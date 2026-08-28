from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.core.axis import get_oriented_transverse_axes
from fdtdx.core.grid import RectilinearGrid
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.linalg import rotate_vector
from fdtdx.core.misc import ensure_slice_tuple
from fdtdx.core.null import Null
from fdtdx.dispersion import effective_inv_permittivity
from fdtdx.objects.sources.source import Source
from fdtdx.typing import SliceTuple3D


def _contract_orientation(
    inv_material: jax.Array | float,
    orientation: jax.Array,
) -> jax.Array:
    """Contract a material tensor with the dipole orientation.

    Returns ``inv_oriented[i, x, y, z] = sum_j inv_material_{ij}(x,y,z) * orientation_j``
    as a ``(3, Nx, Ny, Nz)`` array. For the isotropic and diagonal
    representations this reduces to componentwise multiplication, matching the
    pre-existing behavior. For the 9-component flattened tensor it correctly
    picks up the off-diagonal coupling that was previously dropped by indexing
    a row of the flattened tensor with ``inv_material[axis]``.
    """
    orient = orientation.reshape(3, 1, 1, 1)
    if not isinstance(inv_material, jax.Array) or inv_material.ndim == 0:
        return jnp.asarray(inv_material) * orient
    shape0 = inv_material.shape[0]
    if shape0 == 9:
        tensor = inv_material.reshape(3, 3, *inv_material.shape[1:])
        return jnp.einsum("ij...,j->i...", tensor, orientation)
    # shape0 in (1, 3): isotropic scalar broadcasts against orientation, diagonal
    # multiplies componentwise. Both collapse to this single multiplication.
    return inv_material * orient


def _axis_aligned_diagonal_injection(
    inv_material: jax.Array | float, azimuth_angle: float, elevation_angle: float
) -> bool:
    """Return whether an axis-aligned dipole can update only one field component."""
    if azimuth_angle != 0.0 or elevation_angle != 0.0:
        return False
    if not isinstance(inv_material, jax.Array) or inv_material.ndim == 0:
        return True
    return inv_material.shape[0] in (1, 3)


def _support_from_coordinates(
    coordinates: np.ndarray,
    position: float,
) -> tuple[tuple[int, int], jax.Array]:
    """Return linear interpolation support on a Yee coordinate axis."""
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("Source interpolation coordinates must be a nonempty 1D array")

    if coordinates.size == 1:
        return (0, 1), jnp.ones((1,), dtype=jnp.float32)

    if position <= coordinates[0]:
        return (0, 2), jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    if position >= coordinates[-1]:
        return (
            (coordinates.size - 2, coordinates.size),
            jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        )

    upper = int(np.searchsorted(coordinates, position, side="right"))
    lower = upper - 1
    upper_weight = float((position - coordinates[lower]) / (coordinates[upper] - coordinates[lower]))
    return (
        (lower, upper + 1),
        jnp.asarray([1.0 - upper_weight, upper_weight], dtype=jnp.float32),
    )


def _field_component_coordinates(
    grid: RectilinearGrid,
    source_type: Literal["electric", "magnetic"],
    component_axis: int,
    coordinate_axis: int,
) -> np.ndarray:
    """Return the physical Yee coordinates of one field component."""
    electric_centered = source_type == "electric" and coordinate_axis == component_axis
    magnetic_centered = source_type == "magnetic" and coordinate_axis != component_axis
    if electric_centered or magnetic_centered:
        return np.asarray(grid.centers(coordinate_axis))
    return np.asarray(grid.edges(coordinate_axis)[:-1])


def _outer_product_weights(axis_weights: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
    """Build a separable 3D interpolation stencil."""
    wx, wy, wz = axis_weights
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


@autoinit
class PointDipoleSource(Source):
    """Soft point dipole source (electric or magnetic).

    Injects an impressed current into the Yee grid. With ``interpolate=False`` (the default), the
    current is applied at one grid sample for backward compatibility. With ``interpolate=True``,
    the continuous source position is distributed linearly over the neighboring, component-specific
    Yee samples. The source is "soft": it adds to the field rather than overwriting, so
    scattered/reflected fields pass through without artificial reflections.

    The dipole orientation starts along the ``polarization`` axis and is then
    rotated by ``azimuth_angle`` and ``elevation_angle`` (both in degrees),
    following the same convention as :class:`TFSFPlaneSource`.  When both
    angles are zero the dipole is axis-aligned, recovering the original
    behavior.

    For an electric dipole with unit orientation ``p_hat``, the E-field
    update at each time step is::

        E[i, x, y, z] += -c * (inv_eps @ p_hat)[i] * amplitude * temporal(t)

    where the tensor contraction ``(inv_eps @ p_hat)[i] = sum_j inv_eps_{ij} * p_hat_j``
    collapses to ``inv_eps[i] * p_hat[i]`` for isotropic and diagonal media but
    correctly picks up off-diagonal coupling when the permittivity is a full
    3x3 tensor.

    For a magnetic dipole, the dual applies during the H update with
    inv_permeability replacing inv_permittivity.

    The medium permittivity/permeability is sampled once during :meth:`apply` at every injection
    sample used by the selected mode. Dispersive coefficients are evaluated at the carrier angular
    frequency, so dispersive media are handled without runtime material lookups.
    """

    #: Polarization axis (0=x, 1=y, 2=z).
    polarization: int = frozen_field()

    #: Azimuth angle in degrees (rotation around vertical axis).
    azimuth_angle: float = frozen_field(default=0.0)

    #: Elevation angle in degrees (rotation around horizontal axis).
    elevation_angle: float = frozen_field(default=0.0)

    #: Source type: "electric" injects into E update, "magnetic" into H update.
    source_type: Literal["electric", "magnetic"] = frozen_field(default="electric")

    #: Source amplitude.
    amplitude: float = frozen_field(default=1.0)

    #: Distribute a continuous source position onto neighboring Yee samples.
    #: Disabled by default to preserve the legacy single-cell source behavior.
    interpolate: bool = frozen_field(default=False)

    _inv_eps_local: jax.Array = private_field()
    _inv_mu_local: jax.Array | float = private_field()
    _inv_eps_oriented: jax.Array = private_field()
    _inv_mu_oriented: jax.Array = private_field()
    _source_support_slices: tuple[SliceTuple3D, SliceTuple3D, SliceTuple3D] = private_field()
    _source_weights: tuple[jax.Array, jax.Array, jax.Array] = private_field()
    _inv_eps_oriented_components: tuple[jax.Array, jax.Array, jax.Array] = private_field()
    _inv_mu_oriented_components: tuple[jax.Array, jax.Array, jax.Array] = private_field()

    def __post_init__(self):
        if self.source_type not in ("electric", "magnetic"):
            raise ValueError(f"source_type must be electric or magnetic, got {self.source_type}")
        if self.polarization not in (0, 1, 2):
            raise ValueError(f"polarization must be 0, 1, or 2, got {self.polarization}")
        if not isinstance(self.interpolate, bool):
            raise ValueError(f"interpolate must be a bool, got {self.interpolate!r}")

    def validate_placement(self, objects) -> list[str]:
        """Reject a dipole sitting on a symmetry plane.

        The mirror plane lies on a cell *edge*, so no cell is centred on it. A dipole placed in the
        first cell of the reduced domain therefore does not stand for one dipole on the plane: the
        reduced simulation models it together with its mirror image, i.e. two dipoles half a cell
        apart, which is not the model the user drew in the full domain.
        """
        errors = list(super().validate_placement(objects))
        on_plane = [a for a in range(3) if self.touches_symmetry_plane(a)]
        if on_plane:
            axis_names = ", ".join("xyz"[a] for a in on_plane)
            errors.append(
                f"Point dipole '{self.name}' sits on the {axis_names}-symmetry plane. The plane lies on a "
                f"cell edge, so a single dipole cannot be centred on it - the reduced simulation would "
                f"model the dipole plus its mirror image, half a cell apart. Move the dipole off the "
                f"plane, or drop the symmetry on that axis."
            )
        return errors

    def _requested_position(self, grid: RectilinearGrid) -> tuple[float, float, float]:
        """Return the requested continuous position in resolved-grid coordinates."""
        position = []
        for axis in range(3):
            real_position = self.partial_real_position[axis]
            edges = grid.edges(axis)
            domain_center = 0.5 * (float(edges[0]) + float(edges[-1]))
            if real_position is None:
                start, stop = self.grid_slice_tuple[axis]
                local_edges = grid.edges(axis)[start : stop + 1]
                position.append(0.5 * (float(local_edges[0]) + float(local_edges[-1])))
            else:
                position.append(domain_center + float(real_position))
        return position[0], position[1], position[2]

    def _single_cell_support(
        self,
    ) -> tuple[
        tuple[SliceTuple3D, SliceTuple3D, SliceTuple3D],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        support_slices = (
            self.grid_slice_tuple,
            self.grid_slice_tuple,
            self.grid_slice_tuple,
        )
        weight = jnp.ones(self.grid_shape, dtype=self._config.dtype)
        return support_slices, (weight, weight, weight)

    def _interpolated_support(
        self, grid: RectilinearGrid
    ) -> tuple[
        tuple[SliceTuple3D, SliceTuple3D, SliceTuple3D],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        source_position = self._requested_position(grid)
        support_slices: list[SliceTuple3D] = []
        support_weights = []
        for component_axis in range(3):
            axis_bounds = []
            axis_weights = []
            for coordinate_axis in range(3):
                coordinates = _field_component_coordinates(grid, self.source_type, component_axis, coordinate_axis)
                bounds, weights = _support_from_coordinates(coordinates, source_position[coordinate_axis])
                axis_bounds.append(bounds)
                axis_weights.append(weights.astype(self._config.dtype))
            support_slices.append((axis_bounds[0], axis_bounds[1], axis_bounds[2]))
            support_weights.append(_outer_product_weights((axis_weights[0], axis_weights[1], axis_weights[2])))
        return (
            (support_slices[0], support_slices[1], support_slices[2]),
            (support_weights[0], support_weights[1], support_weights[2]),
        )

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(grid_slice_tuple, config, key)
        grid = self._config.resolved_grid
        if self.interpolate and self.grid_shape != (1, 1, 1):
            raise ValueError(f"Interpolated point dipoles require grid_shape=(1, 1, 1), got {self.grid_shape}")
        if self.interpolate and grid is not None:
            support_slices, support_weights = self._interpolated_support(grid)
        else:
            support_slices, support_weights = self._single_cell_support()
        self = self.aset("_source_support_slices", support_slices, create_new_ok=True)
        return self.aset("_source_weights", support_weights, create_new_ok=True)

    def _component_supports(
        self,
    ) -> tuple[
        tuple[SliceTuple3D, SliceTuple3D, SliceTuple3D],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        return self._source_support_slices, self._source_weights

    @property
    def _orientation(self) -> jnp.ndarray:
        """Normalized orientation vector as a (3,) JAX array.

        Starts as the unit vector along ``polarization`` and is rotated by
        ``azimuth_angle`` / ``elevation_angle`` using the same rotation
        convention as :func:`rotate_vector`.
        """
        base = jnp.zeros(3, dtype=self._config.dtype).at[self.polarization].set(1.0)
        if self.azimuth_angle == 0.0 and self.elevation_angle == 0.0:
            return base
        horizontal_axis, vertical_axis = get_oriented_transverse_axes(self.polarization)
        axes_tuple = (horizontal_axis, vertical_axis, self.polarization)
        return rotate_vector(
            base,
            azimuth_angle=np.deg2rad(self.azimuth_angle),
            elevation_angle=np.deg2rad(self.elevation_angle),
            axes_tuple=axes_tuple,
        )

    def _oriented_components_from_material(
        self,
        material: jax.Array | float,
        support_slices: tuple[SliceTuple3D, SliceTuple3D, SliceTuple3D],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample and contract a material independently on each Yee support."""
        components = []
        for axis, support_slice in enumerate(support_slices):
            if isinstance(material, jax.Array) and material.ndim > 0:
                material_slice: jax.Array | float = material[:, *ensure_slice_tuple(support_slice)]
            else:
                material_slice = material
            components.append(_contract_orientation(material_slice, self._orientation)[axis])
        return components[0], components[1], components[2]

    def _effective_inv_eps_components(
        self,
        inv_permittivities: jax.Array,
        support_slices: tuple[SliceTuple3D, SliceTuple3D, SliceTuple3D],
        dispersive_c1: jax.Array | None,
        dispersive_c2: jax.Array | None,
        dispersive_c3: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample effective inverse permittivity on each electric support."""
        components = []
        for axis, support_slice in enumerate(support_slices):
            source_slice = ensure_slice_tuple(support_slice)
            inv_eps_slice = inv_permittivities[:, *source_slice]
            if dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None:
                inv_eps_slice = effective_inv_permittivity(
                    inv_eps=inv_eps_slice,
                    c1=dispersive_c1[:, :, *source_slice],
                    c2=dispersive_c2[:, :, *source_slice],
                    c3=dispersive_c3[:, :, *source_slice],
                    omega=2.0 * np.pi * self.wave_character.get_frequency(),
                    dt=self._config.time_step_duration,
                )
            components.append(_contract_orientation(inv_eps_slice, self._orientation)[axis])
        return components[0], components[1], components[2]

    def _inject_interpolated(
        self,
        field: jax.Array,
        oriented_components: tuple[jax.Array, jax.Array, jax.Array],
        scale: jax.Array,
        sign: float,
    ) -> jax.Array:
        """Add a weighted source contribution to each component-specific support."""
        support_slices, weights = self._component_supports()
        for axis in range(3):
            injection = scale * oriented_components[axis] * weights[axis]
            field = field.at[axis, *ensure_slice_tuple(support_slices[axis])].add(sign * injection.astype(field.dtype))
        return field

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
        electric_conductivity: jax.Array | None = None,
    ) -> Self:
        del key, electric_conductivity

        if self.interpolate:
            support_slices, _ = self._component_supports()
            if self.source_type == "electric":
                inv_eps_components = self._effective_inv_eps_components(
                    inv_permittivities,
                    support_slices,
                    dispersive_c1,
                    dispersive_c2,
                    dispersive_c3,
                )
                return self.aset("_inv_eps_oriented_components", inv_eps_components, create_new_ok=True)
            inv_mu_components = self._oriented_components_from_material(inv_permeabilities, support_slices)
            return self.aset("_inv_mu_oriented_components", inv_mu_components, create_new_ok=True)

        inv_eps_slice = inv_permittivities[:, *self.grid_slice]

        if dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None:
            c1_slice = dispersive_c1[:, :, *self.grid_slice]
            c2_slice = dispersive_c2[:, :, *self.grid_slice]
            c3_slice = dispersive_c3[:, :, *self.grid_slice]
            inv_eps_slice = effective_inv_permittivity(
                inv_eps=inv_eps_slice,
                c1=c1_slice,
                c2=c2_slice,
                c3=c3_slice,
                omega=2.0 * np.pi * self.wave_character.get_frequency(),
                dt=self._config.time_step_duration,
            )

        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_mu_slice: jax.Array | float = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_mu_slice = inv_permeabilities

        inv_eps_oriented = _contract_orientation(inv_eps_slice, self._orientation)
        inv_mu_oriented = _contract_orientation(inv_mu_slice, self._orientation)

        self = self.aset("_inv_eps_local", inv_eps_slice, create_new_ok=True)
        self = self.aset("_inv_mu_local", inv_mu_slice, create_new_ok=True)
        self = self.aset("_inv_eps_oriented", inv_eps_oriented, create_new_ok=True)
        self = self.aset("_inv_mu_oriented", inv_mu_oriented, create_new_ok=True)
        return self

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permeabilities
        if self.source_type != "electric":
            return E

        dt = self._config.time_step_duration
        c = self._config.courant_number

        amplitude = self.temporal_profile.get_amplitude(
            time=time_step * dt,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )

        sign = -1.0 if not inverse else 1.0
        scale = c * self.amplitude * self.static_amplitude_factor * amplitude

        if self.interpolate:
            support_slices, _ = self._component_supports()
            if isinstance(self._inv_eps_oriented_components, Null):
                oriented_components = self._effective_inv_eps_components(
                    inv_permittivities,
                    support_slices,
                    dispersive_c1=None,
                    dispersive_c2=None,
                    dispersive_c3=None,
                )
            else:
                oriented_components = self._inv_eps_oriented_components
            return self._inject_interpolated(E, oriented_components, scale, sign)

        if isinstance(self._inv_eps_oriented, Null):
            inv_eps_source = inv_permittivities[:, *self.grid_slice]
            inv_eps_oriented = _contract_orientation(inv_eps_source, self._orientation)
        else:
            inv_eps_source = self._inv_eps_local
            inv_eps_oriented = self._inv_eps_oriented

        if _axis_aligned_diagonal_injection(inv_eps_source, self.azimuth_angle, self.elevation_angle):
            injection = scale * inv_eps_oriented[self.polarization]
            E = E.at[self.polarization, *self.grid_slice].add(sign * injection.astype(E.dtype))
        else:
            for axis in range(3):
                injection = scale * inv_eps_oriented[axis]
                E = E.at[axis, *self.grid_slice].add(sign * injection.astype(E.dtype))

        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities
        if self.source_type != "magnetic":
            return H

        dt = self._config.time_step_duration
        c = self._config.courant_number

        amplitude = self.temporal_profile.get_amplitude(
            time=time_step * dt,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )

        sign = -1.0 if not inverse else 1.0
        scale = c * self.amplitude * self.static_amplitude_factor * amplitude

        if self.interpolate:
            support_slices, _ = self._component_supports()
            if isinstance(self._inv_mu_oriented_components, Null):
                oriented_components = self._oriented_components_from_material(inv_permeabilities, support_slices)
            else:
                oriented_components = self._inv_mu_oriented_components
            return self._inject_interpolated(H, oriented_components, scale, sign)

        if isinstance(self._inv_mu_oriented, Null):
            inv_mu_source: jax.Array | float = inv_permeabilities
            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                inv_mu_source = inv_permeabilities[:, *self.grid_slice]
            inv_mu_oriented = _contract_orientation(inv_mu_source, self._orientation)
        else:
            inv_mu_source = self._inv_mu_local
            inv_mu_oriented = self._inv_mu_oriented

        if _axis_aligned_diagonal_injection(inv_mu_source, self.azimuth_angle, self.elevation_angle):
            injection = scale * inv_mu_oriented[self.polarization]
            H = H.at[self.polarization, *self.grid_slice].add(sign * injection.astype(H.dtype))
        else:
            for axis in range(3):
                injection = scale * inv_mu_oriented[axis]
                H = H.at[axis, *self.grid_slice].add(sign * injection.astype(H.dtype))

        return H
