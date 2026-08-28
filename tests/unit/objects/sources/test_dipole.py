"""Unit tests for objects/sources/dipole.py.

Tests for PointDipoleSource: initialization, field updates, inverse behavior,
and arbitrary-orientation via azimuth/elevation angles.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.dipole import PointDipoleSource, _support_from_coordinates


@pytest.fixture
def micro_config():
    """Minimal simulation config for source testing."""
    return SimulationConfig(
        time=100e-15,
        grid=UniformGrid(spacing=100e-9),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )


@pytest.fixture
def jax_key():
    """JAX random key for tests."""
    return jax.random.PRNGKey(42)


class TestPointDipoleSourceInitialization:
    """Tests for PointDipoleSource construction."""

    def test_default_electric_dipole(self):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=2,
        )
        assert source.polarization == 2
        assert source.source_type == "electric"
        assert source.amplitude == 1.0
        assert source.azimuth_angle == 0.0
        assert source.elevation_angle == 0.0

    def test_magnetic_dipole(self):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=0,
            source_type="magnetic",
        )
        assert source.source_type == "magnetic"

    def test_custom_amplitude(self):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=1,
            amplitude=5.0,
        )
        assert source.amplitude == 5.0

    def test_invalid_source_type(self):
        with pytest.raises(ValueError, match="source_type must be electric or magnetic"):
            PointDipoleSource(
                partial_grid_shape=(1, 1, 1),
                wave_character=WaveCharacter(wavelength=1e-6),
                source_type="test",
                polarization=0,
            )

    def test_invalid_polarization(self):
        with pytest.raises(ValueError, match="polarization must be 0, 1, or 2"):
            PointDipoleSource(
                partial_grid_shape=(1, 1, 1),
                wave_character=WaveCharacter(wavelength=1e-6),
                polarization=3,
            )

    def test_invalid_interpolate(self):
        with pytest.raises(ValueError, match="interpolate must be a bool"):
            PointDipoleSource(
                partial_grid_shape=(1, 1, 1),
                wave_character=WaveCharacter(wavelength=1e-6),
                polarization=2,
                interpolate="linear",
            )

    def test_all_polarization_axes(self):
        for axis in (0, 1, 2):
            source = PointDipoleSource(
                partial_grid_shape=(1, 1, 1),
                wave_character=WaveCharacter(wavelength=1e-6),
                polarization=axis,
            )
            assert source.polarization == axis

    def test_with_angles(self):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=2,
            azimuth_angle=30.0,
            elevation_angle=15.0,
        )
        assert source.azimuth_angle == 30.0
        assert source.elevation_angle == 15.0

    def test_interpolation_support_handles_degenerate_and_boundary_axes(self):
        for invalid_coordinates in (np.asarray([]), np.ones((1, 2))):
            with pytest.raises(ValueError, match="nonempty 1D array"):
                _support_from_coordinates(invalid_coordinates, 0.0)

        cases = (
            (np.asarray([1.0]), 1.0, (0, 1), [1.0]),
            (np.asarray([1.0, 2.0, 3.0]), 0.0, (0, 2), [1.0, 0.0]),
            (np.asarray([1.0, 2.0, 3.0]), 4.0, (1, 3), [0.0, 1.0]),
        )
        for coordinates, position, expected_bounds, expected_weights in cases:
            bounds, weights = _support_from_coordinates(coordinates, position)
            assert bounds == expected_bounds
            assert jnp.allclose(weights, jnp.asarray(expected_weights))


class TestPointDipoleSourceUpdateE:
    """Tests for electric dipole update_E behavior."""

    def _make_placed(self, micro_config, jax_key, polarization=2, **kwargs):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=polarization,
            **kwargs,
        )
        return source.place_on_grid(
            grid_slice_tuple=((4, 5), (4, 5), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

    def _make_interpolated_placed(self, micro_config, jax_key):
        config = micro_config.aset(
            "grid",
            RectilinearGrid.uniform(
                shape=(8, 8, 8),
                spacing=1.0,
                origin=(-4.0, -4.0, -4.0),
            ),
        )
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            partial_real_position=(0.25, 0.0, 0.0),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=2,
            interpolate=True,
        )
        return source.place_on_grid(
            grid_slice_tuple=((4, 5), (4, 5), (4, 5)),
            config=config,
            key=jax_key,
        )

    def test_electric_dipole_modifies_E(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # Only the polarization component at the source cell should be modified
        assert not jnp.allclose(E_updated[2, 4, 4, 4], 0.0)
        # Other components should remain zero
        assert jnp.allclose(E_updated[0, 4, 4, 4], 0.0)
        assert jnp.allclose(E_updated[1, 4, 4, 4], 0.0)

    def test_electric_dipole_does_not_modify_H(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)
        assert jnp.allclose(H_updated, H)

    def test_inverse_reverses_update(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_fwd = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        E_back = placed.update_E(E_fwd, inv_perm, 1.0, time_step, inverse=True)
        assert jnp.allclose(E_back, E, atol=1e-6)

    def test_amplitude_scaling(self, micro_config, jax_key):
        placed_1 = self._make_placed(micro_config, jax_key, amplitude=1.0)
        placed_2 = self._make_placed(micro_config, jax_key, amplitude=2.0)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E1 = placed_1.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        E2 = placed_2.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        assert jnp.allclose(E2[2, 4, 4, 4], 2.0 * E1[2, 4, 4, 4])

    def test_only_modifies_source_cell(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # Zero out the source cell — rest should still be zero
        E_check = E_updated.at[2, 4, 4, 4].set(0.0)
        assert jnp.allclose(E_check, 0.0)

    def test_interpolation_uses_expected_yee_samples_and_weights(self, micro_config, jax_key):
        placed = self._make_interpolated_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones_like(E)

        updated = placed.update_E(E, inv_perm, 1.0, jnp.array(10), inverse=False)
        abs_ez = jnp.abs(updated[2])
        nonzero = jnp.argwhere(abs_ez > 0)
        expected = jnp.asarray(
            [[4, 4, 3], [4, 4, 4], [5, 4, 3], [5, 4, 4]],
            dtype=nonzero.dtype,
        )

        assert jnp.array_equal(nonzero, expected)
        assert jnp.allclose(abs_ez[4, 4, 3], abs_ez[4, 4, 4])
        assert jnp.allclose(abs_ez[5, 4, 3], abs_ez[5, 4, 4])
        assert jnp.allclose(abs_ez[4, 4, 3], 3.0 * abs_ez[5, 4, 3])

    def test_interpolation_samples_material_on_each_support(self, micro_config, jax_key):
        placed = self._make_interpolated_placed(micro_config, jax_key)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = inv_perm.at[2, 5, 4, 3].set(2.0)
        placed = placed.apply(jax_key, inv_perm, 1.0)

        updated = placed.update_E(
            jnp.zeros_like(inv_perm),
            jnp.ones_like(inv_perm),
            1.0,
            jnp.array(10),
            inverse=False,
        )
        abs_ez = jnp.abs(updated[2])

        assert jnp.allclose(abs_ez[5, 4, 3], 2.0 * abs_ez[5, 4, 4])
        assert jnp.allclose(abs_ez[4, 4, 3], 1.5 * abs_ez[5, 4, 3])

    def test_interpolation_supports_jit_and_material_gradient(self, micro_config, jax_key):
        placed = self._make_interpolated_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)

        @jax.jit
        def response(local_inv_perm):
            inv_perm = jnp.ones_like(E).at[2, 5, 4, 3].set(local_inv_perm)
            updated = placed.update_E(E, inv_perm, 1.0, jnp.array(10), inverse=False)
            return jnp.sum(updated[2] ** 2)

        value, gradient = jax.value_and_grad(response)(jnp.asarray(2.0))
        assert jnp.isfinite(value)
        assert jnp.isfinite(gradient)
        assert gradient != 0.0

    def test_interpolation_uses_nonuniform_yee_coordinates(self, micro_config, jax_key):
        grid = RectilinearGrid(
            x_edges=jnp.asarray([-4.0, -2.0, 1.0, 4.0]),
            y_edges=jnp.asarray([-4.0, -1.0, 1.0, 4.0]),
            z_edges=jnp.asarray([-4.0, -2.0, 2.0, 4.0]),
        )
        config = micro_config.aset("grid", grid)
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=2,
            interpolate=True,
        ).place_on_grid(((1, 2), (1, 2), (1, 2)), config, jax_key)

        support, weights = source._component_supports()
        assert support[2] == ((1, 3), (1, 3), (1, 3))
        assert jnp.allclose(weights[2].sum(), 1.0)
        assert jnp.allclose(weights[2].sum(axis=(1, 2)), jnp.asarray([0.5, 0.5]))

    def test_interpolation_supports_dispersive_material(self, micro_config, jax_key):
        config = micro_config.aset(
            "grid", RectilinearGrid.uniform(shape=(8, 8, 8), spacing=1e-7, origin=(-4e-7, -4e-7, -4e-7))
        )
        placed = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            partial_real_position=(2.5e-8, 0.0, 0.0),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=2,
            interpolate=True,
        ).place_on_grid(((4, 5), (4, 5), (4, 5)), config, jax_key)
        shape = (3, 8, 8, 8)
        inv_perm = jnp.ones(shape, dtype=jnp.float32)
        coeff_shape = (1, *shape)
        c1 = jnp.full(coeff_shape, 0.4, dtype=jnp.float32)
        c2 = jnp.full(coeff_shape, 0.2, dtype=jnp.float32)
        c3 = jnp.full(coeff_shape, 0.1, dtype=jnp.float32)
        applied = placed.apply(jax_key, inv_perm, 1.0, c1, c2, c3)

        plain = placed.update_E(jnp.zeros(shape), inv_perm, 1.0, jnp.array(10), inverse=False)
        dispersive = applied.update_E(jnp.zeros(shape), inv_perm, 1.0, jnp.array(10), inverse=False)
        assert jnp.all(jnp.isfinite(dispersive))
        assert not jnp.allclose(dispersive, plain)

    def test_interpolation_rejects_non_point_shape(self, micro_config, jax_key):
        source = PointDipoleSource(
            partial_grid_shape=(2, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=2,
            interpolate=True,
        )
        with pytest.raises(ValueError, match="require grid_shape"):
            source.place_on_grid(((3, 5), (4, 5), (4, 5)), micro_config, jax_key)

    def test_axis_aligned_full_tensor_material_keeps_coupled_components(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key, polarization=0)
        inv_perm = jnp.zeros((9, 8, 8, 8), dtype=jnp.float32)
        inv_perm = inv_perm.at[0].set(1.0)
        inv_perm = inv_perm.at[3].set(2.0)
        inv_perm = inv_perm.at[6].set(3.0)
        applied = placed.apply(jax_key, inv_perm, 1.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = applied.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        source_values = E_updated[:, 4, 4, 4]
        assert not jnp.allclose(source_values[0], 0.0)
        assert jnp.allclose(source_values[1], 2.0 * source_values[0], atol=1e-6)
        assert jnp.allclose(source_values[2], 3.0 * source_values[0], atol=1e-6)


class TestPointDipoleSourceUpdateH:
    """Tests for magnetic dipole update_H behavior."""

    def _make_placed(self, micro_config, jax_key, polarization=2, **kwargs):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=polarization,
            source_type="magnetic",
            **kwargs,
        )
        return source.place_on_grid(
            grid_slice_tuple=((4, 5), (4, 5), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

    def test_magnetic_dipole_modifies_H(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)
        assert not jnp.allclose(H_updated[2, 4, 4, 4], 0.0)
        assert jnp.allclose(H_updated[0, 4, 4, 4], 0.0)
        assert jnp.allclose(H_updated[1, 4, 4, 4], 0.0)

    def test_magnetic_dipole_does_not_modify_E(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        assert jnp.allclose(E_updated, E)

    def test_magnetic_inverse_reverses_update(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_fwd = placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)
        H_back = placed.update_H(H_fwd, inv_perm, 1.0, time_step, inverse=True)
        assert jnp.allclose(H_back, H, atol=1e-6)

    def test_interpolated_magnetic_dipole_uses_yee_support(self, micro_config, jax_key):
        config = micro_config.aset(
            "grid", RectilinearGrid.uniform(shape=(8, 8, 8), spacing=1.0, origin=(-4.0, -4.0, -4.0))
        )
        placed = self._make_placed(
            config,
            jax_key,
            partial_real_position=(0.25, 0.0, 0.0),
            interpolate=True,
        )
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        updated = placed.update_H(H, jnp.ones_like(H), 1.0, jnp.array(10), inverse=False)

        assert jnp.count_nonzero(updated[2]) == 4
        assert jnp.allclose(updated[0], 0.0)
        assert jnp.allclose(updated[1], 0.0)

    def test_interpolated_magnetic_dipole_samples_each_support(self, micro_config, jax_key):
        config = micro_config.aset(
            "grid", RectilinearGrid.uniform(shape=(8, 8, 8), spacing=1.0, origin=(-4.0, -4.0, -4.0))
        )
        placed = self._make_placed(
            config,
            jax_key,
            partial_real_position=(0.25, 0.0, 0.0),
            interpolate=True,
        )
        inv_mu = jnp.ones((3, 8, 8, 8), dtype=jnp.float32).at[2, 3, 3, 4].set(2.0)
        applied = placed.apply(jax_key, jnp.ones_like(inv_mu), inv_mu)
        updated = applied.update_H(jnp.zeros_like(inv_mu), jnp.ones_like(inv_mu), inv_mu, jnp.array(10), False)

        abs_hz = jnp.abs(updated[2])
        assert jnp.allclose(abs_hz[3, 3, 4], 2.0 * abs_hz[3, 4, 4])


class TestPointDipoleInterpolatedOrientation:
    def test_tilted_dipole_uses_component_specific_supports(self, micro_config, jax_key):
        config = micro_config.aset(
            "grid", RectilinearGrid.uniform(shape=(8, 8, 8), spacing=1.0, origin=(-4.0, -4.0, -4.0))
        )
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            partial_real_position=(0.25, 0.25, 0.25),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=2,
            azimuth_angle=35.0,
            elevation_angle=20.0,
            interpolate=True,
        ).place_on_grid(((4, 5), (4, 5), (4, 5)), config, jax_key)
        field = source.update_E(jnp.zeros((3, 8, 8, 8)), jnp.ones((3, 8, 8, 8)), 1.0, jnp.array(10), False)

        assert all(int(jnp.count_nonzero(field[axis])) == 8 for axis in range(3))


class TestPointDipoleArbitraryOrientation:
    """Tests for dipole with azimuth/elevation angles."""

    def _make_placed(self, micro_config, jax_key, polarization=2, **kwargs):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=polarization,
            **kwargs,
        )
        return source.place_on_grid(
            grid_slice_tuple=((4, 5), (4, 5), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

    def test_zero_angles_matches_axis_aligned(self, micro_config, jax_key):
        """azimuth=0, elevation=0 should be identical to the axis-aligned case."""
        placed_plain = self._make_placed(micro_config, jax_key, polarization=2)
        placed_angled = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=0.0, elevation_angle=0.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E1 = placed_plain.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        E2 = placed_angled.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        assert jnp.allclose(E1, E2, atol=1e-7)

    def test_angled_dipole_injects_multiple_components(self, micro_config, jax_key):
        """A tilted dipole should inject into more than one field component."""
        placed = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=45.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # The z-component should still be modified
        assert not jnp.allclose(E_updated[2, 4, 4, 4], 0.0)
        # At least one other component should also be modified (tilted away from z)
        other_nonzero = not jnp.allclose(E_updated[0, 4, 4, 4], 0.0) or not jnp.allclose(E_updated[1, 4, 4, 4], 0.0)
        assert other_nonzero

    def test_angled_dipole_preserves_unit_norm(self, micro_config, jax_key):
        """The orientation vector should be unit-norm regardless of angles."""
        placed = self._make_placed(micro_config, jax_key, polarization=1, azimuth_angle=30.0, elevation_angle=20.0)
        orientation = placed._orientation
        assert jnp.allclose(jnp.linalg.norm(orientation), 1.0, atol=1e-6)

    def test_angled_dipole_inverse_reverses(self, micro_config, jax_key):
        """Forward + inverse should cancel for a tilted dipole."""
        placed = self._make_placed(micro_config, jax_key, polarization=0, azimuth_angle=60.0, elevation_angle=25.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_fwd = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        E_back = placed.update_E(E_fwd, inv_perm, 1.0, time_step, inverse=True)
        assert jnp.allclose(E_back, E, atol=1e-6)

    def test_angled_magnetic_dipole(self, micro_config, jax_key):
        """Tilted magnetic dipole should inject into multiple H components."""
        placed = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=45.0, source_type="magnetic")

        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)
        assert not jnp.allclose(H_updated[2, 4, 4, 4], 0.0)
        other_nonzero = not jnp.allclose(H_updated[0, 4, 4, 4], 0.0) or not jnp.allclose(H_updated[1, 4, 4, 4], 0.0)
        assert other_nonzero

    def test_90_degree_azimuth_rotates_to_adjacent_axis(self, micro_config, jax_key):
        """A 90-degree azimuth from z should rotate entirely into an adjacent axis."""
        placed = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=90.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # z-component should be ~zero after 90-degree rotation away from z
        assert jnp.allclose(E_updated[2, 4, 4, 4], 0.0, atol=1e-6)
        # The injection should have gone into one of the other axes
        total_other = jnp.abs(E_updated[0, 4, 4, 4]) + jnp.abs(E_updated[1, 4, 4, 4])
        assert total_other > 0

    def test_only_modifies_source_cell_with_angles(self, micro_config, jax_key):
        """An angled dipole should still only modify the single source cell."""
        placed = self._make_placed(micro_config, jax_key, polarization=1, azimuth_angle=35.0, elevation_angle=20.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # Zero out the source cell for all components — rest should be zero
        E_check = E_updated.at[:, 4, 4, 4].set(0.0)
        assert jnp.allclose(E_check, 0.0)

    def test_45_degree_equal_split(self, micro_config, jax_key):
        """A 45-degree azimuth rotation should split energy equally between two axes."""
        placed = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=45.0)
        orientation = placed._orientation

        # The z-component and the rotated-into component should have equal magnitude
        # (cos(45°) = sin(45°) ≈ 0.707)
        z_mag = jnp.abs(orientation[2])
        # Find which horizontal axis got the energy
        h_mag = jnp.sqrt(orientation[0] ** 2 + orientation[1] ** 2)
        assert jnp.allclose(z_mag, h_mag, atol=1e-5)
        assert jnp.allclose(z_mag, np.cos(np.deg2rad(45.0)), atol=1e-5)
