"""
Regression tests for the reference solvers.

These tests pin down the converged coupled solution for the default
configuration so that refactors of the solvers (or of the physics
modules they depend on) cannot silently change the reference numbers
that the surrogate and the PINN are evaluated against.

Run with:
    pytest tests/test_solvers.py -v
"""

import numpy as np
import pytest

from neutherm._compat import trapezoid
from neutherm.physics.parameters import ProblemConfig
from neutherm.solvers.coupled_solver import solve_coupled
from neutherm.solvers.thermal_solver import compute_surface_temperature


POWER_LEVEL = 200.0  # W/cm — linear power used throughout the project


@pytest.fixture(scope="module")
def config():
    return ProblemConfig.from_yaml("configs/default.yaml")


@pytest.fixture(scope="module")
def solution(config):
    """Converged coupled solution for the default config (computed once)."""
    return solve_coupled(config, power_level=POWER_LEVEL, verbose=False)


# =============================================================================
# Convergence and k_eff regression
# =============================================================================

def test_coupled_solver_converges(solution):
    assert solution.converged, "Picard iteration did not converge"
    # The default problem converges quickly; a blow-up in iteration count
    # signals a regression in the feedback loop.
    assert solution.n_iterations <= 15


def test_k_eff_regression(solution):
    """k_eff for the default configuration (reference: 1.300934)."""
    assert solution.k_eff == pytest.approx(1.300934, abs=1e-4)


# =============================================================================
# Power normalization
# =============================================================================

def test_flux_power_normalization(solution, config):
    """The flux must be normalized so the integrated linear power matches.

    q''' is in W/m³ and the meshes are stored in cm, so convert the
    radius to meters for the integral:  P' = ∫ q'''(r) · 2πr dr  [W/m].
    """
    r_m = solution.r_fuel / 100.0  # cm → m
    linear_power_w_per_m = trapezoid(
        solution.q_volumetric * 2.0 * np.pi * r_m, r_m
    )
    linear_power_w_per_cm = linear_power_w_per_m / 100.0
    assert linear_power_w_per_cm == pytest.approx(POWER_LEVEL, rel=1e-2)


def test_fluxes_positive(solution):
    assert np.all(solution.phi1 > 0.0)
    assert np.all(solution.phi2 > 0.0)


def test_flux_physical_magnitude(solution):
    """Power-normalized PWR pin fluxes should be in the 1e13–1e15 range."""
    assert 1e12 < solution.phi1.max() < 1e16
    assert 1e12 < solution.phi2.max() < 1e16


# =============================================================================
# Thermal profile
# =============================================================================

def test_temperature_profile_shape(solution, config):
    """Centerline must be the hottest point; profile decreases outward."""
    T = solution.temperature
    assert T[0] == pytest.approx(T.max(), rel=1e-6)
    assert np.all(np.diff(T) <= 1e-9), "Temperature must be monotonically decreasing"
    assert T[0] > T[-1]


def test_temperature_regression(solution):
    """Centerline and surface temperatures (reference: 1139.1 / 757.8 K)."""
    assert solution.temperature[0] == pytest.approx(1139.1, abs=2.0)
    assert solution.temperature[-1] == pytest.approx(757.8, abs=2.0)


def test_surface_temperature_consistency(solution, config):
    """The converged fuel surface temperature must match the analytic
    gap/cladding/convection model for the same linear power."""
    T_surf_analytic = compute_surface_temperature(
        POWER_LEVEL * 100.0,  # W/cm → W/m
        config.geometry,
        config.thermal,
    )
    assert solution.temperature[-1] == pytest.approx(T_surf_analytic, abs=1.0)


def test_temperature_above_coolant(solution, config):
    assert np.all(solution.temperature > config.thermal.T_coolant)


# =============================================================================
# Doppler feedback sanity
# =============================================================================

def test_doppler_feedback_reduces_k_eff(config):
    """Doubling the power should lower k_eff (negative Doppler coefficient)."""
    sol_lo = solve_coupled(config, power_level=100.0, verbose=False)
    sol_hi = solve_coupled(config, power_level=400.0, verbose=False)
    assert sol_hi.k_eff < sol_lo.k_eff
