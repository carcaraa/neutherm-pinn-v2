"""
Steady-state radial heat conduction solver for a cylindrical fuel pin.

Solves:
    -1/r d/dr(r k(T) dT/dr) = q'''(r)

with boundary conditions:
    dT/dr(0) = 0                          (symmetry at center)
    T(R_f) = T_surface                    (fuel surface temperature)

The fuel surface temperature is computed from the thermal resistance
chain: fuel surface → gap → cladding → coolant.

The nonlinearity from k(T) is handled by Picard linearization:
we evaluate k at the previous temperature iterate and solve the
resulting linear system.

References
----------
.. [3] Todreas & Kazimi, "Nuclear Systems I" (2012), Ch. 8.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as la

from neutherm._compat import trapezoid
from neutherm.physics.parameters import GeometryParams, ThermalParams
from neutherm.physics.fuel_properties import fuel_conductivity_np


def compute_surface_temperature(
    q_total: float,
    geom: GeometryParams,
    thermal: ThermalParams,
) -> float:
    """Compute the fuel surface temperature from total pin power.

    Uses the thermal resistance chain from fuel surface to coolant:

        T_surface = T_coolant + q'' / h_conv + q'' / h_gap

    where q'' is the surface heat flux computed from the total
    volumetric power integrated over the fuel cross-section.

    In 1D cylindrical geometry, the linear heat rate q' [W/m] is:
        q' = ∫₀^{R_f} q'''(r) * 2π r dr

    And the surface heat flux:
        q'' = q' / (2π R_f)

    Parameters
    ----------
    q_total : float
        Total volumetric heat generation integrated over the pin
        cross-section: ∫ q''' 2πr dr [W/m]. This is the linear heat rate.
    geom : GeometryParams
        Fuel pin geometry.
    thermal : ThermalParams
        Thermal-hydraulic parameters (h_gap, h_conv, T_coolant).

    Returns
    -------
    float
        Fuel outer surface temperature [K].
    """
    # Convert fuel radius from [m] to [m] (it's already in SI)
    R_f = geom.r_fuel  # [m]
    R_c = geom.r_clad  # [m]

    # Surface heat flux at the cladding outer surface
    # q' = q_total (the linear heat rate, already integrated)
    # q''_clad = q' / (2π R_c)
    q_flux_clad = q_total / (2 * np.pi * R_c)

    # q''_fuel = q' / (2π R_f)
    q_flux_fuel = q_total / (2 * np.pi * R_f)

    # Temperature drops across each resistance:
    # 1. Coolant → cladding outer surface: ΔT = q''_clad / h_conv
    # 2. Cladding (thin wall, neglect for now)
    # 3. Gap → fuel surface: ΔT = q''_fuel / h_gap
    T_clad_outer = thermal.T_coolant + q_flux_clad / thermal.h_conv
    T_fuel_surface = T_clad_outer + q_flux_fuel / thermal.h_gap

    return T_fuel_surface


def build_thermal_matrix(
    r: np.ndarray,
    k: np.ndarray,
    dr: float,
) -> np.ndarray:
    """Build the finite-difference matrix for the heat conduction operator.

    Discretizes: -1/r * d/dr(r * k(T) * dT/dr)

    This is structurally identical to the diffusion operator in
    diffusion_solver.py (same cylindrical Laplacian), but applied
    to temperature instead of neutron flux, and with thermal
    conductivity k(T) instead of diffusion coefficient D.

    Parameters
    ----------
    r : np.ndarray
        Radial mesh points [m], shape (N,).
    k : np.ndarray
        Thermal conductivity at each mesh point [W/(m·K)], shape (N,).
    dr : float
        Mesh spacing [m].

    Returns
    -------
    np.ndarray
        Matrix A of shape (N, N) for the system A @ T = q'''.
    """
    N = len(r)
    A = np.zeros((N, N))

    # Interior points: same conservative discretization as diffusion
    for i in range(1, N - 1):
        r_plus = r[i] + dr / 2
        r_minus = r[i] - dr / 2
        k_plus = (k[i] + k[i + 1]) / 2
        k_minus = (k[i] + k[i - 1]) / 2

        coeff_minus = r_minus * k_minus / (r[i] * dr**2)
        coeff_plus = r_plus * k_plus / (r[i] * dr**2)
        coeff_center = coeff_minus + coeff_plus

        A[i, i - 1] = -coeff_minus
        A[i, i] = coeff_center
        A[i, i + 1] = -coeff_plus

    # r = 0: L'Hôpital, symmetry BC dT/dr = 0
    A[0, 0] = 4.0 * k[0] / dr**2
    A[0, 1] = -4.0 * k[0] / dr**2

    # r = R_f: Dirichlet BC (T = T_surface, prescribed)
    A[N - 1, N - 1] = 1.0

    return A


def solve_thermal(
    r: np.ndarray,
    q_volumetric: np.ndarray,
    T_prev: np.ndarray,
    geom: GeometryParams,
    thermal: ThermalParams,
) -> np.ndarray:
    """Solve the radial heat conduction equation for one Picard iteration.

    Given a volumetric heat source q'''(r) (from the neutronics) and
    the previous temperature iterate T_prev (for evaluating k(T)),
    solves for the new temperature profile.

    Parameters
    ----------
    r : np.ndarray
        Radial mesh points [m], shape (N,).
    q_volumetric : np.ndarray
        Volumetric heat generation [W/m³] at each mesh point, shape (N,).
    T_prev : np.ndarray
        Previous temperature iterate [K], shape (N,). Used to evaluate
        the temperature-dependent conductivity k(T).
    geom : GeometryParams
        Fuel pin geometry.
    thermal : ThermalParams
        Thermal parameters (conductivity coefficients, BCs).

    Returns
    -------
    np.ndarray
        New temperature profile [K], shape (N,).
    """
    dr = r[1] - r[0]
    N = len(r)

    # Evaluate thermal conductivity at the previous temperature
    # This is the Picard linearization: k(T^{n}) is used to solve for T^{n+1}
    k = fuel_conductivity_np(T_prev, thermal)

    # Compute the fuel surface temperature from the total pin power
    # Total linear heat rate: q' = ∫₀^{R_f} q'''(r) * 2πr dr
    # Numerical integration using the trapezoidal rule
    q_linear = trapezoid(q_volumetric * 2 * np.pi * r, r)
    T_surface = compute_surface_temperature(q_linear, geom, thermal)

    # Build and solve the linear system
    A = build_thermal_matrix(r, k, dr)
    rhs = q_volumetric.copy()

    # Apply Dirichlet BC at the fuel surface
    rhs[-1] = T_surface

    T_new = la.solve(A, rhs)

    return T_new
