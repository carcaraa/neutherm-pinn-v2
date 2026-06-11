"""
Coupled neutronics-thermal solver using Picard (fixed-point) iteration.

This is the reference solver that generates "ground truth" data for
training the surrogate model and validating the PINN.

The coupling loop:
    1. Start with an initial temperature guess (uniform T_coolant)
    2. Evaluate cross sections at current temperature: Σ(T)
    3. Solve neutron diffusion → φ_1(r), φ_2(r), k_eff
    4. Compute heat generation: q'''(r) = κ * (Σ_f1 φ_1 + Σ_f2 φ_2)
    5. Solve heat conduction → T(r)
    6. Check convergence (temperature and k_eff)
    7. If not converged, go to step 2 with updated T(r)

The convergence is typically fast (5-20 iterations) because the
Doppler feedback is weak (small α coefficients). However, the
nonlinearity from k(T) in the thermal solve can slow convergence
if the temperature change is large between iterations.

References
----------
.. [1] Duderstadt & Hamilton, "Nuclear Reactor Analysis" (1976), Ch. 7.
.. [3] Todreas & Kazimi, "Nuclear Systems I" (2012), Ch. 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neutherm._compat import trapezoid
from neutherm.physics.parameters import ProblemConfig
from neutherm.physics.cross_sections import (
    evaluate_cross_sections_np,
    build_pin_cell_xs_np,
)
from neutherm.physics.fuel_properties import heat_generation_np
from neutherm.solvers.diffusion_solver import solve_diffusion
from neutherm.solvers.thermal_solver import solve_thermal


@dataclass
class CoupledSolution:
    """Container for the converged coupled solution.

    Stores all fields on the radial mesh plus scalar quantities.
    The neutronics mesh covers the full pin cell (fuel + moderator),
    while temperature and heat generation are defined only in the fuel.

    Attributes
    ----------
    r_neutronics : np.ndarray
        Full pin cell radial mesh [cm] (fuel + moderator).
    r_fuel : np.ndarray
        Fuel-only radial mesh [cm].
    phi1 : np.ndarray
        Fast-group neutron flux on the full mesh (normalized to power).
    phi2 : np.ndarray
        Thermal-group neutron flux on the full mesh.
    temperature : np.ndarray
        Temperature profile in the fuel [K].
    q_volumetric : np.ndarray
        Volumetric heat generation in the fuel [W/m³].
    k_eff : float
        Effective multiplication factor.
    n_iterations : int
        Number of Picard iterations to converge.
    converged : bool
        Whether the solution converged within the iteration limit.
    """

    r_neutronics: np.ndarray
    r_fuel: np.ndarray
    phi1: np.ndarray
    phi2: np.ndarray
    temperature: np.ndarray
    q_volumetric: np.ndarray
    k_eff: float
    n_iterations: int
    converged: bool


def solve_coupled(
    config: ProblemConfig,
    power_level: float = 200.0,
    verbose: bool = True,
) -> CoupledSolution:
    """Run the coupled neutronics-thermal Picard iteration.

    The neutronics domain covers the full pin cell (fuel + moderator)
    to correctly capture neutron moderation and reflection. The thermal
    domain covers only the fuel region where heat is generated.

    Parameters
    ----------
    config : ProblemConfig
        Full problem configuration (geometry, physics, solver settings).
    power_level : float
        Target linear heat rate [W/cm] for flux normalization.
        Typical PWR value: ~200 W/cm.
    verbose : bool
        If True, print convergence information at each iteration.

    Returns
    -------
    CoupledSolution
        Converged (or best-effort) solution containing all fields.
    """
    geom = config.geometry
    neutronics = config.neutronics
    thermal_params = config.thermal
    solver = config.solver

    # =========================================================================
    # Set up the radial meshes
    # =========================================================================
    # Neutronics mesh: covers the full pin cell [0, R_cell] in [cm]
    # This includes: fuel (0 to R_fuel) + moderator (R_fuel to R_cell)
    # We skip the thin cladding neutronically (small effect on transport).
    N_fuel = geom.n_radial
    N_mod = geom.n_radial_mod
    R_fuel_cm = geom.r_fuel * 100  # [m] → [cm]
    R_cell_cm = geom.r_cell * 100  # [m] → [cm]

    # Build a composite mesh: uniform in fuel, uniform in moderator
    r_fuel_cm = np.linspace(0, R_fuel_cm, N_fuel)
    r_mod_cm = np.linspace(R_fuel_cm, R_cell_cm, N_mod + 1)[1:]  # Skip duplicate point
    r_full_cm = np.concatenate([r_fuel_cm, r_mod_cm])
    N_total = len(r_full_cm)

    # Thermal mesh: fuel region only [m]
    r_fuel_m = np.linspace(0, geom.r_fuel, N_fuel)

    if verbose:
        print("=" * 70)
        print("Coupled Neutronics-Thermal Solver (Pin Cell Model)")
        print("=" * 70)
        print(f"  Fuel radius:  {R_fuel_cm:.4f} cm")
        print(f"  Cell radius:  {R_cell_cm:.4f} cm")
        print(f"  Mesh (fuel):  {N_fuel} points")
        print(f"  Mesh (mod):   {N_mod} points")
        print(f"  Mesh (total): {N_total} points")
        print(f"  T_coolant:    {thermal_params.T_coolant:.1f} K")
        print(f"  Target q':    {power_level:.1f} W/cm")
        print("-" * 70)
        print(f"  {'Iter':>4s}  {'k_eff':>10s}  {'Δk/k':>10s}  "
              f"{'T_center':>10s}  {'T_surface':>10s}  {'ΔT_max':>10s}")
        print("-" * 70)

    # =========================================================================
    # Initialize the temperature field (fuel only)
    # =========================================================================
    T_fuel = np.full(N_fuel, thermal_params.T_coolant)
    k_eff_old = 0.0
    converged = False
    n_iter = 0

    for iteration in range(solver.max_picard_iter):
        n_iter = iteration + 1

        # =====================================================================
        # Step 1: Build pin cell cross sections
        # =====================================================================
        # Fuel: T-dependent via Doppler model
        # Moderator: fixed cross sections, no fission
        xs = build_pin_cell_xs_np(r_full_cm, T_fuel, R_fuel_cm, neutronics)

        # =====================================================================
        # Step 2: Solve neutron diffusion on the full pin cell
        # =====================================================================
        # The reflective BC (zero net current, dφ/dr = 0) at R_cell is the
        # physically correct condition here: at the Wigner-Seitz cell
        # boundary, neighboring pin cells are mirror images of each other,
        # so the net neutron current vanishes by symmetry. This is what
        # build_diffusion_matrix implements at the outer boundary.
        phi1, phi2, k_eff = solve_diffusion(
            r_full_cm, xs,
            tol=solver.power_iteration_tol,
            max_iter=solver.power_iteration_max,
        )

        # =====================================================================
        # Step 3: Normalize fluxes to match the target power level
        # =====================================================================
        # Extract fuel-region fluxes and cross sections for heat generation
        phi1_fuel = phi1[:N_fuel]
        phi2_fuel = phi2[:N_fuel]
        nu_sf1_fuel = xs.nu_sigma_f1[:N_fuel]
        nu_sf2_fuel = xs.nu_sigma_f2[:N_fuel]

        q_unnorm = heat_generation_np(
            phi1_fuel, phi2_fuel, nu_sf1_fuel, nu_sf2_fuel,
            thermal_params.kappa_fission,
        )

        # Integrate over the fuel cross-section: q' = ∫ q''' 2πr dr [W/cm]
        q_linear_unnorm = trapezoid(q_unnorm * 2 * np.pi * r_fuel_cm, r_fuel_cm)

        # Scale all fluxes so that q' = power_level
        if q_linear_unnorm > 0:
            norm_factor = power_level / q_linear_unnorm
        else:
            norm_factor = 1.0

        phi1 *= norm_factor
        phi2 *= norm_factor

        # Recompute properly normalized heat generation in fuel [W/cm³]
        phi1_fuel = phi1[:N_fuel]
        phi2_fuel = phi2[:N_fuel]
        q_volumetric_cgs = heat_generation_np(
            phi1_fuel, phi2_fuel, nu_sf1_fuel, nu_sf2_fuel,
            thermal_params.kappa_fission,
        )

        # Convert to [W/m³] for the thermal solver
        q_volumetric_si = q_volumetric_cgs * 1e6

        # =====================================================================
        # Step 4: Solve heat conduction in the fuel
        # =====================================================================
        T_old = T_fuel.copy()
        T_fuel = solve_thermal(r_fuel_m, q_volumetric_si, T_fuel, geom, thermal_params)

        # =====================================================================
        # Step 5: Check convergence
        # =====================================================================
        if k_eff_old > 0:
            dk_rel = abs(k_eff - k_eff_old) / abs(k_eff_old)
        else:
            dk_rel = 1.0

        dT_max = np.max(np.abs(T_fuel - T_old))
        dT_rel = np.max(np.abs((T_fuel - T_old) / np.maximum(T_old, 1.0)))

        if verbose:
            print(f"  {n_iter:4d}  {k_eff:10.6f}  {dk_rel:10.2e}  "
                  f"{T_fuel[0]:10.1f}  {T_fuel[-1]:10.1f}  {dT_max:10.2f}")

        if dk_rel < solver.tol_keff and dT_rel < solver.tol_temperature:
            converged = True
            if verbose:
                print("-" * 70)
                print(f"  Converged in {n_iter} iterations!")
                print(f"  k_eff = {k_eff:.6f}")
                print(f"  T_centerline = {T_fuel[0]:.1f} K")
                print(f"  T_surface = {T_fuel[-1]:.1f} K")
                print("=" * 70)
            break

        k_eff_old = k_eff

    if not converged and verbose:
        print("-" * 70)
        print(f"  WARNING: Did not converge after {n_iter} iterations!")
        print(f"  Last dk/k = {dk_rel:.2e}, last dT_max = {dT_max:.2f} K")
        print("=" * 70)

    return CoupledSolution(
        r_neutronics=r_full_cm,
        r_fuel=r_fuel_cm,
        phi1=phi1,
        phi2=phi2,
        temperature=T_fuel,
        q_volumetric=q_volumetric_si,
        k_eff=k_eff,
        n_iterations=n_iter,
        converged=converged,
    )


# =============================================================================
# CLI entry point: run the solver from the command line
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the coupled neutronics-thermal solver."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--power", type=float, default=200.0,
        help="Target linear heat rate [W/cm]."
    )
    args = parser.parse_args()

    # Load configuration
    cfg = ProblemConfig.from_yaml(args.config)
    cfg.validate()

    # Run solver
    solution = solve_coupled(cfg, power_level=args.power, verbose=True)

    # Quick summary plot (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Neutron fluxes (full pin cell)
        axes[0].plot(solution.r_neutronics, solution.phi1, label="Fast (group 1)")
        axes[0].plot(solution.r_neutronics, solution.phi2, label="Thermal (group 2)")
        axes[0].axvline(x=solution.r_fuel[-1], color="gray", ls="--", lw=0.8, label="Fuel/mod boundary")
        axes[0].set_xlabel("r [cm]")
        axes[0].set_ylabel("Neutron flux [a.u.]")
        axes[0].legend()
        axes[0].set_title(f"Neutron flux (k_eff = {solution.k_eff:.5f})")

        # Temperature profile (fuel only)
        axes[1].plot(solution.r_fuel, solution.temperature, color="tab:red")
        axes[1].set_xlabel("r [cm]")
        axes[1].set_ylabel("Temperature [K]")
        axes[1].set_title("Fuel temperature profile")

        # Heat generation (fuel only)
        axes[2].plot(solution.r_fuel, solution.q_volumetric / 1e6, color="tab:orange")
        axes[2].set_xlabel("r [cm]")
        axes[2].set_ylabel("q''' [MW/m³]")
        axes[2].set_title("Volumetric heat generation")

        plt.tight_layout()
        plt.savefig("results/coupled_solution.png", dpi=150)
        print(f"\nPlot saved to results/coupled_solution.png")
        plt.show()
    except ImportError:
        print("\nmatplotlib not available — skipping plot.")
