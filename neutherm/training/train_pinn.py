"""
Training pipeline for the PINN on the full pin cell domain.

The PINN loss has three components:
    L = λ_pde * L_pde + λ_bc * L_bc + λ_data * L_data

Key change from the fuel-only version: the PDE residuals now handle
TWO REGIONS with different physics:
- Fuel [0, R_fuel]: diffusion + fission + heat generation
- Moderator [R_fuel, R_cell]: diffusion + absorption (no fission, no heat)

Boundary conditions:
- r = 0: symmetry (dφ/dr = 0, dT/dr = 0)
- r = R_cell: reflective (dφ/dr = 0) — matches the solver's Wigner-Seitz BC
- r = R_fuel: continuity of flux and temperature (handled implicitly
  by the continuous neural network)

References
----------
.. [4] Raissi et al., "Physics-informed neural networks" (2019). JCP.
.. [6] Elhareef & Wu, "PINN for nuclear reactor calculations" (2023). ANE.
.. [7] Maddu et al., "Inverse Dirichlet weighting for PINNs" (2022).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neutherm.physics.parameters import ProblemConfig, NeutronicsParams, ThermalParams
from neutherm.physics.cross_sections import evaluate_cross_sections_torch
from neutherm.physics.fuel_properties import fuel_conductivity_torch
from neutherm.models.pinn import PINNModel
from neutherm.solvers.thermal_solver import compute_surface_temperature


# =============================================================================
# Autograd derivative utilities
# =============================================================================


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute d(outputs)/d(inputs) via autograd. Creates graph for higher-order derivs."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


# =============================================================================
# PDE residuals for the full pin cell
# =============================================================================


def compute_fuel_residuals(
    model: PINNModel,
    r: torch.Tensor,
    k_eff: torch.Tensor,
    phi_scale: torch.Tensor,
    neutronics: NeutronicsParams,
    thermal: ThermalParams,
    T_base: float,
) -> dict[str, torch.Tensor]:
    """PDE residuals in the FUEL region: diffusion + fission + heat conduction.

    Three equations:
    1. Fast diffusion:   -D₁ (d²φ₁/dr² + 1/r dφ₁/dr) + Σ_r1 φ₁ = (1/k)(νΣ_f1 φ₁ + νΣ_f2 φ₂)
    2. Thermal diffusion: -D₂ (d²φ₂/dr² + 1/r dφ₂/dr) + Σ_a2 φ₂ = Σ_s12 φ₁
    3. Heat conduction:  -(k d²T/dr² + (k/r + dk/dT dT/dr) dT/dr) = κ (νΣ_f1 Φ₁ + νΣ_f2 Φ₂)

    Flux scaling
    ------------
    The network outputs O(1) "shape" fluxes φ_g. The PHYSICAL flux is
        Φ_g = phi_scale * φ_g
    with phi_scale a learnable positive scalar (≈ 10¹³–10¹⁴ n·cm⁻²·s⁻¹)
    fixed by the power-normalization constraint in the training loop.

    The diffusion equations are homogeneous in φ, so they use the O(1)
    shape directly (better numerical conditioning). The heat source is
    NOT homogeneous — it must use the physical flux, otherwise
    q''' ≈ κΣφ ~ 10⁻¹² W/cm³ and the thermal problem decouples entirely.
    """
    out = model(r)
    phi1, phi2, T_raw = out["phi1"], out["phi2"], out["T"]
    T = T_base + T_raw

    # Derivatives via autograd
    dphi1 = grad(phi1, r)
    dphi2 = grad(phi2, r)
    dT = grad(T_raw, r)
    d2phi1 = grad(dphi1, r)
    d2phi2 = grad(dphi2, r)
    d2T = grad(dT, r)

    # Cross sections (temperature-dependent, differentiable)
    T_flat = T.squeeze()
    xs = evaluate_cross_sections_torch(T_flat, neutronics)
    D1 = xs["D1"].unsqueeze(1)
    D2 = xs["D2"].unsqueeze(1)
    sigma_r1 = xs["sigma_r1"].unsqueeze(1)
    sigma_a2 = xs["sigma_a2"].unsqueeze(1)
    nu_sf1 = xs["nu_sigma_f1"].unsqueeze(1)
    nu_sf2 = xs["nu_sigma_f2"].unsqueeze(1)
    sigma_s12 = xs["sigma_s12"].unsqueeze(1)

    # Thermal conductivity and its derivative
    k_th = fuel_conductivity_torch(T_flat, thermal).unsqueeze(1)
    dk_dT = grad(k_th, T_raw)

    # Convert k from W/(m·K) to W/(cm·K) for unit consistency with r in [cm]
    k_cgs = k_th / 100.0
    dk_cgs = dk_dT / 100.0

    r_safe = torch.clamp(r, min=1e-6)

    # Fast group residual (shape flux — equation is homogeneous in φ)
    diff1 = -D1 * (d2phi1 + dphi1 / r_safe)
    fission = (1.0 / k_eff) * (nu_sf1 * phi1 + nu_sf2 * phi2)
    R_fast = diff1 + sigma_r1 * phi1 - fission

    # Thermal group residual (shape flux)
    diff2 = -D2 * (d2phi2 + dphi2 / r_safe)
    R_thermal = diff2 + sigma_a2 * phi2 - sigma_s12 * phi1

    # Heat conduction residual — uses the PHYSICAL flux Φ = phi_scale * φ
    # so that q''' has the correct magnitude [W/cm³] and the thermal
    # problem is actually coupled to the neutronics.
    heat_diff = -(k_cgs * d2T + (k_cgs / r_safe + dk_cgs * dT) * dT)
    q_source = thermal.kappa_fission * phi_scale * (nu_sf1 * phi1 + nu_sf2 * phi2)
    R_heat = heat_diff - q_source

    return {
        "R_fast": R_fast, "R_thermal": R_thermal, "R_heat": R_heat,
        "phi1": phi1, "phi2": phi2, "T": T, "q_source": q_source,
    }


def compute_moderator_residuals(
    model: PINNModel,
    r: torch.Tensor,
    neutronics: NeutronicsParams,
    T_base: float,
) -> dict[str, torch.Tensor]:
    """PDE residuals in the MODERATOR region: diffusion only, no fission, no heat.

    Two equations (no heat equation — moderator temperature is fixed):
    1. Fast diffusion:    -D₁_mod (d²φ₁/dr² + 1/r dφ₁/dr) + Σ_r1_mod φ₁ = 0
    2. Thermal diffusion: -D₂_mod (d²φ₂/dr² + 1/r dφ₂/dr) + Σ_a2_mod φ₂ = Σ_s12_mod φ₁

    Cross sections are CONSTANT in the moderator (no temperature dependence).
    """
    out = model(r)
    phi1, phi2 = out["phi1"], out["phi2"]

    dphi1 = grad(phi1, r)
    dphi2 = grad(phi2, r)
    d2phi1 = grad(dphi1, r)
    d2phi2 = grad(dphi2, r)

    # Moderator cross sections (constant)
    D1 = neutronics.D1_mod
    D2 = neutronics.D2_mod
    sigma_r1 = neutronics.sigma_r1_mod
    sigma_a2 = neutronics.sigma_a2_mod
    sigma_s12 = neutronics.sigma_s12_mod

    r_safe = torch.clamp(r, min=1e-6)

    # Fast group: no fission source in moderator
    R_fast = -D1 * (d2phi1 + dphi1 / r_safe) + sigma_r1 * phi1

    # Thermal group: scattering source from fast group
    R_thermal = -D2 * (d2phi2 + dphi2 / r_safe) + sigma_a2 * phi2 - sigma_s12 * phi1

    return {"R_fast": R_fast, "R_thermal": R_thermal, "phi1": phi1, "phi2": phi2}


def compute_k_balance(
    model: PINNModel,
    r_quad_cell: torch.Tensor,
    fuel_mask: torch.Tensor,
    neutronics: NeutronicsParams,
    T_base: float,
) -> torch.Tensor:
    """Integral neutron-balance estimate of k_eff for the current flux iterate.

    With reflective boundaries the net leakage out of the Wigner–Seitz cell
    vanishes, so for the converged solution

        k_eff = ∫ (νΣ_f1 φ₁ + νΣ_f2 φ₂) dV / ∫ (Σ_a1 φ₁ + Σ_a2 φ₂) dV

    with Σ_a1 = Σ_r1 − Σ_s1→2 (removal minus down-scatter) and dV = 2πr dr.
    Evaluating this on the *current* network fluxes is the PINN analogue of
    the power-iteration eigenvalue update: k_eff becomes a differentiable
    FUNCTION of the flux shape rather than a free optimization variable.

    This removes the spurious shape⊗eigenvalue degeneracy by construction —
    a free k_eff trained by gradient descent drifts towards the flat-flux
    k_inf mode (observed: k → 1.8–2.8 with the PDE loss stuck at the
    irreducible moderator residual), because for *any* shape there is a k
    that locally shrinks the fast-group residual. Tying k to the integral
    balance leaves the shape residuals as the only degrees of freedom,
    exactly as in the reference solver's outer iteration.
    """
    out = model(r_quad_cell)
    phi1, phi2 = out["phi1"], out["phi2"]
    T = T_base + out["T"]

    # Fuel-region cross sections (temperature-dependent, differentiable).
    # They are evaluated at every quadrature point but only used where
    # fuel_mask is 1; the moderator points use the constant moderator set.
    xs = evaluate_cross_sections_torch(T.squeeze(1), neutronics)
    nu_sf1_f = xs["nu_sigma_f1"].unsqueeze(1)
    nu_sf2_f = xs["nu_sigma_f2"].unsqueeze(1)
    sa1_f = (xs["sigma_r1"] - xs["sigma_s12"]).clamp(min=0.0).unsqueeze(1)
    sa2_f = xs["sigma_a2"].unsqueeze(1)

    sa1_m = max(neutronics.sigma_r1_mod - neutronics.sigma_s12_mod, 0.0)
    sa2_m = neutronics.sigma_a2_mod

    mask = fuel_mask.to(phi1.dtype).unsqueeze(1)
    production = mask * (nu_sf1_f * phi1 + nu_sf2_f * phi2)
    absorption = (
        mask * (sa1_f * phi1 + sa2_f * phi2)
        + (1.0 - mask) * (sa1_m * phi1 + sa2_m * phi2)
    )

    w = 2.0 * np.pi * r_quad_cell
    r_axis = r_quad_cell.squeeze(1)
    num = torch.trapezoid((production * w).squeeze(1), r_axis)
    den = torch.trapezoid((absorption * w).squeeze(1), r_axis)

    # Clamp for numerical robustness in the very first epochs, when the
    # untrained shape can make the ratio ill-conditioned.
    return (num / den.clamp(min=1e-12)).clamp(0.3, 3.5)


# =============================================================================
# Boundary condition losses
# =============================================================================


def compute_bc_loss(
    model: PINNModel,
    r_cell_cm: float,
    T_surface: float,
    T_base: float,
    T_scale: float = 100.0,
) -> torch.Tensor:
    """Boundary conditions for the full pin cell.

    1. r = 0: symmetry → dφ₁/dr = 0, dφ₂/dr = 0, dT/dr = 0
    2. r = R_cell: reflective → dφ₁/dr = 0, dφ₂/dr = 0
    3. r = R_fuel: T = T_surface (fuel surface temperature from gap model)

    Loss balancing
    --------------
    The flux-derivative terms are O(1) because the network outputs O(1)
    shape fluxes over an O(1) cm domain. The temperature mismatch,
    however, is measured in KELVIN: an unnormalized (T - T_surface)²
    starts at ~10⁴ K² and completely drowns the flux BCs (this was a
    real bug — the BC loss was ~99% temperature). We therefore divide
    the temperature mismatch by T_scale (a characteristic ΔT of ~100 K)
    so all three BC contributions live on comparable scales.
    """
    device = next(model.parameters()).device

    # --- r = 0: symmetry ---
    r0 = torch.zeros(1, 1, device=device, requires_grad=True)
    out0 = model(r0)
    bc_sym = (
        torch.mean(grad(out0["phi1"], r0) ** 2)
        + torch.mean(grad(out0["phi2"], r0) ** 2)
        + torch.mean(grad(out0["T"], r0) ** 2)
    )

    # --- r = R_cell: reflective (zero net current) ---
    r_cell = torch.full((1, 1), r_cell_cm, device=device, requires_grad=True)
    out_cell = model(r_cell)
    bc_refl = (
        torch.mean(grad(out_cell["phi1"], r_cell) ** 2)
        + torch.mean(grad(out_cell["phi2"], r_cell) ** 2)
    )

    # --- r = R_fuel: prescribed temperature (normalized by T_scale) ---
    r_fuel = torch.full(
        (1, 1), model.r_fuel, device=device, requires_grad=True
    )
    out_fuel = model(r_fuel)
    T_pred = T_base + out_fuel["T"]
    bc_temp = torch.mean(((T_pred - T_surface) / T_scale) ** 2)

    return bc_sym + bc_refl + bc_temp


# =============================================================================
# Training history
# =============================================================================


@dataclass
class PINNHistory:
    """Records PINN training metrics."""
    total_loss: list[float] = field(default_factory=list)
    pde_loss: list[float] = field(default_factory=list)
    bc_loss: list[float] = field(default_factory=list)
    data_loss: list[float] = field(default_factory=list)
    power_loss: list[float] = field(default_factory=list)
    k_eff_history: list[float] = field(default_factory=list)
    phi_scale_history: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_loss: float = float("inf")


# =============================================================================
# Main training function
# =============================================================================


def train_pinn(
    config: ProblemConfig,
    reference_solution=None,
    power_level: float = 200.0,
    device: str = "auto",
    verbose: bool = True,
) -> tuple[PINNModel, PINNHistory, torch.Tensor, torch.Tensor]:
    """Train the PINN on the full pin cell domain.

    Collocation points are sampled in BOTH fuel and moderator regions.
    Fuel points get fuel-physics residuals (diffusion + fission + heat).
    Moderator points get moderator-physics residuals (diffusion only).
    k_eff is a learnable parameter optimized jointly with the network.

    Power normalization
    -------------------
    The diffusion eigenvalue problem only fixes the flux SHAPE — the
    magnitude is arbitrary. The reference solver pins it down by
    normalizing to a target linear heat rate (power_level, W/cm). The
    PINN does the same through:
    - a learnable flux scale Φ = phi_scale · φ_net (phi_scale = exp(s)
      with s trainable, so the scale stays positive), and
    - a power-constraint loss ((q'_pred − power_level)/power_level)²
      where q'_pred = ∫ κ(νΣ_f1 Φ₁ + νΣ_f2 Φ₂) 2πr dr over the fuel.

    Without this constraint the heat source is ~10⁻¹² W/cm³ (flux O(1)),
    the heat equation decouples, the temperature field collapses onto
    the boundary condition and the Doppler feedback disappears — i.e.
    the PINN would NOT be solving the coupled problem at all.

    The fuel surface temperature BC is derived from the same gap/coolant
    resistance chain used by the reference solver (no hardcoded guess).

    Parameters
    ----------
    config : ProblemConfig
        Full problem configuration.
    reference_solution : CoupledSolution, optional
        If given, adds a data-fitting term (hybrid PINN).
    power_level : float
        Target linear heat rate [W/cm] — must match the value used by
        the reference solver for a fair comparison (default 200).
    device : str
        "auto", "cuda", or "cpu".
    verbose : bool
        Print training progress.

    Returns
    -------
    model : PINNModel
    history : PINNHistory
    k_eff : torch.Tensor (scalar, best epoch)
    phi_scale : torch.Tensor (scalar, best epoch)
    """
    pinn_cfg = config.pinn
    geom = config.geometry
    neutronics = config.neutronics
    thermal = config.thermal

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    R_fuel_cm = geom.r_fuel * 100
    R_cell_cm = geom.r_cell * 100

    # Fraction of collocation points in each region (proportional to area)
    fuel_area = R_fuel_cm ** 2
    cell_area = R_cell_cm ** 2
    fuel_frac = fuel_area / cell_area
    n_fuel_pts = int(pinn_cfg.n_collocation * fuel_frac)
    n_mod_pts = pinn_cfg.n_collocation - n_fuel_pts

    # --- Thermal boundary condition from the SAME physics as the solver ---
    # compute_surface_temperature expects the linear heat rate in W/m;
    # power_level is in W/cm → multiply by 100.
    T_surface = compute_surface_temperature(power_level * 100.0, geom, thermal)
    # Output shift: the network learns T_raw = T - T_base. Centering at the
    # surface temperature keeps T_raw at O(ΔT_fuel) ~ O(10²) K.
    T_base = float(T_surface)

    # --- Physical scale of the temperature head ---
    # Classic conduction estimate of the fuel temperature rise:
    #   ΔT_fuel ≈ q' / (4π k̄)     [q' in W/m, k̄ in W/(m·K)]
    # evaluated at a representative mid-fuel temperature. This conditions
    # the T head so its raw output stays O(1); without it the heat residual
    # (normalized by mean(q²)) yields vanishing gradients on T and training
    # stalls on a flat temperature profile.
    k_bar = float(
        fuel_conductivity_torch(
            torch.tensor(T_surface + 200.0), thermal
        ).item()
    )
    T_out_scale = (power_level * 100.0) / (4.0 * np.pi * max(k_bar, 1e-6))

    # Build model over full pin cell
    model = PINNModel(
        hidden_layers=pinn_cfg.hidden_layers,
        activation=pinn_cfg.activation,
        r_fuel=R_fuel_cm,
        r_cell=R_cell_cm,
        T_out_scale=T_out_scale,
    ).to(device)

    # k_eff is NOT a free learnable parameter: it is recomputed every epoch
    # from the integral neutron balance of the current flux iterate (see
    # compute_k_balance). A free k_eff optimized jointly with the network is
    # degenerate with the flux shape and drifts to the spurious k_inf mode.

    # --- Flux scale: learnable, parametrized as exp(s) to stay positive ---
    # Physics-based initial guess: with a flat flux Φ₀, the linear power is
    #   q' ≈ κ · νΣ_f2 · Φ₀ · π R_f²   →   Φ₀ ≈ q' / (κ νΣ_f2 π R_f²)
    phi0_estimate = power_level / (
        thermal.kappa_fission
        * max(neutronics.nu_sigma_f2_ref, 1e-12)
        * np.pi * R_fuel_cm ** 2
    )
    log_phi_scale = nn.Parameter(
        torch.tensor(float(np.log(phi0_estimate)), device=device)
    )

    if verbose:
        print("=" * 70)
        print("PINN Training (Full Pin Cell)")
        print("=" * 70)
        print(f"  Device: {device}")
        print(f"  Parameters: {model.count_parameters():,} + 1 (phi_scale); "
              f"k_eff via integral balance")
        print(f"  Domain: [0, {R_cell_cm:.4f}] cm (fuel: {R_fuel_cm:.4f} cm)")
        print(f"  Collocation: {n_fuel_pts} fuel + {n_mod_pts} moderator")
        print(f"  Target power: {power_level:.1f} W/cm")
        print(f"  T_surface (gap model): {T_surface:.1f} K")
        print(f"  T head scale (q'/4\u03c0k): {T_out_scale:.1f} K")
        print(f"  phi_scale init: {phi0_estimate:.3e}")
        print(f"  Epochs: {pinn_cfg.epochs}")
        print("-" * 70)
        if pinn_cfg.adaptive_weights:
            print("  [NOTE] adaptive_weights=True is not implemented yet; "
                  "using fixed lambda weights.")

    # Optimizer: two parameter groups.
    #   - network weights: base LR;
    #   - log_phi_scale: faster (×10) — it must travel far in log-space and
    #     the power-normalization constraint keeps its gradient stable.
    # k_eff is not optimized: it is recomputed from the integral neutron
    # balance every epoch (see compute_k_balance).
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": pinn_cfg.learning_rate},
        {"params": [log_phi_scale], "lr": pinn_cfg.learning_rate * 10},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pinn_cfg.epochs, eta_min=1e-6
    )

    lambda_pde = pinn_cfg.lambda_pde
    lambda_bc = pinn_cfg.lambda_bc
    lambda_data = pinn_cfg.lambda_data
    lambda_power = pinn_cfg.lambda_power

    # Fixed quadrature grid for the power integral (no spatial derivatives
    # needed here, only network evaluations — gradients flow to the weights
    # and to phi_scale through the integrand).
    n_quad = 128
    r_quad = torch.linspace(0.0, R_fuel_cm, n_quad, device=device).unsqueeze(1)

    # Quadrature grid over the WHOLE cell for the integral neutron balance
    # used to update k_eff each epoch (power-iteration analogue).
    n_quad_cell = 192
    r_quad_cell = torch.linspace(
        0.0, R_cell_cm, n_quad_cell, device=device
    ).unsqueeze(1)
    fuel_mask = r_quad_cell.squeeze(1) <= R_fuel_cm

    # --- Reference data (optional) ---
    has_data = reference_solution is not None
    if has_data:
        ref = reference_solution
        n_full = len(ref.r_neutronics)
        n_data = min(80, n_full)
        idx = np.linspace(0, n_full - 1, n_data, dtype=int)

        r_data = torch.tensor(ref.r_neutronics[idx], dtype=torch.float32).unsqueeze(1).to(device)
        phi1_data = torch.tensor(ref.phi1[idx], dtype=torch.float32).unsqueeze(1).to(device)
        phi2_data = torch.tensor(ref.phi2[idx], dtype=torch.float32).unsqueeze(1).to(device)

        # Temperature data: only available in fuel region
        n_fuel_data = config.geometry.n_radial
        idx_fuel = np.linspace(0, n_fuel_data - 1, min(50, n_fuel_data), dtype=int)
        r_data_fuel = torch.tensor(ref.r_fuel[idx_fuel], dtype=torch.float32).unsqueeze(1).to(device)
        T_data = torch.tensor(ref.temperature[idx_fuel], dtype=torch.float32).unsqueeze(1).to(device)

        # Normalization scales of the reference data (for the data loss)
        phi1_ref_scale = phi1_data.abs().max().clamp(min=1e-10)
        phi2_ref_scale = phi2_data.abs().max().clamp(min=1e-10)
        T_ref_scale = T_data.abs().max().clamp(min=1e-10)

        if verbose:
            print(f"  Reference data: {n_data} flux pts + {len(idx_fuel)} temp pts")
            print(f"  Reference k_eff: {ref.k_eff:.6f}")

    # --- Training loop ---
    history = PINNHistory()
    best_state = None
    best_keff = 1.0
    best_phi_scale = phi0_estimate

    for epoch in range(1, pinn_cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        phi_scale = torch.exp(log_phi_scale)

        # === Eigenvalue from the integral neutron balance ===
        # Differentiable function of the current flux shape (power-iteration
        # analogue) — NOT a free parameter; see compute_k_balance.
        k_eff = compute_k_balance(
            model, r_quad_cell, fuel_mask, neutronics, T_base
        )

        # === Sample collocation points ===
        # Fuel region: (0, R_fuel), avoiding r=0 singularity
        r_fuel_pts = (
            torch.rand(n_fuel_pts, 1, device=device) * R_fuel_cm * 0.98
            + R_fuel_cm * 0.01
        )
        r_fuel_pts.requires_grad_(True)

        # Moderator region: (R_fuel, R_cell)
        r_mod_pts = (
            torch.rand(n_mod_pts, 1, device=device) * (R_cell_cm - R_fuel_cm) * 0.98
            + R_fuel_cm * 1.01
        )
        r_mod_pts.requires_grad_(True)

        # === Fuel PDE residuals ===
        fuel_res = compute_fuel_residuals(
            model, r_fuel_pts, k_eff, phi_scale, neutronics, thermal, T_base
        )

        loss_fuel_fast = torch.mean(fuel_res["R_fast"] ** 2)
        loss_fuel_thermal = torch.mean(fuel_res["R_thermal"] ** 2)
        loss_fuel_heat = torch.mean(fuel_res["R_heat"] ** 2)

        # Normalize by field magnitudes (detached to avoid affecting gradients).
        # Flux equations: scale-invariant normalization by mean(φ²).
        # Heat equation: normalized by mean(q²) — residual and source share
        # the same units [W/cm³], so this keeps the term dimensionless and
        # O(1) regardless of the power level.
        with torch.no_grad():
            s_fast = torch.mean(fuel_res["phi1"] ** 2).clamp(min=1e-10)
            s_therm = torch.mean(fuel_res["phi2"] ** 2).clamp(min=1e-10)
            s_heat = torch.mean(fuel_res["q_source"] ** 2).clamp(min=1e-10)

        loss_pde_fuel = loss_fuel_fast / s_fast + loss_fuel_thermal / s_therm + loss_fuel_heat / s_heat

        # === Moderator PDE residuals ===
        mod_res = compute_moderator_residuals(model, r_mod_pts, neutronics, T_base)

        loss_mod_fast = torch.mean(mod_res["R_fast"] ** 2)
        loss_mod_thermal = torch.mean(mod_res["R_thermal"] ** 2)

        with torch.no_grad():
            sm_fast = torch.mean(mod_res["phi1"] ** 2).clamp(min=1e-10)
            sm_therm = torch.mean(mod_res["phi2"] ** 2).clamp(min=1e-10)

        loss_pde_mod = loss_mod_fast / sm_fast + loss_mod_thermal / sm_therm

        loss_pde = loss_pde_fuel + loss_pde_mod

        # === Power normalization constraint ===
        # q'_pred = ∫₀^{R_f} κ (νΣ_f1 Φ₁ + νΣ_f2 Φ₂) 2πr dr  [W/cm]
        out_q = model(r_quad)
        T_quad = (T_base + out_q["T"]).squeeze()
        xs_q = evaluate_cross_sections_torch(T_quad, neutronics)
        q_quad = thermal.kappa_fission * phi_scale * (
            xs_q["nu_sigma_f1"].unsqueeze(1) * out_q["phi1"]
            + xs_q["nu_sigma_f2"].unsqueeze(1) * out_q["phi2"]
        )
        integrand = (q_quad * 2.0 * np.pi * r_quad).squeeze()
        q_linear_pred = torch.trapezoid(integrand, r_quad.squeeze())
        loss_power = ((q_linear_pred - power_level) / power_level) ** 2

        # === Boundary conditions ===
        loss_bc = compute_bc_loss(model, R_cell_cm, T_surface, T_base)

        # === Data loss (optional) ===
        if has_data:
            # Flux data (full cell mesh) — the reference fluxes are PHYSICAL
            # (power-normalized, ~10¹³), so the network shape flux must be
            # multiplied by phi_scale before comparing. Without this, the
            # data term compares O(1) against O(10¹³) and can never fit.
            r_d = r_data.clone().requires_grad_(True)
            out_d = model(r_d)
            loss_data = (
                torch.mean(((phi_scale * out_d["phi1"] - phi1_data) / phi1_ref_scale) ** 2)
                + torch.mean(((phi_scale * out_d["phi2"] - phi2_data) / phi2_ref_scale) ** 2)
            )
            # Temperature data (fuel only)
            r_df = r_data_fuel.clone().requires_grad_(True)
            out_df = model(r_df)
            T_pred = T_base + out_df["T"]
            loss_data = loss_data + torch.mean(((T_pred - T_data) / T_ref_scale) ** 2)
        else:
            loss_data = torch.tensor(0.0, device=device)

        # === Total loss ===
        loss_total = (
            lambda_pde * loss_pde
            + lambda_bc * loss_bc
            + lambda_data * loss_data
            + lambda_power * loss_power
        )

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # === Record ===
        history.total_loss.append(loss_total.item())
        history.pde_loss.append(loss_pde.item())
        history.bc_loss.append(loss_bc.item())
        history.data_loss.append(loss_data.item())
        history.power_loss.append(loss_power.item())
        history.k_eff_history.append(k_eff.item())
        history.phi_scale_history.append(torch.exp(log_phi_scale).item())
        history.learning_rates.append(optimizer.param_groups[0]["lr"])

        if loss_total.item() < history.best_loss:
            history.best_loss = loss_total.item()
            history.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_keff = k_eff.item()
            best_phi_scale = torch.exp(log_phi_scale).item()

        if verbose and (epoch % max(1, pinn_cfg.epochs // 20) == 0 or epoch == 1):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:5d}/{pinn_cfg.epochs}  "
                f"total={loss_total.item():.6f}  "
                f"pde={loss_pde.item():.4f}  "
                f"bc={loss_bc.item():.4f}  "
                f"pw={loss_power.item():.4f}  "
                f"q'={q_linear_pred.item():7.1f}  "
                f"k_eff={k_eff.item():.5f}  "
                f"lr={lr:.2e}"
            )

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    if verbose:
        print("-" * 70)
        print(f"  Best epoch: {history.best_epoch}")
        print(f"  Learned k_eff: {best_keff:.6f}")
        print(f"  Learned phi_scale: {best_phi_scale:.4e}")
        if has_data:
            print(f"  Reference k_eff: {ref.k_eff:.6f}")
            print(f"  k_eff error: {abs(best_keff - ref.k_eff):.6f} "
                  f"({abs(best_keff - ref.k_eff)/ref.k_eff*100:.2f}%)")
        print("=" * 70)

    return model, history, torch.tensor(best_keff), torch.tensor(best_phi_scale)


def save_pinn(
    model: PINNModel,
    history: PINNHistory,
    k_eff: torch.Tensor,
    phi_scale: torch.Tensor,
    path: str | Path,
    T_base: float | None = None,
    T_surface: float | None = None,
    power_level: float | None = None,
):
    """Save trained PINN model, history, learned k_eff and flux scale.

    The checkpoint is self-contained: it stores the geometry, the
    architecture, the temperature shift (T_base) and the flux scale, so
    the model can be reloaded and evaluated without re-deriving any of
    these from a (possibly changed) YAML config.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "k_eff": k_eff.item(),
        "phi_scale": float(phi_scale.item()),
        "r_fuel": model.r_fuel,
        "r_cell": model.r_cell,
        "T_out_scale": model.T_out_scale,
        "hidden_layers": model.hidden_layers,
        "activation": model.activation_name,
        "T_base": T_base,
        "T_surface": T_surface,
        "power_level": power_level,
        "history": {
            "total_loss": history.total_loss,
            "pde_loss": history.pde_loss,
            "bc_loss": history.bc_loss,
            "data_loss": history.data_loss,
            "power_loss": history.power_loss,
            "k_eff_history": history.k_eff_history,
            "phi_scale_history": history.phi_scale_history,
            "learning_rates": history.learning_rates,
            "best_epoch": history.best_epoch,
            "best_loss": history.best_loss,
        },
    }, path)
    print(f"PINN saved to {path}")


def plot_pinn_training(history: PINNHistory, save_path: str | Path | None = None):
    """Plot PINN training curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history.total_loss) + 1)

    axes[0].semilogy(epochs, history.total_loss, label="Total", alpha=0.7)
    axes[0].semilogy(epochs, history.pde_loss, label="PDE", alpha=0.7)
    axes[0].semilogy(epochs, history.bc_loss, label="BC", alpha=0.7)
    if any(v > 0 for v in history.data_loss):
        axes[0].semilogy(epochs, history.data_loss, label="Data", alpha=0.7)
    if history.power_loss:
        axes[0].semilogy(epochs, history.power_loss, label="Power", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("PINN loss components")
    axes[0].legend()

    axes[1].plot(epochs, history.k_eff_history, color="tab:red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("k_eff")
    axes[1].set_title("Learned k_eff")

    axes[2].plot(epochs, history.learning_rates, color="tab:orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning rate schedule")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the PINN (full pin cell).")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--with-data", action="store_true",
                        help="Use reference solver solution as training data.")
    parser.add_argument("--power", type=float, default=200.0,
                        help="Target linear heat rate [W/cm] (must match the solver).")
    parser.add_argument("--output", type=str, default="results/pinn_model.pt")
    args = parser.parse_args()

    cfg = ProblemConfig.from_yaml(args.config)

    ref_solution = None
    if args.with_data:
        from neutherm.solvers.coupled_solver import solve_coupled
        print("Generating reference solution...")
        ref_solution = solve_coupled(cfg, power_level=args.power, verbose=False)
        print(f"Reference k_eff = {ref_solution.k_eff:.6f}")

    model, history, k_eff, phi_scale = train_pinn(
        cfg, reference_solution=ref_solution, power_level=args.power, verbose=True
    )
    T_surface = compute_surface_temperature(args.power * 100.0, cfg.geometry, cfg.thermal)
    save_pinn(
        model, history, k_eff, phi_scale, args.output,
        T_base=float(T_surface), T_surface=float(T_surface), power_level=args.power,
    )
    plot_pinn_training(history, save_path="results/pinn_training.png")