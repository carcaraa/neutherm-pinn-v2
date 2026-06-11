"""
Comparison of Solver vs Surrogate vs PINN.

Generates side-by-side plots of flux, temperature, and heat generation
profiles, plus error tables and training cost comparison.

Usage:
    python -m neutherm.evaluation.compare --config configs/default.yaml
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from neutherm.physics.parameters import ProblemConfig
from neutherm.solvers.coupled_solver import solve_coupled
from neutherm.models.surrogate import SurrogateModel
from neutherm.models.pinn import PINNModel
from neutherm.evaluation.metrics import relative_l2, relative_linf


def load_surrogate(
    path: str, config: ProblemConfig, device: torch.device
) -> tuple[SurrogateModel, dict | None]:
    """Load a trained surrogate model from checkpoint.

    The architecture is read from the checkpoint itself when available
    (checkpoints created after the fix store hidden_layers/activation),
    falling back to the YAML config for older checkpoints.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    mc = checkpoint["model_config"]

    model = SurrogateModel(
        n_inputs=mc["n_inputs"],
        n_radial_fuel=mc["n_radial_fuel"],
        n_radial_total=mc["n_radial_total"],
        hidden_layers=mc.get("hidden_layers", config.surrogate.hidden_layers),
        activation=mc.get("activation", config.surrogate.activation),
    ).to(device)

    # Initialize normalizer with correct shape before loading state_dict.
    # The actual values will be overwritten by load_state_dict.
    model.set_normalizer(
        torch.zeros(mc["n_inputs"], device=device),
        torch.ones(mc["n_inputs"], device=device),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint.get("norm_stats", None)


def load_pinn(path: str, config: ProblemConfig, device: torch.device) -> tuple:
    """Load a trained PINN model from checkpoint.

    Returns
    -------
    (model, k_eff, phi_scale, T_base)
        phi_scale converts the network's O(1) shape flux into the
        physical, power-normalized flux. T_base is the temperature
        shift used during training. Both fall back to safe defaults
        for checkpoints created before these fields existed.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Use checkpoint geometry/architecture if available, else config
    r_fuel = checkpoint.get("r_fuel", config.geometry.r_fuel * 100)
    r_cell = checkpoint.get("r_cell", config.geometry.r_cell * 100)

    model = PINNModel(
        hidden_layers=checkpoint.get("hidden_layers", config.pinn.hidden_layers),
        activation=checkpoint.get("activation", config.pinn.activation),
        r_fuel=r_fuel,
        r_cell=r_cell,
        T_out_scale=checkpoint.get("T_out_scale", 1.0),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    k_eff = checkpoint["k_eff"]
    phi_scale = checkpoint.get("phi_scale", 1.0)
    T_base = checkpoint.get("T_base")
    if T_base is None:  # legacy checkpoints used T_coolant + 300
        T_base = float(config.thermal.T_coolant + 300.0)
    return model, k_eff, phi_scale, T_base


def predict_surrogate(
    model: SurrogateModel,
    norm_stats: dict,
    params: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run surrogate prediction and denormalize outputs."""
    with torch.no_grad():
        x = torch.tensor(params, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x)

        # Denormalize if norm_stats available
        if norm_stats is not None:
            ns = norm_stats
            phi1 = (pred["phi1"].cpu().numpy()[0] * ns["phi1_std"].cpu().numpy()
                    + ns["phi1_mean"].cpu().numpy())
            phi2 = (pred["phi2"].cpu().numpy()[0] * ns["phi2_std"].cpu().numpy()
                    + ns["phi2_mean"].cpu().numpy())
            temp = (pred["temperature"].cpu().numpy()[0] * ns["temp_std"].cpu().numpy()
                    + ns["temp_mean"].cpu().numpy())
            keff = (pred["k_eff"].cpu().numpy()[0, 0] * ns["keff_std"].cpu().numpy()
                    + ns["keff_mean"].cpu().numpy())
        else:
            phi1 = pred["phi1"].cpu().numpy()[0]
            phi2 = pred["phi2"].cpu().numpy()[0]
            temp = pred["temperature"].cpu().numpy()[0]
            keff = pred["k_eff"].cpu().numpy()[0, 0]

    return {"phi1": phi1, "phi2": phi2, "temperature": temp, "k_eff": float(keff)}


def predict_pinn(
    model: PINNModel,
    r_cm: np.ndarray,
    T_base: float,
    device: torch.device,
    phi_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Run PINN prediction on given radial points.

    phi_scale converts the network's O(1) shape flux to physical units
    (power-normalized), making the fluxes directly comparable to the
    reference solver output.
    """
    with torch.no_grad():
        r_t = torch.tensor(r_cm, dtype=torch.float32).unsqueeze(1).to(device)
        out = model(r_t)
        phi1 = phi_scale * out["phi1"].cpu().numpy().squeeze()
        phi2 = phi_scale * out["phi2"].cpu().numpy().squeeze()
        T = T_base + out["T"].cpu().numpy().squeeze()

    return {"phi1": phi1, "phi2": phi2, "temperature": T}


def run_comparison(config: ProblemConfig, surrogate_path: str, pinn_path: str):
    """Run the full comparison: solver → surrogate → PINN → plots + metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # 1. Reference solver solution
    # =========================================================================
    print("Running reference solver...")
    ref = solve_coupled(config, power_level=200.0, verbose=False)
    print(f"  Solver k_eff = {ref.k_eff:.6f}")

    # =========================================================================
    # 2. Surrogate prediction
    # =========================================================================
    print("Loading surrogate model...")
    surr_model, norm_stats = load_surrogate(surrogate_path, config, device)

    # Input params: default config values
    params = np.array([
        config.thermal.T_coolant,
        config.geometry.r_fuel,
        1.0,  # enrichment factor = 1 (reference)
    ])
    surr_pred = predict_surrogate(surr_model, norm_stats, params, device)
    print(f"  Surrogate k_eff = {surr_pred['k_eff']:.6f}")

    # =========================================================================
    # 3. PINN prediction
    # =========================================================================
    print("Loading PINN model...")
    pinn_model, pinn_keff, pinn_phi_scale, pinn_T_base = load_pinn(
        pinn_path, config, device
    )
    # Fluxes live on the full pin cell (the PINN domain); the temperature
    # head is only physically meaningful inside the fuel, so evaluate it
    # on the fuel mesh.
    pinn_flux = predict_pinn(
        pinn_model, ref.r_neutronics, pinn_T_base, device, phi_scale=pinn_phi_scale
    )
    pinn_temp = predict_pinn(
        pinn_model, ref.r_fuel, pinn_T_base, device, phi_scale=pinn_phi_scale
    )
    pinn_pred = {
        "phi1": pinn_flux["phi1"],
        "phi2": pinn_flux["phi2"],
        "temperature": pinn_temp["temperature"],
    }
    print(f"  PINN k_eff = {pinn_keff:.6f}  (phi_scale = {pinn_phi_scale:.3e})")

    # =========================================================================
    # 4. Compute error metrics
    # =========================================================================
    # Surrogate errors: fluxes on the full pin cell, temperature in the fuel
    surr_errors = {
        "phi1_L2": relative_l2(surr_pred["phi1"], ref.phi1),
        "phi2_L2": relative_l2(surr_pred["phi2"], ref.phi2),
        "temp_L2": relative_l2(surr_pred["temperature"], ref.temperature),
        "keff_rel": abs(surr_pred["k_eff"] - ref.k_eff) / ref.k_eff,
    }

    # PINN errors: same domains (full cell for fluxes, fuel for temperature)
    pinn_errors = {
        "phi1_L2": relative_l2(pinn_pred["phi1"], ref.phi1),
        "phi2_L2": relative_l2(pinn_pred["phi2"], ref.phi2),
        "temp_L2": relative_l2(pinn_pred["temperature"], ref.temperature),
        "keff_rel": abs(pinn_keff - ref.k_eff) / ref.k_eff,
    }

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'Metric':<25s} {'Surrogate':>15s} {'PINN':>15s}")
    print("-" * 65)
    print(f"{'k_eff':<25s} {surr_pred['k_eff']:>15.6f} {pinn_keff:>15.6f}")
    print(f"{'k_eff (reference)':<25s} {ref.k_eff:>15.6f} {ref.k_eff:>15.6f}")
    print(f"{'k_eff rel. error':<25s} {surr_errors['keff_rel']:>14.4%} {pinn_errors['keff_rel']:>14.4%}")
    print(f"{'phi1 rel. L2':<25s} {surr_errors['phi1_L2']:>14.4%} {pinn_errors['phi1_L2']:>14.4%}")
    print(f"{'phi2 rel. L2':<25s} {surr_errors['phi2_L2']:>14.4%} {pinn_errors['phi2_L2']:>14.4%}")
    print(f"{'Temperature rel. L2':<25s} {surr_errors['temp_L2']:>14.4%} {pinn_errors['temp_L2']:>14.4%}")
    print("-" * 65)
    print(f"{'Training data needed':<25s} {'5000 samples':>15s} {'1 ref. solve':>15s}")
    print(f"{'Parameters':<25s} {surr_model.count_parameters():>15,d} {pinn_model.count_parameters():>15,d}")
    print("=" * 65)

    # =========================================================================
    # 5. Comparison plots
    # =========================================================================
    r_all = ref.r_neutronics   # full pin cell mesh [cm]
    r_fuel = ref.r_fuel        # fuel-only mesh [cm]
    R_fuel_cm = config.geometry.r_fuel * 100.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # --- Row 1: Field profiles ---
    # Fast flux (full pin cell)
    axes[0, 0].plot(r_all, ref.phi1, "k-", lw=2, label="Solver")
    axes[0, 0].plot(r_all, surr_pred["phi1"], "b--", lw=1.5, label="Surrogate")
    axes[0, 0].plot(r_all, pinn_pred["phi1"], "r:", lw=1.5, label="PINN")
    axes[0, 0].axvline(R_fuel_cm, color="gray", ls="--", lw=0.8, alpha=0.7)
    axes[0, 0].set_xlabel("r [cm]")
    axes[0, 0].set_ylabel("Fast flux φ₁ [n/cm²·s]")
    axes[0, 0].set_title("Fast-group neutron flux")
    axes[0, 0].legend()

    # Thermal flux (full pin cell)
    axes[0, 1].plot(r_all, ref.phi2, "k-", lw=2, label="Solver")
    axes[0, 1].plot(r_all, surr_pred["phi2"], "b--", lw=1.5, label="Surrogate")
    axes[0, 1].plot(r_all, pinn_pred["phi2"], "r:", lw=1.5, label="PINN")
    axes[0, 1].axvline(R_fuel_cm, color="gray", ls="--", lw=0.8, alpha=0.7)
    axes[0, 1].set_xlabel("r [cm]")
    axes[0, 1].set_ylabel("Thermal flux φ₂ [n/cm²·s]")
    axes[0, 1].set_title("Thermal-group neutron flux")
    axes[0, 1].legend()

    # Temperature (fuel only)
    axes[0, 2].plot(r_fuel, ref.temperature, "k-", lw=2, label="Solver")
    axes[0, 2].plot(r_fuel, surr_pred["temperature"], "b--", lw=1.5, label="Surrogate")
    axes[0, 2].plot(r_fuel, pinn_pred["temperature"], "r:", lw=1.5, label="PINN")
    axes[0, 2].set_xlabel("r [cm]")
    axes[0, 2].set_ylabel("Temperature [K]")
    axes[0, 2].set_title("Fuel temperature profile")
    axes[0, 2].legend()

    # --- Row 2: Errors ---
    from neutherm.evaluation.metrics import pointwise_relative_error

    # Fast flux error (full pin cell)
    surr_err_phi1 = pointwise_relative_error(surr_pred["phi1"], ref.phi1)
    pinn_err_phi1 = pointwise_relative_error(pinn_pred["phi1"], ref.phi1)
    axes[1, 0].semilogy(r_all, surr_err_phi1, "b-", label="Surrogate")
    axes[1, 0].semilogy(r_all, pinn_err_phi1, "r-", label="PINN")
    axes[1, 0].axvline(R_fuel_cm, color="gray", ls="--", lw=0.8, alpha=0.7)
    axes[1, 0].set_xlabel("r [cm]")
    axes[1, 0].set_ylabel("Relative error")
    axes[1, 0].set_title("Fast flux — pointwise error")
    axes[1, 0].legend()

    # Thermal flux error (full pin cell)
    surr_err_phi2 = pointwise_relative_error(surr_pred["phi2"], ref.phi2)
    pinn_err_phi2 = pointwise_relative_error(pinn_pred["phi2"], ref.phi2)
    axes[1, 1].semilogy(r_all, surr_err_phi2, "b-", label="Surrogate")
    axes[1, 1].semilogy(r_all, pinn_err_phi2, "r-", label="PINN")
    axes[1, 1].axvline(R_fuel_cm, color="gray", ls="--", lw=0.8, alpha=0.7)
    axes[1, 1].set_xlabel("r [cm]")
    axes[1, 1].set_ylabel("Relative error")
    axes[1, 1].set_title("Thermal flux — pointwise error")
    axes[1, 1].legend()

    # Temperature error
    surr_err_T = pointwise_relative_error(surr_pred["temperature"], ref.temperature)
    pinn_err_T = pointwise_relative_error(pinn_pred["temperature"], ref.temperature)
    axes[1, 2].semilogy(r_fuel, surr_err_T, "b-", label="Surrogate")
    axes[1, 2].semilogy(r_fuel, pinn_err_T, "r-", label="PINN")
    axes[1, 2].set_xlabel("r [cm]")
    axes[1, 2].set_ylabel("Relative error")
    axes[1, 2].set_title("Temperature — pointwise error")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=150)
    print(f"\nComparison plot saved to results/comparison.png")
    plt.show()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Solver vs Surrogate vs PINN.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--surrogate", type=str, default="results/surrogate_model.pt")
    parser.add_argument("--pinn", type=str, default="results/pinn_model.pt")
    args = parser.parse_args()

    cfg = ProblemConfig.from_yaml(args.config)
    run_comparison(cfg, args.surrogate, args.pinn)