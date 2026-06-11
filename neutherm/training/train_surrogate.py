"""
Training pipeline for the data-driven surrogate model.

This script handles the complete training workflow:
1. Load the parametric dataset (from Etapa 4)
2. Normalize all outputs to zero mean / unit variance
3. Split into train / validation / test sets
4. Build the surrogate model and optimizer
5. Train with early stopping and learning rate scheduling
6. Save the best model checkpoint and training curves

The surrogate is trained purely on data — no physics in the loss function.
This serves as the baseline to compare against the PINN (Etapa 6).

Key design decision: OUTPUT NORMALIZATION
    The flux fields (φ₁, φ₂) are ~10^14, temperature ~10^3, k_eff ~1.
    Without normalization, the network ignores fluxes because their
    contribution to the loss is drowned out by the other fields.
    We normalize ALL outputs to zero mean, unit variance using
    statistics from the training set only (no data leakage).

Usage
-----
From the command line:
    python -m neutherm.training.train_surrogate \\
        --config configs/default.yaml \\
        --data data/dataset.npz
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neutherm.physics.parameters import ProblemConfig
from neutherm.models.surrogate import SurrogateModel
from neutherm.training.dataset import load_dataset, ParametricDataset
from neutherm.training.losses import WeightedMSELoss, relative_l2_error


@dataclass
class NormStats:
    """Normalization statistics for each output field.

    Computed from the training set and applied to all sets.
    Used to denormalize predictions back to physical units.

    The formula is:  x_norm = (x - mean) / std
    To recover:      x = x_norm * std + mean
    """

    phi1_mean: torch.Tensor
    phi1_std: torch.Tensor
    phi2_mean: torch.Tensor
    phi2_std: torch.Tensor
    temp_mean: torch.Tensor
    temp_std: torch.Tensor
    keff_mean: torch.Tensor
    keff_std: torch.Tensor


@dataclass
class TrainingHistory:
    """Records training metrics across epochs."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_losses_by_field: dict[str, list[float]] = field(default_factory=dict)
    val_losses_by_field: dict[str, list[float]] = field(default_factory=dict)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor: (x - mean) / std, with epsilon to avoid division by zero."""
    return (x - mean) / (std + 1e-8)


def _denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize a tensor back to physical units: x * std + mean."""
    return x * std + mean


def prepare_data(
    dataset: ParametricDataset,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, NormStats]:
    """Convert the NumPy dataset into normalized PyTorch TensorDatasets.

    All output fields are normalized to zero mean and unit variance
    using statistics computed from the TRAINING set only. This prevents
    data leakage (val/test must not influence normalization).

    Parameters
    ----------
    dataset : ParametricDataset
        The full dataset from generate_dataset().
    train_split : float
        Fraction of data for training.
    val_split : float
        Fraction of data for validation.
    seed : int
        Random seed for the shuffle.

    Returns
    -------
    train_ds, val_ds, test_ds : TensorDataset
        Each contains (inputs, phi1_norm, phi2_norm, temp_norm, keff_norm).
    norm_stats : NormStats
        Statistics for denormalizing predictions back to physical units.
    """
    # Filter to only converged samples
    mask = dataset.converged
    n_total = int(np.sum(mask))

    # Convert to float32 tensors
    inputs = torch.tensor(dataset.params[mask], dtype=torch.float32)
    phi1 = torch.tensor(dataset.phi1[mask], dtype=torch.float32)
    phi2 = torch.tensor(dataset.phi2[mask], dtype=torch.float32)
    temperature = torch.tensor(dataset.temperature[mask], dtype=torch.float32)
    k_eff = torch.tensor(dataset.k_eff[mask], dtype=torch.float32).unsqueeze(1)

    # Random shuffle
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)

    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # Compute normalization stats FROM TRAINING SET ONLY
    # Using global mean/std across all samples and all radial points
    norm_stats = NormStats(
        phi1_mean=phi1[train_idx].mean(),
        phi1_std=phi1[train_idx].std(),
        phi2_mean=phi2[train_idx].mean(),
        phi2_std=phi2[train_idx].std(),
        temp_mean=temperature[train_idx].mean(),
        temp_std=temperature[train_idx].std(),
        keff_mean=k_eff[train_idx].mean(),
        keff_std=k_eff[train_idx].std(),
    )

    # Normalize ALL sets using training stats
    phi1_n = _normalize(phi1, norm_stats.phi1_mean, norm_stats.phi1_std)
    phi2_n = _normalize(phi2, norm_stats.phi2_mean, norm_stats.phi2_std)
    temp_n = _normalize(temperature, norm_stats.temp_mean, norm_stats.temp_std)
    keff_n = _normalize(k_eff, norm_stats.keff_mean, norm_stats.keff_std)

    def make_ds(idx):
        return TensorDataset(
            inputs[idx], phi1_n[idx], phi2_n[idx], temp_n[idx], keff_n[idx]
        )

    train_ds = make_ds(train_idx)
    val_ds = make_ds(val_idx)
    test_ds = make_ds(test_idx)

    print(f"Data split: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    return train_ds, val_ds, test_ds, norm_stats


def train_surrogate(
    config: ProblemConfig,
    dataset: ParametricDataset,
    device: str = "auto",
    verbose: bool = True,
) -> tuple[SurrogateModel, TrainingHistory, NormStats]:
    """Train the surrogate model on the parametric dataset.

    Training details:
    - Optimizer: Adam (adaptive learning rate per parameter)
    - LR scheduler: Cosine annealing (smooth decay to near-zero)
    - Loss: MSE on normalized outputs (all fields on the same scale)
    - Best model saved based on validation loss

    Parameters
    ----------
    config : ProblemConfig
        Configuration with surrogate hyperparameters.
    dataset : ParametricDataset
        Training data from generate_dataset().
    device : str
        "auto" (detect GPU), "cuda", or "cpu".
    verbose : bool
        Print training progress.

    Returns
    -------
    model : SurrogateModel
        The trained model (best validation checkpoint).
    history : TrainingHistory
        Training metrics across all epochs.
    norm_stats : NormStats
        Output normalization statistics for denormalizing predictions.
    """
    surr = config.surrogate

    # --- Device selection ---
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if verbose:
        print(f"Training on: {device}")

    # --- Prepare data (with output normalization) ---
    train_ds, val_ds, test_ds, norm_stats = prepare_data(
        dataset,
        train_split=surr.train_split,
        val_split=surr.val_split,
    )

    train_loader = DataLoader(
        train_ds, batch_size=surr.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=surr.batch_size, shuffle=False)

    # --- Build model ---
    n_radial_fuel = config.geometry.n_radial
    n_radial_total = config.geometry.n_radial + config.geometry.n_radial_mod

    model = SurrogateModel(
        n_inputs=3,
        n_radial_fuel=n_radial_fuel,
        n_radial_total=n_radial_total,
        hidden_layers=surr.hidden_layers,
        activation=surr.activation,
    ).to(device)

    # Set input normalization from training data
    all_inputs = train_ds.tensors[0]
    input_mean = all_inputs.mean(dim=0).to(device)
    input_std = all_inputs.std(dim=0).to(device)
    model.set_normalizer(input_mean, input_std)

    if verbose:
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Hidden layers: {surr.hidden_layers}")
        print(f"Activation: {surr.activation}")

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=surr.learning_rate)

    if surr.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=surr.epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    # --- Loss function ---
    # Since outputs are already normalized, we use plain MSE with equal weights.
    # No need for variance normalization in the loss — it's already done in the data.
    criterion = WeightedMSELoss(
        w_phi1=1.0, w_phi2=1.0, w_temperature=1.0, w_keff=1.0,
        normalize_by_variance=False,  # Already normalized in prepare_data
    )

    # --- Training loop ---
    history = TrainingHistory()
    best_state = None

    for epoch in range(1, surr.epochs + 1):
        # === Training phase ===
        model.train()
        epoch_losses = {}
        n_batches = 0

        for batch in train_loader:
            inputs_b, phi1_b, phi2_b, temp_b, keff_b = [
                t.to(device) for t in batch
            ]

            predictions = model(inputs_b)
            targets = {
                "phi1": phi1_b, "phi2": phi2_b,
                "temperature": temp_b, "k_eff": keff_b,
            }
            losses = criterion(predictions, targets)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        if scheduler is not None:
            scheduler.step()

        # === Validation phase ===
        model.eval()
        val_losses = {}
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs_b, phi1_b, phi2_b, temp_b, keff_b = [
                    t.to(device) for t in batch
                ]
                predictions = model(inputs_b)
                targets = {
                    "phi1": phi1_b, "phi2": phi2_b,
                    "temperature": temp_b, "k_eff": keff_b,
                }
                losses = criterion(predictions, targets)

                for k, v in losses.items():
                    val_losses[k] = val_losses.get(k, 0.0) + v.item()
                n_val_batches += 1

        for k in val_losses:
            val_losses[k] /= n_val_batches

        # === Record history ===
        history.train_loss.append(epoch_losses["total"])
        history.val_loss.append(val_losses["total"])
        history.learning_rates.append(optimizer.param_groups[0]["lr"])

        for k in epoch_losses:
            if k != "total":
                history.train_losses_by_field.setdefault(k, []).append(epoch_losses[k])
                history.val_losses_by_field.setdefault(k, []).append(val_losses[k])

        # === Save best model ===
        if val_losses["total"] < history.best_val_loss:
            history.best_val_loss = val_losses["total"]
            history.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # === Print progress ===
        if verbose and (epoch % max(1, surr.epochs // 20) == 0 or epoch == 1):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:5d}/{surr.epochs}  "
                f"train={epoch_losses['total']:.6f}  "
                f"val={val_losses['total']:.6f}  "
                f"lr={lr:.2e}  "
                f"{'*' if epoch == history.best_epoch else ''}"
            )

    # Load best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    if verbose:
        print(f"\nBest model at epoch {history.best_epoch} "
              f"(val_loss={history.best_val_loss:.6f})")

    # === Evaluate on test set (in PHYSICAL units) ===
    if verbose:
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

        # Move norm stats to the training device so denormalization works
        # on GPU as well (predictions live on `device`; the stats were
        # computed on CPU tensors in prepare_data).
        ns = NormStats(
            **{
                f.name: getattr(norm_stats, f.name).to(device)
                for f in NormStats.__dataclass_fields__.values()
            }
        )

        with torch.no_grad():
            for batch in test_loader:
                inputs_b, phi1_b, phi2_b, temp_b, keff_b = [
                    t.to(device) for t in batch
                ]
                predictions = model(inputs_b)

                # Denormalize predictions and targets back to physical units
                pred_phi1 = _denormalize(predictions["phi1"], ns.phi1_mean, ns.phi1_std)
                pred_phi2 = _denormalize(predictions["phi2"], ns.phi2_mean, ns.phi2_std)
                pred_temp = _denormalize(predictions["temperature"], ns.temp_mean, ns.temp_std)
                pred_keff = _denormalize(predictions["k_eff"], ns.keff_mean, ns.keff_std)

                tgt_phi1 = _denormalize(phi1_b, ns.phi1_mean, ns.phi1_std)
                tgt_phi2 = _denormalize(phi2_b, ns.phi2_mean, ns.phi2_std)
                tgt_temp = _denormalize(temp_b, ns.temp_mean, ns.temp_std)
                tgt_keff = _denormalize(keff_b, ns.keff_mean, ns.keff_std)

                # Relative L2 errors in physical units
                err_phi1 = relative_l2_error(pred_phi1, tgt_phi1)
                err_phi2 = relative_l2_error(pred_phi2, tgt_phi2)
                err_temp = relative_l2_error(pred_temp, tgt_temp)
                err_keff = relative_l2_error(pred_keff, tgt_keff)

                print(f"\nTest set relative L2 errors:")
                print(f"  phi1:        {err_phi1:.4f} ({err_phi1*100:.2f}%)")
                print(f"  phi2:        {err_phi2:.4f} ({err_phi2*100:.2f}%)")
                print(f"  temperature: {err_temp:.4f} ({err_temp*100:.2f}%)")
                print(f"  k_eff:       {err_keff:.4f} ({err_keff*100:.2f}%)")

    return model, history, norm_stats


def save_model(
    model: SurrogateModel,
    history: TrainingHistory,
    norm_stats: NormStats,
    path: str | Path,
):
    """Save the trained model, training history, and normalization stats.

    Parameters
    ----------
    model : SurrogateModel
        The trained model.
    history : TrainingHistory
        Training metrics.
    norm_stats : NormStats
        Output normalization statistics.
    path : str or Path
        Output file path (should end in .pt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "norm_stats": {
                "phi1_mean": norm_stats.phi1_mean,
                "phi1_std": norm_stats.phi1_std,
                "phi2_mean": norm_stats.phi2_mean,
                "phi2_std": norm_stats.phi2_std,
                "temp_mean": norm_stats.temp_mean,
                "temp_std": norm_stats.temp_std,
                "keff_mean": norm_stats.keff_mean,
                "keff_std": norm_stats.keff_std,
            },
            "history": {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
                "train_losses_by_field": history.train_losses_by_field,
                "val_losses_by_field": history.val_losses_by_field,
                "learning_rates": history.learning_rates,
                "best_epoch": history.best_epoch,
                "best_val_loss": history.best_val_loss,
            },
            "model_config": {
                "n_inputs": model.n_inputs,
                "n_radial_fuel": model.n_radial_fuel,
                "n_radial_total": model.n_radial_total,
                "hidden_layers": model.hidden_layers,
                "activation": model.activation_name,
            },
        },
        path,
    )
    print(f"Model saved to {path}")


def plot_training_curves(history: TrainingHistory, save_path: str | Path | None = None):
    """Plot training and validation loss curves.

    Parameters
    ----------
    history : TrainingHistory
        Training metrics.
    save_path : str or Path, optional
        If provided, save the plot instead of showing.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.train_loss) + 1)
    axes[0].semilogy(epochs, history.train_loss, label="Train", alpha=0.8)
    axes[0].semilogy(epochs, history.val_loss, label="Validation", alpha=0.8)
    axes[0].axvline(
        history.best_epoch, color="gray", ls="--", lw=0.8,
        label=f"Best (epoch {history.best_epoch})",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss (normalized)")
    axes[0].set_title("Training curves")
    axes[0].legend()

    axes[1].plot(epochs, history.learning_rates, color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("Learning rate schedule")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the data-driven surrogate model."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data", type=str, default="data/dataset.npz",
        help="Path to the training dataset.",
    )
    parser.add_argument(
        "--output", type=str, default="results/surrogate_model.pt",
        help="Path to save the trained model.",
    )
    args = parser.parse_args()

    # Load config and data
    cfg = ProblemConfig.from_yaml(args.config)
    dataset = load_dataset(args.data)

    # Train
    model, history, norm_stats = train_surrogate(cfg, dataset)

    # Save model and plots
    save_model(model, history, norm_stats, args.output)
    plot_training_curves(history, save_path="results/surrogate_training.png")