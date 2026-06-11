"""
Loss functions for training the surrogate model and PINN.

This module provides:
- Per-field MSE losses (flux, temperature, k_eff)
- Weighted composite loss combining all fields
- Relative L2 error metric for evaluation

The same loss functions are reused by the PINN (Etapa 6), which adds
physics-based residual terms on top of the data loss defined here.

Notes
-----
Why weighted losses?
    The output fields live on very different scales:
    - φ₁ ~ 10^14 (neutron flux)
    - T ~ 10^3 (temperature in Kelvin)
    - k_eff ~ 1 (dimensionless)

    Without weighting, the loss would be dominated by whichever field
    has the largest magnitude. We normalize each field's contribution
    so they all matter equally during training.
"""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """Weighted mean squared error loss across multiple output fields.

    Computes MSE for each field (phi1, phi2, temperature, k_eff) and
    combines them with configurable weights.

    Parameters
    ----------
    w_phi1 : float
        Weight for the fast-group flux loss.
    w_phi2 : float
        Weight for the thermal-group flux loss.
    w_temperature : float
        Weight for the temperature loss.
    w_keff : float
        Weight for the k_eff loss.
    normalize_by_variance : bool
        If True, divide each field's MSE by the variance of the target.
        Set to False when outputs are already normalized.
    """

    def __init__(
        self,
        w_phi1: float = 1.0,
        w_phi2: float = 1.0,
        w_temperature: float = 1.0,
        w_keff: float = 1.0,
        normalize_by_variance: bool = True,
    ):
        super().__init__()
        self.w_phi1 = w_phi1
        self.w_phi2 = w_phi2
        self.w_temperature = w_temperature
        self.w_keff = w_keff
        self.normalize_by_variance = normalize_by_variance

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute weighted MSE loss for all fields.

        Parameters
        ----------
        predictions : dict[str, torch.Tensor]
            Model predictions with keys: 'phi1', 'phi2', 'temperature', 'k_eff'.
        targets : dict[str, torch.Tensor]
            Ground truth with the same keys.

        Returns
        -------
        dict[str, torch.Tensor]
            Individual losses and the total:
            'loss_phi1', 'loss_phi2', 'loss_temperature', 'loss_k_eff', 'total'.
        """
        losses = {}

        for name, weight in [
            ("phi1", self.w_phi1),
            ("phi2", self.w_phi2),
            ("temperature", self.w_temperature),
            ("k_eff", self.w_keff),
        ]:
            pred = predictions[name]
            tgt = targets[name]

            # Mean squared error for this field
            mse = torch.mean((pred - tgt) ** 2)

            if self.normalize_by_variance:
                # Normalize by target variance so the loss is scale-invariant
                var = torch.var(tgt)
                if var > 1e-10:
                    mse = mse / var

            losses[f"loss_{name}"] = weight * mse

        # Total loss: weighted sum of all fields
        losses["total"] = sum(losses.values())

        return losses


def relative_l2_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute the relative L2 error between prediction and target.

    relative_L2 = ||pred - target||_2 / ||target||_2

    A value of 0.01 means the prediction is within 1% of the target.

    Parameters
    ----------
    prediction : torch.Tensor
        Model prediction, any shape.
    target : torch.Tensor
        Ground truth, same shape as prediction.

    Returns
    -------
    torch.Tensor
        Scalar relative L2 error.
    """
    numerator = torch.norm(prediction - target, p=2)
    denominator = torch.norm(target, p=2)
    return numerator / (denominator + 1e-10)