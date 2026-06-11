"""
Physics-Informed Neural Network for the full pin cell domain.

The PINN approximates the solution as a continuous function of radius:

    NN(r) → (φ₁(r), φ₂(r), T(r))

covering both the fuel region [0, R_fuel] and the moderator region
[R_fuel, R_cell]. This matches the solver domain exactly, enabling
a fair comparison of k_eff and flux profiles.

The network receives two inputs:
- r: radial position (normalized to [0, 1] over the full cell)
- region: a smooth indicator that transitions from 1 (fuel) to 0
  (moderator) near R_fuel. This helps the network learn the
  discontinuity in material properties at the fuel-moderator interface.

Why not a hard region split?
    A hard if/else would break autograd (no gradient through the branch).
    Instead we use a smooth sigmoid transition, which is differentiable
    everywhere. The network can learn the sharp material interface
    through this soft indicator.

References
----------
.. [4] Raissi et al., "Physics-informed neural networks" (2019). JCP.
.. [10] Wang et al., "When and why PINNs fail to train" (2022). JCP.
"""

import torch
import torch.nn as nn


class PINNModel(nn.Module):
    """Neural network approximating the coupled solution over the full pin cell.

    Inputs: (r, region_indicator) → 2 features
    Outputs: (φ₁, φ₂, T) → 3 fields

    The region indicator is computed internally from r:
        indicator(r) = sigmoid((R_fuel - r) / width)
    which is ~1 in fuel and ~0 in moderator, with a smooth transition.

    Parameters
    ----------
    hidden_layers : list[int]
        Width of each hidden layer.
    activation : str
        Activation function: "tanh" (recommended), "silu", or "gelu".
    r_fuel : float
        Fuel radius [cm] — location of the material interface.
    r_cell : float
        Cell radius [cm] — outer boundary of the domain.
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "tanh",
        r_fuel: float = 0.4096,
        r_cell: float = 0.7174,
        T_out_scale: float = 1.0,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]

        self.r_fuel = r_fuel
        self.r_cell = r_cell
        # Stored so checkpoints can rebuild the exact architecture without
        # relying on the YAML config still matching the trained model.
        self.hidden_layers = list(hidden_layers)
        self.activation_name = activation
        # Physical scale of the temperature head [K]. The raw network output
        # is O(1); multiplying by the expected fuel ΔT (≈ q'/(4πk̄), the
        # classic conduction estimate) keeps the head's gradients well
        # conditioned. With T_out_scale = 1 (legacy default) the head must
        # grow its own weights by ~10² before the heat-conduction residual
        # produces any signal, and training stalls on a flat-T plateau.
        self.T_out_scale = float(T_out_scale)

        # Smooth transition width: ~5% of fuel radius gives a sharp
        # but differentiable interface
        self.transition_width = r_fuel * 0.05

        activation_fns = {
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        act_cls = activation_fns[activation.lower()]

        # Input: 2 features (r_normalized, region_indicator)
        layers = []
        in_features = 2

        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(act_cls())
            in_features = width

        # Output: 3 fields (φ₁, φ₂, T)
        layers.append(nn.Linear(in_features, 3))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier uniform initialization for all linear layers."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _region_indicator(self, r: torch.Tensor) -> torch.Tensor:
        """Smooth fuel/moderator indicator: ~1 in fuel, ~0 in moderator.

        Uses a sigmoid centered at R_fuel with width self.transition_width.
        This is differentiable everywhere, so autograd works through it.
        """
        return torch.sigmoid((self.r_fuel - r) / self.transition_width)

    def forward(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass: r → (φ₁, φ₂, T).

        Parameters
        ----------
        r : torch.Tensor
            Radial positions [cm], shape (N, 1). Must have requires_grad=True.

        Returns
        -------
        dict with 'phi1', 'phi2', 'T', each shape (N, 1).
        """
        # Normalize r to [0, 1] over the full cell domain
        r_norm = r / self.r_cell

        # Compute smooth region indicator
        region = self._region_indicator(r)

        # Concatenate inputs: [r_normalized, region_indicator]
        x = torch.cat([r_norm, region], dim=1)  # shape (N, 2)

        # Forward through network
        out = self.network(x)  # shape (N, 3)

        # Fluxes: enforce positivity with softplus
        phi1 = torch.nn.functional.softplus(out[:, 0:1])
        phi2 = torch.nn.functional.softplus(out[:, 1:2])

        # Temperature: deviation from T_base (added by the caller), scaled
        # to the physical ΔT magnitude so the raw head output stays O(1)
        T = self.T_out_scale * out[:, 2:3]

        return {"phi1": phi1, "phi2": phi2, "T": T}

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)