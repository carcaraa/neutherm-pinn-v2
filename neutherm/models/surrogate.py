"""
Data-driven surrogate model for the coupled neutronics-thermal problem.

This module defines a feed-forward neural network (FNN) that learns the
mapping from physical parameters to solution fields:

    (T_coolant, r_fuel, enrichment) → (φ₁(r), φ₂(r), T(r), k_eff)

The network takes 3 scalar inputs and produces:
- φ₁(r): fast-group flux at each radial point (n_total values)
- φ₂(r): thermal-group flux at each radial point (n_total values)
- T(r):  fuel temperature at each radial point (n_radial values)
- k_eff: effective multiplication factor (1 scalar)

Architecture choices:
- Skip (residual) connections every 2 layers for better gradient flow
- Separate output heads for fields vs scalars
- Tanh activation (smooth, works well for physics problems)
- Input normalization built into the model

References
----------
.. [5] Lu et al., "Learning nonlinear operators via DeepONet" (2021).
.. [9] Sun et al., "Surrogate modeling for fluid flows" (2020). CMAME.
"""

import torch
import torch.nn as nn


class InputNormalizer(nn.Module):
    """Normalizes input parameters to zero mean, unit variance.

    Neural networks train much better when inputs are normalized.
    Without normalization, parameters with large magnitudes (T_coolant ~ 580)
    would dominate over small ones (r_fuel ~ 0.004).

    Parameters
    ----------
    input_mean : torch.Tensor
        Mean of each input parameter, shape (n_inputs,).
    input_std : torch.Tensor
        Standard deviation of each input parameter, shape (n_inputs,).
    """

    def __init__(self, input_mean: torch.Tensor, input_std: torch.Tensor):
        super().__init__()
        # register_buffer: saved with model, moved with .to(device),
        # but NOT treated as a trainable parameter.
        self.register_buffer("mean", input_mean)
        self.register_buffer("std", input_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize inputs: x_norm = (x - mean) / std."""
        return (x - self.mean) / (self.std + 1e-8)


class ResidualBlock(nn.Module):
    """A residual (skip connection) block with two linear layers.

    Instead of learning F(x), the block learns F(x) + x, so the
    gradient always has a direct path through the skip connection.

    Architecture:
        x → Linear → Activation → Linear → + x → Activation → out

    Parameters
    ----------
    width : int
        Number of neurons (input and output must match for skip).
    activation : nn.Module
        Activation function.
    """

    def __init__(self, width: int, activation: nn.Module):
        super().__init__()
        self.layer1 = nn.Linear(width, width)
        self.layer2 = nn.Linear(width, width)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        residual = x
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        out = self.activation(out + residual)
        return out


class SurrogateModel(nn.Module):
    """Feed-forward surrogate model for the coupled problem.

    Three sections:
    1. Input layer: projects 3 parameters → hidden_width
    2. Hidden blocks: sequence of ResidualBlocks
    3. Output heads: separate linear projections for each output type

    Why separate output heads?
        k_eff is 1 scalar, φ₁ has 140 values. They live on different
        scales and have different structures. Separate heads let each
        output specialize while sharing the hidden feature extraction.

    Parameters
    ----------
    n_inputs : int
        Number of input parameters (default: 3).
    n_radial_fuel : int
        Number of radial mesh points in the fuel region.
    n_radial_total : int
        Number of radial mesh points in the full pin cell.
    hidden_layers : list[int]
        Width of each hidden layer. All must be the same for skip connections.
    activation : str
        Activation function: "tanh", "relu", "gelu", or "silu".
    """

    def __init__(
        self,
        n_inputs: int = 3,
        n_radial_fuel: int = 100,
        n_radial_total: int = 140,
        hidden_layers: list[int] | None = None,
        activation: str = "tanh",
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 128]

        self.n_inputs = n_inputs
        self.n_radial_fuel = n_radial_fuel
        self.n_radial_total = n_radial_total
        # Stored so checkpoints can rebuild the exact architecture without
        # relying on the YAML config still matching the trained model.
        self.hidden_layers = list(hidden_layers)
        self.activation_name = activation

        activation_fn = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }[activation.lower()]

        hidden_width = hidden_layers[0]
        n_blocks = len(hidden_layers) // 2

        # Input projection: 3 params → hidden_width
        self.input_proj = nn.Sequential(
            nn.Linear(n_inputs, hidden_width),
            activation_fn,
        )

        # Hidden blocks with skip connections
        self.hidden_blocks = nn.Sequential(
            *[ResidualBlock(hidden_width, activation_fn) for _ in range(n_blocks)]
        )

        # Output heads — separate projections for each field
        self.head_phi1 = nn.Linear(hidden_width, n_radial_total)
        self.head_phi2 = nn.Linear(hidden_width, n_radial_total)
        self.head_temperature = nn.Linear(hidden_width, n_radial_fuel)
        self.head_keff = nn.Linear(hidden_width, 1)

        # Input normalization (set after seeing data)
        self.normalizer = None

    def set_normalizer(self, input_mean: torch.Tensor, input_std: torch.Tensor):
        """Set input normalization statistics from training data."""
        self.normalizer = InputNormalizer(input_mean, input_std)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass: parameters → predicted fields.

        Parameters
        ----------
        x : torch.Tensor
            Input parameters, shape (batch_size, 3).

        Returns
        -------
        dict[str, torch.Tensor]
            'phi1': (batch, n_total), 'phi2': (batch, n_total),
            'temperature': (batch, n_fuel), 'k_eff': (batch, 1).
        """
        if self.normalizer is not None:
            x = self.normalizer(x)

        h = self.input_proj(x)
        h = self.hidden_blocks(h)

        return {
            "phi1": self.head_phi1(h),
            "phi2": self.head_phi2(h),
            "temperature": self.head_temperature(h),
            "k_eff": self.head_keff(h),
        }

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)