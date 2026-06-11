"""
Neural network architectures for the surrogate and PINN models.

Submodules
----------
surrogate : Data-driven feed-forward surrogate model.
pinn : Physics-informed neural network (continuous function approximator).
"""

from neutherm.models.pinn import PINNModel
from neutherm.models.surrogate import InputNormalizer, ResidualBlock, SurrogateModel

__all__ = [
    "SurrogateModel",
    "PINNModel",
    "ResidualBlock",
    "InputNormalizer",
]
