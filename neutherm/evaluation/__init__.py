"""
Evaluation and benchmarking: Solver vs Surrogate vs PINN.

Submodules
----------
metrics : Error metrics (L2, Linf, pointwise relative).
compare : Side-by-side comparison with plots and tables.
"""

from neutherm.evaluation.metrics import (
    mean_absolute_error,
    pointwise_relative_error,
    relative_l2,
    relative_linf,
)

__all__ = [
    "relative_l2",
    "relative_linf",
    "pointwise_relative_error",
    "mean_absolute_error",
]
