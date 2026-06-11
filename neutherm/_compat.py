"""
Compatibility helpers for supporting multiple NumPy versions.

NumPy 2.0 renamed ``np.trapz`` to ``np.trapezoid`` (the old name was
removed in 2.0). Since ``pyproject.toml`` allows ``numpy>=1.24``, we
resolve the correct function at import time so the solvers work on
both NumPy 1.x and 2.x.
"""

import numpy as np

# np.trapezoid exists on NumPy >= 2.0; np.trapz on NumPy < 2.0
trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = ["trapezoid"]
