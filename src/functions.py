"""Core objective function definitions for the quadratic optimization task."""

import numpy as np


def quadratic_function(x: np.ndarray, Q: np.ndarray, b: np.ndarray) -> float:
    """Compute f(x) = x^T Q x - b^T x."""
    return float(x.T @ Q @ x - b.T @ x)


def quadratic_gradient(x: np.ndarray, Q: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute grad f(x) = 2Qx - b."""
    return 2.0 * (Q @ x) - b


def quadratic_hessian(Q: np.ndarray) -> np.ndarray:
    """Compute Hessian f(x) = 2Q (constant for this quadratic)."""
    return 2.0 * Q
