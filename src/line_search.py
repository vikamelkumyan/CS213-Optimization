"""Armijo backtracking line search."""

import numpy as np


def armijo_backtracking(
    f,
    grad_f,
    x: np.ndarray,
    direction: np.ndarray,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_backtracks: int = 50,
) -> float:
    """
    Find a step size alpha that satisfies the Armijo condition.

    Armijo condition:
        f(x + alpha * d) <= f(x) + c * alpha * grad_f(x)^T d
    """
    alpha = alpha0
    fx = f(x)
    grad_fx = grad_f(x)
    directional_derivative = grad_fx.T @ direction

    for _ in range(max_backtracks):
        candidate = x + alpha * direction
        if f(candidate) <= fx + c * alpha * directional_derivative:
            return alpha
        alpha *= rho

    return alpha
