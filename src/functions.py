"""Objective functions and derivatives used in optimization experiments."""

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


def rosenbrock_function(x: np.ndarray) -> float:
    """Compute Rosenbrock function f(x, y) = (1 - x)^2 + 100(y - x^2)^2."""
    x1, x2 = x[0], x[1]
    return float((1.0 - x1) ** 2 + 100.0 * (x2 - x1**2) ** 2)


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Compute gradient of the 2D Rosenbrock function."""
    x1, x2 = x[0], x[1]
    grad_x1 = -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1**2)
    grad_x2 = 200.0 * (x2 - x1**2)
    return np.array([grad_x1, grad_x2], dtype=float)


def rosenbrock_hessian(x: np.ndarray) -> np.ndarray:
    """Compute Hessian of the 2D Rosenbrock function."""
    x1, x2 = x[0], x[1]
    h11 = 2.0 - 400.0 * (x2 - x1**2) + 800.0 * x1**2
    h12 = -400.0 * x1
    h22 = 200.0
    return np.array([[h11, h12], [h12, h22]], dtype=float)
