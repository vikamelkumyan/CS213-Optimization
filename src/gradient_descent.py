"""Gradient Descent implementation with Armijo line search."""

import numpy as np

from functions import quadratic_function, quadratic_gradient
from line_search import armijo_backtracking


def gradient_descent(
    x0: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict:
    """Run Gradient Descent and return final state plus iteration history."""
    x = x0.astype(float).copy()

    history = {
        "x": [x.copy()],
        "f": [],
        "grad_norm": [],
    }

    for _ in range(max_iter):
        grad = quadratic_gradient(x, Q, b)
        grad_norm = np.linalg.norm(grad)
        fx = quadratic_function(x, Q, b)

        history["f"].append(fx)
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            break

        direction = -grad

        # Use Armijo backtracking to choose a stable descent step.
        alpha = armijo_backtracking(
            f=lambda z: quadratic_function(z, Q, b),
            grad_f=lambda z: quadratic_gradient(z, Q, b),
            x=x,
            direction=direction,
        )

        x = x + alpha * direction
        history["x"].append(x.copy())

    return {
        "x": x,
        "f": quadratic_function(x, Q, b),
        "grad_norm": np.linalg.norm(quadratic_gradient(x, Q, b)),
        "iterations": len(history["grad_norm"]),
        "history": history,
    }
