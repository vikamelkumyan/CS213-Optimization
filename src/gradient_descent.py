"""Quadratic Gradient Descent wrapper used in the verification task."""

import numpy as np

from functions import quadratic_function, quadratic_gradient
from optimizers import gradient_descent_armijo


def gradient_descent(
    x0: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict:
    """Run Gradient Descent with Armijo line search on f(x)=x^TQx-b^Tx."""
    return gradient_descent_armijo(
        f=lambda z: quadratic_function(z, Q, b),
        grad_f=lambda z: quadratic_gradient(z, Q, b),
        x0=x0,
        max_iter=max_iter,
        tol=tol,
    )
