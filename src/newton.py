"""Quadratic Newton wrapper used in the verification task."""

import numpy as np

from functions import quadratic_function, quadratic_gradient, quadratic_hessian
from optimizers import newton_armijo


def newton_method(
    x0: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Run Newton's Method on f(x)=x^TQx-b^Tx using numpy.linalg.solve."""
    return newton_armijo(
        f=lambda z: quadratic_function(z, Q, b),
        grad_f=lambda z: quadratic_gradient(z, Q, b),
        hess_f=lambda _: quadratic_hessian(Q),
        x0=x0,
        max_iter=max_iter,
        tol=tol,
        use_line_search=False,
    )
