"""Newton's Method implementation for the quadratic objective."""

import numpy as np

from functions import quadratic_function, quadratic_gradient, quadratic_hessian


def newton_method(
    x0: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Run Newton's Method and return final state plus iteration history."""
    x = x0.astype(float).copy()
    H = quadratic_hessian(Q)

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

        # Solve H p = grad, then use x_{k+1} = x_k - p.
        step = np.linalg.solve(H, grad)
        x = x - step
        history["x"].append(x.copy())

    return {
        "x": x,
        "f": quadratic_function(x, Q, b),
        "grad_norm": np.linalg.norm(quadratic_gradient(x, Q, b)),
        "iterations": len(history["grad_norm"]),
        "history": history,
    }
