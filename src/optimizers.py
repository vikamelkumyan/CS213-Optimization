"""Reusable optimization algorithms for Project 7 experiments."""

from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from line_search import armijo_backtracking

Array = np.ndarray
ScalarFunction = Callable[[Array], float]
VectorFunction = Callable[[Array], Array]
MatrixFunction = Callable[[Array], Array]


def make_history(x: Array) -> dict:
    """Create a consistent history container."""
    return {"x": [x.copy()], "f": [], "grad_norm": [], "alpha": []}


def finish_result(
    x: Array,
    f: ScalarFunction,
    grad_f: VectorFunction,
    history: dict,
    status: str,
) -> dict:
    """Build the common result dictionary returned by all custom solvers."""
    return {
        "x": x,
        "f": f(x),
        "grad_norm": float(np.linalg.norm(grad_f(x))),
        "iterations": len(history["grad_norm"]),
        "history": history,
        "status": status,
        "success": status == "converged",
    }


def gradient_descent_armijo(
    f: ScalarFunction,
    grad_f: VectorFunction,
    x0: Array,
    max_iter: int = 1000,
    tol: float = 1e-6,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_backtracks: int = 50,
) -> dict:
    """Run Gradient Descent with Armijo backtracking line search."""
    x = x0.astype(float).copy()
    history = make_history(x)
    status = "max_iter"

    for _ in range(max_iter):
        grad = grad_f(x)
        grad_norm = float(np.linalg.norm(grad))
        history["f"].append(f(x))
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            status = "converged"
            break

        direction = -grad
        alpha = armijo_backtracking(
            f=f,
            grad_f=grad_f,
            x=x,
            direction=direction,
            alpha0=alpha0,
            c=c,
            rho=rho,
            max_backtracks=max_backtracks,
        )
        history["alpha"].append(alpha)

        x = x + alpha * direction
        history["x"].append(x.copy())

    return finish_result(x, f, grad_f, history, status)


def newton_armijo(
    f: ScalarFunction,
    grad_f: VectorFunction,
    hess_f: MatrixFunction,
    x0: Array,
    max_iter: int = 100,
    tol: float = 1e-6,
    use_line_search: bool = True,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_backtracks: int = 60,
) -> dict:
    """Run Newton's Method using numpy.linalg.solve for the Newton system."""
    x = x0.astype(float).copy()
    history = make_history(x)
    status = "max_iter"

    for _ in range(max_iter):
        grad = grad_f(x)
        grad_norm = float(np.linalg.norm(grad))
        history["f"].append(f(x))
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            status = "converged"
            break

        try:
            direction = -np.linalg.solve(hess_f(x), grad)
        except np.linalg.LinAlgError:
            direction = -grad

        if grad.T @ direction >= 0:
            direction = -grad

        if use_line_search:
            alpha = armijo_backtracking(
                f=f,
                grad_f=grad_f,
                x=x,
                direction=direction,
                alpha0=alpha0,
                c=c,
                rho=rho,
                max_backtracks=max_backtracks,
            )
        else:
            alpha = 1.0

        history["alpha"].append(alpha)
        x = x + alpha * direction
        history["x"].append(x.copy())

    return finish_result(x, f, grad_f, history, status)


def fixed_step_gradient_descent(
    f: ScalarFunction,
    grad_f: VectorFunction,
    x0: Array,
    alpha: float,
    max_iter: int = 20000,
    tol: float = 1e-6,
    divergence_threshold: float = 1e10,
) -> dict:
    """Run Gradient Descent with a constant step size."""
    x = x0.astype(float).copy()
    history = make_history(x)
    status = "max_iter"

    for _ in range(max_iter):
        fx = f(x)
        grad = grad_f(x)
        grad_norm = float(np.linalg.norm(grad))
        history["f"].append(fx)
        history["grad_norm"].append(grad_norm)

        if not np.isfinite(fx) or not np.isfinite(grad_norm) or np.linalg.norm(x) > divergence_threshold:
            status = "diverged"
            break

        if grad_norm < tol:
            status = "converged"
            break

        history["alpha"].append(alpha)
        x = x - alpha * grad
        history["x"].append(x.copy())

    final_f = f(x) if np.all(np.isfinite(x)) else np.inf
    final_grad_norm = float(np.linalg.norm(grad_f(x))) if np.all(np.isfinite(x)) else np.inf
    return {
        "x": x,
        "f": final_f,
        "grad_norm": final_grad_norm,
        "iterations": len(history["grad_norm"]),
        "history": history,
        "status": status,
        "success": status == "converged",
    }


def scipy_bfgs(
    f: ScalarFunction,
    grad_f: VectorFunction,
    x0: Array,
    tol: float = 1e-6,
    max_iter: int = 20000,
) -> dict:
    """Run scipy.optimize.minimize with method='BFGS'."""
    scipy_result = minimize(
        fun=f,
        x0=x0,
        jac=grad_f,
        method="BFGS",
        options={"gtol": tol, "maxiter": max_iter},
    )

    x = scipy_result.x
    return {
        "x": x,
        "f": f(x),
        "grad_norm": float(np.linalg.norm(grad_f(x))),
        "iterations": int(scipy_result.nit),
        "history": {},
        "status": "converged" if scipy_result.success else "failed",
        "success": bool(scipy_result.success),
        "message": str(scipy_result.message),
    }


def timed_run(name: str, runner: Callable[[], dict]) -> dict:
    """Run an optimizer and return timing plus convergence summary."""
    start = perf_counter()
    result = runner()
    runtime = perf_counter() - start

    return {
        "method": name,
        "iterations": int(result["iterations"]),
        "runtime_seconds": float(runtime),
        "final_f": float(result["f"]),
        "final_grad_norm": float(result["grad_norm"]),
        "status": result["status"],
        "success": bool(result["success"]),
    }
