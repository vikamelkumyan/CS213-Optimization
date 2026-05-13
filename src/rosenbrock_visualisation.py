"""Visualize Gradient Descent and Newton's Method on the Rosenbrock function."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from functions import rosenbrock_function, rosenbrock_gradient, rosenbrock_hessian
from optimizers import gradient_descent_armijo, newton_armijo


def gradient_descent_rosenbrock(
    x0: np.ndarray,
    max_iter: int = 20000,
    tol: float = 1e-6,
) -> dict:
    """Run Gradient Descent with Armijo backtracking on Rosenbrock."""
    return gradient_descent_armijo(
        f=rosenbrock_function,
        grad_f=rosenbrock_gradient,
        x0=x0,
        max_iter=max_iter,
        tol=tol,
        max_backtracks=60,
    )


def newton_rosenbrock(
    x0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    """Run Newton's Method with Armijo globalization on Rosenbrock."""
    return newton_armijo(
        f=rosenbrock_function,
        grad_f=rosenbrock_gradient,
        hess_f=rosenbrock_hessian,
        x0=x0,
        max_iter=max_iter,
        tol=tol,
        use_line_search=True,
        max_backtracks=60,
    )


def create_level_set_plot(gd_result: dict, newton_result: dict, output_path: Path) -> None:
    """Plot Rosenbrock level sets and both optimization paths."""
    x_grid = np.linspace(-2.0, 2.0, 500)
    y_grid = np.linspace(-1.0, 3.0, 500)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = (1.0 - X) ** 2 + 100.0 * (Y - X**2) ** 2

    gd_path = np.array(gd_result["history"]["x"])
    newton_path = np.array(newton_result["history"]["x"])

    plt.figure(figsize=(9, 7))
    levels = np.logspace(-1, 3.5, 30)
    plt.contour(X, Y, Z, levels=levels, cmap="viridis")

    plt.plot(gd_path[:, 0], gd_path[:, 1], "o-", markersize=3, linewidth=1.5, label="Gradient Descent")
    plt.plot(newton_path[:, 0], newton_path[:, 1], "s-", markersize=3, linewidth=1.5, label="Newton's Method")

    x0 = gd_path[0]
    gd_final = gd_path[-1]
    newton_final = newton_path[-1]

    plt.scatter(x0[0], x0[1], color="black", marker="x", s=80, label="Start x0")
    plt.scatter(gd_final[0], gd_final[1], color="tab:blue", marker="*", s=120, label="GD final")
    plt.scatter(newton_final[0], newton_final[1], color="tab:orange", marker="*", s=120, label="Newton final")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rosenbrock Level Sets with Optimization Paths")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_gradient_norm_plot(gd_result: dict, newton_result: dict, output_path: Path) -> None:
    """Plot log(||grad f(x^k)||) against iteration for both methods."""
    eps = 1e-16

    gd_grad_norms = np.array(gd_result["history"]["grad_norm"])
    newton_grad_norms = np.array(newton_result["history"]["grad_norm"])

    gd_log = np.log(np.maximum(gd_grad_norms, eps))
    newton_log = np.log(np.maximum(newton_grad_norms, eps))

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(gd_log)), gd_log, linewidth=2, label="Gradient Descent")
    plt.plot(np.arange(len(newton_log)), newton_log, linewidth=2, label="Newton's Method")
    plt.xlabel("Iteration")
    plt.ylabel("log(||grad f(x^k)||)")
    plt.title("Rosenbrock Convergence: Gradient Norm Decay")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_summary(method_name: str, result: dict) -> None:
    """Print final point, objective value, iterations, and gradient norm."""
    print(f"\n{method_name}")
    print("-" * len(method_name))
    print(f"Final point: {result['x']}")
    print(f"Final objective value: {result['f']:.10f}")
    print(f"Number of iterations: {result['iterations']}")
    print(f"Final gradient norm: {result['grad_norm']:.10e}")


def main() -> None:
    x0 = np.array([-1.2, 1.0])
    tol = 1e-6

    gd_result = gradient_descent_rosenbrock(x0=x0, max_iter=20000, tol=tol)
    newton_result = newton_rosenbrock(x0=x0, max_iter=200, tol=tol)

    print_summary("Gradient Descent", gd_result)
    print_summary("Newton's Method", newton_result)

    root = Path(__file__).resolve().parent.parent
    paths_plot_path = root / "results" / "rosenbrock_paths.png"
    grad_norm_plot_path = root / "results" / "rosenbrock_gradient_norms.png"

    create_level_set_plot(gd_result, newton_result, paths_plot_path)
    create_gradient_norm_plot(gd_result, newton_result, grad_norm_plot_path)

    print(f"\nSaved plot to: {paths_plot_path}")
    print(f"Saved plot to: {grad_norm_plot_path}")


if __name__ == "__main__":
    main()
