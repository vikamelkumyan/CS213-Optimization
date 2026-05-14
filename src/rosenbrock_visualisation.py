"""Visualize GD, Newton, and BFGS on the Rosenbrock function."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from functions import rosenbrock_function, rosenbrock_gradient, rosenbrock_hessian
from optimizers import gradient_descent_armijo, newton_armijo, scipy_bfgs


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


def bfgs_rosenbrock(
    x0: np.ndarray,
    max_iter: int = 20000,
    tol: float = 1e-6,
) -> dict:
    """Run SciPy BFGS on Rosenbrock with iterate history."""
    return scipy_bfgs(
        f=rosenbrock_function,
        grad_f=rosenbrock_gradient,
        x0=x0,
        max_iter=max_iter,
        tol=tol,
    )


def create_level_set_plot(gd_result: dict, newton_result: dict, bfgs_result: dict, output_path: Path) -> None:
    """Plot Rosenbrock paths with overview, valley zoom, and GD detail."""
    gd_path = np.array(gd_result["history"]["x"])
    newton_path = np.array(newton_result["history"]["x"])
    bfgs_path = np.array(bfgs_result["history"]["x"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    levels = np.logspace(-1, 3.5, 30)

    def draw_contours(axis, xlim, ylim) -> None:
        x_grid = np.linspace(xlim[0], xlim[1], 450)
        y_grid = np.linspace(ylim[0], ylim[1], 450)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = (1.0 - X) ** 2 + 100.0 * (Y - X**2) ** 2
        axis.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=0.9)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, linestyle="--", alpha=0.3)

    def plot_method_paths(axis, show_gd: bool = True) -> None:
        if show_gd:
            gd_markevery = max(1, len(gd_path) // 40)
            axis.plot(
                gd_path[:, 0],
                gd_path[:, 1],
                "o-",
                markersize=3,
                markevery=gd_markevery,
                linewidth=1.1,
                label="Gradient Descent",
            )
        axis.plot(newton_path[:, 0], newton_path[:, 1], "s-", markersize=4, linewidth=1.6, label="Newton")
        axis.plot(bfgs_path[:, 0], bfgs_path[:, 1], "^-", markersize=4, linewidth=1.6, label="BFGS")

    draw_contours(axes[0], (-2.0, 2.0), (-1.0, 3.0))
    plot_method_paths(axes[0])
    axes[0].set_title("Full Path Overview")

    draw_contours(axes[1], (-0.2, 1.2), (-0.15, 1.25))
    plot_method_paths(axes[1])
    axes[1].set_title("Valley Zoom")

    early_limit = min(40, len(gd_path))
    early_gd_path = gd_path[:early_limit]
    x_padding = 0.03
    y_padding = 0.02
    draw_contours(
        axes[2],
        (early_gd_path[:, 0].min() - x_padding, early_gd_path[:, 0].max() + x_padding),
        (early_gd_path[:, 1].min() - y_padding, early_gd_path[:, 1].max() + y_padding),
    )
    scatter = axes[2].scatter(
        early_gd_path[:, 0],
        early_gd_path[:, 1],
        c=np.arange(early_limit),
        cmap="plasma",
        s=20,
        label="GD iterates",
        zorder=3,
    )
    axes[2].plot(
        early_gd_path[:, 0],
        early_gd_path[:, 1],
        color="tab:blue",
        alpha=0.55,
        linewidth=1.2,
    )
    arrow_stride = 12
    arrow_starts = early_gd_path[:-1:arrow_stride]
    arrow_deltas = early_gd_path[1::arrow_stride] - arrow_starts
    axes[2].quiver(
        arrow_starts[:, 0],
        arrow_starts[:, 1],
        arrow_deltas[:, 0],
        arrow_deltas[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
        color="black",
        alpha=0.65,
        zorder=4,
    )
    if early_limit > 32:
        axes[2].annotate(
            "large accepted\nArmijo step",
            xy=early_gd_path[31],
            xytext=(-0.55, 1.04),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=9,
            ha="center",
        )
    axes[2].set_title("GD First 40 Iterations")
    colorbar = fig.colorbar(scatter, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label("GD iteration")

    x0 = gd_path[0]
    final_points = [
        ("GD final", gd_path[-1], "tab:blue"),
        ("Newton final", newton_path[-1], "tab:orange"),
        ("BFGS final", bfgs_path[-1], "tab:green"),
    ]

    for axis in axes[:2]:
        axis.scatter(x0[0], x0[1], color="black", marker="x", s=80, label="Start x0", zorder=4)
        for label, point, color in final_points:
            axis.scatter(point[0], point[1], color=color, marker="*", s=120, label=label, zorder=5)

    axes[2].scatter(x0[0], x0[1], color="black", marker="x", s=80, label="Start x0", zorder=4)

    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[2].legend(loc="upper left", fontsize=9)

    fig.suptitle("Rosenbrock Level Sets with Optimization Paths")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_gradient_norm_plot(gd_result: dict, newton_result: dict, bfgs_result: dict, output_path: Path) -> None:
    """Plot log(||grad f(x^k)||) against iteration for all three methods."""
    eps = 1e-16

    gd_grad_norms = np.array(gd_result["history"]["grad_norm"])
    newton_grad_norms = np.array(newton_result["history"]["grad_norm"])
    bfgs_grad_norms = np.array(bfgs_result["history"]["grad_norm"])

    gd_log = np.log(np.maximum(gd_grad_norms, eps))
    newton_log = np.log(np.maximum(newton_grad_norms, eps))
    bfgs_log = np.log(np.maximum(bfgs_grad_norms, eps))

    series = [
        ("Gradient Descent", gd_log),
        ("Newton's Method", newton_log),
        ("SciPy BFGS", bfgs_log),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label, values in series:
        axes[0].plot(np.arange(len(values)), values, linewidth=2, label=label)
    axes[0].set_title("Full Run")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("log(||grad f(x^k)||)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    zoom_limit = 60
    for label, values in series:
        visible = values[: min(len(values), zoom_limit + 1)]
        axes[1].plot(np.arange(len(visible)), visible, marker="o", markersize=3, linewidth=2, label=label)
    axes[1].set_title("First 60 Iterations")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("log(||grad f(x^k)||)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle("Rosenbrock Convergence: Gradient Norm Decay")
    fig.tight_layout()

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
    bfgs_result = bfgs_rosenbrock(x0=x0, max_iter=20000, tol=tol)

    print_summary("Gradient Descent", gd_result)
    print_summary("Newton's Method", newton_result)
    print_summary("SciPy BFGS", bfgs_result)

    root = Path(__file__).resolve().parent.parent
    paths_plot_path = root / "results" / "rosenbrock_paths.png"
    grad_norm_plot_path = root / "results" / "rosenbrock_gradient_norms.png"

    create_level_set_plot(gd_result, newton_result, bfgs_result, paths_plot_path)
    create_gradient_norm_plot(gd_result, newton_result, bfgs_result, grad_norm_plot_path)

    print(f"\nSaved plot to: {paths_plot_path}")
    print(f"Saved plot to: {grad_norm_plot_path}")


if __name__ == "__main__":
    main()
