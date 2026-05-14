"""Compare custom GD/Newton implementations against scipy BFGS on Rosenbrock."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from functions import rosenbrock_function, rosenbrock_gradient
from optimizers import scipy_bfgs, timed_run
from rosenbrock_visualisation import gradient_descent_rosenbrock, newton_rosenbrock


def print_results_table(results: list[dict]) -> None:
    """Print a compact comparison table."""
    header = (
        f"{'Method':>18} | {'Iters':>8} | {'Runtime (s)':>12} | "
        f"{'Final f':>12} | {'Final ||grad||':>15} | {'Success':>7}"
    )
    separator = "-" * len(header)

    print("\nBFGS Comparison on Rosenbrock")
    print(separator)
    print(header)
    print(separator)

    for row in results:
        print(
            f"{row['method']:>18} | "
            f"{row['iterations']:8d} | "
            f"{row['runtime_seconds']:12.6f} | "
            f"{row['final_f']:12.6e} | "
            f"{row['final_grad_norm']:15.6e} | "
            f"{str(row['success']):>7}"
        )

    print(separator)


def save_results_csv(results: list[dict], output_path: Path) -> None:
    """Save BFGS comparison results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "iterations",
        "runtime_seconds",
        "final_f",
        "final_grad_norm",
        "status",
        "success",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def create_runtime_plot(results: list[dict], output_path: Path) -> None:
    """Create readable comparison plots for iterations, runtime, and accuracy."""
    methods = [row["method"] for row in results]
    iterations = [row["iterations"] for row in results]
    runtimes = [row["runtime_seconds"] for row in results]
    grad_norms = [row["final_grad_norm"] for row in results]

    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def label_bars(axis, bars, values, formatter) -> None:
        for bar, value in zip(bars, values):
            axis.annotate(
                formatter(value),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    bars = axes[0, 0].bar(methods, iterations, color=colors)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Iterations, Log Scale")
    axes[0, 0].set_ylabel("Iterations")
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].grid(True, axis="y", which="both", linestyle="--", alpha=0.35)
    label_bars(axes[0, 0], bars, iterations, lambda value: f"{int(value)}")

    zoom_methods = methods[1:]
    zoom_iterations = iterations[1:]
    zoom_colors = colors[1:]
    bars = axes[0, 1].bar(zoom_methods, zoom_iterations, color=zoom_colors)
    axes[0, 1].set_title("Newton vs BFGS Iterations")
    axes[0, 1].set_ylabel("Iterations")
    axes[0, 1].set_ylim(0, max(zoom_iterations) * 1.3)
    axes[0, 1].grid(True, axis="y", linestyle="--", alpha=0.35)
    label_bars(axes[0, 1], bars, zoom_iterations, lambda value: f"{int(value)}")

    bars = axes[1, 0].bar(methods, runtimes, color=colors)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Runtime, Log Scale")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].grid(True, axis="y", which="both", linestyle="--", alpha=0.35)
    label_bars(axes[1, 0], bars, runtimes, lambda value: f"{value:.2e}")

    bars = axes[1, 1].bar(methods, grad_norms, color=colors)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Final Gradient Norm")
    axes[1, 1].set_ylabel("||grad f(x)||")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].grid(True, axis="y", which="both", linestyle="--", alpha=0.35)
    label_bars(axes[1, 1], bars, grad_norms, lambda value: f"{value:.1e}")

    fig.suptitle("Rosenbrock Optimizer Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    x0 = np.array([-1.2, 1.0])
    tol = 1e-6

    results = [
        timed_run(
            "Gradient Descent",
            lambda: gradient_descent_rosenbrock(x0=x0, max_iter=20000, tol=tol),
        ),
        timed_run(
            "Newton",
            lambda: newton_rosenbrock(x0=x0, max_iter=200, tol=tol),
        ),
        timed_run("SciPy BFGS", lambda: scipy_bfgs(rosenbrock_function, rosenbrock_gradient, x0, tol)),
    ]

    print_results_table(results)

    root = Path(__file__).resolve().parent.parent
    csv_path = root / "results" / "bfgs_comparison.csv"
    plot_path = root / "results" / "bfgs_comparison.png"

    save_results_csv(results, csv_path)
    create_runtime_plot(results, plot_path)

    print(f"\nSaved table to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
