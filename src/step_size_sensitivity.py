"""Study fixed-step Gradient Descent sensitivity on the Rosenbrock function."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from functions import rosenbrock_function, rosenbrock_gradient
from optimizers import fixed_step_gradient_descent


def print_results_table(results: list[dict]) -> None:
    """Print fixed-step sensitivity results."""
    header = (
        f"{'alpha':>8} | {'status':>10} | {'iters':>8} | "
        f"{'final f':>12} | {'final ||grad||':>15}"
    )
    separator = "-" * len(header)

    print("\nFixed Step Size Sensitivity on Rosenbrock")
    print(separator)
    print(header)
    print(separator)

    for row in results:
        print(
            f"{row['alpha']:8.3f} | "
            f"{row['status']:>10} | "
            f"{row['iterations']:8d} | "
            f"{row['f']:12.6e} | "
            f"{row['grad_norm']:15.6e}"
        )

    print(separator)


def save_results_csv(results: list[dict], output_path: Path) -> None:
    """Save fixed-step sensitivity summary to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["alpha", "status", "iterations", "f", "grad_norm"]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fieldnames})


def create_gradient_norm_plot(results: list[dict], output_path: Path) -> None:
    """Plot fixed-step behavior with stable and divergent runs separated."""
    eps = 1e-16
    stable_results = [result for result in results if result["status"] != "diverged"]
    divergent_results = [result for result in results if result["status"] == "diverged"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for result in stable_results:
        grad_norms = np.array(result["history"]["grad_norm"])
        log_grad_norms = np.log(np.maximum(grad_norms, eps))
        axes[0].plot(
            np.arange(len(log_grad_norms)),
            log_grad_norms,
            linewidth=2,
            label=f"alpha={result['alpha']:g} ({result['status']})",
        )

    axes[0].set_title("Stable but Slow Fixed Steps")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("log(||grad f(x^k)||)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    for result in divergent_results:
        grad_norms = np.array(result["history"]["grad_norm"])
        log_grad_norms = np.log(np.maximum(grad_norms, eps))
        axes[1].plot(
            np.arange(len(log_grad_norms)),
            log_grad_norms,
            marker="o",
            linewidth=2,
            label=f"alpha={result['alpha']:g} ({result['status']})",
        )

    axes[1].set_title("Large Steps Diverge Quickly")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("log(||grad f(x^k)||)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle("Rosenbrock Fixed Step Size Sensitivity")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    x0 = np.array([-1.2, 1.0])
    alphas = [0.001, 0.005, 0.01, 0.05]

    results = [
        {
            **fixed_step_gradient_descent(
                f=rosenbrock_function,
                grad_f=rosenbrock_gradient,
                x0=x0,
                alpha=alpha,
                max_iter=20000,
                tol=1e-6,
            ),
            "alpha": alpha,
        }
        for alpha in alphas
    ]

    print_results_table(results)

    root = Path(__file__).resolve().parent.parent
    csv_path = root / "results" / "step_size_sensitivity.csv"
    plot_path = root / "results" / "step_size_sensitivity.png"

    save_results_csv(results, csv_path)
    create_gradient_norm_plot(results, plot_path)

    print("\nConclusion:")
    print("alpha=0.001 is stable but slow.")
    print("alpha=0.005 does not reach the tolerance within the iteration budget.")
    print("alpha=0.01 and alpha=0.05 diverge.")
    print(f"\nSaved table to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
