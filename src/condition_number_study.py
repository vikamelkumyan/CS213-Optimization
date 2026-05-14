"""Condition number study for Gradient Descent and Newton's Method."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gradient_descent import gradient_descent
from newton import newton_method


def generate_spd_with_condition_number(kappa: float, dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Create a symmetric positive definite matrix with target condition number kappa."""
    random_matrix = rng.standard_normal((dimension, dimension))
    orthogonal_matrix, _ = np.linalg.qr(random_matrix)

    # Eigenvalues range from 1 to kappa, so cond(Q) = kappa.
    eigenvalues = np.linspace(1.0, float(kappa), dimension)
    Q = orthogonal_matrix @ np.diag(eigenvalues) @ orthogonal_matrix.T

    # Symmetrize to remove tiny numerical asymmetry from matrix multiplications.
    return 0.5 * (Q + Q.T)


def print_results_table(results: list[dict]) -> None:
    """Print a clear table of iteration counts and final gradient norms."""
    header = (
        f"{'kappa(Q)':>10} | {'GD iters':>8} | {'Newton iters':>12} | "
        f"{'GD final ||grad||':>17} | {'Newton final ||grad||':>21}"
    )
    separator = "-" * len(header)

    print("\nCondition Number Study Results")
    print(separator)
    print(header)
    print(separator)

    for row in results:
        print(
            f"{row['kappa']:10.0f} | "
            f"{row['gd_iters']:8d} | "
            f"{row['newton_iters']:12d} | "
            f"{row['gd_grad_norm']:17.6e} | "
            f"{row['newton_grad_norm']:21.6e}"
        )

    print(separator)


def create_plot(results: list[dict], output_path: Path) -> None:
    """Create and save the iteration-vs-condition-number plot."""
    kappas = [row["kappa"] for row in results]
    gd_iters = [row["gd_iters"] for row in results]
    newton_iters = [row["newton_iters"] for row in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis in axes:
        axis.plot(kappas, gd_iters, marker="o", linewidth=2, label="Gradient Descent")
        axis.plot(kappas, newton_iters, marker="s", linewidth=2, label="Newton's Method")
        axis.set_xscale("log")
        axis.set_xlabel("Condition Number kappa(Q)")
        axis.set_ylabel("Iterations to reach tolerance")
        axis.grid(True, which="both", linestyle="--", alpha=0.4)
        axis.legend()

        for kappa, gd_iter, newton_iter in zip(kappas, gd_iters, newton_iters):
            axis.annotate(f"{gd_iter}", (kappa, gd_iter), textcoords="offset points", xytext=(0, 8), ha="center")
            axis.annotate(
                f"{newton_iter}",
                (kappa, newton_iter),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
            )

    axes[0].set_title("Linear Iteration Scale")
    axes[1].set_title("Log Iteration Scale")
    axes[1].set_yscale("log")

    fig.suptitle("Effect of Condition Number on Convergence")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_results_csv(results: list[dict], output_path: Path) -> None:
    """Save study results to CSV for easy reporting and reuse."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "kappa",
        "gd_iters",
        "newton_iters",
        "gd_grad_norm",
        "newton_grad_norm",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    kappas = [1, 10, 100, 1000]
    dimension = 5
    tol = 1e-6

    # Keep b and x0 fixed across all condition number experiments.
    b = np.array([1.0, -2.0, 3.0, -1.0, 2.0])
    x0 = np.zeros(dimension)

    rng = np.random.default_rng(42)
    results = []

    for kappa in kappas:
        Q = generate_spd_with_condition_number(kappa=kappa, dimension=dimension, rng=rng)

        gd_result = gradient_descent(x0=x0, Q=Q, b=b, max_iter=10000, tol=tol)
        newton_result = newton_method(x0=x0, Q=Q, b=b, max_iter=100, tol=tol)

        results.append(
            {
                "kappa": float(kappa),
                "gd_iters": int(gd_result["iterations"]),
                "newton_iters": int(newton_result["iterations"]),
                "gd_grad_norm": float(gd_result["grad_norm"]),
                "newton_grad_norm": float(newton_result["grad_norm"]),
            }
        )

    print_results_table(results)

    output_path = Path(__file__).resolve().parent.parent / "results" / "condition_number_study.png"
    create_plot(results, output_path)
    print(f"\nSaved plot to: {output_path}")

    csv_output_path = Path(__file__).resolve().parent.parent / "results" / "condition_number_study.csv"
    save_results_csv(results, csv_output_path)
    print(f"Saved table to: {csv_output_path}")

    print("\nExpected conclusion:")
    print("Gradient Descent needs more iterations as kappa(Q) increases,")
    print("while Newton's Method stays almost insensitive to condition number.")


if __name__ == "__main__":
    main()
