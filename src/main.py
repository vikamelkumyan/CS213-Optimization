"""Run and compare Gradient Descent and Newton's Method on a quadratic function."""

import numpy as np

from gradient_descent import gradient_descent
from newton import newton_method


def print_summary(name: str, result: dict) -> None:
    """Print a compact summary of optimization results."""
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Final solution x: {result['x']}")
    print(f"Final objective value: {result['f']:.10f}")
    print(f"Number of iterations: {result['iterations']}")
    print(f"Final gradient norm: {result['grad_norm']:.10e}")


def main() -> None:
    # Example positive definite matrix Q and vector b.
    Q = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0])

    # Initial point.
    x0 = np.array([0.0, 0.0])

    gd_result = gradient_descent(x0, Q, b, max_iter=1000, tol=1e-6)
    newton_result = newton_method(x0, Q, b, max_iter=100, tol=1e-6)

    print_summary("Gradient Descent", gd_result)
    print_summary("Newton's Method", newton_result)

    # Analytical minimizer from setting grad f(x) = 0 -> 2Qx - b = 0.
    x_star = 0.5 * np.linalg.solve(Q, b)
    print("\nAnalytical Solution")
    print("-------------------")
    print(f"x*: {x_star}")

    print("\nDistances to analytical solution")
    print("-------------------------------")
    print(f"Gradient Descent ||x - x*||: {np.linalg.norm(gd_result['x'] - x_star):.10e}")
    print(f"Newton's Method ||x - x*||: {np.linalg.norm(newton_result['x'] - x_star):.10e}")


if __name__ == "__main__":
    main()
