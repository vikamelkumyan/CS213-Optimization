"""Regression tests for the Project 7 optimization experiments."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from condition_number_study import generate_spd_with_condition_number
from functions import (
    quadratic_gradient,
    rosenbrock_function,
    rosenbrock_gradient,
    rosenbrock_hessian,
)
from gradient_descent import gradient_descent
from newton import newton_method
from optimizers import fixed_step_gradient_descent, scipy_bfgs
from rosenbrock_visualisation import newton_rosenbrock


class ProjectExperimentsTest(unittest.TestCase):
    def test_quadratic_methods_reach_analytic_solution(self) -> None:
        Q = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([1.0, 2.0])
        x0 = np.zeros(2)
        x_star = 0.5 * np.linalg.solve(Q, b)

        gd_result = gradient_descent(x0, Q, b, max_iter=1000, tol=1e-6)
        newton_result = newton_method(x0, Q, b, max_iter=100, tol=1e-6)

        self.assertLess(np.linalg.norm(gd_result["x"] - x_star), 1e-5)
        self.assertLess(np.linalg.norm(newton_result["x"] - x_star), 1e-12)
        self.assertLess(np.linalg.norm(quadratic_gradient(x_star, Q, b)), 1e-12)

    def test_condition_number_generator_matches_target(self) -> None:
        rng = np.random.default_rng(7)
        for kappa in [1, 10, 100, 1000]:
            Q = generate_spd_with_condition_number(kappa, dimension=5, rng=rng)
            self.assertTrue(np.allclose(Q, Q.T))
            self.assertAlmostEqual(np.linalg.cond(Q), kappa, delta=1e-8 * kappa)

    def test_newton_converges_on_rosenbrock(self) -> None:
        result = newton_rosenbrock(np.array([-1.2, 1.0]), max_iter=100, tol=1e-6)

        self.assertTrue(result["success"])
        self.assertLess(result["grad_norm"], 1e-6)
        self.assertLess(np.linalg.norm(result["x"] - np.array([1.0, 1.0])), 1e-5)
        self.assertGreater(np.linalg.cond(rosenbrock_hessian(np.array([1.0, 1.0]))), 1000)

    def test_scipy_bfgs_converges_on_rosenbrock(self) -> None:
        result = scipy_bfgs(
            rosenbrock_function,
            rosenbrock_gradient,
            np.array([-1.2, 1.0]),
            tol=1e-6,
        )

        self.assertTrue(result["success"])
        self.assertLess(result["grad_norm"], 1e-6)
        self.assertLess(result["iterations"], 100)

    def test_fixed_large_step_diverges_on_rosenbrock(self) -> None:
        result = fixed_step_gradient_descent(
            rosenbrock_function,
            rosenbrock_gradient,
            np.array([-1.2, 1.0]),
            alpha=0.01,
            max_iter=50,
            tol=1e-6,
        )

        self.assertEqual(result["status"], "diverged")


if __name__ == "__main__":
    unittest.main()
