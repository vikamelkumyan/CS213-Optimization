"""Run every required Project 7 experiment and regenerate deliverables."""

from bfgs_comparison import main as run_bfgs_comparison
from condition_number_study import main as run_condition_number_study
from main import main as run_quadratic_verification
from rosenbrock_visualisation import main as run_rosenbrock_visualisation
from step_size_sensitivity import main as run_step_size_sensitivity


def main() -> None:
    """Execute the full reproducible experiment pipeline."""
    print("\n=== Task 1: Quadratic Verification ===")
    run_quadratic_verification()

    print("\n=== Task 2: Condition Number Study ===")
    run_condition_number_study()

    print("\n=== Task 3: Rosenbrock Visualisation ===")
    run_rosenbrock_visualisation()

    print("\n=== Task 4: SciPy BFGS Comparison ===")
    run_bfgs_comparison()

    print("\n=== Task 5: Step Size Sensitivity ===")
    run_step_size_sensitivity()


if __name__ == "__main__":
    main()
