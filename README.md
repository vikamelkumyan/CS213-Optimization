# Project 7: Convergence Anatomy of Second-Order and Quasi-Newton Methods

This repository contains a reproducible numerical study comparing Gradient Descent,
Newton's Method, and SciPy BFGS on strongly convex quadratics and the Rosenbrock
function.

The project satisfies the five required tasks:

1. Implement Gradient Descent and Newton's Method from scratch.
2. Study the effect of condition number on convergence.
3. Visualize Rosenbrock optimization paths and log-gradient convergence for GD, Newton, and BFGS.
4. Compare against `scipy.optimize.minimize(method="BFGS")`.
5. Study fixed step-size sensitivity on Rosenbrock.

## Structure

```text
optimization_methods/
├── README.md
├── requirements.txt
├── reports/
│   ├── final_report.md
│   └── presentation_slides.md
├── results/
│   ├── bfgs_comparison.csv
│   ├── bfgs_comparison.png
│   ├── condition_number_study.csv
│   ├── condition_number_study.png
│   ├── rosenbrock_gradient_norms.png
│   ├── rosenbrock_paths.png
│   ├── step_size_sensitivity.csv
│   └── step_size_sensitivity.png
├── src/
│   ├── bfgs_comparison.py
│   ├── condition_number_study.py
│   ├── functions.py
│   ├── gradient_descent.py
│   ├── line_search.py
│   ├── main.py
│   ├── newton.py
│   ├── optimizers.py
│   ├── rosenbrock_visualisation.py
│   ├── run_all.py
│   └── step_size_sensitivity.py
└── tests/
    └── test_project.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce All Results

```bash
MPLCONFIGDIR=/private/tmp python src/run_all.py
```

Individual experiments can also be run directly:

```bash
python src/main.py
python src/condition_number_study.py
python src/rosenbrock_visualisation.py
python src/bfgs_comparison.py
python src/step_size_sensitivity.py
```

## Tests

```bash
python -m unittest discover -s tests
```

## Key Results

Condition-number study:

| kappa(Q) | GD iterations | Newton iterations |
|---:|---:|---:|
| 1 | 2 | 2 |
| 10 | 56 | 2 |
| 100 | 680 | 2 |
| 1000 | 6411 | 2 |

Rosenbrock comparison from `(-1.2, 1)`:

| Method | Iterations | Final \|\|grad\|\| |
|---|---:|---:|
| Gradient Descent | 13757 | 9.83e-07 |
| Newton | 22 | 4.47e-10 |
| SciPy BFGS | 33 | 2.59e-08 |

The Rosenbrock path and gradient-norm plots include BFGS alongside GD and Newton.

Fixed-step Gradient Descent on Rosenbrock:

| Step size | Status | Iterations |
|---:|---|---:|
| 0.001 | max_iter, stable but slow | 20000 |
| 0.005 | max_iter, unstable/oscillatory | 20000 |
| 0.01 | diverged | 6 |
| 0.05 | diverged | 4 |

## Main Conclusion

Gradient Descent is simple and robust with line search, but its iteration count
scales badly with conditioning. Newton's Method converges in very few iterations
because it uses curvature information directly. BFGS gets close to Newton-like
behavior without forming or solving with the exact Hessian, which makes it a
practical compromise for many smooth optimization problems.
