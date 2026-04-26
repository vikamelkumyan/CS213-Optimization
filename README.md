# Optimization Methods: Task 1

This project implements **Gradient Descent** and **Newton's Method** from scratch for the quadratic objective:

\[
f(x) = x^T Q x - b^T x
\]

## Project Structure

```text
optimization_methods/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── functions.py
│   ├── line_search.py
│   ├── gradient_descent.py
│   ├── newton.py
│   └── main.py
└── results/
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python src/main.py
```

## What This Implements

- Quadratic objective, gradient, and Hessian
- Armijo backtracking line search
- Gradient Descent with stopping condition:
  - `||grad f(x)|| < 1e-6` or `max_iter` reached
- Newton's Method with stopping condition:
  - `||grad f(x)|| < 1e-6` or `max_iter` reached
- Tracking of:
  - iterate history (`x` values)
  - objective value history
  - gradient norm history
- Comparison against analytical solution:
  - `x* = 0.5 * np.linalg.solve(Q, b)`
