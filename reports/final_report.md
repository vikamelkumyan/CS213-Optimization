# Project 7 Final Report

## Convergence Anatomy of Second-Order and Quasi-Newton Methods

### Objective

The goal of this project is to compare Gradient Descent, Newton's Method, and
BFGS on smooth optimization problems. The experiments focus on how curvature and
conditioning affect convergence speed, stability, and practical runtime.
The quadratic experiments are globally strongly convex. Rosenbrock is not
globally strongly convex, but it has a strongly convex local basin around the
minimizer, which makes it useful for studying practical convergence phases.

### Methods

Gradient Descent was implemented from scratch with Armijo backtracking line
search. At iteration `k`, the method uses

```text
x_{k+1} = x_k - alpha_k grad f(x_k),
```

where `alpha_k` is selected by the Armijo sufficient decrease condition.

Newton's Method was also implemented from scratch. The Newton direction is
computed with `numpy.linalg.solve`:

```text
H_k p_k = grad f(x_k),    x_{k+1} = x_k - p_k.
```

For the Rosenbrock function, Armijo globalization is used with Newton's Method
to improve robustness away from the minimizer. SciPy BFGS was run with
`scipy.optimize.minimize(method="BFGS")` and the analytic gradient.

The stopping tolerance for all main convergence experiments was
`||grad f(x_k)|| < 1e-6`.

### Task 1: Quadratic Verification

The methods were verified on

```text
f(x) = x^T Q x - b^T x
```

with

```text
Q = [[3, 1],
     [1, 2]],    b = [1, 2].
```

The analytic solution satisfies `2Qx - b = 0`, so
`x* = 0.5 * solve(Q, b) = [0, 0.5]`.

| Method | Iterations | Final point | Final gradient norm |
|---|---:|---|---:|
| Gradient Descent | 70 | `[9.97e-08, 5.00000062e-01]` | 8.48e-07 |
| Newton | 2 | `[0, 0.5]` | 0.00e+00 |

Both methods reached the analytic minimizer. Newton reached it essentially in
one update after the initial optimality check because the objective is exactly
quadratic and the Hessian is constant.

### Task 2: Condition Number Study

Symmetric positive definite matrices were generated with prescribed condition
numbers `kappa(Q) in {1, 10, 100, 1000}`. The same starting point and vector `b`
were used for each run.

| kappa(Q) | GD iterations | Newton iterations | GD final \|\|grad\|\| | Newton final \|\|grad\|\| |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 1.17e-15 | 6.66e-16 |
| 10 | 56 | 2 | 7.66e-07 | 4.44e-16 |
| 100 | 680 | 2 | 7.63e-07 | 7.11e-15 |
| 1000 | 6411 | 2 | 9.98e-07 | 7.82e-14 |

![Condition Number Study](../results/condition_number_study.png)

The results confirm the theoretical trend: Gradient Descent degrades strongly as
the condition number increases, while Newton's Method remains almost insensitive
on these quadratics.

### Task 3: Rosenbrock Visualization

The Rosenbrock function was minimized from `x0 = (-1.2, 1)`:

```text
f(x, y) = (1 - x)^2 + 100(y - x^2)^2.
```

The Hessian condition number at the minimizer `(1, 1)` is approximately
`2508`, so the problem has a narrow, curved valley.

![Rosenbrock Paths](../results/rosenbrock_paths.png)

Gradient Descent follows the valley slowly. Newton's Method uses curvature
information and moves to the minimizer in far fewer iterations.

![Rosenbrock Gradient Norms](../results/rosenbrock_gradient_norms.png)

The log-gradient plot shows the main convergence phases. Gradient Descent has a
long linear phase. Newton has a short global phase followed by rapid local
convergence near the minimizer.

### Task 4: BFGS via SciPy

SciPy BFGS was compared against the custom Gradient Descent and Newton
implementations on Rosenbrock.
Runtime values are from the local run and should be interpreted as approximate;
iteration counts are the more stable comparison.

| Method | Iterations | Runtime seconds | Final f | Final \|\|grad\|\| |
|---|---:|---:|---:|---:|
| Gradient Descent | 13757 | 0.2176 | 6.12e-13 | 9.83e-07 |
| Newton | 22 | 0.0003 | 3.74e-21 | 4.47e-10 |
| SciPy BFGS | 33 | 0.0015 | 8.98e-19 | 2.59e-08 |

![BFGS Comparison](../results/bfgs_comparison.png)

BFGS needed more iterations than exact Newton, but far fewer than Gradient
Descent. This matches the expected role of BFGS: it builds curvature information
without using the exact Hessian at every iteration.

### Task 5: Step Size Sensitivity

Fixed-step Gradient Descent was run on Rosenbrock with
`alpha in {0.001, 0.005, 0.01, 0.05}`.

| alpha | Status | Iterations | Final f | Final \|\|grad\|\| |
|---:|---|---:|---:|---:|
| 0.001 | max_iter | 20000 | 1.94e-08 | 1.24e-04 |
| 0.005 | max_iter | 20000 | 7.88e-01 | 2.00e+01 |
| 0.01 | diverged | 6 | 1.82e+105 | 1.12e+80 |
| 0.05 | diverged | 4 | 6.71e+57 | 2.97e+44 |

![Step Size Sensitivity](../results/step_size_sensitivity.png)

The experiment shows why line search matters. A very small step is stable but
slow. Moderately large fixed steps fail to reach the tolerance. Large fixed
steps diverge almost immediately.

### Testing and Reproducibility

The repository includes a unittest suite covering:

- quadratic convergence to the analytic minimizer,
- generated matrix condition numbers,
- Newton convergence on Rosenbrock,
- SciPy BFGS convergence on Rosenbrock,
- divergence for a large fixed GD step on Rosenbrock.

Run:

```bash
python -m unittest discover -s tests
```

Regenerate all figures and CSV files:

```bash
MPLCONFIGDIR=/private/tmp python src/run_all.py
```

### Conclusion

The experiments support the theoretical convergence anatomy:

- Gradient Descent is robust but sensitive to conditioning and step size.
- Newton's Method is extremely fast when the Hessian is available and reliable.
- BFGS is a strong practical compromise because it approximates second-order
  curvature while avoiding explicit Hessian solves.

For ill-conditioned smooth problems like Rosenbrock, curvature information is
the decisive advantage.

### References

- Nocedal, J., and Wright, S. J. Numerical Optimization, Chapters 3 and 6.
- Chong, E. K. P., and Zak, S. H. An Introduction to Optimization, Chapter 8.
- More, J. J., Garbow, B. S., and Hillstrom, K. E. Testing unconstrained optimization software. ACM TOMS, 7(1), 17-41, 1981.
