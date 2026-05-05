# FCM2 Programming Assignment 3

**Piecewise Polynomial Interpolation and Cubic Splines**

Foundations of Computational Mathematics 2 — Florida State University, Spring 2026

## Overview

Implementation of piecewise polynomial interpolation and cubic spline algorithms in C++, with empirical validation and an application to estimating related functions from discrete data.

The project includes five interpolation methods:

1. **Barycentric Form 1** — Global polynomial interpolation using Chebyshev points (1st and 2nd kind) with mapping from [-1,1] to [a,b]
2. **Piecewise Polynomial g_d(x)** — Degrees d=1, 2, 3 plus Hermite cubic, supporting both uniform and Chebyshev-2 interior points, built using Newton divided differences
3. **Cubic Spline Code 1** — Moment (s''_i) parameterization on nonuniform meshes
4. **Cubic Spline Code 2** — Cubic B-spline basis on uniform meshes
5. **Evaluation routines** — For all methods including derivatives

Both spline codes support natural and clamped boundary conditions.

## Files

| File | Description |
|------|-------------|
| `program3.cpp` | Complete C++ implementation (1,600+ lines) |
| `program3_plots.m` | MATLAB script for generating all 11 figures |
| `generate_scipy_reference.py` | Python script for library comparison via binary I/O |
| `FCM2_Programming_Assignment_3_Report.pdf` | Full 32-page report |

## Build and Run

```bash
g++ -std=c++17 -O2 -o program3 program3.cpp -lm
./program3
```

### Optional: Library comparison (Test 8)

```bash
python3 generate_scipy_reference.py    # generates scipy_reference.bin
./program3                              # reads binary file, compares at machine precision
python3 generate_scipy_reference.py --verify  # reverse verification
```

Requires `scipy` and `numpy`. If `scipy_reference.bin` is not present, the C++ program skips Test 8 and runs all other tests normally.

## Validation Tests

The program runs 8 empirical tests:

| Test | Description | Key Result |
|------|-------------|------------|
| 1 | Polynomial reproduction (deg 0–3) | Error ~ 10⁻¹⁵ |
| 2 | Piecewise cubic reproduction | Error ~ 10⁻¹⁴ |
| 3 | Spline Code 1 vs Code 2 agreement | Diff ~ 10⁻¹⁶ |
| 4 | Convergence comparison (all methods) | Rates match theory |
| 5 | Boundary condition verification | Exact to machine precision |
| 6 | Random cubic reproduction (10 trials) | Error ~ 10⁻¹⁴ |
| 7 | Uniform vs Chebyshev-2 interior points | Comparable for low degree |
| 8 | Comparison against scipy via binary I/O | Diff ~ 10⁻¹⁶ |

## Task 2: Function Estimation from Discrete Data

Given 8 data points (t_i, y_i) on a nonuniform mesh, the program estimates three related functions using a natural cubic spline:

- y(t) — interpolated from the data
- f(t) = y(t) + t·y'(t) — requires a continuous derivative
- D(t) = exp(-t·y(t)) — derived from y(t)

Results are tabulated at 40 evaluation points from t=0.5 to t=20 with step 0.5. The program also investigates piecewise polynomial alternatives, demonstrating why standard piecewise methods (d=1,2,3) fail for computing f(t) due to derivative discontinuities, while the Hermite cubic provides a viable but less smooth alternative.

## MATLAB Plots

The MATLAB script generates 11 figures with all data hardcoded (no file uploads needed for MATLAB Online). Run the entire script in MATLAB Online or desktop MATLAB.

## Dependencies

- C++17 compiler (g++, clang++)
- MATLAB or MATLAB Online (for plots)
- Python 3 with scipy, numpy (optional, for Test 8 only)
