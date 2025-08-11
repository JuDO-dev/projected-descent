# Projected-Descent Optimization
A simple, header-only C++ library for solving constrained optimization problems using **projected-descent** methods.

| Constraint Set | QP | NLP |
|---|---|---|
| Box            | ✅ Supported | See [Progradio.jl](https://github.com/JuDO-dev/Progradio.jl) | 
| Simplex        | 📝 To-do     | See [Progradio.jl](https://github.com/JuDO-dev/Progradio.jl) |
| General Linear | 📝 To-do     | 📝 To-do |

## Features
- Uses [Eigen](https://eigen.tuxfamily.org/) for fast linear algebra.
- Supports different scalar types (e.g., float, double) and fixed-size arrays via templates.
- Compatible with C++11 or higher.

## Getting started
1. Ensure you have [Eigen](https://eigen.tuxfamily.org/).
2. Include the relevant header file.
```cpp
#include "projected-descent/box_constrained_qp.hpp"
```

## Example usage
```cpp
#include "projected-descent/box_constrained_qp.hpp"

using T = double;
constexpr int N = 2;
projected_descent::BoxConstrainedQP<T, N> solver;

solver.quadratic_coefs_ << 2.0, -1.0,
                          -1.0,  2.0;
solver.linear_coefs_ = {-1.0, -2.0};
solver.lower_bounds_ = {2.0, 2.0};
solver.upper_bounds_ = {4.0, 4.0};
solver.start_point_  = {1.0, 5.0};

solver.Solve();
double x_0 = solver.get_point()[0];
double x_1 = solver.get_point()[1];
double f_x = solver.get_objective();
```

## References
[Bertsekas, DP *Projected Newton methods for optimization problems with simple constraints* 1982](https://epubs.siam.org/doi/abs/10.1137/0320018)
