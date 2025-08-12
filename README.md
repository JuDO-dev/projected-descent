# Projected-Descent Optimization
A simple, header-only C++ library for solving constrained optimization problems using **projected-descent** methods.

| Constraint Set | QP | NLP |
|---|---|---|
| Box            | ✅ Supported | See [Progradio.jl](https://github.com/JuDO-dev/Progradio.jl) | 
| Simplex        | 📝 To-do     | See [Progradio.jl](https://github.com/JuDO-dev/Progradio.jl) |
| General Linear | 📝 To-do     | 📝 To-do |

## Features
- **[Eigen](https://eigen.tuxfamily.org/)** for fast linear algebra operations.
- **Templates** allow different scalar types (e.g., float, double), and fixed-size arrays.
- **C++11** for wide compatibility across compilers.

## Get via CMake
Simply add the following to your CMakeLists.txt file.
```cmake
include(FetchContent)
FetchContent_Declare(
  eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_Declare(
    projected-descent
    GIT_REPOSITORY https://github.com/JuDO-dev/projected-descent.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(eigen3 projected-descent)
target_link_libraries(YOUR_TARGET
  Eigen3::Eigen
  projected_descent
)
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
1. [Bertsekas, DP *Projected Newton methods for optimization problems with simple constraints* 1982](https://epubs.siam.org/doi/abs/10.1137/0320018)
