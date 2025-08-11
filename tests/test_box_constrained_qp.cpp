#include <limits>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "projected-descent/box_constrained_qp.hpp"

class BoxConstrainedQPTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(BoxConstrainedQP, Unconstrained) {
  using T = double;
  constexpr int N = 2;

  projected_descent::BoxConstrainedQP<T, N> solver_unc;

  solver_unc.quadratic_coefs_ << 2.0, -1.0,
                                -1.0,  2.0;
  solver_unc.linear_coefs_ = {-1.0, -2.0};
  T inf = std::numeric_limits<T>::infinity();
  solver_unc.lower_bounds_ = {-inf, -inf};
  solver_unc.upper_bounds_ = { inf,  inf};
  solver_unc.start_point_  = { 0.0,  0.0};

  solver_unc.Solve();

  EXPECT_NEAR(solver_unc.get_point()[0],  4.0/3, 1e-6);
  EXPECT_NEAR(solver_unc.get_point()[1],  5.0/3, 1e-6);
  EXPECT_NEAR(solver_unc.get_objective(), -7.0/3, 1e-6);
}

TEST(BoxConstrainedQP, Constrained) {
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

  EXPECT_NEAR(solver.get_point()[0],  2.0, 1e-6);
  EXPECT_NEAR(solver.get_point()[1],  2.0, 1e-6);
  EXPECT_NEAR(solver.get_objective(), -2.0, 1e-6);
}

TEST(BoxConstrainedQP, UnsuccessfulCholesky) {
  using T = double;
  constexpr int N = 2;

  projected_descent::BoxConstrainedQP<T, N> solver_ill;

  solver_ill.quadratic_coefs_ << 1.0,  0.0,
                                 0.0, -1.0;
  solver_ill.linear_coefs_ = {-1.0, -2.0};
  T inf = std::numeric_limits<T>::infinity();
  solver_ill.lower_bounds_ = {-inf, -inf};
  solver_ill.upper_bounds_ = { inf,  inf};
  solver_ill.start_point_  = {1.0, 1.0};

  projected_descent::Status status = solver_ill.Solve();

  EXPECT_EQ(status, projected_descent::Status::kIllposed);
}