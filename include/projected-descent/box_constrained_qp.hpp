#ifndef BOX_CONSTRAINED_QP_HPP
#define BOX_CONSTRAINED_QP_HPP

#include <iostream>
#include <Eigen/Dense>
#include "base.hpp"

namespace projected_descent {

template <typename T, int N>
class BoxConstrainedQP : public ProjectedDescent<T, N> {
  using VectorNT = Eigen::Matrix<T, N, 1>;
  using MatrixNT = Eigen::Matrix<T, N, N>;
  using DiagNT   = Eigen::DiagonalMatrix<T, N>;
  using LLTNT    = Eigen::LLT<MatrixNT>;

public:
  // QP Problem
  MatrixNT quadratic_coefs_;
  VectorNT linear_coefs_;
  VectorNT lower_bounds_;
  VectorNT upper_bounds_;
  VectorNT start_point_;

  BoxConstrainedQP() :
    quadratic_coefs_(), linear_coefs_(), lower_bounds_(), upper_bounds_(), start_point_(),
    tol_cost_(T(1e-6)), tol_binding_(T(0.1)), max_steps_(100), max_searches_(20),
    armijo_slope_(T(0.5)), armijo_backtrack_(T(0.6)), verbose_(false),
    current_point_(), current_cost_(), previous_cost_(), current_gradient_(),
    point_steepest_(), point_delta_(), binding_set_mask_(), working_set_mask_(), previous_working_set_mask_(),
    hessian_working_(), factorization_(), ascent_direction_(),
    trial_point_(), trial_cost_()
  {}

  Status Solve() override;

  void set_tol_cost(T tol_cost)                 { tol_cost_ = tol_cost; }
  void set_tol_binding(T tol_binding)           { tol_binding_ = tol_binding; }
  void set_max_steps(int max_steps)             { max_steps_ = max_steps; }
  void set_max_searches(int max_searches)       { max_searches_ = max_searches; }
  void set_armijo_slope(T armijo_slope)         { armijo_slope_ = armijo_slope; }
  void set_armijo_backtrack(T armijo_backtrack) { armijo_backtrack_ = armijo_backtrack; }
  void set_verbose(bool verbose)                { verbose_ = verbose;}

  const VectorNT& get_point()     const override { return current_point_; }
  const T         get_objective() const override { return current_cost_; }
  const VectorNT& get_gradient()  const override { return current_gradient_; }
  const DiagNT&   get_binding_set_mask() const   { return binding_set_mask_; }
  const DiagNT&   get_working_set_mask() const   { return working_set_mask_; }
  const LLTNT&    get_factorization()    const   { return factorization_; }

private:
  // Options
  T    tol_cost_;
  T    tol_binding_;
  int  max_steps_;
  int  max_searches_;
  T    armijo_slope_;
  T    armijo_backtrack_;
  bool verbose_;

  // State
  VectorNT current_point_;
  T        current_cost_;
  T        previous_cost_;
  VectorNT current_gradient_;

  // Constraints state
  VectorNT point_steepest_;
  VectorNT point_delta_;
  DiagNT   binding_set_mask_;
  DiagNT   working_set_mask_;
  DiagNT   previous_working_set_mask_;

  // Direction state
  MatrixNT hessian_working_;
  LLTNT    factorization_;
  VectorNT ascent_direction_;

  // Line-search state
  VectorNT trial_point_;
  T        trial_cost_;
  
  void   Project(VectorNT&) override;
  T      EvaluateCost(const VectorNT&) override;
  void   EvaluateGradient(VectorNT&, const VectorNT&) override;
  void   WarmStart() override;
  void   SetMembership() override;
  Status ComputeDirection(const bool factorize);
  void   ArmijoSearch() override;
};

template <typename T, int N>
Status BoxConstrainedQP<T, N>::Solve() {

  // Reset
  previous_cost_ = std::numeric_limits<T>::max();
  previous_working_set_mask_.setZero();
  bool working_set_convergence = false;
  bool f_convergence = false;
  bool factorize = true;
  Status direction_status = Status::kConverged;

  WarmStart();

  for (int i = 0; i < max_steps_; ++i) {

    EvaluateGradient(current_gradient_, current_point_);

    // Identify which components of current_point_ belong to working and binding sets
    SetMembership();
    working_set_convergence = (working_set_mask_.diagonal() == previous_working_set_mask_.diagonal());
    previous_working_set_mask_ = working_set_mask_;
    
    // Compute Newton direction
    factorize = (i == 0 || !working_set_convergence);
    direction_status = ComputeDirection(factorize);
    if (direction_status == Status::kIllposed) { return direction_status; }

    // Perform Armijo line-search
    ArmijoSearch();    
    current_point_ = trial_point_;
    previous_cost_ = current_cost_;
    current_cost_ = trial_cost_;

    // Check for objective convergence
    f_convergence = (previous_cost_ - current_cost_ < tol_cost_ * (1 + std::abs(current_cost_)));
    if (f_convergence && working_set_convergence) {
      if (verbose_) {
        std::cout << "Converged after " << i + 1 << " iterations." << std::endl;
      }
      return Status::kConverged;
    }
  }

  if (verbose_) {
    std::cout << "Maximum iterations reached without convergence." << std::endl;
  }
  return Status::kExhausted;
}

template <typename T, int N>
void BoxConstrainedQP<T, N>::Project(VectorNT& point) {
  point = point.cwiseMax(lower_bounds_).cwiseMin(upper_bounds_);
}

template <typename T, int N>
T BoxConstrainedQP<T, N>::EvaluateCost(const VectorNT& point) {
  return T(0.5) * point.dot(quadratic_coefs_ * point) + linear_coefs_.dot(point);
}

template <typename T, int N>
void BoxConstrainedQP<T, N>::EvaluateGradient(VectorNT& gradient, const VectorNT& point) {

  gradient.noalias() = quadratic_coefs_ * point + linear_coefs_;

  if (verbose_) {
    std::cout << "Gradient: " << gradient.transpose() << std::endl;
  }
}

template <typename T, int N>
void BoxConstrainedQP<T, N>::WarmStart() {

  current_point_ = start_point_;
  Project(current_point_);
  current_cost_ = EvaluateCost(current_point_);

  if (verbose_) {
    std::cout << "Start point: " << current_point_.transpose() << std::endl;
    std::cout << "Start cost: " << current_cost_ << std::endl;
  }
}

template <typename T, int N>
void BoxConstrainedQP<T, N>::SetMembership() {

  point_steepest_.noalias() = current_point_ - current_gradient_;
  Project(point_steepest_);
  point_delta_.noalias() = current_point_ - point_steepest_;
  T tol_Bx = std::min(point_delta_.norm(), tol_binding_);

  for (int j = 0; j < N; ++j) {
    if (lower_bounds_[j] <= current_point_[j] && current_point_[j] <= (lower_bounds_[j] + tol_Bx) && current_gradient_[j] > 0) {
      working_set_mask_.diagonal()[j] = T(0);
      binding_set_mask_.diagonal()[j] = T(1);
    } else if ((upper_bounds_[j] - tol_Bx) <= current_point_[j] && current_point_[j] <= upper_bounds_[j] && current_gradient_[j] < 0) {
      working_set_mask_.diagonal()[j] = T(0);
      binding_set_mask_.diagonal()[j] = T(1);
    } else {
      working_set_mask_.diagonal()[j] = T(1);
      binding_set_mask_.diagonal()[j] = T(0);
    }
  }
}

template <typename T, int N>
Status BoxConstrainedQP<T, N>::ComputeDirection(const bool factorize) {

  if (factorize) {
    // Use the elements of Q associated with the working set
    hessian_working_ = working_set_mask_ * quadratic_coefs_ * working_set_mask_;
    // Also use the diagonal elements associated with the binding set
    hessian_working_.diagonal() += binding_set_mask_ * quadratic_coefs_.diagonal();

    // Perform the Cholesky factorization of hessian_working_
    factorization_.compute(hessian_working_);

    // Check if the factorization was successful
    if (factorization_.info() == Eigen::Success) {
      if (verbose_) {
        std::cout << "Successfully performed Cholesky factorization." << std::endl;
      }
    } else {
      if (verbose_) {
        std::cout << "Cholesky factorization failed. Problem may be illposed." << std::endl;
      }
      return Status::kIllposed;
    }
  }

  ascent_direction_ = factorization_.solve(current_gradient_);
  if (verbose_) {
    std::cout << "Newton ascent direction: " << ascent_direction_.transpose() << std::endl;
  }
  return Status::kConverged;
}

template <typename T, int N>
void BoxConstrainedQP<T, N>::ArmijoSearch() {

  T    step_size = T(0);
  T    threshold   = T(0);
  bool admissible  = false;

  // Projected Armijo line-search
  T dphi0 = current_gradient_.dot(working_set_mask_ * ascent_direction_);

  for (int k = 0; k < max_searches_; ++k) {

    step_size = std::pow(armijo_backtrack_, k);
    trial_point_.noalias() = current_point_ - step_size * ascent_direction_;
    Project(trial_point_);

    trial_cost_ = EvaluateCost(trial_point_);

    // Check admissibility of the trial step
    threshold = step_size * dphi0 - (current_gradient_.dot(binding_set_mask_ * (current_point_ - trial_point_)));
    admissible = (trial_cost_ - current_cost_) <= (armijo_slope_ * threshold);

    if (admissible) {
      if (verbose_) {
        std::cout << "Admissible step found after " << k + 1 << " searches." << std::endl;
        std::cout << "Line-search step size: " << step_size << std::endl;
        std::cout << "Updated point: " << trial_point_.transpose() << std::endl;
        std::cout << "Updated cost: " << trial_cost_ << std::endl;
      }
      break;
    }
  }
  
  if (verbose_ && !admissible ) {
    std::cout << "Line-search exhausted without finding an admissible step." << std::endl;
    std::cout << "Line-search step size: " << step_size << std::endl;
    std::cout << "Updated point: " << trial_point_.transpose() << std::endl;
    std::cout << "Updated cost: " << trial_cost_ << std::endl;
  }
}

} //namespace projected_descent 

#endif