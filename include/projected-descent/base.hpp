#ifndef BASE_HPP
#define BASE_HPP

#include <Eigen/Dense>

namespace projected_descent {

/**
 * @enum Status
 * @brief Represents the termination reason.
 */
enum class Status {
  kConverged = 0,
  kExhausted = 1,
  kIllposed = 2
};

/**
 * @class ProjectedDescent
 * @brief Abstract base class for projected descent algorithms.
 * 
 * @tparam T Scalar type (e.g., float, double).
 * @tparam N Dimension of the optimization variable.
 */
template <typename T, int N>
class ProjectedDescent {
  using VectorNT = Eigen::Matrix<T, N, 1>;
  using MatrixNT = Eigen::Matrix<T, N, N>;

public:
  virtual ~ProjectedDescent() {};
  virtual Status Solve() = 0;
  virtual const VectorNT& get_point()     const = 0;
  virtual const T         get_objective() const = 0;
  virtual const VectorNT& get_gradient()  const = 0;

protected:
  /**
   * @brief Project a point onto the feasible set.
   * @param[in,out] point Point to be projected (in-place).
   */
  virtual void Project(VectorNT& point) = 0;

  /**
   * @brief Evaluates the cost function for a given point.
   * @param[in] point Point for which to evaluate the cost.
   * @return The cost value.
   */
  virtual T EvaluateCost(const VectorNT& point) = 0;

  /**
   * @brief Evaluates the gradient of the cost function for a given point.
   * @param[out] gradient Gradient vector (in-place).
   * @param[in] point Point for which to evaluate the gradient.
   */
  virtual void EvaluateGradient(VectorNT& gradient, const VectorNT& point) = 0;
  
  /**
   * @brief Initializes the optimization with the provided point.
   */
  virtual void WarmStart() = 0;

  /**
   * @brief Updates the current point's membership of the working and binding sets.
   */
  virtual void SetMembership() = 0;

  /**
   * @brief Performs projected line-search along the computed direction.
   */
  virtual void ArmijoSearch() = 0;
};

} //namespace projected_descent

#endif