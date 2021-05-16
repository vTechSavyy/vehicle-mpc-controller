#include <trajectory.h>

Eigen::VectorXd Trajectory::getStateHistory(size_t dim) const {
  if (dim < 0 || dim >= state_dim_) {
    spdlog::warn(
        " Given dimension value of {} is invalid. State dimension is: {}. "
        "Returning a nan vector ",
        dim, state_dim_);
    return Eigen::VectorXd::Constant(num_steps_,
                                     std::numeric_limits<double>::quiet_NaN());
  }
  return state_history_.row(dim);
}

Eigen::VectorXd Trajectory::getInputHistory(size_t dim) const {
  if (dim < 0 || dim >= input_dim_) {
    spdlog::warn(
        " Given dimension value of {} is invalid. Input dimension is: {}.  "
        "Returning a nan vector ",
        dim, input_dim_);
    return Eigen::VectorXd::Constant(num_steps_,
                                     std::numeric_limits<double>::quiet_NaN());
  }
  return input_history_.row(dim);
}

Eigen::VectorXd Trajectory::getStateAt(size_t step) const {
  if (step < 0 || step >= num_steps_) {
    spdlog::warn(" Given step value of {} is out of trajectory range.  Returning a nan vector", step);
    return Eigen::VectorXd::Constant(state_dim_, 1,
                                     std::numeric_limits<double>::quiet_NaN());
  }

  return state_history_.col(step);
}

Eigen::VectorXd Trajectory::getInputAt(size_t step) const {
  if (step < 0 || step >= num_steps_) {
    spdlog::warn(" Given step value of {} is out of trajectory range.  Returning a nan vector", step);
    return Eigen::VectorXd::Constant(input_dim_, 1,
                                     std::numeric_limits<double>::quiet_NaN());
  }

  return input_history_.col(step);
}

void Trajectory::reset() {
  input_history_.setZero();
  state_history_.setZero();
}

bool Trajectory::setStateAt(size_t step, const Eigen::VectorXd& vec) {
  if (step < 0 || step >= num_steps_) {
    return false;
  }

  state_history_.col(step) = vec;
}

bool Trajectory::setInputAt(size_t step, const Eigen::VectorXd& vec) {
  if (step < 0 || step >= num_steps_) {
    return false;
  }

  input_history_.col(step) = vec;
}

bool Trajectory::setInputHistory(Eigen::MatrixXd&& input_history) {

  if (input_history.rows() != input_dim_) {
    spdlog::warn(" Dimension on input history doesn't match input dimension");
    return false;
  }

  if (input_history.cols() != num_steps_) {
    spdlog::warn(" Length on input history doesn't match number of steps");
    return false;
  }

  input_history_ = input_history;

  return true;
}

Eigen::VectorXd Trajectory::lerpState(double time) const {
  if (time < start_time_ || time > end_time_) {
    return Eigen::VectorXd::Constant(state_dim_, 1,
                                     std::numeric_limits<double>::quiet_NaN());
  }

  double delta_time = time - start_time_;

  size_t lower_index = std::floor(delta_time / time_step_);
  size_t upper_index = lower_index + 1;

  double fraction = (delta_time - (lower_index * time_step_)) / time_step_;

  return state_history_.col(lower_index) * fraction +
         state_history_.col(upper_index) * (1 - fraction);
}

Eigen::VectorXd Trajectory::lerpInput(double time) const {
  if (time < start_time_ || time > end_time_) {
    spdlog::warn(" In lerpInput, time of {} secs is outside input history range", time);
    return Eigen::VectorXd::Constant(input_dim_, 1,
                                     std::numeric_limits<double>::quiet_NaN());
  }

  double delta_time = time - start_time_;

  size_t lower_index = std::floor(delta_time / time_step_);
  size_t upper_index = lower_index + 1;

  double fraction = (delta_time - (lower_index * time_step_)) / time_step_;

  return input_history_.col(lower_index) * fraction +
         input_history_.col(upper_index) * (1 - fraction);
}

// Eigen::MatrixXd Trajectory::slice(size_t from, size_t to) {
//   return data_.block(data_.rows(), to - from, 0, from);
// }