#pragma once

#include <spdlog/spdlog.h>
#include <eigen3/Eigen/Dense>

/**
 * @brief Class used to represnt the trajectory (states and inputs) of a dynamic system
 *
 */
class Trajectory {
 public:
  Trajectory(size_t state_dim, size_t input_dim, double start_time = 0.0,
             double end_time = 10.0, double time_step = 0.1)
      : state_dim_(state_dim),
        input_dim_(input_dim),
        start_time_(start_time),
        end_time_(end_time),
        time_step_(time_step) {
    num_steps_ = std::floor((end_time - start_time) / time_step);
    time_history_ =
        Eigen::VectorXd::LinSpaced(num_steps_, start_time, end_time);
    state_history_ = Eigen::MatrixXd::Zero(state_dim_, num_steps_);
    input_history_ = Eigen::MatrixXd::Zero(input_dim_, num_steps_);
  }

  size_t getNumTimeSteps() const { return num_steps_;};

  const Eigen::VectorXd& getTimeHistory() const { return time_history_; }

  const Eigen::MatrixXd& getStateHistory() const { return state_history_; }

  const Eigen::MatrixXd& getInputHistory() const { return input_history_; }
  
  /**
   * @brief Get the history of a particular state variable, specified by the dimension
   * 
   * @param dim 
   * @return Eigen::VectorXd 
   */
  Eigen::VectorXd getStateHistory(size_t dim) const;

  /**
   * @brief Get the  history of a particular input variable, specified by the dimension
   * 
   * @param dim 
   * @return Eigen::VectorXd 
   */
  Eigen::VectorXd getInputHistory(size_t dim) const;
  
  /**
   * @brief Get the value of the state vector at a given time step
   * 
   * @param step 
   * @return Eigen::VectorXd 
   */
  Eigen::VectorXd getStateAt(size_t step) const;

  /**
   * @brief Get the value of the input vector at a given time step
   * 
   * @param step 
   * @return Eigen::VectorXd 
   */
  Eigen::VectorXd getInputAt(size_t step) const;
 
  /**
   * @brief Get a linearly interpolated value of the state vector at the given time
   * 
   * @param time 
   * @return Eigen::VectorXd 
   */
  Eigen::VectorXd lerpState(double time) const;
 
  /**
   * @brief Get a linearly interpolated value of the input vector at the given time
   * 
   * @param time 
   * @return Eigen::VectorXd 
   */
  Eigen::VectorXd lerpInput(double time) const;

  void reset();
  
  /**
   * @brief Set the value of the state vector at the given time step 
   * 
   * @param step 
   * @param vec 
   * @return true If able to set the state
   * @return false If the given step is outside the trajectory range
   */
  bool setStateAt(size_t step, const Eigen::VectorXd& vec);
  
  /**
   * @brief Set the value of the input vector at the given time step 
   * 
   * @param step 
   * @param vec 
   * @return true If able to set the input
   * @return false If the given step is outside the trajectory range
   */
  bool setInputAt(size_t step, const Eigen::VectorXd& vec);

  /**
   * @brief Set the input history for the system:
   * 
   * @param input_history 
   * @return true 
   * @return false 
   */
  bool setInputHistory(Eigen::MatrixXd&& input_history); 

 private:
  //! Start time:
  double start_time_;  // in seconds

  //! End time:
  double end_time_;  // in seconds

  //! Time step:
  double time_step_;  // in seconds

  //! Number of discrete time steps in the trajectory:
  size_t num_steps_;

  //! Dimensions of the state and input of the system:
  size_t state_dim_;
  size_t input_dim_;

  //! Time, state and input history:
  Eigen::VectorXd time_history_;
  
  Eigen::MatrixXd state_history_;

  Eigen::MatrixXd input_history_;
};

typedef std::shared_ptr<Trajectory> sp_to_traj;