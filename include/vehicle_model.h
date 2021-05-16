#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>
#include <utility>

class VehicleModel {
 public:
  using state_t = Eigen::VectorXd;
  using input_t = Eigen::VectorXd;

  VehicleModel(const std::string& name, int dim_state, int dim_input);

  virtual ~VehicleModel() = 0;

  /**
   * @brief Computes the state transition matrix (F) and the input-state mapping matrix (G)
   *        for the continuous linearized system
   * 
   * @param F 
   * @param G 
   * @param x 
   * @param u 
   */
  virtual void getLinearizedDynamicsContinuous(Eigen::MatrixXd& F,
                                               Eigen::MatrixXd& G,
                                               const state_t& x,
                                               const input_t& u) = 0;

  
  /**
   * @brief Computes the state transition matrix (A) and the input-state mapping matrix (B)
   *        for the discrete linearized system. Internally uses 4th order Runge-Kutta integration
   * 
   * @param A 
   * @param B 
   * @param x 
   * @param u 
   * @param delta_t 
   */
  void getLinearizedDynamicsDiscreteRK4(Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                                        const state_t& x, const input_t& u,
                                        double delta_t);

  const std::function<void(const state_t&, state_t&, const input_t&)>&
  getDynamicsFunc() const {
    return dynamics_function_;
  }

  const std::function<void(const std::vector<double>&, std::vector<double>&,
                           const input_t&)>&
  getDynamicsSimFunc() const {
    return dynamics_function_sim_;
  }

  int getStateDim() const { return dim_state_; }

  int getInputDim() const { return dim_input_; }

  Eigen::VectorXd getInputLowerLimits() { return input_mag_constraints_.first; }

  Eigen::VectorXd getInputUpperLimits() {
    return input_mag_constraints_.second;
  }

  Eigen::VectorXd getInputRateLowerLimits() {
    return input_rate_constraints_.first;
  }

  Eigen::VectorXd getInputRateUpperLimits() {
    return input_rate_constraints_.second;
  }

  Eigen::MatrixXd getProcessNoiseCovariance() { return Q_; }

  void setInputMagConstraints(const Eigen::VectorXd& lower,
                              const Eigen::VectorXd& upper) {
    input_mag_constraints_.first = lower;
    input_mag_constraints_.second = upper;
  }

  void setInputRateConstraints(const Eigen::VectorXd& rate_limit) {
    input_rate_constraints_.first = -1.0 * rate_limit;
    input_rate_constraints_.second = rate_limit;
  }

  void setProcessNoiseCovariance(const Eigen::MatrixXd& Q) { Q_ = Q; }

 protected:
  //! Name of the model:
  std::string name_;

  //! Dimension of the state vector:
  const int dim_state_;

  //! Dimension of the input vector:
  const int dim_input_;

  //! Dynamics function using Eigen data types:
  std::function<void(const state_t&, state_t&, const input_t&)>
      dynamics_function_;

  //! Dynamics function using STL data types (only used with the Boost integrator in simulation)
  std::function<void(const std::vector<double>&, std::vector<double>&,
                     const input_t&)>
      dynamics_function_sim_;

  //! Input magnitude constraints:
  std::pair<Eigen::VectorXd, Eigen::VectorXd> input_mag_constraints_;

  //! Input rate constraints:
  std::pair<Eigen::VectorXd, Eigen::VectorXd> input_rate_constraints_;

  //! Process noise covariance matrix:
  Eigen::MatrixXd Q_;

  int nRKSteps_disc_;
};
