#include <mpc_controller.h>

MPCController::MPCController(const std::shared_ptr<VehicleModel>& model,
                             Eigen::MatrixXd& Q, Eigen::MatrixXd& Q_terminal,
                             Eigen::MatrixXd& R, double Ts,
                             double prediction_horizon_time)
    : model_(model),
      Q_(Q),
      Q_terminal_(Q_terminal),
      R_(R),
      sample_time_(Ts),
      prediction_horizon_time_(prediction_horizon_time) {

  // Dimensions of the model:
  model_state_dim_ = model_->getStateDim();
  model_input_dim_ = model_->getInputDim();

  prediction_horizon_steps_ = std::round(prediction_horizon_time_ / sample_time_);

  current_time_step_ = 0;

  //! Compute the number of variables and constraints:
  num_opt_vars_ = (prediction_horizon_steps_ + 1) * model_state_dim_ + prediction_horizon_steps_ * model_input_dim_;
  num_opt_contrs_ = (prediction_horizon_steps_ + 1) * model_state_dim_ + (2 * prediction_horizon_steps_ ) * model_input_dim_;

  //! Initialize the sparse matrices:
  hessian_ = Eigen::SparseMatrix<double>(num_opt_vars_, num_opt_vars_);
  constraint_mat_ = Eigen::SparseMatrix<double>(num_opt_contrs_, num_opt_vars_);

  gradient_ = Eigen::VectorXd::Zero(num_opt_vars_);

  //! Initialize the upper and lower bound constraint vectors:
  lower_bound_ = Eigen::VectorXd::Zero(num_opt_contrs_);
  upper_bound_ = Eigen::VectorXd::Zero(num_opt_contrs_);

  solver_ = std::unique_ptr<OsqpEigen::Solver>(new OsqpEigen::Solver());

}

void MPCController::setReferenceTrajectory(const sp_to_traj& ref_trajectory) {

  ref_trajectory_ = ref_trajectory;

  //! Compute the linearized systems:
  size_t num_time_steps = ref_trajectory_->getNumTimeSteps();
  linearized_systems_.clear();
  for (size_t i = 0; i < num_time_steps; i++) {
    linearized_systems_.emplace_back(Eigen::MatrixXd::Identity(model_state_dim_, model_state_dim_), 
                                     Eigen::MatrixXd::Zero(model_state_dim_, model_input_dim_));
  }

  for (size_t step = 0; step < num_time_steps; step++) {
    model_->getLinearizedDynamicsDiscreteRK4(
        linearized_systems_[step].first, linearized_systems_[step].second,
        ref_trajectory_->getStateAt(step), ref_trajectory_->getInputAt(step),
        sample_time_);
  }

  prev_input_ = ref_trajectory_->getInputAt(current_time_step_);

}

void MPCController::computeHessianAndGradient() {

  //! 1. Hessian matrix:

  //! a. Initialize the triplet list:
  std::vector<Triplet> hessian_triplet_list;
  hessian_triplet_list.reserve( (prediction_horizon_steps_+ 1)* (model_state_dim_ + model_input_dim_));

  //! b. Add the state weighting matrices (i.e Q matrices)
  for (size_t idx = 0; idx < prediction_horizon_steps_; idx++) {
    utils::setTripletsFromMatrix(Q_, idx * model_state_dim_,
                                     idx * model_state_dim_,
                                     hessian_triplet_list, true);
  }

  //! c. Add the terminal state weighting matrix (i.e Qf matrix):
  utils::setTripletsFromMatrix(
      Q_terminal_, prediction_horizon_steps_ * model_state_dim_,
      prediction_horizon_steps_ * model_state_dim_, hessian_triplet_list,
      true);

  //! d. Add the input weighting matrices (i.e R matrices):
  size_t row_offset = (prediction_horizon_steps_ + 1) * model_state_dim_;
  size_t col_offset = row_offset;
  for (size_t idx = 0; idx < prediction_horizon_steps_; idx++) {
    utils::setTripletsFromMatrix(
        R_, row_offset + idx * model_input_dim_,
        col_offset + idx * model_input_dim_, hessian_triplet_list, true);
  }

  hessian_.setFromTriplets(hessian_triplet_list.begin(), hessian_triplet_list.end());

}

void MPCController::computeConstraintMatrix() {

  std::vector<Triplet> constraint_mat_triplet_list;

  //! 1. Add the initial state constraint: 
  utils::setTripletsFromMatrix(Eigen::MatrixXd::Identity(model_state_dim_,
                                         model_state_dim_), 0, 0, constraint_mat_triplet_list, true);

  spdlog::debug(" Initial state constraint added to linear constraint matrix");

  //! 2. Add the system dynamics constraints
  //!   { A_k * x_k + B_k * u_k - I * x_(k+1) = 0 }
  size_t row_offset = model_state_dim_;
  size_t col_offset = (prediction_horizon_steps_ + 1) * model_state_dim_;
  for (size_t idx = 0; idx < prediction_horizon_steps_; idx++) {
    //! Add the identity matrix for x_(k+1):
    utils::setTripletsFromMatrix(
        -1.0 * Eigen::MatrixXd::Identity(model_state_dim_,
                                         model_state_dim_),
        row_offset + idx * model_state_dim_, (idx + 1) * model_state_dim_,
        constraint_mat_triplet_list, true);

    //! Add the A matrix:
    utils::setTripletsFromMatrix(
        linearized_systems_[current_time_step_ + idx].first,
        row_offset + idx * model_state_dim_, idx * model_state_dim_,
        constraint_mat_triplet_list, false);

    //! Add the B matrix:
    utils::setTripletsFromMatrix(
        linearized_systems_[current_time_step_ + idx].second,
        row_offset + idx * model_state_dim_, col_offset + idx * model_input_dim_,
        constraint_mat_triplet_list, false);
  }

  spdlog::debug(" System dynamics constraints added to linear constraint matrix");

  //! 3. Add the input magnitude constraints:
  //! lower < u_k - u_ref < upper 
  row_offset = col_offset;

  for (size_t idx = 0; idx < prediction_horizon_steps_; idx++) {
    utils::setTripletsFromMatrix(
        Eigen::MatrixXd::Identity(model_input_dim_, model_input_dim_),
        row_offset + idx * model_input_dim_,
        col_offset + idx * model_input_dim_, constraint_mat_triplet_list, true);
  }

  spdlog::debug(" Input magnitude constraints added to linear constraint matrix");

  //! 4 Constraint on the initial input:
  row_offset += prediction_horizon_steps_ * model_input_dim_;
  utils::setTripletsFromMatrix(
        Eigen::MatrixXd::Identity(model_input_dim_, model_input_dim_),
        row_offset,
        col_offset, constraint_mat_triplet_list, true);


  //! 5. Add the input rate constraints:
  row_offset += model_input_dim_;
  for (size_t idx = 0; idx < prediction_horizon_steps_ - 1; idx++) {
    utils::setTripletsFromMatrix(
        -1.0 * Eigen::MatrixXd::Identity(model_input_dim_, model_input_dim_),
        row_offset + idx * model_input_dim_,
        col_offset + idx * model_input_dim_, constraint_mat_triplet_list, true);

    utils::setTripletsFromMatrix(
        Eigen::MatrixXd::Identity(model_input_dim_, model_input_dim_),
        row_offset + idx * model_input_dim_,
        col_offset + (idx + 1) * model_input_dim_, constraint_mat_triplet_list, true);
  }

  spdlog::debug(" Input rate constraints added to linear constraint matrix");

  constraint_mat_.setFromTriplets(constraint_mat_triplet_list.begin(), 
                                  constraint_mat_triplet_list.end());

  spdlog::debug(" Constraint matrix set from triplet list");

}

void MPCController::computeConstraintVector() {

  //! 1. Add the initial state constraint: 
  lower_bound_.head(model_state_dim_) = current_state_;
  upper_bound_.head(model_state_dim_) = current_state_;

  spdlog::debug(" Initial state constraint added to linear constraint vector");

  //! 2. Add the input magnitude bounds:
  size_t row_offset = (prediction_horizon_steps_ + 1) * model_state_dim_;
  for (size_t idx = 0; idx < prediction_horizon_steps_; idx++) {

    Eigen::VectorXd curr_input = ref_trajectory_->getInputAt(current_time_step_ + idx);
    lower_bound_.segment(row_offset + idx * model_input_dim_, model_input_dim_) 
            = model_->getInputLowerLimits() - curr_input;
    upper_bound_.segment(row_offset + idx * model_input_dim_, model_input_dim_) 
            = model_->getInputUpperLimits() - curr_input;
  }

  spdlog::debug(" Initial magnitude constraint added to linear constraint vector");

  //! 3. Add the initial input rate constraint: 
  row_offset += prediction_horizon_steps_ * model_input_dim_;
  Eigen::VectorXd curr_ref_input = ref_trajectory_->getInputAt(current_time_step_);
  Eigen::VectorXd delta = curr_ref_input - prev_input_;

  lower_bound_.segment(row_offset, model_input_dim_) 
          = sample_time_ * model_->getInputRateLowerLimits() - delta;
    upper_bound_.segment(row_offset, model_input_dim_) 
          = sample_time_ * model_->getInputRateUpperLimits() - delta;

  //! 4. Add the input rate bounds: 
  row_offset += model_input_dim_;
  for (size_t idx = 0; idx < prediction_horizon_steps_ - 1; idx++) {

    Eigen::VectorXd curr_ref_input = ref_trajectory_->getInputAt(current_time_step_ + idx);
    Eigen::VectorXd next_ref_input = ref_trajectory_->getInputAt(current_time_step_ + idx + 1);
    Eigen::VectorXd delta_ref_input = next_ref_input - curr_ref_input;

    lower_bound_.segment(row_offset + idx * model_input_dim_, model_input_dim_) 
          = sample_time_ * model_->getInputRateLowerLimits() - delta_ref_input;
    upper_bound_.segment(row_offset + idx * model_input_dim_, model_input_dim_) 
          = sample_time_ * model_->getInputRateUpperLimits() - delta_ref_input;
  }

  spdlog::debug(" Initial rate constraint added to linear constraint vector");

}

bool MPCController::updateConstraints() {

  //! 1. Update the constraint matrix:
  computeConstraintMatrix(); 
  solver_->updateLinearConstraintsMatrix(constraint_mat_);

  //! 2. Update the constraint bounds:
  computeConstraintVector();
  solver_->updateBounds(lower_bound_, upper_bound_);

  return true;
}

bool MPCController::init() {

  solver_->settings()->setVerbosity(false);
  solver_->settings()->setWarmStart(true);

  // set the initial data of the QP solver
  solver_->data()->setNumberOfVariables(num_opt_vars_);
  solver_->data()->setNumberOfConstraints(num_opt_contrs_);

  //! Compute the hessian matrix and gradient vector: 
  computeHessianAndGradient();

  //! Compute the linear constraint matrix: 
  computeConstraintMatrix();

  //! Compute constraint vectors (lower and upper bounds):
  computeConstraintVector();


  if(!solver_->data()->setHessianMatrix(hessian_)) {
    return false;
  } 


  if(!solver_->data()->setGradient(gradient_)) { 
    return false;
  }


  if(!solver_->data()->setLinearConstraintsMatrix(constraint_mat_)) {
    return false;
  }


  if(!solver_->data()->setLowerBound(lower_bound_)) {
    return false;
  }


  if(!solver_->data()->setUpperBound(upper_bound_)) {
    return false;
  }

  // instantiate the solver
  if(!solver_->initSolver()) {
    return false;
  }

  return true;

}

bool MPCController::solveAndStep(Eigen::VectorXd& solution) {

  // solve the QP problem
  if(!solver_->solve()) {
    return false;
  }

  // Extract the controller input
  Eigen::VectorXd qp_solution = solver_->getSolution();

  Eigen::VectorXd delta_solution 
            = qp_solution.segment(model_state_dim_ * (prediction_horizon_steps_ + 1), model_input_dim_);

  solution = ref_trajectory_->getInputAt(current_time_step_) + delta_solution;

  //! Increment current time step:
  current_time_step_++;

  //! Store the previous input to the system:
  prev_input_ = solution;

  return true;
}