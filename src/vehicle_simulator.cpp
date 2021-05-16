#include <vehicle_simulator.h>

VehicleSimulator::VehicleSimulator(const std::shared_ptr<VehicleModel> &model,
                                   double sim_start_time, double sim_end_time,
                                   double sample_time)
    : model_(model),
      sim_start_time_(sim_start_time),
      sim_end_time_(sim_end_time),
      sample_time_(sample_time) {
  // Initialize the integrator:
  integrator_ = std::unique_ptr<rk4_integrator>(new rk4_integrator);

  // 1. End time should be greater than start time
  num_steps_ = (sim_end_time_ - sim_start_time_) / sample_time_;

  num_boost_steps_per_sample_ = 5;

  system_trajectory_ = std::make_shared<Trajectory>(
      model_->getStateDim(), model_->getInputDim(), sim_start_time_,
      sim_end_time_, sample_time_);

  sim_current_step_ = 0;
  sim_current_time_ = sim_start_time_;
  rng_ = std::mt19937(rd_());
}

void VehicleSimulator::reset() {
  rng_ = std::mt19937(rd_());

  sim_current_step_ = 0;
  sim_current_time_ = sim_start_time_;

  system_trajectory_->reset();
}

bool VehicleSimulator::step(const Eigen::VectorXd &input) {
  if (input.size() != model_->getInputDim()) {
    spdlog::warn(" Supplied input dimension does "
                 "not match that of vehicle model");
    return false;
  }

  if (sim_current_time_ >= sim_end_time_) {
    spdlog::warn("Current simulation time of {} secs exceeds "
                 "simulation end time which is {} secs. Cannot perform step", 
              sim_current_time_, sim_end_time_);
    return false;
  }

  double time_step_increment = sample_time_ / num_boost_steps_per_sample_;

  Eigen::VectorXd current_state_eigen =
      system_trajectory_->getStateAt(sim_current_step_);

  std::vector<double> current_state(
      current_state_eigen.data(),
      current_state_eigen.data() + current_state_eigen.size());

  system_trajectory_->setInputAt(sim_current_step_, input);

  for (int i = 0; i < num_boost_steps_per_sample_; i++) {
    integrator_->do_step(
        [this, input](const std::vector<double> &x, std::vector<double> &dxdt,
                      const double t) {
          model_->getDynamicsSimFunc()(x, dxdt, input);
        },
        current_state, sim_current_time_, time_step_increment);

    sim_current_time_ += time_step_increment;
  }

  current_state_eigen = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      current_state.data(), current_state.size());

  //! Add process noise to the system state:  
  utils::addAWGN(current_state_eigen, model_->getProcessNoiseCovariance(), rng_);

  system_trajectory_->setStateAt(++sim_current_step_, current_state_eigen);

  return true;
}

bool VehicleSimulator::runSimulation() {
  double time_step_increment = sample_time_ / num_boost_steps_per_sample_;

  //! Get the initial system state an convert to stl vector:
  Eigen::VectorXd current_state_eigen =
      system_trajectory_->getStateAt(sim_current_step_);
  std::vector<double> current_state(
      current_state_eigen.data(),
      current_state_eigen.data() + current_state_eigen.size());

  //! Main sim loop:
  while (sim_current_step_ < num_steps_) {

    //! Inner runge-kutta loop for the current step:
    for (int i = 0; i < num_boost_steps_per_sample_; i++) {
      integrator_->do_step(
          [this](const std::vector<double> &x, std::vector<double> &dxdt,
                 const double t) {

            Eigen::VectorXd input = system_trajectory_->lerpInput(t);
            model_->getDynamicsSimFunc()(x, dxdt, input);
          },
          current_state, sim_current_time_, time_step_increment);

      sim_current_time_ += time_step_increment;
    }

    //! Convert back to eigen vector and set the state in system trajectory:
    system_trajectory_->setStateAt(
        ++sim_current_step_, Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
                                 current_state.data(), current_state.size()));
  }
  
}

bool VehicleSimulator::simLaneChange(double max_vel, double duration) {

  //! Reset the sim: 
  reset();

  double max_acc = 2.0;  // m/s^2

  if ((sim_end_time_ - sim_start_time_) <=
      (duration + max_vel / max_acc) * 2.0) {
    spdlog::error(
        " Duration of simulation insufficient for double lane change with "
        "given params. Min required duration is {} secs",
        (duration + max_vel / max_acc) * 2.0);
    return false;
  }

  Eigen::ArrayXd time_vector =
      Eigen::ArrayXd::LinSpaced(num_steps_, sim_start_time_, sim_end_time_);
  Eigen::MatrixXd input_trajectory(model_->getInputDim(), num_steps_);

  input_trajectory.setZero();

  // Acceleation input:
  double acc_duration = 1.5 * max_vel / max_acc;
  double acc_freq = 0.5 / acc_duration;
  int acc_steps = acc_duration / sample_time_;
  Eigen::ArrayXd temp_time_vector =
      Eigen::ArrayXd::LinSpaced(acc_steps, 0, acc_duration);

  input_trajectory.row(0).head(acc_steps) =
      max_acc * Eigen::sin(2 * M_PI * acc_freq * temp_time_vector);

  // Steering input:
  double offset = 1.0;  // secs
  double steering_mag = 0.03;
  double steering_start_time = acc_duration + offset;
  double steering_freq = 1 / duration;
  int steering_steps = duration / sample_time_;

  temp_time_vector = Eigen::ArrayXd::LinSpaced(steering_steps, 0, duration);

  input_trajectory.row(1).segment(acc_steps + (offset / sample_time_),
                                  steering_steps) =
      steering_mag * Eigen::sin(2 * M_PI * steering_freq * temp_time_vector);

  system_trajectory_->setInputHistory(std::move(input_trajectory));

  // Run the simulation:
  runSimulation();

  return true;
}

bool VehicleSimulator::simDoubleLaneChange(double max_vel, double duration) {

  //! Reset the sim: 
  reset();

  double max_acc = 4.0;  // m/s^2

  if ((sim_end_time_ - sim_start_time_) <=
      (duration + max_vel / max_acc) * 2.0) {
    spdlog::error(
        " Duration of simulation insufficient for double lane change with "
        "given params. Min required duration is {} secs",
        (duration + max_vel / max_acc) * 2.0);
    return false;
  }

  Eigen::ArrayXd time_vector =
      Eigen::ArrayXd::LinSpaced(num_steps_, sim_start_time_, sim_end_time_);
  Eigen::MatrixXd input_trajectory(model_->getInputDim(), num_steps_);

  input_trajectory.setZero();

  // Acceleation input:
  double acc_duration = 1.5 * max_vel / max_acc;
  double acc_freq = 0.5 / acc_duration;
  int acc_steps = acc_duration / sample_time_;
  Eigen::ArrayXd temp_time_vector =
      Eigen::ArrayXd::LinSpaced(acc_steps, 0, acc_duration);

  input_trajectory.row(0).head(acc_steps) =
      max_acc * Eigen::sin(2 * M_PI * acc_freq * temp_time_vector);

  // Steering input:
  double offset = 1.0;  // secs
  double steering_mag = 0.035;
  double steering_start_time = acc_duration + offset;
  double steering_freq = 1 / duration;
  int steering_steps = duration / sample_time_;

  temp_time_vector = Eigen::ArrayXd::LinSpaced(steering_steps, 0, duration);

  input_trajectory.row(1).segment(acc_steps + (offset / sample_time_),
                                  steering_steps) =
      steering_mag * Eigen::sin(2 * M_PI * steering_freq * temp_time_vector);

  input_trajectory.row(1).segment(
      acc_steps + 2 * (offset / sample_time_) + steering_steps,
      steering_steps) = -1 * steering_mag *
                        Eigen::sin(2 * M_PI * steering_freq * temp_time_vector);

  system_trajectory_->setInputHistory(std::move(input_trajectory));

  // Run the simulation:
  runSimulation();

  return true;
}
