#include <bicycle_model.h>
#include <mpc_controller.h>
#include <plot_utils.h>
#include <vehicle_simulator.h>
#include <eigen3/Eigen/Geometry>
#include <fstream>
#include <json.hpp>
#include <utility>

int main(int argc, char const *argv[]) {
  //! Read in the params from file:
  std::ifstream params_file(
      "/home/savio/Documents/Projects/vehicle-mpc-controller/params/"
      "mpc_bicycle_lane_change.json");

  nlohmann::json json_obj;
  params_file >> json_obj;

  double sim_start_time = json_obj["sim"]["start_time"];
  double sim_end_time = json_obj["sim"]["end_time"];
  double sample_time = json_obj["sim"]["sample_time"];

  spdlog::info("Sim params loaded");

  double wheelbase = json_obj["model"]["wheelbase"];
  double steering_limit = json_obj["model"]["steering_limit"];
  double acc_limit = json_obj["model"]["acc_limit"];
  double steering_rate_limit = json_obj["model"]["steering_rate_limit"];
  double acc_rate_limit = json_obj["model"]["acc_rate_limit"];

  auto process_noise_vars = json_obj["model"]["process_noise_variances"];
  Eigen::VectorXd variances(4);
  size_t idx = 0;
  for (const double var : process_noise_vars) {
    variances(idx++) = var;
  }

  spdlog::info("Bicycle Model params loaded");

  double prediction_horizon = json_obj["mpc"]["prediction_horizon"];  // secs

  //! Create a Bicycle model:
  std::shared_ptr<VehicleModel> bicycle_model =
      std::make_shared<BicycleModel>("bicycle_model", wheelbase);
  Eigen::VectorXd lower_mag(2), upper_mag(2), rate_limit(2);
  lower_mag << -acc_limit, -steering_limit;
  upper_mag << acc_limit, steering_limit;
  rate_limit << acc_rate_limit, steering_rate_limit;

  bicycle_model->setInputMagConstraints(lower_mag, upper_mag);
  bicycle_model->setInputRateConstraints(rate_limit);
  bicycle_model->setProcessNoiseCovariance(variances.asDiagonal());

  //! Create two simulators:
  //! 1. One to simulate the reference trajectory
  //!            (Ideally the reference trajectory should come from a higher
  //!            level planner)
  //! 2. Second to simulate the system with process noise and the mpc controller
  //! inputs
  VehicleSimulator ref_sim(bicycle_model, sim_start_time, sim_end_time,
                           sample_time);
  VehicleSimulator mpc_sim(bicycle_model, sim_start_time, sim_end_time,
                           sample_time);

  // Run the simulation to get the reference trajectory:
  clock_t startTime, endTime;
  startTime = clock();
  ref_sim.simLaneChange();
  endTime = clock();

  spdlog::info("Time to run the double lane change simulation is {} secs",
               (static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC));

  //! Create the MPC controller:
  size_t state_dim = bicycle_model->getStateDim();
  size_t input_dim = bicycle_model->getInputDim();

  //! Cost matrices:
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd Q_terminal = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(input_dim, input_dim);

  Q.diagonal() << json_obj["mpc"]["Q"]["x"], json_obj["mpc"]["Q"]["y"],
      json_obj["mpc"]["Q"]["theta"], json_obj["mpc"]["Q"]["vel"];
  Q_terminal = Q;
  R.diagonal() << json_obj["mpc"]["R"]["acc"], json_obj["mpc"]["R"]["steering"];

  MPCController mpc_cont(bicycle_model, Q, Q_terminal, R, sample_time,
                         prediction_horizon);

  // Set the reference trajectory obtained from the simulator:
  startTime = clock();
  mpc_cont.setReferenceTrajectory(ref_sim.getSystemTrajectory());
  endTime = clock();

  spdlog::info(
      "Time to set ref trajectory and compute linearized systems is {} secs ",
      (static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC));

  //! Set the initial state of the mpc controller and the simulator::
  Eigen::VectorXd init_state(4);
  init_state << json_obj["mpc"]["init_state"]["x"],
      json_obj["mpc"]["init_state"]["y"],
      json_obj["mpc"]["init_state"]["theta"],
      json_obj["mpc"]["init_state"]["vel"];
  mpc_sim.reset();
  mpc_sim.setCurrentState(init_state);
  mpc_cont.setCurrentState(mpc_sim.getCurrentState());

  //! Initialize the mpc controller (Computes the matrices for the QP problem)
  startTime = clock();
  if (!mpc_cont.init()) {
    spdlog::error(" Failed to initialize mpc controller");
  }
  endTime = clock();

  spdlog::info("Time to initialize the mpc controller is {} secs ",
               (static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC));

  //! Run the MPC controller in the main simulation loop:
  size_t idx_end = (sim_end_time - sim_start_time) / sample_time -
                   (prediction_horizon) / sample_time;
  double max_iter_time = std::numeric_limits<double>::min();

  for (size_t idx = 1; idx < idx_end; idx++) {
    //! Solve the QP optimization problem at the current time step:
    Eigen::VectorXd mpc_solution;
    startTime = clock();
    if (!mpc_cont.solveAndStep(mpc_solution)) {
      spdlog::warn(" Failed to solve mpc optimization at time step {}", idx);
    }

    //! Apply the first mpc input to the simulation system:
    mpc_sim.step(mpc_solution);

    //! Update the MPC controller:
    mpc_cont.setCurrentState(mpc_sim.getCurrentState());

    if (!mpc_cont.updateConstraints()) {
      spdlog::warn(" Failed to update constraints at time step {} ", idx);
    };

    endTime = clock();

    double curr_iter_time =
        (static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC);
    if (max_iter_time < curr_iter_time) {
      max_iter_time = curr_iter_time;
    }

    spdlog::debug("MPC Iteration # {}  --> Time is {} seconds", idx,
                  curr_iter_time);
  }

  spdlog::info(" Maximum iteration time is {} secs ", max_iter_time);

  //! Plot the results:
  plot_utils::plotMPCStates(ref_sim.getSystemTrajectory(),
                            mpc_sim.getSystemTrajectory(), idx_end);

  plot_utils::plotMPCInputs(ref_sim.getSystemTrajectory(),
                            mpc_sim.getSystemTrajectory(), idx_end);

  return 0;
}