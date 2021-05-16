#include <gtest/gtest.h>
#include <bicycle_model.h>
#include <mpc_controller.h>
#include <plot_utils.h>
#include <vehicle_simulator.h>
#include <eigen3/Eigen/Geometry>
#include <fstream>
#include <json.hpp>
#include <utility>

TEST(MPCControllerTest, GradientHessianTest) {
  std::shared_ptr<VehicleModel> bicycle_model =
      std::make_shared<BicycleModel>("rw_bicycle", 3.5);

  size_t state_dim = bicycle_model->getStateDim();
  size_t input_dim = bicycle_model->getInputDim();

  //! Cost matrices:
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd Q_terminal = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(input_dim, input_dim);

  Q.diagonal() << 5, 4, 3, 2;

  Q_terminal.diagonal() << 1, 2, 3, 4;

  R.diagonal() << 7, 8;

  // std::cout<<" Q matrix is : \n "<<Q<<std::endl;
  // std::cout<<" Q term matrix is : \n "<<Q_terminal<<std::endl;
  // std::cout<<" R matrix is : \n "<<R<<std::endl;

  MPCController mpc_rw(bicycle_model, Q, Q_terminal, R, 0.1, 0.3);

  mpc_rw.computeHessianAndGradient();
}

TEST(MPCControllerTest, ConstraintMatrixAndVectorTest) {
  double wheelbase = 3.5;

  std::shared_ptr<VehicleModel> bicycle_model =
      std::make_shared<BicycleModel>("bicycle_model", wheelbase);

  Eigen::VectorXd lower_mag(2), upper_mag(2), rate_limit(2);
  lower_mag << 1, 2;
  upper_mag << 3, 4;
  rate_limit << 5, 6;

  bicycle_model->setInputMagConstraints(lower_mag, upper_mag);
  bicycle_model->setInputRateConstraints(rate_limit);

  // Create the MPC controller:
  size_t state_dim = bicycle_model->getStateDim();
  size_t input_dim = bicycle_model->getInputDim();
  double prediction_horizon = 0.3;  // secs
  double sample_time = 0.1;

  //! Cost matrices:
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd Q_terminal = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(input_dim, input_dim);

  Q.diagonal() << 0.5, 0.5, 0.25, 0.25;
  Q_terminal.diagonal() << 1, 1, 0.5, 2;
  R.diagonal() << 0.2, 0.5;
  MPCController mpc_rw(bicycle_model, Q, Q_terminal, R, sample_time,
                       prediction_horizon);

  // Fake the linearized systems matrices:
  Eigen::MatrixXd A(state_dim, state_dim);
  Eigen::MatrixXd B(state_dim, input_dim);

  A.setZero();
  B.setZero();

  A.diagonal() << 2, 2, 2, 2;
  B << 3, 3, 4, 4, 5, 5, 6, 6;
  {
    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd> > linear_systems;
    for (int i = 0; i < 3; i++) {
      linear_systems.push_back(std::make_pair(A, B));
    }

    mpc_rw.setLinearizedSystems(std::move(linear_systems));
  }

  // Compute the constraint matrix:
  mpc_rw.computeConstraintMatrix();

  // Display the constraint matrix:
//   std::cout << " Constraint matrix is : \n " << mpc_rw.getConstraintMat()
//             << std::endl;

  //! Set a dummy reference trajectory:
  sp_to_traj ref_traj =
      std::make_shared<Trajectory>(state_dim, input_dim, 0.0, 0.5, 0.1);
  mpc_rw.setReferenceTrajectory(ref_traj);

//   spdlog::info(" Set reference trajectory");

  //! Set the current state:
  Eigen::VectorXd current_state(state_dim);
  current_state << 7, 7, 7, 7;
  mpc_rw.setCurrentState(current_state);

  // Compute the constraint matrix:
  mpc_rw.computeConstraintVector();

  // Display the constraint matrix:
//   std::cout << " Lower bound is : \n " << mpc_rw.getLowerBound() << std::endl;
//   std::cout << " Upper bound is : \n " << mpc_rw.getUpperBound() << std::endl;

}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  clock_t startTime, endTime;

  startTime = clock();
  bool outcome = RUN_ALL_TESTS();
  endTime = clock();

  std::cerr << "Total time "
            << (static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC)
            << " seconds." << std::endl;

  return 0;
}