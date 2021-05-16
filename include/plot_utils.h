#include <matplotlibcpp.h>
#include <trajectory.h>
#include <utils.h>
#include <eigen3/Eigen/Dense>
#include <memory>

namespace plt = matplotlibcpp;

namespace plot_utils {

void plotMPCStates(const sp_to_traj& ref, const sp_to_traj& mpc, size_t num_steps, const std::string& params_file= "") {

  Eigen::ArrayXd t_ref = ref->getTimeHistory().head(num_steps);
  Eigen::ArrayXd x_ref = ref->getStateHistory(0).head(num_steps);
  Eigen::ArrayXd y_ref = ref->getStateHistory(1).head(num_steps);
  Eigen::ArrayXd theta_ref = ref->getStateHistory(2).head(num_steps);
  Eigen::ArrayXd vel_ref = ref->getStateHistory(3).head(num_steps);

  Eigen::ArrayXd x_mpc = mpc->getStateHistory(0).head(num_steps);
  Eigen::ArrayXd y_mpc = mpc->getStateHistory(1).head(num_steps);
  Eigen::ArrayXd theta_mpc = mpc->getStateHistory(2).head(num_steps);
  Eigen::ArrayXd vel_mpc = mpc->getStateHistory(3).head(num_steps);

  Eigen::ArrayXd squared_dist = (x_mpc - x_ref) * (x_mpc - x_ref) + (y_mpc - y_ref) * (y_mpc - y_ref);
  double pos_rmse = std::sqrt(squared_dist.mean()); 
  double theta_rmse = utils::computeRMSE(theta_mpc, theta_ref);
  double vel_rmse = utils::computeRMSE(vel_mpc, vel_ref);  


  std::string title_str; 

  title_str += " MPC State plots --- RMSE: Position = "; 
  title_str += std::to_string(pos_rmse);
  title_str += " m.   Heading = "; 
  title_str += std::to_string(theta_rmse);
  title_str += " rad.  Velocity = "; 
  title_str += std::to_string(vel_rmse);
  title_str += " m/s "; 

  plt::subplot(3, 1, 1);
  plt::plot(x_ref, y_ref, "b", {{"label", " Reference"}});
  plt::plot(x_mpc, y_mpc, "g--", {{"label", " MPC"}});
  plt::xlabel(" X (meters)");
  plt::ylabel(" Y (meters)");
  plt::grid();
  plt::title(title_str);
  plt::legend();

  plt::subplot(3, 1, 2);
  plt::plot(t_ref, theta_ref, "b", {{"label", " Reference"}});
  plt::plot(t_ref, theta_mpc, "g--", {{"label", " MPC"}});
  plt::xlabel(" Time (sec)");
  plt::ylabel(" Heading (rad)");
  plt::grid();

  plt::subplot(3, 1, 3);
  plt::plot(t_ref, vel_ref, "b", {{"label", " Reference"}});
  plt::plot(t_ref, vel_mpc, "g--", {{"label", " MPC"}});
  plt::xlabel(" Time (sec)");
  plt::ylabel(" Velocity (m/s)");
  plt::grid();

  plt::show();
}

void plotMPCInputs(const sp_to_traj& ref, const sp_to_traj& mpc, size_t num_steps, const std::string& params_file = "") {

  Eigen::ArrayXd t_ref = ref->getTimeHistory().head(num_steps);
  double sample_time = t_ref(1) - t_ref(0);

  Eigen::ArrayXd acc_ref = ref->getInputHistory(0).head(num_steps);
  Eigen::ArrayXd steering_ref = ref->getInputHistory(1).head(num_steps);

  Eigen::ArrayXd acc_mpc = mpc->getInputHistory(0).head(num_steps);
  Eigen::ArrayXd steering_mpc = mpc->getInputHistory(1).head(num_steps);

  Eigen::ArrayXd acc_rates(num_steps - 1);
  for (size_t i = 0; i < num_steps - 1; i++) {
    acc_rates(i) = (std::abs(acc_mpc(i+1) - acc_mpc(i)))/sample_time;
  }

  Eigen::ArrayXd steering_rates(num_steps - 1);
  for (size_t i = 0; i < num_steps - 1; i++) {
    steering_rates(i) = (std::abs(steering_mpc(i+1) - steering_mpc(i)))/sample_time;
  }

  plt::subplot(4, 1, 1);
  plt::plot(t_ref, acc_ref, "b", {{"label", " Reference"}});
  plt::plot(t_ref, acc_mpc, "g--", {{"label", " MPC"}});
  plt::xlabel(" Time (sec)");
  plt::ylabel(" Acceleration (m/s^2)");
  plt::legend();
  plt::grid();

  plt::subplot(4, 1, 2);
  plt::plot(t_ref, steering_ref, "b", {{"label", " Reference"}});
  plt::plot(t_ref, steering_mpc, "g--", {{"label", " MPC"}});
  plt::xlabel(" Time (sec)");
  plt::ylabel(" Steering (rad)");
  plt::grid();

  plt::subplot(4, 1, 3);
  plt::plot(t_ref.head(num_steps - 1), acc_rates);
  plt::xlabel(" Time (sec)");
  plt::ylabel(" Jerk (m/s^3)");
  plt::grid();

  plt::subplot(4, 1, 4);
  plt::plot(t_ref.head(num_steps - 1), steering_rates);
  plt::xlabel(" Time (sec)");
  plt::ylabel(" Steering rate (rad/s)");
  plt::grid();

  plt::show();
}

void plotTrajectory(const sp_to_traj& system_traj) {

  Eigen::ArrayXd time_arr = system_traj->getTimeHistory();
  Eigen::ArrayXd x = system_traj->getStateHistory(0);
  Eigen::ArrayXd y = system_traj->getStateHistory(1);
  Eigen::ArrayXd theta = system_traj->getStateHistory(2);
  Eigen::ArrayXd vel = system_traj->getStateHistory(3);

  plt::subplot(3, 1, 1);
  plt::plot(x, y, "b*--");
  plt::xlabel(" X (meters)");
  plt::ylabel(" Y (meters)");
  plt::grid();

  plt::subplot(3, 1, 2);
  plt::plot(time_arr, theta);
  plt::xlabel(" Time (sec)");
  plt::ylabel(" Heading (rad)");
  plt::grid();

  plt::subplot(3, 1, 3);
  plt::plot(time_arr, vel);
  plt::xlabel(" Time (sec)");
  plt::ylabel(" Velocity (m/s)");
  plt::grid();

  plt::show();
}
}