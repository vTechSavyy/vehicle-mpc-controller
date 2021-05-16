#include <gtest/gtest.h>
#include <bicycle_model.h>
#include <plot_utils.h>
#include <vehicle_simulator.h>

TEST(VehicleSimulatorTest, simLaneChangeTest) {
    
  double sim_start_time = 0.0;
  double sim_end_time = 45.0;
  double sample_time = 0.1;
  double wheelbase = 3.5;

  std::shared_ptr<VehicleModel> bicycle_model =
      std::make_shared<BicycleModel>("bicycle_model", wheelbase);
  VehicleSimulator veh_sim(bicycle_model, sim_start_time, sim_end_time,
                           sample_time);

  // Run the simulation:
  veh_sim.simLaneChange(30, 2.5);

  // Plot the simulated trajectory:
  plot_utils::plotTrajectory(veh_sim.getSystemTrajectory());
}

TEST(VehicleSimulatorTest, simDoubleLaneChangeTest) {

    double sim_start_time = 0.0;
    double sim_end_time = 45;
    double sample_time = 0.1;
    double wheelbase = 3.5;

    std::shared_ptr<VehicleModel> bicycle_model =
    std::make_shared<BicycleModel>("bicycle_model", wheelbase);
    VehicleSimulator veh_sim(bicycle_model, sim_start_time, sim_end_time,
    sample_time);

    // Run the simulation:
    veh_sim.simDoubleLaneChange(30, 2.5);

    // Plot the simulated trajectory:
    plot_utils::plotTrajectory(veh_sim.getSystemTrajectory());

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

  return outcome;
}