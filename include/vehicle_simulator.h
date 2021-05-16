#pragma once 

#include <boost/numeric/odeint.hpp>
#include <vehicle_model.h>
#include <cmath>
#include <utils.h>
#include <trajectory.h>

class VehicleSimulator {

public:

    using rk4_integrator = boost::numeric::odeint::runge_kutta4<std::vector<double> >;

    VehicleSimulator(const std::shared_ptr<VehicleModel>& model, 
                    double sim_start_time = 0.0,
                    double sim_end_time   = 0.0, 
                    double sample_time    = 0.1);
    /**
     * @brief Clears the system trajectory, resets the current time to start time
     * 
     */
    void reset();

    Eigen::VectorXd getCurrentState() {
        return system_trajectory_->getStateAt(sim_current_step_);
    }

    void setCurrentState(const Eigen::VectorXd& state) {
        system_trajectory_->setStateAt(sim_current_step_, state);
    }

    const sp_to_traj& getSystemTrajectory() {
        return system_trajectory_;
    }

    /**
     * @brief Runs a single forward step of the simulation from the current_time_step
     * 
     * @param input 
     * @return true 
     * @return false 
     */
    bool step(const Eigen::VectorXd& input);

    /**
     * @brief Runs the simulation forward from the current time step to the end:
     * 
     * @return true 
     * @return false 
     */
    bool runSimulation();

    /**
     * @brief Utility function to simulate a lane change
     * @todo Move this outside the simulator class
     * @param max_vel 
     * @param duration 
     * @return true 
     * @return false 
     */
    bool simLaneChange(double max_vel = 30.0, double duration = 2.0); 

    /**
     * @brief Utility function to simulate a double lane change
     * @todo Move this outside the simulator clas
     * @param max_vel 
     * @param duration 
     * @return true 
     * @return false 
     */
    bool simDoubleLaneChange(double max_vel = 30.0, double duration = 1.5); 

private:

    //! Numerical ODE integrator to generate forward dynamics: 
    std::unique_ptr<rk4_integrator> integrator_;

    //! Model of the vehicle:
    std::shared_ptr<VehicleModel> model_;

    //! Reference trajectory of states and inputs along with time vector:
    sp_to_traj system_trajectory_;

    //! Start, End and Sample time of the simulation: 
    double sim_start_time_;
    double sim_end_time_;
    double sample_time_;

    //! Number of discrete steps in the simulation: 
    size_t num_steps_;

    //! Number of boost integrator steps per smaple period:
    size_t num_boost_steps_per_sample_;

    //! Current simulation time: 
    double sim_current_time_;  // secs

    //! Current step in the simulation: 
    size_t sim_current_step_;

    //! Used to add process noise to the system state during the simualtion
    std::random_device rd_;
    std::mt19937 rng_;
    
};

