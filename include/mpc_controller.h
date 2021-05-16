#pragma once

#include <vehicle_model.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>
#include <OsqpEigen/Solver.hpp>

#include <trajectory.h>
#include <utils.h>

class MPCController
{

public:

    MPCController(const std::shared_ptr<VehicleModel>& model, Eigen::MatrixXd& Q, Eigen::MatrixXd& Q_terminal, Eigen::MatrixXd& R, double Ts, double prediction_horizon_time);

    void setReferenceTrajectory(const sp_to_traj& ref_trajectory);

    void setCurrentState(const Eigen::VectorXd& current_vehicle_state) {
        current_state_ = current_vehicle_state - ref_trajectory_->getStateAt(current_time_step_);
    }

    const Eigen::VectorXd& getCurrentState() const {
        return current_state_;
    }

    void step() {
        current_time_step_++;
    }

    void computeHessianAndGradient();

    void computeConstraintMatrix();

    void computeConstraintVector();

    bool updateConstraints();
    
    bool init();

    bool solveAndStep(Eigen::VectorXd& solution);

    const Eigen::SparseMatrix<double>& getHessian() {
        return hessian_;
    }

    const Eigen::SparseMatrix<double>& getConstraintMat() {
        return constraint_mat_;
    }

    const Eigen::VectorXd& getLowerBound() {
        return lower_bound_;
    }

    const Eigen::VectorXd& getUpperBound() {
        return upper_bound_;
    }

    const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd> >& getLinearizedSystems() {
        return linearized_systems_;
    }

    void setLinearizedSystems(std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd> >&& linearized_systems) {

        linearized_systems_ = linearized_systems;
    }

private:

    //! Model of the vehicle:
    std::shared_ptr<VehicleModel> model_; 

    //! Dimensions of the model state and input: 
    size_t model_state_dim_;
    size_t model_input_dim_;

    //! Sample time: 
    double sample_time_;

    //! Prediction and control horizons: 
    double prediction_horizon_time_; 
    size_t prediction_horizon_steps_;

    //! Current discrete time step:
    size_t current_time_step_;

    //! Current state of the model (From the simulation or the localization module)
    Eigen::VectorXd current_state_;


    Eigen::VectorXd prev_input_;

    //! Tuning matrices: 
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd Q_terminal_;
    Eigen::MatrixXd R_;

    //! Hessian and Gradient matrices: 
    Eigen::SparseMatrix<double> hessian_;
    Eigen::VectorXd gradient_;

    //! Constraint matrix: 
    Eigen::SparseMatrix<double> constraint_mat_;

    //! Constraint vectors:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;

    //! Number of variables during each optimization problem:
    size_t num_opt_vars_;

    //! Number of constraints during each optimization problem:
    size_t num_opt_contrs_;

    //! OSQP solver: 
    std::unique_ptr<OsqpEigen::Solver> solver_;

    //! Shared pointer to Reference trajectory: (Got either from simulator or planner)
    sp_to_traj ref_trajectory_; 

    //! Linearized systems: 
    std::vector<std::pair< Eigen::MatrixXd, Eigen::MatrixXd> > linearized_systems_;

};
