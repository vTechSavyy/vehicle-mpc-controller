#include <vehicle_model.h>

VehicleModel::VehicleModel(const std::string& name, int dim_state, int dim_input) :name_(name) , dim_state_(dim_state) , dim_input_(dim_input) {
    
    nRKSteps_disc_ = 4;

    input_mag_constraints_.first.resize(dim_input_);
    input_mag_constraints_.second.resize(dim_input_);

    input_rate_constraints_.first.resize(dim_input_);
    input_rate_constraints_.second.resize(dim_input_);
}

VehicleModel::~VehicleModel() {

}

void VehicleModel::getLinearizedDynamicsDiscreteRK4(Eigen::MatrixXd& A, Eigen::MatrixXd& B, const state_t& x, const input_t& u, double sample_period) {

    state_t x_dot(x.size()) , dx(x.size());
    Eigen::MatrixXd F(dim_state_, dim_state_), G(dim_state_ , dim_input_);
    Eigen::MatrixXd dA1(dim_state_, dim_state_) , dA2(dim_state_, dim_state_) , dA3(dim_state_, dim_state_) , dA4(dim_state_, dim_state_);
    Eigen::MatrixXd dB1(dim_state_, dim_input_) , dB2(dim_state_, dim_input_) , dB3(dim_state_, dim_input_) , dB4(dim_state_, dim_input_); 

    A.setIdentity();
    B.setZero();

    double rk_delta_t = sample_period/nRKSteps_disc_;

    for (int step_num = 0; step_num < nRKSteps_disc_; step_num++) {

        // Step 1 of the RK4 integration:
        dynamics_function_(x, x_dot, u);
        getLinearizedDynamicsContinuous(F, G, x, u);
        dx  = x_dot*rk_delta_t;
        dA1 = F*A*rk_delta_t;
        dB1 = (F*B + G)*rk_delta_t;

        dynamics_function_(x + 0.5*dx, x_dot, u);
        getLinearizedDynamicsContinuous(F, G, x + 0.5*dx, u);
        dx  = x_dot*rk_delta_t;
        dA2 = F*(A + 0.5*dA1)*rk_delta_t;
        dB2 = (F*(B + 0.5*dB1) + G)*rk_delta_t;

        // Step 3 of RK4 integration:
        dynamics_function_(x+0.5*dx , x_dot, u);
        getLinearizedDynamicsContinuous(F, G, x + 0.5*dx, u);
        dx  = x_dot*rk_delta_t;
        dA3 = F*(A + 0.5*dA2)*rk_delta_t;
        dB3 = (F*(B + 0.5*dB2) + G)*rk_delta_t;

        // Step 4 of RK4 integration:
        dynamics_function_(x+0.5*dx , x_dot, u);
        getLinearizedDynamicsContinuous(F, G, x + 0.5*dx, u);
        dx  = x_dot*rk_delta_t;
        dA4 = F*(A + 0.5*dA3)*rk_delta_t;
        dB4 = (F*(B + 0.5*dB3) + G)*rk_delta_t;


        // RK4 formula:
        A += ( dA1 + 2*(dA2 + dA3) + dA4)/6;
        B += ( dB1 + 2*(dB2 + dB3) + dB4)/6;

    }
}