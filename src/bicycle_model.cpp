#include <bicycle_model.h>

BicycleModel::BicycleModel(const std::string &model_name, double wheelbase) : VehicleModel(model_name, 4, 2), wheelbase_(wheelbase)
{

    dynamics_function_ = [this](const state_t &x, state_t &dxdt, const input_t &u) {
        dxdt[0] = x[3] * std::cos(x[2]);
        dxdt[1] = x[3] * std::sin(x[2]);
        dxdt[2] = x[3] * (std::tan(u[1]) / wheelbase_);
        dxdt[3] = u[0];
    };

    dynamics_function_sim_ = [this](const std::vector<double> &x, std::vector<double>& dxdt, const input_t& u) {

        dxdt[0] = x[3] * std::cos(x[2]);
        dxdt[1] = x[3] * std::sin(x[2]);
        dxdt[2] = x[3] * (std::tan(u[1]) / wheelbase_);
        dxdt[3] = u[0];
    };

}

void BicycleModel::getLinearizedDynamicsContinuous(Eigen::MatrixXd& F, Eigen::MatrixXd& G, const state_t& x, const input_t& u) {

    // Check for correct size of the matrices:    
    // State transition matrix (A)
    F(0, 0) = 0;     F(0, 1) = 0;     F(0,2) = x[3]*(-1)*std::sin(x[2]);    F(0,3) = std::cos(x[2]);

    F(1, 0) = 0;     F(1, 1) = 0;     F(1,2) = x[3]*std::cos(x[2]);         F(1,3) = std::sin(x[2]);

    F(2, 0) = 0;     F(2, 1) = 0;     F(2,2) = 0;                           F(2,3) = std::tan(u[1]) / wheelbase_;

    F(3, 0) = 0;     F(3, 1) = 0;     F(3,2) = 0;                           F(3,3) = 0;

    // Input to state mapping matrix (B)
    G(0,0) = 0;            G(0,1) = 0;

    G(1,0) = 0;            G(1,1) = 0;

    G(2,0) = 0;            G(2,1) = (x[3]/wheelbase_)*(1/ std::pow(std::cos(u[1]),2));

    G(3,0) = 1;            G(3,1) = 0;

}

