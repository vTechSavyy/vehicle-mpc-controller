#include <unicycle_model.h>

UnicycleModel::UnicycleModel(const std::string& model_name)
    : VehicleModel(model_name, 3, 2) {

  dynamics_function_ = [this](const state_t& x, state_t& dxdt,
                              const input_t& u) {
    dxdt[0] = u[0] * std::cos(x[2]);
    dxdt[1] = u[0] * std::sin(x[2]);
    dxdt[2] = u[1];
  };

  dynamics_function_sim_ = [this](const std::vector<double>& x,
                                  std::vector<double>& dxdt, const input_t& u) {
    dxdt[0] = u[0] * std::cos(x[2]);
    dxdt[1] = u[0] * std::sin(x[2]);
    dxdt[2] = u[1];
  };
}

void UnicycleModel::getLinearizedDynamicsContinuous(Eigen::MatrixXd& F,
                                                   Eigen::MatrixXd& G,
                                                   const state_t& x,
                                                   const input_t& u) {
  // Check for correct size of the matrices:
  // State transition matrix (F)
  F(0, 0) = 0;
  F(0, 1) = 0;
  F(0, 2) = u[0] * (-1) * std::sin(x[2]);

  F(1, 0) = 0;
  F(1, 1) = 0;
  F(1, 2) = u[0] * std::cos(x[2]);

  F(2, 0) = 0;
  F(2, 1) = 0;
  F(2, 2) = 0;

  // Input to state mapping matrix (G)
  G(0, 0) = std::cos(x[2]);
  G(0, 1) = 0;

  G(1, 0) = std::sin(x[2]);
  G(1, 1) = 0;

  G(2, 0) = 0;
  G(2, 1) = 1;
}
