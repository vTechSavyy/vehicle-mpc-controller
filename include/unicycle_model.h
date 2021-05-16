#ifndef UNICYCLE_MODEL_H_
#define UNICYCLE_MODEL_H_

#include <vehicle_model.h>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <string>

class UnicycleModel : public VehicleModel {
 public:
  UnicycleModel(const std::string& model_name);

  void getLinearizedDynamicsContinuous(Eigen::MatrixXd& F, Eigen::MatrixXd& G,
                                       const state_t& x,
                                       const input_t& u) override;
};

#endif