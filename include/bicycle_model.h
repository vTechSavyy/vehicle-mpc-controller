#ifndef BICYCLE_MODEL_H_
#define BICYCLE_MODEL_H_

#include <vehicle_model.h>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <string>

class BicycleModel : public VehicleModel {
 public:
  BicycleModel(const std::string& model_name, double wheelbase);

  void getLinearizedDynamicsContinuous(Eigen::MatrixXd& F, Eigen::MatrixXd& G,
                                       const state_t& x,
                                       const input_t& u) override;

 private:
  //! Params:
  double wheelbase_;
};

#endif