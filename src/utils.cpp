#include <utils.h>

bool utils::setTripletsFromMatrix(const Eigen::MatrixXd& sub_matrix,
                                      size_t start_row, size_t start_col,
                                      std::vector<Triplet>& triplet_list,
                                      bool isDiagonal) {
  if (isDiagonal) {
    for (size_t i = 0; i < sub_matrix.rows(); i++) {
      triplet_list.emplace_back(start_row + i, start_col + i,
                                 sub_matrix(i, i));
    }

    return true;

  } else {
    for (size_t i = 0; i < sub_matrix.rows(); i++) {
      for (size_t j = 0; j < sub_matrix.cols(); j++) {
        triplet_list.emplace_back(start_row + i, start_col + j,
                                   sub_matrix(i, j));
      }
    }

    return true;
  }
}

void utils::addAWGN(Eigen::VectorXd& data,
                        const Eigen::MatrixXd& cov_matrix, std::mt19937& rng) {
  // Dimensions check:

  // Covariance matrix check:

  std::normal_distribution<double> nd(0, 1);

  Eigen::VectorXd noise(data);
  for (size_t i = 0; i < data.size(); i++) {
    noise[i] = nd(rng);
  }

  Eigen::LLT<Eigen::MatrixXd> cholOfCov(cov_matrix);

  noise = cholOfCov.matrixL() * noise;

  data += noise;

  return;
}

double utils::shortestAngularDistance(double source_angle,
                                          double target_angle) {
  // Wrap the angles between the range [-PI, PI]:
  source_angle = std::remainder(source_angle, 2 * M_PI);
  target_angle = std::remainder(target_angle, 2 * M_PI);

  // Compute the difference:
  double diff = target_angle - source_angle;

  if (diff > M_PI) {
    diff -= 2 * M_PI;
    return std::abs(diff);
  }

  if (diff < -M_PI) {
    diff += 2 * M_PI;
    return std::abs(diff);
  }

  return std::abs(diff);
}

double utils::computeRMSE(const Eigen::ArrayXd& data, const Eigen::ArrayXd& ref) {

  Eigen::ArrayXd squared_error = (data - ref) * (data - ref);

  return std::sqrt(squared_error.mean());
}
