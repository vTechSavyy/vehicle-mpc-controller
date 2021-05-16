#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <memory>
#include <random>
#include <vector>

typedef Eigen::Triplet<double> Triplet;

namespace utils {

bool setTripletsFromMatrix(const Eigen::MatrixXd& sub_matrix, size_t start_row,
                           size_t start_col, std::vector<Triplet>& triplet_list,
                           bool isDiagonal = false);

void addAWGN(Eigen::VectorXd& data, const Eigen::MatrixXd& cov_matrix,
             std::mt19937& rng);

double shortestAngularDistance(double source_angle, double target_angle);

double computeRMSE(const Eigen::ArrayXd& data, const Eigen::ArrayXd& ref); 

};
