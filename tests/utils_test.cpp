#include <gtest/gtest.h>
#include <utils.h>

TEST(UtilsTest, SetTripletsFromMatrixTest) {
  Eigen::MatrixXd sub_matrix = Eigen::MatrixXd::Identity(3, 3);
  size_t start_row = 5;
  size_t start_col = 4;

  std::vector<Triplet> triplet_list;

  utils::setTripletsFromMatrix(sub_matrix, start_row, start_col, triplet_list,
                               true);

  size_t count = 2;
  for (auto it = triplet_list.begin(); it != triplet_list.end(); it++) {
    ASSERT_EQ(start_row + count, it->row());
    ASSERT_EQ(start_col + count, it->col());
    ASSERT_EQ(1.0, it->value());
    count--;
  }

  sub_matrix << 0, 1, 2, 3, 4, 5, 6, 7, 8;

  start_row = 10;
  start_col = 12;

  // std::cout<<" Basic sub  matrix is : "<<sub_matrix<<std::endl;

  utils::setTripletsFromMatrix(sub_matrix, start_row, start_col, triplet_list);

  for (auto it = triplet_list.begin(); it != triplet_list.end(); it++) {
    std::cout << " Row : " << it->row() << " --> Col : " << it->col()
              << " --> Value : " << it->value() << std::endl;
  }
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