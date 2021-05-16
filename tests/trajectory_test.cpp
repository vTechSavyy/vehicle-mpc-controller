#include <gtest/gtest.h>
#include <trajectory.h>


TEST(TrajectoryTest, setAndgetStateAtTest) {

    Trajectory traj(2, 1, 0.0, 5.0, 0.1);

    Eigen::VectorXd expected_state(2, 1);
    expected_state << 5.5,  9.2;    

    size_t test_step_num = 20;
    traj.setStateAt(test_step_num, expected_state);
    
    Eigen::VectorXd actual_state = traj.getStateAt(test_step_num);

    EXPECT_DOUBLE_EQ(expected_state(0), actual_state(0));
    EXPECT_DOUBLE_EQ(expected_state(1), actual_state(1));

    //! Invalid input: step < 0
    Eigen::VectorXd invalid_state_1 = traj.getStateAt(-2);

    EXPECT_TRUE(std::isnan(invalid_state_1(0)));
    EXPECT_TRUE(std::isnan(invalid_state_1(1)));

    //! Invalid input: step > num_steps
    Eigen::VectorXd invalid_state_2 = traj.getStateAt(51);

    EXPECT_TRUE(std::isnan(invalid_state_2(0)));
    EXPECT_TRUE(std::isnan(invalid_state_2(1)));
}

TEST(TrajectoryTest, setAndgetInputAtTest) {

    Trajectory traj(3, 3, 0.0, 5.0, 0.1);

    size_t test_step_num = 32;
    Eigen::VectorXd expected_input(3, 1);
    expected_input << 2.1,  15.9, 18.75;    

    traj.setInputAt(test_step_num, expected_input);
    
    Eigen::VectorXd actual_input = traj.getInputAt(test_step_num);

    EXPECT_DOUBLE_EQ(expected_input(0), actual_input(0));
    EXPECT_DOUBLE_EQ(expected_input(1), actual_input(1));

    //! Invalid input: step < 0
    Eigen::VectorXd invalid_input_1 = traj.getInputAt(-2);

    EXPECT_TRUE(std::isnan(invalid_input_1(0)));
    EXPECT_TRUE(std::isnan(invalid_input_1(1)));

    //! Invalid input: step > num_steps
    Eigen::VectorXd invalid_input_2 = traj.getInputAt(51);

    EXPECT_TRUE(std::isnan(invalid_input_2(0)));
    EXPECT_TRUE(std::isnan(invalid_input_2(1)));
}

TEST(TrajectoryTest, lerpStateTest) {

    Trajectory traj(3, 2, 0.0, 5.0, 0.1);

    Eigen::VectorXd first_state(3, 1);
    first_state << 17.0,  9.0, 2.0 ;
    Eigen::VectorXd second_state(3, 1);
    first_state << 20.0,  5.0, -1.0;

    size_t test_step = 25;
    traj.setStateAt(test_step, first_state);
    traj.setStateAt(test_step + 1, second_state);

    double test_time = 2.525;
    Eigen::VectorXd lerp_state = traj.lerpState(test_time);

    std::cout<<" Interp vec is: \n "<<lerp_state<<std::endl;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    clock_t startTime, endTime;

    startTime = clock();
    bool outcome = RUN_ALL_TESTS();
    endTime = clock();

    std::cerr << "Total time " << (static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC)
              << " seconds." << std::endl;

    return outcome;
}