cmake_minimum_required(VERSION 3.0)
project(mpc)

set(CMAKE_BUILD_TYPE Release)
add_definitions(-std=c++11 -Wno-sign-compare -Wno-unused-value)

# Eigen:
find_package(Eigen3 REQUIRED)

# OSQP-Eigen package:
find_package(OsqpEigen REQUIRED)

# The matplotlib-cpp headers are linked against
find_package(PythonLibs 2.7)


include_directories(
  include 
  third_party/spdlog/include
  third_party/jsoncpp/include
  third_party/matplotlibcpp/include
  ${EIGEN3_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS})

message(${PYTHONLIBS_VERSION_STRING})

#############
# LIBRARIES #
#############
set(HEADERS
    include/vehicle_model.h
    include/unicycle_model.h
    include/bicycle_model.h
    include/utils.h
    include/plot_utils.h
    include/trajectory.h
    include/vehicle_simulator.h
    include/mpc_controller.h
  )

set(SOURCES
    src/vehicle_model.cpp
    src/unicycle_model.cpp
    src/bicycle_model.cpp
    src/vehicle_simulator.cpp
    src/utils.cpp
    src/trajectory.cpp
    src/mpc_controller.cpp
  )

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} OsqpEigen::OsqpEigen osqp::osqp ${PYTHON_LIBRARIES})

###############
# EXECUTABLES #
###############

# Examples: 
add_executable(mpc-bicycle-double-lane-change examples/mpc_bicycle_double_lane_change.cpp)
target_link_libraries(mpc-bicycle-double-lane-change PRIVATE ${PROJECT_NAME})

add_executable(mpc-bicycle-lane-change examples/mpc_bicycle_lane_change.cpp)
target_link_libraries(mpc-bicycle-lane-change PRIVATE ${PROJECT_NAME})

###############
# TESTING #
###############
option(BUILD_TESTS " Should the tests be built by CMake" ON)
if(BUILD_TESTS)
add_subdirectory(tests)
endif()

##########
# EXPORT #
##########
