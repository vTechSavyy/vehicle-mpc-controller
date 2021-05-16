cmake_minimum_required(VERSION 3.0)

include(googletest.cmake)

fetch_googletest(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/googletest
    )

function(add_new_test classname)
    set(testsrc ${classname}_test.cpp)
    set(testbinary ${classname}_test)
    set(testname   ${classname}_test)
    add_executable(${testbinary} ${testsrc})
    target_link_libraries(${testbinary} PRIVATE mpc gtest_main)
    add_test(NAME ${testname} COMMAND ${testbinary})
endfunction()
    
# Add individual test suites here:
add_new_test(utils)

add_new_test(vehicle_simulator)

add_new_test(trajectory)

add_new_test(mpc_controller)
