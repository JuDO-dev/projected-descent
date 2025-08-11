add_executable(projected_descent_tests
    test_box_constrained_qp.cpp
)

target_link_libraries(projected_descent_tests
    PRIVATE
    projected_descent
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(projected_descent_tests)