include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(projected_descent_tests
  test_box_constrained_qp.cpp
)

target_link_libraries(projected_descent_tests PRIVATE
  projected_descent
  GTest::gtest
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(projected_descent_tests)