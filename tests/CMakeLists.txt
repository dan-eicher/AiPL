# Test configuration

# Sanity test to verify test harness works
add_executable(test_sanity test_sanity.cpp)
target_link_libraries(test_sanity GTest::gtest)
add_test(NAME SanityTest COMMAND test_sanity)

# Additional test sources (will be added as we implement)
# set(TEST_SOURCES
#     test_lexer.cpp
#     test_value.cpp
#     test_primitives.cpp
#     test_integration.cpp
# )

# Main test executable (will be added later)
# add_executable(run_tests ${TEST_SOURCES})
# target_link_libraries(run_tests
#     GTest::gtest
#     Eigen3::Eigen
#     pthread
# )

# Register tests with CTest
# add_test(NAME AllTests COMMAND run_tests)
