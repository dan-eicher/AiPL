# Test configuration

# Sanity test to verify test harness works
add_executable(test_sanity test_sanity.cpp)
target_link_libraries(test_sanity GTest::gtest)
add_test(NAME SanityTest COMMAND test_sanity)

# Lexer tests
add_executable(test_lexer
    test_lexer.cpp
    ${CMAKE_SOURCE_DIR}/src/token.cpp
    ${CMAKE_SOURCE_DIR}/src/lexer.cpp
)
target_link_libraries(test_lexer GTest::gtest)
add_test(NAME LexerTest COMMAND test_lexer)

# Value system tests
add_executable(test_value
    test_value.cpp
    ${CMAKE_SOURCE_DIR}/src/value.cpp
)
target_link_libraries(test_value GTest::gtest Eigen3::Eigen)
add_test(NAME ValueTest COMMAND test_value)

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
