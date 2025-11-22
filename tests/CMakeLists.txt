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

# Completion records tests
add_executable(test_completion
    test_completion.cpp
    ${CMAKE_SOURCE_DIR}/src/value.cpp
    ${CMAKE_SOURCE_DIR}/src/control.cpp
    ${CMAKE_SOURCE_DIR}/src/token.cpp
    ${CMAKE_SOURCE_DIR}/src/lexer.cpp
)
target_link_libraries(test_completion GTest::gtest Eigen3::Eigen)
add_test(NAME CompletionTest COMMAND test_completion)

# Continuation tests
add_executable(test_continuation
    test_continuation.cpp
    ${CMAKE_SOURCE_DIR}/src/value.cpp
    ${CMAKE_SOURCE_DIR}/src/control.cpp
    ${CMAKE_SOURCE_DIR}/src/continuation.cpp
    ${CMAKE_SOURCE_DIR}/src/heap.cpp
    ${CMAKE_SOURCE_DIR}/src/environment.cpp
    ${CMAKE_SOURCE_DIR}/src/primitives.cpp
    ${CMAKE_SOURCE_DIR}/src/token.cpp
    ${CMAKE_SOURCE_DIR}/src/lexer.cpp
)
target_link_libraries(test_continuation GTest::gtest Eigen3::Eigen)
add_test(NAME ContinuationTest COMMAND test_continuation)

# Heap/GC tests
add_executable(test_heap
    test_heap.cpp
    ${CMAKE_SOURCE_DIR}/src/value.cpp
    ${CMAKE_SOURCE_DIR}/src/control.cpp
    ${CMAKE_SOURCE_DIR}/src/continuation.cpp
    ${CMAKE_SOURCE_DIR}/src/heap.cpp
    ${CMAKE_SOURCE_DIR}/src/environment.cpp
    ${CMAKE_SOURCE_DIR}/src/primitives.cpp
    ${CMAKE_SOURCE_DIR}/src/token.cpp
    ${CMAKE_SOURCE_DIR}/src/lexer.cpp
)
target_link_libraries(test_heap GTest::gtest Eigen3::Eigen)
add_test(NAME HeapTest COMMAND test_heap)

# Machine execution tests
add_executable(test_machine
    test_machine.cpp
    ${CMAKE_SOURCE_DIR}/src/value.cpp
    ${CMAKE_SOURCE_DIR}/src/control.cpp
    ${CMAKE_SOURCE_DIR}/src/continuation.cpp
    ${CMAKE_SOURCE_DIR}/src/heap.cpp
    ${CMAKE_SOURCE_DIR}/src/environment.cpp
    ${CMAKE_SOURCE_DIR}/src/primitives.cpp
    ${CMAKE_SOURCE_DIR}/src/machine.cpp
    ${CMAKE_SOURCE_DIR}/src/token.cpp
    ${CMAKE_SOURCE_DIR}/src/lexer.cpp
)
target_link_libraries(test_machine GTest::gtest Eigen3::Eigen)
add_test(NAME MachineTest COMMAND test_machine)

# Primitives tests
add_executable(test_primitives
    test_primitives.cpp
    ${CMAKE_SOURCE_DIR}/src/value.cpp
    ${CMAKE_SOURCE_DIR}/src/primitives.cpp
    ${CMAKE_SOURCE_DIR}/src/environment.cpp
    ${CMAKE_SOURCE_DIR}/src/heap.cpp
    ${CMAKE_SOURCE_DIR}/src/control.cpp
    ${CMAKE_SOURCE_DIR}/src/continuation.cpp
    ${CMAKE_SOURCE_DIR}/src/token.cpp
    ${CMAKE_SOURCE_DIR}/src/lexer.cpp
)
target_link_libraries(test_primitives GTest::gtest Eigen3::Eigen)
add_test(NAME PrimitivesTest COMMAND test_primitives)

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
