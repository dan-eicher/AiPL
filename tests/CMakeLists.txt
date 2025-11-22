# Test configuration
# All tests link against the apl_vm library (built once in parent CMakeLists.txt)

# Sanity test to verify test harness works
add_executable(test_sanity test_sanity.cpp)
target_link_libraries(test_sanity GTest::gtest)
add_test(NAME SanityTest COMMAND test_sanity)

# Lexer tests
add_executable(test_lexer test_lexer.cpp)
target_link_libraries(test_lexer apl_vm GTest::gtest)
add_test(NAME LexerTest COMMAND test_lexer)

# Value system tests
add_executable(test_value test_value.cpp)
target_link_libraries(test_value apl_vm GTest::gtest)
add_test(NAME ValueTest COMMAND test_value)

# Completion records tests
add_executable(test_completion test_completion.cpp)
target_link_libraries(test_completion apl_vm GTest::gtest)
add_test(NAME CompletionTest COMMAND test_completion)

# Continuation tests
add_executable(test_continuation test_continuation.cpp)
target_link_libraries(test_continuation apl_vm GTest::gtest)
add_test(NAME ContinuationTest COMMAND test_continuation)

# Heap/GC tests
add_executable(test_heap test_heap.cpp)
target_link_libraries(test_heap apl_vm GTest::gtest)
add_test(NAME HeapTest COMMAND test_heap)

# Machine execution tests
add_executable(test_machine test_machine.cpp)
target_link_libraries(test_machine apl_vm GTest::gtest)
add_test(NAME MachineTest COMMAND test_machine)

# Primitives tests
add_executable(test_primitives test_primitives.cpp)
target_link_libraries(test_primitives apl_vm GTest::gtest)
add_test(NAME PrimitivesTest COMMAND test_primitives)
