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

# Format primitive tests (split from test_primitives)
add_executable(test_prim_format test_prim_format.cpp)
target_link_libraries(test_prim_format apl_vm GTest::gtest)
add_test(NAME FormatTest COMMAND test_prim_format)

# Arithmetic primitive tests (split from test_primitives)
add_executable(test_prim_arithmetic test_prim_arithmetic.cpp)
target_link_libraries(test_prim_arithmetic apl_vm GTest::gtest)
add_test(NAME ArithmeticTest COMMAND test_prim_arithmetic)

# Structural primitive tests (split from test_primitives)
add_executable(test_prim_structural test_prim_structural.cpp)
target_link_libraries(test_prim_structural apl_vm GTest::gtest)
add_test(NAME StructuralTest COMMAND test_prim_structural)

# Parser tests (syntax, structure, continuation types)
add_executable(test_parser test_parser.cpp)
target_link_libraries(test_parser apl_vm GTest::gtest)
add_test(NAME ParserTest COMMAND test_parser)

# Eval tests (expression evaluation, primitives via eval pipeline)
add_executable(test_eval test_eval.cpp)
target_link_libraries(test_eval apl_vm GTest::gtest)
add_test(NAME EvalTest COMMAND test_eval)

# Dfn tests (user-defined functions)
add_executable(test_dfn test_dfn.cpp)
target_link_libraries(test_dfn apl_vm GTest::gtest)
add_test(NAME DfnTest COMMAND test_dfn)

# Statement tests (Phase 3.3)
add_executable(test_statements test_statements.cpp)
target_link_libraries(test_statements apl_vm GTest::gtest)
add_test(NAME StatementTest COMMAND test_statements)

# Operator tests (Phase 5)
add_executable(test_operators test_operators.cpp)
target_link_libraries(test_operators apl_vm GTest::gtest)
add_test(NAME OperatorsTest COMMAND test_operators)

# String tests (STRING type, char vectors, UTF-8)
add_executable(test_strings test_strings.cpp)
target_link_libraries(test_strings apl_vm GTest::gtest)
add_test(NAME StringTest COMMAND test_strings)
