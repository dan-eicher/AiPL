// Statement tests - Phase 3.3

#include <gtest/gtest.h>
#include "parser.h"
#include "machine.h"
#include "continuation.h"
#include "environment.h"

using namespace apl;

class StatementTest : public ::testing::Test {
protected:
    Machine* machine;
    Parser* parser;

    void SetUp() override {
        machine = new Machine();
        init_global_environment(machine->env);  // Initialize built-in operators
        parser = new Parser(machine);
    }

    void TearDown() override {
        delete parser;
        delete machine;
    }

    // Helper: evaluate parsed continuation using the CEK machine
    Value* eval(Continuation* k) {
        machine->push_kont(k);
        Value* result = machine->execute();
        return result;
    }
};

// ============================================================================
// Multi-Statement Sequence Tests
// ============================================================================

// Test empty program
TEST_F(StatementTest, EmptyProgram) {
    Continuation* k = parser->parse_program("");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    // Empty program returns 0
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test single statement program
TEST_F(StatementTest, SingleStatement) {
    Continuation* k = parser->parse_program("42");

    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test two statements separated by newline
TEST_F(StatementTest, TwoStatementsNewline) {
    Continuation* k = parser->parse_program("x ← 10\nx");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);

    // Result should be the last statement (x)
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);

    // Verify x was assigned
    Value* x = machine->env->lookup("x");
    ASSERT_NE(x, nullptr);
    EXPECT_DOUBLE_EQ(x->as_scalar(), 10.0);
}

// Test two statements separated by diamond
TEST_F(StatementTest, TwoStatementsDiamond) {
    Continuation* k = parser->parse_program("x ← 5 ⋄ x + 3");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);

    // Result should be the last statement (x + 3 = 8)
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// Test multiple statements with mixed separators
TEST_F(StatementTest, MultipleStatementsMixed) {
    Continuation* k = parser->parse_program("a ← 1\nb ← 2 ⋄ c ← 3\na + b + c");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);

    // Result should be 1 + 2 + 3 = 6
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// Test statements with leading/trailing separators
TEST_F(StatementTest, LeadingTrailingSeparators) {
    Continuation* k = parser->parse_program("\n\n42\n\n");

    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test multiple assignments
TEST_F(StatementTest, MultipleAssignments) {
    Continuation* k = parser->parse_program("x ← 10\ny ← 20\nz ← x + y\nz");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 30.0);

    // Verify all variables were assigned
    EXPECT_DOUBLE_EQ(machine->env->lookup("x")->as_scalar(), 10.0);
    EXPECT_DOUBLE_EQ(machine->env->lookup("y")->as_scalar(), 20.0);
    EXPECT_DOUBLE_EQ(machine->env->lookup("z")->as_scalar(), 30.0);
}

// Test expressions in sequence
TEST_F(StatementTest, ExpressionSequence) {
    Continuation* k = parser->parse_program("1 + 2\n3 * 4\n5 - 1");

    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    // Result should be the last expression (5 - 1 = 4)
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

// Test sequence with array operations
TEST_F(StatementTest, SequenceWithArrays) {
    Continuation* k = parser->parse_program("vec ← 1 2 3\n⍴ vec");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);

    // Result should be shape of vec: [3]
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
