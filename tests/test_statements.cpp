// Statement tests - Phase 3.3

#include <gtest/gtest.h>
#include "parser.h"
#include "machine.h"
#include "continuation.h"

using namespace apl;

class StatementTest : public ::testing::Test {
protected:
    Machine* machine;
    Parser* parser;

    void SetUp() override {
        machine = new Machine();
        parser = machine->parser;
    }

    void TearDown() override {
        delete machine;
    }
};

// ============================================================================
// Multi-Statement Sequence Tests
// ============================================================================

// Test empty program
TEST_F(StatementTest, EmptyProgram) {
    Continuation* k = parser->parse("");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    // Empty program returns 0
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test single statement program
TEST_F(StatementTest, SingleStatement) {
    Continuation* k = parser->parse("42");

    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test two statements separated by newline
TEST_F(StatementTest, TwoStatementsNewline) {
    Continuation* k = parser->parse("x ← 10\nx");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

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
    Continuation* k = parser->parse("x ← 5 ⋄ x + 3");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // Result should be the last statement (x + 3 = 8)
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// Test multiple statements with mixed separators
TEST_F(StatementTest, MultipleStatementsMixed) {
    Continuation* k = parser->parse("a ← 1\nb ← 2 ⋄ c ← 3\na + b + c");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // Result should be 1 + 2 + 3 = 6
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// Test statements with leading/trailing separators
TEST_F(StatementTest, LeadingTrailingSeparators) {
    Continuation* k = parser->parse("\n\n42\n\n");

    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test multiple assignments
TEST_F(StatementTest, MultipleAssignments) {
    Continuation* k = parser->parse("x ← 10\ny ← 20\nz ← x + y\nz");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 30.0);

    // Verify all variables were assigned
    EXPECT_DOUBLE_EQ(machine->env->lookup("x")->as_scalar(), 10.0);
    EXPECT_DOUBLE_EQ(machine->env->lookup("y")->as_scalar(), 20.0);
    EXPECT_DOUBLE_EQ(machine->env->lookup("z")->as_scalar(), 30.0);
}

// Test expressions in sequence
TEST_F(StatementTest, ExpressionSequence) {
    Continuation* k = parser->parse("1 + 2\n3 * 4\n5 - 1");

    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    // Result should be the last expression (5 - 1 = 4)
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

// Test sequence with array operations
TEST_F(StatementTest, SequenceWithArrays) {
    Continuation* k = parser->parse("vec ← 1 2 3\n⍴ vec");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // Result should be shape of vec: [3]
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);
}

// ============================================================================
// If/Else Conditional Tests
// ============================================================================

// Test simple if with true condition
TEST_F(StatementTest, IfTrue) {
    Continuation* k = parser->parse(":If 1\nx ← 42\n:EndIf\nx");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test simple if with false condition
TEST_F(StatementTest, IfFalse) {
    Continuation* k = parser->parse("x ← 10\n:If 0\nx ← 42\n:EndIf\nx");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // x should remain 10 since condition was false
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test if-else with true condition
TEST_F(StatementTest, IfElseTrue) {
    Continuation* k = parser->parse(":If 1\nx ← 100\n:Else\nx ← 200\n:EndIf\nx");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 100.0);
}

// Test if-else with false condition
TEST_F(StatementTest, IfElseFalse) {
    Continuation* k = parser->parse(":If 0\nx ← 100\n:Else\nx ← 200\n:EndIf\nx");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 200.0);
}

// Test if with expression condition
TEST_F(StatementTest, IfExpressionCondition) {
    Continuation* k = parser->parse("x ← 5\n:If x - 3\ny ← 10\n:Else\ny ← 20\n:EndIf\ny");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // x - 3 = 2, which is non-zero (true), so y = 10
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test nested if statements
TEST_F(StatementTest, NestedIf) {
    Continuation* k = parser->parse(
        "x ← 5\n"
        ":If x\n"
        "  :If x - 3\n"
        "    y ← 100\n"
        "  :Else\n"
        "    y ← 50\n"
        "  :EndIf\n"
        ":Else\n"
        "  y ← 0\n"
        ":EndIf\n"
        "y"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // x is 5 (true), x - 3 is 2 (true), so y = 100
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 100.0);
}

// Test if without else branch (false condition)
TEST_F(StatementTest, IfWithoutElseFalse) {
    Continuation* k = parser->parse("x ← 5\n:If 0\nx ← 10\n:EndIf\nx");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // x should remain 5
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test multiple statements in if branches
TEST_F(StatementTest, IfMultipleStatements) {
    Continuation* k = parser->parse(
        ":If 1\n"
        "a ← 10\n"
        "b ← 20\n"
        "c ← a + b\n"
        ":EndIf\n"
        "c"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 30.0);
}

// ============================================================================
// While Loop Tests
// ============================================================================

// Test simple while loop with counter
TEST_F(StatementTest, WhileSimple) {
    Continuation* k = parser->parse(
        "i ← 1\n"
        "sum ← 0\n"
        ":While i - 5\n"
        "  sum ← sum + i\n"
        "  i ← i + 1\n"
        ":EndWhile\n"
        "sum"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // When i=1: sum=1, i=2
    // When i=2: sum=3, i=3
    // When i=3: sum=6, i=4
    // When i=4: sum=10, i=5
    // When i=5: condition is 0 (false), exit
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test while loop that never executes (false condition)
TEST_F(StatementTest, WhileNeverExecutes) {
    Continuation* k = parser->parse(
        "x ← 5\n"
        ":While 0\n"
        "  x ← 10\n"
        ":EndWhile\n"
        "x"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // x should remain 5
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test while loop with expression condition
TEST_F(StatementTest, WhileExpressionCondition) {
    Continuation* k = parser->parse(
        "n ← 10\n"
        ":While n\n"
        "  n ← n - 1\n"
        ":EndWhile\n"
        "n"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // n counts down to 0
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test nested while loops
TEST_F(StatementTest, WhileNested) {
    Continuation* k = parser->parse(
        "sum ← 0\n"
        "i ← 0\n"
        ":While i - 3\n"
        "  j ← 0\n"
        "  :While j - 2\n"
        "    sum ← sum + 1\n"
        "    j ← j + 1\n"
        "  :EndWhile\n"
        "  i ← i + 1\n"
        ":EndWhile\n"
        "sum"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // 3 outer iterations × 2 inner iterations = 6
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// Test while loop modifying multiple variables
TEST_F(StatementTest, WhileMultipleVariables) {
    Continuation* k = parser->parse(
        "a ← 1\n"
        "b ← 1\n"
        "n ← 5\n"
        ":While n\n"
        "  c ← a + b\n"
        "  a ← b\n"
        "  b ← c\n"
        "  n ← n - 1\n"
        ":EndWhile\n"
        "b"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // Fibonacci: 1,1,2,3,5,8,13 - after 5 iterations, b = 13
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 13.0);
}

// ============================================================================
// For Loop Tests
// ============================================================================

// Test simple for loop over vector
TEST_F(StatementTest, ForSimple) {
    Continuation* k = parser->parse(
        "sum ← 0\n"
        ":For x :In 1 2 3 4 5\n"
        "  sum ← sum + x\n"
        ":EndFor\n"
        "sum"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // sum = 1+2+3+4+5 = 15
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

// Test for loop over scalar (single iteration)
TEST_F(StatementTest, ForScalar) {
    Continuation* k = parser->parse(
        "result ← 0\n"
        ":For x :In 42\n"
        "  result ← x\n"
        ":EndFor\n"
        "result"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test for loop with expression array
TEST_F(StatementTest, ForExpression) {
    Continuation* k = parser->parse(
        "arr ← 10 20 30\n"
        "sum ← 0\n"
        ":For val :In arr\n"
        "  sum ← sum + val\n"
        ":EndFor\n"
        "sum"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // sum = 10+20+30 = 60
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 60.0);
}

// Test nested for loops
TEST_F(StatementTest, ForNested) {
    Continuation* k = parser->parse(
        "sum ← 0\n"
        ":For i :In 1 2 3\n"
        "  :For j :In 10 20\n"
        "    sum ← sum + i + j\n"
        "  :EndFor\n"
        ":EndFor\n"
        "sum"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // i=1: j=10 (11), j=20 (21) -> 32
    // i=2: j=10 (12), j=20 (22) -> 34
    // i=3: j=10 (13), j=20 (23) -> 36
    // Total: 32+34+36 = 102
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 102.0);
}

// Test for loop with multiple statements
TEST_F(StatementTest, ForMultipleStatements) {
    Continuation* k = parser->parse(
        "sum ← 0\n"
        "product ← 1\n"
        ":For x :In 2 3 4\n"
        "  sum ← sum + x\n"
        "  product ← product × x\n"
        ":EndFor\n"
        "sum + product"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // sum = 2+3+4 = 9
    // product = 2*3*4 = 24
    // result = 9+24 = 33
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 33.0);
}

// ============================================================================
// Leave/Return Tests
// ============================================================================

// Test :Leave from While loop
TEST_F(StatementTest, LeaveFromWhile) {
    Continuation* k = parser->parse(
        "i ← 5\n"
        ":While 1\n"
        "  i ← i - 1\n"
        "  :If i\n"
        "    :Leave\n"
        "  :EndIf\n"
        ":EndWhile\n"
        "i"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // Loop exits when i becomes 0 after decrement, leaving i at 0
    // Wait no - we decrement first, then check. So: i=5, dec to 4 (truthy, Leave)
    // After :Leave, we still need to evaluate final 'i' statement
    ASSERT_NE(result, nullptr) << "Result should not be null after Leave";
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

// Test :Leave from For loop
TEST_F(StatementTest, LeaveFromFor) {
    Continuation* k = parser->parse(
        "sum ← 0\n"
        ":For x :In 10 20 30 40 50\n"
        "  sum ← sum + x\n"
        "  :If sum = 60\n"
        "    :Leave\n"
        "  :EndIf\n"
        ":EndFor\n"
        "sum"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // sum progression: 10, 30, 60
    // When sum=60: sum=60 is 1 (truthy), Leave
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 60.0);
}

// Test :Leave from nested loops (exits innermost)
TEST_F(StatementTest, LeaveFromNested) {
    Continuation* k = parser->parse(
        "count ← 0\n"
        ":For i :In 1 2 3\n"
        "  :For j :In 1 2 3\n"
        "    count ← count + 1\n"
        "    :If count - 2\n"
        "      :Leave\n"
        "    :EndIf\n"
        "  :EndFor\n"
        ":EndFor\n"
        "count"
    );

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();

    // First inner loop: count=1 (1-2=-1, truthy, Leave), exits with count=1
    // Wait, we want to test nested loop exit, so let's do it differently
    // Inner loop: j=1, count=1 (1-2=-1, truthy, Leave after first iteration)
    // This would exit too early. Let me reconsider...
    // Actually: count=1, check 1-2=-1 (truthy), Leave → count=1
    // Then second outer: count=2, check 2-2=0 (falsy), continue
    // count=3, check 3-2=1 (truthy), Leave → count=3
    // Third outer: count=4, check 4-2=2 (truthy), Leave → count=4
    // Total: 1, 3, 4 → final count=4
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
