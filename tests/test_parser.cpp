// Parser tests

#include <gtest/gtest.h>
#include "parser.h"
#include "machine.h"
#include "continuation.h"

using namespace apl;

class ParserTest : public ::testing::Test {
protected:
    Machine* machine;
    Parser* parser;

    void SetUp() override {
        machine = new Machine();
        parser = machine->parser;
    }

    void TearDown() override {
        delete machine;  // Machine will delete its parser
    }

    // Helper: evaluate parsed continuation using the CEK machine
    Value* eval(Continuation* k) {
        // The parsed continuation graph needs to be evaluated
        // Push continuation onto stack and execute via trampoline
        machine->push_kont(k);
        Value* result = machine->execute();

        return result;
    }

};

// Test parsing a simple literal
TEST_F(ParserTest, ParseLiteral) {
    Continuation* k = parser->parse("42");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test parsing a negative literal
TEST_F(ParserTest, ParseNegativeLiteral) {
    Continuation* k = parser->parse("-3.14");

    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -3.14);
}

// Test parsing zero
TEST_F(ParserTest, ParseZero) {
    Continuation* k = parser->parse("0");

    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test parsing failure with invalid input
TEST_F(ParserTest, ParseInvalidInput) {
    // Use @ which is not a valid token
    Continuation* k = parser->parse("@");

    EXPECT_EQ(k, nullptr);
    EXPECT_NE(parser->get_error(), "");
}

// Test parse-time safety: no Values allocated during parse
TEST_F(ParserTest, ParseTimeSafety) {
    size_t values_before = machine->heap->total_size();

    Continuation* k = parser->parse("123");

    size_t values_after = machine->heap->total_size();

    // NO Values should be allocated during parsing
    EXPECT_EQ(values_before, values_after);

    // Values are only created when we EVALUATE
    eval(k);

    size_t values_after_invoke = machine->heap->total_size();
    EXPECT_GT(values_after_invoke, values_before);
}

// Test parsing simple addition
TEST_F(ParserTest, ParseAddition) {
    Continuation* k = parser->parse("2 + 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test DyadicK with trampoline
TEST_F(ParserTest, TestDyadicKWithTrampoline) {
    // Manually construct a DyadicK: 2 + 3
    LiteralK* left = machine->heap->allocate<LiteralK>(2.0);
    LiteralK* right = machine->heap->allocate<LiteralK>(3.0);

    // Intern the operator name
    const char* plus_name = machine->string_pool.intern("+");

    DyadicK* dyadic = machine->heap->allocate<DyadicK>(plus_name, left, right);

    // Allocate in heap for GC

    // Use trampoline instead of direct invoke
    machine->push_kont(dyadic);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test right-to-left evaluation: 2 + 3 * 4 (where * is power) = 2 + 3^4 = 2 + 81 = 83
TEST_F(ParserTest, ParseRightToLeft) {
    Continuation* k = parser->parse("2 + 3 * 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 83.0);  // 2 + (3^4) = 2 + 81 = 83
}

// Test with APL symbols: 2 + 3 × 4
TEST_F(ParserTest, ParseAPLSymbols) {
    Continuation* k = parser->parse("2 + 3 × 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test division
TEST_F(ParserTest, ParseDivision) {
    Continuation* k = parser->parse("10 ÷ 2");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test subtraction
TEST_F(ParserTest, ParseSubtraction) {
    Continuation* k = parser->parse("10 - 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

// Test longer chain: 1 + 2 + 3 + 4 = 1 + 2 + 7 = 1 + 9 = 10 (right-to-left)
TEST_F(ParserTest, ParseLongChain) {
    Continuation* k = parser->parse("1 + 2 + 3 + 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test mixed operators: 10 - 2 × 3 = 10 - (2 × 3) = 10 - 6 = 4
TEST_F(ParserTest, ParseMixedOperators) {
    Continuation* k = parser->parse("10 - 2 × 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

// ============================================================================
// Parenthesized Expression Tests (Phase 3.2.1)
// ============================================================================

// Test simple parenthesized literal
TEST_F(ParserTest, ParseParenthesizedLiteral) {
    Continuation* k = parser->parse("(42)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test parenthesized expression
TEST_F(ParserTest, ParseParenthesizedExpression) {
    Continuation* k = parser->parse("(2 + 3)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test parentheses change evaluation order: 2 × (3 + 4) = 2 × 7 = 14
TEST_F(ParserTest, ParseParenthesesPrecedence) {
    Continuation* k = parser->parse("2 × (3 + 4)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test without parentheses for comparison: 2 × 3 + 4 = 2 × (3 + 4) = 2 × 7 = 14 (right-to-left)
TEST_F(ParserTest, ParseWithoutParentheses) {
    Continuation* k = parser->parse("2 × 3 + 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);  // 2 * (3 + 4) right-to-left
}

// Test nested parentheses: ((2 + 3) × 4) = 5 × 4 = 20
TEST_F(ParserTest, ParseNestedParentheses) {
    Continuation* k = parser->parse("((2 + 3) × 4)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

// Test complex nested expression: (10 - (2 + 3)) × 2 = (10 - 5) × 2 = 5 × 2 = 10
TEST_F(ParserTest, ParseComplexNested) {
    Continuation* k = parser->parse("(10 - (2 + 3)) × 2");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test multiple parenthesized subexpressions: (1 + 2) + (3 + 4) = 3 + 7 = 10
TEST_F(ParserTest, ParseMultipleParentheses) {
    Continuation* k = parser->parse("(1 + 2) + (3 + 4)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test parentheses with division: (12 ÷ 2) ÷ 3 = 6 ÷ 3 = 2
TEST_F(ParserTest, ParseParenthesesDivision) {
    Continuation* k = parser->parse("(12 ÷ 2) ÷ 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// Test unclosed parenthesis error
TEST_F(ParserTest, ParseUnclosedParenthesis) {
    Continuation* k = parser->parse("(2 + 3");

    EXPECT_EQ(k, nullptr);
    EXPECT_NE(parser->get_error(), "");
}

// Test unmatched closing parenthesis
TEST_F(ParserTest, ParseUnmatchedClosingParen) {
    Continuation* k = parser->parse("2 + 3)");

    EXPECT_EQ(k, nullptr);
    EXPECT_NE(parser->get_error(), "");
}

// Test empty parentheses error
TEST_F(ParserTest, ParseEmptyParentheses) {
    Continuation* k = parser->parse("()");

    EXPECT_EQ(k, nullptr);
    EXPECT_NE(parser->get_error(), "");
}

// ============================================================================
// Array Strand Tests (Phase 3.2.2)
// ============================================================================

// Test simple 2-element strand
TEST_F(ParserTest, ParseSimpleStrand) {
    Continuation* k = parser->parse("1 2");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 2);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
}

// Test 3-element strand
TEST_F(ParserTest, ParseThreeElementStrand) {
    Continuation* k = parser->parse("1 2 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

// Test longer strand
TEST_F(ParserTest, ParseLongerStrand) {
    Continuation* k = parser->parse("10 20 30 40 50");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 5);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*m)(4, 0), 50.0);
}

// ISO 13751: Strands are lexical - only numeric literals form strands
// Tests for complex strand expressions (variables, parenthesized expressions) removed
// Use ravel operator (,) for creating vectors from computed values

// Test strand with decimals
TEST_F(ParserTest, ParseStrandWithDecimals) {
    Continuation* k = parser->parse("1.5 2.25 3.75");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.5);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.25);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.75);
}

// Test strand as operand to binary operation: (1 2 3) + 10 = 11 12 13
TEST_F(ParserTest, ParseStrandAsLeftOperand) {
    Continuation* k = parser->parse("1 2 3 + 10");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 13.0);
}

// Test strand as right operand: 10 + (1 2 3) = 11 12 13
TEST_F(ParserTest, ParseStrandAsRightOperand) {
    Continuation* k = parser->parse("10 + 1 2 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 13.0);
}

// Test two strands: (1 2) + (3 4) = 4 6
TEST_F(ParserTest, ParseTwoStrands) {
    Continuation* k = parser->parse("1 2 + 3 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 2);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 6.0);
}

// Test parenthesized strand: (1 2 3) should still be a strand
TEST_F(ParserTest, ParseParenthesizedStrand) {
    Continuation* k = parser->parse("(1 2 3)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

// ============================================================================
// Variable Parsing Tests (Phase 3.2.3)
// ============================================================================

// Test simple variable lookup
TEST_F(ParserTest, ParseSimpleVariable) {
    // Define a variable in the environment
    Value* val = machine->heap->allocate_scalar(42.0);
    machine->env->define("x", val);

    Continuation* k = parser->parse("x");
    ASSERT_NE(k, nullptr);

    // Check that it parsed as LookupK
    LookupK* lookup = dynamic_cast<LookupK*>(k);
    ASSERT_NE(lookup, nullptr);
    EXPECT_STREQ(lookup->var_name, "x");

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test variable in expression
TEST_F(ParserTest, ParseVariableInExpression) {
    // Define variables
    machine->env->define("a", machine->heap->allocate_scalar(10.0));
    machine->env->define("b", machine->heap->allocate_scalar(5.0));

    Continuation* k = parser->parse("a + b");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

// Test undefined variable error
TEST_F(ParserTest, ParseUndefinedVariable) {
    Continuation* k = parser->parse("undefined");
    ASSERT_NE(k, nullptr);

    // Phase 1: Now throws exception instead of returning nullptr
    EXPECT_THROW(eval(k), std::runtime_error);
}

// ============================================================================
// Monadic Operator Tests
// ============================================================================

// Test simple negation: -5 = -5
TEST_F(ParserTest, ParseMonadicNegate) {
    Continuation* k = parser->parse("-5");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

// Test negation with expression: -(2 + 3) = -5
TEST_F(ParserTest, ParseMonadicNegateExpression) {
    Continuation* k = parser->parse("-(2 + 3)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

// Test reciprocal: ÷4 = 0.25
TEST_F(ParserTest, ParseMonadicReciprocal) {
    Continuation* k = parser->parse("÷4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.25);
}

// Test exponential: *0 = 1 (e^0)
TEST_F(ParserTest, ParseMonadicExponential) {
    Continuation* k = parser->parse("*0");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// Test exponential: *1 = e
TEST_F(ParserTest, ParseMonadicExponentialE) {
    Continuation* k = parser->parse("*1");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 2.71828, 0.0001);
}

// Test identity/conjugate: +5 = 5
TEST_F(ParserTest, ParseMonadicIdentity) {
    Continuation* k = parser->parse("+5");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test signum: ×5 = 1, ×(-5) = -1, ×0 = 0
TEST_F(ParserTest, ParseMonadicSignum) {
    Continuation* k1 = parser->parse("×5");
    ASSERT_NE(k1, nullptr);
    Value* r1 = eval(k1);
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    Continuation* k2 = parser->parse("×(-5)");
    ASSERT_NE(k2, nullptr);
    Value* r2 = eval(k2);
    EXPECT_DOUBLE_EQ(r2->as_scalar(), -1.0);

    Continuation* k3 = parser->parse("×0");
    ASSERT_NE(k3, nullptr);
    Value* r3 = eval(k3);
    EXPECT_DOUBLE_EQ(r3->as_scalar(), 0.0);
}

// Test monadic with dyadic: 3 - -5 = 3 - (-5) = 8
TEST_F(ParserTest, ParseMonadicInDyadicContext) {
    Continuation* k = parser->parse("3 - -5");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// Test double negation: --5 = 5
TEST_F(ParserTest, ParseDoubleNegation) {
    Continuation* k = parser->parse("--5");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// ========== Assignment Tests ==========

// Test basic assignment: x ← 5
TEST_F(ParserTest, ParseBasicAssignment) {
    Continuation* k = parser->parse("x ← 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);

    // Check that x is now in the environment
    Value* x = machine->env->lookup("x");
    ASSERT_NE(x, nullptr);
    EXPECT_TRUE(x->is_scalar());
    EXPECT_DOUBLE_EQ(x->as_scalar(), 5.0);
}

// Test assignment with expression: y ← 3 + 4
TEST_F(ParserTest, ParseAssignmentWithExpression) {
    Continuation* k = parser->parse("y ← 3 + 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

    // Check environment
    Value* y = machine->env->lookup("y");
    ASSERT_NE(y, nullptr);
    EXPECT_DOUBLE_EQ(y->as_scalar(), 7.0);
}

// Test variable lookup after assignment
TEST_F(ParserTest, AssignmentThenLookup) {
    // First assign
    Continuation* k1 = parser->parse("z ← 10");
    ASSERT_NE(k1, nullptr);
    eval(k1);

    // Then use the variable
    Continuation* k2 = parser->parse("z");
    ASSERT_NE(k2, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k2);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test variable in expression after assignment: z + 5
TEST_F(ParserTest, AssignmentThenUseInExpression) {
    // Assign z ← 10
    Continuation* k1 = parser->parse("z ← 10");
    ASSERT_NE(k1, nullptr);
    eval(k1);

    // Use z in expression
    Continuation* k2 = parser->parse("z + 5");
    ASSERT_NE(k2, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k2);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

// Test function assignment and monadic application: f ← +, then f 3
TEST_F(ParserTest, FunctionAssignmentMonadic) {
    // First, manually add + to a variable for testing
    // (We can't parse "f ← +" yet because + is a token, not a name)
    Value* plus_fn = machine->env->lookup("+");
    ASSERT_NE(plus_fn, nullptr) << "+ should be in global environment";
    machine->env->define("f", plus_fn);

    // Now parse "f 3" - should apply f (which is +) monadically to 3
    // Monadic + is identity, so result should be 3
    Continuation* k = parser->parse("f 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// Test function assignment and dyadic application: f ← +, then 2 f 3
TEST_F(ParserTest, FunctionAssignmentDyadic) {
    // Manually add + to variable f
    Value* plus_fn = machine->env->lookup("+");
    ASSERT_NE(plus_fn, nullptr) << "+ should be in global environment";
    machine->env->define("f", plus_fn);

    // Now parse "2 f 3" - should apply f (which is +) dyadically to 2 and 3
    // Dyadic + is addition, so result should be 5
    Continuation* k = parser->parse("2 f 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Array operation tests
TEST_F(ParserTest, IotaMonadic) {
    // Parse "⍳ 5" - should generate vector [0 1 2 3 4]
    Continuation* k = parser->parse("⍳ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 4.0);
}

TEST_F(ParserTest, RavelMonadic) {
    // Parse ", 1 2 3" - ravel of vector is identity
    Continuation* k = parser->parse(", 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(ParserTest, ShapeMonadic) {
    // Parse "⍴ 1 2 3" - should return shape [3]
    Continuation* k = parser->parse("⍴ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 1);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 3.0);
}

TEST_F(ParserTest, ReshapeDyadic) {
    // Parse "2 3 ⍴ ⍳ 6" - reshape iota(6) into 2x3 matrix
    // APL uses row-major order (ISO 13751 §8.2.1):
    //   0 1 2
    //   3 4 5
    Continuation* k = parser->parse("2 3 ⍴ ⍳ 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(result->is_vector());  // Should be a 2D matrix
    EXPECT_FALSE(result->is_scalar());

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    // Row 0: 0, 1, 2
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 2.0);
    // Row 1: 3, 4, 5
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 5.0);
}

TEST_F(ParserTest, CatenateDyadic) {
    // Parse "1 2 , 3 4" - concatenate two vectors
    Continuation* k = parser->parse("1 2 , 3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
}

TEST_F(ParserTest, EqualDyadicTrue) {
    // Parse "5 = 5" - equality test (true)
    Continuation* k = parser->parse("5 = 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 1 = true
}

TEST_F(ParserTest, EqualDyadicFalse) {
    // Parse "5 = 3" - equality test (false)
    Continuation* k = parser->parse("5 = 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 0 = false
}

TEST_F(ParserTest, EqualDyadicVector) {
    // Parse "1 2 3 = 1 5 3" - element-wise equality
    Continuation* k = parser->parse("1 2 3 = 1 5 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1=1 true
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 2=5 false
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // 3=3 true
}

// ============================================================================
// Dfn (Dynamic Function) Tests
// ============================================================================

// Test parsing a simple monadic dfn
TEST_F(ParserTest, ParseMonadicDfn) {
    // {⍵×2} should create a closure
    Continuation* k = parser->parse("{⍵×2}");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

// Test applying a monadic dfn
TEST_F(ParserTest, ApplyMonadicDfn) {
    // ({⍵×2} 5) should evaluate to 10
    // Note: Need to use dyadic syntax since we're applying the dfn to an argument
    Continuation* k = parser->parse("5 {⍵×2} 0");  // Using a dummy left arg for now

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    // This will fail until we implement monadic application properly
    // For now, just test that it parses
}

// Test parsing a dyadic dfn
TEST_F(ParserTest, ParseDyadicDfn) {
    // {⍺+⍵} should create a closure
    Continuation* k = parser->parse("{⍺+⍵}");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

// Test dfn that just returns omega
TEST_F(ParserTest, DfnJustOmega) {
    // 0 {⍵} 5 should return 5
    Continuation* k = parser->parse("0 {⍵} 5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test applying a dyadic dfn
TEST_F(ParserTest, ApplyDyadicDfn) {
    // 3 {⍺+⍵} 5 should evaluate to 8
    Continuation* k = parser->parse("3 {⍺+⍵} 5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR) << "Result should be SCALAR, got tag=" << static_cast<int>(result->tag);

    if (result->tag == ValueType::SCALAR) {
        double val = result->as_scalar();
        EXPECT_DOUBLE_EQ(val, 8.0) << "Expected 8.0, got " << val;
    }
}

// Test assigning a dfn to a variable
TEST_F(ParserTest, AssignDfn) {
    // square ← {⍵×⍵}
    Continuation* k = parser->parse("square ← {⍵×⍵}");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    // Check that the variable was assigned
    Value* square = machine->env->lookup("square");
    ASSERT_NE(square, nullptr);
    EXPECT_EQ(square->tag, ValueType::CLOSURE);
}

// Test dfn with nested expressions
TEST_F(ParserTest, DfnNestedExpression) {
    // {(⍺×2)+(⍵×3)} should parse correctly
    Continuation* k = parser->parse("{(⍺×2)+(⍵×3)}");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

// Test applying nested dfn
TEST_F(ParserTest, ApplyNestedDfn) {
    // 4 {(⍺×2)+(⍵×3)} 5 should evaluate to (4×2)+(5×3) = 8+15 = 23
    Continuation* k = parser->parse("4 {(⍺×2)+(⍵×3)} 5");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 23.0);
}

// ============================================================================
// GC Integration Tests (Phase 5.2)
// ============================================================================

// Test that GC can run during parsing without breaking anything
TEST_F(ParserTest, GCDuringParsing) {
    // Force a GC before parsing
    machine->heap->collect(machine);

    size_t heap_size_before = machine->heap->total_size();

    // Parse a complex expression
    Continuation* k = parser->parse("((1+2)×(3+4))+(5×6)");
    ASSERT_NE(k, nullptr);

    // Force another GC after parsing
    machine->heap->collect(machine);

    // Execute the parsed continuation
    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 51.0);  // (3×7)+30 = 21+30 = 51

    // Verify heap is stable
    size_t heap_size_after = machine->heap->total_size();
    EXPECT_GT(heap_size_after, 0);  // Heap should have objects
}

// Test that parser can handle large expressions without leaking
TEST_F(ParserTest, LargeExpressionGC) {
    // Build a large nested expression that creates 100+ continuations
    // Expression: 1+2+3+4+...+50 (creates many continuation nodes)
    std::string expr = "1";
    for (int i = 2; i <= 50; i++) {
        expr += "+" + std::to_string(i);
    }

    // Parse the large expression
    Continuation* k = parser->parse(expr.c_str());
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    // Execute it
    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1275.0);  // Sum 1 to 50 = 50×51/2 = 1275

    // Force GC - should not crash or corrupt heap
    machine->heap->collect(machine);

    // Verify heap is still functional after GC
    Continuation* k2 = parser->parse("1+2+3");
    ASSERT_NE(k2, nullptr);
    Value* result2 = eval(k2);
    EXPECT_DOUBLE_EQ(result2->as_scalar(), 6.0);
}

// Test that parser doesn't leak continuations across multiple parses
TEST_F(ParserTest, ParserNoLeakAcrossParses) {
    // Parse and execute several expressions
    for (int i = 0; i < 10; i++) {
        Continuation* k = parser->parse("1+2+3+4+5");
        ASSERT_NE(k, nullptr);

        Value* result = eval(k);
        ASSERT_NE(result, nullptr);
        EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);

        // Force GC between iterations
        machine->heap->collect(machine);
    }

    // After all iterations, heap should be stable (not growing unbounded)
    size_t final_size = machine->heap->total_size();
    EXPECT_GT(final_size, 0);

    // Parse one more time and verify heap is still reasonable
    size_t before_last = machine->heap->total_size();
    Continuation* k = parser->parse("1+2+3+4+5");
    ASSERT_NE(k, nullptr);
    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);

    size_t after_last = machine->heap->total_size();
    // Heap shouldn't grow dramatically for the same expression
    EXPECT_LT(after_last, before_last * 2);
}

// Test GC during complex nested parsing
TEST_F(ParserTest, GCWithNestedExpressions) {
    // Create deeply nested expression
    std::string expr = "(((((1+2)+3)+4)+5)+6)";

    Continuation* k = parser->parse(expr.c_str());
    ASSERT_NE(k, nullptr);

    // Force GC while continuation graph exists
    machine->heap->collect(machine);

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 21.0);

    // GC after execution
    machine->heap->collect(machine);

    // Verify heap is still functional
    Continuation* k2 = parser->parse("10+20");
    ASSERT_NE(k2, nullptr);
    Value* result2 = eval(k2);
    EXPECT_DOUBLE_EQ(result2->as_scalar(), 30.0);
}

TEST_F(ParserTest, StrandIsNotJuxtapose) {
    // "2 3" is a lexical strand (single TOK_NUMBER_VECTOR token)
    // It should create StrandK, NOT JuxtaposeK
    Continuation* k = parser->parse("2 3");
    ASSERT_NE(k, nullptr);

    StrandK* strand = dynamic_cast<StrandK*>(k);
    ASSERT_NE(strand, nullptr) << "2 3 is a strand, not juxtaposition";

    // Verify it's NOT juxtapose
    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    EXPECT_EQ(jux, nullptr) << "Strand should not be JuxtaposeK";
}

TEST_F(ParserTest, MonadicNotJuxtapose) {
    // "- 5" is TWO tokens: TOK_MINUS ("-") and TOK_NUMBER ("5")
    // In G2 grammar, "-" is an identifier (fb-term), so this is juxtaposition
    Continuation* k = parser->parse("- 5");
    ASSERT_NE(k, nullptr);

    // In G2, this should be JuxtaposeK
    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "- 5 should be JuxtaposeK in G2 grammar";

    // Verify it's NOT the old MonadicK
    MonadicK* monadic = dynamic_cast<MonadicK*>(k);
    EXPECT_EQ(monadic, nullptr) << "G2 grammar uses juxtaposition, not MonadicK";
}

TEST_F(ParserTest, FunctionNameAndStrand) {
    // "× 5 6" is TWO tokens: TOK_TIMES ("×") and TOK_NUMBER_VECTOR ([5, 6])
    // In G2 grammar, "×" is an identifier, so this is juxtaposition
    Continuation* k = parser->parse("× 5 6");
    ASSERT_NE(k, nullptr);

    // In G2, this should be JuxtaposeK
    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "× 5 6 should be JuxtaposeK in G2 grammar";

    // The right side should be a strand
    StrandK* strand = dynamic_cast<StrandK*>(jux->right);
    EXPECT_NE(strand, nullptr) << "Right operand should be strand [5, 6]";
}

TEST_F(ParserTest, NameNameCreatesJuxtapose) {
    // "f g" where both are variables should create JuxtaposeK
    // This is two TOK_NAME tokens, so juxtaposition should occur
    Value* plus_fn = machine->env->lookup("+");
    Value* minus_fn = machine->env->lookup("-");
    machine->env->define("f", plus_fn);
    machine->env->define("g", minus_fn);

    Continuation* k = parser->parse("f g");
    ASSERT_NE(k, nullptr);

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "f g should create JuxtaposeK (two NAME tokens)";
}

TEST_F(ParserTest, NameNumberCreatesJuxtapose) {
    // "f 3" where f is a variable and 3 is a number should create JuxtaposeK
    // This is two separate tokens: TOK_NAME and TOK_NUMBER
    Value* plus_fn = machine->env->lookup("+");
    machine->env->define("f", plus_fn);

    Continuation* k = parser->parse("f 3");
    ASSERT_NE(k, nullptr);

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "f 3 should create JuxtaposeK (NAME followed by NUMBER)";
}

TEST_F(ParserTest, NumberNameCreatesJuxtapose) {
    // "2 f" should create JuxtaposeK (two separate tokens)
    Value* plus_fn = machine->env->lookup("+");
    machine->env->define("f", plus_fn);

    Continuation* k = parser->parse("2 f");
    ASSERT_NE(k, nullptr);

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "2 f should create JuxtaposeK (NUMBER followed by NAME)";
}

TEST_F(ParserTest, NumberNameNumberCreatesJuxtapose) {
    // "2 f 3" should create nested JuxtaposeK
    Value* plus_fn = machine->env->lookup("+");
    machine->env->define("f", plus_fn);

    Continuation* k = parser->parse("2 f 3");
    ASSERT_NE(k, nullptr);

    // Right-associative (APL style): 2 (f 3)
    JuxtaposeK* outer = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(outer, nullptr) << "2 f 3 should create outer JuxtaposeK";

    JuxtaposeK* inner = dynamic_cast<JuxtaposeK*>(outer->right);
    ASSERT_NE(inner, nullptr) << "Right side (f 3) should also be JuxtaposeK";
}

TEST_F(ParserTest, OuterProductScalarParseStructure) {
    // "3 ∘.× 5" with single scalars (simpler than strand case)
    Continuation* k = parser->parse("3 ∘.× 5");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    EXPECT_NE(jux, nullptr) << "3 ∘.× 5 should have JuxtaposeK at top level";
}

TEST_F(ParserTest, OuterProductStrandParseStructure) {
    // Outer product "3 4 ∘.× 5 6" should parse as:
    // JuxtaposeK(StrandK(3,4), JuxtaposeK(DerivedOperatorK(×,"∘."), StrandK(5,6)))
    // This is: (3 4) ((∘.×) (5 6))
    Continuation* k = parser->parse("3 4 ∘.× 5 6");
    ASSERT_NE(k, nullptr);

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr);

    // Left side should be StrandK (left array)
    StrandK* left_strand = dynamic_cast<StrandK*>(jux->left);
    ASSERT_NE(left_strand, nullptr) << "Left side should be StrandK";

    // Right side should be JuxtaposeK (derived operator applied to right array)
    JuxtaposeK* right_jux = dynamic_cast<JuxtaposeK*>(jux->right);
    ASSERT_NE(right_jux, nullptr) << "Right side should be JuxtaposeK";

    // Right.Left should be DerivedOperatorK
    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(right_jux->left);
    ASSERT_NE(derived, nullptr) << "Right.Left should be DerivedOperatorK";
    EXPECT_STREQ(derived->op_name, "∘.");
}

TEST_F(ParserTest, OuterProductEvaluatesToMatrix) {
    Continuation* k = parser->parse("3 4 ∘.× 5 6");
    ASSERT_NE(k, nullptr);

    // Debug: print parse tree structure
    std::cerr << "Outer product parse tree: " << typeid(*k).name() << std::endl;
    if (auto* jux = dynamic_cast<JuxtaposeK*>(k)) {
        std::cerr << "  Left: " << typeid(*jux->left).name() << std::endl;
        std::cerr << "  Right: " << typeid(*jux->right).name() << std::endl;
        if (auto* jux_right = dynamic_cast<JuxtaposeK*>(jux->right)) {
            std::cerr << "    Right.Left: " << typeid(*jux_right->left).name() << std::endl;
            std::cerr << "    Right.Right: " << typeid(*jux_right->right).name() << std::endl;
            if (auto* derived = dynamic_cast<DerivedOperatorK*>(jux_right->left)) {
                std::cerr << "      DerivedOp.operand: " << typeid(*derived->operand_cont).name() << std::endl;
                std::cerr << "      DerivedOp.op_name: " << derived->op_name << std::endl;
            }
        }
    }

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::MATRIX) << "Result should be MATRIX, got tag " << static_cast<int>(result->tag);
}

// ============================================================================
// Comprehensive Juxtaposition Tests
// Test all cases of juxtaposition in G2 grammar
// ============================================================================

// Test: Primitive function followed by operator creates derived operator
TEST_F(ParserTest, PrimitiveFunctionWithOperator) {
    Continuation* k = parser->parse("+/");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // Should be DerivedOperatorK (+ with /)
    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(k);
    ASSERT_NE(derived, nullptr) << "+/ should create DerivedOperatorK";
    EXPECT_STREQ(derived->op_name, "/");
}

// Test: Derived operator followed by primitive function (juxtaposition)
TEST_F(ParserTest, DerivedOperatorWithPrimitiveFunctionJuxtaposition) {
    Continuation* k = parser->parse("+/ ×");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // "+/" is fb-term, "×" triggers juxtaposition
    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "+/ × should create JuxtaposeK";
}

// Test: Chained operators with juxtaposition "+/ ×/ 1 2 3"
TEST_F(ParserTest, ChainedOperatorsWithJuxtaposition) {
    Continuation* k = parser->parse("+/ ×/ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // Should parse as: (+/) ((×/) (1 2 3))
    // Top level should be JuxtaposeK
    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "+/ ×/ 1 2 3 should create JuxtaposeK at top level";

    // Left side should be DerivedOperatorK (+/)
    DerivedOperatorK* left_derived = dynamic_cast<DerivedOperatorK*>(jux->left);
    ASSERT_NE(left_derived, nullptr) << "Left side should be +/";
    EXPECT_STREQ(left_derived->op_name, "/");
}

// Test: Primitive function with commute operator
TEST_F(ParserTest, PrimitiveFunctionWithCommute) {
    Continuation* k = parser->parse("+⍨");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // Should be DerivedOperatorK (+ with ⍨)
    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(k);
    ASSERT_NE(derived, nullptr) << "+⍨ should create DerivedOperatorK";
    EXPECT_STREQ(derived->op_name, "⍨");
}

// Test: Number juxtaposed with function and commute "2 +⍨ 3"
TEST_F(ParserTest, NumberFunctionCommuteNumber) {
    Continuation* k = parser->parse("2 +⍨ 3");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // Should parse as: ((2 (+⍨)) 3) via left-to-right juxtaposition
    // Top level is JuxtaposeK
    JuxtaposeK* top_jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(top_jux, nullptr) << "2 +⍨ 3 should create JuxtaposeK at top level";
}

// Test: Inner product juxtaposition "3 4 +.× 5 6"
TEST_F(ParserTest, InnerProductJuxtaposition) {
    Continuation* k = parser->parse("3 4 +.× 5 6");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // Should parse - structure will be juxtaposition of strand, derived op, strand
    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "3 4 +.× 5 6 should create JuxtaposeK";
}

// Test: Primitive function followed by dot and another function "+.×"
TEST_F(ParserTest, PrimitiveFunctionDotPrimitiveFunction) {
    Continuation* k = parser->parse("+.×");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // Should parse as: (+.) followed by × creating juxtaposition or application
    // The key is it should parse without error
}

// Test: Primitive function followed by number "2 + 3"
TEST_F(ParserTest, PrimitiveFunctionJuxtaposedWithNumbers) {
    Continuation* k = parser->parse("2 + 3");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // Should parse as: (2 (+)) 3 via juxtaposition
    // Top level is JuxtaposeK
    JuxtaposeK* top_jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(top_jux, nullptr) << "2 + 3 should create JuxtaposeK at top level";

    // Can also evaluate to verify it works
    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test: Multiple primitive functions in a row "2 + 3 × 4"
TEST_F(ParserTest, MultiplePrimitiveFunctionsJuxtaposition) {
    Continuation* k = parser->parse("2 + 3 × 4");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // Should parse via left-to-right juxtaposition
    JuxtaposeK* top_jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(top_jux, nullptr) << "2 + 3 × 4 should create JuxtaposeK at top level";

    // Evaluate to verify right-to-left application: 2 + (3 × 4) = 2 + 12 = 14
    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// ============================================================================
// Derived Operator Tests (commute/duplicate)
// ============================================================================

TEST_F(ParserTest, DerivedOperatorCommuteParsing) {
    // "+⍨" should parse as a derived operator (DerivedOperatorK)
    Continuation* k = parser->parse("+⍨");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(k);
    ASSERT_NE(derived, nullptr) << "+⍨ should parse as DerivedOperatorK";
}

TEST_F(ParserTest, DerivedOperatorCommuteMonadic) {
    // "+⍨ 3" should be JuxtaposeK with derived operator on left
    Continuation* k = parser->parse("+⍨ 3");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "+⍨ 3 should be JuxtaposeK";

    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(jux->left);
    EXPECT_NE(derived, nullptr) << "Left side should be DerivedOperatorK";
}

TEST_F(ParserTest, DerivedOperatorCommuteDyadic) {
    // "2 +⍨ 3" should be JuxtaposeK: 2 ((+⍨) 3)
    // Right-associative, so: 2 (juxtapose) ((+⍨) (juxtapose) 3)
    Continuation* k = parser->parse("2 +⍨ 3");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    JuxtaposeK* outer = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(outer, nullptr) << "2 +⍨ 3 should be JuxtaposeK at top level";

    // Left should be LiteralK(2)
    LiteralK* left_lit = dynamic_cast<LiteralK*>(outer->left);
    EXPECT_NE(left_lit, nullptr) << "Left should be LiteralK(2)";

    // Right should be JuxtaposeK: (+⍨) 3
    JuxtaposeK* inner = dynamic_cast<JuxtaposeK*>(outer->right);
    ASSERT_NE(inner, nullptr) << "Right should be JuxtaposeK (+⍨ 3)";

    // Inner left should be DerivedOperatorK
    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(inner->left);
    EXPECT_NE(derived, nullptr) << "Inner left should be DerivedOperatorK (+⍨)";

    // Inner right should be LiteralK(3)
    LiteralK* right_lit = dynamic_cast<LiteralK*>(inner->right);
    EXPECT_NE(right_lit, nullptr) << "Inner right should be LiteralK(3)";
}

// ============================================================================
// Comparison Operator Tests (≠ < > ≤ ≥)
// ============================================================================

TEST_F(ParserTest, NotEqualDyadicTrue) {
    Continuation* k = parser->parse("5 ≠ 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5≠3 is true
}

TEST_F(ParserTest, NotEqualDyadicFalse) {
    Continuation* k = parser->parse("5 ≠ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 5≠5 is false
}

TEST_F(ParserTest, NotEqualDyadicVector) {
    Continuation* k = parser->parse("1 2 3 ≠ 1 5 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);  // 1≠1 false
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);  // 2≠5 true
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);  // 3≠3 false
}

TEST_F(ParserTest, LessThanDyadicTrue) {
    Continuation* k = parser->parse("3 < 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 3<5 is true
}

TEST_F(ParserTest, LessThanDyadicFalse) {
    Continuation* k = parser->parse("5 < 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 5<3 is false
}

TEST_F(ParserTest, LessThanDyadicVector) {
    Continuation* k = parser->parse("1 5 3 < 2 3 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1<2 true
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 5<3 false
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);  // 3<3 false
}

TEST_F(ParserTest, GreaterThanDyadicTrue) {
    Continuation* k = parser->parse("5 > 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5>3 is true
}

TEST_F(ParserTest, GreaterThanDyadicFalse) {
    Continuation* k = parser->parse("3 > 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 3>5 is false
}

TEST_F(ParserTest, GreaterThanDyadicVector) {
    Continuation* k = parser->parse("5 2 3 > 3 4 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 5>3 true
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 2>4 false
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);  // 3>3 false
}

TEST_F(ParserTest, LessOrEqualDyadicLess) {
    Continuation* k = parser->parse("3 ≤ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 3≤5 is true
}

TEST_F(ParserTest, LessOrEqualDyadicEqual) {
    Continuation* k = parser->parse("5 ≤ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5≤5 is true
}

TEST_F(ParserTest, LessOrEqualDyadicFalse) {
    Continuation* k = parser->parse("7 ≤ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 7≤5 is false
}

TEST_F(ParserTest, LessOrEqualDyadicVector) {
    Continuation* k = parser->parse("1 5 3 ≤ 2 3 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1≤2 true
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 5≤3 false
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // 3≤3 true
}

TEST_F(ParserTest, GreaterOrEqualDyadicGreater) {
    Continuation* k = parser->parse("7 ≥ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 7≥5 is true
}

TEST_F(ParserTest, GreaterOrEqualDyadicEqual) {
    Continuation* k = parser->parse("5 ≥ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5≥5 is true
}

TEST_F(ParserTest, GreaterOrEqualDyadicFalse) {
    Continuation* k = parser->parse("3 ≥ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 3≥5 is false
}

TEST_F(ParserTest, GreaterOrEqualDyadicVector) {
    Continuation* k = parser->parse("5 2 3 ≥ 3 4 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 5≥3 true
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 2≥4 false
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // 3≥3 true
}

// Test comparisons in expressions
TEST_F(ParserTest, ComparisonInExpression) {
    // Count elements less than 3: +/ (⍳5) < 3
    // ⍳5 = 0 1 2 3 4
    // < 3 = 1 1 1 0 0
    // +/ = 3
    Continuation* k = parser->parse("+/ (⍳5) < 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// ============================================================================
// Min/Max Tests (⌈ ⌊)
// ============================================================================

TEST_F(ParserTest, CeilingMonadic) {
    Continuation* k = parser->parse("⌈ 3.2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

TEST_F(ParserTest, CeilingMonadicNegative) {
    Continuation* k = parser->parse("⌈ ¯3.2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -3.0);
}

TEST_F(ParserTest, FloorMonadic) {
    Continuation* k = parser->parse("⌊ 3.7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(ParserTest, FloorMonadicNegative) {
    Continuation* k = parser->parse("⌊ ¯3.2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -4.0);
}

TEST_F(ParserTest, MaximumDyadic) {
    Continuation* k = parser->parse("3 ⌈ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(ParserTest, MinimumDyadic) {
    Continuation* k = parser->parse("3 ⌊ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(ParserTest, MaximumDyadicVector) {
    Continuation* k = parser->parse("1 5 3 ⌈ 4 2 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 4.0);  // max(1,4)
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 5.0);  // max(5,2)
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 6.0);  // max(3,6)
}

TEST_F(ParserTest, MinimumDyadicVector) {
    Continuation* k = parser->parse("1 5 3 ⌊ 4 2 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // min(1,4)
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);  // min(5,2)
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);  // min(3,6)
}

// ============================================================================
// Logical Function Tests (∧ ∨ ~ ⍲ ⍱)
// ============================================================================

TEST_F(ParserTest, AndDyadicTrue) {
    Continuation* k = parser->parse("1 ∧ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, AndDyadicFalse) {
    Continuation* k = parser->parse("1 ∧ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(ParserTest, OrDyadicTrue) {
    Continuation* k = parser->parse("0 ∨ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, OrDyadicFalse) {
    Continuation* k = parser->parse("0 ∨ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(ParserTest, NotMonadicTrue) {
    Continuation* k = parser->parse("~ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, NotMonadicFalse) {
    Continuation* k = parser->parse("~ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(ParserTest, NotMonadicVector) {
    Continuation* k = parser->parse("~ 1 0 1 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 1.0);
}

TEST_F(ParserTest, NandDyadic) {
    // 1 ⍲ 1 = 0 (not (1 and 1))
    Continuation* k = parser->parse("1 ⍲ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(ParserTest, NandDyadicTrue) {
    // 1 ⍲ 0 = 1 (not (1 and 0))
    Continuation* k = parser->parse("1 ⍲ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, NorDyadic) {
    // 0 ⍱ 0 = 1 (not (0 or 0))
    Continuation* k = parser->parse("0 ⍱ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, NorDyadicFalse) {
    // 0 ⍱ 1 = 0 (not (0 or 1))
    Continuation* k = parser->parse("0 ⍱ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(ParserTest, LogicalVectorAnd) {
    Continuation* k = parser->parse("1 0 1 1 ∧ 1 1 0 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1∧1
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 0∧1
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);  // 1∧0
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 1.0);  // 1∧1
}

TEST_F(ParserTest, LogicalVectorOr) {
    Continuation* k = parser->parse("1 0 0 1 ∨ 0 0 1 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1∨0
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 0∨0
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // 0∨1
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 1.0);  // 1∨1
}

// Test logical operations in expressions
TEST_F(ParserTest, LogicalExpression) {
    // Count where both conditions are true
    // (⍳5) gives 0 1 2 3 4
    // > 1 gives 0 0 1 1 1
    // < 4 gives 1 1 1 1 0
    // ∧ gives 0 0 1 1 0
    // +/ gives 2
    Continuation* k = parser->parse("+/ ((⍳5) > 1) ∧ ((⍳5) < 4)");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// ============================================================================
// Comprehensive Basic Arithmetic Tests (+ - × ÷ *)
// ============================================================================

// Addition (+) comprehensive tests
TEST_F(ParserTest, AdditionDyadicVectors) {
    Continuation* k = parser->parse("1 2 3 + 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 9.0);
}

TEST_F(ParserTest, AdditionScalarExtensionLeft) {
    Continuation* k = parser->parse("10 + 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 13.0);
}

TEST_F(ParserTest, AdditionScalarExtensionRight) {
    Continuation* k = parser->parse("1 2 3 + 10");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 13.0);
}

TEST_F(ParserTest, IdentityMonadicVector) {
    // Monadic + is identity/conjugate
    Continuation* k = parser->parse("+ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

// Subtraction (-) comprehensive tests
TEST_F(ParserTest, SubtractionDyadicVectors) {
    Continuation* k = parser->parse("10 20 30 - 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 9.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 18.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 27.0);
}

TEST_F(ParserTest, SubtractionScalarExtension) {
    Continuation* k = parser->parse("100 - 10 20 30");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 90.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 80.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 70.0);
}

TEST_F(ParserTest, NegationMonadicVector) {
    Continuation* k = parser->parse("- 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), -3.0);
}

// Multiplication (×) comprehensive tests
TEST_F(ParserTest, MultiplicationDyadicVectors) {
    Continuation* k = parser->parse("2 3 4 × 5 6 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 18.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 28.0);
}

TEST_F(ParserTest, MultiplicationScalarExtension) {
    Continuation* k = parser->parse("10 × 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 20.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 30.0);
}

TEST_F(ParserTest, SignumMonadicVector) {
    // Monadic × is signum: -1, 0, or 1
    Continuation* k = parser->parse("× ¯5 0 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);
}

// Division (÷) comprehensive tests
TEST_F(ParserTest, DivisionDyadicVectors) {
    Continuation* k = parser->parse("10 20 30 ÷ 2 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 6.0);
}

TEST_F(ParserTest, DivisionScalarExtension) {
    Continuation* k = parser->parse("100 ÷ 2 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 50.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 25.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 20.0);
}

TEST_F(ParserTest, ReciprocalMonadicVector) {
    // Monadic ÷ is reciprocal
    Continuation* k = parser->parse("÷ 2 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.5);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.25);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.2);
}

// Power (*) comprehensive tests
TEST_F(ParserTest, PowerDyadicVectors) {
    Continuation* k = parser->parse("2 3 4 * 2 2 2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 9.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 16.0);
}

TEST_F(ParserTest, PowerScalarExtension) {
    Continuation* k = parser->parse("2 * 1 2 3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 8.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 16.0);
}

TEST_F(ParserTest, ExponentialMonadicVector) {
    // Monadic * is e^x
    Continuation* k = parser->parse("* 0 1 2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);                // e^0 = 1
    EXPECT_NEAR((*vec)(1, 0), 2.718281828, 1e-6);       // e^1 ≈ 2.718
    EXPECT_NEAR((*vec)(2, 0), 7.389056099, 1e-6);       // e^2 ≈ 7.389
}

// ============================================================================
// Comprehensive Array Operation Tests (⍴ , ⍉ ⍳ ↑ ↓)
// ============================================================================

// Transpose (⍉) tests
TEST_F(ParserTest, TransposeMatrix) {
    // Create 2x3 matrix and transpose to 3x2
    Continuation* k = parser->parse("⍉ 2 3 ⍴ 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 2);
    // Original was: 1 2 3
    //               4 5 6
    // Transposed:   1 4
    //               2 5
    //               3 6
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 1), 6.0);
}

TEST_F(ParserTest, TransposeVector) {
    // Transpose of vector is itself
    Continuation* k = parser->parse("⍉ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

// Take (↑) tests
TEST_F(ParserTest, TakePositive) {
    // Take first 3 elements
    Continuation* k = parser->parse("3 ↑ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(ParserTest, TakeNegative) {
    // Take last 3 elements
    Continuation* k = parser->parse("¯3 ↑ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);
}

TEST_F(ParserTest, TakeOverextend) {
    // Take more than available - pads with zeros
    Continuation* k = parser->parse("5 ↑ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 0.0);
}

// Drop (↓) tests
TEST_F(ParserTest, DropPositive) {
    // Drop first 2 elements
    Continuation* k = parser->parse("2 ↓ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);
}

TEST_F(ParserTest, DropNegative) {
    // Drop last 2 elements
    Continuation* k = parser->parse("¯2 ↓ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(ParserTest, DropAll) {
    // Drop more than available - empty result
    Continuation* k = parser->parse("10 ↓ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 0);
}

// Iota (⍳) additional tests
TEST_F(ParserTest, IotaZero) {
    // ⍳0 gives empty vector
    Continuation* k = parser->parse("⍳ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 0);
}

TEST_F(ParserTest, IotaOne) {
    // ⍳1 gives [0]
    Continuation* k = parser->parse("⍳ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 1);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
}

// Shape (⍴) additional tests
TEST_F(ParserTest, ShapeScalar) {
    // Shape of scalar is empty vector
    Continuation* k = parser->parse("⍴ 42");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 0);
}

TEST_F(ParserTest, ShapeMatrix) {
    // Shape of 2x3 matrix is [2, 3]
    Continuation* k = parser->parse("⍴ 2 3 ⍴ ⍳ 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 2);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 3.0);
}

// Reshape (⍴) additional tests
TEST_F(ParserTest, ReshapeWithCycling) {
    // Reshape cycles data: 2 3 ⍴ 1 2 gives matrix with 1 2 1 2 1 2
    Continuation* k = parser->parse("2 3 ⍴ 1 2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 2.0);
}

TEST_F(ParserTest, ReshapeToVector) {
    // Single dimension reshape creates vector
    Continuation* k = parser->parse("6 ⍴ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 6);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(5, 0), 3.0);
}

// Catenate (,) additional tests
TEST_F(ParserTest, CatenateScalars) {
    // Catenate two scalars creates a 2-element vector
    Continuation* k = parser->parse("1 , 2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 2);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
}

TEST_F(ParserTest, CatenateVectorScalar) {
    // Catenate vector with scalar
    Continuation* k = parser->parse("1 2 3 , 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
}

// Ravel (,) additional tests
TEST_F(ParserTest, RavelMatrix) {
    // Ravel 2x3 matrix to 6-element vector
    Continuation* k = parser->parse(", 2 3 ⍴ ⍳ 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 6);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(5, 0), 5.0);
}

TEST_F(ParserTest, RavelScalar) {
    // Ravel scalar creates 1-element vector
    Continuation* k = parser->parse(", 42");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 1);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 42.0);
}

// ============================================================================
// Arithmetic Extension Tests (| ⍟ !)
// ============================================================================

// Magnitude (|) tests
TEST_F(ParserTest, MagnitudeMonadic) {
    Continuation* k = parser->parse("| ¯5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(ParserTest, MagnitudePositive) {
    Continuation* k = parser->parse("| 3.5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.5);
}

TEST_F(ParserTest, MagnitudeVector) {
    Continuation* k = parser->parse("| ¯1 2 ¯3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
}

// Residue (|) tests - APL: A|B means B mod A
TEST_F(ParserTest, ResidueDyadic) {
    // 3|7 = 1 (7 mod 3)
    Continuation* k = parser->parse("3 | 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, ResidueZeroDivisor) {
    // 0|5 = 5 (special case)
    Continuation* k = parser->parse("0 | 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(ParserTest, ResidueVector) {
    // 3| 0 1 2 3 4 5 6 = 0 1 2 0 1 2 0
    Continuation* k = parser->parse("3 | 0 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 7);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(5, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(6, 0), 0.0);
}

// Natural Log (⍟) tests
TEST_F(ParserTest, NaturalLogMonadic) {
    // ⍟ e ≈ 1
    Continuation* k = parser->parse("⍟ 2.718281828");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 1.0, 1e-6);
}

TEST_F(ParserTest, NaturalLogOne) {
    // ⍟ 1 = 0
    Continuation* k = parser->parse("⍟ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Logarithm (⍟) dyadic tests
TEST_F(ParserTest, LogarithmDyadic) {
    // 10 ⍟ 100 = 2 (log base 10 of 100)
    Continuation* k = parser->parse("10 ⍟ 100");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 2.0, 1e-10);
}

TEST_F(ParserTest, LogarithmBase2) {
    // 2 ⍟ 8 = 3 (log base 2 of 8)
    Continuation* k = parser->parse("2 ⍟ 8");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 3.0, 1e-10);
}

// Factorial (!) tests
TEST_F(ParserTest, FactorialMonadic) {
    // !5 = 120
    Continuation* k = parser->parse("! 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 120.0);
}

TEST_F(ParserTest, FactorialZero) {
    // !0 = 1
    Continuation* k = parser->parse("! 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, FactorialVector) {
    // ! 0 1 2 3 4 = 1 1 2 6 24
    Continuation* k = parser->parse("! 0 1 2 3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 6.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 24.0);
}

// Binomial (!) dyadic tests - k!n = C(n,k) = "n choose k"
TEST_F(ParserTest, BinomialDyadic) {
    // 2!5 = C(5,2) = 10
    Continuation* k = parser->parse("2 ! 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(ParserTest, BinomialZero) {
    // 0!n = 1 for any n
    Continuation* k = parser->parse("0 ! 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, BinomialSame) {
    // n!n = 1
    Continuation* k = parser->parse("5 ! 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, BinomialPascalsRow) {
    // 0 1 2 3 4 ! 4 = 1 4 6 4 1 (row 4 of Pascal's triangle)
    Continuation* k = parser->parse("(0 1 2 3 4) ! 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 6.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 1.0);
}

// ============================================================================
// Reverse/Rotate Tests (⌽ ⊖ ≢)
// ============================================================================

// Reverse (⌽) monadic tests
TEST_F(ParserTest, ReverseVector) {
    Continuation* k = parser->parse("⌽ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 1.0);
}

TEST_F(ParserTest, ReverseScalar) {
    Continuation* k = parser->parse("⌽ 42");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(ParserTest, ReverseMatrix) {
    // Reverse last axis of matrix (reverse columns within each row)
    Continuation* k = parser->parse("⌽ 2 3 ⍴ 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    // Row 0: 3 2 1
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 1.0);
    // Row 1: 6 5 4
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 4.0);
}

// Rotate (⌽) dyadic tests
TEST_F(ParserTest, RotateVectorLeft) {
    // Positive rotation = left rotate
    Continuation* k = parser->parse("2 ⌽ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 2.0);
}

TEST_F(ParserTest, RotateVectorRight) {
    // Negative rotation = right rotate
    Continuation* k = parser->parse("¯2 ⌽ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 3.0);
}

TEST_F(ParserTest, RotateZero) {
    // Zero rotation = identity
    Continuation* k = parser->parse("0 ⌽ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

// Reverse First (⊖) monadic tests
TEST_F(ParserTest, ReverseFirstVector) {
    // For vectors, first axis is the only axis
    Continuation* k = parser->parse("⊖ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 1.0);
}

TEST_F(ParserTest, ReverseFirstMatrix) {
    // Reverse first axis of matrix (reverse rows)
    Continuation* k = parser->parse("⊖ 2 3 ⍴ 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    // Row 0 is now old row 1: 4 5 6
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 6.0);
    // Row 1 is now old row 0: 1 2 3
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 3.0);
}

// Rotate First (⊖) dyadic tests
TEST_F(ParserTest, RotateFirstMatrix) {
    // Rotate rows of matrix
    Continuation* k = parser->parse("1 ⊖ 3 2 ⍴ 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 2);
    // Original: [[1,2],[3,4],[5,6]], rotated 1 up: [[3,4],[5,6],[1,2]]
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 1), 2.0);
}

// Tally (≢) tests
TEST_F(ParserTest, TallyVector) {
    Continuation* k = parser->parse("≢ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(ParserTest, TallyScalar) {
    // Tally of scalar is 1
    Continuation* k = parser->parse("≢ 42");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, TallyMatrix) {
    // Tally of matrix is number of rows
    Continuation* k = parser->parse("≢ 3 4 ⍴ ⍳12");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(ParserTest, TallyEmpty) {
    // Tally of empty vector is 0
    Continuation* k = parser->parse("≢ 0 ↑ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// ============================================================================
// Search Functions (⍳ dyadic, ∊)
// ============================================================================

TEST_F(ParserTest, IndexOfScalar) {
    // 1 2 3 4 5 ⍳ 3 → 2 (0-origin)
    Continuation* k = parser->parse("1 2 3 4 5 ⍳ 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(ParserTest, IndexOfNotFound) {
    // 1 2 3 ⍳ 7 → 3 (length of haystack)
    Continuation* k = parser->parse("1 2 3 ⍳ 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(ParserTest, IndexOfVector) {
    // 10 20 30 40 ⍳ 30 10 99 → 2 0 4
    Continuation* k = parser->parse("10 20 30 40 ⍳ 30 10 99");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 2.0);  // 30 at index 2
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 10 at index 0
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 4.0);  // 99 not found
}

TEST_F(ParserTest, IndexOfDuplicates) {
    // First occurrence is returned
    // 5 3 5 7 3 ⍳ 3 5 → 1 0
    Continuation* k = parser->parse("5 3 5 7 3 ⍳ 3 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 3 first at index 1
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 5 first at index 0
}

TEST_F(ParserTest, MemberOfScalarFound) {
    // 3 ∊ 1 2 3 4 5 → 1
    Continuation* k = parser->parse("3 ∊ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(ParserTest, MemberOfScalarNotFound) {
    // 7 ∊ 1 2 3 → 0
    Continuation* k = parser->parse("7 ∊ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(ParserTest, MemberOfVector) {
    // 1 5 3 7 ∊ 1 2 3 → 1 0 1 0
    Continuation* k = parser->parse("1 5 3 7 ∊ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1 in set
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 5 not in set
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // 3 in set
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 0.0);  // 7 not in set
}

TEST_F(ParserTest, MemberOfWithIota) {
    // 5 7 2 ∊ ⍳10 → 1 1 1 (all found in 0..9)
    Continuation* k = parser->parse("5 7 2 ∊ ⍳10");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);
}

TEST_F(ParserTest, EnlistVector) {
    // ∊ 1 2 3 → 1 2 3 (identity for simple vector)
    Continuation* k = parser->parse("∊ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(ParserTest, EnlistScalar) {
    // ∊ 5 → 5 (1-element vector)
    Continuation* k = parser->parse("∊ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 1);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
}

TEST_F(ParserTest, EnlistMatrix) {
    // ∊ 2 3 ⍴ ⍳6 → 0 1 2 3 4 5 (flatten to vector)
    Continuation* k = parser->parse("∊ 2 3 ⍴ ⍳6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 6);
    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ((*vec)(i, 0), static_cast<double>(i));
    }
}

TEST_F(ParserTest, IndexOfWithArithmetic) {
    // Combined with arithmetic
    // (⍳5) ⍳ 2+1 → 3 (find 3 in 0 1 2 3 4)
    Continuation* k = parser->parse("(⍳5) ⍳ 2+1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    Value* result = eval(k);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
