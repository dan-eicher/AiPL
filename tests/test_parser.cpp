// Parser tests - tests parser mechanics, syntax, and continuation structure
// For evaluation tests (does "1+2" equal 3?), see test_eval.cpp

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
};

// Test parsing a simple literal
TEST_F(ParserTest, ParseLiteral) {
    Continuation* k = parser->parse("42");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test parsing a negative literal
TEST_F(ParserTest, ParseNegativeLiteral) {
    Continuation* k = parser->parse("-3.14");

    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -3.14);
}

// Test parsing zero
TEST_F(ParserTest, ParseZero) {
    Continuation* k = parser->parse("0");

    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

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
    machine->push_kont(k);
    machine->execute();

    size_t values_after_invoke = machine->heap->total_size();
    EXPECT_GT(values_after_invoke, values_before);
}

// Test parsing simple addition
TEST_F(ParserTest, ParseAddition) {
    Continuation* k = parser->parse("2 + 3");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 83.0);  // 2 + (3^4) = 2 + 81 = 83
}

// Test with APL symbols: 2 + 3 × 4
TEST_F(ParserTest, ParseAPLSymbols) {
    Continuation* k = parser->parse("2 + 3 × 4");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test division
TEST_F(ParserTest, ParseDivision) {
    Continuation* k = parser->parse("10 ÷ 2");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test subtraction
TEST_F(ParserTest, ParseSubtraction) {
    Continuation* k = parser->parse("10 - 3");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

// Test longer chain: 1 + 2 + 3 + 4 = 1 + 2 + 7 = 1 + 9 = 10 (right-to-left)
TEST_F(ParserTest, ParseLongChain) {
    Continuation* k = parser->parse("1 + 2 + 3 + 4");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test mixed operators: 10 - 2 × 3 = 10 - (2 × 3) = 10 - 6 = 4
TEST_F(ParserTest, ParseMixedOperators) {
    Continuation* k = parser->parse("10 - 2 × 3");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test parenthesized expression
TEST_F(ParserTest, ParseParenthesizedExpression) {
    Continuation* k = parser->parse("(2 + 3)");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test parentheses change evaluation order: 2 × (3 + 4) = 2 × 7 = 14
TEST_F(ParserTest, ParseParenthesesPrecedence) {
    Continuation* k = parser->parse("2 × (3 + 4)");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test without parentheses for comparison: 2 × 3 + 4 = 2 × (3 + 4) = 2 × 7 = 14 (right-to-left)
TEST_F(ParserTest, ParseWithoutParentheses) {
    Continuation* k = parser->parse("2 × 3 + 4");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);  // 2 * (3 + 4) right-to-left
}

// Test nested parentheses: ((2 + 3) × 4) = 5 × 4 = 20
TEST_F(ParserTest, ParseNestedParentheses) {
    Continuation* k = parser->parse("((2 + 3) × 4)");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

// Test complex nested expression: (10 - (2 + 3)) × 2 = (10 - 5) × 2 = 5 × 2 = 10
TEST_F(ParserTest, ParseComplexNested) {
    Continuation* k = parser->parse("(10 - (2 + 3)) × 2");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test multiple parenthesized subexpressions: (1 + 2) + (3 + 4) = 3 + 7 = 10
TEST_F(ParserTest, ParseMultipleParentheses) {
    Continuation* k = parser->parse("(1 + 2) + (3 + 4)");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test parentheses with division: (12 ÷ 2) ÷ 3 = 6 ÷ 3 = 2
TEST_F(ParserTest, ParseParenthesesDivision) {
    Continuation* k = parser->parse("(12 ÷ 2) ÷ 3");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();

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

    machine->push_kont(k);
    Value* result = machine->execute();
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

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

// Test undefined variable error
TEST_F(ParserTest, ParseUndefinedVariable) {
    Continuation* k = parser->parse("undefined");
    ASSERT_NE(k, nullptr);

    // Phase 1: Now throws exception instead of returning nullptr
    EXPECT_THROW({
        machine->push_kont(k);
        machine->execute();
    }, APLError);
}

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

// Test: Reduce without axis has nullptr axis_cont
TEST_F(ParserTest, ReduceWithoutAxisHasNullAxis) {
    Continuation* k = parser->parse("+/");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(k);
    ASSERT_NE(derived, nullptr);
    EXPECT_EQ(derived->axis_cont, nullptr) << "+/ should have no axis";
}

// Test: Reduce with axis [1] has non-null axis_cont
TEST_F(ParserTest, ReduceWithAxisHasAxisCont) {
    Continuation* k = parser->parse("+/[1]");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(k);
    ASSERT_NE(derived, nullptr) << "+/[1] should create DerivedOperatorK";
    EXPECT_STREQ(derived->op_name, "/");
    ASSERT_NE(derived->axis_cont, nullptr) << "+/[1] should have axis continuation";

    // The axis should be a literal 1
    LiteralK* axis_lit = dynamic_cast<LiteralK*>(derived->axis_cont);
    ASSERT_NE(axis_lit, nullptr) << "Axis should be a literal";
    EXPECT_DOUBLE_EQ(axis_lit->literal_value, 1.0);
}

// Test: Reduce with axis expression [1+1]
TEST_F(ParserTest, ReduceWithAxisExpression) {
    Continuation* k = parser->parse("+/[1+1]");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(k);
    ASSERT_NE(derived, nullptr);
    EXPECT_STREQ(derived->op_name, "/");
    ASSERT_NE(derived->axis_cont, nullptr) << "+/[1+1] should have axis continuation";
    // Axis is an expression, not just a literal
}

// Test: Scan with axis
TEST_F(ParserTest, ScanWithAxisHasAxisCont) {
    Continuation* k = parser->parse("+\\[2]");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(k);
    ASSERT_NE(derived, nullptr) << "+\\[2] should create DerivedOperatorK";
    EXPECT_STREQ(derived->op_name, "\\");
    ASSERT_NE(derived->axis_cont, nullptr) << "+\\[2] should have axis continuation";
}

// Test: Reduce-first with axis
TEST_F(ParserTest, ReduceFirstWithAxisHasAxisCont) {
    Continuation* k = parser->parse("+⌿[1]");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(k);
    ASSERT_NE(derived, nullptr);
    EXPECT_STREQ(derived->op_name, "⌿");
    ASSERT_NE(derived->axis_cont, nullptr);
}

// Test: N-wise reduction parsing "2 +/ 1 2 3 4 5"
// This should create a JuxtaposeK with derived operator in the middle
TEST_F(ParserTest, NwiseReductionParsesSuccessfully) {
    Continuation* k = parser->parse("2 +/ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    // Should parse as a juxtaposition
    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "2 +/ 1 2 3 4 5 should create JuxtaposeK";

    // The left should be 2
    LiteralK* left_lit = dynamic_cast<LiteralK*>(jux->left);
    ASSERT_NE(left_lit, nullptr) << "Left should be LiteralK(2)";
    EXPECT_DOUBLE_EQ(left_lit->literal_value, 2.0);

    // The right should be a JuxtaposeK for "+/ 1 2 3 4 5"
    JuxtaposeK* inner_jux = dynamic_cast<JuxtaposeK*>(jux->right);
    ASSERT_NE(inner_jux, nullptr) << "Right should be inner JuxtaposeK";

    // The inner left should be the derived operator +/
    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(inner_jux->left);
    ASSERT_NE(derived, nullptr) << "Inner left should be DerivedOperatorK";
    EXPECT_STREQ(derived->op_name, "/");
}

// Test: N-wise reduction actually evaluates correctly
TEST_F(ParserTest, NwiseReductionEvaluates) {
    Continuation* k = parser->parse("2 +/ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);  // 5 - 2 + 1 = 4 windows
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);   // 1+2
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 5.0);   // 2+3
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 7.0);   // 3+4
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 9.0);   // 4+5
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
    machine->push_kont(k);
    Value* result = machine->execute();
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
    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// String literal parsing tests
// ============================================================================

TEST_F(ParserTest, StringLiteralParses) {
    Continuation* k = parser->parse("'hello'");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "hello");
}

TEST_F(ParserTest, EmptyStringParses) {
    Continuation* k = parser->parse("''");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");
}

TEST_F(ParserTest, StringAsArgument) {
    // String should work as an argument (is_basic_value)
    // Just parse, don't need to eval since ⍎ isn't about parsing
    Continuation* k = parser->parse("⍎'1'");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
