// Evaluation tests - tests that APL expressions evaluate correctly
// These tests verify primitives, operators, and expressions through
// the full parse → continuation → execute pipeline.

#include <gtest/gtest.h>
#include "parser.h"
#include "machine.h"
#include "continuation.h"

using namespace apl;

class EvalTest : public ::testing::Test {
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
// Monadic Operator Tests
// ============================================================================

// Test simple negation: -5 = -5
TEST_F(EvalTest, ParseMonadicNegate) {
    Continuation* k = parser->parse("-5");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

// Test negation with expression: -(2 + 3) = -5
TEST_F(EvalTest, ParseMonadicNegateExpression) {
    Continuation* k = parser->parse("-(2 + 3)");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

// Test reciprocal: ÷4 = 0.25
TEST_F(EvalTest, ParseMonadicReciprocal) {
    Continuation* k = parser->parse("÷4");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.25);
}

// Test exponential: *0 = 1 (e^0)
TEST_F(EvalTest, ParseMonadicExponential) {
    Continuation* k = parser->parse("*0");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// Test exponential: *1 = e
TEST_F(EvalTest, ParseMonadicExponentialE) {
    Continuation* k = parser->parse("*1");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 2.71828, 0.0001);
}

// Test identity/conjugate: +5 = 5
TEST_F(EvalTest, ParseMonadicIdentity) {
    Continuation* k = parser->parse("+5");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test signum: ×5 = 1, ×(-5) = -1, ×0 = 0
TEST_F(EvalTest, ParseMonadicSignum) {
    Continuation* k1 = parser->parse("×5");
    ASSERT_NE(k1, nullptr);
    machine->push_kont(k1);
    Value* r1 = machine->execute();
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    Continuation* k2 = parser->parse("×(-5)");
    ASSERT_NE(k2, nullptr);
    machine->push_kont(k2);
    Value* r2 = machine->execute();
    EXPECT_DOUBLE_EQ(r2->as_scalar(), -1.0);

    Continuation* k3 = parser->parse("×0");
    ASSERT_NE(k3, nullptr);
    machine->push_kont(k3);
    Value* r3 = machine->execute();
    EXPECT_DOUBLE_EQ(r3->as_scalar(), 0.0);
}

// Test monadic with dyadic: 3 - -5 = 3 - (-5) = 8
TEST_F(EvalTest, ParseMonadicInDyadicContext) {
    Continuation* k = parser->parse("3 - -5");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// Test double negation: --5 = 5
TEST_F(EvalTest, ParseDoubleNegation) {
    Continuation* k = parser->parse("--5");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// ========== Assignment Tests ==========

// Test basic assignment: x ← 5
TEST_F(EvalTest, ParseBasicAssignment) {
    Continuation* k = parser->parse("x ← 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, ParseAssignmentWithExpression) {
    Continuation* k = parser->parse("y ← 3 + 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

    // Check environment
    Value* y = machine->env->lookup("y");
    ASSERT_NE(y, nullptr);
    EXPECT_DOUBLE_EQ(y->as_scalar(), 7.0);
}

// Test variable lookup after assignment
TEST_F(EvalTest, AssignmentThenLookup) {
    // First assign
    Continuation* k1 = parser->parse("z ← 10");
    ASSERT_NE(k1, nullptr);
    machine->push_kont(k1);
    machine->execute();

    // Then use the variable
    Continuation* k2 = parser->parse("z");
    ASSERT_NE(k2, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k2);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test variable in expression after assignment: z + 5
TEST_F(EvalTest, AssignmentThenUseInExpression) {
    // Assign z ← 10
    Continuation* k1 = parser->parse("z ← 10");
    ASSERT_NE(k1, nullptr);
    machine->push_kont(k1);
    machine->execute();

    // Use z in expression
    Continuation* k2 = parser->parse("z + 5");
    ASSERT_NE(k2, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k2);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

// Test function assignment and monadic application: f ← +, then f 3
TEST_F(EvalTest, FunctionAssignmentMonadic) {
    // First, manually add + to a variable for testing
    // (We can't parse "f ← +" yet because + is a token, not a name)
    Value* plus_fn = machine->env->lookup("+");
    ASSERT_NE(plus_fn, nullptr) << "+ should be in global environment";
    machine->env->define("f", plus_fn);

    // Now parse "f 3" - should apply f (which is +) monadically to 3
    // Monadic + is identity, so result should be 3
    Continuation* k = parser->parse("f 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// Test function assignment and dyadic application: f ← +, then 2 f 3
TEST_F(EvalTest, FunctionAssignmentDyadic) {
    // Manually add + to variable f
    Value* plus_fn = machine->env->lookup("+");
    ASSERT_NE(plus_fn, nullptr) << "+ should be in global environment";
    machine->env->define("f", plus_fn);

    // Now parse "2 f 3" - should apply f (which is +) dyadically to 2 and 3
    // Dyadic + is addition, so result should be 5
    Continuation* k = parser->parse("2 f 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Array operation tests
TEST_F(EvalTest, IotaMonadic) {
    // Parse "⍳ 5" - should generate vector [1 2 3 4 5]
    Continuation* k = parser->parse("⍳ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1-based per ISO 13751 (⎕IO=1)
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 5.0);
}

TEST_F(EvalTest, RavelMonadic) {
    // Parse ", 1 2 3" - ravel of vector is identity
    Continuation* k = parser->parse(", 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(EvalTest, ShapeMonadic) {
    // Parse "⍴ 1 2 3" - should return shape [3]
    Continuation* k = parser->parse("⍴ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 1);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 3.0);
}

TEST_F(EvalTest, ReshapeDyadic) {
    // Parse "2 3 ⍴ ⍳ 6" - reshape iota(6) into 2x3 matrix
    // APL uses row-major order (ISO 13751 §8.2.1):
    //   1 2 3
    //   4 5 6
    Continuation* k = parser->parse("2 3 ⍴ ⍳ 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(result->is_vector());  // Should be a 2D matrix
    EXPECT_FALSE(result->is_scalar());

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    // Row 0: 1, 2, 3 (1-based per ISO 13751)
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 3.0);
    // Row 1: 4, 5, 6
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 6.0);
}

TEST_F(EvalTest, CatenateDyadic) {
    // Parse "1 2 , 3 4" - concatenate two vectors
    Continuation* k = parser->parse("1 2 , 3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
}

TEST_F(EvalTest, EqualDyadicTrue) {
    // Parse "5 = 5" - equality test (true)
    Continuation* k = parser->parse("5 = 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 1 = true
}

TEST_F(EvalTest, EqualDyadicFalse) {
    // Parse "5 = 3" - equality test (false)
    Continuation* k = parser->parse("5 = 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 0 = false
}

TEST_F(EvalTest, EqualDyadicVector) {
    // Parse "1 2 3 = 1 5 3" - element-wise equality
    Continuation* k = parser->parse("1 2 3 = 1 5 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, ParseMonadicDfn) {
    // {⍵×2} should create a closure
    Continuation* k = parser->parse("{⍵×2}");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

// Test applying a monadic dfn
TEST_F(EvalTest, ApplyMonadicDfn) {
    // ({⍵×2} 5) should evaluate to 10
    // Note: Need to use dyadic syntax since we're applying the dfn to an argument
    Continuation* k = parser->parse("5 {⍵×2} 0");  // Using a dummy left arg for now

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    // This will fail until we implement monadic application properly
    // For now, just test that it parses
}

// Test parsing a dyadic dfn
TEST_F(EvalTest, ParseDyadicDfn) {
    // {⍺+⍵} should create a closure
    Continuation* k = parser->parse("{⍺+⍵}");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

// Test dfn that just returns omega
TEST_F(EvalTest, DfnJustOmega) {
    // 0 {⍵} 5 should return 5
    Continuation* k = parser->parse("0 {⍵} 5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test applying a dyadic dfn
TEST_F(EvalTest, ApplyDyadicDfn) {
    // 3 {⍺+⍵} 5 should evaluate to 8
    Continuation* k = parser->parse("3 {⍺+⍵} 5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR) << "Result should be SCALAR, got tag=" << static_cast<int>(result->tag);

    if (result->tag == ValueType::SCALAR) {
        double val = result->as_scalar();
        EXPECT_DOUBLE_EQ(val, 8.0) << "Expected 8.0, got " << val;
    }
}

// Test assigning a dfn to a variable
TEST_F(EvalTest, AssignDfn) {
    // square ← {⍵×⍵}
    Continuation* k = parser->parse("square ← {⍵×⍵}");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    // Check that the variable was assigned
    Value* square = machine->env->lookup("square");
    ASSERT_NE(square, nullptr);
    EXPECT_EQ(square->tag, ValueType::CLOSURE);
}

// Test dfn with nested expressions
TEST_F(EvalTest, DfnNestedExpression) {
    // {(⍺×2)+(⍵×3)} should parse correctly
    Continuation* k = parser->parse("{(⍺×2)+(⍵×3)}");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

// Test applying nested dfn
TEST_F(EvalTest, ApplyNestedDfn) {
    // 4 {(⍺×2)+(⍵×3)} 5 should evaluate to (4×2)+(5×3) = 8+15 = 23
    Continuation* k = parser->parse("4 {(⍺×2)+(⍵×3)} 5");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 23.0);
}

// Test monadic dfn application (direct, without left argument)
TEST_F(EvalTest, DfnMonadicDirect) {
    // {⍵+1}5 should evaluate to 6
    Continuation* k = parser->parse("{⍵+1}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// Test named dfn called monadically
TEST_F(EvalTest, DfnNamedMonadic) {
    // F←{⍵×2} ⋄ F 5 should return 10
    Continuation* k = parser->parse("F←{⍵×2} ⋄ F 5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test dfn with local assignment
TEST_F(EvalTest, DfnLocalAssignment) {
    // {x←5 ⋄ x+⍵}3 should return 8
    Continuation* k = parser->parse("{x←5 ⋄ x+⍵}3");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// Test dfn returning vector
TEST_F(EvalTest, DfnReturnsVector) {
    // {⍵ ⍵ ⍵}5 should return 5 5 5
    Continuation* k = parser->parse("{⍵ ⍵ ⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    ASSERT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 5.0);
}

// Test dfn with vector argument
TEST_F(EvalTest, DfnVectorArgument) {
    // {+/⍵}1 2 3 should return 6
    Continuation* k = parser->parse("{+/⍵}1 2 3");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// Test guard with true condition
TEST_F(EvalTest, DfnGuardTrue) {
    // {⍵>0: ⍵}5 should return 5
    Continuation* k = parser->parse("{⍵>0: ⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test guard with false condition followed by default
TEST_F(EvalTest, DfnGuardFalseWithDefault) {
    // {⍵>0: ⍵ ⋄ 0}¯5 should return 0
    Continuation* k = parser->parse("{⍵>0: ⍵ ⋄ 0}¯5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test multiple guards (first matching wins)
TEST_F(EvalTest, DfnMultipleGuards) {
    // {⍵<0: ¯1 ⋄ ⍵=0: 0 ⋄ 1}5 should return 1 (positive)
    Continuation* k = parser->parse("{⍵<0: ¯1 ⋄ ⍵=0: 0 ⋄ 1}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// Test multiple guards - second condition matches
TEST_F(EvalTest, DfnMultipleGuardsSecond) {
    // {⍵<0: ¯1 ⋄ ⍵=0: 0 ⋄ 1}0 should return 0
    Continuation* k = parser->parse("{⍵<0: ¯1 ⋄ ⍵=0: 0 ⋄ 1}0");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test multiple guards - first condition matches
TEST_F(EvalTest, DfnMultipleGuardsFirst) {
    // {⍵<0: ¯1 ⋄ ⍵=0: 0 ⋄ 1}¯5 should return ¯1
    Continuation* k = parser->parse("{⍵<0: ¯1 ⋄ ⍵=0: 0 ⋄ 1}¯5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -1.0);
}

// Test recursive dfn with ∇ (factorial)
TEST_F(EvalTest, DfnRecursiveFactorial) {
    // {⍵≤1: 1 ⋄ ⍵×∇ ⍵-1}5 should return 120 (5!)
    Continuation* k = parser->parse("{⍵≤1: 1 ⋄ ⍵×∇ ⍵-1}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 120.0);
}

// Test recursive dfn with ∇ (fibonacci)
TEST_F(EvalTest, DfnRecursiveFibonacci) {
    // {⍵≤1: ⍵ ⋄ (∇ ⍵-1)+∇ ⍵-2}6 should return 8 (fib(6))
    Continuation* k = parser->parse("{⍵≤1: ⍵ ⋄ (∇ ⍵-1)+∇ ⍵-2}6");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// Test named recursive dfn
TEST_F(EvalTest, DfnNamedRecursive) {
    // fact←{⍵≤1: 1 ⋄ ⍵×∇ ⍵-1} ⋄ fact 5 should return 120
    Continuation* k = parser->parse("fact←{⍵≤1: 1 ⋄ ⍵×∇ ⍵-1} ⋄ fact 5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 120.0);
}

// Test dfn with alpha when called monadically (alpha should error or use default)
TEST_F(EvalTest, DfnAlphaMonadicError) {
    // {⍺+⍵}5 - calling a dfn that uses ⍺ monadically should error
    // because ⍺ is not defined when called without left argument
    Continuation* k = parser->parse("{⍺+⍵}5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    // Expect VALUE ERROR for undefined ⍺
    EXPECT_THROW(machine->execute(), APLError);
}

// Test nested dfn (dfn defined inside dfn)
TEST_F(EvalTest, DfnNested) {
    // {F←{⍵+1} ⋄ F ⍵}5 should return 6
    Continuation* k = parser->parse("{F←{⍵+1} ⋄ F ⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// Test dfn returning a dfn (higher-order)
TEST_F(EvalTest, DfnReturningDfn) {
    // adder←{⍵{⍺+⍵}} creates a function that adds ⍵
    // Not all APLs support this - test for graceful handling
    Continuation* k = parser->parse("adder←{{⍺+⍵}⍵}");

    // May or may not parse - test doesn't crash
    if (k != nullptr) {
        machine->push_kont(k);
        machine->execute();
    }
}

// Test dfn as reduce operand
TEST_F(EvalTest, DfnAsReduceOperand) {
    // {⍺+⍵}/1 2 3 4 should return 10
    Continuation* k = parser->parse("{⍺+⍵}/1 2 3 4");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test dfn as each operand
TEST_F(EvalTest, DfnAsEachOperand) {
    // {⍵×2}¨1 2 3 should return 2 4 6
    Value* result = machine->eval("{⍵×2}¨1 2 3");

    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    ASSERT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 6.0);
}

// Test alpha default value syntax
TEST_F(EvalTest, DfnAlphaDefault) {
    // {⍺←10 ⋄ ⍺+⍵}5 should return 15 (alpha defaults to 10)
    Continuation* k = parser->parse("{⍺←10 ⋄ ⍺+⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

// Test alpha default value overridden by caller
TEST_F(EvalTest, DfnAlphaDefaultOverride) {
    // 3{⍺←10 ⋄ ⍺+⍵}5 should return 8 (alpha=3 overrides default)
    Continuation* k = parser->parse("3{⍺←10 ⋄ ⍺+⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// Test dfn with matrix argument
TEST_F(EvalTest, DfnMatrixArgument) {
    // {⍴⍵}2 3⍴⍳6 should return 2 3
    Continuation* k = parser->parse("{⍴⍵}2 3⍴⍳6");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    ASSERT_EQ(result->size(), 2);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);
}

// Test dfn calling another named dfn
TEST_F(EvalTest, DfnCallsNamedDfn) {
    // double←{⍵×2} ⋄ {double ⍵}5 should return 10
    Continuation* k = parser->parse("double←{⍵×2} ⋄ {double ⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test empty dfn body
TEST_F(EvalTest, DfnEmpty) {
    // {} - empty dfn, behavior varies by implementation
    Continuation* k = parser->parse("{}5");

    // Should either error or return something predictable
    // Test that it doesn't crash
    if (k != nullptr) {
        machine->push_kont(k);
        machine->execute();
    }
}

// Test dfn with multiple local variables
TEST_F(EvalTest, DfnMultipleLocals) {
    // {a←1 ⋄ b←2 ⋄ c←3 ⋄ a+b+c+⍵}10 should return 16
    Continuation* k = parser->parse("{a←1 ⋄ b←2 ⋄ c←3 ⋄ a+b+c+⍵}10");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 16.0);
}

// Test dfn with scan operator
TEST_F(EvalTest, DfnAsScanOperand) {
    // {⍺+⍵}\1 2 3 4 should return 1 3 6 10
    Continuation* k = parser->parse("{⍺+⍵}\\1 2 3 4");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    ASSERT_EQ(result->size(), 4);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 10.0);
}

// Test dfn with outer product
TEST_F(EvalTest, DfnAsOuterProduct) {
    // 1 2∘.{⍺×⍵}3 4 should return 2x2 matrix: 3 4 / 6 8
    Continuation* k = parser->parse("1 2∘.{⍺×⍵}3 4");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 2);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 8.0);
}

// ============================================================================
// GC Integration Tests (Phase 5.2)
// ============================================================================

// Test that GC can run during parsing without breaking anything
TEST_F(EvalTest, GCDuringParsing) {
    // Force a GC before parsing
    machine->heap->collect(machine);

    size_t heap_size_before = machine->heap->total_size();

    // Parse a complex expression
    Continuation* k = parser->parse("((1+2)×(3+4))+(5×6)");
    ASSERT_NE(k, nullptr);

    // Force another GC after parsing
    machine->heap->collect(machine);

    // Execute the parsed continuation
    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 51.0);  // (3×7)+30 = 21+30 = 51

    // Verify heap is stable
    size_t heap_size_after = machine->heap->total_size();
    EXPECT_GT(heap_size_after, 0);  // Heap should have objects
}

// Test that parser can handle large expressions without leaking
TEST_F(EvalTest, LargeExpressionGC) {
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
    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1275.0);  // Sum 1 to 50 = 50×51/2 = 1275

    // Force GC - should not crash or corrupt heap
    machine->heap->collect(machine);

    // Verify heap is still functional after GC
    Continuation* k2 = parser->parse("1+2+3");
    ASSERT_NE(k2, nullptr);
    machine->push_kont(k2);
    Value* result2 = machine->execute();
    EXPECT_DOUBLE_EQ(result2->as_scalar(), 6.0);
}

// Test that parser doesn't leak continuations across multiple parses
TEST_F(EvalTest, ParserNoLeakAcrossParses) {
    // Parse and execute several expressions
    for (int i = 0; i < 10; i++) {
        Continuation* k = parser->parse("1+2+3+4+5");
        ASSERT_NE(k, nullptr);

        machine->push_kont(k);
    Value* result = machine->execute();
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
    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);

    size_t after_last = machine->heap->total_size();
    // Heap shouldn't grow dramatically for the same expression
    EXPECT_LT(after_last, before_last * 2);
}

// Test GC during complex nested parsing
TEST_F(EvalTest, GCWithNestedExpressions) {
    // Create deeply nested expression
    std::string expr = "(((((1+2)+3)+4)+5)+6)";

    Continuation* k = parser->parse(expr.c_str());
    ASSERT_NE(k, nullptr);

    // Force GC while continuation graph exists
    machine->heap->collect(machine);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 21.0);

    // GC after execution
    machine->heap->collect(machine);

    // Verify heap is still functional
    Continuation* k2 = parser->parse("10+20");
    ASSERT_NE(k2, nullptr);
    machine->push_kont(k2);
    Value* result2 = machine->execute();
    EXPECT_DOUBLE_EQ(result2->as_scalar(), 30.0);
}

TEST_F(EvalTest, StrandIsNotJuxtapose) {
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

TEST_F(EvalTest, MonadicNotJuxtapose) {
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

TEST_F(EvalTest, FunctionNameAndStrand) {
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

TEST_F(EvalTest, NameNameCreatesJuxtapose) {
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

TEST_F(EvalTest, NameNumberCreatesJuxtapose) {
    // "f 3" where f is a variable and 3 is a number should create JuxtaposeK
    // This is two separate tokens: TOK_NAME and TOK_NUMBER
    Value* plus_fn = machine->env->lookup("+");
    machine->env->define("f", plus_fn);

    Continuation* k = parser->parse("f 3");
    ASSERT_NE(k, nullptr);

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "f 3 should create JuxtaposeK (NAME followed by NUMBER)";
}

TEST_F(EvalTest, NumberNameCreatesJuxtapose) {
    // "2 f" should create JuxtaposeK (two separate tokens)
    Value* plus_fn = machine->env->lookup("+");
    machine->env->define("f", plus_fn);

    Continuation* k = parser->parse("2 f");
    ASSERT_NE(k, nullptr);

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "2 f should create JuxtaposeK (NUMBER followed by NAME)";
}

TEST_F(EvalTest, NumberNameNumberCreatesJuxtapose) {
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

TEST_F(EvalTest, OuterProductScalarParseStructure) {
    // "3 ∘.× 5" with single scalars (simpler than strand case)
    Continuation* k = parser->parse("3 ∘.× 5");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    EXPECT_NE(jux, nullptr) << "3 ∘.× 5 should have JuxtaposeK at top level";
}

TEST_F(EvalTest, OuterProductStrandParseStructure) {
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

TEST_F(EvalTest, OuterProductEvaluatesToMatrix) {
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

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::MATRIX) << "Result should be MATRIX, got tag " << static_cast<int>(result->tag);
}

// ============================================================================
// Derived Operator Tests (commute/duplicate)
// ============================================================================

TEST_F(EvalTest, DerivedOperatorCommuteParsing) {
    // "+⍨" should parse as a derived operator (DerivedOperatorK)
    Continuation* k = parser->parse("+⍨");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(k);
    ASSERT_NE(derived, nullptr) << "+⍨ should parse as DerivedOperatorK";
}

TEST_F(EvalTest, DerivedOperatorCommuteMonadic) {
    // "+⍨ 3" should be JuxtaposeK with derived operator on left
    Continuation* k = parser->parse("+⍨ 3");
    ASSERT_NE(k, nullptr) << "Parser error: " << parser->get_error();

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "+⍨ 3 should be JuxtaposeK";

    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(jux->left);
    EXPECT_NE(derived, nullptr) << "Left side should be DerivedOperatorK";
}

TEST_F(EvalTest, DerivedOperatorCommuteDyadic) {
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

TEST_F(EvalTest, NotEqualDyadicTrue) {
    Continuation* k = parser->parse("5 ≠ 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5≠3 is true
}

TEST_F(EvalTest, NotEqualDyadicFalse) {
    Continuation* k = parser->parse("5 ≠ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 5≠5 is false
}

TEST_F(EvalTest, NotEqualDyadicVector) {
    Continuation* k = parser->parse("1 2 3 ≠ 1 5 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);  // 1≠1 false
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);  // 2≠5 true
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);  // 3≠3 false
}

TEST_F(EvalTest, LessThanDyadicTrue) {
    Continuation* k = parser->parse("3 < 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 3<5 is true
}

TEST_F(EvalTest, LessThanDyadicFalse) {
    Continuation* k = parser->parse("5 < 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 5<3 is false
}

TEST_F(EvalTest, LessThanDyadicVector) {
    Continuation* k = parser->parse("1 5 3 < 2 3 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1<2 true
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 5<3 false
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);  // 3<3 false
}

TEST_F(EvalTest, GreaterThanDyadicTrue) {
    Continuation* k = parser->parse("5 > 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5>3 is true
}

TEST_F(EvalTest, GreaterThanDyadicFalse) {
    Continuation* k = parser->parse("3 > 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 3>5 is false
}

TEST_F(EvalTest, GreaterThanDyadicVector) {
    Continuation* k = parser->parse("5 2 3 > 3 4 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 5>3 true
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 2>4 false
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);  // 3>3 false
}

TEST_F(EvalTest, LessOrEqualDyadicLess) {
    Continuation* k = parser->parse("3 ≤ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 3≤5 is true
}

TEST_F(EvalTest, LessOrEqualDyadicEqual) {
    Continuation* k = parser->parse("5 ≤ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5≤5 is true
}

TEST_F(EvalTest, LessOrEqualDyadicFalse) {
    Continuation* k = parser->parse("7 ≤ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 7≤5 is false
}

TEST_F(EvalTest, LessOrEqualDyadicVector) {
    Continuation* k = parser->parse("1 5 3 ≤ 2 3 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1≤2 true
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 5≤3 false
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // 3≤3 true
}

TEST_F(EvalTest, GreaterOrEqualDyadicGreater) {
    Continuation* k = parser->parse("7 ≥ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 7≥5 is true
}

TEST_F(EvalTest, GreaterOrEqualDyadicEqual) {
    Continuation* k = parser->parse("5 ≥ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5≥5 is true
}

TEST_F(EvalTest, GreaterOrEqualDyadicFalse) {
    Continuation* k = parser->parse("3 ≥ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 3≥5 is false
}

TEST_F(EvalTest, GreaterOrEqualDyadicVector) {
    Continuation* k = parser->parse("5 2 3 ≥ 3 4 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 5≥3 true
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 2≥4 false
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // 3≥3 true
}

// Test comparisons in expressions
TEST_F(EvalTest, ComparisonInExpression) {
    // Count elements less than 3: +/ (⍳5) < 3
    // ⍳5 = 1 2 3 4 5
    // < 3 = 1 1 0 0 0
    // +/ = 2
    Continuation* k = parser->parse("+/ (⍳5) < 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// ============================================================================
// Min/Max Tests (⌈ ⌊)
// ============================================================================

TEST_F(EvalTest, CeilingMonadic) {
    Continuation* k = parser->parse("⌈ 3.2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

TEST_F(EvalTest, CeilingMonadicNegative) {
    Continuation* k = parser->parse("⌈ ¯3.2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -3.0);
}

TEST_F(EvalTest, FloorMonadic) {
    Continuation* k = parser->parse("⌊ 3.7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, FloorMonadicNegative) {
    Continuation* k = parser->parse("⌊ ¯3.2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -4.0);
}

TEST_F(EvalTest, MaximumDyadic) {
    Continuation* k = parser->parse("3 ⌈ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, MinimumDyadic) {
    Continuation* k = parser->parse("3 ⌊ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, MaximumDyadicVector) {
    Continuation* k = parser->parse("1 5 3 ⌈ 4 2 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 4.0);  // max(1,4)
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 5.0);  // max(5,2)
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 6.0);  // max(3,6)
}

TEST_F(EvalTest, MinimumDyadicVector) {
    Continuation* k = parser->parse("1 5 3 ⌊ 4 2 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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

TEST_F(EvalTest, AndDyadicTrue) {
    Continuation* k = parser->parse("1 ∧ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, AndDyadicFalse) {
    Continuation* k = parser->parse("1 ∧ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, OrDyadicTrue) {
    Continuation* k = parser->parse("0 ∨ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, OrDyadicFalse) {
    Continuation* k = parser->parse("0 ∨ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, NotMonadicTrue) {
    Continuation* k = parser->parse("~ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, NotMonadicFalse) {
    Continuation* k = parser->parse("~ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, NotMonadicVector) {
    Continuation* k = parser->parse("~ 1 0 1 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 1.0);
}

TEST_F(EvalTest, NandDyadic) {
    // 1 ⍲ 1 = 0 (not (1 and 1))
    Continuation* k = parser->parse("1 ⍲ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, NandDyadicTrue) {
    // 1 ⍲ 0 = 1 (not (1 and 0))
    Continuation* k = parser->parse("1 ⍲ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, NorDyadic) {
    // 0 ⍱ 0 = 1 (not (0 or 0))
    Continuation* k = parser->parse("0 ⍱ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, NorDyadicFalse) {
    // 0 ⍱ 1 = 0 (not (0 or 1))
    Continuation* k = parser->parse("0 ⍱ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, LogicalVectorAnd) {
    Continuation* k = parser->parse("1 0 1 1 ∧ 1 1 0 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1∧1
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 0∧1
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.0);  // 1∧0
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 1.0);  // 1∧1
}

TEST_F(EvalTest, LogicalVectorOr) {
    Continuation* k = parser->parse("1 0 0 1 ∨ 0 0 1 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, LogicalExpression) {
    // Count where both conditions are true
    // (⍳5) gives 1 2 3 4 5
    // > 1 gives 0 1 1 1 1
    // < 4 gives 1 1 1 0 0
    // ∧ gives 0 1 1 0 0
    // +/ gives 2
    Continuation* k = parser->parse("+/ ((⍳5) > 1) ∧ ((⍳5) < 4)");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// ============================================================================
// Comprehensive Basic Arithmetic Tests (+ - × ÷ *)
// ============================================================================

// Addition (+) comprehensive tests
TEST_F(EvalTest, AdditionDyadicVectors) {
    Continuation* k = parser->parse("1 2 3 + 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 9.0);
}

TEST_F(EvalTest, AdditionScalarExtensionLeft) {
    Continuation* k = parser->parse("10 + 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 13.0);
}

TEST_F(EvalTest, AdditionScalarExtensionRight) {
    Continuation* k = parser->parse("1 2 3 + 10");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 13.0);
}

TEST_F(EvalTest, IdentityMonadicVector) {
    // Monadic + is identity/conjugate
    Continuation* k = parser->parse("+ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

// Subtraction (-) comprehensive tests
TEST_F(EvalTest, SubtractionDyadicVectors) {
    Continuation* k = parser->parse("10 20 30 - 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 9.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 18.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 27.0);
}

TEST_F(EvalTest, SubtractionScalarExtension) {
    Continuation* k = parser->parse("100 - 10 20 30");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 90.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 80.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 70.0);
}

TEST_F(EvalTest, NegationMonadicVector) {
    Continuation* k = parser->parse("- 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), -3.0);
}

// Multiplication (×) comprehensive tests
TEST_F(EvalTest, MultiplicationDyadicVectors) {
    Continuation* k = parser->parse("2 3 4 × 5 6 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 18.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 28.0);
}

TEST_F(EvalTest, MultiplicationScalarExtension) {
    Continuation* k = parser->parse("10 × 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 20.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 30.0);
}

TEST_F(EvalTest, SignumMonadicVector) {
    // Monadic × is signum: -1, 0, or 1
    Continuation* k = parser->parse("× ¯5 0 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);
}

// Division (÷) comprehensive tests
TEST_F(EvalTest, DivisionDyadicVectors) {
    Continuation* k = parser->parse("10 20 30 ÷ 2 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 6.0);
}

TEST_F(EvalTest, DivisionScalarExtension) {
    Continuation* k = parser->parse("100 ÷ 2 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 50.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 25.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 20.0);
}

TEST_F(EvalTest, ReciprocalMonadicVector) {
    // Monadic ÷ is reciprocal
    Continuation* k = parser->parse("÷ 2 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.5);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.25);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 0.2);
}

// Power (*) comprehensive tests
TEST_F(EvalTest, PowerDyadicVectors) {
    Continuation* k = parser->parse("2 3 4 * 2 2 2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 9.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 16.0);
}

TEST_F(EvalTest, PowerScalarExtension) {
    Continuation* k = parser->parse("2 * 1 2 3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 8.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 16.0);
}

TEST_F(EvalTest, ExponentialMonadicVector) {
    // Monadic * is e^x
    Continuation* k = parser->parse("* 0 1 2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, TransposeMatrix) {
    // Create 2x3 matrix and transpose to 3x2
    Continuation* k = parser->parse("⍉ 2 3 ⍴ 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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

TEST_F(EvalTest, TransposeVector) {
    // Transpose of vector is itself
    Continuation* k = parser->parse("⍉ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

// Take (↑) tests
TEST_F(EvalTest, TakePositive) {
    // Take first 3 elements
    Continuation* k = parser->parse("3 ↑ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(EvalTest, TakeNegative) {
    // Take last 3 elements
    Continuation* k = parser->parse("¯3 ↑ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);
}

TEST_F(EvalTest, TakeOverextend) {
    // Take more than available - pads with zeros
    Continuation* k = parser->parse("5 ↑ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, DropPositive) {
    // Drop first 2 elements
    Continuation* k = parser->parse("2 ↓ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);
}

TEST_F(EvalTest, DropNegative) {
    // Drop last 2 elements
    Continuation* k = parser->parse("¯2 ↓ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(EvalTest, DropAll) {
    // Drop more than available - empty result
    Continuation* k = parser->parse("10 ↓ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 0);
}

// Iota (⍳) additional tests
TEST_F(EvalTest, IotaZero) {
    // ⍳0 gives empty vector
    Continuation* k = parser->parse("⍳ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 0);
}

TEST_F(EvalTest, IotaOne) {
    // ⍳1 gives [1]
    Continuation* k = parser->parse("⍳ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 1);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
}

// Shape (⍴) additional tests
TEST_F(EvalTest, ShapeScalar) {
    // Shape of scalar is empty vector
    Continuation* k = parser->parse("⍴ 42");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 0);
}

TEST_F(EvalTest, ShapeMatrix) {
    // Shape of 2x3 matrix is [2, 3]
    Continuation* k = parser->parse("⍴ 2 3 ⍴ ⍳ 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 2);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 3.0);
}

// Reshape (⍴) additional tests
TEST_F(EvalTest, ReshapeWithCycling) {
    // Reshape cycles data: 2 3 ⍴ 1 2 gives matrix with 1 2 1 2 1 2
    Continuation* k = parser->parse("2 3 ⍴ 1 2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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

TEST_F(EvalTest, ReshapeToVector) {
    // Single dimension reshape creates vector
    Continuation* k = parser->parse("6 ⍴ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, CatenateScalars) {
    // Catenate two scalars creates a 2-element vector
    Continuation* k = parser->parse("1 , 2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 2);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
}

TEST_F(EvalTest, CatenateVectorScalar) {
    // Catenate vector with scalar
    Continuation* k = parser->parse("1 2 3 , 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, RavelMatrix) {
    // Ravel 2x3 matrix to 6-element vector
    Continuation* k = parser->parse(", 2 3 ⍴ ⍳ 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 6);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1-based per ISO 13751 (⎕IO=1)
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(5, 0), 6.0);
}

TEST_F(EvalTest, RavelScalar) {
    // Ravel scalar creates 1-element vector
    Continuation* k = parser->parse(", 42");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, MagnitudeMonadic) {
    Continuation* k = parser->parse("| ¯5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, MagnitudePositive) {
    Continuation* k = parser->parse("| 3.5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.5);
}

TEST_F(EvalTest, MagnitudeVector) {
    Continuation* k = parser->parse("| ¯1 2 ¯3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, ResidueDyadic) {
    // 3|7 = 1 (7 mod 3)
    Continuation* k = parser->parse("3 | 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, ResidueZeroDivisor) {
    // 0|5 = 5 (special case)
    Continuation* k = parser->parse("0 | 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, ResidueVector) {
    // 3| 0 1 2 3 4 5 6 = 0 1 2 0 1 2 0
    Continuation* k = parser->parse("3 | 0 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, NaturalLogMonadic) {
    // ⍟ e ≈ 1
    Continuation* k = parser->parse("⍟ 2.718281828");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 1.0, 1e-6);
}

TEST_F(EvalTest, NaturalLogOne) {
    // ⍟ 1 = 0
    Continuation* k = parser->parse("⍟ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Logarithm (⍟) dyadic tests
TEST_F(EvalTest, LogarithmDyadic) {
    // 10 ⍟ 100 = 2 (log base 10 of 100)
    Continuation* k = parser->parse("10 ⍟ 100");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 2.0, 1e-10);
}

TEST_F(EvalTest, LogarithmBase2) {
    // 2 ⍟ 8 = 3 (log base 2 of 8)
    Continuation* k = parser->parse("2 ⍟ 8");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 3.0, 1e-10);
}

// Factorial (!) tests
TEST_F(EvalTest, FactorialMonadic) {
    // !5 = 120
    Continuation* k = parser->parse("! 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 120.0);
}

TEST_F(EvalTest, FactorialZero) {
    // !0 = 1
    Continuation* k = parser->parse("! 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, FactorialVector) {
    // ! 0 1 2 3 4 = 1 1 2 6 24
    Continuation* k = parser->parse("! 0 1 2 3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, BinomialDyadic) {
    // 2!5 = C(5,2) = 10
    Continuation* k = parser->parse("2 ! 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(EvalTest, BinomialZero) {
    // 0!n = 1 for any n
    Continuation* k = parser->parse("0 ! 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, BinomialSame) {
    // n!n = 1
    Continuation* k = parser->parse("5 ! 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, BinomialPascalsRow) {
    // 0 1 2 3 4 ! 4 = 1 4 6 4 1 (row 4 of Pascal's triangle)
    Continuation* k = parser->parse("(0 1 2 3 4) ! 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, ReverseVector) {
    Continuation* k = parser->parse("⌽ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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

TEST_F(EvalTest, ReverseScalar) {
    Continuation* k = parser->parse("⌽ 42");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(EvalTest, ReverseMatrix) {
    // Reverse last axis of matrix (reverse columns within each row)
    Continuation* k = parser->parse("⌽ 2 3 ⍴ 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, RotateVectorLeft) {
    // Positive rotation = left rotate
    Continuation* k = parser->parse("2 ⌽ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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

TEST_F(EvalTest, RotateVectorRight) {
    // Negative rotation = right rotate
    Continuation* k = parser->parse("¯2 ⌽ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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

TEST_F(EvalTest, RotateZero) {
    // Zero rotation = identity
    Continuation* k = parser->parse("0 ⌽ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

// Reverse First (⊖) monadic tests
TEST_F(EvalTest, ReverseFirstVector) {
    // For vectors, first axis is the only axis
    Continuation* k = parser->parse("⊖ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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

TEST_F(EvalTest, ReverseFirstMatrix) {
    // Reverse first axis of matrix (reverse rows)
    Continuation* k = parser->parse("⊖ 2 3 ⍴ 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, RotateFirstMatrix) {
    // Rotate rows of matrix
    Continuation* k = parser->parse("1 ⊖ 3 2 ⍴ 1 2 3 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
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
TEST_F(EvalTest, TallyVector) {
    Continuation* k = parser->parse("≢ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, TallyScalar) {
    // Tally of scalar is 1
    Continuation* k = parser->parse("≢ 42");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, TallyMatrix) {
    // Tally of matrix is number of rows
    Continuation* k = parser->parse("≢ 3 4 ⍴ ⍳12");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, TallyEmpty) {
    // Tally of empty vector is 0
    Continuation* k = parser->parse("≢ 0 ↑ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// ============================================================================
// Search Functions (⍳ dyadic, ∊)
// ============================================================================

TEST_F(EvalTest, IndexOfScalar) {
    // 1 2 3 4 5 ⍳ 3 → 3 (1-origin per ISO 13751)
    Continuation* k = parser->parse("1 2 3 4 5 ⍳ 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, IndexOfNotFound) {
    // 1 2 3 ⍳ 7 → 4 (1 + length of haystack, per ISO 13751)
    Continuation* k = parser->parse("1 2 3 ⍳ 7");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

TEST_F(EvalTest, IndexOfVector) {
    // 10 20 30 40 ⍳ 30 10 99 → 3 1 5 (1-origin per ISO 13751)
    Continuation* k = parser->parse("10 20 30 40 ⍳ 30 10 99");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 3.0);  // 30 at index 3 (1-origin)
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);  // 10 at index 1 (1-origin)
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);  // 99 not found → 5 (1+length)
}

TEST_F(EvalTest, IndexOfDuplicates) {
    // First occurrence is returned
    // 5 3 5 7 3 ⍳ 3 5 → 2 1 (1-origin per ISO 13751)
    Continuation* k = parser->parse("5 3 5 7 3 ⍳ 3 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 2.0);  // 3 first at index 2 (1-origin)
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);  // 5 first at index 1 (1-origin)
}

TEST_F(EvalTest, MemberOfScalarFound) {
    // 3 ∊ 1 2 3 4 5 → 1
    Continuation* k = parser->parse("3 ∊ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, MemberOfScalarNotFound) {
    // 7 ∊ 1 2 3 → 0
    Continuation* k = parser->parse("7 ∊ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, MemberOfVector) {
    // 1 5 3 7 ∊ 1 2 3 → 1 0 1 0
    Continuation* k = parser->parse("1 5 3 7 ∊ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1 in set
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);  // 5 not in set
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // 3 in set
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 0.0);  // 7 not in set
}

TEST_F(EvalTest, MemberOfWithIota) {
    // 5 7 2 ∊ ⍳10 → 1 1 1 (all found in 1..10)
    Continuation* k = parser->parse("5 7 2 ∊ ⍳10");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);
}

TEST_F(EvalTest, EnlistVector) {
    // ∊ 1 2 3 → 1 2 3 (identity for simple vector)
    Continuation* k = parser->parse("∊ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(EvalTest, EnlistScalar) {
    // ∊ 5 → 5 (1-element vector)
    Continuation* k = parser->parse("∊ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 1);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
}

TEST_F(EvalTest, EnlistMatrix) {
    // ∊ 2 3 ⍴ ⍳6 → 1 2 3 4 5 6 (flatten to vector)
    Continuation* k = parser->parse("∊ 2 3 ⍴ ⍳6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 6);
    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ((*vec)(i, 0), static_cast<double>(i + 1));
    }
}

TEST_F(EvalTest, EnlistEmpty) {
    // ∊ ⍬ → ⍬ (empty vector stays empty)
    Value* result = machine->eval("∊ ⍬");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(EvalTest, IndexOfWithArithmetic) {
    // Combined with arithmetic
    // (⍳5) ⍳ 2+1 → 3 (find 3 in 1 2 3 4 5, returns 1-based index per ISO 13751)
    Continuation* k = parser->parse("(⍳5) ⍳ 2+1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// ============================================================================
// Grade Functions (⍋ ⍒)
// ============================================================================

TEST_F(EvalTest, GradeUpVector) {
    // ⍋ 3 1 4 1 5 → 2 4 1 3 5 (indices for ascending sort, 1-origin per ISO 13751)
    Continuation* k = parser->parse("⍋ 3 1 4 1 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 2.0);  // index of first 1
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);  // index of second 1
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // index of 3
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 3.0);  // index of 4
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 5.0);  // index of 5
}

TEST_F(EvalTest, GradeDownVector) {
    // ⍒ 3 1 4 1 5 → 5 3 1 2 4 (indices for descending sort, 1-origin per ISO 13751)
    Continuation* k = parser->parse("⍒ 3 1 4 1 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);  // index of 5 (largest)
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 3.0);  // index of 4
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);  // index of 3
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 2.0);  // index of first 1
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 4.0);  // index of second 1
}

TEST_F(EvalTest, GradeUpScalar) {
    // ⍋ 5 → RANK ERROR (grade requires array per ISO 13751)
    EXPECT_THROW(machine->eval("⍋ 5"), APLError);
}

TEST_F(EvalTest, GradeDownScalar) {
    // ⍒ 5 → RANK ERROR (grade requires array per ISO 13751)
    EXPECT_THROW(machine->eval("⍒ 5"), APLError);
}

TEST_F(EvalTest, GradeUpWithIota) {
    // ⍋ ⍳5 → 1 2 3 4 5 (already sorted, returns 1-based indices)
    Continuation* k = parser->parse("⍋ ⍳5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*vec)(i, 0), static_cast<double>(i + 1));
    }
}

TEST_F(EvalTest, GradeDownWithIota) {
    // ⍒ ⍳5 → 5 4 3 2 1 (reverse order, returns 1-based indices)
    Continuation* k = parser->parse("⍒ ⍳5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*vec)(i, 0), static_cast<double>(5 - i));
    }
}

TEST_F(EvalTest, GradeNegativeValues) {
    // ⍋ ¯3 1 ¯2 0 → 1 3 4 2 (sorted: -3 -2 0 1, 1-origin)
    Continuation* k = parser->parse("⍋ ¯3 1 ¯2 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // index of -3 (smallest)
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 3.0);  // index of -2
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 4.0);  // index of 0
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 2.0);  // index of 1 (largest)
}

// ============================================================================
// Replicate Function (/)
// ============================================================================

TEST_F(EvalTest, ReplicateBasic) {
    // 2 0 3 / 1 2 3 → 1 1 3 3 3
    Continuation* k = parser->parse("2 0 3 / 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);  // 2+0+3 = 5
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 2 copies of 1
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);  // 0 copies of 2, 3 copies of 3
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 3.0);
}

TEST_F(EvalTest, ReplicateCompress) {
    // 1 0 1 / 4 5 6 → 4 6 (compression with boolean mask)
    Continuation* k = parser->parse("1 0 1 / 4 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 2);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 6.0);
}

TEST_F(EvalTest, ReplicateAllZero) {
    // 0 0 0 / 1 2 3 → empty vector
    Continuation* k = parser->parse("0 0 0 / 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 0);
}

TEST_F(EvalTest, ReplicateScalar) {
    // 3 / 5 → 5 5 5
    Continuation* k = parser->parse("3 / 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);
}

TEST_F(EvalTest, ReplicateWithIota) {
    // 1 2 3 / ⍳3 → 1 2 2 3 3 3
    Continuation* k = parser->parse("1 2 3 / ⍳3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 6);  // 1+2+3 = 6
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1 copy of 1
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);  // 2 copies of 2
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 3.0);  // 3 copies of 3
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(5, 0), 3.0);
}

TEST_F(EvalTest, ReduceStillWorks) {
    // Verify that +/ still works as reduce (not replicate)
    // +/ 1 2 3 4 → 10
    Continuation* k = parser->parse("+/ 1 2 3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(EvalTest, ReplicateVsReduceDistinction) {
    // Array / Array is replicate
    // Function / Array is reduce
    // Test both in sequence
    Continuation* k1 = parser->parse("×/ 1 2 3 4");  // reduce: 24
    ASSERT_NE(k1, nullptr);
    machine->push_kont(k1);
    Value* result1 = machine->execute();
    EXPECT_DOUBLE_EQ(result1->as_scalar(), 24.0);

    Continuation* k2 = parser->parse("2 2 2 2 / 1 2 3 4");  // replicate: 1 1 2 2 3 3 4 4
    ASSERT_NE(k2, nullptr);
    machine->push_kont(k2);
    Value* result2 = machine->execute();
    const Eigen::MatrixXd* vec = result2->as_matrix();
    EXPECT_EQ(vec->rows(), 8);
}

// ============================================================================
// Set Functions (∪ ~)
// ============================================================================

TEST_F(EvalTest, UniqueVector) {
    // ∪ 1 2 2 3 1 4 → 1 2 3 4
    Continuation* k = parser->parse("∪ 1 2 2 3 1 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
}

TEST_F(EvalTest, UniqueScalar) {
    // ∪ 5 → 5
    Continuation* k = parser->parse("∪ 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, UniqueWithIota) {
    // ∪ 1 0 2 0 3 0 → 1 0 2 3 (preserves order of first appearance)
    Continuation* k = parser->parse("∪ 1 0 2 0 3 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 3.0);
}

TEST_F(EvalTest, UnionBasic) {
    // 1 2 3 ∪ 3 4 5 → 1 2 3 4 5
    Continuation* k = parser->parse("1 2 3 ∪ 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 5.0);
}

TEST_F(EvalTest, UnionNoOverlap) {
    // 1 2 ∪ 3 4 → 1 2 3 4
    Continuation* k = parser->parse("1 2 ∪ 3 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
}

TEST_F(EvalTest, WithoutBasic) {
    // 1 2 3 4 5 ~ 2 4 → 1 3 5
    Continuation* k = parser->parse("1 2 3 4 5 ~ 2 4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);
}

TEST_F(EvalTest, WithoutNoMatch) {
    // 1 2 3 ~ 4 5 → 1 2 3
    Continuation* k = parser->parse("1 2 3 ~ 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
}

TEST_F(EvalTest, WithoutAllMatch) {
    // 1 2 3 ~ 1 2 3 → (empty)
    Continuation* k = parser->parse("1 2 3 ~ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 0);
}

TEST_F(EvalTest, MonadicNotStillWorks) {
    // Verify ~ still works as monadic not
    // ~ 0 1 0 1 → 1 0 1 0
    Continuation* k = parser->parse("~ 0 1 0 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 0.0);
}

TEST_F(EvalTest, SetFunctionsWithArithmetic) {
    // ∪ (⍳5) + 0 → 1 2 3 4 5 (unique of already unique)
    Continuation* k = parser->parse("∪ (⍳5) + 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
}

// ============================================================================
// First (↑ monadic) Tests
// ============================================================================

TEST_F(EvalTest, FirstOfVector) {
    // ↑ 10 20 30 → 10
    Continuation* k = parser->parse("↑ 10 20 30");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(EvalTest, FirstOfScalar) {
    // ↑ 42 → 42
    Continuation* k = parser->parse("↑ 42");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(EvalTest, FirstOfIota) {
    // ↑ ⍳5 → 1
    Continuation* k = parser->parse("↑ ⍳5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, FirstOfMatrix) {
    // ↑ 2 3 ⍴ ⍳6 → 1 2 3 (first row)
    Continuation* k = parser->parse("↑ 2 3 ⍴ ⍳6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(EvalTest, DyadicTakeStillWorks) {
    // 3 ↑ 1 2 3 4 5 → 1 2 3
    Continuation* k = parser->parse("3 ↑ 1 2 3 4 5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

// ============================================================================
// Circular Functions (○) Tests
// ============================================================================

TEST_F(EvalTest, PiTimesScalar) {
    // ○ 1 → π
    Continuation* k = parser->parse("○ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), M_PI, 1e-10);
}

TEST_F(EvalTest, PiTimesHalf) {
    // ○ 0.5 → π/2
    Continuation* k = parser->parse("○ 0.5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), M_PI / 2.0, 1e-10);
}

TEST_F(EvalTest, CircularSin) {
    // 1 ○ (○ 0.5) → sin(π/2) = 1
    Continuation* k = parser->parse("1 ○ (○ 0.5)");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 1.0, 1e-10);
}

TEST_F(EvalTest, CircularCos) {
    // 2 ○ 0 → cos(0) = 1
    Continuation* k = parser->parse("2 ○ 0");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 1.0, 1e-10);
}

TEST_F(EvalTest, CircularSqrt) {
    // 0 ○ 0.6 → sqrt(1-0.36) = 0.8
    Continuation* k = parser->parse("0 ○ 0.6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), 0.8, 1e-10);
}

TEST_F(EvalTest, CircularAtan) {
    // ¯3 ○ 1 → atan(1) = π/4
    Continuation* k = parser->parse("¯3 ○ 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), M_PI / 4.0, 1e-10);
}

TEST_F(EvalTest, PiTimesVector) {
    // ○ 0 0.5 1 2 → 0, π/2, π, 2π
    Continuation* k = parser->parse("○ 0 0.5 1 2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_NEAR((*vec)(0, 0), 0.0, 1e-10);
    EXPECT_NEAR((*vec)(1, 0), M_PI / 2.0, 1e-10);
    EXPECT_NEAR((*vec)(2, 0), M_PI, 1e-10);
    EXPECT_NEAR((*vec)(3, 0), M_PI * 2.0, 1e-10);
}

// ========================================================================
// Roll/Deal Tests (?)
// ========================================================================

TEST_F(EvalTest, RollBasic) {
    // ?6 returns random integer in [1,6] (1-based per ISO 13751)
    Continuation* k = parser->parse("?6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    double r = result->as_scalar();
    EXPECT_GE(r, 1.0);  // 1-based per ISO 13751 (⎕IO=1)
    EXPECT_LE(r, 6.0);
}

TEST_F(EvalTest, RollVector) {
    // ?3 3 3 returns vector of random integers in [1,3] (1-based per ISO 13751)
    Continuation* k = parser->parse("?3 3 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE((*vec)(i, 0), 1.0);  // 1-based per ISO 13751 (⎕IO=1)
        EXPECT_LE((*vec)(i, 0), 3.0);
    }
}

TEST_F(EvalTest, DealBasic) {
    // 3?10 returns 3 unique random integers from [1,10] (1-based per ISO 13751)
    Continuation* k = parser->parse("3?10");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);

    // All unique
    double v0 = (*vec)(0, 0);
    double v1 = (*vec)(1, 0);
    double v2 = (*vec)(2, 0);
    EXPECT_NE(v0, v1);
    EXPECT_NE(v0, v2);
    EXPECT_NE(v1, v2);
}

TEST_F(EvalTest, DealPermutation) {
    // 5?5 returns a permutation of 1 2 3 4 5
    Continuation* k = parser->parse("5?5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);

    // Should contain each of 1-5 exactly once
    double sum = 0;
    for (int i = 0; i < 5; ++i) {
        sum += (*vec)(i, 0);
    }
    EXPECT_DOUBLE_EQ(sum, 15.0);  // 1+2+3+4+5 = 15
}

// ========================================================================
// Expand Tests (\)
// ========================================================================

TEST_F(EvalTest, ExpandBasic) {
    // 1 0 1 1 \ 1 2 3 → 1 0 2 3
    Continuation* k = parser->parse("1 0 1 1 \\ 1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 3.0);
}

TEST_F(EvalTest, ExpandLeadingZeros) {
    // 0 0 1 1 \ 5 6 → 0 0 5 6
    Continuation* k = parser->parse("0 0 1 1 \\ 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 6.0);
}

TEST_F(EvalTest, ScanVsExpand) {
    // +\1 2 3 → 1 3 6 (scan with function)
    Continuation* k = parser->parse("+\\1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 6.0);
}

// ========================================================================
// Decode/Encode Tests (⊥ ⊤)
// ========================================================================

TEST_F(EvalTest, DecodeBinary) {
    // 2⊥1 0 1 1 → 11
    Continuation* k = parser->parse("2⊥1 0 1 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 11.0);
}

TEST_F(EvalTest, DecodeDecimal) {
    // 10⊥1 2 3 → 123
    Continuation* k = parser->parse("10⊥1 2 3");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 123.0);
}

TEST_F(EvalTest, DecodeMixedRadix) {
    // 24 60 60⊥1 30 45 → 5445
    Continuation* k = parser->parse("24 60 60⊥1 30 45");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5445.0);
}

TEST_F(EvalTest, EncodeBinary) {
    // 2 2 2 2⊤11 → 1 0 1 1
    Continuation* k = parser->parse("2 2 2 2⊤11");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 1.0);
}

TEST_F(EvalTest, EncodeDecimal) {
    // 10 10 10⊤345 → 3 4 5
    Continuation* k = parser->parse("10 10 10⊤345");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 5.0);
}

TEST_F(EvalTest, EncodeMixedRadix) {
    // 24 60 60⊤5445 → 1 30 45
    Continuation* k = parser->parse("24 60 60⊤5445");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 30.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 45.0);
}

TEST_F(EvalTest, DecodeEncodeRoundtrip) {
    // 2⊥2 2 2 2⊤13 → 13 (encode then decode)
    Continuation* k = parser->parse("2⊥2 2 2 2⊤13");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 13.0);
}

// ============================================================================
// Matrix Inverse (⌹) monadic tests
// ============================================================================

TEST_F(EvalTest, MatrixInverseScalar) {
    // ⌹4 → 0.25
    Continuation* k = parser->parse("⌹4");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.25);
}

TEST_F(EvalTest, MatrixInverseMatrix) {
    // ⌹2 2⍴1 0 0 1 → identity (inverse of identity is identity)
    Continuation* k = parser->parse("⌹2 2⍴1 0 0 1");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(result->is_scalar());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 2);
    EXPECT_NEAR((*mat)(0, 0), 1.0, 1e-10);
    EXPECT_NEAR((*mat)(1, 1), 1.0, 1e-10);
}

// ============================================================================
// Matrix Divide (⌹) dyadic tests
// ============================================================================

TEST_F(EvalTest, MatrixDivideScalars) {
    // 6⌹2 → 3
    Continuation* k = parser->parse("6⌹2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, MatrixDivideVectorByScalar) {
    // (2 4 6)⌹2 → 1 2 3
    Continuation* k = parser->parse("(2 4 6)⌹2");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

// ============================================================================
// Dyadic Transpose (⍉) tests
// ============================================================================

TEST_F(EvalTest, DyadicTransposeIdentity) {
    // 0 1⍉2 3⍴⍳6 → same matrix
    Continuation* k = parser->parse("0 1⍉2 3⍴⍳6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(result->is_scalar());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
}

TEST_F(EvalTest, DyadicTransposeSwap) {
    // 1 0⍉2 3⍴⍳6 → 3x2 transpose
    Continuation* k = parser->parse("1 0⍉2 3⍴⍳6");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(result->is_scalar());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 2);
    // Original: 1 2 3 / 4 5 6 (1-based per ISO 13751), transposed: 1 4 / 2 5 / 3 6
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 1), 6.0);
}

TEST_F(EvalTest, DyadicTransposeEqualsMonadic) {
    // 1 0⍉M ≡ ⍉M for 2D matrix
    // Both should give same result
    Continuation* k1 = parser->parse("1 0⍉2 3⍴⍳6");
    ASSERT_NE(k1, nullptr) << "Parse error: " << parser->get_error();
    machine->push_kont(k1);
    Value* result1 = machine->execute();

    Continuation* k2 = parser->parse("⍉2 3⍴⍳6");
    ASSERT_NE(k2, nullptr) << "Parse error: " << parser->get_error();
    machine->push_kont(k2);
    Value* result2 = machine->execute();

    const Eigen::MatrixXd* mat1 = result1->as_matrix();
    const Eigen::MatrixXd* mat2 = result2->as_matrix();
    EXPECT_EQ(mat1->rows(), mat2->rows());
    EXPECT_EQ(mat1->cols(), mat2->cols());
    EXPECT_TRUE(mat1->isApprox(*mat2));
}

// ============================================================================
// Indexed Reference (A[I]) Tests
// ============================================================================

TEST_F(EvalTest, IndexedRefScalar) {
    // A←10 20 30 ⋄ A[2] → 20 (1-based indexing)
    machine->eval("A←10 20 30");
    Continuation* k = parser->parse("A[2]");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

TEST_F(EvalTest, IndexedRefVector) {
    // A←10 20 30 40 50 ⋄ A[2 4] → 20 40
    machine->eval("A←10 20 30 40 50");
    Continuation* k = parser->parse("A[2 4]");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 2);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 20.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 40.0);
}

TEST_F(EvalTest, IndexedRefWithExpression) {
    // A←⍳10 ⋄ A[1+2] → 3 (index 3, value 3)
    machine->eval("A←⍳10");
    Continuation* k = parser->parse("A[1+2]");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// ============================================================================
// Indexed Assignment (A[I]←V) Tests
// ============================================================================

TEST_F(EvalTest, IndexedAssignScalar) {
    // A←1 2 3 ⋄ A[2]←99 ⋄ A → 1 99 3
    machine->eval("A←1 2 3");
    Continuation* k = parser->parse("A[2]←99");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    // Indexed assignment returns the assigned value
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 99.0);

    // Verify A was modified
    Value* a_val = machine->eval("A");
    ASSERT_NE(a_val, nullptr);
    EXPECT_TRUE(a_val->is_vector());
    const Eigen::MatrixXd* vec = a_val->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 99.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(EvalTest, IndexedAssignFirst) {
    // A←10 20 30 ⋄ A[1]←5 ⋄ A → 5 20 30
    machine->eval("A←10 20 30");
    machine->eval("A[1]←5");

    Value* a_val = machine->eval("A");
    ASSERT_NE(a_val, nullptr);
    const Eigen::MatrixXd* vec = a_val->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 20.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 30.0);
}

TEST_F(EvalTest, IndexedAssignLast) {
    // A←10 20 30 ⋄ A[3]←99 ⋄ A → 10 20 99
    machine->eval("A←10 20 30");
    machine->eval("A[3]←99");

    Value* a_val = machine->eval("A");
    ASSERT_NE(a_val, nullptr);
    const Eigen::MatrixXd* vec = a_val->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 20.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 99.0);
}

TEST_F(EvalTest, IndexedAssignWithIota) {
    // A←⍳5 ⋄ A[2]←100 (iota + simple index)
    machine->eval("A←⍳5");

    // Debug: Check what type is stored
    Value* stored = machine->env->lookup("A");
    ASSERT_NE(stored, nullptr) << "A should be defined after A←⍳5";
    EXPECT_TRUE(stored->is_array()) << "A should be array, got tag=" << static_cast<int>(stored->tag);

    machine->eval("A[2]←100");

    Value* result = machine->eval("A[2]");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 100.0);
}

TEST_F(EvalTest, IndexedAssignWithExpressionIndex) {
    // A←⍳5 ⋄ A[2+1]←100 ⋄ A[3] → 100
    machine->eval("A←⍳5");
    machine->eval("A[2+1]←100");

    Value* result = machine->eval("A[3]");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 100.0);
}

TEST_F(EvalTest, IndexedAssignWithExpressionValue) {
    // A←1 2 3 ⋄ A[2]←10×5 ⋄ A[2] → 50
    machine->eval("A←1 2 3");
    machine->eval("A[2]←10×5");

    Value* result = machine->eval("A[2]");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 50.0);
}

TEST_F(EvalTest, IndexedAssignReturnsValue) {
    // The return value of A[I]←V is V
    machine->eval("A←1 2 3");
    Continuation* k = parser->parse("A[1]←42");
    ASSERT_NE(k, nullptr);

    machine->push_kont(k);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(EvalTest, IndexedAssignMatrix) {
    // A←2 3⍴⍳6 ⋄ A[4]←99 ⋄ A[4] → 99 (linear index into matrix)
    machine->eval("A←2 3⍴⍳6");

    Value* stored = machine->env->lookup("A");
    ASSERT_NE(stored, nullptr) << "A should be defined";
    EXPECT_TRUE(stored->is_matrix()) << "A should be matrix, got tag=" << static_cast<int>(stored->tag);

    machine->eval("A[4]←99");

    Value* result = machine->eval("A[4]");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 99.0);
}

TEST_F(EvalTest, IndexedAssignChained) {
    // Multiple indexed assignments in sequence
    machine->eval("A←1 2 3 4 5");
    machine->eval("A[1]←10");
    machine->eval("A[3]←30");
    machine->eval("A[5]←50");

    Value* a_val = machine->eval("A");
    const Eigen::MatrixXd* vec = a_val->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 30.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 50.0);
}

// --- Phase 6: Assignment Edge Cases (ISO 13751) ---

TEST_F(EvalTest, IndexedAssignOutOfBounds) {
    // A←1 2 3 ⋄ A[10]←5 → INDEX ERROR
    machine->eval("A←1 2 3");
    EXPECT_THROW(machine->eval("A[10]←5"), APLError);
}

TEST_F(EvalTest, IndexedAssignZeroIndex) {
    // A←1 2 3 ⋄ A[0]←5 → INDEX ERROR (⎕IO=1)
    machine->eval("A←1 2 3");
    EXPECT_THROW(machine->eval("A[0]←5"), APLError);
}

TEST_F(EvalTest, IndexedAssignNegativeIndex) {
    // A←1 2 3 ⋄ A[¯1]←5 → INDEX ERROR
    machine->eval("A←1 2 3");
    EXPECT_THROW(machine->eval("A[¯1]←5"), APLError);
}

TEST_F(EvalTest, IndexRefOutOfBounds) {
    // A←1 2 3 ⋄ A[10] → INDEX ERROR
    machine->eval("A←1 2 3");
    EXPECT_THROW(machine->eval("A[10]"), APLError);
}

TEST_F(EvalTest, IndexRefZeroIndex) {
    // A←1 2 3 ⋄ A[0] → INDEX ERROR (⎕IO=1)
    machine->eval("A←1 2 3");
    EXPECT_THROW(machine->eval("A[0]"), APLError);
}

TEST_F(EvalTest, IndexedAssignVectorIndices) {
    // A←10 20 30 40 50 ⋄ A[2 4]←99 88 ⋄ A → 10 99 30 88 50
    machine->eval("A←10 20 30 40 50");
    machine->eval("A[2 4]←99 88");

    Value* a_val = machine->eval("A");
    ASSERT_NE(a_val, nullptr);
    const Eigen::MatrixXd* vec = a_val->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 99.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 30.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 88.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 50.0);
}

TEST_F(EvalTest, IndexedAssignScalarToMultiple) {
    // A←10 20 30 40 50 ⋄ A[2 4]←0 ⋄ A → 10 0 30 0 50 (scalar extends)
    machine->eval("A←10 20 30 40 50");
    machine->eval("A[2 4]←0");

    Value* a_val = machine->eval("A");
    ASSERT_NE(a_val, nullptr);
    const Eigen::MatrixXd* vec = a_val->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 30.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 50.0);
}

TEST_F(EvalTest, IndexedAssignEmptyIndex) {
    // A←1 2 3 ⋄ A[⍳0]←⍳0 ⋄ A → 1 2 3 (no-op assignment)
    machine->eval("A←1 2 3");
    machine->eval("A[⍳0]←⍳0");

    Value* a_val = machine->eval("A");
    ASSERT_NE(a_val, nullptr);
    const Eigen::MatrixXd* vec = a_val->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(EvalTest, IndexedRefEmptyIndex) {
    // A←1 2 3 ⋄ A[⍳0] → ⍳0 (empty result)
    machine->eval("A←1 2 3");
    Value* result = machine->eval("A[⍳0]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 0);
}

// ============================================================================
// GC Stress Tests
// ============================================================================

TEST_F(EvalTest, GCStressManyAllocations) {
    // Execute many operations to stress GC with allocations
    // Each iteration creates temporaries that should be collected
    for (int i = 0; i < 100; i++) {
        Value* result = machine->eval("⍳100");
        ASSERT_NE(result, nullptr);
        EXPECT_TRUE(result->is_vector());
    }
    // Should complete without running out of memory
}

TEST_F(EvalTest, GCStressDeepRecursion) {
    // Define a recursive function and call it with moderate depth
    machine->eval("fact←{⍵≤1: 1 ⋄ ⍵×∇ ⍵-1}");
    Value* result = machine->eval("fact 10");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3628800.0);  // 10!
}

TEST_F(EvalTest, GCStressLargeArrays) {
    // Create and operate on large arrays
    Value* result = machine->eval("+/⍳1000");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 500500.0);  // sum 1..1000

    // Matrix operations
    result = machine->eval("+/,10 10⍴⍳100");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5050.0);  // sum 1..100
}

TEST_F(EvalTest, GCStressNestedDfnCalls) {
    // Multiple function definitions and calls creating many environments
    machine->eval("F←{⍵+1}");
    machine->eval("G←{F F ⍵}");
    machine->eval("H←{G G ⍵}");
    Value* result = machine->eval("H 0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);  // 0+1+1+1+1
}

// ============================================================================
// Table Function (⍪) Tests
// ============================================================================

TEST_F(EvalTest, TableScalar) {
    // ⍪ 5 → 1×1 matrix
    Value* result = machine->eval("⍪ 5");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_FALSE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 1);
    EXPECT_EQ(mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 5.0);
}

TEST_F(EvalTest, TableVector) {
    // ⍪ ⍳4 → 4×1 matrix
    Value* result = machine->eval("⍪ ⍳4");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_FALSE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 4);
    EXPECT_EQ(mat->cols(), 1);
}

TEST_F(EvalTest, TableShapeScalar) {
    // ⍴⍪5 → 1 1
    Value* result = machine->eval("⍴⍪5");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* shape = result->as_matrix();
    EXPECT_EQ(shape->size(), 2);
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 1.0);
}

TEST_F(EvalTest, TableShapeVector) {
    // ⍴⍪⍳5 → 5 1
    Value* result = machine->eval("⍴⍪⍳5");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* shape = result->as_matrix();
    EXPECT_EQ(shape->size(), 2);
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 1.0);
}

// ============================================================================
// Catenate First (dyadic ⍪) Tests - ISO 13751 Section 8.3.2
// ============================================================================

TEST_F(EvalTest, CatenateFirstVectors) {
    // (1 2 3)⍪(4 5 6) → 2×3 matrix
    Value* result = machine->eval("(1 2 3)⍪(4 5 6)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 6.0);
}

TEST_F(EvalTest, CatenateFirstScalars) {
    // 1⍪2 → 2-element column vector
    Value* result = machine->eval("1⍪2");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
}

TEST_F(EvalTest, CatenateFirstMatrices) {
    // Stack two 2×3 matrices → 4×3 matrix
    Value* result = machine->eval("(2 3⍴⍳6)⍪(2 3⍴7 8 9 10 11 12)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 4);
    EXPECT_EQ(mat->cols(), 3);
    // First matrix: 1 2 3 / 4 5 6
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 6.0);
    // Second matrix: 7 8 9 / 10 11 12
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 7.0);
    EXPECT_DOUBLE_EQ((*mat)(3, 2), 12.0);
}

TEST_F(EvalTest, CatenateFirstShape) {
    // ⍴(1 2 3)⍪(4 5 6) → 2 3
    Value* result = machine->eval("⍴(1 2 3)⍪(4 5 6)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* shape = result->as_matrix();
    EXPECT_EQ(shape->size(), 2);
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 3.0);
}

// ============================================================================
// Depth (≡) Tests - ISO 13751 Section 8.2.5
// ============================================================================

TEST_F(EvalTest, DepthScalar) {
    // ≡ 5 → 0 (scalar has depth 0)
    Value* result = machine->eval("≡ 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, DepthVector) {
    // ≡ 1 2 3 → 1 (simple array has depth 1)
    Value* result = machine->eval("≡ 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, DepthMatrix) {
    // ≡ 2 3⍴⍳6 → 1 (simple array has depth 1)
    Value* result = machine->eval("≡ 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, DepthEmptyVector) {
    // ≡ ⍬ → 1 (empty vector has depth 1)
    Value* result = machine->eval("≡ ⍬");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, MatchDyadicBasic) {
    // Dyadic ≡ (match) - returns 1 if identical, 0 otherwise
    // ISO 13751 Section 10.2.53
    Value* result = machine->eval("1 2 3 ≡ 1 2 3");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // Identical arrays match

    result = machine->eval("1 2 3 ≡ 1 2 4");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // Different values don't match

    result = machine->eval("5 ≡ 5");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // Identical scalars match

    result = machine->eval("5 ≡ 1 2 3");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // Different shapes don't match
}

// ============================================================================
// Left Tack (⊣) and Right Tack (⊢) Tests - ISO 13751 Section 10.2.17-18
// ============================================================================

TEST_F(EvalTest, LeftTackDyadic) {
    // ⍺⊣⍵ returns ⍺ (left argument)
    Value* result = machine->eval("3⊣5");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, RightTackDyadic) {
    // ⍺⊢⍵ returns ⍵ (right argument)
    Value* result = machine->eval("3⊢5");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, LeftTackMonadic) {
    // ⊣⍵ returns ⍵ (identity)
    Value* result = machine->eval("⊣7");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(EvalTest, RightTackMonadic) {
    // ⊢⍵ returns ⍵ (identity)
    Value* result = machine->eval("⊢7");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(EvalTest, LeftTackWithVector) {
    // ⍺⊣⍵ returns ⍺ even with vector arguments
    Value* result = machine->eval("1 2 3 ⊣ 4 5 6");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
}

TEST_F(EvalTest, RightTackWithVector) {
    // ⍺⊢⍵ returns ⍵ even with vector arguments
    Value* result = machine->eval("1 2 3 ⊢ 4 5 6");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 6.0);
}

// ============================================================================
// Squad (⌷) Tests - ISO 13751 Section 10.2.5
// ============================================================================

TEST_F(EvalTest, SquadScalarIndex) {
    // I⌷V - scalar index into vector
    Value* result = machine->eval("2⌷10 20 30 40");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

TEST_F(EvalTest, SquadVectorIndex) {
    // I⌷V - vector index into vector
    Value* result = machine->eval("2 4⌷10 20 30 40 50");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 20.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 40.0);
}

TEST_F(EvalTest, SquadWithIota) {
    // (⍳n)⌷V - iota indices into vector
    Value* result = machine->eval("(⍳3)⌷10 20 30 40 50");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 20.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 30.0);
}

// ============================================================================
// Zilde (⍬) Tests - Empty Vector Literal
// ============================================================================

TEST_F(EvalTest, ZildeIsEmptyVector) {
    // ⍬ should be an empty vector
    Value* result = machine->eval("⍬");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(EvalTest, ZildeShape) {
    // ⍴⍬ → 0 (shape of empty vector is a 1-element vector containing 0)
    Value* result = machine->eval("⍴⍬");
    ASSERT_NE(result, nullptr);
    // Shape returns a vector; for empty vector the shape is [0]
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 0.0);
}

TEST_F(EvalTest, ZildeAssignment) {
    // x←⍬ then ⍴x → 0
    machine->eval("x←⍬");
    Value* result = machine->eval("⍴x");
    ASSERT_NE(result, nullptr);
    // Shape returns a vector; for empty vector the shape is [0]
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 0.0);
}

TEST_F(EvalTest, ZildeCatenate) {
    // 1 2 3,⍬ → 1 2 3 (catenate with empty)
    Value* result = machine->eval("1 2 3,⍬");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
}

TEST_F(EvalTest, ZildeCatenateReverse) {
    // ⍬,1 2 3 → 1 2 3 (catenate empty with vector)
    Value* result = machine->eval("⍬,1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

TEST_F(EvalTest, ZildeFormatRoundTrip) {
    // format_value outputs ⍬ for empty vectors
    // This test verifies the round-trip: ⍬ → parse → eval → format → ⍬
    Value* result = machine->eval("⍬");
    ASSERT_NE(result, nullptr);
    std::string formatted = format_value(result);
    EXPECT_EQ(formatted, "⍬");
}

// ============================================================================
// ISO 13751 Section 6: Syntax & Evaluation Compliance Tests
// ============================================================================

// Section 6.3.9: Assignment returns committed-value (the assigned value)
TEST_F(EvalTest, AssignmentReturnsValue) {
    // X←5 returns 5, so 1+X←5 should equal 6
    Value* result = machine->eval("1+X←5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(EvalTest, AssignmentChaining) {
    // Y←X←5 should set both X and Y to 5
    machine->eval("Y←X←5");

    Value* x = machine->eval("X");
    ASSERT_NE(x, nullptr);
    EXPECT_DOUBLE_EQ(x->as_scalar(), 5.0);

    Value* y = machine->eval("Y");
    ASSERT_NE(y, nullptr);
    EXPECT_DOUBLE_EQ(y->as_scalar(), 5.0);
}

TEST_F(EvalTest, AssignmentChainingThreeVars) {
    // Z←Y←X←42 should set all three to 42
    machine->eval("Z←Y←X←42");

    Value* x = machine->eval("X");
    Value* y = machine->eval("Y");
    Value* z = machine->eval("Z");

    EXPECT_DOUBLE_EQ(x->as_scalar(), 42.0);
    EXPECT_DOUBLE_EQ(y->as_scalar(), 42.0);
    EXPECT_DOUBLE_EQ(z->as_scalar(), 42.0);
}

TEST_F(EvalTest, AssignmentInExpression) {
    // (X←3)×(Y←4) should set X=3, Y=4, return 12
    Value* result = machine->eval("(X←3)×(Y←4)");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);

    Value* x = machine->eval("X");
    Value* y = machine->eval("Y");
    EXPECT_DOUBLE_EQ(x->as_scalar(), 3.0);
    EXPECT_DOUBLE_EQ(y->as_scalar(), 4.0);
}

// Section 6.3.10: Indexed assignment to undefined variable signals VALUE ERROR
TEST_F(EvalTest, IndexedAssignUndefinedVariable) {
    // UNDEFINED_VAR[1]←5 should signal VALUE ERROR
    EXPECT_THROW(machine->eval("UNDEFINED_VAR_XYZ[1]←5"), APLError);
}

// Section 6.3.11: Referencing undefined variable signals VALUE ERROR
TEST_F(EvalTest, UndefinedVariableError) {
    // Referencing an undefined variable should signal VALUE ERROR
    EXPECT_THROW(machine->eval("NEVER_DEFINED_VAR"), APLError);
}

// Section 6.1: Statement separator (⋄) allows multiple statements
TEST_F(EvalTest, StatementSeparatorBasic) {
    // X←1 ⋄ X+1 → 2
    Value* result = machine->eval("X←1 ⋄ X+1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(EvalTest, StatementSeparatorMultiple) {
    // X←1 ⋄ Y←2 ⋄ X+Y → 3
    Value* result = machine->eval("X←1 ⋄ Y←2 ⋄ X+Y");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, StatementSeparatorReturnsLast) {
    // 1 ⋄ 2 ⋄ 3 → 3 (returns last statement's value)
    Value* result = machine->eval("1 ⋄ 2 ⋄ 3");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// Main function

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
