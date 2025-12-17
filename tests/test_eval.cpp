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

TEST_F(EvalTest, LiteralStrandIsNotJuxtapose) {
    // "2 3" is a lexical strand (single TOK_NUMBER_VECTOR token)
    // It should create LiteralStrandK, NOT JuxtaposeK
    Continuation* k = parser->parse("2 3");
    ASSERT_NE(k, nullptr);

    LiteralStrandK* strand = dynamic_cast<LiteralStrandK*>(k);
    ASSERT_NE(strand, nullptr) << "2 3 is a literal strand, not juxtaposition";

    // Verify it's NOT juxtapose
    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    EXPECT_EQ(jux, nullptr) << "Literal strand should not be JuxtaposeK";
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

TEST_F(EvalTest, FunctionNameAndLiteralStrand) {
    // "× 5 6" is TWO tokens: TOK_TIMES ("×") and TOK_NUMBER_VECTOR ([5, 6])
    // In G2 grammar, "×" is an identifier, so this is juxtaposition
    Continuation* k = parser->parse("× 5 6");
    ASSERT_NE(k, nullptr);

    // In G2, this should be JuxtaposeK
    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr) << "× 5 6 should be JuxtaposeK in G2 grammar";

    // The right side should be a literal strand
    LiteralStrandK* strand = dynamic_cast<LiteralStrandK*>(jux->right);
    EXPECT_NE(strand, nullptr) << "Right operand should be literal strand [5, 6]";
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

TEST_F(EvalTest, OuterProductLiteralStrandParseStructure) {
    // Outer product "3 4 ∘.× 5 6" should parse as:
    // JuxtaposeK(LiteralStrandK(3,4), JuxtaposeK(DerivedOperatorK(×,"∘."), LiteralStrandK(5,6)))
    // This is: (3 4) ((∘.×) (5 6))
    Continuation* k = parser->parse("3 4 ∘.× 5 6");
    ASSERT_NE(k, nullptr);

    JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(k);
    ASSERT_NE(jux, nullptr);

    // Left side should be LiteralStrandK (left array)
    LiteralStrandK* left_strand = dynamic_cast<LiteralStrandK*>(jux->left);
    ASSERT_NE(left_strand, nullptr) << "Left side should be LiteralStrandK";

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
    // 1 2⍉2 3⍴⍳6 → same matrix (⎕IO=1)
    Continuation* k = parser->parse("1 2⍉2 3⍴⍳6");
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
    // 2 1⍉2 3⍴⍳6 → 3x2 transpose (⎕IO=1)
    Continuation* k = parser->parse("2 1⍉2 3⍴⍳6");
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
    // 2 1⍉M ≡ ⍉M for 2D matrix (⎕IO=1)
    // Both should give same result
    Continuation* k1 = parser->parse("2 1⍉2 3⍴⍳6");
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

// ============================================================================
// System Variables Tests
// ============================================================================

TEST_F(EvalTest, SysVarIORead) {
    // ⎕IO default is 1
    Value* result = machine->eval("⎕IO");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, SysVarPPRead) {
    // ⎕PP default is 10
    Value* result = machine->eval("⎕PP");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(EvalTest, SysVarIOAssign) {
    // ⎕IO←0 sets index origin to 0
    Value* result = machine->eval("⎕IO←0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
    EXPECT_EQ(machine->io, 0);
}

TEST_F(EvalTest, SysVarIOAffectsIota) {
    // With ⎕IO←0, iota should return 0 1 2 3 4
    machine->eval("⎕IO←0");
    Value* result = machine->eval("⍳5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 4.0);
}

TEST_F(EvalTest, SysVarPPAssign) {
    // ⎕PP←5 sets print precision to 5
    Value* result = machine->eval("⎕PP←5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
    EXPECT_EQ(machine->pp, 5);
}

TEST_F(EvalTest, SysVarIOInvalidValueError) {
    // ⎕IO←2 should error (only 0 or 1 allowed)
    EXPECT_THROW(machine->eval("⎕IO←2"), APLError);
}

TEST_F(EvalTest, SysVarPPInvalidValueError) {
    // ⎕PP←0 should error (must be 1-17)
    EXPECT_THROW(machine->eval("⎕PP←0"), APLError);
}

TEST_F(EvalTest, SysVarInExpression) {
    // System variables can be used in expressions
    Value* result = machine->eval("⎕IO + 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);  // 1 + 5
}

// ============================================================================
// Comparison Tolerance (⎕CT) Tests
// ============================================================================

TEST_F(EvalTest, SysVarCTRead) {
    // ⎕CT default is 0 (exact comparisons, Eigen fast path)
    Value* result = machine->eval("⎕CT");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, SysVarCTAssign) {
    // ⎕CT←0 sets comparison tolerance to 0
    Value* result = machine->eval("⎕CT←0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
    EXPECT_DOUBLE_EQ(machine->ct, 0.0);
}

TEST_F(EvalTest, SysVarCTAssignLarger) {
    // ⎕CT←0.1 sets comparison tolerance to 0.1
    Value* result = machine->eval("⎕CT←0.1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.1);
    EXPECT_DOUBLE_EQ(machine->ct, 0.1);
}

TEST_F(EvalTest, SysVarCTInvalidNegative) {
    // ⎕CT←¯1 should error (must be nonnegative)
    EXPECT_THROW(machine->eval("⎕CT←¯1"), APLError);
}

TEST_F(EvalTest, SysVarCTTolerantEquality) {
    // With large tolerance, nearly equal values should be equal
    machine->eval("⎕CT←0.1");
    Value* result = machine->eval("1.05 = 1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // Tolerantly equal
}

TEST_F(EvalTest, SysVarCTExactEquality) {
    // With CT=0, exact comparison only
    machine->eval("⎕CT←0");
    Value* result = machine->eval("1.0000000001 = 1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // Not exactly equal
}

TEST_F(EvalTest, SysVarCTTolerantLessThan) {
    // With large tolerance, tolerantly equal values are NOT less than
    machine->eval("⎕CT←0.1");
    Value* result = machine->eval("0.95 < 1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // Tolerantly equal, so not <
}

TEST_F(EvalTest, SysVarCTTolerantFloor) {
    // Floor with tolerance: 2.9999999999 should round to 3
    machine->eval("⎕CT←1e-9");
    Value* result = machine->eval("⌊2.9999999999");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, SysVarCTTolerantCeiling) {
    // Ceiling with tolerance: 3.0000000001 should round to 3
    machine->eval("⎕CT←1e-9");
    Value* result = machine->eval("⌈3.0000000001");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, SysVarCTZeroFloorExact) {
    // With CT=0, floor is exact
    machine->eval("⎕CT←0");
    Value* result = machine->eval("⌊2.9999999999");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// ============================================================================
// Random Link (⎕RL) Tests
// ============================================================================

TEST_F(EvalTest, SysVarRLRead) {
    // ⎕RL is seeded from system at startup (positive integer)
    Value* result = machine->eval("⎕RL");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_GT(result->as_scalar(), 0.0);  // Must be positive
}

TEST_F(EvalTest, SysVarRLAssign) {
    // ⎕RL←12345 sets random seed
    Value* result = machine->eval("⎕RL←12345");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12345.0);
    EXPECT_EQ(machine->rl, 12345u);
}

TEST_F(EvalTest, SysVarRLInvalidZero) {
    // ⎕RL←0 should error (must be positive)
    EXPECT_THROW(machine->eval("⎕RL←0"), APLError);
}

TEST_F(EvalTest, SysVarRLInvalidNegative) {
    // ⎕RL←¯1 should error (must be positive)
    EXPECT_THROW(machine->eval("⎕RL←¯1"), APLError);
}

TEST_F(EvalTest, SysVarRLInvalidNonInteger) {
    // ⎕RL←1.5 should error (must be integer)
    EXPECT_THROW(machine->eval("⎕RL←1.5"), APLError);
}

TEST_F(EvalTest, SysVarRLReproducibility) {
    // Same seed produces same sequence
    machine->eval("⎕RL←42");
    Value* r1 = machine->eval("?100");
    double first1 = r1->as_scalar();

    machine->eval("⎕RL←42");  // Reset to same seed
    Value* r2 = machine->eval("?100");
    double first2 = r2->as_scalar();

    EXPECT_DOUBLE_EQ(first1, first2);
}

TEST_F(EvalTest, SysVarRLDifferentSeeds) {
    // Different seeds produce different sequences (with high probability)
    machine->eval("⎕RL←1");
    Value* r1 = machine->eval("?1000000");
    double val1 = r1->as_scalar();

    machine->eval("⎕RL←2");
    Value* r2 = machine->eval("?1000000");
    double val2 = r2->as_scalar();

    EXPECT_NE(val1, val2);
}

TEST_F(EvalTest, SysVarRLDealReproducibility) {
    // Deal also uses ⎕RL for reproducibility
    machine->eval("⎕RL←123");
    Value* d1 = machine->eval("5?10");
    const Eigen::MatrixXd* v1 = d1->as_matrix();

    machine->eval("⎕RL←123");  // Reset to same seed
    Value* d2 = machine->eval("5?10");
    const Eigen::MatrixXd* v2 = d2->as_matrix();

    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*v1)(i, 0), (*v2)(i, 0));
    }
}

// ============================================================================
// ISO 13751 Section 5 - Defined Operations Tests
// ============================================================================

// Direction/Signum (ISO 13751 §5.2.5)
// "Direction of A: returns zero if A is zero, otherwise -1, 0, or 1"
TEST_F(EvalTest, DirectionPositive) {
    Value* result = machine->eval("×42");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, DirectionNegative) {
    Value* result = machine->eval("×¯42");
    EXPECT_DOUBLE_EQ(result->as_scalar(), -1.0);
}

TEST_F(EvalTest, DirectionZero) {
    Value* result = machine->eval("×0");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, DirectionSmallPositive) {
    // Very small positive number still has direction 1
    Value* result = machine->eval("×1E¯300");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, DirectionSmallNegative) {
    // Very small negative number still has direction -1
    Value* result = machine->eval("×¯1E¯300");
    EXPECT_DOUBLE_EQ(result->as_scalar(), -1.0);
}

// Tolerant Equality edge cases (ISO 13751 §5.2.5)
TEST_F(EvalTest, TolerantEqualitySameValue) {
    // "If A equals B, then Z is one" - exact equality always true
    machine->eval("⎕CT←0.1");
    Value* result = machine->eval("5=5");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, TolerantEqualityWithinTolerance) {
    // Values within CT×max(|A|,|B|) are equal
    machine->eval("⎕CT←0.01");
    Value* result = machine->eval("100=100.5");  // diff=0.5, tol=0.01×100=1
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, TolerantEqualityOutsideTolerance) {
    // Values outside tolerance are not equal
    machine->eval("⎕CT←0.001");
    Value* result = machine->eval("100=101");  // diff=1, tol=0.001×100=0.1
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, TolerantEqualityZeroTolerance) {
    // CT=0 means exact comparison only
    machine->eval("⎕CT←0");
    Value* r1 = machine->eval("1.0=1.0");
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    Value* r2 = machine->eval("1.0=1.0000000001");
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 0.0);
}

// Tolerant comparison affects relational operators (ISO 13751)
TEST_F(EvalTest, TolerantLessThanNotEqualMeansLess) {
    // With tolerance, tolerantly equal values are NOT less than
    machine->eval("⎕CT←0.01");
    Value* result = machine->eval("100<100.5");  // tolerantly equal
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, TolerantGreaterThanNotEqualMeansGreater) {
    // With tolerance, tolerantly equal values are NOT greater than
    machine->eval("⎕CT←0.01");
    Value* result = machine->eval("100.5>100");  // tolerantly equal
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, TolerantLessEqualIncludesTolerantEqual) {
    // ≤ includes tolerantly equal values
    machine->eval("⎕CT←0.01");
    Value* result = machine->eval("100≤100.5");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(EvalTest, TolerantGreaterEqualIncludesTolerantEqual) {
    // ≥ includes tolerantly equal values
    machine->eval("⎕CT←0.01");
    Value* result = machine->eval("100.5≥100");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}
// ============================================================================
// G2 Rule 1: fbn-term ::= fb-term fbn-term (Juxtaposition)
// Semantics: if type(x₁) = bas then x₂(x₁) else x₁(x₂)
// ============================================================================

// Test: Basic value as left operand (bas × v → v(bas))
TEST_F(EvalTest, JuxtapositionBasicLeft) {
    // "5 -" is invalid syntax (- needs right operand)
    // Instead test: value function value → middle applies to left
    // "5 + 3" with left-associative parsing: (5 +) 3
    // 5 is basic, so +(5) creates curried function, then apply to 3
    Value* result = machine->eval("5 + 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);  // 5 + 3
}

// Test: Function as left operand (v × bas → v(bas))
TEST_F(EvalTest, JuxtapositionFunctionLeft) {
    // "- 5" should parse as: - is not basic, so apply -(5)
    Value* result = machine->eval("- 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(EvalTest, JuxtapositionRightAssociative) {
    // G2 Grammar: "2 + 3 × 4" should parse right-associatively as: 2 (+ (3 (× 4)))
    // Evaluation (right-to-left in APL):
    // 3 × 4: 3 is basic, so (× 4) curries to a function that multiplies by 4
    // + (× 4): + is function, (× 4) is basic (curried fn), so (+ (× 4)) curries
    // 2 (+ (× 4)): applies the composition to 2
    // Result: 2 + (3 × 4) = 2 + 12 = 14
    Value* result = machine->eval("2 + 3 × 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test: Chain of function applications
TEST_F(EvalTest, JuxtapositionChain) {
    // "- - - 5" should parse left-to-right as ((- -) -) 5
    // But each "-" is a function, so we need to handle function composition
    // Actually, let's test something simpler first

    // "- + 5" with left-associative parsing:
    // (- +) 5
    // - is function, + is function, so - applies to + (composition? error?)
    // This gets complex - let me test the paper's example instead
}

// Test: Dyadic application via currying (g' transformation)
TEST_F(EvalTest, DyadicViaGPrime) {
    // "2 + 3" should work as:
    // Left-to-right: (2 +) 3
    // 2 is basic, so +(2) creates curried function +₂
    // Then +₂(3) = 2+3 = 5
    Value* result = machine->eval("2 + 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test: Multiple dyadic applications
TEST_F(EvalTest, MultipleDyadics) {
    // "2 + 3 × 4" with APL right-to-left evaluation:
    // 2 + (3 × 4)
    // = 2 + 12
    // = 14
    Value* result = machine->eval("2 + 3 × 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test: Right-to-left evaluation within expression
TEST_F(EvalTest, RightToLeftEval) {
    // APL evaluates right-to-left for dyadic operations
    // "2 × 3 + 4" should be 2 × (3 + 4) = 2 × 7 = 14
    Value* result = machine->eval("2 × 3 + 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// ============================================================================
// G2 Rule 2: fb-term ::= fb-term monadic-operator
// Semantics: x₂(x₁) - operator takes operand to left
// ============================================================================

TEST_F(EvalTest, MonadicOperatorLeft) {
    // "+/ 1 2 3" should parse as (+ /) (1 2 3)
    // / takes + to its left, creating reduce-with-plus
    // Then apply to vector 1 2 3 → 6
    Value* result = machine->eval("+/1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(EvalTest, MonadicOperatorChain) {
    // Test multiple monadic operators
    // "f o1 o2" should parse as (f o1) o2
    // Each operator takes the result to its left

    // Using real operators: "+ / /" would be reduce-reduce-plus
    // But that doesn't make semantic sense
    // Let's test with each operator: "+ ¨" (plus each)
    Value* result = machine->eval("+¨1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    // +¨ applies + monadically to each element (conjugate, which is identity for reals)
    EXPECT_EQ(result->rows(), 3);
}

// ============================================================================
// G2 Rule 3: fb-term ::= derived-operator fb
// Semantics: x₁(x₂) - derived operator on right applies to operand
// ============================================================================

// Diagnostic test: check parse tree for × 5 6
TEST_F(EvalTest, InnerProductParseCheck) {
    // Check what "× 5 6" parses to
    Continuation* k = machine->parser->parse("× 5 6");
    ASSERT_NE(k, nullptr);
    std::cerr << "Parse of '× 5 6': " << typeid(*k).name() << std::endl;

    if (auto* jux = dynamic_cast<JuxtaposeK*>(k)) {
        std::cerr << "  Left: " << typeid(*jux->left).name() << std::endl;
        std::cerr << "  Right: " << typeid(*jux->right).name() << std::endl;
        if (auto* jux2 = dynamic_cast<JuxtaposeK*>(jux->right)) {
            std::cerr << "    Right.Left: " << typeid(*jux2->left).name() << std::endl;
            std::cerr << "    Right.Right: " << typeid(*jux2->right).name() << std::endl;
        }
    }

    // Now check "(× 5 6)" with parens to force grouping
    Continuation* k2 = machine->parser->parse("(×) 5 6");
    ASSERT_NE(k2, nullptr);
    std::cerr << "Parse of '(×) 5 6': " << typeid(*k2).name() << std::endl;
}

TEST_F(EvalTest, DerivedOperatorRight) {
    // "+. × 3 4" should parse as (+.) (× 3 4)
    // Wait, that doesn't follow the grammar...
    //
    // Actually: "× +. 3 4"
    // Parses as: ((×) (+.)) (3 4)
    // × is fb-term, +. is derived-operator
    // Rule 4: fb-term dyadic-operator → derived-operator
    // So + is fb-term, . is dyadic-operator → +. is derived-operator
    // Then rule 3: derived-operator fb → (+.) ×
    // Hmm, this is getting complex. Let me test inner product directly.

    // First, let's see what the parser creates
    Continuation* k = machine->parser->parse("3 4 +.× 5 6");
    ASSERT_NE(k, nullptr) << "Parse error: " << machine->parser->get_error();

    std::cerr << "Parsed continuation type: " << typeid(*k).name() << std::endl;
    if (auto* jux = dynamic_cast<JuxtaposeK*>(k)) {
        std::cerr << "  Left: " << typeid(*jux->left).name() << std::endl;
        if (auto* derived = dynamic_cast<DerivedOperatorK*>(jux->left)) {
            std::cerr << "    DerivedOp.operand: " << typeid(*derived->operand_cont).name() << std::endl;
            std::cerr << "    DerivedOp.op_name: " << derived->op_name << std::endl;
            if (auto* jux_operand = dynamic_cast<JuxtaposeK*>(derived->operand_cont)) {
                std::cerr << "      Operand.Left: " << typeid(*jux_operand->left).name() << std::endl;
                std::cerr << "      Operand.Right: " << typeid(*jux_operand->right).name() << std::endl;
            }
        }
        std::cerr << "  Right: " << typeid(*jux->right).name() << std::endl;
        if (auto* jux_right = dynamic_cast<JuxtaposeK*>(jux->right)) {
            std::cerr << "    Right.Left: " << typeid(*jux_right->left).name() << std::endl;
            if (auto* derived_right = dynamic_cast<DerivedOperatorK*>(jux_right->left)) {
                std::cerr << "      Right.DerivedOp.operand: " << typeid(*derived_right->operand_cont).name() << std::endl;
                std::cerr << "      Right.DerivedOp.op_name: " << derived_right->op_name << std::endl;
            }
            std::cerr << "    Right.Right: " << typeid(*jux_right->right).name() << std::endl;
        }
    }

    Value* result = machine->eval("3 4 +.× 5 6");
    ASSERT_NE(result, nullptr);
    if (!result->is_scalar()) {
        std::cerr << "Result is not scalar, tag=" << static_cast<int>(result->tag) << std::endl;
        if (result->tag == ValueType::CURRIED_FN) {
            Value::CurriedFnData* cd = result->data.curried_fn;
            std::cerr << "CurriedFn: fn tag=" << static_cast<int>(cd->fn->tag)
                      << " curry_type=" << static_cast<int>(cd->curry_type) << std::endl;
        } else if (result->tag == ValueType::VECTOR) {
            std::cerr << "Vector size=" << result->size() << std::endl;
        } else if (result->tag == ValueType::MATRIX) {
            std::cerr << "Matrix rows=" << result->rows() << " cols=" << result->cols() << std::endl;
        }
    }
    EXPECT_TRUE(result->is_scalar());
    // 3×5 + 4×6 = 15 + 24 = 39
    EXPECT_DOUBLE_EQ(result->as_scalar(), 39.0);
}

// ============================================================================
// G2 Rule 4: derived-operator ::= fb-term dyadic-operator
// Semantics: x₂(x₁) - operator takes operand to left
// ============================================================================

TEST_F(EvalTest, DyadicOperatorCurrying) {
    // "+. ×" should parse as: + is fb-term, . is dyadic-operator
    // Creates derived operator (+.)
    // Then × is the right operand: (+.) ×
    // This creates the inner product operator +.×

    // Full expression: "3 4 +.× 5 6"
    // Already tested above in DerivedOperatorRight

    // Let's test outer product instead: "3 4 ∘.× 5 6"
    Value* result = machine->eval("3 4 ∘.× 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::MATRIX);
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
    // [3×5  3×6] = [15 18]
    // [4×5  4×6]   [20 24]
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0,0), 15.0);
    EXPECT_DOUBLE_EQ((*m)(0,1), 18.0);
    EXPECT_DOUBLE_EQ((*m)(1,0), 20.0);
    EXPECT_DOUBLE_EQ((*m)(1,1), 24.0);
}

// ============================================================================
// g' Transformation Tests (Section 4 of paper)
// g' = λx. λy. if null(y) then g₁(x)
//             else if bas(y) then g₂(x,y)
//             else y(g₁(x))
//
// The g' transformation is applied to functions that have both monadic and
// dyadic forms. When such a function f is applied to a value x, it creates
// a "curried" function g'(f,x) that waits to see what comes next:
//   - If nothing (null(y)): apply f monadically to x
//   - If a basic value (bas(y)): apply f dyadically with x as left arg
//   - If another function (y is function): apply f monadically to x,
//     then pass the result to y
// ============================================================================

// ---------------------------------------------------------------------------
// Case 1: null(y) - Monadic application at top level
// ---------------------------------------------------------------------------

TEST_F(EvalTest, GPrimeNullCase_Plus) {
    // "+ 3" at top level: + creates G_PRIME(+, 3), y is null → monadic +
    // Monadic + is conjugate (identity for reals)
    Value* result = machine->eval("+ 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, GPrimeNullCase_Iota) {
    // "⍳5" at top level: ⍳ has both forms, creates G_PRIME(⍳, 5)
    // y is null → apply monadic ⍳ → 1 2 3 4 5
    Value* result = machine->eval("⍳5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(4, 0), 5.0);
}

TEST_F(EvalTest, GPrimeNullCase_Minus) {
    // "- 5" at top level: - has both forms, creates G_PRIME(-, 5)
    // y is null → apply monadic - → -5
    Value* result = machine->eval("- 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(EvalTest, GPrimeNullCase_Rho) {
    // "⍴ 1 2 3" at top level: monadic ⍴ gives shape
    Value* result = machine->eval("⍴ 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 1);  // Shape of 3-element vector is [3]
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);
}

// ---------------------------------------------------------------------------
// Case 2: bas(y) - Dyadic application when second arg is basic value
// ---------------------------------------------------------------------------

TEST_F(EvalTest, GPrimeBasicCase_Plus) {
    // "2 + 3" → +(2) creates G_PRIME, sees 3 (basic) → dyadic + → 5
    Value* result = machine->eval("2 + 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, GPrimeBasicCase_Iota) {
    // "1 2 3 ⍳ 2" → dyadic ⍳ (index-of): find 2 in vector 1 2 3 → index 2 (1-origin per ISO 13751)
    Value* result = machine->eval("1 2 3 ⍳ 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(EvalTest, GPrimeBasicCase_IotaVector) {
    // "1 2 3 ⍳ 3 1 5" → find indices of 3,1,5 in 1 2 3 → 3 1 4(not found) (1-origin per ISO 13751)
    Value* result = machine->eval("1 2 3 ⍳ 3 1 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);  // 3 is at index 3 (1-origin)
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // 1 is at index 1 (1-origin)
    EXPECT_DOUBLE_EQ((*m)(2, 0), 4.0);  // 5 not found, returns 1+length
}

TEST_F(EvalTest, GPrimeBasicCase_Minus) {
    // "10 - 3" → dyadic - (subtract) → 7
    Value* result = machine->eval("10 - 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(EvalTest, GPrimeBasicCase_Rho) {
    // "2 3 ⍴ 1 2 3 4 5 6" → dyadic ⍴ (reshape) → 2x3 matrix
    Value* result = machine->eval("2 3 ⍴ 1 2 3 4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
}

// ---------------------------------------------------------------------------
// Case 3: y is function - Composition: y(g₁(x))
// This is the critical case that was buggy before the fix!
// When the curried function sees another function, it should:
//   1. Apply its function monadically to its captured argument
//   2. Pass the result to the incoming function
// ---------------------------------------------------------------------------

TEST_F(EvalTest, GPrimeFunctionCase_IotaLessThan) {
    // "(⍳5) < 3" is the canonical test case
    // Parse: ((⍳ 5) <) 3
    // 1. ⍳ sees 5 (basic) → creates G_PRIME(⍳, 5)
    // 2. G_PRIME sees < (function) → apply ⍳ monadically: ⍳5 = 1 2 3 4 5
    //    Then < sees 1 2 3 4 5 (basic) → creates G_PRIME(<, 1 2 3 4 5)
    // 3. G_PRIME(<, 1 2 3 4 5) sees 3 (basic) → dyadic <: (1 2 3 4 5) < 3
    // Result: 1 1 0 0 0
    Value* result = machine->eval("(⍳5) < 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);  // 1 < 3 = true
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // 2 < 3 = true
    EXPECT_DOUBLE_EQ((*m)(2, 0), 0.0);  // 3 < 3 = false
    EXPECT_DOUBLE_EQ((*m)(3, 0), 0.0);  // 4 < 3 = false
    EXPECT_DOUBLE_EQ((*m)(4, 0), 0.0);  // 5 < 3 = false
}

TEST_F(EvalTest, GPrimeFunctionCase_IotaEquals) {
    // "(⍳5) = 2" → ⍳5 = 1 2 3 4 5, then (1 2 3 4 5) = 2 → 0 1 0 0 0
    Value* result = machine->eval("(⍳5) = 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);  // 1 = 2 → 0
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // 2 = 2 → 1
    EXPECT_DOUBLE_EQ((*m)(2, 0), 0.0);  // 3 = 2 → 0
    EXPECT_DOUBLE_EQ((*m)(3, 0), 0.0);  // 4 = 2 → 0
    EXPECT_DOUBLE_EQ((*m)(4, 0), 0.0);  // 5 = 2 → 0
}

TEST_F(EvalTest, GPrimeFunctionCase_NegateAdd) {
    // "(- 5) + 3" → negate 5 first, then add 3
    // - sees 5 (basic) → G_PRIME(-, 5)
    // G_PRIME sees + (function) → apply - monadically: -5
    // + sees -5 (basic) → G_PRIME(+, -5)
    // G_PRIME sees 3 (basic) → dyadic +: -5 + 3 = -2
    Value* result = machine->eval("(- 5) + 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -2.0);
}

TEST_F(EvalTest, GPrimeFunctionCase_IotaTimes) {
    // "(⍳4) × 2" → (1 2 3 4) × 2 → 2 4 6 8
    Value* result = machine->eval("(⍳4) × 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), 8.0);
}

TEST_F(EvalTest, GPrimeFunctionCase_ChainedComposition) {
    // "(- 5) + - 3" → complex case with multiple function compositions
    // Parse: (((- 5) +) -) 3
    // - 5 → G_PRIME(-, 5), sees + → monadic -5, + creates G_PRIME(+, -5)
    // G_PRIME(+, -5) sees - → apply monadic +(-5)=-5, - creates G_PRIME(-, -5)
    // G_PRIME(-, -5) sees 3 → dyadic -: -5 - 3 = -8
    Value* result = machine->eval("(- 5) + - 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -8.0);
}

TEST_F(EvalTest, GPrimeFunctionCase_IotaWithOperator) {
    // "(⍳5) +/ 1 2 3 4 5" - tests that iota resolves before operator expression
    // ⍳5 → 1 2 3 4 5, then... wait this is complex
    // Let's test simpler: "+/ ⍳5" should sum 1+2+3+4+5 = 15
    Value* result = machine->eval("+/ ⍳5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(EvalTest, GPrimeFunctionCase_IotaPlusIota) {
    // "(⍳3) + ⍳3" → (1 2 3) + (1 2 3) → 2 4 6
    // First ⍳3 creates G_PRIME, sees + (function), applies monadically
    // + creates G_PRIME(+, 1 2 3), sees ⍳ (function), applies monadically
    // But ⍳ 3 creates G_PRIME(⍳, 3), which at end resolves to 1 2 3
    // Then dyadic +: (1 2 3) + (1 2 3) = 2 4 6
    Value* result = machine->eval("(⍳3) + ⍳3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);
}

TEST_F(EvalTest, GPrimeFunctionCase_Member) {
    // "⍳5 ∊ 2 3 7" → first ⍳5 = 1 2 3 4 5, then membership test
    // (1 2 3 4 5) ∊ (2 3 7) → 0 1 1 0 0
    Value* result = machine->eval("(⍳5) ∊ 2 3 7");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);  // 1 not in {2,3,7}
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // 2 in {2,3,7}
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // 3 in {2,3,7}
    EXPECT_DOUBLE_EQ((*m)(3, 0), 0.0);  // 4 not in {2,3,7}
    EXPECT_DOUBLE_EQ((*m)(4, 0), 0.0);  // 5 not in {2,3,7}
}

// ---------------------------------------------------------------------------
// Combined/stress tests for g' transformation
// ---------------------------------------------------------------------------

TEST_F(EvalTest, GPrimeMixedExpression) {
    // "1 + (⍳3) × 2" → APL right-to-left: 1 + ((⍳3) × 2)
    // (⍳3) × 2 → (1 2 3) × 2 → 2 4 6
    // 1 + (2 4 6) → 3 5 7
    Value* result = machine->eval("1 + (⍳3) × 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 7.0);
}

TEST_F(EvalTest, GPrimeWithReduce) {
    // "+/ (⍳4) × 2" → ⍳4 = 1 2 3 4, × 2 → 2 4 6 8, +/ → 20
    Value* result = machine->eval("+/ (⍳4) × 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

TEST_F(EvalTest, GPrimeComparisonInExpression) {
    // Test the original failing case more thoroughly
    // "1 + (⍳5) < 3" → (⍳5) < 3 = 1 1 0 0 0, then 1 + that = 2 2 1 1 1
    Value* result = machine->eval("1 + (⍳5) < 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(4, 0), 1.0);
}

TEST_F(EvalTest, GPrimeLogicalExpression) {
    // "(⍳5) > 2 ∧ (⍳5) < 4" - should find 3 (values where 2 < x < 4)
    // ⍳5 = 1 2 3 4 5
    // (⍳5) > 2 = 0 0 1 1 1
    // (⍳5) < 4 = 1 1 1 0 0
    // Result: 0 0 1 0 0 (only value 3 satisfies both)
    Value* result = machine->eval("((⍳5) > 2) ∧ (⍳5) < 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(4, 0), 0.0);
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_F(EvalTest, NestedParentheses) {
    // Parentheses should force evaluation order
    // "(2 + 3) × 4" should be 5 × 4 = 20
    Value* result = machine->eval("(2 + 3) × 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

TEST_F(EvalTest, ComplexNesting) {
    // "((2 + 3) × 4) - 1" = 20 - 1 = 19
    Value* result = machine->eval("((2 + 3) × 4) - 1");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 19.0);
}

TEST_F(EvalTest, VectorOperations) {
    // "1 2 3 + 4 5 6" with vectors
    Value* result = machine->eval("1 2 3 + 4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0,0), 5.0);
    EXPECT_DOUBLE_EQ((*m)(1,0), 7.0);
    EXPECT_DOUBLE_EQ((*m)(2,0), 9.0);
}

TEST_F(EvalTest, ScalarVectorMixed) {
    // "5 + 1 2 3" should broadcast: 5+1=6, 5+2=7, 5+3=8
    Value* result = machine->eval("5 + 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0,0), 6.0);
    EXPECT_DOUBLE_EQ((*m)(1,0), 7.0);
    EXPECT_DOUBLE_EQ((*m)(2,0), 8.0);
}

TEST_F(EvalTest, ReductionWithOperators) {
    // "+/×/1 2 3 4" should parse as (+/) ((×/) (1 2 3 4))
    // ×/ 1 2 3 4 = 1×2×3×4 = 24
    // +/ 24 = 24 (reduction of single element)
    Value* result = machine->eval("+/×/1 2 3 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 24.0);
}

TEST_F(EvalTest, CommuteDuplicate) {
    // "2 +⍨ 3" should be 3 + 2 (commuted) = 5
    Value* result = machine->eval("2 +⍨ 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, EachOperator) {
    // "×¨ 1 2 3" should apply × monadically to each element
    // Monadic × is sign function: sign(1)=1, sign(2)=1, sign(3)=1
    Value* result = machine->eval("×¨ 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0,0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1,0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(2,0), 1.0);
}

TEST_F(EvalTest, LexicalStrandVsJuxtaposition) {
    // Test distinction between lexical strand and runtime juxtaposition
    // "1 2 3" is a LEXICAL STRAND (single TOK_NUMBER_VECTOR token) - creates vector [1,2,3]
    Value* strand = machine->eval("1 2 3");
    ASSERT_NE(strand, nullptr);
    EXPECT_TRUE(strand->is_array());
    Eigen::MatrixXd* m1 = strand->as_matrix();
    EXPECT_EQ(m1->rows(), 3);
    EXPECT_DOUBLE_EQ((*m1)(0,0), 1.0);
    EXPECT_DOUBLE_EQ((*m1)(1,0), 2.0);
    EXPECT_DOUBLE_EQ((*m1)(2,0), 3.0);

    // But "- 1 2 3" is JUXTAPOSITION: (- (1 2 3))
    // The minus function applied to the strand [1,2,3] = [-1,-2,-3]
    Value* result = machine->eval("- 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_array());
    Eigen::MatrixXd* m2 = result->as_matrix();
    EXPECT_EQ(m2->rows(), 3);
    EXPECT_DOUBLE_EQ((*m2)(0,0), -1.0);
    EXPECT_DOUBLE_EQ((*m2)(1,0), -2.0);
    EXPECT_DOUBLE_EQ((*m2)(2,0), -3.0);
}

// ============================================================================
// Rank Operator Tests (ISO 13751 §9)
// ============================================================================

TEST_F(EvalTest, RankMonadicFullRankSimple) {
    // -⍤2 on a simple 2x3 matrix (full rank = apply to whole matrix)
    Value* result = machine->eval("-⍤2 (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
}

TEST_F(EvalTest, RankMonadicFullRank) {
    // -⍤2 on matrix → applies - to whole matrix (rank 2 = full)
    Value* result = machine->eval("-⍤2 (2 3⍴⍳6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);  // -1
    EXPECT_DOUBLE_EQ((*m)(0, 1), -2.0);  // -2
    EXPECT_DOUBLE_EQ((*m)(1, 2), -6.0);  // -6
}

TEST_F(EvalTest, RankMonadicRank0Vector) {
    // -⍤0 vector → applies - to each scalar (0-cells)
    Value* result = machine->eval("-⍤0 (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), -3.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), -4.0);
}

TEST_F(EvalTest, RankMonadicRank1Matrix) {
    // -⍤1 on matrix → applies - to each row (1-cells)
    Value* result = machine->eval("-⍤1 (3 2⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_EQ(m->cols(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), -2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 1), -6.0);
}

TEST_F(EvalTest, RankDyadicFullRank) {
    // A +⍤2 B → applies + to whole arrays
    Value* result = machine->eval("1 2 3 +⍤2 (10 20 30)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 22.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 33.0);
}

TEST_F(EvalTest, RankDyadicRank0) {
    // A +⍤0 B → element-wise (same as regular +)
    Value* result = machine->eval("1 2 3 +⍤0 (10 20 30)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 22.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 33.0);
}

TEST_F(EvalTest, RankScalarArg) {
    // -⍤0 on scalar → just negate it
    Value* result = machine->eval("-⍤0 (42)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -42.0);
}

TEST_F(EvalTest, RankWithReductionSimple) {
    // Verify +/ works on a simple vector
    Value* result = machine->eval("+/ (1 2)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, RankWithReduction) {
    // +/⍤1 on matrix → sum each row
    // Matrix is 3×2, sum each row gives vector of 3 sums
    Value* result = machine->eval("+/⍤1 (3 2⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);   // 1+2
    EXPECT_DOUBLE_EQ((*m)(1, 0), 7.0);   // 3+4
    EXPECT_DOUBLE_EQ((*m)(2, 0), 11.0);  // 5+6
}

// ============================================================================
// Reduce Operator Tests (via grammar)
// ============================================================================

TEST_F(EvalTest, ReduceVector) {
    // +/ 1 2 3 4 → 10
    Value* result = machine->eval("+/ (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(EvalTest, ReduceVectorMultiply) {
    // ×/ 1 2 3 4 → 24
    Value* result = machine->eval("×/ (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 24.0);
}

TEST_F(EvalTest, ReduceMatrix) {
    // +/ on 2×3 matrix → vector [6, 15]
    Value* result = machine->eval("+/ (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ((*m)(1, 0), 15.0);  // 4+5+6
}

TEST_F(EvalTest, ReduceFirstMatrix) {
    // +⌿ on 2×3 matrix → vector [5, 7, 9]
    Value* result = machine->eval("+⌿ (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 5.0);   // 1+4
    EXPECT_DOUBLE_EQ((*m)(1, 0), 7.0);   // 2+5
    EXPECT_DOUBLE_EQ((*m)(2, 0), 9.0);   // 3+6
}

// ============================================================================
// Scan Operator Tests (via grammar)
// ============================================================================

TEST_F(EvalTest, ScanVector) {
    // +\ 1 2 3 4 → 1 3 6 10
    Value* result = machine->eval("+\\ (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), 10.0);
}

TEST_F(EvalTest, ScanVectorNonAssociative) {
    // -\ 1 2 3 4 → 1 -1 2 -2 (prefix reductions)
    Value* result = machine->eval("-\\ (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -1.0);  // 1-2
    EXPECT_DOUBLE_EQ((*m)(2, 0), 2.0);   // 1-(2-3)
    EXPECT_DOUBLE_EQ((*m)(3, 0), -2.0);  // 1-(2-(3-4))
}

TEST_F(EvalTest, ScanMatrix) {
    // +\ on 2×3 matrix → prefix sums along rows
    Value* result = machine->eval("+\\ (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    // Row 0: 1, 1+2=3, 1+2+3=6
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 3.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), 6.0);
    // Row 1: 4, 4+5=9, 4+5+6=15
    EXPECT_DOUBLE_EQ((*m)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 9.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 15.0);
}

TEST_F(EvalTest, ScanFirstMatrix) {
    // +⍀ on 2×3 matrix → prefix sums along columns
    Value* result = machine->eval("+⍀ (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    // Col 0: 1, 1+4=5
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 5.0);
    // Col 1: 2, 2+5=7
    EXPECT_DOUBLE_EQ((*m)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 7.0);
    // Col 2: 3, 3+6=9
    EXPECT_DOUBLE_EQ((*m)(0, 2), 3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 9.0);
}

// ============================================================================
// Each Operator Tests (via grammar)
// ============================================================================

TEST_F(EvalTest, EachScalar) {
    // -¨5 → -5
    Value* result = machine->eval("-¨ 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(EvalTest, EachVector) {
    // -¨1 2 3 → -1 -2 -3
    Value* result = machine->eval("-¨ (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), -3.0);
}

TEST_F(EvalTest, EachMatrix) {
    // -¨ on 2×2 matrix → negate each element
    Value* result = machine->eval("-¨ (2 2⍴1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), -2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), -4.0);
}

// ============================================================================
// Derived Operator Tests - these exercise the continuation-based iteration
// which enables non-primitive functions with reduce/scan/each
// ============================================================================

TEST_F(EvalTest, ReduceWithDerivedOperator) {
    // Use commute with reduce: +⍨/ (1 2 3)
    // This is (+⍨)/ which means reduce using the commuted plus
    // Since + is commutative, result is same as +/: 6
    Value* result = machine->eval("+⍨/ (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(EvalTest, ReduceWithDerivedOperatorNonCommutative) {
    // -⍨/ (10 3 1) → reduce with commuted minus
    // -⍨ means {⍵-⍺}, so A -⍨ B = B - A
    // Right-to-left: 10 -⍨ (3 -⍨ 1)
    //   3 -⍨ 1 = 1 - 3 = -2
    //   10 -⍨ (-2) = -2 - 10 = -12
    Value* result = machine->eval("-⍨/ (10 3 1)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -12.0);
}

TEST_F(EvalTest, ScanWithDerivedOperator) {
    // +⍨\ (1 2 3) → scan with commuted plus
    // Since + is commutative: 1, 3, 6
    Value* result = machine->eval("+⍨\\ (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);
}

TEST_F(EvalTest, EachWithDerivedOperator) {
    // ×⍨¨ (2 3 4) → square each element (x times itself)
    // Results: 4, 9, 16
    Value* result = machine->eval("×⍨¨ (2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 4.0);   // 2×2
    EXPECT_DOUBLE_EQ((*m)(1, 0), 9.0);   // 3×3
    EXPECT_DOUBLE_EQ((*m)(2, 0), 16.0);  // 4×4
}

TEST_F(EvalTest, EachWithReduceDerived) {
    // (+/)¨ applied to vectors would reduce each
    // But we don't have nested arrays yet, so test with matrix rows
    // For now, test that derived operators chain: +/¨ is valid syntax
    // Apply to a simple case
    Value* result = machine->eval("+/¨ (1 2 3)");
    ASSERT_NE(result, nullptr);
    // Each element is a scalar, +/ of scalar is the scalar
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

TEST_F(EvalTest, NestedDerivedOperators) {
    // -⍨⍨ negates the commute (back to normal minus order)
    // -⍨⍨/ (10 3) → same as -/ (10 3) = 10-3 = 7
    Value* result = machine->eval("-⍨⍨/ (10 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(EvalTest, MatrixReduceWithDerivedOperator) {
    // ×⍨/ on 2×3 matrix → product of each row (since × is commutative)
    // Row 0: 1×2×3 = 6, Row 1: 4×5×6 = 120
    Value* result = machine->eval("×⍨/ (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 120.0);
}

TEST_F(EvalTest, MatrixScanWithDerivedOperator) {
    // +⍨\ on 2×3 matrix → cumulative sums along rows
    Value* result = machine->eval("+⍨\\ (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    // Row 0: 1, 3, 6
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 3.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), 6.0);
    // Row 1: 4, 9, 15
    EXPECT_DOUBLE_EQ((*m)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 9.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 15.0);
}

// ============================================================================
// Outer Product Tests (via grammar)
// ============================================================================

TEST_F(EvalTest, OuterProductScalars) {
    // 5 ∘.+ 3 → 8
    Value* result = machine->eval("5 ∘.+ 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

TEST_F(EvalTest, OuterProductVectorVector) {
    // (1 2 3) ∘.+ (10 20) → 2D matrix
    Value* result = machine->eval("(1 2 3) ∘.+ (10 20)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_EQ(m->cols(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);  // 1+10
    EXPECT_DOUBLE_EQ((*m)(0, 1), 21.0);  // 1+20
    EXPECT_DOUBLE_EQ((*m)(1, 0), 12.0);  // 2+10
    EXPECT_DOUBLE_EQ((*m)(1, 1), 22.0);  // 2+20
    EXPECT_DOUBLE_EQ((*m)(2, 0), 13.0);  // 3+10
    EXPECT_DOUBLE_EQ((*m)(2, 1), 23.0);  // 3+20
}

TEST_F(EvalTest, OuterProductMultiply) {
    // (2 3) ∘.× (10 100) → multiplication table
    Value* result = machine->eval("(2 3) ∘.× (10 100)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 20.0);   // 2×10
    EXPECT_DOUBLE_EQ((*m)(0, 1), 200.0);  // 2×100
    EXPECT_DOUBLE_EQ((*m)(1, 0), 30.0);   // 3×10
    EXPECT_DOUBLE_EQ((*m)(1, 1), 300.0);  // 3×100
}

TEST_F(EvalTest, OuterProductWithDerivedOperator) {
    // (1 2) ∘.+⍨ (10 20) → uses commuted plus (same result since + is commutative)
    Value* result = machine->eval("(1 2) ∘.+⍨ (10 20)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 21.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 22.0);
}

TEST_F(EvalTest, OuterProductWithCommuteNonCommutative) {
    // (10 20) ∘.-⍨ (1 2) → -⍨ swaps args: (1-10, 1-20, 2-10, 2-20)
    Value* result = machine->eval("(10 20) ∘.-⍨ (1 2)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 2);
    // -⍨ means rhs - lhs: outer product applies f(lhs[i], rhs[j]) → rhs[j] - lhs[i]
    EXPECT_DOUBLE_EQ((*m)(0, 0), -9.0);   // 1-10
    EXPECT_DOUBLE_EQ((*m)(0, 1), -8.0);   // 2-10
    EXPECT_DOUBLE_EQ((*m)(1, 0), -19.0);  // 1-20
    EXPECT_DOUBLE_EQ((*m)(1, 1), -18.0);  // 2-20
}

// ============================================================================
// Inner Product Tests (via grammar)
// ============================================================================

TEST_F(EvalTest, InnerProductVectorDot) {
    // (1 2 3) +.× (4 5 6) → dot product: 1×4 + 2×5 + 3×6 = 32
    Value* result = machine->eval("(1 2 3) +.× (4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 32.0);
}

TEST_F(EvalTest, InnerProductMatrixMultiply) {
    // Matrix multiplication: (2 2⍴1 2 3 4) +.× (2 2⍴5 6 7 8)
    // [1 2] × [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    Value* result = machine->eval("(2 2⍴1 2 3 4) +.× (2 2⍴5 6 7 8)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 19.0);  // 1×5 + 2×7
    EXPECT_DOUBLE_EQ((*m)(0, 1), 22.0);  // 1×6 + 2×8
    EXPECT_DOUBLE_EQ((*m)(1, 0), 43.0);  // 3×5 + 4×7
    EXPECT_DOUBLE_EQ((*m)(1, 1), 50.0);  // 3×6 + 4×8
}

TEST_F(EvalTest, InnerProductMatrixVector) {
    // Matrix × vector: (2 2⍴1 2 3 4) +.× (5 7)
    Value* result = machine->eval("(2 2⍴1 2 3 4) +.× (5 7)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 19.0);  // 1×5 + 2×7
    EXPECT_DOUBLE_EQ((*m)(1, 0), 43.0);  // 3×5 + 4×7
}

TEST_F(EvalTest, InnerProductWithDerivedOperator) {
    // Use commuted multiply: (1 2 3) +.×⍨ (4 5 6)
    // ×⍨ is same as × (commutative), so same result as dot product
    Value* result = machine->eval("(1 2 3) +.×⍨ (4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 32.0);
}

TEST_F(EvalTest, InnerProductDifferentOperators) {
    // (10 20 30) +.÷ (2 4 5) → 10÷2 + 20÷4 + 30÷5 = 5 + 5 + 6 = 16
    Value* result = machine->eval("(10 20 30) +.÷ (2 4 5)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 16.0);
}

// ============================================================================
// Execute (⍎) end-to-end tests
// ============================================================================

TEST_F(EvalTest, ExecuteSimpleNumber) {
    // ⍎'42' → 42
    Value* result = machine->eval("⍎'42'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(EvalTest, ExecuteArithmetic) {
    // ⍎'1+2' → 3
    Value* result = machine->eval("⍎'1+2'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, ExecuteVector) {
    // ⍎'1 2 3' → 1 2 3
    Value* result = machine->eval("⍎'1 2 3'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
}

TEST_F(EvalTest, ExecuteWithReduce) {
    // ⍎'+/⍳5' → 15 (sum of 1 2 3 4 5)
    Value* result = machine->eval("⍎'+/⍳5'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(EvalTest, ExecuteComplex) {
    // ⍎'2×3+4' → 14 (right-to-left: 3+4=7, 2×7=14)
    Value* result = machine->eval("⍎'2×3+4'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

TEST_F(EvalTest, ExecuteInExpression) {
    // 1 + ⍎'2' → 3
    Value* result = machine->eval("1 + ⍎'2'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// ============================================================================
// G2 Grammar: G_PRIME Curry Tests
// Per Georgeff et al. "Parsing and Evaluation of APL with Operators"
// ============================================================================

TEST_F(EvalTest, GPrimeMonadicOnlyInDyadicContextErrors) {
    // Monadic-only functions used in dyadic context should error
    // ≢ (tally) has no dyadic form
    EXPECT_THROW(machine->eval("5 ≢ 1 2 3"), APLError);
}

// ============================================================================
// Dyadic Character Grade Tests (ISO 13751 Section 10.2.20-21)
// ============================================================================

TEST_F(EvalTest, DyadicGradeUpBasic) {
    // A⍋B - grade up using collating sequence A
    // 'abc' ⍋ 'cab' should give indices that sort 'cab' to 'abc' order
    Value* result = machine->eval("'abc' ⍋ 'cab'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // 'cab' positions: c=2, a=0, b=1 -> sorted order is a,b,c -> indices 2,3,1 (1-indexed)
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 1.0);
}

TEST_F(EvalTest, DyadicGradeDownBasic) {
    // A⍒B - grade down using collating sequence A
    Value* result = machine->eval("'abc' ⍒ 'cab'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // Descending order: c,b,a -> indices 1,3,2 (1-indexed)
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 2.0);
}

TEST_F(EvalTest, DyadicGradeUnknownCharsLast) {
    // Characters not in collating sequence sort after known characters
    Value* result = machine->eval("'ab' ⍋ 'axb'");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->size(), 3);
    // 'a'=pos0, 'x'=unknown, 'b'=pos1 -> sorted: a,b,x -> indices 1,3,2
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 2.0);
}

TEST_F(EvalTest, DyadicGradeDownUnknownCharsStillLast) {
    // Even in descending order, unknowns sort LAST (ISO 13751)
    Value* result = machine->eval("'ab' ⍒ 'axb'");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->size(), 3);
    // Descending known chars first: b,a then unknowns: x -> indices 3,1,2
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 2.0);
}

TEST_F(EvalTest, DyadicGradeMatrixRows) {
    // Matrix B sorts rows lexicographically
    Value* result = machine->eval("'abc' ⍋ 3 3 ⍴ 'cabbbaabc'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // Rows: 'cab', 'bba', 'abc' -> sorted: 'abc'(3), 'bba'(2), 'cab'(1)
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 1.0);
}

TEST_F(EvalTest, DyadicGradeScalarCollatingError) {
    // Scalar A should signal RANK ERROR - use ↑ (first) to get actual scalar
    EXPECT_THROW(machine->eval("(↑'a') ⍋ 'abc'"), APLError);
    EXPECT_THROW(machine->eval("(↑'a') ⍒ 'abc'"), APLError);
}

TEST_F(EvalTest, DyadicGradeDomainError) {
    // Non-character B should signal DOMAIN ERROR
    EXPECT_THROW(machine->eval("'abc' ⍋ 1 2 3"), APLError);
    EXPECT_THROW(machine->eval("'abc' ⍒ 1 2 3"), APLError);
}

TEST_F(EvalTest, GPrimeOverloadedFunctionDyadic) {
    // Overloaded functions (both monadic and dyadic) should apply dyadically
    // ≡ (depth/match) has both forms - dyadic match returns 1 if identical
    Value* result = machine->eval("1 2 3 ≡ 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);

    result = machine->eval("1 2 3 ≡ 4 5 6");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(EvalTest, GPrimeNestedCurryFinalization) {
    // Execute (⍎) creates G_PRIME curry, inner code may also create curries
    // All should finalize properly
    Value* result = machine->eval("⍎'⍳5'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);

    result = machine->eval("⍎'+/⍳5'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);

    result = machine->eval("⍎'≢1 2 3 4 5'");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, GPrimeCurryInExpressionContext) {
    // G_PRIME curry as argument to another function should unwrap
    Value* result = machine->eval("2 + ⍎'3'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);

    result = machine->eval("10 - ≢1 2 3");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(EvalTest, GPrimeStrandFormationInDfn) {
    // Value-value juxtaposition in dfn body forms strands
    Value* result = machine->eval("{⍵ ⍵}5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);

    result = machine->eval("{⍵ ⍵ ⍵}5");
    EXPECT_EQ(result->size(), 3);

    // But A f B should NOT strand if f has dyadic form
    result = machine->eval("3{⍺ + ⍵}5");  // Should add, not strand
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// ============================================================================
// ISO 13751 Section 6 Compliance Tests
// ============================================================================

// Section 6.3.3: Evaluate-Niladic-Function (Pattern N)
TEST_F(EvalTest, NiladicDfnBasic) {
    // Niladic dfn: no ⍵ or ⍺ references, just returns a value
    Value* result = machine->eval("F←{42} ⋄ F");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(EvalTest, NiladicDfnExpression) {
    // Niladic dfn with expression inside
    Value* result = machine->eval("F←{2+3} ⋄ F");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, NiladicDfnWithReduce) {
    // Niladic dfn that uses reduction
    Value* result = machine->eval("F←{+/⍳10} ⋄ F");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 55.0);
}

// Section 6.3.2: Remove-Parentheses - error cases
TEST_F(EvalTest, ParenthesesWithValue) {
    // (B) where B is a value - should work
    Value* result = machine->eval("(5)");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);

    result = machine->eval("(1 2 3)");
    EXPECT_EQ(result->size(), 3);
}

TEST_F(EvalTest, ParenthesesWithExpression) {
    // ((2+3)) nested parens
    Value* result = machine->eval("((2+3))");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Section 6.3.4: Evaluate-Monadic-Function with axis (X F[C] B)
TEST_F(EvalTest, MonadicFunctionWithAxis) {
    // Reverse with axis on matrix
    machine->eval("M←2 3⍴⍳6");
    Value* result = machine->eval("⌽[1]M");  // Reverse along first axis (rows)
    ASSERT_NE(result, nullptr);
    // Original: 1 2 3 / 4 5 6, reversed rows: 4 5 6 / 1 2 3
    const auto* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);
}

TEST_F(EvalTest, ReverseWithAxis) {
    machine->eval("M←2 3⍴⍳6");  // [[1,2,3],[4,5,6]]

    // ⌽[1] reverses along first axis (rows)
    Value* r1 = machine->eval("⌽[1]M");
    ASSERT_NE(r1, nullptr);
    const auto* m1 = r1->as_matrix();
    EXPECT_DOUBLE_EQ((*m1)(0, 0), 4.0);  // First row is now [4,5,6]
    EXPECT_DOUBLE_EQ((*m1)(1, 0), 1.0);  // Second row is now [1,2,3]

    // ⌽[2] reverses along second axis (columns)
    Value* r2 = machine->eval("⌽[2]M");
    ASSERT_NE(r2, nullptr);
    const auto* m2 = r2->as_matrix();
    EXPECT_DOUBLE_EQ((*m2)(0, 0), 3.0);  // [3,2,1]
    EXPECT_DOUBLE_EQ((*m2)(0, 2), 1.0);
    EXPECT_DOUBLE_EQ((*m2)(1, 0), 6.0);  // [6,5,4]

    // ⊖[1] is same as ⌽[1] for reverse first
    Value* r3 = machine->eval("⊖[1]M");
    ASSERT_NE(r3, nullptr);
    const auto* m3 = r3->as_matrix();
    EXPECT_DOUBLE_EQ((*m3)(0, 0), 4.0);

    // ⊖[2] reverses along second axis
    Value* r4 = machine->eval("⊖[2]M");
    ASSERT_NE(r4, nullptr);
    const auto* m4 = r4->as_matrix();
    EXPECT_DOUBLE_EQ((*m4)(0, 0), 3.0);
}

TEST_F(EvalTest, TakeWithAxis) {
    machine->eval("M←3 4⍴⍳12");  // [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

    // 2↑[1]M takes 2 rows
    Value* t1 = machine->eval("2↑[1]M");
    ASSERT_NE(t1, nullptr);
    const auto* m1 = t1->as_matrix();
    EXPECT_EQ(m1->rows(), 2);
    EXPECT_EQ(m1->cols(), 4);
    EXPECT_DOUBLE_EQ((*m1)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m1)(1, 0), 5.0);

    // 2↑[2]M takes 2 columns
    Value* t2 = machine->eval("2↑[2]M");
    ASSERT_NE(t2, nullptr);
    const auto* m2 = t2->as_matrix();
    EXPECT_EQ(m2->rows(), 3);
    EXPECT_EQ(m2->cols(), 2);
    EXPECT_DOUBLE_EQ((*m2)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m2)(0, 1), 2.0);

    // Negative take: ¯2↑[1]M takes last 2 rows
    Value* t3 = machine->eval("¯2↑[1]M");
    ASSERT_NE(t3, nullptr);
    const auto* m3 = t3->as_matrix();
    EXPECT_EQ(m3->rows(), 2);
    EXPECT_DOUBLE_EQ((*m3)(0, 0), 5.0);   // Second row
    EXPECT_DOUBLE_EQ((*m3)(1, 0), 9.0);   // Third row
}

TEST_F(EvalTest, DropWithAxis) {
    machine->eval("M←3 4⍴⍳12");  // [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

    // 1↓[1]M drops 1 row
    Value* d1 = machine->eval("1↓[1]M");
    ASSERT_NE(d1, nullptr);
    const auto* m1 = d1->as_matrix();
    EXPECT_EQ(m1->rows(), 2);
    EXPECT_EQ(m1->cols(), 4);
    EXPECT_DOUBLE_EQ((*m1)(0, 0), 5.0);  // Starts at row 2

    // 1↓[2]M drops 1 column
    Value* d2 = machine->eval("1↓[2]M");
    ASSERT_NE(d2, nullptr);
    const auto* m2 = d2->as_matrix();
    EXPECT_EQ(m2->rows(), 3);
    EXPECT_EQ(m2->cols(), 3);
    EXPECT_DOUBLE_EQ((*m2)(0, 0), 2.0);  // Starts at column 2

    // Negative drop: ¯1↓[2]M drops last column
    Value* d3 = machine->eval("¯1↓[2]M");
    ASSERT_NE(d3, nullptr);
    const auto* m3 = d3->as_matrix();
    EXPECT_EQ(m3->cols(), 3);
    EXPECT_DOUBLE_EQ((*m3)(0, 2), 3.0);  // Last col is now 3,7,11
}

TEST_F(EvalTest, LaminateWithAxis) {
    // ,[0.5] creates new first axis (vectors become rows)
    Value* l1 = machine->eval("1 2 3 ,[0.5] 4 5 6");
    ASSERT_NE(l1, nullptr);
    const auto* m1 = l1->as_matrix();
    EXPECT_EQ(m1->rows(), 2);
    EXPECT_EQ(m1->cols(), 3);
    EXPECT_DOUBLE_EQ((*m1)(0, 0), 1.0);  // First row: 1 2 3
    EXPECT_DOUBLE_EQ((*m1)(1, 0), 4.0);  // Second row: 4 5 6

    // ,[1.5] creates new second axis (vectors become columns)
    Value* l2 = machine->eval("1 2 3 ,[1.5] 4 5 6");
    ASSERT_NE(l2, nullptr);
    const auto* m2 = l2->as_matrix();
    EXPECT_EQ(m2->rows(), 3);
    EXPECT_EQ(m2->cols(), 2);
    EXPECT_DOUBLE_EQ((*m2)(0, 0), 1.0);  // First col: 1,2,3
    EXPECT_DOUBLE_EQ((*m2)(0, 1), 4.0);  // Second col: 4,5,6
}

// Section 6.3.6: Evaluate-Dyadic-Function with axis (A F[C] B)
TEST_F(EvalTest, DyadicFunctionWithAxisCatenate) {
    // ,[1] on vectors catenates along axis 1 (the only axis) = regular catenate
    Value* result = machine->eval("1 2 3 ,[1] 4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 6);

    // Laminate requires fractional axis (ISO 13751): ,[0.5] creates new first axis
    Value* laminated = machine->eval("1 2 3 ,[0.5] 4 5 6");
    ASSERT_NE(laminated, nullptr);
    EXPECT_TRUE(laminated->is_matrix());
    const auto* m = laminated->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);

    // Matrix catenation with axis
    machine->eval("A←2 3⍴⍳6");   // [[1,2,3],[4,5,6]]
    machine->eval("B←2 3⍴7+⍳6"); // [[8,9,10],[11,12,13]]

    // ,[1] catenates along first axis (rows) - same as ⍪
    Value* cat1 = machine->eval("A,[1]B");
    ASSERT_NE(cat1, nullptr);
    EXPECT_TRUE(cat1->is_matrix());
    const auto* m1 = cat1->as_matrix();
    EXPECT_EQ(m1->rows(), 4);
    EXPECT_EQ(m1->cols(), 3);

    // ,[2] catenates along second axis (columns)
    Value* cat2 = machine->eval("A,[2]B");
    ASSERT_NE(cat2, nullptr);
    EXPECT_TRUE(cat2->is_matrix());
    const auto* m2 = cat2->as_matrix();
    EXPECT_EQ(m2->rows(), 2);
    EXPECT_EQ(m2->cols(), 6);
}

TEST_F(EvalTest, DyadicFunctionWithAxisCatenateFirst) {
    // Catenate along first axis
    machine->eval("A←2 3⍴⍳6");
    machine->eval("B←2 3⍴7+⍳6");
    Value* result = machine->eval("A⍪B");  // First-axis catenate
    ASSERT_NE(result, nullptr);
    const auto* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_EQ(m->cols(), 3);
}

// Section 6.3.5: Evaluate-Monadic-Operator with axis (F M[C] B)
TEST_F(EvalTest, MonadicOperatorWithAxisReduce) {
    // +/[1] reduces along first axis (columns)
    machine->eval("M←2 3⍴⍳6");  // [[1,2,3],[4,5,6]]
    Value* result = machine->eval("+/[1]M");  // Sum columns: 5 7 9
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const auto* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*v)(2, 0), 9.0);
}

TEST_F(EvalTest, MonadicOperatorWithAxisScan) {
    // +\[1] scans along first axis
    machine->eval("M←2 3⍴⍳6");  // [[1,2,3],[4,5,6]]
    Value* result = machine->eval("+\\[1]M");  // Running sum down columns
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
}

// Section 6.3.7: Evaluate-Dyadic-Operator tests
TEST_F(EvalTest, DyadicOperatorInnerProduct) {
    // Inner product A +.× B
    Value* result = machine->eval("1 2 3 +.× 4 5 6");
    ASSERT_NE(result, nullptr);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_DOUBLE_EQ(result->as_scalar(), 32.0);
}

TEST_F(EvalTest, DyadicOperatorOuterProduct) {
    // Outer product A ∘.× B
    Value* result = machine->eval("1 2 ∘.× 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const auto* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    // [[3,4,5],[6,8,10]]
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 10.0);
}

// Section 6.3.8: Evaluate-Indexed-Reference (A[K])
TEST_F(EvalTest, IndexedReferenceScalar) {
    // Simple scalar index
    machine->eval("A←10 20 30 40 50");
    Value* result = machine->eval("A[3]");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 30.0);
}

TEST_F(EvalTest, IndexedReferenceVector) {
    // Vector index returns vector
    machine->eval("A←10 20 30 40 50");
    Value* result = machine->eval("A[2 4]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
}

TEST_F(EvalTest, IndexedReferenceExpression) {
    // Index can be an expression
    machine->eval("A←10 20 30 40 50");
    Value* result = machine->eval("A[1+2]");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 30.0);
}


TEST_F(EvalTest, AssignmentChain) {
    // Chained assignment: A←B←5
    Value* result = machine->eval("A←B←5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);

    result = machine->eval("A");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);

    result = machine->eval("B");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Section 6.3.10: Evaluate-Indexed-Assignment (V[K]←B)
TEST_F(EvalTest, IndexedAssignmentBasic) {
    machine->eval("A←1 2 3 4 5");
    Value* result = machine->eval("A[3]←99");
    ASSERT_NE(result, nullptr);

    result = machine->eval("A");
    const auto* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(2, 0), 99.0);  // 3rd element (0-indexed: 2)
}

TEST_F(EvalTest, IndexedAssignmentMultiple) {
    // Assign to multiple indices
    machine->eval("A←1 2 3 4 5");
    machine->eval("A[2 4]←10 20");

    Value* result = machine->eval("A");
    const auto* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(1, 0), 10.0);  // A[2]
    EXPECT_DOUBLE_EQ((*v)(3, 0), 20.0);  // A[4]
}

// Section 6.3.11: Evaluate-Variable (V)
TEST_F(EvalTest, VariableLookup) {
    machine->eval("X←123");
    Value* result = machine->eval("X");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 123.0);
}


// Section 6.3.13: Process-End-of-Statement
TEST_F(EvalTest, EmptyStatementReturnsNil) {
    // Empty line/statement handling
    // The diamond separator creates multiple statements
    Value* result = machine->eval("1 ⋄ 2 ⋄ 3");
    ASSERT_NE(result, nullptr);
    // Last statement's result
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(EvalTest, BranchToZeroExitsFunction) {
    // →0 exits function, returning previous result
    Value* result = machine->eval("{5 ⋄ →0 ⋄ 99}0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(EvalTest, BranchToEmptyExitsFunction) {
    // →⍬ exits function (empty vector target)
    Value* result = machine->eval("{5 ⋄ →⍬ ⋄ 99}0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Section 6 Error Cases
TEST_F(EvalTest, IndexErrorOutOfBounds) {
    machine->eval("A←1 2 3");
    EXPECT_THROW(machine->eval("A[10]"), APLError);  // INDEX ERROR
}

TEST_F(EvalTest, RankErrorScalarIndexing) {
    // Scalars cannot be indexed (rank mismatch)
    machine->eval("S←42");
    EXPECT_THROW(machine->eval("S[1]"), APLError);  // RANK ERROR
}

TEST_F(EvalTest, DomainErrorNonIntegerIndex) {
    machine->eval("A←1 2 3");
    EXPECT_THROW(machine->eval("A[1.5]"), APLError);  // DOMAIN ERROR (non-integer)
}

TEST_F(EvalTest, AxisErrorUnsupportedFunction) {
    // Functions that don't support axis should signal AXIS ERROR
    EXPECT_THROW(machine->eval("+[1] 5"), APLError);      // + doesn't support axis
    EXPECT_THROW(machine->eval("×[2] 3 4"), APLError);    // × doesn't support axis
    EXPECT_THROW(machine->eval("-[1] 10"), APLError);     // - doesn't support axis
    EXPECT_THROW(machine->eval("⍳[1] 5"), APLError);      // ⍳ doesn't support axis
    EXPECT_THROW(machine->eval("⍴[1] 3 4"), APLError);    // ⍴ doesn't support axis
}

// ============================================================================
// ISO 13751 Section 10.2 Gap Tests - Dyadic Mixed Functions
// ============================================================================

// --- 10.2.1 Join Along Axis ---

// ISO 13751 10.2.1: Laminate with scalar extension
TEST_F(EvalTest, LaminateScalarExtension) {
    // 1 2 3,[1.5]4 → 3×2 matrix with 4 replicated
    Value* result = machine->eval("1 2 3,[1.5]4");
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 2);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*m)(2, 1), 4.0);
}

// ISO 13751 10.2.1: Both scalars with integer axis is error
TEST_F(EvalTest, CatenateBothScalarsError) {
    // 5,[1]3 with both scalars → AXIS ERROR
    EXPECT_THROW(machine->eval("5,[1]3"), APLError);
}

// ISO 13751 10.2.1: Laminate with fractional axis (both scalars OK)
TEST_F(EvalTest, LaminateBothScalars) {
    // 5,[0.5]3 → 2-element vector [5,3]
    Value* result = machine->eval("5,[0.5]3");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
}

// --- 10.2.2 Index Of ---

// ISO 13751 10.2.2: Non-vector left arg signals RANK ERROR
TEST_F(EvalTest, IndexOfNonVectorLeftError) {
    // (2 3⍴⍳6)⍳3 → RANK ERROR
    EXPECT_THROW(machine->eval("(2 3⍴⍳6)⍳3"), APLError);
}

// ISO 13751 10.2.2: ⎕CT affects tolerant equality
TEST_F(EvalTest, IndexOfCTEffect) {
    machine->eval("⎕CT←0.1");
    Value* result = machine->eval("1 2 3⍳1.05");
    // 1.05 is within tolerance of 1, should find at index 1
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
    machine->eval("⎕CT←1E¯14");
}

// ISO 13751 10.2.2: Character index-of
TEST_F(EvalTest, IndexOfCharacter) {
    Value* result = machine->eval("'ABC'⍳'B'");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(EvalTest, IndexOfCharacterNotFound) {
    // 'ABC'⍳'D' → 4 (one past end)
    Value* result = machine->eval("'ABC'⍳'D'");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

// --- 10.2.3 Member Of ---

// ISO 13751 10.2.3: ⎕CT affects tolerant equality
TEST_F(EvalTest, MemberOfCTEffect) {
    machine->eval("⎕CT←0.1");
    Value* result = machine->eval("1.05∊1 2 3");
    // 1.05 is within tolerance of 1
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
    machine->eval("⎕CT←1E¯14");
}

// ISO 13751 10.2.3: Matrix B - result has shape of A
TEST_F(EvalTest, MemberOfMatrixRight) {
    // 3 5∊2 3⍴⍳6 → check 3 and 5 membership
    Value* result = machine->eval("3 5∊2 3⍴⍳6");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);  // 3 is in 1..6
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // 5 is in 1..6
}

// --- 10.2.5 Replicate ---

// ISO 13751 10.2.5: Scalar left argument expansion
TEST_F(EvalTest, ReplicateScalarLeftExpansion) {
    // 2/1 2 3 → 1 1 2 2 3 3
    Value* result = machine->eval("2/1 2 3");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 6);
}

// ISO 13751 10.2.5: Replicate with axis
TEST_F(EvalTest, ReplicateWithAxis) {
    // 1 0 1/[2] 2 3⍴⍳6 → select columns 1 and 3
    Value* result = machine->eval("1 0 1/[2] 2 3⍴⍳6");
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
}

// --- 10.2.6 Expand ---

// ISO 13751 10.2.6: Non-boolean signals DOMAIN ERROR
TEST_F(EvalTest, ExpandNonBooleanError) {
    EXPECT_THROW(machine->eval("1 2 1\\1 2"), APLError);
}

// ISO 13751 10.2.6: Expand with axis
TEST_F(EvalTest, ExpandWithAxis) {
    // 1 0 1\[1] 2 3⍴⍳6 → insert zero row
    Value* result = machine->eval("1 0 1\\[1] 2 3⍴⍳6");
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 3);
}

// --- 10.2.11 Take ---

// ISO 13751 10.2.11: Matrix take
TEST_F(EvalTest, TakeMatrix) {
    // 2 2↑3 4⍴⍳12 → top-left 2×2 corner
    Value* result = machine->eval("2 2↑3 4⍴⍳12");
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
}

// ISO 13751 10.2.11: Rank>1 left arg signals RANK ERROR
TEST_F(EvalTest, TakeRankLeftError) {
    EXPECT_THROW(machine->eval("(2 2⍴⍳4)↑1 2 3 4 5"), APLError);
}

// --- 10.2.12 Drop ---

// ISO 13751 10.2.12: Matrix drop
TEST_F(EvalTest, DropMatrix) {
    // 1 1↓3 4⍴⍳12 → drop first row and column
    Value* result = machine->eval("1 1↓3 4⍴⍳12");
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
}

// ISO 13751 10.2.12: Rank>1 left arg signals RANK ERROR
TEST_F(EvalTest, DropRankLeftError) {
    EXPECT_THROW(machine->eval("(2 2⍴⍳4)↓1 2 3 4 5"), APLError);
}

// --- 10.2.14-15 Indexed Reference/Assignment ---

// ISO 13751 10.2.14: Multi-dimensional indexing
TEST_F(EvalTest, DISABLED_IndexedRefMultiDim) {
    machine->eval("M←3 4⍴⍳12");
    Value* result = machine->eval("M[2;3]");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);  // Row 2, Col 3 of 1-indexed
}

// ISO 13751 10.2.14: Elided index (all elements along axis)
TEST_F(EvalTest, DISABLED_IndexedRefElidedIndex) {
    machine->eval("M←3 4⍴⍳12");
    Value* result = machine->eval("M[2;]");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);  // All columns of row 2
}

// ISO 13751 10.2.14: Scalar cannot be indexed
TEST_F(EvalTest, IndexedRefScalarError) {
    machine->eval("S←42");
    EXPECT_THROW(machine->eval("S[1]"), APLError);
}

// ISO 13751 10.2.15: Multi-dimensional indexed assignment
TEST_F(EvalTest, DISABLED_IndexedAssignMultiDim) {
    machine->eval("M←3 4⍴⍳12");
    machine->eval("M[2;3]←99");
    Value* result = machine->eval("M[2;3]");
    EXPECT_DOUBLE_EQ(result->as_scalar(), 99.0);
}


// Main function

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
