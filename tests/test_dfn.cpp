// Dfn (Direct Function) tests - user-defined functions with braces
// Tests dfn parsing, application, guards, recursion, closures, and operators

#include <gtest/gtest.h>
#include "parser.h"
#include "machine.h"
#include "continuation.h"

using namespace apl;

class DfnTest : public ::testing::Test {
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
// Basic Dfn Parsing
// ============================================================================

TEST_F(DfnTest, ParseMonadicDfn) {
    // {omega*2} should create a closure
    Continuation* k = parser->parse("{⍵×2}");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

TEST_F(DfnTest, ApplyMonadicDfn) {
    // ({omega*2} 5) should evaluate to 10
    // Note: Need to use dyadic syntax since we're applying the dfn to an argument
    Continuation* k = parser->parse("5 {⍵×2} 0");  // Using a dummy left arg for now

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    // This will fail until we implement monadic application properly
    // For now, just test that it parses
}

TEST_F(DfnTest, ParseDyadicDfn) {
    // {alpha+omega} should create a closure
    Continuation* k = parser->parse("{⍺+⍵}");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

// ============================================================================
// Dfn Application
// ============================================================================

TEST_F(DfnTest, DfnJustOmega) {
    // 0 {omega} 5 should return 5
    Continuation* k = parser->parse("0 {⍵} 5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(DfnTest, ApplyDyadicDfn) {
    // 3 {alpha+omega} 5 should evaluate to 8
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

TEST_F(DfnTest, AssignDfn) {
    // square <- {omega*omega}
    Continuation* k = parser->parse("square ← {⍵×⍵}");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    // Check that the variable was assigned
    Value* square = machine->env->lookup(machine->string_pool.intern("square"));
    ASSERT_NE(square, nullptr);
    EXPECT_EQ(square->tag, ValueType::CLOSURE);
}

TEST_F(DfnTest, DfnNestedExpression) {
    // {(alpha*2)+(omega*3)} should parse correctly
    Continuation* k = parser->parse("{(⍺×2)+(⍵×3)}");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

TEST_F(DfnTest, ApplyNestedDfn) {
    // 4 {(alpha*2)+(omega*3)} 5 should evaluate to (4*2)+(5*3) = 8+15 = 23
    Continuation* k = parser->parse("4 {(⍺×2)+(⍵×3)} 5");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 23.0);
}

TEST_F(DfnTest, DfnMonadicDirect) {
    // {omega+1}5 should evaluate to 6
    Continuation* k = parser->parse("{⍵+1}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(DfnTest, DfnNamedMonadic) {
    // F<-{omega*2} diamond F 5 should return 10
    Continuation* k = parser->parse("F←{⍵×2} ⋄ F 5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// ============================================================================
// Local Variables
// ============================================================================

TEST_F(DfnTest, DfnLocalAssignment) {
    // {x<-5 diamond x+omega}3 should return 8
    Continuation* k = parser->parse("{x←5 ⋄ x+⍵}3");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

TEST_F(DfnTest, DfnMultipleLocals) {
    // {a<-1 diamond b<-2 diamond c<-3 diamond a+b+c+omega}10 should return 16
    Continuation* k = parser->parse("{a←1 ⋄ b←2 ⋄ c←3 ⋄ a+b+c+⍵}10");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 16.0);
}

// ============================================================================
// Return Values
// ============================================================================

TEST_F(DfnTest, DfnValueValueJuxtapositionIsSyntaxError) {
    // ISO 13751: value-value juxtaposition is SYNTAX ERROR, even in dfn bodies
    // {⍵ ⍵ ⍵}5 has adjacent values with no function - SYNTAX ERROR
    EXPECT_THROW(machine->eval("{⍵ ⍵ ⍵}5"), APLError);
    EXPECT_THROW(machine->eval("{⍵ ⍵}5"), APLError);
}

TEST_F(DfnTest, DfnVectorArgument) {
    // {+/omega}1 2 3 should return 6
    Continuation* k = parser->parse("{+/⍵}1 2 3");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(DfnTest, DfnMatrixArgument) {
    // {rho omega}2 3 rho iota 6 should return 2 3
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

// ============================================================================
// Guards
// ============================================================================

TEST_F(DfnTest, DfnGuardTrue) {
    // {omega>0: omega}5 should return 5
    Continuation* k = parser->parse("{⍵>0: ⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(DfnTest, DfnGuardFalseWithDefault) {
    // {omega>0: omega diamond 0}neg5 should return 0
    Continuation* k = parser->parse("{⍵>0: ⍵ ⋄ 0}¯5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(DfnTest, DfnMultipleGuards) {
    // {omega<0: neg1 diamond omega=0: 0 diamond 1}5 should return 1 (positive)
    Continuation* k = parser->parse("{⍵<0: ¯1 ⋄ ⍵=0: 0 ⋄ 1}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(DfnTest, DfnMultipleGuardsSecond) {
    // {omega<0: neg1 diamond omega=0: 0 diamond 1}0 should return 0
    Continuation* k = parser->parse("{⍵<0: ¯1 ⋄ ⍵=0: 0 ⋄ 1}0");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(DfnTest, DfnMultipleGuardsFirst) {
    // {omega<0: neg1 diamond omega=0: 0 diamond 1}neg5 should return neg1
    Continuation* k = parser->parse("{⍵<0: ¯1 ⋄ ⍵=0: 0 ⋄ 1}¯5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -1.0);
}

// ============================================================================
// Recursion (del)
// ============================================================================

TEST_F(DfnTest, DfnRecursiveFactorial) {
    // {omega<=1: 1 diamond omega*del omega-1}5 should return 120 (5!)
    Continuation* k = parser->parse("{⍵≤1: 1 ⋄ ⍵×∇ ⍵-1}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 120.0);
}

TEST_F(DfnTest, DfnRecursiveFibonacci) {
    // {omega<=1: omega diamond (del omega-1)+del omega-2}6 should return 8 (fib(6))
    Continuation* k = parser->parse("{⍵≤1: ⍵ ⋄ (∇ ⍵-1)+∇ ⍵-2}6");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

TEST_F(DfnTest, DfnNamedRecursive) {
    // fact<-{omega<=1: 1 diamond omega*del omega-1} diamond fact 5 should return 120
    Continuation* k = parser->parse("fact←{⍵≤1: 1 ⋄ ⍵×∇ ⍵-1} ⋄ fact 5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 120.0);
}

// ============================================================================
// Alpha Handling
// ============================================================================

TEST_F(DfnTest, DfnAlphaMonadicError) {
    // {alpha+omega}5 - calling a dfn that uses alpha monadically should error
    // because alpha is not defined when called without left argument
    Continuation* k = parser->parse("{⍺+⍵}5");
    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();

    machine->push_kont(k);
    // Expect VALUE ERROR for undefined alpha
    EXPECT_THROW(machine->execute(), APLError);
}

TEST_F(DfnTest, DfnAlphaDefault) {
    // {alpha<-10 diamond alpha+omega}5 should return 15 (alpha defaults to 10)
    Continuation* k = parser->parse("{⍺←10 ⋄ ⍺+⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(DfnTest, DfnAlphaDefaultOverride) {
    // 3{alpha<-10 diamond alpha+omega}5 should return 8 (alpha=3 overrides default)
    Continuation* k = parser->parse("3{⍺←10 ⋄ ⍺+⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// ============================================================================
// Nested and Higher-Order Dfns
// ============================================================================

TEST_F(DfnTest, DfnNested) {
    // {F<-{omega+1} diamond F omega}5 should return 6
    Continuation* k = parser->parse("{F←{⍵+1} ⋄ F ⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(DfnTest, DfnReturningDfn) {
    // adder<-{omega{alpha+omega}} creates a function that adds omega
    // Not all APLs support this - test for graceful handling
    Continuation* k = parser->parse("adder←{{⍺+⍵}⍵}");

    // May or may not parse - test doesn't crash
    if (k != nullptr) {
        machine->push_kont(k);
        machine->execute();
    }
}

TEST_F(DfnTest, DfnCallsNamedDfn) {
    // double<-{omega*2} diamond {double omega}5 should return 10
    Continuation* k = parser->parse("double←{⍵×2} ⋄ {double ⍵}5");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(DfnTest, DfnEmpty) {
    // {} - empty dfn, behavior varies by implementation
    Continuation* k = parser->parse("{}5");

    // Should either error or return something predictable
    // Test that it doesn't crash
    if (k != nullptr) {
        machine->push_kont(k);
        machine->execute();
    }
}

// ============================================================================
// Dfns with Operators
// ============================================================================

TEST_F(DfnTest, DfnAsReduceOperand) {
    // {alpha+omega}/1 2 3 4 should return 10
    Continuation* k = parser->parse("{⍺+⍵}/1 2 3 4");

    ASSERT_NE(k, nullptr) << "Parse error: " << parser->get_error();
    EXPECT_EQ(parser->get_error(), "");

    machine->push_kont(k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(DfnTest, DfnAsEachOperand) {
    // {omega*2}each 1 2 3 should return 2 4 6
    Value* result = machine->eval("{⍵×2}¨1 2 3");

    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    ASSERT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 6.0);
}

TEST_F(DfnTest, DfnAsScanOperand) {
    // {alpha+omega}\1 2 3 4 should return 1 3 6 10
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

TEST_F(DfnTest, DfnAsOuterProduct) {
    // 1 2 outer.{alpha*omega} 3 4 should return 2x2 matrix: 3 4 / 6 8
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
// GC Stress with Dfns
// ============================================================================

TEST_F(DfnTest, GCStressNestedDfnCalls) {
    // Multiple function definitions and calls creating many environments
    machine->eval("F←{⍵+1}");
    machine->eval("G←{F F ⍵}");
    machine->eval("H←{G G ⍵}");
    Value* result = machine->eval("H 0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);  // 0+1+1+1+1
}

// ============================================================================
// ISO 13751 Section 13: Guard Edge Cases
// ============================================================================

TEST_F(DfnTest, GuardNonBooleanScalar) {
    // Guard with value 2 - should be treated as truthy (non-zero)
    Value* result = machine->eval("{2: 'yes' ⋄ 'no'}1");
    ASSERT_NE(result, nullptr);
    // Non-zero values are truthy
    EXPECT_TRUE(result->is_string() || result->is_char_data());
}

TEST_F(DfnTest, GuardZeroIsFalsy) {
    // Guard with 0 should fall through
    Value* result = machine->eval("{0: 'yes' ⋄ 'no'}1");
    ASSERT_NE(result, nullptr);
}

TEST_F(DfnTest, GuardVectorResultError) {
    // Guard with vector result should error (must be scalar)
    EXPECT_THROW(machine->eval("{(1 0): 'yes' ⋄ 'no'}1"), APLError);
}

TEST_F(DfnTest, GuardExpressionEvaluated) {
    // Guard expression should be fully evaluated
    Value* result = machine->eval("{(3>2): 'yes' ⋄ 'no'}1");
    ASSERT_NE(result, nullptr);
}

// ============================================================================
// ISO 13751 Section 13: Local Variable Scoping
// ============================================================================

TEST_F(DfnTest, LocalVariableShadowsGlobal) {
    // Local variable should shadow global
    machine->eval("x←100");
    Value* result = machine->eval("{x←5 ⋄ x}1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);

    // Global should be unchanged
    Value* global = machine->eval("x");
    ASSERT_NE(global, nullptr);
    EXPECT_DOUBLE_EQ(global->as_scalar(), 100.0);
}

TEST_F(DfnTest, LocalVariableDoesNotLeak) {
    // Variable defined inside dfn should not leak out
    machine->eval("{localvar←42 ⋄ localvar}1");

    // Accessing localvar should error (undefined)
    EXPECT_THROW(machine->eval("localvar"), APLError);
}

TEST_F(DfnTest, NestedDfnScoping) {
    // Inner dfn should see outer dfn's locals
    Value* result = machine->eval("{x←10 ⋄ {x+⍵}5}1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(DfnTest, GlobalVariableAccessible) {
    // Dfn should be able to read global variable
    machine->eval("globalval←99");
    Value* result = machine->eval("{globalval+⍵}1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 100.0);
}

// ============================================================================
// ISO 13751 Section 13: Result Handling
// ============================================================================

TEST_F(DfnTest, DfnNoExplicitResult) {
    // Dfn with only assignment - result is the assigned value
    Value* result = machine->eval("{x←42}1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(DfnTest, DfnMultipleStatementsLastResult) {
    // Result is the last evaluated expression
    Value* result = machine->eval("{1 ⋄ 2 ⋄ 3}0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(DfnTest, DfnGuardEarlyReturn) {
    // Guard causes early return, subsequent code not executed
    Value* result = machine->eval("{⍵>0: 'positive' ⋄ ⍵<0: 'negative' ⋄ 'zero'}5");
    ASSERT_NE(result, nullptr);
    // First guard matches, returns immediately
}

// ============================================================================
// ISO 13751 Section 13: Error Handling
// ============================================================================

TEST_F(DfnTest, ErrorInDfnPropagates) {
    // Error inside dfn should propagate
    EXPECT_THROW(machine->eval("{1÷0}1"), APLError);
}

TEST_F(DfnTest, ErrorInNestedDfnPropagates) {
    // Error in nested dfn should propagate through
    machine->eval("F←{1÷⍵}");
    EXPECT_THROW(machine->eval("{F 0}1"), APLError);
}

TEST_F(DfnTest, UndefinedVariableError) {
    // Reference to undefined variable should error
    EXPECT_THROW(machine->eval("{undefined_var}1"), APLError);
}

// ============================================================================
// ISO 13751 Section 13: Recursion Edge Cases
// ============================================================================

TEST_F(DfnTest, DyadicRecursionInDfn) {
    // ∇ can be called dyadically within dfn
    Value* result = machine->eval("{⍵≤1: ⍵ ⋄ ⍵+∇ ⍵-1}5");
    ASSERT_NE(result, nullptr);
    // 5 + 4 + 3 + 2 + 1 = 15
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(DfnTest, ClosureWithGuardAndAlpha) {
    // Dfn that uses ⍺ in guard result should create closure
    Value* result = machine->eval("{⍵≤0: ⍺ ⋄ ⍵}");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

TEST_F(DfnTest, ClosureWithMonadicRecursion) {
    // Dfn with monadic ∇ should create closure
    Value* result = machine->eval("{⍵≤0: ⍵ ⋄ ∇ ⍵-1}");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

TEST_F(DfnTest, ClosureWithDyadicRecursion) {
    // Dfn with dyadic ∇ should create closure
    Value* result = machine->eval("{⍵≤0: ⍺ ⋄ (⍺+⍵)∇ ⍵-1}");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CLOSURE);
}

TEST_F(DfnTest, RecursionWithAccumulator) {
    // Apply dyadic recursion: 0 f 5 = sum from 5 to 1 = 15
    Value* sum = machine->eval("0{⍵≤0: ⍺ ⋄ (⍺+⍵)∇ ⍵-1}5");
    ASSERT_NE(sum, nullptr);
    EXPECT_DOUBLE_EQ(sum->as_scalar(), 15.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
