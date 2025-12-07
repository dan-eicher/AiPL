// G2 Grammar Comprehensive Tests
// Based on "Parsing and Evaluation of APL with Operators" (Georgeff et al.)
//
// Tests the G2 grammar rules and evaluation semantics from Table 1:
// 1. fbn-term ::= fb-term fbn-term (juxtaposition)
// 2. fb-term ::= fb-term monadic-operator
// 3. fb-term ::= derived-operator fb
// 4. derived-operator ::= fb-term dyadic-operator

#include <gtest/gtest.h>
#include "machine.h"
#include "parser.h"
#include "value.h"

using namespace apl;

class GrammarTest : public ::testing::Test {
protected:
    Machine* machine = nullptr;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
    }

    // Helper: evaluate an expression (delegates to machine)
    Value* eval(const std::string& input) {
        Value* result = machine->eval(input);
        if (!result) {
            ADD_FAILURE() << "Eval failed: " << machine->parser->get_error();
        }
        return result;
    }
};

// ============================================================================
// G2 Rule 1: fbn-term ::= fb-term fbn-term (Juxtaposition)
// Semantics: if type(x₁) = bas then x₂(x₁) else x₁(x₂)
// ============================================================================

// Test: Basic value as left operand (bas × v → v(bas))
TEST_F(GrammarTest, JuxtapositionBasicLeft) {
    // "5 -" is invalid syntax (- needs right operand)
    // Instead test: value function value → middle applies to left
    // "5 + 3" with left-associative parsing: (5 +) 3
    // 5 is basic, so +(5) creates curried function, then apply to 3
    Value* result = eval("5 + 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);  // 5 + 3
}

// Test: Function as left operand (v × bas → v(bas))
TEST_F(GrammarTest, JuxtapositionFunctionLeft) {
    // "- 5" should parse as: - is not basic, so apply -(5)
    Value* result = eval("- 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(GrammarTest, JuxtapositionRightAssociative) {
    // G2 Grammar: "2 + 3 × 4" should parse right-associatively as: 2 (+ (3 (× 4)))
    // Evaluation (right-to-left in APL):
    // 3 × 4: 3 is basic, so (× 4) curries to a function that multiplies by 4
    // + (× 4): + is function, (× 4) is basic (curried fn), so (+ (× 4)) curries
    // 2 (+ (× 4)): applies the composition to 2
    // Result: 2 + (3 × 4) = 2 + 12 = 14
    Value* result = eval("2 + 3 × 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test: Chain of function applications
TEST_F(GrammarTest, JuxtapositionChain) {
    // "- - - 5" should parse left-to-right as ((- -) -) 5
    // But each "-" is a function, so we need to handle function composition
    // Actually, let's test something simpler first

    // "- + 5" with left-associative parsing:
    // (- +) 5
    // - is function, + is function, so - applies to + (composition? error?)
    // This gets complex - let me test the paper's example instead
}

// Test: Dyadic application via currying (g' transformation)
TEST_F(GrammarTest, DyadicViaGPrime) {
    // "2 + 3" should work as:
    // Left-to-right: (2 +) 3
    // 2 is basic, so +(2) creates curried function +₂
    // Then +₂(3) = 2+3 = 5
    Value* result = eval("2 + 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test: Multiple dyadic applications
TEST_F(GrammarTest, MultipleDyadics) {
    // "2 + 3 × 4" with APL right-to-left evaluation:
    // 2 + (3 × 4)
    // = 2 + 12
    // = 14
    Value* result = eval("2 + 3 × 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test: Right-to-left evaluation within expression
TEST_F(GrammarTest, RightToLeftEval) {
    // APL evaluates right-to-left for dyadic operations
    // "2 × 3 + 4" should be 2 × (3 + 4) = 2 × 7 = 14
    Value* result = eval("2 × 3 + 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// ============================================================================
// G2 Rule 2: fb-term ::= fb-term monadic-operator
// Semantics: x₂(x₁) - operator takes operand to left
// ============================================================================

TEST_F(GrammarTest, MonadicOperatorLeft) {
    // "+/ 1 2 3" should parse as (+ /) (1 2 3)
    // / takes + to its left, creating reduce-with-plus
    // Then apply to vector 1 2 3 → 6
    Value* result = eval("+/1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(GrammarTest, MonadicOperatorChain) {
    // Test multiple monadic operators
    // "f o1 o2" should parse as (f o1) o2
    // Each operator takes the result to its left

    // Using real operators: "+ / /" would be reduce-reduce-plus
    // But that doesn't make semantic sense
    // Let's test with each operator: "+ ¨" (plus each)
    Value* result = eval("+¨1 2 3");
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
TEST_F(GrammarTest, InnerProductParseCheck) {
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

TEST_F(GrammarTest, DerivedOperatorRight) {
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

    Value* result = eval("3 4 +.× 5 6");
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

TEST_F(GrammarTest, DyadicOperatorCurrying) {
    // "+. ×" should parse as: + is fb-term, . is dyadic-operator
    // Creates derived operator (+.)
    // Then × is the right operand: (+.) ×
    // This creates the inner product operator +.×

    // Full expression: "3 4 +.× 5 6"
    // Already tested above in DerivedOperatorRight

    // Let's test outer product instead: "3 4 ∘.× 5 6"
    Value* result = eval("3 4 ∘.× 5 6");
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

TEST_F(GrammarTest, GPrimeNullCase_Plus) {
    // "+ 3" at top level: + creates G_PRIME(+, 3), y is null → monadic +
    // Monadic + is conjugate (identity for reals)
    Value* result = eval("+ 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(GrammarTest, GPrimeNullCase_Iota) {
    // "⍳5" at top level: ⍳ has both forms, creates G_PRIME(⍳, 5)
    // y is null → apply monadic ⍳ → 0 1 2 3 4
    Value* result = eval("⍳5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(4, 0), 4.0);
}

TEST_F(GrammarTest, GPrimeNullCase_Minus) {
    // "- 5" at top level: - has both forms, creates G_PRIME(-, 5)
    // y is null → apply monadic - → -5
    Value* result = eval("- 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(GrammarTest, GPrimeNullCase_Rho) {
    // "⍴ 1 2 3" at top level: monadic ⍴ gives shape
    Value* result = eval("⍴ 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 1);  // Shape of 3-element vector is [3]
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);
}

// ---------------------------------------------------------------------------
// Case 2: bas(y) - Dyadic application when second arg is basic value
// ---------------------------------------------------------------------------

TEST_F(GrammarTest, GPrimeBasicCase_Plus) {
    // "2 + 3" → +(2) creates G_PRIME, sees 3 (basic) → dyadic + → 5
    Value* result = eval("2 + 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(GrammarTest, GPrimeBasicCase_Iota) {
    // "1 2 3 ⍳ 2" → dyadic ⍳ (index-of): find 2 in vector 1 2 3 → index 1
    Value* result = eval("1 2 3 ⍳ 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(GrammarTest, GPrimeBasicCase_IotaVector) {
    // "1 2 3 ⍳ 3 1 5" → find indices of 3,1,5 in 1 2 3 → 2 0 3(not found)
    Value* result = eval("1 2 3 ⍳ 3 1 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);  // 3 is at index 2
    EXPECT_DOUBLE_EQ((*m)(1, 0), 0.0);  // 1 is at index 0
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);  // 5 not found, returns length
}

TEST_F(GrammarTest, GPrimeBasicCase_Minus) {
    // "10 - 3" → dyadic - (subtract) → 7
    Value* result = eval("10 - 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(GrammarTest, GPrimeBasicCase_Rho) {
    // "2 3 ⍴ 1 2 3 4 5 6" → dyadic ⍴ (reshape) → 2x3 matrix
    Value* result = eval("2 3 ⍴ 1 2 3 4 5 6");
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

TEST_F(GrammarTest, GPrimeFunctionCase_IotaLessThan) {
    // "(⍳5) < 3" is the canonical test case
    // Parse: ((⍳ 5) <) 3
    // 1. ⍳ sees 5 (basic) → creates G_PRIME(⍳, 5)
    // 2. G_PRIME sees < (function) → apply ⍳ monadically: ⍳5 = 0 1 2 3 4
    //    Then < sees 0 1 2 3 4 (basic) → creates G_PRIME(<, 0 1 2 3 4)
    // 3. G_PRIME(<, 0 1 2 3 4) sees 3 (basic) → dyadic <: (0 1 2 3 4) < 3
    // Result: 1 1 1 0 0
    Value* result = eval("(⍳5) < 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);  // 0 < 3 = true
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // 1 < 3 = true
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // 2 < 3 = true
    EXPECT_DOUBLE_EQ((*m)(3, 0), 0.0);  // 3 < 3 = false
    EXPECT_DOUBLE_EQ((*m)(4, 0), 0.0);  // 4 < 3 = false
}

TEST_F(GrammarTest, GPrimeFunctionCase_IotaEquals) {
    // "(⍳5) = 2" → ⍳5 = 0 1 2 3 4, then (0 1 2 3 4) = 2 → 0 0 1 0 0
    Value* result = eval("(⍳5) = 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);  // 0 = 2 → 0
    EXPECT_DOUBLE_EQ((*m)(1, 0), 0.0);  // 1 = 2 → 0
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // 2 = 2 → 1
    EXPECT_DOUBLE_EQ((*m)(3, 0), 0.0);  // 3 = 2 → 0
    EXPECT_DOUBLE_EQ((*m)(4, 0), 0.0);  // 4 = 2 → 0
}

TEST_F(GrammarTest, GPrimeFunctionCase_NegateAdd) {
    // "(- 5) + 3" → negate 5 first, then add 3
    // - sees 5 (basic) → G_PRIME(-, 5)
    // G_PRIME sees + (function) → apply - monadically: -5
    // + sees -5 (basic) → G_PRIME(+, -5)
    // G_PRIME sees 3 (basic) → dyadic +: -5 + 3 = -2
    Value* result = eval("(- 5) + 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -2.0);
}

TEST_F(GrammarTest, GPrimeFunctionCase_IotaTimes) {
    // "(⍳4) × 2" → (0 1 2 3) × 2 → 0 2 4 6
    Value* result = eval("(⍳4) × 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), 6.0);
}

TEST_F(GrammarTest, GPrimeFunctionCase_ChainedComposition) {
    // "(- 5) + - 3" → complex case with multiple function compositions
    // Parse: (((- 5) +) -) 3
    // - 5 → G_PRIME(-, 5), sees + → monadic -5, + creates G_PRIME(+, -5)
    // G_PRIME(+, -5) sees - → apply monadic +(-5)=-5, - creates G_PRIME(-, -5)
    // G_PRIME(-, -5) sees 3 → dyadic -: -5 - 3 = -8
    Value* result = eval("(- 5) + - 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -8.0);
}

TEST_F(GrammarTest, GPrimeFunctionCase_IotaWithOperator) {
    // "(⍳5) +/ 1 2 3 4 5" - tests that iota resolves before operator expression
    // ⍳5 → 0 1 2 3 4, then... wait this is complex
    // Let's test simpler: "+/ ⍳5" should sum 0+1+2+3+4 = 10
    Value* result = eval("+/ ⍳5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(GrammarTest, GPrimeFunctionCase_IotaPlusIota) {
    // "(⍳3) + ⍳3" → (0 1 2) + (0 1 2) → 0 2 4
    // First ⍳3 creates G_PRIME, sees + (function), applies monadically
    // + creates G_PRIME(+, 0 1 2), sees ⍳ (function), applies monadically
    // But ⍳ 3 creates G_PRIME(⍳, 3), which at end resolves to 0 1 2
    // Then dyadic +: (0 1 2) + (0 1 2) = 0 2 4
    Value* result = eval("(⍳3) + ⍳3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 4.0);
}

TEST_F(GrammarTest, GPrimeFunctionCase_Member) {
    // "⍳5 ∊ 2 3 7" → first ⍳5 = 0 1 2 3 4, then membership test
    // (0 1 2 3 4) ∊ (2 3 7) → 0 0 1 1 0
    Value* result = eval("(⍳5) ∊ 2 3 7");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);  // 0 not in {2,3,7}
    EXPECT_DOUBLE_EQ((*m)(1, 0), 0.0);  // 1 not in {2,3,7}
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // 2 in {2,3,7}
    EXPECT_DOUBLE_EQ((*m)(3, 0), 1.0);  // 3 in {2,3,7}
    EXPECT_DOUBLE_EQ((*m)(4, 0), 0.0);  // 4 not in {2,3,7}
}

// ---------------------------------------------------------------------------
// Combined/stress tests for g' transformation
// ---------------------------------------------------------------------------

TEST_F(GrammarTest, GPrimeMixedExpression) {
    // "1 + (⍳3) × 2" → APL right-to-left: 1 + ((⍳3) × 2)
    // (⍳3) × 2 → (0 1 2) × 2 → 0 2 4
    // 1 + (0 2 4) → 1 3 5
    Value* result = eval("1 + (⍳3) × 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 5.0);
}

TEST_F(GrammarTest, GPrimeWithReduce) {
    // "+/ (⍳4) × 2" → ⍳4 = 0 1 2 3, × 2 → 0 2 4 6, +/ → 12
    Value* result = eval("+/ (⍳4) × 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);
}

TEST_F(GrammarTest, GPrimeComparisonInExpression) {
    // Test the original failing case more thoroughly
    // "1 + (⍳5) < 3" → (⍳5) < 3 = 1 1 1 0 0, then 1 + that = 2 2 2 1 1
    Value* result = eval("1 + (⍳5) < 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(4, 0), 1.0);
}

TEST_F(GrammarTest, GPrimeLogicalExpression) {
    // "(⍳5) > 2 ∧ (⍳5) < 4" - should find 3 (indices where 2 < x < 4)
    // ⍳5 = 0 1 2 3 4
    // (⍳5) > 2 = 0 0 0 1 1
    // (⍳5) < 4 = 1 1 1 1 0
    // Result: 0 0 0 1 0 (only index 3 satisfies both)
    Value* result = eval("((⍳5) > 2) ∧ (⍳5) < 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 5);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(4, 0), 0.0);
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_F(GrammarTest, NestedParentheses) {
    // Parentheses should force evaluation order
    // "(2 + 3) × 4" should be 5 × 4 = 20
    Value* result = eval("(2 + 3) × 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

TEST_F(GrammarTest, ComplexNesting) {
    // "((2 + 3) × 4) - 1" = 20 - 1 = 19
    Value* result = eval("((2 + 3) × 4) - 1");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 19.0);
}

TEST_F(GrammarTest, VectorOperations) {
    // "1 2 3 + 4 5 6" with vectors
    Value* result = eval("1 2 3 + 4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0,0), 5.0);
    EXPECT_DOUBLE_EQ((*m)(1,0), 7.0);
    EXPECT_DOUBLE_EQ((*m)(2,0), 9.0);
}

TEST_F(GrammarTest, ScalarVectorMixed) {
    // "5 + 1 2 3" should broadcast: 5+1=6, 5+2=7, 5+3=8
    Value* result = eval("5 + 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0,0), 6.0);
    EXPECT_DOUBLE_EQ((*m)(1,0), 7.0);
    EXPECT_DOUBLE_EQ((*m)(2,0), 8.0);
}

TEST_F(GrammarTest, ReductionWithOperators) {
    // "+/×/1 2 3 4" should parse as (+/) ((×/) (1 2 3 4))
    // ×/ 1 2 3 4 = 1×2×3×4 = 24
    // +/ 24 = 24 (reduction of single element)
    Value* result = eval("+/×/1 2 3 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 24.0);
}

TEST_F(GrammarTest, CommuteDuplicate) {
    // "2 +⍨ 3" should be 3 + 2 (commuted) = 5
    Value* result = eval("2 +⍨ 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(GrammarTest, EachOperator) {
    // "×¨ 1 2 3" should apply × monadically to each element
    // Monadic × is sign function: sign(1)=1, sign(2)=1, sign(3)=1
    Value* result = eval("×¨ 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0,0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1,0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(2,0), 1.0);
}

TEST_F(GrammarTest, LexicalStrandVsJuxtaposition) {
    // Test distinction between lexical strand and runtime juxtaposition
    // "1 2 3" is a LEXICAL STRAND (single TOK_NUMBER_VECTOR token) - creates vector [1,2,3]
    Value* strand = eval("1 2 3");
    ASSERT_NE(strand, nullptr);
    EXPECT_TRUE(strand->is_array());
    Eigen::MatrixXd* m1 = strand->as_matrix();
    EXPECT_EQ(m1->rows(), 3);
    EXPECT_DOUBLE_EQ((*m1)(0,0), 1.0);
    EXPECT_DOUBLE_EQ((*m1)(1,0), 2.0);
    EXPECT_DOUBLE_EQ((*m1)(2,0), 3.0);

    // But "- 1 2 3" is JUXTAPOSITION: (- (1 2 3))
    // The minus function applied to the strand [1,2,3] = [-1,-2,-3]
    Value* result = eval("- 1 2 3");
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

TEST_F(GrammarTest, RankMonadicFullRankSimple) {
    // -⍤2 on a simple 2x3 matrix (full rank = apply to whole matrix)
    Value* result = eval("-⍤2 (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
}

TEST_F(GrammarTest, RankMonadicFullRank) {
    // -⍤2 on matrix → applies - to whole matrix (rank 2 = full)
    Value* result = eval("-⍤2 (2 3⍴⍳6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);   // -0
    EXPECT_DOUBLE_EQ((*m)(0, 1), -1.0);  // -1
    EXPECT_DOUBLE_EQ((*m)(1, 2), -5.0);  // -5
}

TEST_F(GrammarTest, RankMonadicRank0Vector) {
    // -⍤0 vector → applies - to each scalar (0-cells)
    Value* result = eval("-⍤0 (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), -3.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), -4.0);
}

TEST_F(GrammarTest, RankMonadicRank1Matrix) {
    // -⍤1 on matrix → applies - to each row (1-cells)
    Value* result = eval("-⍤1 (3 2⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_EQ(m->cols(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), -2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 1), -6.0);
}

TEST_F(GrammarTest, RankDyadicFullRank) {
    // A +⍤2 B → applies + to whole arrays
    Value* result = eval("1 2 3 +⍤2 (10 20 30)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 22.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 33.0);
}

TEST_F(GrammarTest, RankDyadicRank0) {
    // A +⍤0 B → element-wise (same as regular +)
    Value* result = eval("1 2 3 +⍤0 (10 20 30)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 22.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 33.0);
}

TEST_F(GrammarTest, RankScalarArg) {
    // -⍤0 on scalar → just negate it
    Value* result = eval("-⍤0 (42)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -42.0);
}

TEST_F(GrammarTest, RankWithReductionSimple) {
    // Verify +/ works on a simple vector
    Value* result = eval("+/ (1 2)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(GrammarTest, RankWithReduction) {
    // +/⍤1 on matrix → sum each row
    // Matrix is 3×2, sum each row gives vector of 3 sums
    Value* result = eval("+/⍤1 (3 2⍴1 2 3 4 5 6)");
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

TEST_F(GrammarTest, ReduceVector) {
    // +/ 1 2 3 4 → 10
    Value* result = eval("+/ (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(GrammarTest, ReduceVectorMultiply) {
    // ×/ 1 2 3 4 → 24
    Value* result = eval("×/ (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 24.0);
}

TEST_F(GrammarTest, ReduceMatrix) {
    // +/ on 2×3 matrix → vector [6, 15]
    Value* result = eval("+/ (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ((*m)(1, 0), 15.0);  // 4+5+6
}

TEST_F(GrammarTest, ReduceFirstMatrix) {
    // +⌿ on 2×3 matrix → vector [5, 7, 9]
    Value* result = eval("+⌿ (2 3⍴1 2 3 4 5 6)");
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

TEST_F(GrammarTest, ScanVector) {
    // +\ 1 2 3 4 → 1 3 6 10
    Value* result = eval("+\\ (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), 10.0);
}

TEST_F(GrammarTest, ScanVectorNonAssociative) {
    // -\ 1 2 3 4 → 1 -1 2 -2 (prefix reductions)
    Value* result = eval("-\\ (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -1.0);  // 1-2
    EXPECT_DOUBLE_EQ((*m)(2, 0), 2.0);   // 1-(2-3)
    EXPECT_DOUBLE_EQ((*m)(3, 0), -2.0);  // 1-(2-(3-4))
}

TEST_F(GrammarTest, ScanMatrix) {
    // +\ on 2×3 matrix → prefix sums along rows
    Value* result = eval("+\\ (2 3⍴1 2 3 4 5 6)");
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

TEST_F(GrammarTest, ScanFirstMatrix) {
    // +⍀ on 2×3 matrix → prefix sums along columns
    Value* result = eval("+⍀ (2 3⍴1 2 3 4 5 6)");
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

TEST_F(GrammarTest, EachScalar) {
    // -¨5 → -5
    Value* result = eval("-¨ 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(GrammarTest, EachVector) {
    // -¨1 2 3 → -1 -2 -3
    Value* result = eval("-¨ (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), -3.0);
}

TEST_F(GrammarTest, EachMatrix) {
    // -¨ on 2×2 matrix → negate each element
    Value* result = eval("-¨ (2 2⍴1 2 3 4)");
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

TEST_F(GrammarTest, ReduceWithDerivedOperator) {
    // Use commute with reduce: +⍨/ (1 2 3)
    // This is (+⍨)/ which means reduce using the commuted plus
    // Since + is commutative, result is same as +/: 6
    Value* result = eval("+⍨/ (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(GrammarTest, ReduceWithDerivedOperatorNonCommutative) {
    // -⍨/ (10 3 1) → reduce with commuted minus
    // -⍨ means {⍵-⍺}, so A -⍨ B = B - A
    // Right-to-left: 10 -⍨ (3 -⍨ 1)
    //   3 -⍨ 1 = 1 - 3 = -2
    //   10 -⍨ (-2) = -2 - 10 = -12
    Value* result = eval("-⍨/ (10 3 1)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -12.0);
}

TEST_F(GrammarTest, ScanWithDerivedOperator) {
    // +⍨\ (1 2 3) → scan with commuted plus
    // Since + is commutative: 1, 3, 6
    Value* result = eval("+⍨\\ (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);
}

TEST_F(GrammarTest, EachWithDerivedOperator) {
    // ×⍨¨ (2 3 4) → square each element (x times itself)
    // Results: 4, 9, 16
    Value* result = eval("×⍨¨ (2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 4.0);   // 2×2
    EXPECT_DOUBLE_EQ((*m)(1, 0), 9.0);   // 3×3
    EXPECT_DOUBLE_EQ((*m)(2, 0), 16.0);  // 4×4
}

TEST_F(GrammarTest, EachWithReduceDerived) {
    // (+/)¨ applied to vectors would reduce each
    // But we don't have nested arrays yet, so test with matrix rows
    // For now, test that derived operators chain: +/¨ is valid syntax
    // Apply to a simple case
    Value* result = eval("+/¨ (1 2 3)");
    ASSERT_NE(result, nullptr);
    // Each element is a scalar, +/ of scalar is the scalar
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

TEST_F(GrammarTest, NestedDerivedOperators) {
    // -⍨⍨ negates the commute (back to normal minus order)
    // -⍨⍨/ (10 3) → same as -/ (10 3) = 10-3 = 7
    Value* result = eval("-⍨⍨/ (10 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(GrammarTest, MatrixReduceWithDerivedOperator) {
    // ×⍨/ on 2×3 matrix → product of each row (since × is commutative)
    // Row 0: 1×2×3 = 6, Row 1: 4×5×6 = 120
    Value* result = eval("×⍨/ (2 3⍴1 2 3 4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 120.0);
}

TEST_F(GrammarTest, MatrixScanWithDerivedOperator) {
    // +⍨\ on 2×3 matrix → cumulative sums along rows
    Value* result = eval("+⍨\\ (2 3⍴1 2 3 4 5 6)");
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

TEST_F(GrammarTest, OuterProductScalars) {
    // 5 ∘.+ 3 → 8
    Value* result = eval("5 ∘.+ 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

TEST_F(GrammarTest, OuterProductVectorVector) {
    // (1 2 3) ∘.+ (10 20) → 2D matrix
    Value* result = eval("(1 2 3) ∘.+ (10 20)");
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

TEST_F(GrammarTest, OuterProductMultiply) {
    // (2 3) ∘.× (10 100) → multiplication table
    Value* result = eval("(2 3) ∘.× (10 100)");
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

TEST_F(GrammarTest, OuterProductWithDerivedOperator) {
    // (1 2) ∘.+⍨ (10 20) → uses commuted plus (same result since + is commutative)
    Value* result = eval("(1 2) ∘.+⍨ (10 20)");
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

TEST_F(GrammarTest, OuterProductWithCommuteNonCommutative) {
    // (10 20) ∘.-⍨ (1 2) → -⍨ swaps args: (1-10, 1-20, 2-10, 2-20)
    Value* result = eval("(10 20) ∘.-⍨ (1 2)");
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

TEST_F(GrammarTest, InnerProductVectorDot) {
    // (1 2 3) +.× (4 5 6) → dot product: 1×4 + 2×5 + 3×6 = 32
    Value* result = eval("(1 2 3) +.× (4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 32.0);
}

TEST_F(GrammarTest, InnerProductMatrixMultiply) {
    // Matrix multiplication: (2 2⍴1 2 3 4) +.× (2 2⍴5 6 7 8)
    // [1 2] × [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    Value* result = eval("(2 2⍴1 2 3 4) +.× (2 2⍴5 6 7 8)");
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

TEST_F(GrammarTest, InnerProductMatrixVector) {
    // Matrix × vector: (2 2⍴1 2 3 4) +.× (5 7)
    Value* result = eval("(2 2⍴1 2 3 4) +.× (5 7)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 19.0);  // 1×5 + 2×7
    EXPECT_DOUBLE_EQ((*m)(1, 0), 43.0);  // 3×5 + 4×7
}

TEST_F(GrammarTest, InnerProductWithDerivedOperator) {
    // Use commuted multiply: (1 2 3) +.×⍨ (4 5 6)
    // ×⍨ is same as × (commutative), so same result as dot product
    Value* result = eval("(1 2 3) +.×⍨ (4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 32.0);
}

TEST_F(GrammarTest, InnerProductDifferentOperators) {
    // (10 20 30) +.÷ (2 4 5) → 10÷2 + 20÷4 + 30÷5 = 5 + 5 + 6 = 16
    Value* result = eval("(10 20 30) +.÷ (2 4 5)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 16.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
