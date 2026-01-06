// Tests for APL operators

#include <gtest/gtest.h>
#include "machine.h"
#include "heap.h"
#include "primitives.h"
#include "operators.h"
#include "continuation.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

using namespace apl;

class OperatorsTest : public ::testing::Test {
protected:
    Machine* machine;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
    }
};

// Helper to evaluate APL expression
static Value* eval(Machine* m, const char* expr) {
    return m->eval(expr);
}

// ========================================================================
// Inner Product Error Tests (functional tests are in test_grammar.cpp)
// ========================================================================

TEST_F(OperatorsTest, InnerProductDimensionMismatch) {
    // Test LENGTH ERROR when dimensions don't match
    Eigen::VectorXd lhs_vec(3);
    lhs_vec << 1, 2, 3;
    Value* lhs = machine->heap->allocate_vector(lhs_vec);

    Eigen::VectorXd rhs_vec(2);
    rhs_vec << 4, 5;
    Value* rhs = machine->heap->allocate_vector(rhs_vec);

    Value* f = machine->heap->allocate_primitive(&prim_plus);
    Value* g = machine->heap->allocate_primitive(&prim_times);

    machine->kont_stack.clear();
    op_inner_product(machine, nullptr, lhs, f, g, rhs);

    // Should have pushed a ThrowErrorK
    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// ========================================================================
// Duplicate/Commute Operator Tests
// ========================================================================

TEST_F(OperatorsTest, DuplicateScalar) {
    // +⍨3 → 3+3 = 6
    Value* result = eval(machine, "+⍨3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(OperatorsTest, DuplicateVector) {
    // ×⍨(2 3 4) → (2 3 4) × (2 3 4) (element-wise)
    Value* result = eval(machine, "×⍨(2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 4.0);   // 2*2
    EXPECT_DOUBLE_EQ((*m)(1, 0), 9.0);   // 3*3
    EXPECT_DOUBLE_EQ((*m)(2, 0), 16.0);  // 4*4
}

TEST_F(OperatorsTest, CommuteScalars) {
    // 3-⍨4 → 4-3 = 1 (commute swaps arguments)
    Value* result = eval(machine, "3-⍨4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 4-3, not 3-4
}

TEST_F(OperatorsTest, CommuteVectors) {
    // (10 20 30) -⍨ (1 2 3) → (1 2 3) - (10 20 30) = ¯9 ¯18 ¯27
    Value* result = eval(machine, "(10 20 30)-⍨(1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());

    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), -9.0);   // 1-10
    EXPECT_DOUBLE_EQ((*m)(1, 0), -18.0);  // 2-20
    EXPECT_DOUBLE_EQ((*m)(2, 0), -27.0);  // 3-30
}

TEST_F(OperatorsTest, ReplicateViaReduce) {
    // When reduce receives an array as "function", it's replicate
    // 1 2 3 / 4 5 6 → 4 5 5 6 6 6
    Eigen::VectorXd counts(3);
    counts << 1.0, 2.0, 3.0;
    Value* count_vec = machine->heap->allocate_vector(counts);

    Eigen::VectorXd data(3);
    data << 4.0, 5.0, 6.0;
    Value* data_vec = machine->heap->allocate_vector(data);

    fn_reduce(machine, nullptr, count_vec, data_vec);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* result = machine->result->as_matrix();
    EXPECT_EQ(result->rows(), 6);  // 1+2+3 = 6
    EXPECT_DOUBLE_EQ((*result)(0, 0), 4.0);  // 1 copy of 4
    EXPECT_DOUBLE_EQ((*result)(1, 0), 5.0);  // 2 copies of 5
    EXPECT_DOUBLE_EQ((*result)(2, 0), 5.0);
    EXPECT_DOUBLE_EQ((*result)(3, 0), 6.0);  // 3 copies of 6
    EXPECT_DOUBLE_EQ((*result)(4, 0), 6.0);
    EXPECT_DOUBLE_EQ((*result)(5, 0), 6.0);
}

TEST_F(OperatorsTest, ErrorScanNonFunction) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_scan(machine, nullptr, vec, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// ========================================================================
// Additional Duplicate/Commute Tests
// ========================================================================

TEST_F(OperatorsTest, DuplicateWithDivide) {
    // ÷⍨4 → 4÷4 = 1
    Value* result = eval(machine, "÷⍨4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, CommuteWithDivide) {
    // 3÷⍨12 → 12÷3 = 4
    Value* result = eval(machine, "3÷⍨12");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

TEST_F(OperatorsTest, CommuteMatrix) {
    // lhs -⍨ rhs → rhs - lhs = (1-10, 2-20, 3-30, 4-40) = (-9, -18, -27, -36)
    Value* result = eval(machine, "(2 2⍴10 20 30 40)-⍨(2 2⍴1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), -9.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), -18.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -27.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), -36.0);
}

TEST_F(OperatorsTest, DuplicateMatrix) {
    // Test duplicate with matrix subtraction: -⍨ mat → mat - mat = zero matrix
    Value* result = eval(machine, "-⍨(2 2⍴5 6 7 8)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());

    const Eigen::MatrixXd* m = result->as_matrix();
    // Matrix - itself = zero matrix
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 0.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 0.0);
}

// Test postfix monadic operator: +¨ creates DERIVED_OPERATOR
TEST_F(OperatorsTest, PostfixMonadicOperator) {
    // Create continuation that looks up "+" and applies "¨" operator to it
    // This should create a DERIVED_OPERATOR that captures ¨ and +
    LookupK* lookup = machine->heap->allocate<LookupK>(machine->string_pool.intern("+"));
    const char* op_name = machine->string_pool.intern("¨");
    DerivedOperatorK* derived_k = machine->heap->allocate<DerivedOperatorK>(lookup, op_name);

    machine->push_kont(derived_k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    // Should create a DERIVED_OPERATOR value
    EXPECT_TRUE(result->is_derived_operator());
    EXPECT_EQ(result->data.derived_op->primitive_op, &op_diaeresis);
    // The first_operand should be the plus function
    EXPECT_TRUE(result->data.derived_op->first_operand->is_primitive());
}

// ========================================================================
// Rank Operator Error Tests (synchronous validation)
// Full rank operator tests are in test_grammar.cpp
// ========================================================================

TEST_F(OperatorsTest, RankRequiresFunction) {
    // Test error when first operand is not a function
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* omega = machine->heap->allocate_vector(vec);
    Value* not_fn = machine->heap->allocate_scalar(5.0);
    Value* rank_spec = machine->heap->allocate_scalar(0.0);

    machine->kont_stack.clear();
    op_rank(machine, nullptr, nullptr, not_fn, rank_spec, omega);

    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(OperatorsTest, RankCellCountMismatch) {
    // Test LENGTH ERROR when cell counts don't match
    Eigen::VectorXd vec3(3);
    vec3 << 1, 2, 3;
    Value* lhs = machine->heap->allocate_vector(vec3);

    Eigen::VectorXd vec4(4);
    vec4 << 10, 20, 30, 40;
    Value* rhs = machine->heap->allocate_vector(vec4);

    Value* fn = machine->heap->allocate_primitive(&prim_plus);
    Value* rank_spec = machine->heap->allocate_scalar(0.0);

    machine->kont_stack.clear();
    op_rank(machine, nullptr, lhs, fn, rank_spec, rhs);

    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// ========================================================================
// Axis Specification Tests (f/[k] and f\[k])
// ========================================================================

TEST_F(OperatorsTest, ReduceAxisLastOnMatrix) {
    // +/[2] is same as +/ (reduce along last axis)
    // ⍳6 = 1 2 3 4 5 6 (1-origin)
    // 2 3⍴⍳6 = [[1,2,3], [4,5,6]]
    Value* result = eval(machine, "+/[2] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    // Row sums: 1+2+3=6, 4+5+6=15
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 15.0);
}

TEST_F(OperatorsTest, ReduceAxisFirstOnMatrix) {
    // +/[1] is same as +⌿ (reduce along first axis)
    // ⍳6 = 1 2 3 4 5 6 (1-origin)
    // 2 3⍴⍳6 = [[1,2,3], [4,5,6]]
    Value* result = eval(machine, "+/[1] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // Column sums: 1+4=5, 2+5=7, 3+6=9
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 7.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 9.0);
}

TEST_F(OperatorsTest, ReduceFirstAxisEquivalent) {
    // +⌿[1] should give same result as +⌿
    Value* result1 = eval(machine, "+⌿ 2 3⍴⍳6");
    Value* result2 = eval(machine, "+⌿[1] 2 3⍴⍳6");
    ASSERT_NE(result1, nullptr);
    ASSERT_NE(result2, nullptr);
    EXPECT_TRUE(result1->is_vector());
    EXPECT_TRUE(result2->is_vector());
    EXPECT_EQ(result1->size(), result2->size());
    for (int i = 0; i < result1->size(); i++) {
        EXPECT_DOUBLE_EQ(result1->as_matrix()->coeff(i, 0),
                         result2->as_matrix()->coeff(i, 0));
    }
}

TEST_F(OperatorsTest, ScanAxisLastOnMatrix) {
    // Scan with axis [2] is same as scan along last axis
    // ⍳6 = 1 2 3 4 5 6 (1-origin)
    // 2 3⍴⍳6 = [[1,2,3], [4,5,6]]
    Value* result = eval(machine, "+\\[2] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    // Running row sums:
    // Row 0: 1, 1+2=3, 1+2+3=6
    // Row 1: 4, 4+5=9, 4+5+6=15
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 9.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 15.0);
}

TEST_F(OperatorsTest, ScanAxisFirstOnMatrix) {
    // Scan with axis [1] is same as scan-first (along first axis)
    // ⍳6 = 1 2 3 4 5 6 (1-origin)
    // 2 3⍴⍳6 = [[1,2,3], [4,5,6]]
    Value* result = eval(machine, "+\\[1] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    // Running column sums:
    // Row 0: 1, 2, 3 (first row unchanged)
    // Row 1: 1+4=5, 2+5=7, 3+6=9
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 7.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 9.0);
}

TEST_F(OperatorsTest, ReduceAxisOnVector) {
    // +/[1] on vector is same as +/
    Value* result = eval(machine, "+/[1] 1 2 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, ScanAxisOnVector) {
    // Scan with axis [1] on vector is same as regular scan
    Value* result = eval(machine, "+\\[1] 1 2 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 10.0);
    EXPECT_DOUBLE_EQ((*mat)(4, 0), 15.0);
}

TEST_F(OperatorsTest, ReduceAxisExpression) {
    // Axis can be an expression
    // ⍳6 = 1 2 3 4 5 6 (1-origin)
    // 2 3⍴⍳6 = [[1,2,3], [4,5,6]]
    Value* result = eval(machine, "+/[1+1] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    // Same as +/[2] = row sums: 1+2+3=6, 4+5+6=15
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 15.0);
}

// ============================================================================
// N-wise Reduction Tests (ISO 13751 §9.2.3)
// ============================================================================

TEST_F(OperatorsTest, NwiseReduceN1) {
    // 1 +/ 1 2 3 4 5 -> window of 1: each element (5 results)
    Value* result = eval(machine, "1 +/ 1 2 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(4, 0), 5.0);
}

TEST_F(OperatorsTest, NwiseReduceVectorPairwise) {
    // 2 +/ 1 2 3 4 5 -> pairwise sums: 3 5 7 9
    Value* result = eval(machine, "2 +/ 1 2 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);   // 1+2
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 5.0);   // 2+3
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 7.0);   // 3+4
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 9.0);   // 4+5
}

TEST_F(OperatorsTest, NwiseReduceVectorTriplets) {
    // 3 +/ 1 2 3 4 5 -> sums of 3: 6 9 12
    Value* result = eval(machine, "3 +/ 1 2 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 9.0);   // 2+3+4
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 12.0);  // 3+4+5
}

TEST_F(OperatorsTest, NwiseReduceFullVector) {
    // N = length of vector -> single result (full reduce)
    // Per ISO 9.2.3: "If M1 equals the length of B, return f/B2"
    // So 5+/1 2 3 4 5 returns the reduction result directly (scalar 15)
    Value* result = eval(machine, "5 +/ 1 2 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, NwiseReduceNonCommutative) {
    // Test with non-commutative function: subtraction
    // 2 -/ 10 3 2 1 -> (10-3) (3-2) (2-1) = 7 1 1
    Value* result = eval(machine, "2 -/ 10 3 2 1");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 7.0);   // 10-3
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);   // 3-2
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);   // 2-1
}

TEST_F(OperatorsTest, NwiseReduceMatrixAxis2) {
    // 2 +/ on 2x4 matrix (default axis 2)
    // 2 4⍴⍳8 = [[1,2,3,4],[5,6,7,8]]
    // 2 +/ gives pairwise sums: [[3,5,7],[11,13,15]]
    Value* result = eval(machine, "2 +/ 2 4⍴⍳8");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);  // 4 - 2 + 1 = 3
    auto* mat = result->as_matrix();
    // Row 0: 1+2=3, 2+3=5, 3+4=7
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 7.0);
    // Row 1: 5+6=11, 6+7=13, 7+8=15
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 11.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 13.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 15.0);
}

TEST_F(OperatorsTest, NwiseReduceMatrixAxis1) {
    // 2 +/[1] on 3x2 matrix
    // 3 2⍴⍳6 = [[1,2],[3,4],[5,6]]
    // 2 +/[1] gives pairwise sums along first axis: [[4,6],[8,10]]
    Value* result = eval(machine, "2 +/[1] 3 2⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);  // 3 - 2 + 1 = 2
    EXPECT_EQ(result->cols(), 2);
    auto* mat = result->as_matrix();
    // Col 0: 1+3=4, 3+5=8
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 8.0);
    // Col 1: 2+4=6, 4+6=10
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 10.0);
}

TEST_F(OperatorsTest, NwiseReduceFirstOperator) {
    // ⌿ with N-wise: 2 +⌿ on 3x2 matrix (default axis 1)
    // Same as 2 +/[1]
    Value* result = eval(machine, "2 +⌿ 3 2⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 8.0);
}

// --- Phase 7: N-wise Reduction Edge Cases (ISO 13751) ---

TEST_F(OperatorsTest, NwiseZeroPlus) {
    // 0+/1 2 3 → 0 0 0 0 (identity for +, repeated N+1 times)
    Value* result = eval(machine, "0+/1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);  // Length + 1 windows
    auto* mat = result->as_matrix();
    for (int i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ((*mat)(i, 0), 0.0);
    }
}

TEST_F(OperatorsTest, NwiseZeroTimes) {
    // 0×/1 2 3 → 1 1 1 1 (identity for ×)
    Value* result = eval(machine, "0×/1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    auto* mat = result->as_matrix();
    for (int i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ((*mat)(i, 0), 1.0);
    }
}

TEST_F(OperatorsTest, NwiseZeroMax) {
    // 0⌈/1 2 3 → -∞ -∞ -∞ -∞ (identity for ⌈)
    Value* result = eval(machine, "0⌈/1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    auto* mat = result->as_matrix();
    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(std::isinf((*mat)(i, 0)) && (*mat)(i, 0) < 0);
    }
}

TEST_F(OperatorsTest, NwiseNegative) {
    // ISO 13751 §9.2.3: ¯2+/1 2 3 → 3 5 (reverse each window, then reduce)
    // Windows: [1,2], [2,3] → Reversed: [2,1], [3,2]
    // Reduced: 2+1=3, 3+2=5
    Value* result = eval(machine, "¯2+/1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 5.0);
}

TEST_F(OperatorsTest, NwiseNegativeTriplets) {
    // ISO 13751 §9.2.3: ¯3-/1 2 3 4 → 2 3 (reverse each window, then reduce)
    // Windows: [1,2,3], [2,3,4] → Reversed: [3,2,1], [4,3,2]
    // Reduced (right-to-left): 3-(2-1)=2, 4-(3-2)=3
    Value* result = eval(machine, "¯3-/1 2 3 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);
}

TEST_F(OperatorsTest, NwiseTooLarge) {
    // 5+/1 2 3 → DOMAIN ERROR (window size > array length)
    EXPECT_THROW(eval(machine, "5+/1 2 3"), APLError);
}

TEST_F(OperatorsTest, NwiseOnScalar) {
    // 2+/5 → works: single scalar, window of 2 doesn't make sense
    // Per ISO 13751, this should work if we treat scalar as 1-element vector
    // 2+/ on length-1 → result length = 1-2+1 = 0 (empty)
    Value* result = eval(machine, "2+/,5");  // ,5 makes it a 1-element vector
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);  // Empty result
}

TEST_F(OperatorsTest, NwiseWindowEqualsLength) {
    // 3+/1 2 3 → single element: 1+2+3 = 6
    Value* result = eval(machine, "3+/1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar() || (result->is_vector() && result->size() == 1));
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
    } else {
        EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 6.0);
    }
}

TEST_F(OperatorsTest, ReduceAxisInvalidHigh) {
    // +/[3] on 2D matrix → DOMAIN ERROR (only axes 1,2 valid)
    EXPECT_THROW(eval(machine, "+/[3] 2 3⍴⍳6"), APLError);
}

TEST_F(OperatorsTest, ReduceAxisInvalidZero) {
    // +/[0] → DOMAIN ERROR (axes are 1-based)
    EXPECT_THROW(eval(machine, "+/[0] 1 2 3"), APLError);
}

TEST_F(OperatorsTest, ReduceAxisInvalidOnVector) {
    // +/[2] on vector → DOMAIN ERROR (only axis 1 valid for vectors)
    EXPECT_THROW(eval(machine, "+/[2] 1 2 3"), APLError);
}

TEST_F(OperatorsTest, ScanAxisInvalidHigh) {
    // +\[3] on 2D matrix → DOMAIN ERROR
    EXPECT_THROW(eval(machine, "+\\[3] 2 3⍴⍳6"), APLError);
}

TEST_F(OperatorsTest, ScanAxisInvalidZero) {
    // +\[0] → DOMAIN ERROR (axes are 1-based)
    EXPECT_THROW(eval(machine, "+\\[0] 1 2 3"), APLError);
}

TEST_F(OperatorsTest, ReduceAxisWithDifferentFunction) {
    // ×/[1] - product along first axis
    // ⍳6 = 1 2 3 4 5 6 (1-origin)
    // 2 3⍴⍳6 = [[1,2,3], [4,5,6]]
    Value* result = eval(machine, "×/[1] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // Column products: 1×4=4, 2×5=10, 3×6=18
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 18.0);
}

// ========================================================================
// Catenate/Laminate with Axis: ,[k]
// ========================================================================

TEST_F(OperatorsTest, CatenateAxisVectors) {
    // 1 2 3 ,[1] 4 5 6 - catenate vectors along axis 1
    Value* result = eval(machine, "1 2 3 ,[1] 4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 6);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(4, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(5, 0), 6.0);
}

TEST_F(OperatorsTest, LaminateVectorsAxis05) {
    // 1 2 3 ,[0.5] 4 5 6 - laminate creates 2×3 matrix
    Value* result = eval(machine, "1 2 3 ,[0.5] 4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    auto* mat = result->as_matrix();
    // Row 0: 1 2 3
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 3.0);
    // Row 1: 4 5 6
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 6.0);
}

TEST_F(OperatorsTest, LaminateVectorsAxis15) {
    // 1 2 3 ,[1.5] 4 5 6 - laminate creates 3×2 matrix
    Value* result = eval(machine, "1 2 3 ,[1.5] 4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 2);
    auto* mat = result->as_matrix();
    // Col 0: 1 2 3
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
    // Col 1: 4 5 6
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 1), 6.0);
}

TEST_F(OperatorsTest, CatenateMatrixAxis1) {
    // Vertical catenation: stack matrices along axis 1
    // (2 2⍴1 2 3 4) ,[1] (2 2⍴5 6 7 8)
    Value* result = eval(machine, "(2 2⍴1 2 3 4) ,[1] (2 2⍴5 6 7 8)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 4);
    EXPECT_EQ(result->cols(), 2);
}

TEST_F(OperatorsTest, CatenateMatrixAxis2) {
    // Horizontal catenation: stack matrices along axis 2
    // (2 2⍴1 2 3 4) ,[2] (2 2⍴5 6 7 8)
    Value* result = eval(machine, "(2 2⍴1 2 3 4) ,[2] (2 2⍴5 6 7 8)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 4);
}

// ========================================================================
// Reduction Identity Elements (ISO 13751 Table 5)
// Empty vector reduction should return the identity element
// Note: Using ⍳0 to create empty vector since ⍬ (zilde) not yet in lexer
// ========================================================================

TEST_F(OperatorsTest, ReduceEmptyPlus) {
    // +/⍳0 → 0
    Value* result = eval(machine, "+/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, ReduceEmptyMinus) {
    // -/⍳0 → 0
    Value* result = eval(machine, "-/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, ReduceEmptyTimes) {
    // ×/⍳0 → 1
    Value* result = eval(machine, "×/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, ReduceEmptyDivide) {
    // ÷/⍳0 → 1
    Value* result = eval(machine, "÷/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, ReduceEmptyMinimum) {
    // ⌊/⍳0 → +∞ (positive-number-limit per ISO 13751)
    Value* result = eval(machine, "⌊/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_TRUE(std::isinf(result->as_scalar()) && result->as_scalar() > 0);
}

TEST_F(OperatorsTest, ReduceEmptyMaximum) {
    // ⌈/⍳0 → -∞ (negative-number-limit per ISO 13751)
    Value* result = eval(machine, "⌈/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_TRUE(std::isinf(result->as_scalar()) && result->as_scalar() < 0);
}

TEST_F(OperatorsTest, ReduceEmptyAnd) {
    // ∧/⍳0 → 1 (identity for logical AND)
    Value* result = eval(machine, "∧/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, ReduceEmptyOr) {
    // ∨/⍳0 → 0 (identity for logical OR)
    Value* result = eval(machine, "∨/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, ReduceEmptyLessThan) {
    // </⍳0 → 0
    Value* result = eval(machine, "</⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, ReduceEmptyLessEqual) {
    // ≤/⍳0 → 1
    Value* result = eval(machine, "≤/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, ReduceEmptyEqual) {
    // =/⍳0 → 1
    Value* result = eval(machine, "=/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, ReduceEmptyGreaterEqual) {
    // ≥/⍳0 → 1
    Value* result = eval(machine, "≥/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, ReduceEmptyGreaterThan) {
    // >/⍳0 → 0
    Value* result = eval(machine, ">/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, ReduceEmptyNotEqual) {
    // ≠/⍳0 → 0
    Value* result = eval(machine, "≠/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, ReduceEmptyPower) {
    // */⍳0 → 1 (identity for power)
    Value* result = eval(machine, "*/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, ReduceEmptyLog) {
    // ⍟/⍳0 → DOMAIN ERROR (no identity element)
    EXPECT_THROW(eval(machine, "⍟/⍳0"), APLError);
}

TEST_F(OperatorsTest, ReduceEmptyCircle) {
    // ○/⍳0 → DOMAIN ERROR (no identity element)
    EXPECT_THROW(eval(machine, "○/⍳0"), APLError);
}

TEST_F(OperatorsTest, ReduceEmptyNand) {
    // ⍲/⍳0 → DOMAIN ERROR (no identity element)
    EXPECT_THROW(eval(machine, "⍲/⍳0"), APLError);
}

TEST_F(OperatorsTest, ReduceEmptyNor) {
    // ⍱/⍳0 → DOMAIN ERROR (no identity element)
    EXPECT_THROW(eval(machine, "⍱/⍳0"), APLError);
}

// ============================================================================
// ISO 13751 Section 9.2.1: Reduce Edge Cases
// ============================================================================

TEST_F(OperatorsTest, ReduceVector) {
    // +/1 2 3 4 → 10 (basic vector reduce)
    Value* result = eval(machine, "+/1 2 3 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(OperatorsTest, ReduceScalar) {
    // +/5 → 5 (ISO 13751: "If B is a scalar, return B")
    Value* result = eval(machine, "+/5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(OperatorsTest, ReduceSingleElement) {
    // +/,5 → 5 (single element vector reduces to scalar)
    Value* result = eval(machine, "+/,5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// ============================================================================
// ISO 13751 Section 9.2.2: Scan Edge Cases
// ============================================================================

TEST_F(OperatorsTest, ScanVector) {
    // +\1 2 3 4 → 1 3 6 10 (basic vector scan)
    Value* result = eval(machine, "+\\1 2 3 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    const Eigen::MatrixXd* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*v)(2, 0), 6.0);
    EXPECT_DOUBLE_EQ((*v)(3, 0), 10.0);
}

TEST_F(OperatorsTest, ScanScalar) {
    // +\5 → 5 (ISO 13751: "If B is a scalar, return B")
    Value* result = eval(machine, "+\\5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(OperatorsTest, ScanSingleElement) {
    // +\,5 → ,5 (single element vector stays as vector)
    Value* result = eval(machine, "+\\,5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 5.0);
}

TEST_F(OperatorsTest, ScanEmpty) {
    // +\⍬ → ⍬ (scan of empty returns empty)
    Value* result = eval(machine, "+\\⍬");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

// ============================================================================
// ISO 13751 Section 9.2.6: Each Edge Cases
// ============================================================================

TEST_F(OperatorsTest, EachDyadic) {
    // 1 2 3 +¨ 4 5 6 → 5 7 9
    Value* result = eval(machine, "1 2 3 +¨ 4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*v)(2, 0), 9.0);
}

TEST_F(OperatorsTest, EachScalarExtensionLeft) {
    // 10 -¨ 1 2 3 → 9 8 7 (scalar extends to match vector)
    Value* result = eval(machine, "10 -¨ 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), 9.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), 8.0);
    EXPECT_DOUBLE_EQ((*v)(2, 0), 7.0);
}

TEST_F(OperatorsTest, EachScalarExtensionRight) {
    // 1 2 3 +¨ 10 → 11 12 13 (scalar extends to match vector)
    Value* result = eval(machine, "1 2 3 +¨ 10");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*v)(2, 0), 13.0);
}

TEST_F(OperatorsTest, EachMonadic) {
    // -¨1 2 3 → ¯1 ¯2 ¯3 (negate each element)
    Value* result = eval(machine, "-¨1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*v)(2, 0), -3.0);
}

TEST_F(OperatorsTest, EachMonadicEmpty) {
    // -¨⍬ → ⍬ (each on empty returns empty)
    Value* result = eval(machine, "-¨⍬");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(OperatorsTest, EachDyadicEmpty) {
    // ⍬ +¨ ⍬ → ⍬ (dyadic each on empty returns empty)
    Value* result = eval(machine, "⍬ +¨ ⍬");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(OperatorsTest, EachMonadicScalar) {
    // -¨5 → ¯5 (each on scalar returns scalar)
    Value* result = eval(machine, "-¨5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(OperatorsTest, EachDyadicBothScalars) {
    // 3 +¨ 4 → 7 (both scalars)
    Value* result = eval(machine, "3 +¨ 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(OperatorsTest, EachMonadicMatrix) {
    // -¨ 2 2⍴1 2 3 4 → matrix of negated values
    Value* result = eval(machine, "-¨2 2⍴1 2 3 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), -2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), -4.0);
}

TEST_F(OperatorsTest, EachDyadicMatrix) {
    // (2 2⍴1 2 3 4) +¨ (2 2⍴10 20 30 40) → element-wise add
    Value* result = eval(machine, "(2 2⍴1 2 3 4) +¨ (2 2⍴10 20 30 40)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 22.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 33.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 44.0);
}

TEST_F(OperatorsTest, EachLengthMismatch) {
    // 1 2 +¨ 1 2 3 → LENGTH ERROR (shapes must match)
    EXPECT_THROW(eval(machine, "1 2 +¨ 1 2 3"), APLError);
}

// ============================================================================
// ISO 13751 Section 9.3: Dyadic Operators - Outer Product
// ============================================================================

TEST_F(OperatorsTest, OuterProductEmptyLeft) {
    // (⍳0) ∘.+ (1 2 3) → empty matrix with shape 0 3
    Value* result = eval(machine, "(⍳0) ∘.+ (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 0);
    EXPECT_EQ(result->cols(), 3);
}

TEST_F(OperatorsTest, OuterProductEmptyRight) {
    // (1 2 3) ∘.+ (⍳0) → empty matrix with shape 3 0
    Value* result = eval(machine, "(1 2 3) ∘.+ (⍳0)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 0);
}

TEST_F(OperatorsTest, OuterProductBothEmpty) {
    // (⍳0) ∘.+ (⍳0) → empty matrix with shape 0 0
    Value* result = eval(machine, "(⍳0) ∘.+ (⍳0)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 0);
    EXPECT_EQ(result->cols(), 0);
}

TEST_F(OperatorsTest, OuterProductScalarLeft) {
    // 5 ∘.× (1 2 3) → 1×3 matrix: [[5,10,15]]
    Value* result = eval(machine, "5 ∘.× (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 1);
    EXPECT_EQ(result->cols(), 3);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 10.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), 15.0);
}

TEST_F(OperatorsTest, OuterProductScalarRight) {
    // (1 2 3) ∘.× 5 → 3×1 matrix: [[5],[10],[15]]
    Value* result = eval(machine, "(1 2 3) ∘.× 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 1);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 10.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 15.0);
}

TEST_F(OperatorsTest, OuterProductNonCommutative) {
    // (1 2) ∘.- (10 20 30) → subtraction table
    Value* result = eval(machine, "(1 2) ∘.- (10 20 30)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    const Eigen::MatrixXd* m = result->as_matrix();
    // Row 0: 1-10=-9, 1-20=-19, 1-30=-29
    EXPECT_DOUBLE_EQ((*m)(0, 0), -9.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), -19.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), -29.0);
    // Row 1: 2-10=-8, 2-20=-18, 2-30=-28
    EXPECT_DOUBLE_EQ((*m)(1, 0), -8.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), -18.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), -28.0);
}

TEST_F(OperatorsTest, OuterProductComparison) {
    // (1 2 3) ∘.= (1 2 3) → identity-like matrix
    Value* result = eval(machine, "(1 2 3) ∘.= (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 3);
    const Eigen::MatrixXd* m = result->as_matrix();
    // Diagonal should be 1, off-diagonal 0
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 1.0);
    EXPECT_DOUBLE_EQ((*m)(2, 2), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 0.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 0.0);
}

// ============================================================================
// ISO 13751 Section 9.3: Dyadic Operators - Inner Product
// ============================================================================

TEST_F(OperatorsTest, InnerProductEmptyVectors) {
    // (⍳0) +.× (⍳0) → 0 (sum of empty products = identity for +)
    Value* result = eval(machine, "(⍳0) +.× (⍳0)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, InnerProductScalarScalar) {
    // 3 +.× 4 → 12 (scalar inner product is just multiply)
    Value* result = eval(machine, "3 +.× 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);
}

TEST_F(OperatorsTest, InnerProductVectorMatrix) {
    // (1 2) +.× (2 3⍴⍳6) → vector × matrix
    // [1,2] × [[1,2,3],[4,5,6]] = [1×1+2×4, 1×2+2×5, 1×3+2×6] = [9, 12, 15]
    Value* result = eval(machine, "(1 2) +.× (2 3⍴⍳6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), 9.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*v)(2, 0), 15.0);
}

TEST_F(OperatorsTest, InnerProductMaxTimes) {
    // (1 5 2) ⌈.× (2 1 3) → max of products: max(1×2, 5×1, 2×3) = max(2,5,6) = 6
    Value* result = eval(machine, "(1 5 2) ⌈.× (2 1 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(OperatorsTest, InnerProductMinPlus) {
    // (1 2 3) ⌊.+ (4 5 6) → min of sums: min(1+4, 2+5, 3+6) = min(5,7,9) = 5
    Value* result = eval(machine, "(1 2 3) ⌊.+ (4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(OperatorsTest, InnerProductAndEqual) {
    // (1 2 3) ∧.= (1 2 4) → all equal? 1∧1∧0 = 0
    Value* result = eval(machine, "(1 2 3) ∧.= (1 2 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, InnerProductAndEqualTrue) {
    // (1 2 3) ∧.= (1 2 3) → all equal? 1∧1∧1 = 1
    Value* result = eval(machine, "(1 2 3) ∧.= (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, InnerProductOrLess) {
    // (1 5 3) ∨.< (2 4 4) → any less? (1<2)∨(5<4)∨(3<4) = 1∨0∨1 = 1
    Value* result = eval(machine, "(1 5 3) ∨.< (2 4 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// ============================================================================
// ISO 13751 Section 9.3: Dyadic Operators - Rank Operator
// ============================================================================

TEST_F(OperatorsTest, RankMonadicEmptyVector) {
    // -⍤0 (⍳0) → empty vector (apply to 0-cells of empty)
    Value* result = eval(machine, "-⍤0 (⍳0)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(OperatorsTest, RankMonadicEmptyMatrix) {
    // -⍤1 (0 3⍴0) → empty matrix (apply to 1-cells of 0×3)
    Value* result = eval(machine, "-⍤1 (0 3⍴0)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 0);
}

TEST_F(OperatorsTest, RankHigherThanDimension) {
    // -⍤5 (1 2 3) → rank 5 on rank-1 array clamps to full array
    // Should apply - to the whole vector (same as -⍤1)
    Value* result = eval(machine, "-⍤5 (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*v)(2, 0), -3.0);
}

TEST_F(OperatorsTest, RankWithReverse) {
    // ⌽⍤1 on matrix → reverse each row
    // 2 3⍴⍳6 = [[1,2,3],[4,5,6]]
    // Reversed rows: [[3,2,1],[6,5,4]]
    Value* result = eval(machine, "⌽⍤1 (2 3⍴⍳6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 6.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 5.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 4.0);
}

TEST_F(OperatorsTest, RankDyadicScalarExtension) {
    // 10 +⍤0 (1 2 3) → add 10 to each element
    Value* result = eval(machine, "10 +⍤0 (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*v)(2, 0), 13.0);
}

TEST_F(OperatorsTest, RankWithIota) {
    // ⍳⍤0 (2 3) → apply iota to each scalar: gives nested result
    // Since we don't support nested arrays, this should apply ⍳ to each element
    // ⍳2 = 1 2, ⍳3 = 1 2 3 - but without nesting, behavior may vary
    // For now, test that it doesn't crash and returns something reasonable
    Value* result = eval(machine, "⍳⍤0 (2 3)");
    ASSERT_NE(result, nullptr);
    // Result structure depends on implementation
}

TEST_F(OperatorsTest, RankDyadicMatrixVector) {
    // Matrix +⍤1 vector → add vector to each row
    // (2 3⍴1 2 3 4 5 6) +⍤1 (10 20 30)
    // [[1,2,3],[4,5,6]] + [10,20,30] = [[11,22,33],[14,25,36]]
    Value* result = eval(machine, "(2 3⍴1 2 3 4 5 6) +⍤1 (10 20 30)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 22.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), 33.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 14.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 25.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 36.0);
}

TEST_F(OperatorsTest, RankNonIntegerError) {
    // -⍤1.5 (1 2 3) → DOMAIN ERROR (rank must be integer)
    EXPECT_THROW(eval(machine, "-⍤1.5 (1 2 3)"), APLError);
}

TEST_F(OperatorsTest, RankNegative) {
    // -⍤¯1 (2 3⍴⍳6) → apply to rank-(r-1) = 1-cells (rows)
    // Same as -⍤1 for a matrix
    Value* result = eval(machine, "(-⍤¯1) (2 3⍴⍳6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), -3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -4.0);
}

// ============================================================================
// Additional Outer Product Tests (ISO 9.3.1)
// ============================================================================

TEST_F(OperatorsTest, OuterProductBothScalars) {
    // 3 ∘.× 4 → scalar result 12
    Value* result = eval(machine, "3 ∘.× 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);
}

TEST_F(OperatorsTest, OuterProductMultiplicationTable) {
    // (1 2 3) ∘.× (1 2 3 4) → 3×4 multiplication table
    Value* result = eval(machine, "(1 2 3) ∘.× (1 2 3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 4);
    const Eigen::MatrixXd* m = result->as_matrix();
    // Row 0: 1×1=1, 1×2=2, 1×3=3, 1×4=4
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 3), 4.0);
    // Row 2: 3×1=3, 3×4=12
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(2, 3), 12.0);
}

// ============================================================================
// Additional Inner Product Tests (ISO 9.3.2)
// ============================================================================

TEST_F(OperatorsTest, InnerProductMatrixVector) {
    // (2 3⍴⍳6) +.× (1 2 3) → matrix × vector = vector
    // [[1,2,3],[4,5,6]] +.× [1,2,3] = [1+4+9, 4+10+18] = [14, 32]
    Value* result = eval(machine, "(2 3⍴⍳6) +.× (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    const Eigen::MatrixXd* v = result->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), 14.0);
    EXPECT_DOUBLE_EQ((*v)(1, 0), 32.0);
}

TEST_F(OperatorsTest, InnerProductMatrixMatrix) {
    // (2 3⍴⍳6) +.× (3 2⍴⍳6) → 2×2 matrix multiplication
    // [[1,2,3],[4,5,6]] +.× [[1,2],[3,4],[5,6]]
    // = [[1×1+2×3+3×5, 1×2+2×4+3×6], [4×1+5×3+6×5, 4×2+5×4+6×6]]
    // = [[1+6+15, 2+8+18], [4+15+30, 8+20+36]] = [[22,28],[49,64]]
    Value* result = eval(machine, "(2 3⍴⍳6) +.× (3 2⍴⍳6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 22.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 28.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 49.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 64.0);
}

TEST_F(OperatorsTest, InnerProductScalarVector) {
    // Per ISO 9.3.2: scalar extended to match vector
    // 2 +.× (1 2 3) → scalar extended to (2 2 2), then 2+4+6 = 12
    Value* result = eval(machine, "2 +.× (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);
}

TEST_F(OperatorsTest, InnerProductVectorScalar) {
    // Per ISO 9.3.2: scalar extended to match vector
    // (1 2 3) +.× 2 → scalar extended to (2 2 2), then 2+4+6 = 12
    Value* result = eval(machine, "(1 2 3) +.× 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);
}

TEST_F(OperatorsTest, InnerProductOneElementVector) {
    // Per ISO 9.3.2: one-element vector treated like scalar
    // (,5) +.× (1 2 3) → 5+10+15 = 30
    Value* result = eval(machine, "(,5) +.× (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 30.0);
}

// ============================================================================
// ISO 13751 Section 9: Additional Operator Edge Cases
// ============================================================================

// --- Table 5: Missing Empty Vector Reduction Identities ---

TEST_F(OperatorsTest, ReduceEmptyResidue) {
    // ISO Table 5: |/⍳0 → 0 (residue identity)
    Value* result = eval(machine, "|/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, ReduceEmptyBinomial) {
    // ISO Table 5: !/⍳0 → 1 (binomial identity)
    Value* result = eval(machine, "!/⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// --- Table 6: N-wise Reduction Empty Vector Identities ---

TEST_F(OperatorsTest, NwiseZeroResidue) {
    // ISO Table 6: 0|/⍳3 → 4 zeros (R reshape zero)
    Value* result = eval(machine, "0|/⍳3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    const Eigen::MatrixXd* m = result->as_matrix();
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ((*m)(i, 0), 0.0);
    }
}

TEST_F(OperatorsTest, NwiseZeroBinomial) {
    // ISO Table 6: 0!/⍳3 → 4 ones (R reshape one)
    Value* result = eval(machine, "0!/⍳3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    const Eigen::MatrixXd* m = result->as_matrix();
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ((*m)(i, 0), 1.0);
    }
}

TEST_F(OperatorsTest, NwiseZeroLogError) {
    // ISO Table 6: 0⍟/⍳3 → DOMAIN ERROR (no identity)
    EXPECT_THROW(eval(machine, "0⍟/⍳3"), APLError);
}

TEST_F(OperatorsTest, NwiseZeroCircleError) {
    // ISO Table 6: 0○/⍳3 → DOMAIN ERROR (no identity)
    EXPECT_THROW(eval(machine, "0○/⍳3"), APLError);
}

TEST_F(OperatorsTest, NwiseZeroNandError) {
    // ISO Table 6: 0⍲/⍳3 → DOMAIN ERROR (no identity)
    EXPECT_THROW(eval(machine, "0⍲/⍳3"), APLError);
}

TEST_F(OperatorsTest, NwiseZeroNorError) {
    // ISO Table 6: 0⍱/⍳3 → DOMAIN ERROR (no identity)
    EXPECT_THROW(eval(machine, "0⍱/⍳3"), APLError);
}

// --- Scan First Axis (⍀) Tests ---

TEST_F(OperatorsTest, ScanFirstVector) {
    // ISO 9.2.2: +⍀ on vector is same as +\ (since there's only one axis)
    Value* result = eval(machine, "+⍀1 2 3 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);   // 1
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);   // 1+2
    EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ((*m)(3, 0), 10.0);  // 1+2+3+4
}

TEST_F(OperatorsTest, ScanFirstMatrix) {
    // ISO 9.2.2: +⍀ (2 3⍴⍳6) → scan along first axis (rows)
    // Matrix: 1 2 3    Scan along axis 1 gives: 1 2 3
    //         4 5 6                              5 7 9 (1+4, 2+5, 3+6)
    Value* result = eval(machine, "+⍀2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    const Eigen::MatrixXd* m = result->as_matrix();
    // First row unchanged
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), 3.0);
    // Second row is cumulative sum
    EXPECT_DOUBLE_EQ((*m)(1, 0), 5.0);   // 1+4
    EXPECT_DOUBLE_EQ((*m)(1, 1), 7.0);   // 2+5
    EXPECT_DOUBLE_EQ((*m)(1, 2), 9.0);   // 3+6
}

TEST_F(OperatorsTest, ScanFirstScalar) {
    // ISO 9.2.2: f⍀ scalar → scalar (identity)
    Value* result = eval(machine, "+⍀5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// --- Empty Matrix Reduce/Scan (ISO 9.2.1 example) ---

TEST_F(OperatorsTest, ReduceEmptyMatrixRows) {
    // ISO 9.2.1: +/2 0⍴5 → vector 0 0 (per ISO example: "µ+/2 0µ5.1" = 2)
    // Reducing along last axis of 2×0 matrix gives 2-element vector
    Value* result = eval(machine, "+/2 0⍴5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 0.0);
}

TEST_F(OperatorsTest, ReduceEmptyMatrixCols) {
    // +⌿0 3⍴5 → reduce along first axis of 0×3 matrix gives 3-element vector
    Value* result = eval(machine, "+⌿0 3⍴5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* m = result->as_matrix();
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ((*m)(i, 0), 0.0);
    }
}

// --- Duplicate (⍨ monadic) Additional Tests ---

TEST_F(OperatorsTest, DuplicateWithMinus) {
    // -⍨5 → 5-5 = 0
    Value* result = eval(machine, "-⍨5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, DuplicateWithPower) {
    // *⍨3 → 3*3 = 27
    Value* result = eval(machine, "*⍨3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 27.0);
}

// --- Commute Additional Tests ---

TEST_F(OperatorsTest, CommuteWithPower) {
    // 2*⍨3 → 3*2 = 9
    Value* result = eval(machine, "2*⍨3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 9.0);
}

// ============================================================================
// Defined Operator Tests (User-defined operators)
// ============================================================================

TEST_F(OperatorsTest, DefinedMonadicOperatorTwice) {
    // Define TWICE operator: applies function twice
    eval(machine, "(F TWICE) ← {F F ⍵}");

    // -TWICE 5 should return 5 (negate twice = identity)
    Value* result = eval(machine, "-TWICE 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(OperatorsTest, DefinedMonadicOperatorWithCurriedFunction) {
    // To use a curried function as operand, need explicit parentheses
    // (2×)TWICE 3 = apply (2×) twice to 3 = 2×(2×3) = 2×6 = 12
    eval(machine, "(F TWICE) ← {F F ⍵}");

    Value* result = eval(machine, "(2×)TWICE 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);
}

TEST_F(OperatorsTest, DefinedMonadicOperatorSignum) {
    // ×TWICE 5 = signum(signum(5)) = signum(1) = 1
    // Note: monadic + is identity (conjugate), × is signum
    eval(machine, "(F TWICE) ← {F F ⍵}");
    Value* result = eval(machine, "×TWICE 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, DefinedDyadicOperatorCompose) {
    // Define COMPOSE operator: applies g then f
    eval(machine, "(F COMPOSE G) ← {F G ⍵}");

    // -COMPOSE÷ 4 = -(÷4) = -0.25
    Value* result = eval(machine, "-COMPOSE÷ 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -0.25);
}

TEST_F(OperatorsTest, DefinedDyadicOperatorWithParens) {
    // Same but with explicit parentheses
    eval(machine, "(F COMPOSE G) ← {F G ⍵}");

    Value* result = eval(machine, "(-COMPOSE÷) 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -0.25);
}

TEST_F(OperatorsTest, DefinedDyadicOperatorComposeWithVector) {
    // -COMPOSE⌽ on vector
    eval(machine, "(F COMPOSE G) ← {F G ⍵}");

    // Reverse then negate: -COMPOSE⌽ 1 2 3 = -(⌽1 2 3) = -(3 2 1) = ¯3 ¯2 ¯1
    Value* result = eval(machine, "-COMPOSE⌽ 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), -3.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), -1.0);
}

TEST_F(OperatorsTest, DefinedOperatorAmbivalent) {
    // Ambivalent operator: can be used monadically or dyadically
    eval(machine, "(F OP) ← {⍺←0 ⋄ ⍺ F ⍵}");

    // Monadic: 0 + ⍵
    Value* result1 = eval(machine, "+OP 5");
    ASSERT_NE(result1, nullptr);
    EXPECT_DOUBLE_EQ(result1->as_scalar(), 5.0);

    // Dyadic: ⍺ + ⍵
    Value* result2 = eval(machine, "10 +OP 5");
    ASSERT_NE(result2, nullptr);
    EXPECT_DOUBLE_EQ(result2->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, DefinedOperatorWithClosure) {
    // Define a function and use it with operator
    eval(machine, "double ← {⍵+⍵}");
    eval(machine, "(F TWICE) ← {F F ⍵}");

    // doubleTWICE 3 = double(double(3)) = double(6) = 12
    Value* result = eval(machine, "double TWICE 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);
}

TEST_F(OperatorsTest, DefinedDyadicOperatorBothClosures) {
    // Both operands are user-defined functions
    eval(machine, "square ← {⍵×⍵}");
    eval(machine, "double ← {⍵+⍵}");
    eval(machine, "(F COMPOSE G) ← {F G ⍵}");

    // square COMPOSE double 3 = square(double(3)) = square(6) = 36
    Value* result = eval(machine, "square COMPOSE double 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 36.0);
}

TEST_F(OperatorsTest, DefinedOperatorChained) {
    // Chain operators: +/TWICE means apply +/ twice
    eval(machine, "(F TWICE) ← {F F ⍵}");

    // +/TWICE 1 2 3 4 = +/(+/1 2 3 4) = +/10 = 10
    Value* result = eval(machine, "+/TWICE 1 2 3 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// ============================================================================
// Additional Defined Operator Tests - Edge Cases and ⍺⍺/⍵⍵ Syntax
// ============================================================================

TEST_F(OperatorsTest, DefinedOperatorWithAlphaAlphaSyntax) {
    // Use ⍺⍺ syntax directly instead of named operand F
    eval(machine, "(F APPLY) ← {⍺⍺ ⍵}");

    // -APPLY 5 = -5
    Value* result = eval(machine, "-APPLY 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(OperatorsTest, DefinedDyadicOperatorWithOmegaOmegaSyntax) {
    // Use ⍵⍵ syntax for second operand
    eval(machine, "(F THEN G) ← {⍵⍵ ⍺⍺ ⍵}");

    // -THEN÷ 4 = ÷(-4) = -0.25
    Value* result = eval(machine, "-THEN÷ 4");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -0.25);
}

TEST_F(OperatorsTest, DefinedOperatorWithDyadicApplication) {
    // Operator where derived function takes left argument
    eval(machine, "(F SWAP) ← {⍵ F ⍺}");

    // 3 -SWAP 10 = 10 - 3 = 7
    Value* result = eval(machine, "3 -SWAP 10");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(OperatorsTest, DefinedOperatorMultipleApplications) {
    // Apply operand multiple times
    eval(machine, "(F TRIPLE) ← {F F F ⍵}");

    // -TRIPLE 5 = ---5 = -5
    Value* result = eval(machine, "-TRIPLE 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(OperatorsTest, DefinedOperatorWithVectorOperand) {
    // Operand that works on vectors
    eval(machine, "(F EACH2) ← {F¨ ⍵}");

    // -EACH2 1 2 3 = -¨ 1 2 3 = ¯1 ¯2 ¯3
    Value* result = eval(machine, "-EACH2 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), -3.0);
}

TEST_F(OperatorsTest, DefinedOperatorReturningScalar) {
    // Operator that reduces result to scalar
    eval(machine, "(F SUM) ← {+/F ⍵}");

    // ⍳SUM 5 = +/(⍳5) = +/1 2 3 4 5 = 15
    Value* result = eval(machine, "⍳SUM 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, DefinedOperatorWithReduceOperand) {
    // Use reduce as operand
    eval(machine, "(F TWICE) ← {F F ⍵}");

    // ×/TWICE 2 3 = ×/(×/2 3) = ×/6 = 6
    Value* result = eval(machine, "×/TWICE 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

TEST_F(OperatorsTest, DefinedOperatorPreservesGPrimeCurry) {
    // Parenthesized curry (2×) should be preserved, not finalized to signum(2)
    eval(machine, "(F TWICE) ← {F F ⍵}");

    // (3+)TWICE 5 = (3+)(3+)5 = 3+(3+5) = 3+8 = 11
    Value* result = eval(machine, "(3+)TWICE 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 11.0);
}

TEST_F(OperatorsTest, ValueValueJuxtapositionWithReduceIsSyntaxError) {
    // ISO 13751: value-value juxtaposition is SYNTAX ERROR
    // 0 (+/1 2 3) 10 has adjacent values - SYNTAX ERROR
    EXPECT_THROW(eval(machine, "0 (+/1 2 3) 10"), APLError);

    // But reduce result can be used in arithmetic (not stranding)
    Value* result = eval(machine, "(+/1 2 3) + 10");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 16.0);  // 6 + 10
}

TEST_F(OperatorsTest, DefinedOperatorNestedApplication) {
    // Two different operators
    eval(machine, "(F TWICE) ← {F F ⍵}");
    eval(machine, "(F COMPOSE G) ← {F G ⍵}");

    // -TWICE COMPOSE ⍳ 3 = apply (-TWICE) to (⍳3) = --(1 2 3) = 1 2 3
    Value* result = eval(machine, "-TWICE COMPOSE ⍳ 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
}

TEST_F(OperatorsTest, DefinedOperatorWithDefaultAlpha) {
    // Ambivalent derived function with default left arg
    eval(machine, "(F WITHDEFAULT) ← {⍺←100 ⋄ ⍺ F ⍵}");

    // Monadic: uses default 100
    Value* result1 = eval(machine, "+WITHDEFAULT 5");
    ASSERT_NE(result1, nullptr);
    EXPECT_DOUBLE_EQ(result1->as_scalar(), 105.0);

    // Dyadic: uses provided left arg
    Value* result2 = eval(machine, "10 +WITHDEFAULT 5");
    ASSERT_NE(result2, nullptr);
    EXPECT_DOUBLE_EQ(result2->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, DefinedOperatorIdentity) {
    // Identity operator - just returns operand applied to arg
    eval(machine, "(F ID) ← {F ⍵}");

    Value* result = eval(machine, "⍳ID 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
}

TEST_F(OperatorsTest, DefinedDyadicOperatorPowerCompose) {
    // Compose with power: F^n
    eval(machine, "(F POW N) ← {N=0: ⍵ ⋄ F (F POW (N-1)) ⍵}");

    // -POW 3 applied to 5 = ---5 = -5
    Value* result = eval(machine, "-POW 3 ⊢ 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

// ========================================================================
// ISO 13751 §13 - ∇∇ (Del-Del) Self-Reference in Defined Operators
// ========================================================================

TEST_F(OperatorsTest, DelDelBasicBinding) {
    // ∇∇ should be bound within a defined operator's body
    // Simple test: operator just applies its operand
    eval(machine, "(F SELFOP) ← {⍺⍺ ⍵}");

    Value* result = eval(machine, "-SELFOP 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(OperatorsTest, DelDelWithVectorArg) {
    // ∇∇ with vector argument
    eval(machine, "(F APPLY) ← {⍺⍺ ⍵}");

    Value* result = eval(machine, "-APPLY 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), -2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), -3.0);
}

TEST_F(OperatorsTest, DelDelRecursiveMonadicOperator) {
    // Recursive operator using ∇∇ - apply function N times
    // (F NTIMES N) applies F N times to argument
    eval(machine, "(F NTIMES N) ← {N≤0: ⍵ ⋄ ⍺⍺ (⍺⍺ ∇∇ (N-1)) ⍵}");

    // Apply negate 3 times to 5: ---5 = -5
    Value* result = eval(machine, "-NTIMES 3 ⊢ 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);

    // Apply negate 2 times to 5: --5 = 5
    Value* result2 = eval(machine, "-NTIMES 2 ⊢ 5");
    ASSERT_NE(result2, nullptr);
    EXPECT_DOUBLE_EQ(result2->as_scalar(), 5.0);
}

TEST_F(OperatorsTest, DelDelRecursiveDyadicOperator) {
    // Dyadic operator with ∇∇ recursion
    // (F COMPOSE G) applies G to result of F
    eval(machine, "(F COMPOSE G) ← {⍵⍵ ⍺⍺ ⍵}");

    // Compose negate and double: first negate, then double
    // Using dfn {2×⍵} instead of train (2×⊢)
    Value* result = eval(machine, "- COMPOSE {2×⍵} ⊢ 3");
    ASSERT_NE(result, nullptr);
    // -3 then ×2 = -6
    EXPECT_DOUBLE_EQ(result->as_scalar(), -6.0);
}

TEST_F(OperatorsTest, DelDelWithDfnOperand) {
    // ∇∇ with a dfn as operand
    eval(machine, "(F WRAP) ← {⍺⍺ ⍵}");

    Value* result = eval(machine, "{⍵+10}WRAP 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, DelDelWithNamedFunctionOperand) {
    // ∇∇ with a named function as operand
    eval(machine, "DOUBLE ← {2×⍵}");
    eval(machine, "(F APPLY) ← {⍺⍺ ⍵}");

    Value* result = eval(machine, "DOUBLE APPLY 7");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

TEST_F(OperatorsTest, DelDelDyadicApplication) {
    // ∇∇ in operator with dyadic function application
    eval(machine, "(F DAPPLY) ← {⍺ ⍺⍺ ⍵}");

    Value* result = eval(machine, "10 +DAPPLY 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, DelDelAmbivalentOperator) {
    // ∇∇ in operator that can work monadically and dyadically
    // ISO 13751 §13.1.2.1: Operators can be ambivalent
    // Note: ⎕NC is not implemented, so we test via separate monadic/dyadic operators

    // Monadic operator - always applies operand monadically
    eval(machine, "(F MWRAP) ← {⍺⍺ ⍵}");
    Value* result = eval(machine, "-MWRAP 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);

    // Dyadic operator application - applies operand dyadically
    eval(machine, "(F DWRAP) ← {⍺ ⍺⍺ ⍵}");
    Value* result2 = eval(machine, "10 +DWRAP 5");
    ASSERT_NE(result2, nullptr);
    EXPECT_DOUBLE_EQ(result2->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, DelDelNestedOperator) {
    // Nested operator calls with ∇∇
    eval(machine, "(F OUTER) ← {⍺⍺ ⍵}");
    eval(machine, "(F INNER) ← {⍺⍺ ⍵}");

    // Compose operators: -OUTER applied through INNER
    Value* result = eval(machine, "-OUTER INNER 5");
    ASSERT_NE(result, nullptr);
    // This tests derived function from OUTER being passed to INNER
}

TEST_F(OperatorsTest, DelDelWithReduceOperand) {
    // ∇∇ with reduce as operand
    eval(machine, "(F WRAP) ← {⍺⍺ ⍵}");

    Value* result = eval(machine, "+/WRAP 1 2 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, DelDelCountingRecursion) {
    // Simple counting recursion to verify ∇∇ works
    // Operator that counts down and sums
    eval(machine, "(F COUNTDOWN) ← {⍵≤0: 0 ⋄ ⍵ + (⍺⍺ ∇∇) (⍵-1)}");

    // Sum of 1+2+3+4+5 = 15
    Value* result = eval(machine, "⊢COUNTDOWN 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, DelDelFactorial) {
    // Classic factorial using ∇∇
    eval(machine, "(F FACT) ← {⍵≤1: 1 ⋄ ⍵ × (⍺⍺ ∇∇) (⍵-1)}");

    Value* result = eval(machine, "×FACT 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 120.0);  // 5! = 120
}

TEST_F(OperatorsTest, DelDelFibonacci) {
    // Fibonacci using ∇∇
    eval(machine, "(F FIB) ← {⍵≤1: ⍵ ⋄ ((⍺⍺ ∇∇) (⍵-1)) + (⍺⍺ ∇∇) (⍵-2)}");

    Value* result = eval(machine, "+FIB 10");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 55.0);  // F(10) = 55
}

TEST_F(OperatorsTest, DelDelDyadicRecursive) {
    // Dyadic operator with ∇∇ - power function compose
    // Applies F N times where N is the right operand
    eval(machine, "(F POW N) ← {N=0: ⍵ ⋄ ⍺⍺ (⍺⍺ ∇∇ (N-1)) ⍵}");

    // Apply +1 three times: 5+1+1+1 = 8
    // Using dfn {1+⍵} instead of train (1+⊢)
    Value* result = eval(machine, "{1+⍵} POW 3 ⊢ 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

// ========================================================================
// ISO 13751 Error Type Tests
// ========================================================================

// AXIS ERROR: Invalid axis specification (ISO 13751 §5.3.4.1)
TEST_F(OperatorsTest, AxisErrorReduceInvalidAxis) {
    // Reduce with axis out of range for vector (vectors only have axis 1)
    EXPECT_THROW(eval(machine, "+/[2] 1 2 3"), APLError);
}

TEST_F(OperatorsTest, AxisErrorReduceAxisTooHigh) {
    // Axis 3 is out of range for a 2D matrix
    EXPECT_THROW(eval(machine, "+/[3] 2 3⍴⍳6"), APLError);
}

TEST_F(OperatorsTest, AxisErrorCatenateInvalidAxis) {
    // Catenate vectors with axis 2 (vectors only have axis 1)
    EXPECT_THROW(eval(machine, "1 2 3 ,[2] 4 5 6"), APLError);
}

TEST_F(OperatorsTest, AxisErrorMonadicCatenateInvalidAxis) {
    // Monadic catenate (ravel) with invalid axis
    EXPECT_THROW(eval(machine, ",[3] 1 2 3"), APLError);
}

TEST_F(OperatorsTest, AxisErrorTransposeInvalidPerm) {
    // Invalid axis permutation for transpose
    EXPECT_THROW(eval(machine, "0 1⍉1 2 3"), APLError);  // Vector only has axis 0
}

TEST_F(OperatorsTest, AxisErrorNonScalarAxis) {
    // Axis must be scalar
    EXPECT_THROW(eval(machine, "+/[1 2] 2 3⍴⍳6"), APLError);
}

// LIMIT ERROR: Implementation limits exceeded (ISO 13751 §A.3)
TEST_F(OperatorsTest, LimitErrorIotaTooLarge) {
    // Iota with value exceeding INT_MAX
    EXPECT_THROW(eval(machine, "⍳3000000000"), APLError);
}

TEST_F(OperatorsTest, LimitErrorReshapeTooLarge) {
    // Reshape to size exceeding INT_MAX
    EXPECT_THROW(eval(machine, "3000000000⍴1"), APLError);
}

// Valid-axis: one-element-vector as axis (ISO 13751 §5.3.2)
// "K is a valid-axis of A... if K is a scalar or one-element-vector"
TEST_F(OperatorsTest, ValidAxisOneElementVector) {
    // Reduce with one-element-vector axis should work same as scalar
    Value* result = eval(machine, "+/[(⍳1)] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // Column sums: 1+4=5, 2+5=7, 3+6=9
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 7.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 9.0);
}

TEST_F(OperatorsTest, ValidAxisOneElementVectorLastAxis) {
    // One-element-vector [2] should work as axis for reduce
    Value* result = eval(machine, "+/[1⍴2] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    // Row sums: 1+2+3=6, 4+5+6=15
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 15.0);
}

// Near-integer axis (ISO 13751 §5.2.5, §5.3.2)
// Axis must be "near-integer" - tolerantly close to an integer within INTEGER_TOLERANCE
TEST_F(OperatorsTest, NearIntegerAxisAccepted) {
    // 1.0 is exactly an integer - should work
    Value* result = eval(machine, "+/[1.0] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
}

TEST_F(OperatorsTest, NearIntegerAxisTolerance) {
    // Values within INTEGER_TOLERANCE (1e-10) of an integer should be accepted
    // 1+1e-11 is within tolerance of 1
    Value* result = eval(machine, "+/[1.00000000001] 2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    // Should reduce along axis 1 (first axis), giving 3 results
    EXPECT_EQ(result->size(), 3);
}

TEST_F(OperatorsTest, NonIntegerAxisRejected) {
    // 1.5 is NOT a near-integer - should fail
    EXPECT_THROW(eval(machine, "+/[1.5] 2 3⍴⍳6"), APLError);
}

// Laminate vs Catenate distinguished by fractional axis (ISO 13751)
TEST_F(OperatorsTest, LaminateWithFractionalAxis) {
    // ,[0.5] is laminate (fractional axis creates new dimension)
    Value* result = eval(machine, "1 2 3 ,[0.5] 4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    // Laminate creates 2×3 matrix
    EXPECT_EQ(result->as_matrix()->rows(), 2);
    EXPECT_EQ(result->as_matrix()->cols(), 3);
}

// ============================================================================
// ISO 13751 Section 9: Additional Edge Cases
// ============================================================================

// --- 9.2.1 Reduction: Character vectors ---

TEST_F(OperatorsTest, ReduceEqualSingleChar) {
    // ISO 9.2.1: =/'A' → 'A' (single element returns that element)
    Value* result = eval(machine, "=/'A'");
    ASSERT_NE(result, nullptr);
    // Single element reduce returns scalar
    EXPECT_TRUE(result->is_scalar() || result->size() == 1);
}

TEST_F(OperatorsTest, ReduceEqualIdenticalChars) {
    // ISO 9.2.1: =/'AA' → 1 (all identical)
    Value* result = eval(machine, "=/'AA'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, ReduceEqualDifferentChars) {
    // ISO 9.2.1: =/'AB' → 0 (not all identical)
    Value* result = eval(machine, "=/'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(OperatorsTest, ReduceEqualTripleChars) {
    // ISO 9.2.1: =/'AAA' → 0 (right-to-left: 'A'=('A'='A') = 'A'=1 = 65=1 = 0)
    Value* result = eval(machine, "=/'AAA'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// --- 9.2.2 Scan: Additional edge cases ---

TEST_F(OperatorsTest, ScanNonCommutative) {
    // ISO 9.2.2: -\1 1 1 → 1 0 1 (prefix reductions, right-to-left)
    // -/1=1, -/1 1=0, -/1 1 1=1-(1-1)=1
    Value* result = eval(machine, "-\\1 1 1");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 1.0);
}

TEST_F(OperatorsTest, ScanDivide) {
    // ÷\2 4 8 → 2, 0.5, 4 (prefix reductions, right-to-left)
    // ÷/2=2, ÷/2 4=2÷4=0.5, ÷/2 4 8=2÷(4÷8)=2÷0.5=4
    Value* result = eval(machine, "÷\\2 4 8");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 0.5);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 4.0);
}

TEST_F(OperatorsTest, ScanAndLogical) {
    // ISO 9.2.2: ∧\1 1 1 0 0 0 1 1 1 → 1 1 1 0 0 0 0 0 0
    Value* result = eval(machine, "∧\\1 1 1 0 0 0 1 1 1");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 9);
    // First three: 1 1 1, then all zeros after first 0
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(3, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(8, 0), 0.0);
}

// --- 9.2.3 N-wise: Spec examples ---

TEST_F(OperatorsTest, NwiseSubtractionNegative) {
    // ISO 9.2.3: ¯2-/1 4 9 16 25 → 3 5 7 9 (reversed pairs)
    Value* result = eval(machine, "¯2-/1 4 9 16 25");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 7.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(3, 0), 9.0);
}

TEST_F(OperatorsTest, NwiseSubtractionPositive) {
    // ISO 9.2.3: 2-/1 4 9 16 25 → ¯3 ¯5 ¯7 ¯9
    Value* result = eval(machine, "2-/1 4 9 16 25");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), -3.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), -5.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), -7.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(3, 0), -9.0);
}

// --- 9.2.4/9.2.5 Duplicate/Commute: Additional cases ---

TEST_F(OperatorsTest, CommutePower) {
    // ISO 9.2.5: 2*⍨3 → 3*2 = 9
    Value* result = eval(machine, "2*⍨3");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 9.0);
}

TEST_F(OperatorsTest, DuplicateMinus) {
    // ISO 9.2.4: -⍨5 → 5-5 = 0
    Value* result = eval(machine, "-⍨5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// --- 9.3.1 Outer Product: Spec examples ---

TEST_F(OperatorsTest, OuterProductAdditionTable) {
    // ISO 9.3.1: 10 20 30 ∘.+ 1 2 3 → addition table
    Value* result = eval(machine, "10 20 30 ∘.+ 1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 3);
    // Check corners
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 11.0);  // 10+1
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 13.0);  // 10+3
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 31.0);  // 30+1
    EXPECT_DOUBLE_EQ((*mat)(2, 2), 33.0);  // 30+3
}

// --- 9.3.2 Inner Product: Spec examples ---

TEST_F(OperatorsTest, InnerProductSpecExample) {
    // ISO 9.3.2: 4 2 1 +.× 1 0 1 → 5 (4×1 + 2×0 + 1×1)
    Value* result = eval(machine, "4 2 1 +.× 1 0 1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(OperatorsTest, InnerProductDimensionError) {
    // ISO 9.3.2: Inner dims must match
    EXPECT_THROW(eval(machine, "(2 3⍴⍳6) +.× (2 3⍴⍳6)"), APLError);
}

// --- 9.3.3-5 Rank: Additional edge cases ---

TEST_F(OperatorsTest, RankZeroCells) {
    // f⍤0 applies f to each scalar (0-cells)
    // -⍤0 (1 2 3) → ¯1 ¯2 ¯3
    Value* result = eval(machine, "-⍤0 (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), -2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), -3.0);
}

TEST_F(OperatorsTest, RankTwoElementVector) {
    // Two-element rank vector: left and right cell ranks
    // 10 +⍤0 0 (1 2 3) → apply + to 0-cells of both (scalar extension)
    Value* result = eval(machine, "10 +⍤0 0 (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 12.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 13.0);
}

TEST_F(OperatorsTest, RankThreeElementVector) {
    // Three-element rank vector: monadic, left, right
    // For dyadic use, elements 2 and 3 (second and third) are used
    Value* result = eval(machine, "10 +⍤99 0 0 (1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    // 0 0 means scalar cells for both, so 10+1, 10+2, 10+3
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 11.0);
}

// ========================================================================
// Operator Argument Validation Tests
// ========================================================================

TEST_F(OperatorsTest, ReduceRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("+/+"), APLError);
}

TEST_F(OperatorsTest, ReduceFirstRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("+⌿+"), APLError);
}

TEST_F(OperatorsTest, ScanRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("+\\+"), APLError);
}

TEST_F(OperatorsTest, ScanFirstRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("+⍀+"), APLError);
}

TEST_F(OperatorsTest, NwiseReduceRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("2+/+"), APLError);
}

TEST_F(OperatorsTest, NwiseReduceFirstRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("2+⌿+"), APLError);
}

TEST_F(OperatorsTest, RankRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("+⍤1 +"), APLError);
}

// ============================================================================
// Operators with Strands
// ============================================================================

TEST_F(OperatorsTest, EachWithStrand) {
    // ⍴¨ applied to strand - result must have same shape as input (ISO 9.2.6)
    // Note: ⊂ on scalar returns scalar, so use ⊂1 2 3 to create a strand
    Value* result = machine->eval("⍴¨⊂1 2 3");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 1);
    // Shape of 1 2 3 is the 1-element vector (3)
    Value* inner = (*result->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_EQ(inner->size(), 1);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), 3.0);
}

TEST_F(OperatorsTest, EachShapeOfStrand) {
    // ⍴¨ applied to strand of vectors - 1-element vectors returned as strand (ISO 8.2.2: shape returns vector)
    Value* result = machine->eval("⍴¨(⊂1 2 3),⊂4 5");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2);
    // First element shape is 1-element vector (3)
    Value* shape1 = (*result->as_strand())[0];
    ASSERT_TRUE(shape1->is_vector());
    EXPECT_DOUBLE_EQ((*shape1->as_matrix())(0, 0), 3.0);
    // Second element shape is 1-element vector (2)
    Value* shape2 = (*result->as_strand())[1];
    ASSERT_TRUE(shape2->is_vector());
    EXPECT_DOUBLE_EQ((*shape2->as_matrix())(0, 0), 2.0);
}

TEST_F(OperatorsTest, ReduceStrandSingleElement) {
    // +/⊂1 2 3 → returns the enclosed vector
    Value* result = machine->eval("+/⊂1 2 3");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(2, 0), 3.0);
}

TEST_F(OperatorsTest, ReduceStrandTwoElements) {
    // +/(⊂1 2 3),⊂4 5 6 → element-wise sum: 5 7 9
    Value* result = machine->eval("+/(⊂1 2 3),⊂4 5 6");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(2, 0), 9.0);
}

TEST_F(OperatorsTest, ReduceStrandCatenate) {
    // ,/(⊂1 2 3),⊂4 5 → concatenate: 1 2 3 4 5
    Value* result = machine->eval(",/(⊂1 2 3),⊂4 5");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(4, 0), 5.0);
}

TEST_F(OperatorsTest, ScanStrandSingleElement) {
    // +\⊂1 2 3 → returns the strand unchanged
    Value* result = machine->eval("+\\⊂1 2 3");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 1);
}

TEST_F(OperatorsTest, ScanStrandTwoElements) {
    // +\(⊂1 2 3),⊂4 5 6 → ((1 2 3), (5 7 9))
    Value* result = machine->eval("+\\(⊂1 2 3),⊂4 5 6");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2);

    // First element: 1 2 3
    Value* first = (*result->as_strand())[0];
    ASSERT_TRUE(first->is_vector());
    EXPECT_DOUBLE_EQ((*first->as_matrix())(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*first->as_matrix())(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*first->as_matrix())(2, 0), 3.0);

    // Second element: 5 7 9 (running sum)
    Value* second = (*result->as_strand())[1];
    ASSERT_TRUE(second->is_vector());
    EXPECT_DOUBLE_EQ((*second->as_matrix())(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*second->as_matrix())(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*second->as_matrix())(2, 0), 9.0);
}

TEST_F(OperatorsTest, ScanStrandThreeElements) {
    // +\(⊂1 2 3),(⊂4 5 6),⊂7 8 9 → ((1 2 3), (5 7 9), (12 15 18))
    Value* result = machine->eval("+\\(⊂1 2 3),(⊂4 5 6),⊂7 8 9");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 3);

    // Third element: 12 15 18
    Value* third = (*result->as_strand())[2];
    ASSERT_TRUE(third->is_vector());
    EXPECT_DOUBLE_EQ((*third->as_matrix())(0, 0), 12.0);
    EXPECT_DOUBLE_EQ((*third->as_matrix())(1, 0), 15.0);
    EXPECT_DOUBLE_EQ((*third->as_matrix())(2, 0), 18.0);
}

// ============================================================================
// Dyadic Each with Mixed Types (ISO 9.2.6)
// ============================================================================

TEST_F(OperatorsTest, DyadicEachVectorPlusStrand) {
    // 1 2+¨(⊂10 20),(⊂30 40) → ((11 21), (32 42))
    // Per ISO 9.2.6: match shapes element-wise
    Value* result = machine->eval("1 2+¨(⊂10 20),(⊂30 40)");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2);

    // First element: 1+(10 20) = 11 21
    Value* first = (*result->as_strand())[0];
    ASSERT_TRUE(first->is_vector());
    EXPECT_DOUBLE_EQ((*first->as_matrix())(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*first->as_matrix())(1, 0), 21.0);

    // Second element: 2+(30 40) = 32 42
    Value* second = (*result->as_strand())[1];
    ASSERT_TRUE(second->is_vector());
    EXPECT_DOUBLE_EQ((*second->as_matrix())(0, 0), 32.0);
    EXPECT_DOUBLE_EQ((*second->as_matrix())(1, 0), 42.0);
}

TEST_F(OperatorsTest, DyadicEachStrandPlusVector) {
    // ((⊂10 20),(⊂30 40))+¨1 2 → ((11 21), (32 42))
    // Note: need explicit parens because APL parses right-to-left
    Value* result = machine->eval("((⊂10 20),(⊂30 40))+¨1 2");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2);

    Value* first = (*result->as_strand())[0];
    ASSERT_TRUE(first->is_vector());
    EXPECT_DOUBLE_EQ((*first->as_matrix())(0, 0), 11.0);
}

TEST_F(OperatorsTest, DyadicEachVectorTimesStrand) {
    // 2 3×¨(⊂10 20),(⊂30 40) → ((20 40), (90 120))
    Value* result = machine->eval("2 3×¨(⊂10 20),(⊂30 40)");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2);

    Value* first = (*result->as_strand())[0];
    ASSERT_TRUE(first->is_vector());
    EXPECT_DOUBLE_EQ((*first->as_matrix())(0, 0), 20.0);
    EXPECT_DOUBLE_EQ((*first->as_matrix())(1, 0), 40.0);
}

// ============================================================================
// Outer Product with Strands (ISO 9.3.1)
// ============================================================================

TEST_F(OperatorsTest, OuterProductVectorVector) {
    // Basic outer product: 1 2∘.+10 20 → 2×2 matrix
    Value* result = machine->eval("1 2∘.+10 20");
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 1), 21.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 1), 22.0);
}

TEST_F(OperatorsTest, OuterProductStrandStrand) {
    // Outer product with strands: (⊂1 2),(⊂3 4)∘.+⊂10 20
    // Result shape is 2×1 (strand with strand)
    // Each element is strand[i] + strand[j]
    Value* result = machine->eval("((⊂1 2),(⊂3 4))∘.+(⊂10 20)");
    // Result should be a 2×1 arrangement where each cell is a vector
    // [0,0]: (1 2)+(10 20) = (11 22)
    // [1,0]: (3 4)+(10 20) = (13 24)
    ASSERT_TRUE(result->is_strand() || result->is_matrix());
}

TEST_F(OperatorsTest, OuterProductVectorStrand) {
    // 1 2∘.+(⊂10 20),(⊂30 40) → 2×2 result
    // [0,0]: 1+(10 20) = (11 21)
    // [0,1]: 1+(30 40) = (31 41)
    // [1,0]: 2+(10 20) = (12 22)
    // [1,1]: 2+(30 40) = (32 42)
    Value* result = machine->eval("1 2∘.+((⊂10 20),(⊂30 40))");
    // Result is 2×2 with nested vectors
    ASSERT_TRUE(result->is_strand() || result->is_matrix());
    // Check depth - should be nested
    Value* depth = machine->eval("≡1 2∘.+((⊂10 20),(⊂30 40))");
    EXPECT_GE(depth->as_scalar(), 2.0);
}

TEST_F(OperatorsTest, OuterProductStrandVector) {
    // (⊂1 2),(⊂3 4)∘.+10 20 → 2×2 result
    // [0,0]: (1 2)+10 = (11 12)
    // [0,1]: (1 2)+20 = (21 22)
    // [1,0]: (3 4)+10 = (13 14)
    // [1,1]: (3 4)+20 = (23 24)
    Value* result = machine->eval("((⊂1 2),(⊂3 4))∘.+10 20");
    ASSERT_TRUE(result->is_strand() || result->is_matrix());
}

// ============================================================================
// Inner Product with Strands
// ============================================================================

TEST_F(OperatorsTest, InnerProductVectorVectorBaseline) {
    // Baseline: 1 2 3 +.× 4 5 6 = +/ 1 2 3 × 4 5 6 = +/ 4 10 18 = 32
    Value* result = machine->eval("1 2 3+.×4 5 6");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 32.0);
}

TEST_F(OperatorsTest, InnerProductStrandStrand) {
    // Inner product with strands: each element paired, g applied, then f reduced
    // ((⊂1 2),(⊂3 4)) +.× ((⊂10),(⊂20))
    // = +/ ((1 2)×(10)) ((3 4)×(20))
    // = +/ (10 20) (60 80)
    // Result depends on how + works with strands
    Value* result = machine->eval("((⊂1 2),(⊂3 4))+.×((⊂10),(⊂20))");
    // Should produce some result without crashing
    ASSERT_NE(result, nullptr);
}

TEST_F(OperatorsTest, InnerProductVectorStrand) {
    // 1 2 +.× ((⊂10 20),(⊂30 40))
    // = +/ (1×(10 20)) (2×(30 40))
    // = +/ (10 20) (60 80)
    // = (70 100) if + works element-wise on strands
    Value* result = machine->eval("1 2+.×((⊂10 20),(⊂30 40))");
    ASSERT_NE(result, nullptr);
}

TEST_F(OperatorsTest, InnerProductStrandVector) {
    // ((⊂1 2),(⊂3 4)) +.× 10 20
    // = +/ ((1 2)×10) ((3 4)×20)
    // = +/ (10 20) (60 80)
    Value* result = machine->eval("((⊂1 2),(⊂3 4))+.×10 20");
    ASSERT_NE(result, nullptr);
}

// ============================================================================
// Commute/Duplicate with Strands (ISO 9.2.4, 9.2.5)
// ============================================================================

TEST_F(OperatorsTest, DuplicateStrand) {
    // Duplicate: f⍨ B = B f B
    // ,⍨ strand = strand , strand (catenate strand with itself)
    Value* result = machine->eval(",⍨(⊂1 2),(⊂3 4)");
    ASSERT_NE(result, nullptr);
    // Should produce 4-element strand: (⊂1 2),(⊂3 4),(⊂1 2),(⊂3 4)
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 4);
}

TEST_F(OperatorsTest, CommuteStrandScalar) {
    // Commute: A f⍨ B = B f A
    // 10 -⍨ (⊂1 2),(⊂3 4) = ((⊂1 2),(⊂3 4)) - 10
    // Each element of strand minus 10
    Value* result = machine->eval("10-⍨((⊂1 2),(⊂3 4))");
    ASSERT_NE(result, nullptr);
    // Result should be strand with (1 2)-10 and (3 4)-10
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 2);
    // First element: (1 2) - 10 = (-9 -8)
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(0, 0), -9.0);
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(1, 0), -8.0);
}

TEST_F(OperatorsTest, CommuteScalarStrand) {
    // Commute: A f⍨ B = B f A
    // (⊂1 2),(⊂3 4) -⍨ 10 = 10 - ((⊂1 2),(⊂3 4))
    Value* result = machine->eval("((⊂1 2),(⊂3 4))-⍨10");
    ASSERT_NE(result, nullptr);
    // Result: 10 - (⊂1 2) and 10 - (⊂3 4)
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 2);
    // First element: 10 - (1 2) = (9 8)
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(0, 0), 9.0);
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(1, 0), 8.0);
}

TEST_F(OperatorsTest, DirectStrandStrandSubtract) {
    // Direct strand-strand subtraction without commute
    Value* result = machine->eval("(10,20)-(1,2)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0,0), 9.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1,0), 18.0);
}

TEST_F(OperatorsTest, CommuteSimpleStrandStrand) {
    // Commute with simple strands (no enclose): (1,2) -⍨ (10,20) = (10,20) - (1,2)
    Value* result = machine->eval("(1,2)-⍨(10,20)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0,0), 9.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1,0), 18.0);
}

TEST_F(OperatorsTest, CommuteStrandStrand) {
    // Commute with nested strands (enclosed vectors): A f⍨ B = B f A
    // ((⊂1 2),(⊂3 4)) -⍨ ((⊂10 20),(⊂30 40)) = ((⊂10 20),(⊂30 40)) - ((⊂1 2),(⊂3 4))
    // Element 0: (10 20) - (1 2) = (9 18)
    // Element 1: (30 40) - (3 4) = (27 36)
    Value* result = machine->eval("((⊂1 2),(⊂3 4))-⍨((⊂10 20),(⊂30 40))");
    ASSERT_NE(result, nullptr);
    // Result is a strand of vectors
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 2);
    // First element: (9 18)
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(0,0), 9.0);
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(1,0), 18.0);
    // Second element: (27 36)
    ASSERT_TRUE((*strand)[1]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[1]->as_matrix())(0,0), 27.0);
    EXPECT_DOUBLE_EQ((*(*strand)[1]->as_matrix())(1,0), 36.0);
}

// ============================================================================
// Rank Operator with Strands (ISO 9.3.3-9.3.5)
// Strand has rank 1: 0-cells are elements, 1-cells is whole strand
// ============================================================================

TEST_F(OperatorsTest, RankMonadicStrandRank0) {
    // -⍤0 on strand: apply - to each 0-cell (each element)
    // Per ISO 9.3.4: apply f to rank-0 cells of B
    Value* result = machine->eval("-⍤0 ((⊂1 2),(⊂3 4))");
    ASSERT_NE(result, nullptr);
    // Result should be strand with negated vectors
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 2);
    // First: -(1 2) = (-1 -2)
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(0,0), -1.0);
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(1,0), -2.0);
}

TEST_F(OperatorsTest, RankMonadicStrandRank1) {
    // ⌽⍤1 on strand: apply ⌽ to 1-cells (the whole strand, since strand is rank 1)
    // Per ISO 9.3.4: if y exceeds rank, use rank of B (which is 1)
    Value* result = machine->eval("⌽⍤1 ((⊂1 2),(⊂3 4))");
    ASSERT_NE(result, nullptr);
    // Reverse the strand: ((⊂3 4),(⊂1 2))
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 2);
    // First element should now be (3 4)
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(0,0), 3.0);
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(1,0), 4.0);
}

TEST_F(OperatorsTest, RankDyadicStrandStrandRank0) {
    // Per ISO 9.3.5: apply f between rank-0 cells of A and rank-0 cells of B
    // ((⊂1 2),(⊂3 4)) +⍤0 ((⊂10 20),(⊂30 40))
    // Pairs: (1 2)+(10 20), (3 4)+(30 40)
    Value* result = machine->eval("((⊂1 2),(⊂3 4))+⍤0 ((⊂10 20),(⊂30 40))");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 2);
    // First: (1 2)+(10 20) = (11 22)
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(0,0), 11.0);
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(1,0), 22.0);
    // Second: (3 4)+(30 40) = (33 44)
    ASSERT_TRUE((*strand)[1]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[1]->as_matrix())(0,0), 33.0);
    EXPECT_DOUBLE_EQ((*(*strand)[1]->as_matrix())(1,0), 44.0);
}

// N-wise Reduction with Strands (ISO 9.2.3)
// N-wise reduction applies f to N-length windows of the argument
// For strands, the "windows" are consecutive groups of strand elements

TEST_F(OperatorsTest, NwiseReduceStrandPairwise) {
    // 2+/ on strand: add consecutive pairs
    // 2+/((⊂1 2),(⊂3 4),(⊂5 6)) should give strand of 2 elements:
    // (1 2)+(3 4) = (4 6) and (3 4)+(5 6) = (8 10)
    Value* result = machine->eval("2+/((⊂1 2),(⊂3 4),(⊂5 6))");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 2);
    // First: (1 2)+(3 4) = (4 6)
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(0,0), 4.0);
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(1,0), 6.0);
    // Second: (3 4)+(5 6) = (8 10)
    ASSERT_TRUE((*strand)[1]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[1]->as_matrix())(0,0), 8.0);
    EXPECT_DOUBLE_EQ((*(*strand)[1]->as_matrix())(1,0), 10.0);
}

TEST_F(OperatorsTest, NwiseReduceStrandFullReduction) {
    // 3+/ on 3-element strand: reduces all to single element
    // 3+/((⊂1 2),(⊂3 4),(⊂5 6)) = (1 2)+(3 4)+(5 6) = (9 12)
    Value* result = machine->eval("3+/((⊂1 2),(⊂3 4),(⊂5 6))");
    ASSERT_NE(result, nullptr);
    // Result should be a single vector (not strand) since it's fully reduced
    ASSERT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0,0), 9.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1,0), 12.0);
}

TEST_F(OperatorsTest, NwiseReduceStrandIdentity) {
    // 1+/ should return unchanged (each element is its own window)
    Value* result = machine->eval("1+/((⊂1 2),(⊂3 4))");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 2);
}

TEST_F(OperatorsTest, NwiseReduceStrandZero) {
    // 0+/ should return identity-extended result (N+1 elements of identity)
    // For +, identity is 0
    Value* result = machine->eval("0+/((⊂1 2),(⊂3 4))");
    ASSERT_NE(result, nullptr);
    // Per ISO 9.2.3: 0+/ on 2-element vector gives 3-element result of zeros
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

TEST_F(OperatorsTest, NwiseReduceStrandNegative) {
    // Negative N reverses subarrays before applying f
    // Per ISO 9.2.3: negative N reverses each window before reduction
    // ¯2-/((⊂1 2),(⊂3 4),(⊂5 6)) with 3 elements, window size 2:
    // Window 0 reversed: (3 4)-(1 2) = (2 2)
    // Window 1 reversed: (5 6)-(3 4) = (2 2)
    Value* result = machine->eval("¯2-/((⊂1 2),(⊂3 4),(⊂5 6))");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 2);
    // First: (3 4)-(1 2) = (2 2)
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(0,0), 2.0);
    EXPECT_DOUBLE_EQ((*(*strand)[0]->as_matrix())(1,0), 2.0);
    // Second: (5 6)-(3 4) = (2 2)
    ASSERT_TRUE((*strand)[1]->is_vector());
    EXPECT_DOUBLE_EQ((*(*strand)[1]->as_matrix())(0,0), 2.0);
    EXPECT_DOUBLE_EQ((*(*strand)[1]->as_matrix())(1,0), 2.0);
}

// ========================================================================
// NDARRAY Reduce Tests (ISO 13751 §9.2.1)
// ========================================================================

// ISO 9.2.1: f/B on rank>2 array reduces along last axis
// Result shape: shape of B with last dimension removed
TEST_F(OperatorsTest, NDArrayReduceLastAxis) {
    // +/2 3 4⍴⍳24 - reduce 2×3×4 along last axis → 2×3 matrix
    Value* result = machine->eval("+/2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    const auto* mat = result->as_matrix();
    // Row 0: sums of 1-4, 5-8, 9-12
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 10.0);   // 1+2+3+4
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 26.0);   // 5+6+7+8
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 42.0);   // 9+10+11+12
    // Row 1: sums of 13-16, 17-20, 21-24
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 58.0);   // 13+14+15+16
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 74.0);   // 17+18+19+20
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 90.0);   // 21+22+23+24
}

// ISO 9.2.1: f/[K]B reduces along axis K
TEST_F(OperatorsTest, NDArrayReduceAxisFirst) {
    // +/[1]2 3 4⍴⍳24 - reduce along first axis → 3×4 matrix
    Value* result = machine->eval("+/[1]2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 4);
    const auto* mat = result->as_matrix();
    // Position [0,0]: 1+13=14, [0,1]: 2+14=16, etc.
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 14.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 16.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 18.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 3), 20.0);
}

// ISO 9.2.1: f/[K]B reduces along middle axis
TEST_F(OperatorsTest, NDArrayReduceAxisMiddle) {
    // +/[2]2 3 4⍴⍳24 - reduce along axis 2 → 2×4 matrix
    Value* result = machine->eval("+/[2]2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 4);
    const auto* mat = result->as_matrix();
    // Position [0,0]: 1+5+9=15, [0,1]: 2+6+10=18, etc.
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 15.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 18.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 21.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 3), 24.0);
    // Position [1,0]: 13+17+21=51
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 51.0);
}

// ISO 9.2.1: f⌿B reduces along first axis (reduce-first)
TEST_F(OperatorsTest, NDArrayReduceFirst) {
    // +⌿2 3 4⍴⍳24 - same as +/[1]
    Value* result = machine->eval("+⌿2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 4);
    const auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 14.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 16.0);
}

// ISO 9.2.1: Reduce on 4D array produces 3D result
TEST_F(OperatorsTest, NDArrayReduce4D) {
    // +/2 3 4 5⍴⍳120 - reduce along last axis → 2×3×4 NDARRAY
    Value* result = machine->eval("+/2 3 4 5⍴⍳120");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    // First element: 1+2+3+4+5 = 15
    EXPECT_DOUBLE_EQ((*nd->data)(0), 15.0);
}

// ISO 9.2.1: Reduce with ⎕IO=0
TEST_F(OperatorsTest, NDArrayReduceIO0) {
    // With ⎕IO=0, axes are 0-indexed
    Value* result = machine->eval("⎕IO←0 ⋄ +/2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const auto* mat = result->as_matrix();
    // With ⎕IO=0, ⍳24 is 0..23, sum of 0+1+2+3=6
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 6.0);
}

// ISO 9.2.1: Reduce with explicit axis and ⎕IO=0
TEST_F(OperatorsTest, NDArrayReduceAxisIO0) {
    // +/[0] with ⎕IO=0 reduces along first axis
    Value* result = machine->eval("⎕IO←0 ⋄ +/[0]2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 4);
    const auto* mat = result->as_matrix();
    // Position [0,0]: 0+12=12
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 12.0);
}

// ISO 9.2.1: Times reduce (product)
TEST_F(OperatorsTest, NDArrayReduceTimes) {
    // ×/2 2 3⍴⍳12
    Value* result = machine->eval("×/2 2 3⍴⍳12");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const auto* mat = result->as_matrix();
    // Position [0,0]: 1×2×3=6
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 6.0);
    // Position [0,1]: 4×5×6=120
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 120.0);
}

// ISO 9.2.1: Max reduce
TEST_F(OperatorsTest, NDArrayReduceMax) {
    Value* result = machine->eval("⌈/2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const auto* mat = result->as_matrix();
    // Max of 1 2 3 4 is 4
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 4.0);
    // Max of 5 6 7 8 is 8
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 8.0);
}

// ISO 9.2.1: Minus reduce (right-to-left)
TEST_F(OperatorsTest, NDArrayReduceMinus) {
    // -/2 2 4⍴1 2 3 4 5 6 7 8
    // First row: 1-2-3-4 = 1-(2-(3-4)) = 1-(2-(-1)) = 1-3 = -2
    Value* result = machine->eval("-/2 2 4⍴1 2 3 4 5 6 7 8");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), -2.0);  // 1-(2-(3-4))
    EXPECT_DOUBLE_EQ((*mat)(0, 1), -2.0);  // 5-(6-(7-8))
}

// ========================================================================
// NDARRAY N-wise Reduce Tests (ISO 13751 §9.2.3)
// ========================================================================

// ISO 9.2.3: N f/B applies f between successive N-element windows
// For NDARRAY, reduces along last axis by default
TEST_F(OperatorsTest, NDArrayNwiseReduceLastAxis) {
    // 2+/2 3 4⍴⍳24 - pairwise sums along last axis → 2×3×3 NDARRAY
    Value* result = machine->eval("2+/2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 3);  // 4 - 2 + 1 = 3
    // First fiber [0,0,*]: pairwise sums of 1,2,3,4 → 3,5,7
    EXPECT_DOUBLE_EQ((*nd->data)(0), 3.0);   // 1+2
    EXPECT_DOUBLE_EQ((*nd->data)(1), 5.0);   // 2+3
    EXPECT_DOUBLE_EQ((*nd->data)(2), 7.0);   // 3+4
}

// ISO 9.2.3: N f/[K]B applies N-wise reduction along axis K
TEST_F(OperatorsTest, NDArrayNwiseReduceFirstAxis) {
    // 2+/[1]2 3 4⍴⍳24 - pairwise along first axis → 1×3×4 which collapses to 3×4 matrix
    Value* result = machine->eval("2+/[1]2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    // 2 - 2 + 1 = 1 along first axis, result is 1×3×4 (or scalar/matrix depending on impl)
    // Position [0,0,0]: 1+13=14
    if (result->is_ndarray()) {
        const auto* nd = result->as_ndarray();
        EXPECT_EQ(nd->shape[0], 1);
        EXPECT_EQ(nd->shape[1], 3);
        EXPECT_EQ(nd->shape[2], 4);
        EXPECT_DOUBLE_EQ((*nd->data)(0), 14.0);  // 1+13
    } else if (result->is_matrix()) {
        // Collapsed to matrix
        const auto* mat = result->as_matrix();
        EXPECT_DOUBLE_EQ((*mat)(0, 0), 14.0);
    }
}

// ISO 9.2.3: N-wise along middle axis
TEST_F(OperatorsTest, NDArrayNwiseReduceMiddleAxis) {
    // 2+/[2]2 3 4⍴⍳24 - pairwise along axis 2 → 2×2×4 NDARRAY
    Value* result = machine->eval("2+/[2]2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 2);  // 3 - 2 + 1 = 2
    EXPECT_EQ(nd->shape[2], 4);
    // Position [0,0,0]: 1+5=6 (row 0, col 0 of first plane + row 1, col 0)
    EXPECT_DOUBLE_EQ((*nd->data)(0), 6.0);
}

// ISO 9.2.3: Triplets (N=3)
TEST_F(OperatorsTest, NDArrayNwiseReduceTriplets) {
    // 3+/2 3 5⍴⍳30 - triplet sums along last axis → 2×3×3 NDARRAY
    Value* result = machine->eval("3+/2 3 5⍴⍳30");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape[2], 3);  // 5 - 3 + 1 = 3
    // First fiber: sums of [1,2,3], [2,3,4], [3,4,5] → 6, 9, 12
    EXPECT_DOUBLE_EQ((*nd->data)(0), 6.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 9.0);
    EXPECT_DOUBLE_EQ((*nd->data)(2), 12.0);
}

// ISO 9.2.3: Negative N reverses window elements
TEST_F(OperatorsTest, NDArrayNwiseReduceReversed) {
    // ¯2-/2 2 4⍴1 2 3 4 5 6 7 8 - reversed pairwise difference
    // Windows [1,2],[2,3],[3,4] reversed to [2,1],[3,2],[4,3]
    // Results: 2-1=1, 3-2=1, 4-3=1
    Value* result = machine->eval("¯2-/2 2 4⍴1 2 3 4 5 6 7 8");
    ASSERT_NE(result, nullptr);
    // First row, first plane: differences of 1,2,3,4 reversed → all 1s
    if (result->is_ndarray()) {
        const auto* nd = result->as_ndarray();
        EXPECT_DOUBLE_EQ((*nd->data)(0), 1.0);  // 2-1
        EXPECT_DOUBLE_EQ((*nd->data)(1), 1.0);  // 3-2
        EXPECT_DOUBLE_EQ((*nd->data)(2), 1.0);  // 4-3
    } else if (result->is_matrix()) {
        const auto* mat = result->as_matrix();
        EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
        EXPECT_DOUBLE_EQ((*mat)(0, 1), 1.0);
        EXPECT_DOUBLE_EQ((*mat)(0, 2), 1.0);
    }
}

// ISO 9.2.3: N equals axis length returns one result per fiber
TEST_F(OperatorsTest, NDArrayNwiseReduceFullWindow) {
    // 4+/2 3 4⍴⍳24 - window size equals last axis → 2×3 matrix
    Value* result = machine->eval("4+/2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const auto* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    // Same as full reduce
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 10.0);  // 1+2+3+4
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 26.0);  // 5+6+7+8
}

// ISO 9.2.3: 4D N-wise reduction
TEST_F(OperatorsTest, NDArrayNwiseReduce4D) {
    // 2+/2 2 3 4⍴⍳48 - pairwise along last axis → 2×2×3×3 NDARRAY
    Value* result = machine->eval("2+/2 2 3 4⍴⍳48");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 4);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 2);
    EXPECT_EQ(nd->shape[2], 3);
    EXPECT_EQ(nd->shape[3], 3);  // 4 - 2 + 1 = 3
}

// ISO 9.2.3: ⎕IO=0 support
TEST_F(OperatorsTest, NDArrayNwiseReduceIO0) {
    // With ⎕IO=0, axis numbers are 0-indexed
    Value* result = machine->eval("⎕IO←0 ⋄ 2+/[0]2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    // Axis 0 with window 2 on dimension 2 → 1×3×4
    if (result->is_ndarray()) {
        const auto* nd = result->as_ndarray();
        EXPECT_EQ(nd->shape[0], 1);
        // 0+12=12 (⎕IO=0 so ⍳24 is 0..23)
        EXPECT_DOUBLE_EQ((*nd->data)(0), 12.0);
    }
}

// ISO 9.2.3: Times N-wise
TEST_F(OperatorsTest, NDArrayNwiseReduceTimes) {
    // 2×/2 3 4⍴⍳24 - pairwise products
    Value* result = machine->eval("2×/2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    // First fiber: products of [1,2],[2,3],[3,4] → 2, 6, 12
    EXPECT_DOUBLE_EQ((*nd->data)(0), 2.0);   // 1×2
    EXPECT_DOUBLE_EQ((*nd->data)(1), 6.0);   // 2×3
    EXPECT_DOUBLE_EQ((*nd->data)(2), 12.0);  // 3×4
}

// ========================================================================
// NDARRAY Scan Tests (ISO 13751 §9.2.2)
// ========================================================================

// ISO 9.2.2: f\B - result has same shape as B, contains prefix reductions
TEST_F(OperatorsTest, NDArrayScanLastAxis) {
    // +\2 3 4⍴⍳24 - scan along last axis, result is 2×3×4
    Value* result = machine->eval("+\\2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    // First fiber [0,0,*]: 1, 1+2=3, 3+3=6, 6+4=10
    EXPECT_DOUBLE_EQ((*nd->data)(0), 1.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 3.0);
    EXPECT_DOUBLE_EQ((*nd->data)(2), 6.0);
    EXPECT_DOUBLE_EQ((*nd->data)(3), 10.0);
}

// ISO 9.2.2: f⍀B - scan along first axis
TEST_F(OperatorsTest, NDArrayScanFirst) {
    // +⍀2 3 4⍴⍳24 - scan along first axis
    Value* result = machine->eval("+⍀2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape[0], 2);
    // First "plane" unchanged: 1..12
    EXPECT_DOUBLE_EQ((*nd->data)(0), 1.0);
    // Second "plane" [1,0,0]: 1+13=14
    EXPECT_DOUBLE_EQ((*nd->data)(12), 14.0);
    // [1,0,1]: 2+14=16
    EXPECT_DOUBLE_EQ((*nd->data)(13), 16.0);
}

// ISO 9.2.2: f\[K]B - scan along axis K
TEST_F(OperatorsTest, NDArrayScanAxisMiddle) {
    // +\[2]2 3 4⍴⍳24 - scan along axis 2 (middle)
    Value* result = machine->eval("+\\[2]2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    // Shape unchanged
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    // Fiber [0,*,0]: 1, 1+5=6, 6+9=15
    EXPECT_DOUBLE_EQ((*nd->data)(0), 1.0);
    EXPECT_DOUBLE_EQ((*nd->data)(4), 6.0);
    EXPECT_DOUBLE_EQ((*nd->data)(8), 15.0);
}

// ISO 9.2.2: Scan on 4D array
TEST_F(OperatorsTest, NDArrayScan4D) {
    // +\2 2 2 3⍴⍳24 - scan along last axis
    Value* result = machine->eval("+\\2 2 2 3⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 4);
    // First fiber: 1, 1+2=3, 3+3=6
    EXPECT_DOUBLE_EQ((*nd->data)(0), 1.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 3.0);
    EXPECT_DOUBLE_EQ((*nd->data)(2), 6.0);
}

// ISO 9.2.2: Times scan (running product)
TEST_F(OperatorsTest, NDArrayScanTimes) {
    // ×\2 2 3⍴⍳12 - 3D NDARRAY
    Value* result = machine->eval("×\\2 2 3⍴⍳12");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    // First fiber [0,0,*]: 1, 1×2=2, 2×3=6
    EXPECT_DOUBLE_EQ((*nd->data)(0), 1.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 2.0);
    EXPECT_DOUBLE_EQ((*nd->data)(2), 6.0);
}

// ISO 9.2.2: Scan with ⎕IO=0
TEST_F(OperatorsTest, NDArrayScanIO0) {
    Value* result = machine->eval("⎕IO←0 ⋄ +\\2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    // With ⎕IO=0, ⍳24 is 0..23
    // First fiber: 0, 0+1=1, 1+2=3, 3+3=6
    EXPECT_DOUBLE_EQ((*nd->data)(0), 0.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 1.0);
    EXPECT_DOUBLE_EQ((*nd->data)(2), 3.0);
    EXPECT_DOUBLE_EQ((*nd->data)(3), 6.0);
}

// ============================================================================
// NDARRAY Duplicate/Commute (f⍨) Tests - ISO 9.2.4-5
// ============================================================================
// ISO 9.2.4 Duplicate: f⍨ B → B f B
// ISO 9.2.5 Commute: A f⍨ B → B f A

// Duplicate on 3D NDARRAY: +⍨ preserves shape, doubles values
TEST_F(OperatorsTest, NDArrayDuplicatePlus) {
    Value* result = eval(machine, "+⍨ 2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    // Each element doubled: 1+1=2, 2+2=4, etc.
    EXPECT_DOUBLE_EQ((*nd->data)(0), 2.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 4.0);
    EXPECT_DOUBLE_EQ((*nd->data)(23), 48.0);
}

// Duplicate on 3D NDARRAY: ×⍨ squares each element
TEST_F(OperatorsTest, NDArrayDuplicateTimes) {
    Value* result = eval(machine, "×⍨ 2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3);
    // Each element squared: 1×1=1, 2×2=4, 3×3=9, etc.
    EXPECT_DOUBLE_EQ((*nd->data)(0), 1.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 4.0);
    EXPECT_DOUBLE_EQ((*nd->data)(2), 9.0);
    EXPECT_DOUBLE_EQ((*nd->data)(23), 576.0);  // 24×24
}

// Duplicate on 3D NDARRAY: -⍨ gives zeros (B-B=0)
TEST_F(OperatorsTest, NDArrayDuplicateMinus) {
    Value* result = eval(machine, "-⍨ 2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    // Each element minus itself = 0
    for (int i = 0; i < 24; i++) {
        EXPECT_DOUBLE_EQ((*nd->data)(i), 0.0);
    }
}

// Commute with scalar and NDARRAY: scalar -⍨ NDARRAY → NDARRAY - scalar
TEST_F(OperatorsTest, NDArrayCommuteScalarLeft) {
    Value* result = eval(machine, "10 -⍨ 2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3);
    // B - A: 1-10=-9, 2-10=-8, etc.
    EXPECT_DOUBLE_EQ((*nd->data)(0), -9.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), -8.0);
    EXPECT_DOUBLE_EQ((*nd->data)(9), 0.0);   // 10-10
    EXPECT_DOUBLE_EQ((*nd->data)(23), 14.0); // 24-10
}

// Commute with NDARRAY and scalar: NDARRAY -⍨ scalar → scalar - NDARRAY
TEST_F(OperatorsTest, NDArrayCommuteScalarRight) {
    Value* result = eval(machine, "(2 3 4⍴⍳24) -⍨ 100");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3);
    // B - A: 100-1=99, 100-2=98, etc.
    EXPECT_DOUBLE_EQ((*nd->data)(0), 99.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 98.0);
    EXPECT_DOUBLE_EQ((*nd->data)(23), 76.0);  // 100-24
}

// Commute with two NDARRAYs of same shape
TEST_F(OperatorsTest, NDArrayCommuteBothNDArray) {
    Value* result = eval(machine, "(2 3 4⍴⍳24) -⍨ (2 3 4⍴24-⍳24)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3);
    // Left: 1,2,3,...,24  Right: 23,22,21,...,0
    // Commute: Right - Left = (23-1), (22-2), etc. = 22, 20, 18, ...
    EXPECT_DOUBLE_EQ((*nd->data)(0), 22.0);   // 23-1
    EXPECT_DOUBLE_EQ((*nd->data)(1), 20.0);   // 22-2
    EXPECT_DOUBLE_EQ((*nd->data)(11), 0.0);   // 12-12
}

// Commute on 4D NDARRAY
TEST_F(OperatorsTest, NDArrayCommute4D) {
    Value* result = eval(machine, "1 +⍨ 2 2 2 3⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 4);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 2);
    EXPECT_EQ(nd->shape[2], 2);
    EXPECT_EQ(nd->shape[3], 3);
    // B + A: 1+1=2, 2+1=3, etc.
    EXPECT_DOUBLE_EQ((*nd->data)(0), 2.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 3.0);
}

// Duplicate with *⍨ (power): B*B = B squared
TEST_F(OperatorsTest, NDArrayDuplicatePower) {
    Value* result = eval(machine, "*⍨ 2 3⍴1 2 3 4 5 6");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    // 1^1=1, 2^2=4, 3^3=27, 4^4=256, 5^5=3125, 6^6=46656
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), 27.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 256.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 3125.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 46656.0);
}

// Commute with ÷⍨: A ÷⍨ B → B ÷ A
TEST_F(OperatorsTest, NDArrayCommuteDivide) {
    Value* result = eval(machine, "2 ÷⍨ 2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    // B ÷ A: 1÷2=0.5, 2÷2=1, 3÷2=1.5, etc.
    EXPECT_DOUBLE_EQ((*nd->data)(0), 0.5);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 1.0);
    EXPECT_DOUBLE_EQ((*nd->data)(2), 1.5);
    EXPECT_DOUBLE_EQ((*nd->data)(3), 2.0);
}

// ============================================================================
// NDARRAY Each (f¨) Tests - ISO 9.2.6
// ============================================================================

// ISO 9.2.6: Monadic each on 3D NDARRAY - apply function to each element
TEST_F(OperatorsTest, NDArrayEachMonadic) {
    Value* result = machine->eval("-¨2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    // Each element negated: 1→-1, 2→-2, etc.
    EXPECT_DOUBLE_EQ((*nd->data)(0), -1.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), -2.0);
    EXPECT_DOUBLE_EQ((*nd->data)(23), -24.0);
}

// ISO 9.2.6: Dyadic each with scalar extension (scalar + NDARRAY)
TEST_F(OperatorsTest, NDArrayEachScalarLeft) {
    Value* result = machine->eval("10+¨2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    // 10 + each element: 10+1=11, 10+2=12, etc.
    EXPECT_DOUBLE_EQ((*nd->data)(0), 11.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 12.0);
    EXPECT_DOUBLE_EQ((*nd->data)(23), 34.0);
}

// ISO 9.2.6: Dyadic each with scalar extension (NDARRAY + scalar)
TEST_F(OperatorsTest, NDArrayEachScalarRight) {
    Value* result = machine->eval("(2 3 4⍴⍳24)×¨10");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    // Each element × 10: 1×10=10, 2×10=20, etc.
    EXPECT_DOUBLE_EQ((*nd->data)(0), 10.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 20.0);
    EXPECT_DOUBLE_EQ((*nd->data)(23), 240.0);
}

// ISO 9.2.6: Dyadic each with matching NDARRAY shapes
TEST_F(OperatorsTest, NDArrayEachBothNDArray) {
    Value* result = machine->eval("(2 3 4⍴⍳24)+¨2 3 4⍴24-⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    // (⍳24) +¨ (24-⍳24) = 24 for all elements
    // 1+(24-1)=24, 2+(24-2)=24, etc.
    for (int i = 0; i < 24; i++) {
        EXPECT_DOUBLE_EQ((*nd->data)(i), 24.0);
    }
}

// ISO 9.2.6: Each preserves 4D shape
TEST_F(OperatorsTest, NDArrayEach4D) {
    Value* result = machine->eval("⌈¨2 2 2 3⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 4);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 2);
    EXPECT_EQ(nd->shape[2], 2);
    EXPECT_EQ(nd->shape[3], 3);
}

// ISO 9.2.6: Each with ⎕IO=0
TEST_F(OperatorsTest, NDArrayEachIO0) {
    Value* result = machine->eval("⎕IO←0 ⋄ -¨2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    // With ⎕IO=0, ⍳24 is 0..23, negated: 0, -1, -2, ..., -23
    EXPECT_DOUBLE_EQ((*nd->data)(0), 0.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), -1.0);
    EXPECT_DOUBLE_EQ((*nd->data)(23), -23.0);
}

// ============================================================================
// NDARRAY Outer Product (∘.f) Tests - ISO 9.3.1
// ============================================================================

// ISO 9.3.1: Matrix ∘.f vector produces 3D NDARRAY
// Result shape: (⍴A),⍴B = (2 3),(4) = 2 3 4
TEST_F(OperatorsTest, NDArrayOuterMatrixVector) {
    Value* result = machine->eval("(2 3⍴⍳6)∘.+⍳4");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    // Element [0,0,0] = mat[0,0] + vec[0] = 1+1 = 2
    EXPECT_DOUBLE_EQ((*nd->data)(0), 2.0);
    // Element [0,0,1] = mat[0,0] + vec[1] = 1+2 = 3
    EXPECT_DOUBLE_EQ((*nd->data)(1), 3.0);
    // Element [0,0,3] = mat[0,0] + vec[3] = 1+4 = 5
    EXPECT_DOUBLE_EQ((*nd->data)(3), 5.0);
    // Element [0,1,0] = mat[0,1] + vec[0] = 2+1 = 3
    EXPECT_DOUBLE_EQ((*nd->data)(4), 3.0);
}

// ISO 9.3.1: Vector ∘.f matrix produces 3D NDARRAY
// Result shape: (⍴A),⍴B = (3),(2 4) = 3 2 4
TEST_F(OperatorsTest, NDArrayOuterVectorMatrix) {
    Value* result = machine->eval("(⍳3)∘.×2 4⍴⍳8");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 3);
    EXPECT_EQ(nd->shape[1], 2);
    EXPECT_EQ(nd->shape[2], 4);
    // Element [0,0,0] = vec[0] × mat[0,0] = 1×1 = 1
    EXPECT_DOUBLE_EQ((*nd->data)(0), 1.0);
    // Element [0,0,1] = vec[0] × mat[0,1] = 1×2 = 2
    EXPECT_DOUBLE_EQ((*nd->data)(1), 2.0);
    // Element [1,0,0] = vec[1] × mat[0,0] = 2×1 = 2
    EXPECT_DOUBLE_EQ((*nd->data)(8), 2.0);
}

// ISO 9.3.1: Matrix ∘.f matrix produces 4D NDARRAY
// Result shape: (⍴A),⍴B = (2 3),(2 2) = 2 3 2 2
TEST_F(OperatorsTest, NDArrayOuterMatrixMatrix) {
    Value* result = machine->eval("(2 3⍴⍳6)∘.+2 2⍴⍳4");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 4);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 2);
    EXPECT_EQ(nd->shape[3], 2);
    // Element [0,0,0,0] = A[0,0] + B[0,0] = 1+1 = 2
    EXPECT_DOUBLE_EQ((*nd->data)(0), 2.0);
    // Element [0,0,0,1] = A[0,0] + B[0,1] = 1+2 = 3
    EXPECT_DOUBLE_EQ((*nd->data)(1), 3.0);
}

// ISO 9.3.1: NDARRAY ∘.f vector produces higher-dim result
// Result shape: (⍴A),⍴B = (2 3 4),(2) = 2 3 4 2
TEST_F(OperatorsTest, NDArrayOuterNDArrayVector) {
    Value* result = machine->eval("(2 3 4⍴⍳24)∘.+1 2");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 4);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    EXPECT_EQ(nd->shape[3], 2);
    // Element [0,0,0,0] = nd[0,0,0] + vec[0] = 1+1 = 2
    EXPECT_DOUBLE_EQ((*nd->data)(0), 2.0);
    // Element [0,0,0,1] = nd[0,0,0] + vec[1] = 1+2 = 3
    EXPECT_DOUBLE_EQ((*nd->data)(1), 3.0);
}

// ISO 9.3.1: Vector ∘.f NDARRAY produces higher-dim result
// Result shape: (⍴A),⍴B = (2),(2 3 4) = 2 2 3 4
TEST_F(OperatorsTest, NDArrayOuterVectorNDArray) {
    Value* result = machine->eval("(1 2)∘.×2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 4);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 2);
    EXPECT_EQ(nd->shape[2], 3);
    EXPECT_EQ(nd->shape[3], 4);
    // Element [0,0,0,0] = vec[0] × nd[0,0,0] = 1×1 = 1
    EXPECT_DOUBLE_EQ((*nd->data)(0), 1.0);
    // Element [1,0,0,0] = vec[1] × nd[0,0,0] = 2×1 = 2
    EXPECT_DOUBLE_EQ((*nd->data)(24), 2.0);
}

// ISO 9.3.1: Outer product with multiplication (table)
TEST_F(OperatorsTest, NDArrayOuterMultiply3D) {
    Value* result = machine->eval("(2 3⍴1 2 3 4 5 6)∘.×⍳4");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    // Shape is 2 3 4
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
}

// ISO 9.3.1: Outer product with ⎕IO=0
TEST_F(OperatorsTest, NDArrayOuterIO0) {
    Value* result = machine->eval("⎕IO←0 ⋄ (2 3⍴⍳6)∘.+⍳4");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    // With ⎕IO=0: mat is 0..5, vec is 0..3
    // Element [0,0,0] = mat[0,0] + vec[0] = 0+0 = 0
    EXPECT_DOUBLE_EQ((*nd->data)(0), 0.0);
    // Element [0,0,1] = mat[0,0] + vec[1] = 0+1 = 1
    EXPECT_DOUBLE_EQ((*nd->data)(1), 1.0);
}

// ============================================================================
// NDARRAY Inner Product (f.g) Tests - ISO 9.3.2
// ============================================================================
// Result shape: (⍴A)[⍳0⌈¯1+⍴⍴A],(⍴B)[1+⍳0⌈¯1+⍴⍴B]
// i.e., (¯1↓⍴A),1↓⍴B - all but last of A, all but first of B

// ISO 9.3.2: 3D NDARRAY +.× vector
// Shape: (2×3×4) +.× (4) → 2×3 matrix
// Last dim of A (4) = length of B (4)
TEST_F(OperatorsTest, NDArrayInnerNDArrayVector) {
    Value* result = machine->eval("(2 3 4⍴⍳24)+.×⍳4");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    // Result[0,0] = +/ (1 2 3 4) × (1 2 3 4) = 1+4+9+16 = 30
    const auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 30.0);
    // Result[0,1] = +/ (5 6 7 8) × (1 2 3 4) = 5+12+21+32 = 70
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 70.0);
}

// ISO 9.3.2: vector +.× 3D NDARRAY
// Shape: (4) +.× (4×3×2) → 3×2 matrix
TEST_F(OperatorsTest, NDArrayInnerVectorNDArray) {
    Value* result = machine->eval("(⍳4)+.×4 3 2⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 2);
    // Result[0,0] = +/ (1 2 3 4) × (1 7 13 19) = 1+14+39+76 = 130
    const auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 130.0);
}

// ISO 9.3.2: Matrix +.× 3D NDARRAY → 3D result
// Shape: (2×4) +.× (4×3×2) → 2×3×2 (3D NDARRAY)
TEST_F(OperatorsTest, NDArrayInnerMatrixNDArray) {
    Value* result = machine->eval("(2 4⍴⍳8)+.×4 3 2⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 2);
}

// ISO 9.3.2: 3D NDARRAY +.× matrix → 3D result
// Shape: (2×3×4) +.× (4×5) → 2×3×5 (3D NDARRAY)
TEST_F(OperatorsTest, NDArrayInnerNDArrayMatrix) {
    Value* result = machine->eval("(2 3 4⍴⍳24)+.×4 5⍴⍳20");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 5);
}

// ISO 9.3.2: 3D +.× 3D → 4D result
// Shape: (2×3×4) +.× (4×5×6) → 2×3×5×6 (4D NDARRAY)
TEST_F(OperatorsTest, NDArrayInnerNDArrayNDArray) {
    Value* result = machine->eval("(2 3 4⍴⍳24)+.×4 5 6⍴⍳120");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 4);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 5);
    EXPECT_EQ(nd->shape[3], 6);
}

// ISO 9.3.2: Dimension mismatch error
// Last dim of lhs (4) must equal first dim of rhs - here rhs first dim is 5
TEST_F(OperatorsTest, NDArrayInnerDimensionMismatch) {
    Value* lhs = machine->eval("2 3 4⍴⍳24");
    Value* rhs = machine->eval("5 6⍴⍳30");
    Value* f = machine->heap->allocate_primitive(&prim_plus);
    Value* g = machine->heap->allocate_primitive(&prim_times);

    machine->kont_stack.clear();
    op_inner_product(machine, nullptr, lhs, f, g, rhs);

    // Should have pushed a ThrowErrorK for LENGTH ERROR
    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// ISO 9.3.2: Inner product with ⌈.+ (max-plus)
TEST_F(OperatorsTest, NDArrayInnerMaxPlus) {
    Value* result = machine->eval("(2 3 4⍴⍳24)⌈.+4 5⍴⍳20");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 5);
}

// ============================================================================
// NDARRAY Rank Operator Tests (ISO 9.3.3-5)
// ============================================================================

// ISO 9.3.4: Monadic rank on 3D NDARRAY - apply +/ to each 1-cell (vector)
// +/⍤1 on 2×3×4 array reduces each row (4 elements) to scalar
// Result shape: 2×3 (frame shape)
TEST_F(OperatorsTest, NDArrayRankMonadicReduce1Cell) {
    Value* result = machine->eval("+/⍤1 (2 3 4⍴⍳24)");
    ASSERT_NE(result, nullptr);
    // Result should be 2×3 matrix (frame of 2×3×4 with 1-cells removed)
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    // First row: sums of rows 1-3 of first matrix
    // Row 0: 1+2+3+4=10, Row 1: 5+6+7+8=26, Row 2: 9+10+11+12=42
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 10.0);  // 1+2+3+4
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 26.0);  // 5+6+7+8
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 42.0);  // 9+10+11+12
}

// ISO 9.3.4: Apply ⌽ (reverse) to each 2-cell of 3D array
// ⌽⍤2 on 2×3×4 reverses within each row of each 3×4 matrix
TEST_F(OperatorsTest, NDArrayRankMonadicReverse2Cell) {
    Value* result = machine->eval("⌽⍤2 (2 3 4⍴⍳24)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    // ⌽ reverses along LAST axis (within each row)
    // Original row 0: [1,2,3,4] → reversed: [4,3,2,1]
    EXPECT_DOUBLE_EQ((*nd->data)(0), 4.0);   // First element: 4
    EXPECT_DOUBLE_EQ((*nd->data)(3), 1.0);   // Last of first row: 1
    // Original row 2: [9,10,11,12] → reversed: [12,11,10,9]
    EXPECT_DOUBLE_EQ((*nd->data)(8), 12.0);  // First of third row: 12
}

// ISO 9.3.5: Dyadic rank - matrix + each 2-cell of 3D array
// N34+⍤2 N234 adds 3×4 matrix to each 3×4 cell of 2×3×4 array
TEST_F(OperatorsTest, NDArrayRankDyadicMatrixPlus2Cell) {
    Value* result = machine->eval("(3 4⍴10×⍳12)+⍤2 (2 3 4⍴⍳24)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
    // First element: 10 + 1 = 11
    EXPECT_DOUBLE_EQ((*nd->data)(0), 11.0);
    // Element at [0,1,0]: 50 + 5 = 55
    EXPECT_DOUBLE_EQ((*nd->data)(4), 55.0);
}

// ISO 9.3.5: Scalar extension - scalar ⍤0 applied to each 0-cell
// 10+⍤0 on 2×3×4 array adds 10 to each element
TEST_F(OperatorsTest, NDArrayRankDyadicScalarExtension) {
    Value* result = machine->eval("10+⍤0 (2 3 4⍴⍳24)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    // Each element increased by 10
    EXPECT_DOUBLE_EQ((*nd->data)(0), 11.0);   // 1+10
    EXPECT_DOUBLE_EQ((*nd->data)(23), 34.0);  // 24+10
}

// ISO 9.3.5: Vector catenated to each row of matrix (spec example: N3,⍤1 N34)
// Left: 3-vector (frame empty, 1 cell), Right: 3×4 matrix (frame [3], 3 cells)
// With scalar extension, same vector is catenated to each row → 3×7 matrix
TEST_F(OperatorsTest, NDArrayRankDyadicVectorCatRow) {
    Value* result = machine->eval("(100 200 300),⍤1 (3 4⍴⍳12)");
    ASSERT_NE(result, nullptr);
    // Per ISO 9.3.5: A conforms to B when left frame empty
    // Result: 3 rows × 7 cols (3-vector + 4-element row)
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 7);
    const Eigen::MatrixXd* mat = result->as_matrix();
    // Each row: (100 200 300) , (row from matrix)
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 100.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 3), 1.0);   // First element of first matrix row
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 100.0); // Same vector repeated
    EXPECT_DOUBLE_EQ((*mat)(1, 3), 5.0);   // First element of second matrix row
}

// ISO 9.3.4: Apply ⍳ to each 0-cell of vector → matrix result
// ⍳⍤0 on 3-vector gives matrix where row i is ⍳(i)
TEST_F(OperatorsTest, NDArrayRankIotaEach0Cell) {
    Value* result = machine->eval("⍳⍤0 (1 2 3)");
    ASSERT_NE(result, nullptr);
    // Result: matrix with rows ⍳1, ⍳2, ⍳3
    // But these have different lengths! So should be strand of vectors
    ASSERT_TRUE(result->is_strand());
    const auto* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 3);
    // ⍳1 = 1
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_EQ((*strand)[0]->size(), 1);
    // ⍳2 = 1 2
    ASSERT_TRUE((*strand)[1]->is_vector());
    EXPECT_EQ((*strand)[1]->size(), 2);
    // ⍳3 = 1 2 3
    ASSERT_TRUE((*strand)[2]->is_vector());
    EXPECT_EQ((*strand)[2]->size(), 3);
}

// 4D NDARRAY rank test
TEST_F(OperatorsTest, NDArrayRank4D) {
    // +/⍤1 on 2×2×3×4 reduces each innermost row
    // Result shape: 2×2×3
    Value* result = machine->eval("+/⍤1 (2 2 3 4⍴⍳48)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 2);
    EXPECT_EQ(nd->shape[2], 3);
    // First cell: 1+2+3+4=10
    EXPECT_DOUBLE_EQ((*nd->data)(0), 10.0);
}

// Negative rank specification (relative to array rank)
TEST_F(OperatorsTest, NDArrayRankNegative) {
    // +/⍤¯1 means rank = array_rank + (¯1) = 3 - 1 = 2
    // So +/⍤¯1 on 2×3×4 applies +/ to 2-cells (3×4 matrices)
    // +/ on 3×4 matrix reduces along LAST axis (sums each row)
    // Each 3×4 matrix → 3-vector (one sum per row)
    // Frame [2] + cell result [3] = 2×3 matrix
    Value* result = machine->eval("+/⍤¯1 (2 3 4⍴⍳24)");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
    // First matrix rows: [1,2,3,4]=10, [5,6,7,8]=26, [9,10,11,12]=42
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 26.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 42.0);
}

// Length error: frame mismatch in dyadic rank (ISO 9.3.5)
TEST_F(OperatorsTest, NDArrayRankLengthError) {
    // Frame mismatch: [2,3] vs [3,2] frames
    Value* lhs = machine->eval("2 3 4⍴1");
    Value* rhs = machine->eval("3 2 4⍴1");
    Value* fn = machine->heap->allocate_primitive(&prim_plus);
    Value* rank_spec = machine->heap->allocate_scalar(1.0);

    machine->kont_stack.clear();
    op_rank(machine, nullptr, lhs, fn, rank_spec, rhs);

    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
