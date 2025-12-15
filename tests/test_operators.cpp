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
    Value* omega = machine->heap->allocate_scalar(3.0);
    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    op_commute(machine, nullptr, fn, omega);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 6.0);
}

TEST_F(OperatorsTest, DuplicateVector) {
    // ×⍨vector → vector × vector (element-wise)
    Eigen::VectorXd vec(3);
    vec << 2, 3, 4;
    Value* omega = machine->heap->allocate_vector(vec);
    Value* fn = machine->heap->allocate_primitive(&prim_times);

    op_commute(machine, nullptr, fn, omega);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_vector());

    const Eigen::MatrixXd* result = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*result)(0, 0), 4.0);   // 2*2
    EXPECT_DOUBLE_EQ((*result)(1, 0), 9.0);   // 3*3
    EXPECT_DOUBLE_EQ((*result)(2, 0), 16.0);  // 4*4
}

TEST_F(OperatorsTest, CommuteScalars) {
    // 3-⍨4 → 4-3 = 1 (commute swaps arguments)
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Value* rhs = machine->heap->allocate_scalar(4.0);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    op_commute_dyadic(machine, nullptr, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);  // 4-3, not 3-4
}

TEST_F(OperatorsTest, CommuteVectors) {
    // vector1 - ⍨ vector2 → vector2 - vector1
    Eigen::VectorXd vec1(3);
    vec1 << 10, 20, 30;
    Value* lhs = machine->heap->allocate_vector(vec1);

    Eigen::VectorXd vec2(3);
    vec2 << 1, 2, 3;
    Value* rhs = machine->heap->allocate_vector(vec2);

    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    op_commute_dyadic(machine, nullptr, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_vector());

    const Eigen::MatrixXd* result = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*result)(0, 0), -9.0);   // 1-10
    EXPECT_DOUBLE_EQ((*result)(1, 0), -18.0);  // 2-20
    EXPECT_DOUBLE_EQ((*result)(2, 0), -27.0);  // 3-30
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
    Value* omega = machine->heap->allocate_scalar(4.0);
    Value* fn = machine->heap->allocate_primitive(&prim_divide);

    op_commute(machine, nullptr, fn, omega);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, CommuteWithDivide) {
    // 3÷⍨12 → 12÷3 = 4
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Value* rhs = machine->heap->allocate_scalar(12.0);
    Value* fn = machine->heap->allocate_primitive(&prim_divide);

    op_commute_dyadic(machine, nullptr, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 4.0);
}

TEST_F(OperatorsTest, CommuteMatrix) {
    // Matrix commute: swap left and right matrix args
    Eigen::MatrixXd lmat(2, 2);
    lmat << 10, 20, 30, 40;
    Eigen::MatrixXd rmat(2, 2);
    rmat << 1, 2, 3, 4;
    Value* lhs = machine->heap->allocate_matrix(lmat);
    Value* rhs = machine->heap->allocate_matrix(rmat);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    // lhs -⍨ rhs → rhs - lhs = (1-10, 2-20, 3-30, 4-40) = (-9, -18, -27, -36)
    op_commute_dyadic(machine, nullptr, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_matrix());
    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), -9.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), -18.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -27.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), -36.0);
}

TEST_F(OperatorsTest, DuplicateMatrix) {
    // Test duplicate with matrix subtraction
    Eigen::MatrixXd mat(2, 2);
    mat << 5, 6,
           7, 8;
    Value* omega = machine->heap->allocate_matrix(mat);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    op_commute(machine, nullptr, fn, omega);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_matrix());

    const Eigen::MatrixXd* result = machine->result->as_matrix();
    // Matrix - itself = zero matrix
    EXPECT_DOUBLE_EQ((*result)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*result)(0, 1), 0.0);
    EXPECT_DOUBLE_EQ((*result)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*result)(1, 1), 0.0);
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

// Helper to evaluate APL expression
static Value* eval(Machine* m, const char* expr) {
    return m->eval(expr);
}

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
    Value* result = eval(machine, "5 +/ 1 2 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 15.0);
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

TEST_F(OperatorsTest, DefinedOperatorFinalizesReduceCurry) {
    // Parenthesized reduce (+/1 2 3) should finalize to 6
    // Then strand with other values
    Value* result = eval(machine, "0 (+/1 2 3) 10");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 6.0);   // +/1 2 3 = 6
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 10.0);
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
