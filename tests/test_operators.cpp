// Tests for APL operators

#include <gtest/gtest.h>
#include "machine.h"
#include "heap.h"
#include "primitives.h"
#include "operators.h"
#include "continuation.h"
#include <Eigen/Dense>

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
    op_inner_product(machine, lhs, f, g, rhs);

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

    op_commute(machine, fn, omega);

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

    op_commute(machine, fn, omega);

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

    op_commute_dyadic(machine, lhs, fn, nullptr, rhs);

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

    op_commute_dyadic(machine, lhs, fn, nullptr, rhs);

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

    fn_reduce(machine, count_vec, data_vec);

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

    fn_scan(machine, vec, vec);
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

    op_commute(machine, fn, omega);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, CommuteWithDivide) {
    // 3÷⍨12 → 12÷3 = 4
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Value* rhs = machine->heap->allocate_scalar(12.0);
    Value* fn = machine->heap->allocate_primitive(&prim_divide);

    op_commute_dyadic(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 4.0);
}

TEST_F(OperatorsTest, DuplicateMatrix) {
    // Test duplicate with matrix subtraction
    Eigen::MatrixXd mat(2, 2);
    mat << 5, 6,
           7, 8;
    Value* omega = machine->heap->allocate_matrix(mat);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    op_commute(machine, fn, omega);

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
    EXPECT_EQ(result->data.derived_op->op, &op_diaeresis);
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
    op_rank(machine, nullptr, not_fn, rank_spec, omega);

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
    op_rank(machine, lhs, fn, rank_spec, rhs);

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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
