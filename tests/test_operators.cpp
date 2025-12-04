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
        init_global_environment(machine);
    }

    void TearDown() override {
        delete machine;
    }
};

// ========================================================================
// Outer Product Tests
// ========================================================================

TEST_F(OperatorsTest, OuterProductScalarScalar) {
    // 10 ∘.+ 5 → 15
    Value* lhs = machine->heap->allocate_scalar(10.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);
    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    op_outer_product(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_scalar());
    EXPECT_DOUBLE_EQ(machine->ctrl.value->as_scalar(), 15.0);
}

TEST_F(OperatorsTest, OuterProductVectorScalar) {
    // 1 2 3 ∘.+ 10 → 11 12 13
    Eigen::VectorXd lhs_vec(3);
    lhs_vec << 1, 2, 3;
    Value* lhs = machine->heap->allocate_vector(lhs_vec);
    Value* rhs = machine->heap->allocate_scalar(10.0);
    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    op_outer_product(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_vector());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 1);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*result)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*result)(2, 0), 13.0);
}

TEST_F(OperatorsTest, OuterProductScalarVector) {
    // 10 ∘.× 1 2 3 → 1×3 matrix: [10 20 30]
    // Per ISO spec: shape is (⍴A),⍴B = (⍬),3 = 1 3
    Value* lhs = machine->heap->allocate_scalar(10.0);

    Eigen::VectorXd rhs_vec(3);
    rhs_vec << 1, 2, 3;
    Value* rhs = machine->heap->allocate_vector(rhs_vec);

    Value* fn = machine->heap->allocate_primitive(&prim_times);

    op_outer_product(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_matrix());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 1);
    EXPECT_EQ(result->cols(), 3);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*result)(0, 1), 20.0);
    EXPECT_DOUBLE_EQ((*result)(0, 2), 30.0);
}

TEST_F(OperatorsTest, OuterProductVectorVector) {
    // 10 20 30 ∘.+ 1 2 3
    //   11 12 13
    //   21 22 23
    //   31 32 33

    Eigen::VectorXd lhs_vec(3);
    lhs_vec << 10, 20, 30;
    Value* lhs = machine->heap->allocate_vector(lhs_vec);

    Eigen::VectorXd rhs_vec(3);
    rhs_vec << 1, 2, 3;
    Value* rhs = machine->heap->allocate_vector(rhs_vec);

    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    op_outer_product(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_matrix());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 3);

    // Check all values
    EXPECT_DOUBLE_EQ((*result)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*result)(0, 1), 12.0);
    EXPECT_DOUBLE_EQ((*result)(0, 2), 13.0);
    EXPECT_DOUBLE_EQ((*result)(1, 0), 21.0);
    EXPECT_DOUBLE_EQ((*result)(1, 1), 22.0);
    EXPECT_DOUBLE_EQ((*result)(1, 2), 23.0);
    EXPECT_DOUBLE_EQ((*result)(2, 0), 31.0);
    EXPECT_DOUBLE_EQ((*result)(2, 1), 32.0);
    EXPECT_DOUBLE_EQ((*result)(2, 2), 33.0);
}

TEST_F(OperatorsTest, OuterProductMultiplication) {
    // 2 3 4 ∘.× 10 100
    //   20  200
    //   30  300
    //   40  400

    Eigen::VectorXd lhs_vec(3);
    lhs_vec << 2, 3, 4;
    Value* lhs = machine->heap->allocate_vector(lhs_vec);

    Eigen::VectorXd rhs_vec(2);
    rhs_vec << 10, 100;
    Value* rhs = machine->heap->allocate_vector(rhs_vec);

    Value* fn = machine->heap->allocate_primitive(&prim_times);

    op_outer_product(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_matrix());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 2);

    EXPECT_DOUBLE_EQ((*result)(0, 0), 20.0);
    EXPECT_DOUBLE_EQ((*result)(0, 1), 200.0);
    EXPECT_DOUBLE_EQ((*result)(1, 0), 30.0);
    EXPECT_DOUBLE_EQ((*result)(1, 1), 300.0);
    EXPECT_DOUBLE_EQ((*result)(2, 0), 40.0);
    EXPECT_DOUBLE_EQ((*result)(2, 1), 400.0);
}

TEST_F(OperatorsTest, OuterProductRequiresDyadicFunction) {
    // Test that monadic-only functions produce an error
    Eigen::VectorXd vec(2);
    vec << 1, 2;
    Value* lhs = machine->heap->allocate_vector(vec);
    Value* rhs = machine->heap->allocate_vector(vec);

    // prim_transpose has no dyadic form
    Value* fn = machine->heap->allocate_primitive(&prim_transpose);

    machine->kont_stack.clear();
    op_outer_product(machine, lhs, fn, nullptr, rhs);

    // Should have pushed a ThrowErrorK
    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// ========================================================================
// Inner Product Tests
// ========================================================================

TEST_F(OperatorsTest, InnerProductVectorDotProduct) {
    // 4 2 1 +.× 1 0 1 → 5 (dot product)
    Eigen::VectorXd lhs_vec(3);
    lhs_vec << 4, 2, 1;
    Value* lhs = machine->heap->allocate_vector(lhs_vec);

    Eigen::VectorXd rhs_vec(3);
    rhs_vec << 1, 0, 1;
    Value* rhs = machine->heap->allocate_vector(rhs_vec);

    Value* f = machine->heap->allocate_primitive(&prim_plus);
    Value* g = machine->heap->allocate_primitive(&prim_times);

    machine->kont_stack.clear();
    op_inner_product(machine, lhs, f, g, rhs);

    // Check for error first
    if (!machine->kont_stack.empty()) {
        ThrowErrorK* err = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
        if (err) {
            FAIL() << "Unexpected error: " << err->error_message;
        }
    }

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_scalar());
    EXPECT_DOUBLE_EQ(machine->ctrl.value->as_scalar(), 5.0);  // 4*1 + 2*0 + 1*1 = 5
}

TEST_F(OperatorsTest, InnerProductMatrixMultiplication) {
    // 2×2 matrix multiplication using +.×
    // [1 2]  ×  [5 6]  =  [19 22]
    // [3 4]     [7 8]     [43 50]

    Eigen::MatrixXd lhs_mat(2, 2);
    lhs_mat << 1, 2,
               3, 4;
    Value* lhs = machine->heap->allocate_matrix(lhs_mat);

    Eigen::MatrixXd rhs_mat(2, 2);
    rhs_mat << 5, 6,
               7, 8;
    Value* rhs = machine->heap->allocate_matrix(rhs_mat);

    Value* f = machine->heap->allocate_primitive(&prim_plus);
    Value* g = machine->heap->allocate_primitive(&prim_times);

    op_inner_product(machine, lhs, f, g, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_matrix());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 19.0);  // 1*5 + 2*7 = 19
    EXPECT_DOUBLE_EQ((*result)(0, 1), 22.0);  // 1*6 + 2*8 = 22
    EXPECT_DOUBLE_EQ((*result)(1, 0), 43.0);  // 3*5 + 4*7 = 43
    EXPECT_DOUBLE_EQ((*result)(1, 1), 50.0);  // 3*6 + 4*8 = 50
}

TEST_F(OperatorsTest, InnerProductMatrixVector) {
    // Matrix × vector: [1 2] +.× [5]  =  [19]
    //                  [3 4]       [7]     [43]

    Eigen::MatrixXd lhs_mat(2, 2);
    lhs_mat << 1, 2,
               3, 4;
    Value* lhs = machine->heap->allocate_matrix(lhs_mat);

    Eigen::VectorXd rhs_vec(2);
    rhs_vec << 5, 7;
    Value* rhs = machine->heap->allocate_vector(rhs_vec);

    Value* f = machine->heap->allocate_primitive(&prim_plus);
    Value* g = machine->heap->allocate_primitive(&prim_times);

    op_inner_product(machine, lhs, f, g, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_vector());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 1);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 19.0);  // 1*5 + 2*7
    EXPECT_DOUBLE_EQ((*result)(1, 0), 43.0);  // 3*5 + 4*7
}

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
// Each Operator Tests
// ========================================================================

TEST_F(OperatorsTest, EachScalar) {
    // -¨5 → -5
    Value* omega = machine->heap->allocate_scalar(5.0);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    op_each(machine, fn, omega);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_scalar());
    EXPECT_DOUBLE_EQ(machine->ctrl.value->as_scalar(), -5.0);
}

TEST_F(OperatorsTest, EachVector) {
    // -¨1 2 3 → -1 -2 -3
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* omega = machine->heap->allocate_vector(vec);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    op_each(machine, fn, omega);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_vector());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 1);
    EXPECT_DOUBLE_EQ((*result)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*result)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*result)(2, 0), -3.0);
}

TEST_F(OperatorsTest, EachMatrix) {
    // ÷¨ on 2×2 matrix (reciprocal of each element)
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2,
           4, 5;
    Value* omega = machine->heap->allocate_matrix(mat);
    Value* fn = machine->heap->allocate_primitive(&prim_divide);  // ÷ monadic is reciprocal

    op_each(machine, fn, omega);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_matrix());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 1.0);    // 1/1
    EXPECT_DOUBLE_EQ((*result)(0, 1), 0.5);    // 1/2
    EXPECT_DOUBLE_EQ((*result)(1, 0), 0.25);   // 1/4
    EXPECT_DOUBLE_EQ((*result)(1, 1), 0.2);    // 1/5
}

// ========================================================================
// Duplicate/Commute Operator Tests
// ========================================================================

TEST_F(OperatorsTest, DuplicateScalar) {
    // +⍨3 → 3+3 = 6
    Value* omega = machine->heap->allocate_scalar(3.0);
    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    op_commute(machine, fn, omega);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_scalar());
    EXPECT_DOUBLE_EQ(machine->ctrl.value->as_scalar(), 6.0);
}

TEST_F(OperatorsTest, DuplicateVector) {
    // ×⍨vector → vector × vector (element-wise)
    Eigen::VectorXd vec(3);
    vec << 2, 3, 4;
    Value* omega = machine->heap->allocate_vector(vec);
    Value* fn = machine->heap->allocate_primitive(&prim_times);

    op_commute(machine, fn, omega);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_vector());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
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

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_scalar());
    EXPECT_DOUBLE_EQ(machine->ctrl.value->as_scalar(), 1.0);  // 4-3, not 3-4
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

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_vector());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_DOUBLE_EQ((*result)(0, 0), -9.0);   // 1-10
    EXPECT_DOUBLE_EQ((*result)(1, 0), -18.0);  // 2-20
    EXPECT_DOUBLE_EQ((*result)(2, 0), -27.0);  // 3-30
}

// ========================================================================
// Reduce Operator Tests
// ========================================================================

TEST_F(OperatorsTest, ReduceVector) {
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    fn_reduce(machine, plus_fn, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);  // 1+2+3+4
}

TEST_F(OperatorsTest, ReduceWithMultiply) {
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* times_fn = machine->heap->allocate_primitive(&prim_times);

    fn_reduce(machine, times_fn, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 24.0);  // 1*2*3*4
}

TEST_F(OperatorsTest, ReduceMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    fn_reduce(machine, plus_fn, mat);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 2);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 15.0);  // 4+5+6
}

TEST_F(OperatorsTest, ReduceFirstMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    fn_reduce_first(machine, plus_fn, mat);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);  // 1+4
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 7.0);  // 2+5
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 9.0);  // 3+6
}

TEST_F(OperatorsTest, ReduceMultiply) {
    Value* func = machine->heap->allocate_primitive(&prim_times);
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reduce(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 24.0);  // 1*2*3*4 = 24
}

TEST_F(OperatorsTest, ReduceSubtract) {
    Value* func = machine->heap->allocate_primitive(&prim_minus);
    Eigen::VectorXd v(4);
    v << 10.0, 3.0, 2.0, 1.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reduce(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);  // APL right-to-left: 10-(3-(2-1)) = 10-2 = 8
}

TEST_F(OperatorsTest, ReduceDivide) {
    Value* func = machine->heap->allocate_primitive(&prim_divide);
    Eigen::VectorXd v(3);
    v << 100.0, 5.0, 2.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reduce(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 40.0);  // APL right-to-left: 100/(5/2) = 100/2.5 = 40
}

TEST_F(OperatorsTest, ReducePower) {
    Value* func = machine->heap->allocate_primitive(&prim_star);
    Eigen::VectorXd v(3);
    v << 2.0, 3.0, 2.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reduce(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 512.0);  // right-to-left: 2^(3^2) = 2^9 = 512
}

TEST_F(OperatorsTest, ReduceSingleElement) {
    Value* func = machine->heap->allocate_primitive(&prim_plus);
    Eigen::VectorXd v(1);
    v << 42.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reduce(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(OperatorsTest, ReduceEmptyVector) {
    Value* func = machine->heap->allocate_primitive(&prim_plus);
    Eigen::VectorXd v(0);
    Value* vec = machine->heap->allocate_vector(v);

    fn_reduce(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // +/⍬ → 0 (identity)
}

TEST_F(OperatorsTest, ReduceFirstAxis) {
    Value* func = machine->heap->allocate_primitive(&prim_plus);
    Eigen::MatrixXd m(3, 4);
    m << 1.0, 2.0, 3.0, 4.0,
         5.0, 6.0, 7.0, 8.0,
         9.0, 10.0, 11.0, 12.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_reduce_first(machine, func, mat);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 4);
    EXPECT_EQ(res_mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 15.0);  // 1+5+9
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 18.0);  // 2+6+10
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 21.0);  // 3+7+11
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 24.0);  // 4+8+12
}

TEST_F(OperatorsTest, ReduceEmptyVectorPlus) {
    Eigen::VectorXd empty_vec(0);
    Value* vec = machine->heap->allocate_vector(empty_vec);
    Value* func = machine->heap->allocate_primitive(&prim_plus);

    fn_reduce(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // +/⍬ → 0
}

TEST_F(OperatorsTest, ReduceEmptyVectorTimes) {
    Eigen::VectorXd empty_vec(0);
    Value* vec = machine->heap->allocate_vector(empty_vec);
    Value* func = machine->heap->allocate_primitive(&prim_times);

    fn_reduce(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // ×/⍬ → 1
}

TEST_F(OperatorsTest, ReduceEmptyDimensionMatrix) {
    Eigen::MatrixXd empty_mat(3, 0);  // 3 rows, 0 columns
    Value* mat = machine->heap->allocate_matrix(empty_mat);
    Value* func = machine->heap->allocate_primitive(&prim_plus);

    fn_reduce(machine, func, mat);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 3);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 0.0);
}

TEST_F(OperatorsTest, ErrorReduceNonFunction) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reduce(machine, vec, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// ========================================================================
// Scan Operator Tests
// ========================================================================

TEST_F(OperatorsTest, ScanVector) {
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    fn_scan(machine, plus_fn, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    // Left-to-right cumulative (ISO-13751): Item I is f/B[⍳I]
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);   // +/1 = 1
    EXPECT_DOUBLE_EQ((*res)(1, 0), 3.0);   // +/1 2 = 3
    EXPECT_DOUBLE_EQ((*res)(2, 0), 6.0);   // +/1 2 3 = 6
    EXPECT_DOUBLE_EQ((*res)(3, 0), 10.0);  // +/1 2 3 4 = 10
}

TEST_F(OperatorsTest, ScanMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    fn_scan(machine, plus_fn, mat);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 2);
    EXPECT_EQ(res->cols(), 3);
    // Left-to-right scan along last axis (ISO-13751)
    // Row 0: [1, 3, 6] = [1, 1+2, 1+2+3]
    // Row 1: [4, 9, 15] = [4, 4+5, 4+5+6]
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 3.0);
    EXPECT_DOUBLE_EQ((*res)(0, 2), 6.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 9.0);
    EXPECT_DOUBLE_EQ((*res)(1, 2), 15.0);
}

TEST_F(OperatorsTest, ScanMultiply) {
    Value* func = machine->heap->allocate_primitive(&prim_times);
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_scan(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 4);
    // Left-to-right scan (ISO-13751): Item I is ×/1 2 ... I
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);   // ×/1 = 1
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 2.0);   // ×/1 2 = 2
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 6.0);   // ×/1 2 3 = 6
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 24.0);  // ×/1 2 3 4 = 24
}

TEST_F(OperatorsTest, ScanSubtract) {
    Value* func = machine->heap->allocate_primitive(&prim_minus);
    Eigen::VectorXd v(5);
    v << 10.0, 1.0, 1.0, 1.0, 1.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_scan(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    // ISO-13751: Item I is -/B[⍳I] where reduce is RIGHT-TO-LEFT
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 10.0);  // -/10 = 10
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 9.0);   // -/10 1 = 10-1 = 9
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 10.0);  // -/10 1 1 = 10-(1-1) = 10
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 9.0);   // -/10 1 1 1 = 10-(1-(1-1)) = 10-1 = 9
    EXPECT_DOUBLE_EQ((*res_mat)(4, 0), 10.0);  // -/10 1 1 1 1 = 10-(1-(1-(1-1))) = 10-0 = 10
}

TEST_F(OperatorsTest, ScanDivide) {
    Value* func = machine->heap->allocate_primitive(&prim_divide);
    Eigen::VectorXd v(4);
    v << 100.0, 2.0, 5.0, 2.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_scan(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 100.0);  // ÷/100
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 50.0);   // ÷/100 2 = 100÷2
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 250.0);  // ÷/100 2 5 = 100÷(2÷5)
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 125.0);  // ÷/100 2 5 2 = 100÷(2÷(5÷2))
}

TEST_F(OperatorsTest, ScanSingleElement) {
    Value* func = machine->heap->allocate_primitive(&prim_plus);
    Eigen::VectorXd v(1);
    v << 99.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_scan(machine, func, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 1);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 99.0);
}

TEST_F(OperatorsTest, ScanFirstAxis) {
    Value* func = machine->heap->allocate_primitive(&prim_plus);
    Eigen::MatrixXd m(3, 2);
    m << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_scan_first(machine, func, mat);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 3);
    EXPECT_EQ(res_mat->cols(), 2);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 9.0);   // 1+(3+5)
    EXPECT_DOUBLE_EQ((*res_mat)(0, 1), 12.0);  // 2+(4+6)
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 8.0);   // 3+5
    EXPECT_DOUBLE_EQ((*res_mat)(1, 1), 10.0);  // 4+6
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 5.0);   // 5
    EXPECT_DOUBLE_EQ((*res_mat)(2, 1), 6.0);   // 6
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
// Composition Tests with Reduce/Scan
// ========================================================================

TEST_F(OperatorsTest, CompositionIotaTakeReduce) {
    Value* n = machine->heap->allocate_scalar(10.0);
    fn_iota(machine, n);

    Value* iota_result = machine->ctrl.value;

    Value* five = machine->heap->allocate_scalar(5.0);
    fn_take(machine, five, iota_result);

    Value* taken = machine->ctrl.value;

    Value* func = machine->heap->allocate_primitive(&prim_plus);
    fn_reduce(machine, func, taken);

    Value* sum = machine->ctrl.value;
    ASSERT_TRUE(sum->is_scalar());
    EXPECT_DOUBLE_EQ(sum->as_scalar(), 10.0);  // 0+1+2+3+4 = 10
}

TEST_F(OperatorsTest, CompositionMultiplyReduce) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    Value* func = machine->heap->allocate_primitive(&prim_times);
    fn_reduce(machine, func, mat);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 2);
    EXPECT_EQ(res_mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 6.0);    // 1*2*3
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 120.0);  // 4*5*6
}

TEST_F(OperatorsTest, CompositionDropScan) {
    Value* n = machine->heap->allocate_scalar(10.0);
    fn_iota(machine, n);

    Value* iota_result = machine->ctrl.value;

    Value* three = machine->heap->allocate_scalar(3.0);
    fn_drop(machine, three, iota_result);

    Value* dropped = machine->ctrl.value;

    Value* func = machine->heap->allocate_primitive(&prim_plus);
    fn_scan(machine, func, dropped);

    Value* scanned = machine->ctrl.value;
    ASSERT_TRUE(scanned->is_vector());
    const Eigen::MatrixXd* res_mat = scanned->as_matrix();
    EXPECT_EQ(res_mat->rows(), 7);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 3.0);   // +/3
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 7.0);   // +/3 4
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 12.0);  // +/3 4 5
    EXPECT_DOUBLE_EQ((*res_mat)(6, 0), 42.0);  // +/3 4 5 6 7 8 9
}

TEST_F(OperatorsTest, CompositionNestedReduce) {
    Eigen::MatrixXd m(3, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0;
    Value* mat = machine->heap->allocate_matrix(m);

    Value* func = machine->heap->allocate_primitive(&prim_plus);
    fn_reduce(machine, func, mat);

    Value* row_sums = machine->ctrl.value;

    fn_reduce(machine, func, row_sums);

    Value* total = machine->ctrl.value;
    ASSERT_TRUE(total->is_scalar());
    EXPECT_DOUBLE_EQ(total->as_scalar(), 45.0);  // 1+2+...+9 = 45
}

// ========================================================================
// Additional Outer Product Tests (ISO-13751 compliance)
// ========================================================================

TEST_F(OperatorsTest, OuterProductTwoScalars) {
    // 5 ∘.+ 3 → 8 (scalar result)
    Value* lhs = machine->heap->allocate_scalar(5.0);
    Value* rhs = machine->heap->allocate_scalar(3.0);
    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    op_outer_product(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_scalar());
    EXPECT_DOUBLE_EQ(machine->ctrl.value->as_scalar(), 8.0);
}

TEST_F(OperatorsTest, OuterProductMatrixVector) {
    // 2×2 matrix ∘.+ vector[2] → 2×2×2 result
    Eigen::MatrixXd lhs_mat(2, 2);
    lhs_mat << 1, 2,
               3, 4;
    Value* lhs = machine->heap->allocate_matrix(lhs_mat);

    Eigen::VectorXd rhs_vec(2);
    rhs_vec << 10, 20;
    Value* rhs = machine->heap->allocate_vector(rhs_vec);

    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    op_outer_product(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_matrix());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 4);
    EXPECT_EQ(result->cols(), 2);
    // First row of matrix + first element of vector
    EXPECT_DOUBLE_EQ((*result)(0, 0), 11.0);  // 1+10
    EXPECT_DOUBLE_EQ((*result)(0, 1), 21.0);  // 1+20
    EXPECT_DOUBLE_EQ((*result)(1, 0), 12.0);  // 2+10
    EXPECT_DOUBLE_EQ((*result)(1, 1), 22.0);  // 2+20
}

TEST_F(OperatorsTest, OuterProductDifferentOperators) {
    // Test with equality operator: 1 2 3 ∘.= 2 3
    Eigen::VectorXd lhs_vec(3);
    lhs_vec << 1, 2, 3;
    Value* lhs = machine->heap->allocate_vector(lhs_vec);

    Eigen::VectorXd rhs_vec(2);
    rhs_vec << 2, 3;
    Value* rhs = machine->heap->allocate_vector(rhs_vec);

    Value* fn = machine->heap->allocate_primitive(&prim_equal);

    op_outer_product(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 2);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 0.0);  // 1=2 → 0
    EXPECT_DOUBLE_EQ((*result)(0, 1), 0.0);  // 1=3 → 0
    EXPECT_DOUBLE_EQ((*result)(1, 0), 1.0);  // 2=2 → 1
    EXPECT_DOUBLE_EQ((*result)(1, 1), 0.0);  // 2=3 → 0
    EXPECT_DOUBLE_EQ((*result)(2, 0), 0.0);  // 3=2 → 0
    EXPECT_DOUBLE_EQ((*result)(2, 1), 1.0);  // 3=3 → 1
}

// ========================================================================
// Additional Inner Product Tests (ISO-13751 compliance)
// ========================================================================

TEST_F(OperatorsTest, InnerProductLargerMatrices) {
    // 3×4 matrix +.× 4×2 matrix → 3×2 matrix
    Eigen::MatrixXd lhs_mat(3, 4);
    lhs_mat << 1, 2, 3, 4,
               5, 6, 7, 8,
               9, 10, 11, 12;
    Value* lhs = machine->heap->allocate_matrix(lhs_mat);

    Eigen::MatrixXd rhs_mat(4, 2);
    rhs_mat << 1, 0,
               0, 1,
               1, 0,
               0, 1;
    Value* rhs = machine->heap->allocate_matrix(rhs_mat);

    Value* f = machine->heap->allocate_primitive(&prim_plus);
    Value* g = machine->heap->allocate_primitive(&prim_times);

    op_inner_product(machine, lhs, f, g, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_matrix());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 2);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 4.0);   // 1*1 + 2*0 + 3*1 + 4*0 = 4
    EXPECT_DOUBLE_EQ((*result)(0, 1), 6.0);   // 1*0 + 2*1 + 3*0 + 4*1 = 6
    EXPECT_DOUBLE_EQ((*result)(1, 0), 12.0);  // 5*1 + 6*0 + 7*1 + 8*0 = 12
    EXPECT_DOUBLE_EQ((*result)(1, 1), 14.0);  // 5*0 + 6*1 + 7*0 + 8*1 = 14
}

TEST_F(OperatorsTest, InnerProductDifferentOperators) {
    // Test max-min product: ⌈.⌊ (not implemented yet, so skip for now)
    // Instead test with subtraction-division
    Eigen::VectorXd lhs_vec(3);
    lhs_vec << 10, 20, 30;
    Value* lhs = machine->heap->allocate_vector(lhs_vec);

    Eigen::VectorXd rhs_vec(3);
    rhs_vec << 2, 4, 5;
    Value* rhs = machine->heap->allocate_vector(rhs_vec);

    Value* f = machine->heap->allocate_primitive(&prim_plus);
    Value* g = machine->heap->allocate_primitive(&prim_divide);

    op_inner_product(machine, lhs, f, g, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_scalar());
    EXPECT_DOUBLE_EQ(machine->ctrl.value->as_scalar(), 16.0);  // 10/2 + 20/4 + 30/5 = 5+5+6 = 16
}

TEST_F(OperatorsTest, InnerProductRectangularMatrices) {
    // 2×3 +.× 3×4 → 2×4
    Eigen::MatrixXd lhs_mat(2, 3);
    lhs_mat << 1, 2, 3,
               4, 5, 6;
    Value* lhs = machine->heap->allocate_matrix(lhs_mat);

    Eigen::MatrixXd rhs_mat(3, 4);
    rhs_mat << 1, 0, 0, 1,
               0, 1, 0, 1,
               0, 0, 1, 1;
    Value* rhs = machine->heap->allocate_matrix(rhs_mat);

    Value* f = machine->heap->allocate_primitive(&prim_plus);
    Value* g = machine->heap->allocate_primitive(&prim_times);

    op_inner_product(machine, lhs, f, g, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_matrix());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 4);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 1.0);  // 1*1 + 2*0 + 3*0
    EXPECT_DOUBLE_EQ((*result)(0, 1), 2.0);  // 1*0 + 2*1 + 3*0
    EXPECT_DOUBLE_EQ((*result)(0, 2), 3.0);  // 1*0 + 2*0 + 3*1
    EXPECT_DOUBLE_EQ((*result)(0, 3), 6.0);  // 1*1 + 2*1 + 3*1
}

// ========================================================================
// Additional Scan Tests (ISO-13751 compliance)
// ========================================================================

TEST_F(OperatorsTest, ScanEmptyVector) {
    // Per spec: scan of empty returns empty
    Eigen::VectorXd v(0);
    Value* vec = machine->heap->allocate_vector(v);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    fn_scan(machine, plus_fn, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 0);
}

TEST_F(OperatorsTest, ScanSingleElementScalar) {
    // Scan of single element returns that element
    Value* scalar = machine->heap->allocate_scalar(42.0);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    fn_scan(machine, plus_fn, scalar);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(OperatorsTest, ScanWithEqual) {
    // Scan with = operator
    // For vector [65, 66]: scan right-to-left gives [65=(66), 66]
    // But = is only defined dyadically, so this doesn't work with scalars
    // Let's use a different test case with 3 elements
    Eigen::VectorXd v(3);
    v << 1.0, 1.0, 1.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* equal_fn = machine->heap->allocate_primitive(&prim_equal);

    fn_scan(machine, equal_fn, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    // Right-to-left: [1=(1=1), 1=1, 1] → [1=1, 1, 1] → [1, 1, 1]
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);
}

// ========================================================================
// Additional Reduce Tests (ISO-13751 compliance)
// ========================================================================

TEST_F(OperatorsTest, ReduceScalar) {
    // Reduction of scalar returns the scalar
    Value* scalar = machine->heap->allocate_scalar(99.0);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    fn_reduce(machine, plus_fn, scalar);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 99.0);
}

TEST_F(OperatorsTest, ReduceWithEqual) {
    // Test with different operator: right-to-left: 5=(5=5) → 5=1 → 0
    Eigen::VectorXd v(3);
    v << 5.0, 5.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* equal_fn = machine->heap->allocate_primitive(&prim_equal);

    fn_reduce(machine, equal_fn, vec);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 5=(5=5) → 5=1 → 0
}

TEST_F(OperatorsTest, ReduceLargerMatrix) {
    // Test 4×5 matrix reduction
    Eigen::MatrixXd m(4, 5);
    m << 1, 2, 3, 4, 5,
         6, 7, 8, 9, 10,
         11, 12, 13, 14, 15,
         16, 17, 18, 19, 20;
    Value* mat = machine->heap->allocate_matrix(m);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    fn_reduce(machine, plus_fn, mat);

    Value* result = machine->ctrl.value;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 4);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 15.0);   // 1+2+3+4+5
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 40.0);   // 6+7+8+9+10
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 65.0);   // 11+12+13+14+15
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 90.0);   // 16+17+18+19+20
}

// ========================================================================
// Additional Duplicate/Commute Tests
// ========================================================================

TEST_F(OperatorsTest, DuplicateWithDivide) {
    // ÷⍨4 → 4÷4 = 1
    Value* omega = machine->heap->allocate_scalar(4.0);
    Value* fn = machine->heap->allocate_primitive(&prim_divide);

    op_commute(machine, fn, omega);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_scalar());
    EXPECT_DOUBLE_EQ(machine->ctrl.value->as_scalar(), 1.0);
}

TEST_F(OperatorsTest, CommuteWithDivide) {
    // 3÷⍨12 → 12÷3 = 4
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Value* rhs = machine->heap->allocate_scalar(12.0);
    Value* fn = machine->heap->allocate_primitive(&prim_divide);

    op_commute_dyadic(machine, lhs, fn, nullptr, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_scalar());
    EXPECT_DOUBLE_EQ(machine->ctrl.value->as_scalar(), 4.0);
}

TEST_F(OperatorsTest, DuplicateMatrix) {
    // Test duplicate with matrix subtraction
    Eigen::MatrixXd mat(2, 2);
    mat << 5, 6,
           7, 8;
    Value* omega = machine->heap->allocate_matrix(mat);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    op_commute(machine, fn, omega);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_matrix());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
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
// ISO-13751 Compliance Tests
// ========================================================================

TEST_F(OperatorsTest, ScanISOExample) {
    // ISO-13751 example: +\1 1 1 → 1 2 3 (LEFT-TO-RIGHT cumulative)
    // Item I is f/B[⍳I]:
    // - Item 1: +/1 = 1
    // - Item 2: +/1 1 = 2
    // - Item 3: +/1 1 1 = 3
    Eigen::VectorXd vec(3);
    vec << 1, 1, 1;
    Value* omega = machine->heap->allocate_vector(vec);
    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    fn_scan(machine, fn, omega);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_vector());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_EQ(result->rows(), 3);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 1.0);  // +/1
    EXPECT_DOUBLE_EQ((*result)(1, 0), 2.0);  // +/1 1
    EXPECT_DOUBLE_EQ((*result)(2, 0), 3.0);  // +/1 1 1
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
