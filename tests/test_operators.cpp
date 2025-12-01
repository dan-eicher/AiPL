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

    op_commute_dyadic(machine, lhs, fn, rhs);

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

    op_commute_dyadic(machine, lhs, fn, rhs);

    ASSERT_NE(machine->ctrl.value, nullptr);
    EXPECT_TRUE(machine->ctrl.value->is_vector());

    const Eigen::MatrixXd* result = machine->ctrl.value->as_matrix();
    EXPECT_DOUBLE_EQ((*result)(0, 0), -9.0);   // 1-10
    EXPECT_DOUBLE_EQ((*result)(1, 0), -18.0);  // 2-20
    EXPECT_DOUBLE_EQ((*result)(2, 0), -27.0);  // 3-30
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
