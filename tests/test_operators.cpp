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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
