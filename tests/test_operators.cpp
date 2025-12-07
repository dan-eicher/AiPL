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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
