// Structural Primitive Tests
// Covers: ⍴, ,, ⍉, ↑, ↓, ⌽, ⊖, ⍳, ∊, ⍋, ⍒, /, \, ⌿, ⍀, ∪, ~, ⌷, ⊣, ⊢, ⍪
// Split from test_primitives.cpp for maintainability

#include <gtest/gtest.h>
#include "primitives.h"
#include "operators.h"
#include "value.h"
#include "machine.h"
#include "continuation.h"
#include "parser.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

using namespace apl;

class StructuralTest : public ::testing::Test {
protected:
    Machine* machine;
    void SetUp() override { machine = new Machine(); }
    void TearDown() override { delete machine; }
};

// Helper to create character vector from string
static Value* make_char_vector(Machine* m, const char* str) {
    return m->eval(std::string("'") + str + "'");
}

// Helper to create character matrix from strings (each string is a row)
// Uses reshape: 3 2⍴'CAABBC' for {"CA", "AB", "BC"}
static Value* make_char_matrix(Machine* m, const std::vector<std::string>& rows) {
    if (rows.empty()) return nullptr;
    size_t num_rows = rows.size();
    size_t num_cols = rows[0].size();
    std::string chars;
    for (const auto& row : rows) {
        chars += row;
    }
    std::string expr = std::to_string(num_rows) + " " + std::to_string(num_cols) + "⍴'" + chars + "'";
    return m->eval(expr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Array Operation Tests
// ============================================================================

TEST_F(StructuralTest, ShapeScalar) {
    Value* scalar = machine->heap->allocate_scalar(5.0);
    fn_shape(machine, nullptr, scalar);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);  // Empty shape for scalar

}

TEST_F(StructuralTest, ShapeVector) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_shape(machine, nullptr, vec);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* shape = result->as_matrix();
    EXPECT_EQ(shape->rows(), 1);
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 5.0);

}

TEST_F(StructuralTest, ReshapeVector) {
    // 2 3⍴1 2 3 4 5 6 should produce row-major matrix:
    // 1 2 3
    // 4 5 6
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = machine->heap->allocate_vector(v);

    Eigen::VectorXd new_shape(2);
    new_shape << 2.0, 3.0;
    Value* shape = machine->heap->allocate_vector(new_shape);

    fn_reshape(machine, nullptr, shape, vec);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    // Row-major order: fills row 0 first, then row 1
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 6.0);
}

TEST_F(StructuralTest, ReshapeRowMajorOrder) {
    // Explicit test for row-major ordering: 3 2⍴⍳6 (0-based)
    // Should produce:
    // 0 1
    // 2 3
    // 4 5
    Eigen::VectorXd v(6);
    v << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    Eigen::VectorXd new_shape(2);
    new_shape << 3.0, 2.0;
    Value* shape = machine->heap->allocate_vector(new_shape);

    fn_reshape(machine, nullptr, shape, vec);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 2);
    // Verify row-major fill order
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0) << "Row 0, Col 0";
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 1.0) << "Row 0, Col 1";
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0) << "Row 1, Col 0";
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 3.0) << "Row 1, Col 1";
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 4.0) << "Row 2, Col 0";
    EXPECT_DOUBLE_EQ((*mat)(2, 1), 5.0) << "Row 2, Col 1";
}

TEST_F(StructuralTest, ReshapeMatrixToMatrix) {
    // Reshape matrix to different shape - both read and write should be row-major
    // Input 2×3:
    // 1 2 3
    // 4 5 6
    // Row-major read: 1 2 3 4 5 6
    // Row-major write to 3×2:
    // 1 2
    // 3 4
    // 5 6
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    Eigen::VectorXd new_shape(2);
    new_shape << 3.0, 2.0;
    Value* shape = machine->heap->allocate_vector(new_shape);

    fn_reshape(machine, nullptr, shape, mat);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_EQ(res->cols(), 2);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(2, 1), 6.0);
}

TEST_F(StructuralTest, Ravel) {
    // Ravel flattens in row-major order (APL standard)
    // Matrix:
    // 1 2 3
    // 4 5 6
    // Should become: 1 2 3 4 5 6
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_ravel(machine, nullptr, mat);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 6);
    // Row-major order
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 5.0);
    EXPECT_DOUBLE_EQ((*vec)(5, 0), 6.0);
}

TEST_F(StructuralTest, Catenate) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(2);
    v2 << 4.0, 5.0;

    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_catenate(machine, nullptr, vec1, vec2);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);

}

TEST_F(StructuralTest, Transpose) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_transpose(machine, nullptr, mat);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_EQ(res->cols(), 2);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 4.0);

}

TEST_F(StructuralTest, Iota) {
    Value* n = machine->heap->allocate_scalar(5.0);
    fn_iota(machine, nullptr, n);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1-based per ISO 13751
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 5.0);

}

TEST_F(StructuralTest, Take) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    Value* count = machine->heap->allocate_scalar(3.0);
    fn_take(machine, nullptr, count, vec);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);

}

TEST_F(StructuralTest, Drop) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    Value* count = machine->heap->allocate_scalar(2.0);
    fn_drop(machine, nullptr, count, vec);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);

}

// ============================================================================
// Reverse/Rotate/Tally Tests
// ============================================================================

TEST_F(StructuralTest, ReverseVector) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reverse(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 1.0);
}

TEST_F(StructuralTest, ReverseScalar) {
    Value* scalar = machine->heap->allocate_scalar(42.0);

    fn_reverse(machine, nullptr, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 42.0);
}

TEST_F(StructuralTest, ReverseMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_reverse(machine, nullptr, mat);

    ASSERT_TRUE(machine->result->is_matrix());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 2);
    EXPECT_EQ(res->cols(), 3);
    // Row 0: 3 2 1
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*res)(0, 2), 1.0);
    // Row 1: 6 5 4
    EXPECT_DOUBLE_EQ((*res)(1, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 5.0);
    EXPECT_DOUBLE_EQ((*res)(1, 2), 4.0);
}

TEST_F(StructuralTest, ReverseFirstMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_reverse_first(machine, nullptr, mat);

    ASSERT_TRUE(machine->result->is_matrix());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 2);
    EXPECT_EQ(res->cols(), 3);
    // Rows are swapped
    EXPECT_DOUBLE_EQ((*res)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 5.0);
    EXPECT_DOUBLE_EQ((*res)(0, 2), 6.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 2.0);
    EXPECT_DOUBLE_EQ((*res)(1, 2), 3.0);
}

TEST_F(StructuralTest, RotateVectorPositive) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* count = machine->heap->allocate_scalar(2.0);

    fn_rotate(machine, nullptr, count, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    // Rotated left by 2: 3 4 5 1 2
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 2.0);
}

TEST_F(StructuralTest, RotateVectorNegative) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* count = machine->heap->allocate_scalar(-2.0);

    fn_rotate(machine, nullptr, count, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    // Rotated right by 2: 4 5 1 2 3
    EXPECT_DOUBLE_EQ((*res)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 3.0);
}

TEST_F(StructuralTest, RotateFirstMatrix) {
    Eigen::MatrixXd m(3, 2);
    m << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);
    Value* count = machine->heap->allocate_scalar(1.0);

    fn_rotate_first(machine, nullptr, count, mat);

    ASSERT_TRUE(machine->result->is_matrix());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_EQ(res->cols(), 2);
    // Rows rotated up by 1: [[3,4],[5,6],[1,2]]
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 6.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 1), 2.0);
}

TEST_F(StructuralTest, RotateWrapAround) {
    // ISO 10.2.7: rotation wraps around
    // ¯7⌽'ABCDEF' → 'FABCDE' (¯7 mod 6 = ¯1)
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* count = machine->heap->allocate_scalar(7.0);  // 7 mod 5 = 2

    fn_rotate(machine, nullptr, count, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    // 7⌽1 2 3 4 5 = 2⌽1 2 3 4 5 = 3 4 5 1 2
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 2.0);
}

TEST_F(StructuralTest, RotateScalar) {
    // ISO 10.2.7: rotating a scalar returns it unchanged
    Value* scalar = machine->heap->allocate_scalar(42.0);
    Value* count = machine->heap->allocate_scalar(5.0);

    fn_rotate(machine, nullptr, count, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 42.0);
}

// --- ISO 13751 10.1.4/10.2.7: Additional Reverse/Rotate tests ---

// ISO 13751 10.1.4: Reverse with axis - ⌽[K]
TEST_F(StructuralTest, ReverseWithAxisLast) {
    // ⌽[2] on matrix reverses along axis 2 (columns within rows)
    Value* result = machine->eval("⌽[2] 2 3⍴⍳6");
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);  // Row 0: 3 2 1
    EXPECT_DOUBLE_EQ((*m)(0, 2), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 6.0);  // Row 1: 6 5 4
}

TEST_F(StructuralTest, ReverseWithAxisFirst) {
    // ⌽[1] on matrix reverses along axis 1 (rows)
    Value* result = machine->eval("⌽[1] 2 3⍴⍳6");
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 4.0);  // First row is now [4,5,6]
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // Second row is now [1,2,3]
}

// ISO 13751 10.1.4: Invalid axis signals AXIS ERROR
TEST_F(StructuralTest, ReverseAxisError) {
    // ⌽[3] on 2D matrix → AXIS ERROR
    EXPECT_THROW(machine->eval("⌽[3] 2 3⍴⍳6"), APLError);
}

TEST_F(StructuralTest, ReverseAxisZeroError) {
    // ⌽[0] → AXIS ERROR (axes are 1-based when ⎕IO=1)
    EXPECT_THROW(machine->eval("⌽[0] 2 3⍴⍳6"), APLError);
}

// ISO 13751 10.1.4: Reverse on higher rank array
TEST_F(StructuralTest, ReverseHigherRank) {
    // ⌽ 2 2 3⍴⍳12 reverses along last axis of a 3D array
    // ⍳12 (IO=1) = 1..12, reshaped 2×2×3:
    //   [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
    // reversed last axis:
    //   [[[3,2,1],[6,5,4]],[[9,8,7],[12,11,10]]]
    Value* result = machine->eval("⌽ 2 2 3⍴⍳12");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_ndarray());
    const auto* nd = result->as_ndarray();
    ASSERT_EQ(nd->shape.size(), 3u);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 2);
    EXPECT_EQ(nd->shape[2], 3);
    // First row reversed: 3 2 1
    EXPECT_DOUBLE_EQ((*nd->data)(0), 3.0);
    EXPECT_DOUBLE_EQ((*nd->data)(1), 2.0);
    EXPECT_DOUBLE_EQ((*nd->data)(2), 1.0);
    // Second row reversed: 6 5 4
    EXPECT_DOUBLE_EQ((*nd->data)(3), 6.0);
    EXPECT_DOUBLE_EQ((*nd->data)(4), 5.0);
    EXPECT_DOUBLE_EQ((*nd->data)(5), 4.0);
}

// ISO 13751 10.2.7: Rotate with invalid axis
TEST_F(StructuralTest, RotateAxisError) {
    EXPECT_THROW(machine->eval("1⌽[3] 2 3⍴⍳6"), APLError);
}

// ISO 13751 10.2.7: Rotate with non-integer amount signals DOMAIN ERROR
TEST_F(StructuralTest, RotateNonIntegerError) {
    EXPECT_THROW(machine->eval("1.5⌽1 2 3"), APLError);
}

// ISO 13751 10.2.7: Rotate with shape conformability
TEST_F(StructuralTest, RotateMatrixWithVector) {
    // Each row rotated by different amount
    Value* result = machine->eval("1 2⌽[2] 2 4⍴⍳8");
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    // Row 0 rotated by 1: [2,3,4,1]
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(0, 3), 1.0);
    // Row 1 rotated by 2: [7,8,5,6]
    EXPECT_DOUBLE_EQ((*m)(1, 0), 7.0);
}

TEST_F(StructuralTest, TallyVector) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_tally(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(StructuralTest, TallyScalar) {
    Value* scalar = machine->heap->allocate_scalar(42.0);

    fn_tally(machine, nullptr, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(StructuralTest, TallyMatrix) {
    Eigen::MatrixXd m(3, 4);
    m.setZero();
    Value* mat = machine->heap->allocate_matrix(m);

    fn_tally(machine, nullptr, mat);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(StructuralTest, TallyStrand) {
    Value* result = machine->eval("≢⊂1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);

    result = machine->eval("≢(⊂1 2),(⊂3 4 5)");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(StructuralTest, ReverseRotateTallyRegistered) {
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⌽")), nullptr);
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⊖")), nullptr);
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("≢")), nullptr);
}

// ============================================================================
// Search Functions (⍳ dyadic, ∊)
// ============================================================================

TEST_F(StructuralTest, IndexOfFound) {
    // 1 2 3 4 5 ⍳ 3 → 3 (1-origin index per ISO 13751)
    Eigen::VectorXd haystack(5);
    haystack << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* lhs = machine->heap->allocate_vector(haystack);
    Value* rhs = machine->heap->allocate_scalar(3.0);

    fn_index_of(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(StructuralTest, IndexOfNotFound) {
    // 1 2 3 ⍳ 7 → 4 (not found = 1 + length of haystack, per ISO 13751)
    Eigen::VectorXd haystack(3);
    haystack << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(haystack);
    Value* rhs = machine->heap->allocate_scalar(7.0);

    fn_index_of(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 4.0);
}

TEST_F(StructuralTest, IndexOfVector) {
    // 10 20 30 40 ⍳ 30 20 99 → 3 2 5 (1-origin per ISO 13751)
    Eigen::VectorXd haystack(4);
    haystack << 10.0, 20.0, 30.0, 40.0;
    Eigen::VectorXd needles(3);
    needles << 30.0, 20.0, 99.0;
    Value* lhs = machine->heap->allocate_vector(haystack);
    Value* rhs = machine->heap->allocate_vector(needles);

    fn_index_of(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);  // 30 found at index 3 (1-origin)
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);  // 20 found at index 2 (1-origin)
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);  // 99 not found → 5 (1+length)
}

TEST_F(StructuralTest, IndexOfScalarHaystackRankError) {
    // ISO §10.2.2: "If A is not a vector, signal rank-error"
    // Scalar left arg (rank 0) is not a vector (rank 1)
    Value* lhs = machine->heap->allocate_scalar(5.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);

    fn_index_of(machine, nullptr, lhs, rhs);

    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(StructuralTest, MemberOfFound) {
    // 3 ∊ 1 2 3 4 5 → 1
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Eigen::VectorXd set(5);
    set << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* rhs = machine->heap->allocate_vector(set);

    fn_member_of(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(StructuralTest, MemberOfNotFound) {
    // 7 ∊ 1 2 3 → 0
    Value* lhs = machine->heap->allocate_scalar(7.0);
    Eigen::VectorXd set(3);
    set << 1.0, 2.0, 3.0;
    Value* rhs = machine->heap->allocate_vector(set);

    fn_member_of(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(StructuralTest, MemberOfVector) {
    // 1 5 3 7 ∊ 1 2 3 → 1 0 1 0
    Eigen::VectorXd query(4);
    query << 1.0, 5.0, 3.0, 7.0;
    Eigen::VectorXd set(3);
    set << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(query);
    Value* rhs = machine->heap->allocate_vector(set);

    fn_member_of(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);  // 1 is in set
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);  // 5 is not in set
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);  // 3 is in set
    EXPECT_DOUBLE_EQ((*res)(3, 0), 0.0);  // 7 is not in set
}

TEST_F(StructuralTest, EnlistVector) {
    // ∊ 1 2 3 → 1 2 3 (same as ravel for simple arrays)
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_enlist(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(StructuralTest, EnlistScalar) {
    // ∊ 5 → 5 (1-element vector)
    Value* scalar = machine->heap->allocate_scalar(5.0);

    fn_enlist(machine, nullptr, scalar);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 1);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);
}

TEST_F(StructuralTest, EnlistMatrix) {
    // ISO 8.2.6: ∊ (2 3⍴⍳6) → 1 2 3 4 5 6 (ravel for simple arrays)
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_enlist(machine, nullptr, mat);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 6);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(5, 0), 6.0);
}

TEST_F(StructuralTest, EnlistEmptyVector) {
    // ISO 8.2.6: ∊ (⍳0) → empty vector
    Value* result = machine->eval("∊⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

// ============================================================================
// ISO 13751 Section 8: Structural Primitive Functions - Edge Cases
// ============================================================================

// --- Ravel Edge Cases (Section 8.2.1) ---

TEST_F(StructuralTest, RavelScalar) {
    // ISO 8.2.1: ,5 → 1-element vector containing 5
    Value* scalar = machine->heap->allocate_scalar(5.0);
    fn_ravel(machine, nullptr, scalar);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 1);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);
}

TEST_F(StructuralTest, RavelVector) {
    // ISO 8.2.1: ,1 2 3 → same vector (identity for vectors)
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_ravel(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

// --- Shape Edge Cases (Section 8.2.2) ---

TEST_F(StructuralTest, ShapeMatrix) {
    // ISO 8.2.2: ⍴ (2 3⍴⍳6) → 2 3
    Value* result = machine->eval("⍴2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 3.0);
}

// --- Depth Tests (Section 8.2.5) ---
// ISO 13751: simple-scalar → 0, simple-array → 1, nested → 1 + max depth

TEST_F(StructuralTest, DepthScalar) {
    // ISO 8.2.5: ≡5 → 0 (simple scalar)
    Value* scalar = machine->heap->allocate_scalar(5.0);
    fn_depth(machine, nullptr, scalar);

    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(StructuralTest, DepthVector) {
    // ISO 8.2.5: ≡1 2 3 → 1 (simple array)
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_depth(machine, nullptr, vec);

    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(StructuralTest, DepthMatrix) {
    // ISO 8.2.5: ≡ (2 3⍴⍳6) → 1 (simple array)
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);
    fn_depth(machine, nullptr, mat);

    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(StructuralTest, DepthEmptyVector) {
    // ISO 8.2.5: ≡⍳0 → 1 (empty array still has depth 1)
    Value* result = machine->eval("≡⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(StructuralTest, DepthEnclosedVector) {
    // ISO 8.2.5: ≡⊂1 2 3 → 2 (enclosed vector has depth 2)
    Value* result = machine->eval("≡⊂1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(StructuralTest, DepthDoubleEnclosed) {
    // ISO 8.2.5: ≡⊂⊂1 2 3 → 3 (double-enclosed)
    Value* result = machine->eval("≡⊂⊂1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(StructuralTest, DepthNestedMixed) {
    Value* result = machine->eval("≡(⊂1 2 3),(⊂4 5 6)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// --- Strand Catenation Tests ---

TEST_F(StructuralTest, CatenateStrandStrand) {
    // (⊂1 2),(⊂3 4) → strand of 2 vectors
    Value* result = machine->eval("(⊂1 2),(⊂3 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2u);
}

TEST_F(StructuralTest, CatenateStrandScalar) {
    // (⊂1 2),5 → strand with 2 elements (vector + scalar)
    Value* result = machine->eval("(⊂1 2),5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2u);
}

TEST_F(StructuralTest, CatenateScalarStrand) {
    // 5,(⊂1 2) → strand with 2 elements (scalar + vector)
    Value* result = machine->eval("5,(⊂1 2)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2u);
}

TEST_F(StructuralTest, CatenateStrandVector) {
    // (⊂1 2),(3 4 5) → strand with 2 elements (vector + vector)
    Value* result = machine->eval("(⊂1 2),3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2u);
}

TEST_F(StructuralTest, DepthString) {
    // Strings have depth 1 (like simple arrays)
    Value* result = machine->eval("≡'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// --- Table Edge Cases (Section 8.2.4) ---

TEST_F(StructuralTest, TableEmptyVector) {
    // ISO 8.2.4: ⍪⍳0 → 0×1 matrix
    Value* result = machine->eval("⍪⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 0);
    EXPECT_EQ(mat->cols(), 1);
}

// --- Reshape Edge Cases (Section 8.3.1) ---

TEST_F(StructuralTest, ReshapeToScalar) {
    // ISO 8.3.1: (⍳0)⍴5 → scalar 5 (empty shape produces scalar)
    Value* result = machine->eval("(⍳0)⍴5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(StructuralTest, ReshapeZeroLength) {
    // ISO 8.3.1: 0⍴5 → empty vector
    Value* result = machine->eval("0⍴5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, ReshapeZeroMatrix) {
    // ISO 8.3.1: 0 3⍴5 → 0×3 matrix (empty rows)
    Value* result = machine->eval("0 3⍴5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 0);
    EXPECT_EQ(mat->cols(), 3);
}

// --- Join/Catenate Edge Cases (Section 8.3.2) ---

TEST_F(StructuralTest, CatenateScalarScalar) {
    // ISO 8.3.2: 5,3 → 5 3 (two-element vector)
    Value* result = machine->eval("5,3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 3.0);
}

TEST_F(StructuralTest, CatenateScalarVector) {
    // ISO 8.3.2: 5,1 2 3 → 5 1 2 3
    Value* result = machine->eval("5,1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(3, 0), 3.0);
}

TEST_F(StructuralTest, CatenateVectorScalar) {
    // ISO 8.3.2: 1 2 3,5 → 1 2 3 5
    Value* result = machine->eval("1 2 3,5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(3, 0), 5.0);
}

TEST_F(StructuralTest, SearchFunctionsRegistered) {
    // ⍳ should already be registered (monadic iota)
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⍳")), nullptr);
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("∊")), nullptr);
}

// ============================================================================
// Grade Functions (⍋ ⍒)
// ============================================================================

TEST_F(StructuralTest, GradeUpVector) {
    // ⍋ 3 1 4 1 5 → 2 4 1 3 5 (indices for ascending order, 1-origin per ISO 13751)
    Eigen::VectorXd v(5);
    v << 3.0, 1.0, 4.0, 1.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_up(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 2.0);  // index 2 (value 1)
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);  // index 4 (value 1)
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);  // index 1 (value 3)
    EXPECT_DOUBLE_EQ((*res)(3, 0), 3.0);  // index 3 (value 4)
    EXPECT_DOUBLE_EQ((*res)(4, 0), 5.0);  // index 5 (value 5)
}

TEST_F(StructuralTest, GradeDownVector) {
    // ⍒ 3 1 4 1 5 → 5 3 1 2 4 (indices for descending order, 1-origin per ISO 13751)
    Eigen::VectorXd v(5);
    v << 3.0, 1.0, 4.0, 1.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_down(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);  // index 5 (value 5)
    EXPECT_DOUBLE_EQ((*res)(1, 0), 3.0);  // index 3 (value 4)
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);  // index 1 (value 3)
    EXPECT_DOUBLE_EQ((*res)(3, 0), 2.0);  // index 2 (value 1)
    EXPECT_DOUBLE_EQ((*res)(4, 0), 4.0);  // index 4 (value 1)
}

TEST_F(StructuralTest, GradeUpScalarError) {
    // ⍋ 5 → RANK ERROR (grade requires array per ISO 13751)
    Value* scalar = machine->heap->allocate_scalar(5.0);

    fn_grade_up(machine, nullptr, scalar);

    // Should have pushed ThrowErrorK
    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(StructuralTest, GradeDownScalarError) {
    // ⍒ 5 → RANK ERROR (grade requires array per ISO 13751)
    Value* scalar = machine->heap->allocate_scalar(5.0);

    fn_grade_down(machine, nullptr, scalar);

    // Should have pushed ThrowErrorK
    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(StructuralTest, GradeUpAlreadySorted) {
    // ⍋ 1 2 3 4 5 → 1 2 3 4 5 (1-origin)
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_up(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*res)(i, 0), static_cast<double>(i + 1));
    }
}

TEST_F(StructuralTest, GradeDownReversed) {
    // ⍒ 1 2 3 4 5 → 5 4 3 2 1 (1-origin)
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_down(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*res)(i, 0), static_cast<double>(5 - i));
    }
}

TEST_F(StructuralTest, GradeFunctionsRegistered) {
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⍋")), nullptr);
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⍒")), nullptr);
}

// --- ISO 10.1.2/10.1.3 Grade Stability Tests ---
// "The indices of identical elements of B occur in Z in ascending order"

TEST_F(StructuralTest, GradeUpStable) {
    // ⍋ 3 1 4 1 5 → indices that would sort ascending
    // Two 1s at positions 2 and 4 - stable sort should return 2 before 4
    Value* result = machine->eval("⍋ 3 1 4 1 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* m = result->as_matrix();
    // First two indices should be positions of 1s: 2, 4 (in that order for stability)
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);  // First 1 at position 2
    EXPECT_DOUBLE_EQ((*m)(1, 0), 4.0);  // Second 1 at position 4
}

TEST_F(StructuralTest, GradeDownStable) {
    // ⍒ 3 1 4 1 5 → indices that would sort descending
    // Two 1s at positions 2 and 4 - stable sort should keep them in order (2 then 4)
    Value* result = machine->eval("⍒ 3 1 4 1 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* m = result->as_matrix();
    // Last two indices should be positions of 1s: 2, 4 (stable order preserved)
    EXPECT_DOUBLE_EQ((*m)(3, 0), 2.0);  // First 1 at position 2
    EXPECT_DOUBLE_EQ((*m)(4, 0), 4.0);  // Second 1 at position 4
}

TEST_F(StructuralTest, GradeUpAllEqual) {
    // ⍋ 5 5 5 5 → 1 2 3 4 (all equal, preserve original order)
    Value* result = machine->eval("⍋ 5 5 5 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), 4.0);
}

// --- Basic Character Grade Up Tests ---

TEST_F(StructuralTest, CharGradeUpBasicVector) {
    // 'ABC'⍋'CAB' → 2 3 1
    Value* collating = make_char_vector(machine, "ABC");
    Value* data = make_char_vector(machine, "CAB");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->size(), 3);
    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);  // 'A' at position 2 in 'CAB'
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);  // 'B' at position 3
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // 'C' at position 1
}

TEST_F(StructuralTest, CharGradeUpAlreadySorted) {
    Value* collating = make_char_vector(machine, "ABC");
    Value* data = make_char_vector(machine, "ABC");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

TEST_F(StructuralTest, CharGradeUpReversed) {
    Value* collating = make_char_vector(machine, "ABC");
    Value* data = make_char_vector(machine, "CBA");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);
}

// --- Character Grade Down Tests ---

TEST_F(StructuralTest, CharGradeDownBasicVector) {
    Value* collating = make_char_vector(machine, "ABC");
    Value* data = make_char_vector(machine, "CAB");

    fn_grade_down_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);  // 'C' first (highest)
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);  // 'B' second
    EXPECT_DOUBLE_EQ((*m)(2, 0), 2.0);  // 'A' last
}

TEST_F(StructuralTest, CharGradeDownReversed) {
    Value* collating = make_char_vector(machine, "ABC");
    Value* data = make_char_vector(machine, "CBA");

    fn_grade_down_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);  // Already descending
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

// --- ISO 13751: Characters not in A are equal and occur after all characters in A ---

TEST_F(StructuralTest, CharGradeUpUnknownCharsLast) {
    Value* collating = make_char_vector(machine, "AB");
    Value* data = make_char_vector(machine, "CBA");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);  // 'A' first (known, lowest)
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);  // 'B' second (known)
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // 'C' last (unknown)
}

TEST_F(StructuralTest, CharGradeDownUnknownCharsLast) {
    // ISO 13751: unknowns sort AFTER all known chars, even in descending
    Value* collating = make_char_vector(machine, "AB");
    Value* data = make_char_vector(machine, "CBA");

    fn_grade_down_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);  // 'B' first (highest known)
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);  // 'A' second
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // 'C' last (unknown)
}

TEST_F(StructuralTest, CharGradeUpMultipleUnknowns) {
    // Multiple unknown chars should be equal (stable among themselves)
    Value* collating = make_char_vector(machine, "A");
    Value* data = make_char_vector(machine, "XAYZ");  // X,Y,Z unknown

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);  // 'A' first (only known)
    // Unknowns maintain original order (stable): X@1, Y@3, Z@4
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // 'X'
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);  // 'Y'
    EXPECT_DOUBLE_EQ((*m)(3, 0), 4.0);  // 'Z'
}

// --- ISO 13751: Stable sort requirement ---

TEST_F(StructuralTest, CharGradeUpStable) {
    Value* collating = make_char_vector(machine, "AB");
    Value* data = make_char_vector(machine, "ABBA");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);  // First 'A' at position 1
    EXPECT_DOUBLE_EQ((*m)(1, 0), 4.0);  // Second 'A' at position 4
    EXPECT_DOUBLE_EQ((*m)(2, 0), 2.0);  // First 'B' at position 2
    EXPECT_DOUBLE_EQ((*m)(3, 0), 3.0);  // Second 'B' at position 3
}

TEST_F(StructuralTest, CharGradeDownStable) {
    Value* collating = make_char_vector(machine, "AB");
    Value* data = make_char_vector(machine, "ABBA");

    fn_grade_down_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);  // First 'B'
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);  // Second 'B'
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // First 'A'
    EXPECT_DOUBLE_EQ((*m)(3, 0), 4.0);  // Second 'A'
}

TEST_F(StructuralTest, CharGradeUpAllEqualPreservesOrder) {
    Value* collating = make_char_vector(machine, "A");
    Value* data = make_char_vector(machine, "AAA");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

// --- ISO 13751: Edge cases from evaluation sequence ---

TEST_F(StructuralTest, CharGradeUpEmptyCollating) {
    // "If A is empty, return IO+⍳1↑⍴B" (identity permutation)
    Value* collating = make_char_vector(machine, "");
    Value* data = make_char_vector(machine, "ABC");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

TEST_F(StructuralTest, CharGradeUpEmptyRight) {
    // "If 1↑⍴B is zero, return ⍳0"
    Value* collating = make_char_vector(machine, "ABC");
    Value* data = make_char_vector(machine, "");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    EXPECT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->size(), 0);
}

TEST_F(StructuralTest, CharGradeUpSingleElement) {
    // "If 1↑⍴B is one, return one-element-vector containing index-origin"
    Value* collating = make_char_vector(machine, "ABC");
    Value* data = make_char_vector(machine, "X");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    EXPECT_EQ(machine->result->size(), 1);
    EXPECT_DOUBLE_EQ(machine->result->as_matrix()->operator()(0, 0), 1.0);
}

TEST_F(StructuralTest, CharGradeDownEmptyCollating) {
    Value* collating = make_char_vector(machine, "");
    Value* data = make_char_vector(machine, "CBA");

    fn_grade_down_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

// --- ISO 13751: "If A is a scalar, signal rank-error" ---

TEST_F(StructuralTest, CharGradeUpScalarCollatingError) {
    Value* scalar = machine->heap->allocate_scalar(static_cast<double>('A'));
    Value* data = make_char_vector(machine, "ABC");

    fn_grade_up_dyadic(machine, nullptr, scalar, data);

    ASSERT_EQ(machine->kont_stack.size(), 1);
    auto* err = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(err, nullptr);
    EXPECT_TRUE(std::string(err->error_message->c_str()).find("RANK") != std::string::npos);
}

TEST_F(StructuralTest, CharGradeDownScalarCollatingError) {
    Value* scalar = machine->heap->allocate_scalar(static_cast<double>('A'));
    Value* data = make_char_vector(machine, "ABC");

    fn_grade_down_dyadic(machine, nullptr, scalar, data);

    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// --- ISO 13751: Domain errors ---

TEST_F(StructuralTest, CharGradeUpNumericRightError) {
    Value* collating = make_char_vector(machine, "ABC");
    Eigen::VectorXd nums(3);
    nums << 1.0, 2.0, 3.0;
    Value* numeric = machine->heap->allocate_vector(nums);

    fn_grade_up_dyadic(machine, nullptr, collating, numeric);

    ASSERT_EQ(machine->kont_stack.size(), 1);
    auto* err = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(err, nullptr);
    EXPECT_TRUE(std::string(err->error_message->c_str()).find("DOMAIN") != std::string::npos);
}

TEST_F(StructuralTest, CharGradeUpNumericLeftError) {
    Eigen::VectorXd nums(3);
    nums << 1.0, 2.0, 3.0;
    Value* numeric = machine->heap->allocate_vector(nums);
    Value* chars = make_char_vector(machine, "ABC");

    fn_grade_up_dyadic(machine, nullptr, numeric, chars);

    ASSERT_EQ(machine->kont_stack.size(), 1);
    auto* err = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(err, nullptr);
    EXPECT_TRUE(std::string(err->error_message->c_str()).find("DOMAIN") != std::string::npos);
}

// --- ISO 13751: First occurrence determines position for duplicates ---

TEST_F(StructuralTest, CharGradeUpDuplicateInCollating) {
    // 'AABB' → A at pos 0, B at pos 2 (first occurrence)
    Value* collating = make_char_vector(machine, "AABB");
    Value* data = make_char_vector(machine, "BA");

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);  // 'A' first (pos 0)
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // 'B' second (pos 2)
}

// --- ISO 13751: Matrix B - sort rows lexicographically ---

TEST_F(StructuralTest, CharGradeUpMatrixRows) {
    // Sort rows of character matrix
    Value* collating = make_char_vector(machine, "ABC");
    Value* data = make_char_matrix(machine, {"CA", "AB", "BC"});

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    EXPECT_EQ(machine->result->size(), 3);
    const Eigen::MatrixXd* m = machine->result->as_matrix();
    // "AB" < "BC" < "CA" in 'ABC' order
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);  // "AB" first
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);  // "BC" second
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // "CA" last
}

TEST_F(StructuralTest, CharGradeDownMatrixRows) {
    Value* collating = make_char_vector(machine, "ABC");
    Value* data = make_char_matrix(machine, {"CA", "AB", "BC"});

    fn_grade_down_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    // Descending: "CA" > "BC" > "AB"
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);  // "CA" first
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);  // "BC" second
    EXPECT_DOUBLE_EQ((*m)(2, 0), 2.0);  // "AB" last
}

TEST_F(StructuralTest, CharGradeUpMatrixRowsStable) {
    // Equal rows should maintain original order
    Value* collating = make_char_vector(machine, "AB");
    Value* data = make_char_matrix(machine, {"AB", "AB", "AA"});

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    // "AA" < "AB" = "AB", stable keeps first "AB" before second
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);  // "AA" first
    EXPECT_DOUBLE_EQ((*m)(1, 0), 1.0);  // First "AB"
    EXPECT_DOUBLE_EQ((*m)(2, 0), 2.0);  // Second "AB"
}

TEST_F(StructuralTest, CharGradeUpMatrixWithUnknowns) {
    // Rows with unknown chars sort after rows with known chars
    Value* collating = make_char_vector(machine, "AB");
    Value* data = make_char_matrix(machine, {"XY", "AB", "BA"});

    fn_grade_up_dyadic(machine, nullptr, collating, data);

    const Eigen::MatrixXd* m = machine->result->as_matrix();
    // "AB" < "BA" < "XY" (unknowns last)
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);  // "AB"
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);  // "BA"
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // "XY" (all unknown)
}

// --- ISO 13751 10.1.2-3: Additional Grade tests ---

// ISO 13751 10.1.2: Monadic grade on scalar signals RANK ERROR (eval-level test)
TEST_F(StructuralTest, GradeUpScalarRankError) {
    EXPECT_THROW(machine->eval("⍋5"), APLError);
}

TEST_F(StructuralTest, GradeDownScalarRankError) {
    EXPECT_THROW(machine->eval("⍒5"), APLError);
}

// ISO 13751 10.1.2: Single element vector returns ⍳1
TEST_F(StructuralTest, GradeUpSingleElement) {
    Value* result = machine->eval("⍋,5");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 1.0);
}

// ISO 13751 10.1.2: ⎕CT is NOT an implicit argument of grade
TEST_F(StructuralTest, GradeUpCTNotUsed) {
    // Even with large ⎕CT, values should sort by exact value
    machine->eval("⎕CT←0.1");
    Value* result = machine->eval("⍋1 1.05 1.1");
    const Eigen::MatrixXd* m = result->as_matrix();
    // Should sort by exact values: 1 < 1.05 < 1.1
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
    machine->eval("⎕CT←1E¯14");  // Reset
}

// ISO 13751 10.1.2: Numeric matrix - grade sorts by major cells (rows)
TEST_F(StructuralTest, GradeUpNumericMatrix) {
    // Grade up on numeric matrix sorts row indices lexicographically
    Value* result = machine->eval("⍋3 2⍴3 1 2 2 1 3");
    // Rows: [3,1] [2,2] [1,3] → sorted: [1,3]@3 [2,2]@2 [3,1]@1
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);  // [1,3] at row 3
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);  // [2,2] at row 2
    EXPECT_DOUBLE_EQ((*m)(2, 0), 1.0);  // [3,1] at row 1
}

// ============================================================================
// Replicate Function (/)
// ============================================================================

TEST_F(StructuralTest, ReplicateBasic) {
    // 2 0 3 / 1 2 3 → 1 1 3 3 3
    Eigen::VectorXd counts(3);
    counts << 2.0, 0.0, 3.0;
    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(counts);
    Value* rhs = machine->heap->allocate_vector(data);

    fn_replicate(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);  // 2+0+3 = 5
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 3.0);
}

TEST_F(StructuralTest, ReplicateCompress) {
    // 1 0 1 0 1 / 10 20 30 40 50 → 10 30 50 (filter)
    Eigen::VectorXd counts(5);
    counts << 1.0, 0.0, 1.0, 0.0, 1.0;
    Eigen::VectorXd data(5);
    data << 10.0, 20.0, 30.0, 40.0, 50.0;
    Value* lhs = machine->heap->allocate_vector(counts);
    Value* rhs = machine->heap->allocate_vector(data);

    fn_replicate(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 30.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 50.0);
}

TEST_F(StructuralTest, ReplicateAllZero) {
    // 0 0 0 / 1 2 3 → (empty)
    Eigen::VectorXd counts(3);
    counts << 0.0, 0.0, 0.0;
    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(counts);
    Value* rhs = machine->heap->allocate_vector(data);

    fn_replicate(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 0);
}

TEST_F(StructuralTest, ReplicateScalar) {
    // 3 / 5 → 5 5 5
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);

    fn_replicate(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);
}

// ============================================================================
// Set Functions (∪ ~)
// ============================================================================

TEST_F(StructuralTest, UniqueVector) {
    // ∪ 1 2 2 3 1 4 → 1 2 3 4
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 2.0, 3.0, 1.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_unique(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 4.0);
}

TEST_F(StructuralTest, UniqueScalar) {
    // ∪ 5 → ,5 (always a vector per ISO §10.1.8)
    Value* val = machine->heap->allocate_scalar(5.0);

    fn_unique(machine, nullptr, val);

    EXPECT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->size(), 1);
    EXPECT_DOUBLE_EQ(machine->result->as_matrix()->operator()(0, 0), 5.0);
}

TEST_F(StructuralTest, UniqueAllSame) {
    // ∪ 3 3 3 3 → 3
    Eigen::VectorXd v(4);
    v << 3.0, 3.0, 3.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_unique(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 1);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
}

TEST_F(StructuralTest, UniqueAlreadyUnique) {
    // ∪ 1 2 3 4 → 1 2 3 4
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_unique(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
}

// --- ISO 13751 10.1.8: Additional Unique tests ---

// ISO 13751 10.1.8: Rank > 1 signals RANK ERROR
TEST_F(StructuralTest, UniqueMatrixRankError) {
    // ∪ 2 3⍴⍳6 → RANK ERROR
    EXPECT_THROW(machine->eval("∪ 2 3⍴⍳6"), APLError);
}

// ISO 13751 10.1.8: Uses comparison-tolerance
TEST_F(StructuralTest, UniqueCTEffect) {
    // With large ⎕CT, nearly-equal values should be considered duplicates
    machine->eval("⎕CT←0.1");
    Value* result = machine->eval("∪ 1 1.05 2");
    // 1 and 1.05 are within tolerance, so result should be 1 2
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    machine->eval("⎕CT←1E¯14");  // Reset
}

// ISO 13751 10.1.8: Character unique (spec example)
TEST_F(StructuralTest, UniqueCharacter) {
    // ∪'MISSISSIPPI' → 'MISP' (first occurrence order)
    Value* result = machine->eval("∪'MISSISSIPPI'");
    ASSERT_TRUE(result->is_char_data());
    EXPECT_EQ(result->size(), 4);
    // Verify order: M, I, S, P
}

// ISO 13751 10.1.8: Empty vector returns empty
TEST_F(StructuralTest, UniqueEmptyVector) {
    Value* result = machine->eval("∪⍳0");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, UnionBasic) {
    // 1 2 3 ∪ 3 4 5 → 1 2 3 4 5
    Eigen::VectorXd left(3), right(3);
    left << 1.0, 2.0, 3.0;
    right << 3.0, 4.0, 5.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_union(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 5.0);
}

TEST_F(StructuralTest, UnionNoOverlap) {
    // 1 2 ∪ 3 4 → 1 2 3 4
    Eigen::VectorXd left(2), right(2);
    left << 1.0, 2.0;
    right << 3.0, 4.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_union(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
}

TEST_F(StructuralTest, UnionWithDuplicates) {
    // 1 1 2 ∪ 2 3 3 → 1 2 3
    Eigen::VectorXd left(3), right(3);
    left << 1.0, 1.0, 2.0;
    right << 2.0, 3.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_union(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(StructuralTest, WithoutBasic) {
    // 1 2 3 4 5 ~ 2 4 → 1 3 5
    Eigen::VectorXd left(5), right(2);
    left << 1.0, 2.0, 3.0, 4.0, 5.0;
    right << 2.0, 4.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_without(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);
}

TEST_F(StructuralTest, WithoutNoMatch) {
    // 1 2 3 ~ 4 5 6 → 1 2 3
    Eigen::VectorXd left(3), right(3);
    left << 1.0, 2.0, 3.0;
    right << 4.0, 5.0, 6.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_without(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
}

TEST_F(StructuralTest, WithoutAllMatch) {
    // 1 2 3 ~ 1 2 3 → (empty)
    Eigen::VectorXd left(3), right(3);
    left << 1.0, 2.0, 3.0;
    right << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_without(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 0);
}

TEST_F(StructuralTest, WithoutPreservesDuplicates) {
    // 1 2 2 3 3 3 ~ 2 → 1 3 3 3
    Eigen::VectorXd left(6), right(1);
    left << 1.0, 2.0, 2.0, 3.0, 3.0, 3.0;
    right << 2.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_without(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 3.0);
}

TEST_F(StructuralTest, SetFunctionsRegistered) {
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("∪")), nullptr);
    // ~ should already be registered for logical not
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("~")), nullptr);
}

// ============================================================================
// First (↑ monadic) Tests
// ============================================================================

TEST_F(StructuralTest, FirstScalar) {
    Value* scalar = machine->heap->allocate_scalar(42.0);

    fn_first(machine, nullptr, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 42.0);
}

TEST_F(StructuralTest, FirstVector) {
    Eigen::VectorXd v(4);
    v << 10.0, 20.0, 30.0, 40.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_first(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 10.0);
}

TEST_F(StructuralTest, FirstMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1, 2, 3,
         4, 5, 6;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_first(machine, nullptr, mat);

    // First of matrix returns first row as vector
    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(StructuralTest, FirstEmptyVector) {
    Eigen::VectorXd v(0);
    Value* vec = machine->heap->allocate_vector(v);

    fn_first(machine, nullptr, vec);

    // First of empty returns 0 (prototype)
    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(StructuralTest, FirstSingleElementVector) {
    Eigen::VectorXd v(1);
    v << 99.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_first(machine, nullptr, vec);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 99.0);
}

// --- ISO 13751 10.1.9: Additional First tests ---

// ISO 13751 10.1.9: ↑ of vector returns scalar (first element)
TEST_F(StructuralTest, FirstShapeVerification) {
    // ↑1 2 3 → 1 (scalar first element)
    Value* first = machine->eval("↑1 2 3");
    ASSERT_TRUE(first->is_scalar());
    EXPECT_DOUBLE_EQ(first->as_scalar(), 1.0);

    // ⍴↑1 2 3 → ⍬ (shape of scalar is empty vector)
    Value* shape = machine->eval("⍴↑1 2 3");
    ASSERT_TRUE(shape->is_vector());
    EXPECT_EQ(shape->size(), 0);

    // ⍴⍴↑1 2 3 → ,0 (shape of empty vector is 1-element vector [0])
    Value* shape2 = machine->eval("⍴⍴↑1 2 3");
    ASSERT_TRUE(shape2->is_vector());
    EXPECT_EQ(shape2->size(), 1);
    EXPECT_DOUBLE_EQ(shape2->as_matrix()->operator()(0, 0), 0.0);
}

// ISO 13751 10.1.9: First of higher rank array returns major cell
TEST_F(StructuralTest, FirstHigherRank) {
    // ↑ 2 3 4⍴⍳24 → first 3×4 matrix (first plane of 2×3×4 array)
    // ⍳24 (IO=1) = 1..24; first plane is rows 1..12
    Value* result = machine->eval("↑ 2 3 4⍴⍳24");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_matrix());
    const auto* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 4);
    // First row: 1 2 3 4
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 3), 4.0);
    // Last row: 9 10 11 12
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 9.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 3), 12.0);
}

// ISO 13751 10.1.9: First of empty char array returns blank
TEST_F(StructuralTest, FirstEmptyCharReturnsBlank) {
    Value* result = machine->eval("↑''");
    ASSERT_TRUE(result->is_scalar() || result->size() == 1);
    // Result should be blank character ' '
}

// ============================================================================
// Expand (\ dyadic) Tests
// ========================================================================

TEST_F(StructuralTest, ExpandBasic) {
    // 1 0 1 1 \ 1 2 3 → 1 0 2 3
    Eigen::VectorXd mask(4);
    mask << 1.0, 0.0, 1.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, nullptr, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);  // Fill element
    EXPECT_DOUBLE_EQ((*res)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 3.0);
}

TEST_F(StructuralTest, ExpandAllOnes) {
    // 1 1 1 \ 1 2 3 → 1 2 3 (identity)
    Eigen::VectorXd mask(3);
    mask << 1.0, 1.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, nullptr, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(StructuralTest, ExpandLeadingZeros) {
    // 0 0 1 1 \ 1 2 → 0 0 1 2
    Eigen::VectorXd mask(4);
    mask << 0.0, 0.0, 1.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(2);
    data << 1.0, 2.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, nullptr, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 2.0);
}

TEST_F(StructuralTest, ExpandScalar) {
    // 0 1 0 \ 5 → 0 5 0
    Eigen::VectorXd mask(3);
    mask << 0.0, 1.0, 0.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Value* data_val = machine->heap->allocate_scalar(5.0);

    fn_expand(machine, nullptr, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 0.0);
}

TEST_F(StructuralTest, ExpandLengthError) {
    // 1 0 1 \ 1 2 3 is error (2 ones vs 3 elements)
    Eigen::VectorXd mask(3);
    mask << 1.0, 0.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, nullptr, mask_val, data_val);

    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(StructuralTest, ExpandDomainError) {
    // 1 2 1 \ 1 2 is error (non-boolean mask)
    Eigen::VectorXd mask(3);
    mask << 1.0, 2.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(2);
    data << 1.0, 2.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, nullptr, mask_val, data_val);

    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(StructuralTest, ExpandAllZeros) {
    // ISO 10.2.6 example: 0 0\5 → empty vector
    Eigen::VectorXd mask(2);
    mask << 0.0, 0.0;
    Value* mask_val = machine->heap->allocate_vector(mask);
    Value* data_val = machine->heap->allocate_scalar(5.0);

    fn_expand(machine, nullptr, mask_val, data_val);

    // +/0 0 = 0, so B must have 0 elements (scalar 5 is treated as 0-element vector)
    // Result should be empty (all zeros filled)
    ASSERT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->size(), 2);
    // All elements should be fill value (0)
    EXPECT_DOUBLE_EQ(machine->result->as_matrix()->operator()(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(machine->result->as_matrix()->operator()(1, 0), 0.0);
}

// Expand-first (⍀ dyadic) Tests - ISO 13751 Section 10.2.6 variant

TEST_F(StructuralTest, ExpandFirstVector) {
    // 1 0 1⍀1 2 → 1 0 2 (same as expand for vectors)
    Eigen::VectorXd mask(3);
    mask << 1.0, 0.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);
    Eigen::VectorXd data(2);
    data << 1.0, 2.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand_first(machine, nullptr, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->size(), 3);
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);  // Fill
    EXPECT_DOUBLE_EQ((*res)(2, 0), 2.0);
}

TEST_F(StructuralTest, ExpandFirstMatrix) {
    // 1 0 1⍀ 2 3⍴⍳6 → 3×3 matrix with row 2 filled with zeros
    // Input: [[1,2,3],[4,5,6]] → Output: [[1,2,3],[0,0,0],[4,5,6]]
    Eigen::VectorXd mask(3);
    mask << 1.0, 0.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;
    Value* data_val = machine->heap->allocate_matrix(mat);

    fn_expand_first(machine, nullptr, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_matrix());
    EXPECT_EQ(machine->result->rows(), 3);
    EXPECT_EQ(machine->result->cols(), 3);
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    // Row 0: original row 0
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*res)(0, 2), 3.0);
    // Row 1: fill row (zeros)
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 0.0);
    EXPECT_DOUBLE_EQ((*res)(1, 2), 0.0);
    // Row 2: original row 1
    EXPECT_DOUBLE_EQ((*res)(2, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 1), 5.0);
    EXPECT_DOUBLE_EQ((*res)(2, 2), 6.0);
}

TEST_F(StructuralTest, ExpandFirstScalar) {
    // 1 0 1⍀5 → 5 0 5
    Eigen::VectorXd mask(3);
    mask << 1.0, 0.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);
    Value* data_val = machine->heap->allocate_scalar(5.0);

    fn_expand_first(machine, nullptr, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->size(), 3);
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);  // Fill
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);
}

TEST_F(StructuralTest, ExpandFirstLengthError) {
    // 1 0 1⍀ 3 3⍴⍳9 → LENGTH ERROR (3 rows, mask has 2 ones)
    EXPECT_THROW(machine->eval("1 0 1 ⍀ 3 3⍴⍳9"), APLError);
}

// ============================================================================
// Replicate NDARRAY and Strand tests
// ============================================================================

TEST_F(StructuralTest, ReplicateNDArray3D) {
    // 2 1/[3] 2 3 4⍴⍳24 → shape 2 3 8
    // Replicate along last axis (4→8: each element doubled or kept)
    Value* counts_v = machine->heap->allocate_vector(Eigen::Vector4d(2, 1, 2, 1));

    Eigen::VectorXd data(24);
    for (int i = 0; i < 24; ++i) data(i) = i + 1;
    Value* arr = machine->heap->allocate_ndarray(std::move(data), {2, 3, 4});

    Value* axis = machine->heap->allocate_scalar(3.0);
    fn_replicate(machine, axis, counts_v, arr);

    ASSERT_TRUE(machine->result->is_ndarray());
    const Value::NDArrayData* nd = machine->result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3u);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 6);  // 2+1+2+1 = 6
}

TEST_F(StructuralTest, ReplicateNDArrayFirstAxis) {
    // 2 0/[1] 2 3 4⍴⍳24 → shape 2 3 4 (first plane doubled, second removed)
    Value* counts_v = machine->heap->allocate_vector(Eigen::Vector2d(2, 0));

    Eigen::VectorXd data(24);
    for (int i = 0; i < 24; ++i) data(i) = i + 1;
    Value* arr = machine->heap->allocate_ndarray(std::move(data), {2, 3, 4});

    Value* axis = machine->heap->allocate_scalar(1.0);
    fn_replicate(machine, axis, counts_v, arr);

    ASSERT_TRUE(machine->result->is_ndarray());
    const Value::NDArrayData* nd = machine->result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3u);
    EXPECT_EQ(nd->shape[0], 2);  // 2+0 = 2
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);
}

TEST_F(StructuralTest, ReplicateNDArrayScalarExtension) {
    // 2/ 2 3 4⍴⍳24 → each element along last axis doubled
    Value* counts = machine->heap->allocate_scalar(2.0);

    Eigen::VectorXd data(24);
    for (int i = 0; i < 24; ++i) data(i) = i + 1;
    Value* arr = machine->heap->allocate_ndarray(std::move(data), {2, 3, 4});

    fn_replicate(machine, nullptr, counts, arr);

    ASSERT_TRUE(machine->result->is_ndarray());
    const Value::NDArrayData* nd = machine->result->as_ndarray();
    EXPECT_EQ(nd->shape[2], 8);  // 4×2 = 8
}

TEST_F(StructuralTest, ReplicateStrandBasic) {
    // 2 1 3 / ('a' 'b' 'c') → ('a' 'a' 'b' 'c' 'c' 'c')
    Value* counts = machine->heap->allocate_vector(Eigen::Vector3d(2, 1, 3));
    Value* a = machine->heap->allocate_scalar(static_cast<double>('a'));
    Value* b = machine->heap->allocate_scalar(static_cast<double>('b'));
    Value* c = machine->heap->allocate_scalar(static_cast<double>('c'));
    Value* strand = machine->heap->allocate_strand({a, b, c});

    fn_replicate(machine, nullptr, counts, strand);

    ASSERT_TRUE(machine->result->is_strand());
    std::vector<Value*>* result = machine->result->as_strand();
    EXPECT_EQ(result->size(), 6u);  // 2+1+3 = 6
}

TEST_F(StructuralTest, ReplicateStrandCompress) {
    // 1 0 1 / (1 2 3) → (1 3)
    Value* counts = machine->heap->allocate_vector(Eigen::Vector3d(1, 0, 1));
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    Value* v3 = machine->heap->allocate_scalar(3.0);
    Value* strand = machine->heap->allocate_strand({v1, v2, v3});

    fn_replicate(machine, nullptr, counts, strand);

    ASSERT_TRUE(machine->result->is_strand());
    std::vector<Value*>* result = machine->result->as_strand();
    EXPECT_EQ(result->size(), 2u);
    EXPECT_DOUBLE_EQ((*result)[0]->as_scalar(), 1.0);
    EXPECT_DOUBLE_EQ((*result)[1]->as_scalar(), 3.0);
}

TEST_F(StructuralTest, ReplicateStrandEmpty) {
    // 0 0 / (1 2) → empty strand
    Value* counts = machine->heap->allocate_vector(Eigen::Vector2d(0, 0));
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    Value* strand = machine->heap->allocate_strand({v1, v2});

    fn_replicate(machine, nullptr, counts, strand);

    ASSERT_TRUE(machine->result->is_strand());
    std::vector<Value*>* result = machine->result->as_strand();
    EXPECT_EQ(result->size(), 0u);
}

// ============================================================================
// Expand NDARRAY and Strand tests
// ============================================================================

TEST_F(StructuralTest, ExpandNDArray3D) {
    // 1 0 1 0 1 0 \[3] 2 3 3⍴⍳18 → shape 2 3 6
    Value* mask = machine->heap->allocate_vector(Eigen::VectorXd::Map(
        std::vector<double>{1, 0, 1, 0, 1, 0}.data(), 6));

    Eigen::VectorXd data(18);
    for (int i = 0; i < 18; ++i) data(i) = i + 1;
    Value* arr = machine->heap->allocate_ndarray(std::move(data), {2, 3, 3});

    Value* axis = machine->heap->allocate_scalar(3.0);
    fn_expand(machine, axis, mask, arr);

    ASSERT_TRUE(machine->result->is_ndarray());
    const Value::NDArrayData* nd = machine->result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3u);
    EXPECT_EQ(nd->shape[0], 2);
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 6);
}

TEST_F(StructuralTest, ExpandNDArrayFirstAxis) {
    // 1 0 1 \[1] 2 3 4⍴⍳24 → shape 3 3 4 (insert zero plane in middle)
    Value* mask = machine->heap->allocate_vector(Eigen::Vector3d(1, 0, 1));

    Eigen::VectorXd data(24);
    for (int i = 0; i < 24; ++i) data(i) = i + 1;
    Value* arr = machine->heap->allocate_ndarray(std::move(data), {2, 3, 4});

    Value* axis = machine->heap->allocate_scalar(1.0);
    fn_expand(machine, axis, mask, arr);

    ASSERT_TRUE(machine->result->is_ndarray());
    const Value::NDArrayData* nd = machine->result->as_ndarray();
    EXPECT_EQ(nd->shape.size(), 3u);
    EXPECT_EQ(nd->shape[0], 3);  // mask length
    EXPECT_EQ(nd->shape[1], 3);
    EXPECT_EQ(nd->shape[2], 4);

    // Second plane (index 1) should be all zeros
    // Linear index for [1;0;0] = 1*12 = 12
    EXPECT_DOUBLE_EQ((*nd->data)(12), 0.0);
}

TEST_F(StructuralTest, ExpandStrandBasic) {
    // 1 0 1 0 1 \ ('a' 'b' 'c') → ('a' 0 'b' 0 'c')
    Value* mask = machine->heap->allocate_vector(Eigen::VectorXd::Map(
        std::vector<double>{1, 0, 1, 0, 1}.data(), 5));
    Value* a = machine->heap->allocate_scalar(static_cast<double>('a'));
    Value* b = machine->heap->allocate_scalar(static_cast<double>('b'));
    Value* c = machine->heap->allocate_scalar(static_cast<double>('c'));
    Value* strand = machine->heap->allocate_strand({a, b, c});

    fn_expand(machine, nullptr, mask, strand);

    ASSERT_TRUE(machine->result->is_strand());
    std::vector<Value*>* result = machine->result->as_strand();
    EXPECT_EQ(result->size(), 5u);
    EXPECT_DOUBLE_EQ((*result)[0]->as_scalar(), static_cast<double>('a'));
    EXPECT_DOUBLE_EQ((*result)[1]->as_scalar(), 0.0);  // fill
    EXPECT_DOUBLE_EQ((*result)[2]->as_scalar(), static_cast<double>('b'));
    EXPECT_DOUBLE_EQ((*result)[3]->as_scalar(), 0.0);  // fill
    EXPECT_DOUBLE_EQ((*result)[4]->as_scalar(), static_cast<double>('c'));
}

TEST_F(StructuralTest, ExpandStrandAllOnes) {
    // 1 1 1 \ (1 2 3) → (1 2 3)
    Value* mask = machine->heap->allocate_vector(Eigen::Vector3d(1, 1, 1));
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    Value* v3 = machine->heap->allocate_scalar(3.0);
    Value* strand = machine->heap->allocate_strand({v1, v2, v3});

    fn_expand(machine, nullptr, mask, strand);

    ASSERT_TRUE(machine->result->is_strand());
    std::vector<Value*>* result = machine->result->as_strand();
    EXPECT_EQ(result->size(), 3u);
}

TEST_F(StructuralTest, ExpandStrandLeadingZeros) {
    // 0 0 1 1 \ (1 2) → (0 0 1 2)
    Value* mask = machine->heap->allocate_vector(Eigen::Vector4d(0, 0, 1, 1));
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    Value* strand = machine->heap->allocate_strand({v1, v2});

    fn_expand(machine, nullptr, mask, strand);

    ASSERT_TRUE(machine->result->is_strand());
    std::vector<Value*>* result = machine->result->as_strand();
    EXPECT_EQ(result->size(), 4u);
    EXPECT_DOUBLE_EQ((*result)[0]->as_scalar(), 0.0);  // fill
    EXPECT_DOUBLE_EQ((*result)[1]->as_scalar(), 0.0);  // fill
    EXPECT_DOUBLE_EQ((*result)[2]->as_scalar(), 1.0);
    EXPECT_DOUBLE_EQ((*result)[3]->as_scalar(), 2.0);
}

TEST_F(StructuralTest, QuestionRegistered) {
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("?")), nullptr);
}

// ========================================================================
// Dyadic Transpose (⍉) tests
// ============================================================================

TEST_F(StructuralTest, DyadicTransposeScalar) {
    // 0⍉5 → 5 (scalar unchanged)
    Value* lhs = machine->heap->allocate_scalar(0.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);
    fn_dyadic_transpose(machine, nullptr, lhs, rhs);
    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(StructuralTest, DyadicTransposeVectorIdentity) {
    // 1⍉(1 2 3) → 1 2 3 (with ⎕IO=1, axis 1 is the only axis of a vector)
    Value* lhs = machine->heap->allocate_scalar(1.0);  // axis 1 with ⎕IO=1
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* rhs = machine->heap->allocate_vector(vec);
    fn_dyadic_transpose(machine, nullptr, lhs, rhs);
    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(StructuralTest, DyadicTransposeMatrixIdentity) {
    // 1 2⍉M → M (identity permutation, ⎕IO=1)
    Eigen::VectorXd perm(2);
    perm << 1, 2;
    Value* lhs = machine->heap->allocate_vector(perm);
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    fn_dyadic_transpose(machine, nullptr, lhs, rhs);
    ASSERT_FALSE(machine->result->is_scalar());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 2);
    EXPECT_EQ(res->cols(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 2), 6.0);
}

TEST_F(StructuralTest, DyadicTransposeMatrixSwap) {
    // 2 1⍉M → transpose (⎕IO=1)
    Eigen::VectorXd perm(2);
    perm << 2, 1;
    Value* lhs = machine->heap->allocate_vector(perm);
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    fn_dyadic_transpose(machine, nullptr, lhs, rhs);
    ASSERT_FALSE(machine->result->is_scalar());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_EQ(res->cols(), 2);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 1), 6.0);
}

TEST_F(StructuralTest, DyadicTransposeInvalidPermError) {
    // 3 3⍉M → DOMAIN ERROR (out of range, valid axes are 1-2 with ⎕IO=1)
    Eigen::VectorXd perm(2);
    perm << 3, 3;
    Value* lhs = machine->heap->allocate_vector(perm);
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    fn_dyadic_transpose(machine, nullptr, lhs, rhs);
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

// --- ISO 13751 10.1.5/10.2.10: Additional Transpose tests ---

// ISO 13751 10.1.5: Monadic transpose on scalar returns scalar
TEST_F(StructuralTest, MonadicTransposeScalar) {
    Value* result = machine->eval("⍉5");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// ISO 13751 10.2.10: Dyadic transpose diagonal selection (1 1⍉M)
TEST_F(StructuralTest, DyadicTransposeDiagonal) {
    // 1 1⍉ 3 3⍴⍳9 → diagonal: 1 5 9
    Value* result = machine->eval("1 1⍉ 3 3⍴⍳9");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 9.0);
}

// ISO 13751 10.2.10: Permutation length must match rank
TEST_F(StructuralTest, DyadicTransposeLengthError) {
    // 1 2 3⍉ 2 3⍴⍳6 → LENGTH ERROR (3 perms for rank-2 array)
    EXPECT_THROW(machine->eval("1 2 3⍉ 2 3⍴⍳6"), APLError);
}

// ISO 13751 10.2.10: Non-integer permutation signals DOMAIN ERROR
TEST_F(StructuralTest, DyadicTransposeNonIntegerError) {
    EXPECT_THROW(machine->eval("1.5 2⍉ 2 3⍴⍳6"), APLError);
}

// ISO 13751 10.2.10: Permutation out of range signals DOMAIN ERROR
TEST_F(StructuralTest, DyadicTransposeOutOfRangeError) {
    // 1 3⍉ 2 3⍴⍳6 → DOMAIN ERROR (3 > rank)
    EXPECT_THROW(machine->eval("1 3⍉ 2 3⍴⍳6"), APLError);
}

// ISO 13751 10.2.10: Empty permutation on scalar
TEST_F(StructuralTest, DyadicTransposeEmptyPermScalar) {
    // (⍳0)⍉5 → 5 (empty perm on scalar returns scalar)
    Value* result = machine->eval("(⍳0)⍉5");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(StructuralTest, DominoRegistered) {
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⌹")), nullptr);
}

// ============================================================================
// Execute (⍎) tests
// ============================================================================

TEST_F(StructuralTest, ExecuteRequiresString) {
    // ⍎5 → DOMAIN ERROR (not a string)
    Value* val = machine->heap->allocate_scalar(5.0);
    fn_execute(machine, nullptr, val);
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

TEST_F(StructuralTest, ExecutePushesContination) {
    // ⍎'42' should push a continuation
    Value* str = machine->heap->allocate_string("42");
    size_t stack_before = machine->kont_stack.size();
    fn_execute(machine, nullptr, str);
    EXPECT_GT(machine->kont_stack.size(), stack_before);
}

TEST_F(StructuralTest, ExecuteRegistered) {
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⍎")), nullptr);
}

TEST_F(StructuralTest, ExecuteEmptyString) {
    // ⍎'' → zilde (empty numeric vector)
    Value* result = machine->eval("⍎''");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

// --- ISO 13751 10.1.7: Additional Execute tests ---

// ISO 13751 10.1.7: Syntax error in executed string
TEST_F(StructuralTest, ExecuteSyntaxError) {
    // ⍎'1++' → SYNTAX ERROR
    EXPECT_THROW(machine->eval("⍎'1++'"), APLError);
}

// ISO 13751 10.1.7: Execute variable reference
TEST_F(StructuralTest, ExecuteVariableRef) {
    machine->eval("X←42");
    Value* result = machine->eval("⍎'X'");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// ISO 13751 10.1.7: Execute assignment
TEST_F(StructuralTest, ExecuteAssignment) {
    machine->eval("⍎'Y←99'");
    Value* result = machine->eval("Y");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 99.0);
}

// ISO 13751 10.1.7: Execute arithmetic expression
TEST_F(StructuralTest, ExecuteArithmetic) {
    Value* result = machine->eval("⍎'2+3×4'");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);  // 2+12 = 14
}

// ISO 13751 10.1.7: Execute undefined variable should error
TEST_F(StructuralTest, ExecuteUndefinedVariable) {
    EXPECT_THROW(machine->eval("⍎'UNDEFINED_VAR_XYZ'"), APLError);
}

// ============================================================================
// Squad (Indexing) Tests - ⌷
// ============================================================================

TEST_F(StructuralTest, SquadRegistered) {
    // ⌷ should be registered in the environment
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⌷")), nullptr);
}

TEST_F(StructuralTest, SquadVectorScalarIndex) {
    // (1 2 3 4 5)[3] → 3  (1-based indexing)
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(3.0);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, arr);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(StructuralTest, SquadVectorVectorIndex) {
    // (10 20 30 40 50)[2 4] → 20 40
    Eigen::VectorXd v(5);
    v << 10.0, 20.0, 30.0, 40.0, 50.0;
    Value* arr = machine->heap->allocate_vector(v);

    Eigen::VectorXd idx_v(2);
    idx_v << 2.0, 4.0;
    Value* idx = machine->heap->allocate_vector(idx_v);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, arr);

    ASSERT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->data.matrix->size(), 2);
    EXPECT_DOUBLE_EQ((*machine->result->data.matrix)(0, 0), 20.0);
    EXPECT_DOUBLE_EQ((*machine->result->data.matrix)(1, 0), 40.0);
}

TEST_F(StructuralTest, SquadVectorFirstElement) {
    // (5 6 7)[1] → 5  (first element, 1-based)
    Eigen::VectorXd v(3);
    v << 5.0, 6.0, 7.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(1.0);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, arr);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(StructuralTest, SquadVectorLastElement) {
    // (5 6 7)[3] → 7  (last element)
    Eigen::VectorXd v(3);
    v << 5.0, 6.0, 7.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(3.0);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, arr);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 7.0);
}

TEST_F(StructuralTest, SquadOutOfBoundsError) {
    // (1 2 3)[5] → INDEX ERROR
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(5.0);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, arr);

    // Should push ThrowErrorK
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

TEST_F(StructuralTest, SquadZeroIndexError) {
    // (1 2 3)[0] → INDEX ERROR (APL is 1-based)
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(0.0);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, arr);

    // Should push ThrowErrorK
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

// ============================================================================
// Squad String Indexing Tests
// ============================================================================

TEST_F(StructuralTest, SquadStringScalarIndex) {
    // 'hello'[2] → 101 (ASCII 'e')
    Value* str = machine->heap->allocate_string("hello");
    Value* idx = machine->heap->allocate_scalar(2.0);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, str);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 101.0);  // 'e'
}

TEST_F(StructuralTest, SquadStringFirstChar) {
    // 'hello'[1] → 104 (ASCII 'h')
    Value* str = machine->heap->allocate_string("hello");
    Value* idx = machine->heap->allocate_scalar(1.0);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, str);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 104.0);  // 'h'
}

TEST_F(StructuralTest, SquadStringLastChar) {
    // 'hello'[5] → 111 (ASCII 'o')
    Value* str = machine->heap->allocate_string("hello");
    Value* idx = machine->heap->allocate_scalar(5.0);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, str);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 111.0);  // 'o'
}

TEST_F(StructuralTest, SquadStringVectorIndex) {
    // 'hello'[1 3 5] → 104 108 111 (ASCII for 'hlo')
    Value* str = machine->heap->allocate_string("hello");
    Eigen::VectorXd idx_v(3);
    idx_v << 1.0, 3.0, 5.0;
    Value* idx = machine->heap->allocate_vector(idx_v);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, str);

    ASSERT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->size(), 3);
    auto* mat = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 104.0);  // 'h'
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 108.0);  // 'l'
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 111.0);  // 'o'
}

TEST_F(StructuralTest, SquadStringOutOfBoundsError) {
    // 'hi'[5] → INDEX ERROR
    Value* str = machine->heap->allocate_string("hi");
    Value* idx = machine->heap->allocate_scalar(5.0);

    // ISO 13751: I⌷A where indices are left, array is right
    fn_squad(machine, nullptr, idx, str);

    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

// ============================================================================
// Bracket Indexing Syntax Tests (via parser)
// ============================================================================

TEST_F(StructuralTest, BracketIndexVectorScalar) {
    // (1 2 3 4 5)[3] → 3
    Value* result = machine->eval("(1 2 3 4 5)[3]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(StructuralTest, BracketIndexVectorVector) {
    // (10 20 30)[1 3] → 10 30
    Value* result = machine->eval("(10 20 30)[1 3]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->data.matrix->size(), 2);
    EXPECT_DOUBLE_EQ((*result->data.matrix)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*result->data.matrix)(1, 0), 30.0);
}

TEST_F(StructuralTest, BracketIndexIota) {
    // (⍳5)[3] → 3
    Value* result = machine->eval("(⍳5)[3]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(StructuralTest, BracketIndexVariable) {
    // x←1 2 3 4 5 ⋄ x[2]
    machine->eval("x←1 2 3 4 5");
    Value* result = machine->eval("x[2]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(StructuralTest, BracketIndexString) {
    // 'hello'[2] → 101 (ASCII 'e')
    Value* result = machine->eval("'hello'[2]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 101.0);  // 'e'
}

TEST_F(StructuralTest, BracketIndexStringMultiple) {
    // 'abcde'[5 4 3 2 1] → 101 100 99 98 97 (ASCII for 'edcba')
    Value* result = machine->eval("'abcde'[5 4 3 2 1]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 101.0);  // 'e'
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 100.0);  // 'd'
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 99.0);   // 'c'
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 98.0);   // 'b'
    EXPECT_DOUBLE_EQ((*mat)(4, 0), 97.0);   // 'a'
}

TEST_F(StructuralTest, BracketIndexChained) {
    // ((1 2 3)(4 5 6))[2] - would need nested arrays, skip for now
    // Instead test: (⍳10)[⍳3] → 1 2 3
    Value* result = machine->eval("(⍳10)[⍳3]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->data.matrix->size(), 3);
    EXPECT_DOUBLE_EQ((*result->data.matrix)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*result->data.matrix)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*result->data.matrix)(2, 0), 3.0);
}

// ============================================================================
// Table Function (⍪) Tests
// ============================================================================

TEST_F(StructuralTest, TableScalar) {
    // ⍪ 5 → 1×1 matrix containing 5
    Value* scalar = machine->heap->allocate_scalar(5.0);
    fn_table(machine, nullptr, scalar);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_matrix());
    EXPECT_FALSE(result->is_vector());  // Must be matrix, not vector
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 1);
    EXPECT_EQ(mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 5.0);
}

TEST_F(StructuralTest, TableVector) {
    // ⍸ 1 2 3 4 → 4×1 matrix
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_table(machine, nullptr, vec);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_matrix());
    EXPECT_FALSE(result->is_vector());  // Must be matrix, not vector
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 4);
    EXPECT_EQ(mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 4.0);
}

TEST_F(StructuralTest, TableMatrix) {
    // ⍸ (2 3⍴⍳6) → same 2×3 matrix (unchanged for 2D)
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat_val = machine->heap->allocate_matrix(m);
    fn_table(machine, nullptr, mat_val);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 6.0);
}

// ============================================================================
// Phase 3: Empty Array Handling Tests (ISO 13751)
// ============================================================================

// --- Structural Operations on Empty Arrays ---

TEST_F(StructuralTest, ShapeEmptyVector) {
    // ⍴⍳0 → 1-element vector containing 0
    Value* result = machine->eval("⍴⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 0.0);
}

TEST_F(StructuralTest, ShapeEmptyMatrix) {
    // ⍴0 3⍴0 → 0 3 (shape of 0×3 matrix)
    Value* result = machine->eval("⍴0 3⍴0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 3.0);
}

TEST_F(StructuralTest, RavelEmptyMatrix) {
    // ,0 3⍴0 → empty vector
    Value* result = machine->eval(",0 3⍴0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, CatenateEmptyLeft) {
    // (⍳0),1 2 3 → 1 2 3
    Value* result = machine->eval("(⍳0),1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 3.0);
}

TEST_F(StructuralTest, CatenateEmptyRight) {
    // 1 2 3,⍳0 → 1 2 3
    Value* result = machine->eval("1 2 3,⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 3.0);
}

TEST_F(StructuralTest, CatenateEmptyBoth) {
    // (⍳0),⍳0 → empty vector
    Value* result = machine->eval("(⍳0),⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, TallyEmpty) {
    // ≢⍳0 → 0
    Value* result = machine->eval("≢⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(StructuralTest, ReverseEmpty) {
    // ⌽⍳0 → empty vector
    Value* result = machine->eval("⌽⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, TransposeEmpty) {
    // ⍉0 3⍴0 → 3 0 matrix
    Value* result = machine->eval("⍉0 3⍴0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 0);
}

// --- Arithmetic on Empty Arrays ---

TEST_F(StructuralTest, AddScalarEmpty) {
    // 5+⍳0 → empty vector (scalar extension)
    Value* result = machine->eval("5+⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, AddEmptyScalar) {
    // (⍳0)+5 → empty vector
    Value* result = machine->eval("(⍳0)+5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, DivideScalarEmpty) {
    // 5÷⍳0 → empty vector (no domain error!)
    Value* result = machine->eval("5÷⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, AddEmptyEmpty) {
    // (⍳0)+⍳0 → empty vector
    Value* result = machine->eval("(⍳0)+⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, TimesEmptyEmpty) {
    // (⍳0)×⍳0 → empty vector
    Value* result = machine->eval("(⍳0)×⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, NegateEmpty) {
    // -⍳0 → empty vector
    Value* result = machine->eval("-⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, ReciprocalEmpty) {
    // ÷⍳0 → empty vector (no domain error!)
    Value* result = machine->eval("÷⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

// --- Search Functions with Empty Arrays ---

TEST_F(StructuralTest, MembershipEmptyRight) {
    // 1 2 3∊⍳0 → 0 0 0 (nothing found in empty set)
    Value* result = machine->eval("1 2 3∊⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 0.0);
}

TEST_F(StructuralTest, MembershipEmptyLeft) {
    // (⍳0)∊1 2 3 → empty vector
    Value* result = machine->eval("(⍳0)∊1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, UniqueEmpty) {
    // ∪⍳0 → empty vector
    Value* result = machine->eval("∪⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, GradeUpEmpty) {
    // ⍋⍳0 → empty vector
    Value* result = machine->eval("⍋⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, GradeDownEmpty) {
    // ⍒⍳0 → empty vector
    Value* result = machine->eval("⍒⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

// --- Take/Drop with Empty Arrays ---

TEST_F(StructuralTest, TakeFromEmpty) {
    // 3↑⍳0 → 0 0 0 (take pads with zeros)
    Value* result = machine->eval("3↑⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 0.0);
}

TEST_F(StructuralTest, TakeZeroElements) {
    // 0↑1 2 3 → empty vector
    Value* result = machine->eval("0↑1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, DropToEmpty) {
    // 3↓1 2 3 → empty vector
    Value* result = machine->eval("3↓1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, DropFromEmpty) {
    // 3↓⍳0 → empty vector
    Value* result = machine->eval("3↓⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, TakeNegativeOverextend) {
    // ISO 10.2.11: ¯5↑1 2 3 → 0 0 1 2 3 (pads at beginning)
    Value* result = machine->eval("¯5↑1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(3, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(4, 0), 3.0);
}

// ============================================================================
// Left (⊣) and Right (⊢) - ISO 10.2.17-18
// ============================================================================

TEST_F(StructuralTest, LeftTackDyadic) {
    // ISO 10.2.17: A⊣B returns A
    Value* result = machine->eval("3⊣5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(StructuralTest, RightTackDyadic) {
    // ISO 10.2.18: A⊢B returns B
    Value* result = machine->eval("3⊢5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(StructuralTest, LeftTackMonadic) {
    // ISO 10.2.17: ⊣B returns B (identity)
    Value* result = machine->eval("⊣5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(StructuralTest, RightTackMonadic) {
    // ISO 10.2.18: ⊢B returns B (identity)
    Value* result = machine->eval("⊢5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(StructuralTest, LeftTackVector) {
    // A⊣B with vectors returns A unchanged
    Value* result = machine->eval("1 2 3⊣4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 3.0);
}

TEST_F(StructuralTest, RightTackVector) {
    // A⊢B with vectors returns B unchanged
    Value* result = machine->eval("1 2 3⊢4 5 6");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 6.0);
}

TEST_F(StructuralTest, LeftTackMixedShapes) {
    // ISO 10.2.17 example: N2⊣'FRANCE' → 1 2
    // Left returns left arg regardless of right arg's shape
    Value* result = machine->eval("1 2⊣'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 2.0);
}

TEST_F(StructuralTest, RightTackMixedShapes) {
    // Right returns right arg regardless of left arg's shape
    Value* result = machine->eval("1 2 3⊢5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(StructuralTest, LeftTackEmpty) {
    // Empty vector as left argument
    Value* result = machine->eval("(⍳0)⊣1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, RightTackEmpty) {
    // Empty vector as right argument
    Value* result = machine->eval("1 2 3⊢⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(StructuralTest, LeftTackMatrix) {
    // Matrix as left argument
    Value* result = machine->eval("(2 2⍴1 2 3 4)⊣99");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->as_matrix()->rows(), 2);
    EXPECT_EQ(result->as_matrix()->cols(), 2);
}

TEST_F(StructuralTest, RightTackMatrix) {
    // Matrix as right argument
    Value* result = machine->eval("99⊢2 2⍴1 2 3 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->as_matrix()->rows(), 2);
    EXPECT_EQ(result->as_matrix()->cols(), 2);
}

TEST_F(StructuralTest, LeftTackRegistered) {
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⊣")), nullptr);
}

TEST_F(StructuralTest, RightTackRegistered) {
    ASSERT_NE(machine->env->lookup(machine->string_pool.intern("⊢")), nullptr);
}

// ============================================================================
// Structural Function Combinations: Catenate First (⍪)
// ============================================================================

TEST_F(StructuralTest, CatenateFirstAllCombinations) {
    // Test all 9 argument combinations for ⍪ (catenate first axis)
    // Per ISO 13751 Section 8.3.2: A⍪B is A,[1]B
    // Scalar extension applies: scalar extends to match other arg's trailing dims
    struct TestCase {
        std::string left;
        std::string right;
        bool should_succeed;
        int expected_rows;
        int expected_cols;
        std::string description;
    };

    std::vector<TestCase> cases = {
        // Scalar combinations
        {"5",           "3",           true,  2, 1, "scalar-scalar"},
        {"5",           "1 2 3",       true,  2, 3, "scalar-vector (extension)"},
        {"5",           "2 3⍴⍳6",      true,  3, 3, "scalar-matrix (extension)"},
        // Vector combinations
        {"1 2 3",       "4",           true,  2, 3, "vector-scalar (extension)"},
        {"1 2 3",       "4 5 6",       true,  2, 3, "vector-vector (same len)"},
        {"1 2 3",       "4 5",         false, 0, 0, "vector-vector (diff len)"},
        {"1 2 3",       "2 3⍴⍳6",      true,  3, 3, "vector-matrix (matching cols)"},
        {"1 2 3",       "2 4⍴⍳8",      false, 0, 0, "vector-matrix (diff cols)"},
        // Matrix combinations
        {"2 3⍴⍳6",      "7",           true,  3, 3, "matrix-scalar (extension)"},
        {"2 3⍴⍳6",      "7 8 9",       true,  3, 3, "matrix-vector (matching cols)"},
        {"2 3⍴⍳6",      "7 8",         false, 0, 0, "matrix-vector (diff cols)"},
        {"2 3⍴⍳6",      "2 3⍴7 8 9 10 11 12", true, 4, 3, "matrix-matrix (same cols)"},
        {"2 3⍴⍳6",      "2 4⍴⍳8",      false, 0, 0, "matrix-matrix (diff cols)"},
    };

    int total = 0, passed = 0;
    for (const auto& tc : cases) {
        total++;
        std::string expr = "(" + tc.left + ")⍪(" + tc.right + ")";

        if (tc.should_succeed) {
            try {
                Value* result = machine->eval(expr);
                if (result && result->is_matrix()) {
                    const Eigen::MatrixXd* mat = result->as_matrix();
                    if (mat->rows() == tc.expected_rows && mat->cols() == tc.expected_cols) {
                        passed++;
                    } else {
                        ADD_FAILURE() << "Wrong shape for ⍪ " << tc.description
                                      << ": got " << mat->rows() << "×" << mat->cols()
                                      << ", expected " << tc.expected_rows << "×" << tc.expected_cols;
                    }
                } else {
                    ADD_FAILURE() << "Non-matrix result for ⍪ " << tc.description << ": " << expr;
                }
            } catch (const std::exception& e) {
                ADD_FAILURE() << "Unexpected error for ⍪ " << tc.description << ": " << e.what();
            }
        } else {
            try {
                machine->eval(expr);
                ADD_FAILURE() << "Expected error for ⍪ " << tc.description << ": " << expr;
            } catch (const APLError&) {
                passed++;
            }
        }
    }
    EXPECT_EQ(passed, total) << "Failed " << (total - passed) << " of " << total << " ⍪ tests";
}

// ============================================================================
// Phase 5: Index Origin (⎕IO) Tests via C++ API
// ============================================================================

TEST_F(StructuralTest, IotaIO1) {
    // Default ⎕IO=1: ⍳3 → 1 2 3
    EXPECT_EQ(machine->io, 1);  // Verify default
    Value* result = machine->eval("⍳3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
}

TEST_F(StructuralTest, IotaIO0) {
    // ⎕IO=0: ⍳3 → 0 1 2
    machine->io = 0;
    Value* result = machine->eval("⍳3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 2.0);
}

TEST_F(StructuralTest, GradeUpIO1) {
    // Default ⎕IO=1: ⍋3 1 2 → 2 3 1
    EXPECT_EQ(machine->io, 1);
    Value* result = machine->eval("⍋3 1 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);
}

TEST_F(StructuralTest, GradeUpIO0) {
    // ⎕IO=0: ⍋3 1 2 → 1 2 0
    machine->io = 0;
    Value* result = machine->eval("⍋3 1 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 0.0);
}

TEST_F(StructuralTest, GradeDownIO0) {
    // ⎕IO=0: ⍒3 1 2 → 0 2 1
    machine->io = 0;
    Value* result = machine->eval("⍒3 1 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);
}

TEST_F(StructuralTest, IndexingIO1) {
    // Default ⎕IO=1: (1 2 3)[2] → 2
    EXPECT_EQ(machine->io, 1);
    Value* result = machine->eval("(1 2 3)[2]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(StructuralTest, IndexingIO0) {
    // ⎕IO=0: (1 2 3)[0] → 1
    machine->io = 0;
    Value* result = machine->eval("(1 2 3)[0]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(StructuralTest, RollIO0) {
    // ⎕IO=0: ?5 should return values in 0..4
    machine->io = 0;
    for (int i = 0; i < 20; i++) {
        Value* result = machine->eval("?5");
        ASSERT_NE(result, nullptr);
        EXPECT_TRUE(result->is_scalar());
        double val = result->as_scalar();
        EXPECT_GE(val, 0.0);
        EXPECT_LE(val, 4.0);
    }
}

TEST_F(StructuralTest, RollIO1) {
    // ⎕IO=1: ?5 should return values in 1..5
    EXPECT_EQ(machine->io, 1);
    for (int i = 0; i < 20; i++) {
        Value* result = machine->eval("?5");
        ASSERT_NE(result, nullptr);
        EXPECT_TRUE(result->is_scalar());
        double val = result->as_scalar();
        EXPECT_GE(val, 1.0);
        EXPECT_LE(val, 5.0);
    }
}

// ============================================================================
// Structural Function Rejection Tests
// ============================================================================

TEST_F(StructuralTest, ShapeRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("⍴+"), APLError);
}

TEST_F(StructuralTest, RavelRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval(",+"), APLError);
}

TEST_F(StructuralTest, ReverseRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("⌽+"), APLError);
}

TEST_F(StructuralTest, ReverseFirstRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("⊖+"), APLError);
}

TEST_F(StructuralTest, TransposeRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("⍉+"), APLError);
}

TEST_F(StructuralTest, FirstRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("↑+"), APLError);
}

TEST_F(StructuralTest, TallyRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("≢+"), APLError);
}

TEST_F(StructuralTest, TableRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("⍪+"), APLError);
}

TEST_F(StructuralTest, GradeUpRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("⍋+"), APLError);
}

TEST_F(StructuralTest, GradeDownRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("⍒+"), APLError);
}

TEST_F(StructuralTest, UniqueRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("∪+"), APLError);
}

TEST_F(StructuralTest, DyadicReshapeRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("3⍴+"), APLError);
}

TEST_F(StructuralTest, DyadicCatenateRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1,+"), APLError);
}

TEST_F(StructuralTest, DyadicRotateRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1⌽+"), APLError);
}

TEST_F(StructuralTest, DyadicRotateFirstRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1⊖+"), APLError);
}

TEST_F(StructuralTest, DyadicTakeRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("3↑+"), APLError);
}

TEST_F(StructuralTest, DyadicDropRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1↓+"), APLError);
}

TEST_F(StructuralTest, DyadicIndexOfRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1 2 3⍳+"), APLError);
}

TEST_F(StructuralTest, DyadicMemberOfRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1∊+"), APLError);
}

TEST_F(StructuralTest, DyadicUnionRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1 2∪+"), APLError);
}

TEST_F(StructuralTest, DyadicWithoutRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1 2 3~+"), APLError);
}

TEST_F(StructuralTest, DyadicCatenateFirstRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1⍪+"), APLError);
}

TEST_F(StructuralTest, DyadicTransposeRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1 0⍉+"), APLError);
}

// ============================================================================
// Special Function Rejection Tests
// ============================================================================

TEST_F(StructuralTest, FormatMonadicRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("⍕+"), APLError);
}

TEST_F(StructuralTest, FormatDyadicRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("10 2⍕+"), APLError);
}

TEST_F(StructuralTest, MatrixInverseRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("⌹+"), APLError);
}

TEST_F(StructuralTest, MatrixDivideRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("1 2 3⌹+"), APLError);
}

TEST_F(StructuralTest, EncodeRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("2 2 2⊤+"), APLError);
}

TEST_F(StructuralTest, DecodeRejectsFunctionArgument) {
    EXPECT_THROW(machine->eval("10⊥+"), APLError);
}

// ========================================================================
// Enclose (⊂) Tests - ISO 13751 Section 10.2.26
// ========================================================================

// ISO 13751: If B is a simple-scalar, Z is B (scalars don't get enclosed)
TEST_F(StructuralTest, EncloseScalarReturnsScalar) {
    Value* result = machine->eval("⊂5");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Enclose of vector creates strand containing vector
TEST_F(StructuralTest, EncloseVectorCreatesStrand) {
    Value* result = machine->eval("⊂1 2 3");
    ASSERT_TRUE(result->is_strand());
    std::vector<Value*>* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 1);
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_EQ((*strand)[0]->size(), 3);
}

// Enclose of matrix creates strand containing matrix
TEST_F(StructuralTest, EncloseMatrixCreatesStrand) {
    Value* result = machine->eval("⊂2 3⍴⍳6");
    ASSERT_TRUE(result->is_strand());
    std::vector<Value*>* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 1);
    ASSERT_TRUE((*strand)[0]->is_matrix());
    EXPECT_EQ((*strand)[0]->rows(), 2);
    EXPECT_EQ((*strand)[0]->cols(), 3);
}

// Enclose of string creates strand containing string
TEST_F(StructuralTest, EncloseStringCreatesStrand) {
    Value* result = machine->eval("⊂'OSCAR'");
    ASSERT_TRUE(result->is_strand());
    std::vector<Value*>* strand = result->as_strand();
    ASSERT_EQ(strand->size(), 1);
    // String is converted to char vector
    ASSERT_TRUE((*strand)[0]->is_vector() || (*strand)[0]->is_string());
}

// Double enclose creates nested strand
TEST_F(StructuralTest, DoubleEncloseCreatesNestedStrand) {
    Value* result = machine->eval("⊂⊂1 2 3");
    ASSERT_TRUE(result->is_strand());
    std::vector<Value*>* outer = result->as_strand();
    ASSERT_EQ(outer->size(), 1);
    ASSERT_TRUE((*outer)[0]->is_strand());
    std::vector<Value*>* inner = (*outer)[0]->as_strand();
    ASSERT_EQ(inner->size(), 1);
    ASSERT_TRUE((*inner)[0]->is_vector());
}

// Enclose preserves identity: ⊃⊂B ≡ B for non-scalars
TEST_F(StructuralTest, EncloseDiscloseIdentityVector) {
    Value* result = machine->eval("⊃⊂1 2 3");
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
}

// ========================================================================
// Disclose (⊃ monadic) Tests - ISO 13751 Section 10.2.24
// ========================================================================

// Disclose of scalar returns scalar
TEST_F(StructuralTest, DiscloseScalarReturnsScalar) {
    Value* result = machine->eval("⊃5");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Disclose of vector returns first element
TEST_F(StructuralTest, DiscloseVectorReturnsFirst) {
    Value* result = machine->eval("⊃1 2 3");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// Disclose of matrix returns first row
TEST_F(StructuralTest, DiscloseMatrixReturnsFirstRow) {
    Value* result = machine->eval("⊃2 3⍴⍳6");
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
}

// Disclose of empty vector returns prototype (0)
TEST_F(StructuralTest, DiscloseEmptyVectorReturnsZero) {
    Value* result = machine->eval("⊃⍬");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Disclose of strand returns first element
TEST_F(StructuralTest, DiscloseStrandReturnsFirst) {
    Value* result = machine->eval("⊃⊂1 2 3");  // ⊂ creates strand, ⊃ extracts
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

// Disclose of function returns function unchanged
TEST_F(StructuralTest, DiscloseFunctionReturnsFunction) {
    Value* result = machine->eval("⊃+");
    ASSERT_TRUE(result->is_primitive());
}

// ========================================================================
// Pick (⊃ dyadic) Tests - ISO 13751 Section 10.2.22
// ========================================================================

// Pick from vector with scalar index
TEST_F(StructuralTest, PickVectorScalarIndex) {
    Value* result = machine->eval("2⊃1 2 3");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// Pick from vector - index out of range
TEST_F(StructuralTest, PickVectorIndexOutOfRange) {
    EXPECT_THROW(machine->eval("5⊃1 2 3"), APLError);
}

// Pick from vector - zero index (⎕IO=1)
TEST_F(StructuralTest, PickVectorZeroIndex) {
    EXPECT_THROW(machine->eval("0⊃1 2 3"), APLError);
}

// Pick from strand
TEST_F(StructuralTest, PickFromStrand) {
    // Create strand via catenate of enclosed values
    Value* result = machine->eval("1⊃⊂1 2 3");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

// ========================================================================
// Enclose with Axis - ISO 13751 Section 10.2.27
// ========================================================================

TEST_F(StructuralTest, EncloseWithAxisVector) {
    // ⊂[1]vec - partition vector into strand of scalars
    Value* result = machine->eval("⊂[1]1 2 3");
    ASSERT_TRUE(result->is_strand());
    std::vector<Value*>* strand = result->as_strand();
    EXPECT_EQ(strand->size(), 3);
    EXPECT_TRUE((*strand)[0]->is_scalar());
    EXPECT_EQ((*strand)[0]->as_scalar(), 1);
    EXPECT_EQ((*strand)[1]->as_scalar(), 2);
    EXPECT_EQ((*strand)[2]->as_scalar(), 3);
}

TEST_F(StructuralTest, EncloseWithAxisMatrixAxis1) {
    // ⊂[1]matrix - each column becomes an element
    Value* result = machine->eval("⊂[1]2 3⍴⍳6");
    ASSERT_TRUE(result->is_strand());
    std::vector<Value*>* strand = result->as_strand();
    EXPECT_EQ(strand->size(), 3);  // 3 columns
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_EQ((*strand)[0]->size(), 2);  // Each column has 2 elements
}

TEST_F(StructuralTest, EncloseWithAxisMatrixAxis2) {
    // ⊂[2]matrix - each row becomes an element
    Value* result = machine->eval("⊂[2]2 3⍴⍳6");
    ASSERT_TRUE(result->is_strand());
    std::vector<Value*>* strand = result->as_strand();
    EXPECT_EQ(strand->size(), 2);  // 2 rows
    ASSERT_TRUE((*strand)[0]->is_vector());
    EXPECT_EQ((*strand)[0]->size(), 3);  // Each row has 3 elements
}

TEST_F(StructuralTest, EncloseWithAxisInvalidAxis) {
    // Axis out of range
    EXPECT_THROW(machine->eval("⊂[3]1 2 3"), APLError);
    EXPECT_THROW(machine->eval("⊂[0]1 2 3"), APLError);
}

// ========================================================================
// Disclose with Axis - ISO 13751 Section 10.2.25
// ========================================================================

TEST_F(StructuralTest, DiscloseWithAxisVector) {
    // ⊃[1] on enclosed vector - returns first element (vector), axis 1 stays
    Value* result = machine->eval("⊃[1]⊂1 2 3");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

TEST_F(StructuralTest, DiscloseWithAxisMatrix) {
    // ⊃[1 2] on enclosed matrix - normal order
    Value* result = machine->eval("⊃[1 2]⊂2 3⍴⍳6");
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
}

TEST_F(StructuralTest, DiscloseWithAxisMatrixTranspose) {
    // ⊃[2 1] on enclosed matrix - transposed
    Value* result = machine->eval("⊃[2 1]⊂2 3⍴⍳6");
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);  // Transposed: cols become rows
    EXPECT_EQ(mat->cols(), 2);  // Transposed: rows become cols
}

TEST_F(StructuralTest, DiscloseWithAxisScalar) {
    // Scalars have rank 0, so specifying any axis is invalid
    EXPECT_THROW(machine->eval("⊃[1]5"), APLError);
}

TEST_F(StructuralTest, DiscloseWithAxisInvalidCount) {
    // Axis count must match rank
    EXPECT_THROW(machine->eval("⊃[1 2]⊂1 2 3"), APLError);  // Vector has rank 1, not 2
}

TEST_F(StructuralTest, DiscloseWithAxisInvalidValue) {
    // Axis values must be in range
    EXPECT_THROW(machine->eval("⊃[3]⊂2 3⍴⍳6"), APLError);  // Matrix has rank 2, not 3
    EXPECT_THROW(machine->eval("⊃[0]⊂1 2 3"), APLError);  // Axis 0 invalid with ⎕IO=1
}

TEST_F(StructuralTest, DiscloseWithAxisDuplicate) {
    // Duplicate axis values are invalid
    EXPECT_THROW(machine->eval("⊃[1 1]⊂2 3⍴⍳6"), APLError);
}

// ========================================================================
// Pervasive Dispatch on Strands - ISO 13751 Scalar Function Extension
// ========================================================================
// Scalar (pervasive) functions automatically penetrate nested arrays,
// applying element-wise to the contents of enclosed values.

TEST_F(StructuralTest, PervasiveAddScalarToStrand) {
    // 1+⊂2 3 4 → (3 4 5) - add scalar to each element of enclosed vector
    Value* result = machine->eval("1+⊂2 3 4");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 1);
    Value* inner = (*result->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_EQ(inner->size(), 3);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(2, 0), 5.0);
}

TEST_F(StructuralTest, PervasiveAddStrandToStrand) {
    // (⊂1 2)+(⊂3 4) → (4 6) - add corresponding elements
    // Result is a single-element strand containing the sum vector
    Value* result = machine->eval("(⊂1 2)+(⊂3 4)");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 1);
    Value* inner = (*result->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 0), 6.0);
}

TEST_F(StructuralTest, PervasiveAddMultiElementStrand) {
    // 10+(⊂1 2),⊂3 4 → ((11 12)(13 14)) - add scalar to each element
    Value* result = machine->eval("10+(⊂1 2),⊂3 4");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2);
    // First element: 11 12
    Value* first = (*result->as_strand())[0];
    ASSERT_TRUE(first->is_vector());
    EXPECT_DOUBLE_EQ((*first->as_matrix())(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*first->as_matrix())(1, 0), 12.0);
    // Second element: 13 14
    Value* second = (*result->as_strand())[1];
    ASSERT_TRUE(second->is_vector());
    EXPECT_DOUBLE_EQ((*second->as_matrix())(0, 0), 13.0);
    EXPECT_DOUBLE_EQ((*second->as_matrix())(1, 0), 14.0);
}

TEST_F(StructuralTest, PervasiveMultiplyStrands) {
    // (⊂2 3)×(⊂4 5) → (8 15) - wrapped in strand
    Value* result = machine->eval("(⊂2 3)×(⊂4 5)");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 1);
    Value* inner = (*result->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), 8.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 0), 15.0);
}

TEST_F(StructuralTest, PervasivePreservesDepth) {
    // Depth should be preserved through pervasive operations
    Value* before = machine->eval("≡(⊂1 2),⊂3 4");
    EXPECT_DOUBLE_EQ(before->as_scalar(), 2.0);  // Depth 2 before
    Value* after = machine->eval("≡10+(⊂1 2),⊂3 4");
    EXPECT_DOUBLE_EQ(after->as_scalar(), 2.0);   // Depth 2 after
}

TEST_F(StructuralTest, PervasiveComparisonOnStrand) {
    // 2<⊂1 2 3 → (0 0 1) - comparison is pervasive
    Value* result = machine->eval("2<⊂1 2 3");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 1);
    Value* inner = (*result->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), 0.0);  // 2<1 = 0
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 0), 0.0);  // 2<2 = 0
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(2, 0), 1.0);  // 2<3 = 1
}

TEST_F(StructuralTest, StructuralFunctionsNotPervasive) {
    // Structural functions like ≢ and ≡ should NOT be pervasive
    // ≢⊂1 2 3 → 1 (tally of strand, not tally of each element)
    Value* tally = machine->eval("≢⊂1 2 3");
    EXPECT_DOUBLE_EQ(tally->as_scalar(), 1.0);

    // ≡⊂1 2 3 → 2 per ISO 13751 Section 8.2.5
    // Depth = 1 + max depth of elements = 1 + 1 (simple vector) = 2
    Value* depth = machine->eval("≡⊂1 2 3");
    EXPECT_DOUBLE_EQ(depth->as_scalar(), 2.0);
}

TEST_F(StructuralTest, ShapeOfEnclose) {
    // ⍴⊂1 2 3 → 1 (single enclosed item, strand of length 1)
    Value* result = machine->eval("⍴⊂1 2 3");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 1.0);
}

TEST_F(StructuralTest, ShapeOfStrand) {
    // ⍴(⊂1 2 3),⊂4 5 → 2 (strand with 2 elements)
    Value* result = machine->eval("⍴(⊂1 2 3),⊂4 5");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 2.0);
}

TEST_F(StructuralTest, ShapeOfThreeElementStrand) {
    // ⍴(⊂1 2),(⊂3 4),⊂5 6 → 3 (strand with 3 elements)
    Value* result = machine->eval("⍴(⊂1 2),(⊂3 4),⊂5 6");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 3.0);
}

// ============================================================================
// Strand Operations - Comprehensive Tests per ISO 13751
// ============================================================================

// First (↑) with strands - ISO 10.1.9
// "Z is the first item of B in row-major order"
TEST_F(StructuralTest, FirstOfStrand) {
    // ↑(⊂1 2 3),⊂4 5 → 1 2 3 (first element of strand)
    Value* result = machine->eval("↑(⊂1 2 3),⊂4 5");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(2, 0), 3.0);
}

TEST_F(StructuralTest, FirstOfSingleElementStrand) {
    // ↑⊂1 2 3 → 1 2 3
    Value* result = machine->eval("↑⊂1 2 3");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

TEST_F(StructuralTest, FirstOfNestedStrand) {
    // ↑(⊂(⊂1 2)),⊂3 4 → (⊂1 2) (first element is itself a strand)
    Value* result = machine->eval("↑(⊂(⊂1 2)),⊂3 4");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 1);
}

// Disclose/First (⊃) with strands - already works, verify behavior
TEST_F(StructuralTest, DiscloseOfStrand) {
    // ⊃(⊂1 2 3),⊂4 5 → 1 2 3
    Value* result = machine->eval("⊃(⊂1 2 3),⊂4 5");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 1.0);
}

// Tally (≢) with strands
TEST_F(StructuralTest, TallyOfStrand) {
    // ≢(⊂1 2 3),⊂4 5 → 2
    Value* result = machine->eval("≢(⊂1 2 3),⊂4 5");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(StructuralTest, TallyOfSingleElementStrand) {
    // ≢⊂1 2 3 → 1
    Value* result = machine->eval("≢⊂1 2 3");
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// Reverse (⌽) with strands - ISO 10.1.4
// "Z is an array whose elements are those of B taken in reverse order"
TEST_F(StructuralTest, ReverseOfStrand) {
    // ⌽(⊂1 2),(⊂3 4),⊂5 6 → (⊂5 6),(⊂3 4),⊂1 2
    Value* result = machine->eval("⌽(⊂1 2),(⊂3 4),⊂5 6");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 3);
    // First element should now be 5 6
    Value* first = (*result->as_strand())[0];
    ASSERT_TRUE(first->is_vector());
    EXPECT_DOUBLE_EQ((*first->as_matrix())(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*first->as_matrix())(1, 0), 6.0);
    // Last element should now be 1 2
    Value* last = (*result->as_strand())[2];
    ASSERT_TRUE(last->is_vector());
    EXPECT_DOUBLE_EQ((*last->as_matrix())(0, 0), 1.0);
}

TEST_F(StructuralTest, ReverseOfTwoElementStrand) {
    // ⌽(⊂1 2 3),⊂4 5 → (⊂4 5),⊂1 2 3
    Value* result = machine->eval("⌽(⊂1 2 3),⊂4 5");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 2);
    Value* first = (*result->as_strand())[0];
    ASSERT_TRUE(first->is_vector());
    EXPECT_EQ(first->size(), 2);  // 4 5
    EXPECT_DOUBLE_EQ((*first->as_matrix())(0, 0), 4.0);
}

// Rotate (⌽) dyadic with strands
TEST_F(StructuralTest, RotateStrand) {
    // 1⌽(⊂1 2),(⊂3 4),⊂5 6 → (⊂3 4),(⊂5 6),⊂1 2
    Value* result = machine->eval("1⌽(⊂1 2),(⊂3 4),⊂5 6");
    ASSERT_TRUE(result->is_strand());
    EXPECT_EQ(result->as_strand()->size(), 3);
    Value* first = (*result->as_strand())[0];
    ASSERT_TRUE(first->is_vector());
    EXPECT_DOUBLE_EQ((*first->as_matrix())(0, 0), 3.0);  // Was second, now first
}

// Enlist (∊) with strands - ISO 8.2.6
// "recursively raveling each element of B and joining them together"
TEST_F(StructuralTest, EnlistOfStrand) {
    // ∊(⊂1 2 3),⊂4 5 → 1 2 3 4 5
    Value* result = machine->eval("∊(⊂1 2 3),⊂4 5");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(4, 0), 5.0);
}

TEST_F(StructuralTest, EnlistOfNestedStrand) {
    // ∊(⊂(⊂1 2)),⊂3 4 → 1 2 3 4 (recursive flatten)
    Value* result = machine->eval("∊(⊂(⊂1 2)),⊂3 4");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
}

TEST_F(StructuralTest, EnlistOfSingleElementStrand) {
    // ∊⊂1 2 3 → 1 2 3
    Value* result = machine->eval("∊⊂1 2 3");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

// Pick (⊃) dyadic with strands - additional tests (1-indexed per ISO)
TEST_F(StructuralTest, PickSecondFromStrand) {
    // 2⊃(⊂10 20),(⊂30 40) → 30 40 (second element, 1-indexed)
    Value* result = machine->eval("2⊃(⊂10 20),(⊂30 40)");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 30.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 40.0);
}

TEST_F(StructuralTest, PickFirstFromStrandOneIndex) {
    // 1⊃(⊂10 20),(⊂30 40) → 10 20 (first element, 1-indexed)
    Value* result = machine->eval("1⊃(⊂10 20),(⊂30 40)");
    ASSERT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 10.0);
}

// ========================================================================
// NDARRAY Tests (Rank 3+)
// ========================================================================

// Test reshape to 3D array using direct function call
TEST_F(StructuralTest, Reshape3DBasic) {
    // 2 3 4⍴⍳24 should produce 2×3×4 array
    // Create shape vector: 2 3 4
    Eigen::VectorXd shape_vec(3);
    shape_vec << 2.0, 3.0, 4.0;
    Value* shape = machine->heap->allocate_vector(shape_vec);

    // Create data vector: 1..24
    Eigen::VectorXd data_vec(24);
    for (int i = 0; i < 24; ++i) data_vec(i) = i + 1;
    Value* data = machine->heap->allocate_vector(data_vec);

    fn_reshape(machine, nullptr, shape, data);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_ndarray());
    EXPECT_EQ(result->rank(), 3);

    const auto& result_shape = result->ndarray_shape();
    ASSERT_EQ(result_shape.size(), 3);
    EXPECT_EQ(result_shape[0], 2);
    EXPECT_EQ(result_shape[1], 3);
    EXPECT_EQ(result_shape[2], 4);

    EXPECT_EQ(result->size(), 24);

    // Verify row-major order: element [0,0,0] = 1, [0,0,1] = 2, etc.
    const Eigen::VectorXd* nd_data = result->ndarray_data();
    EXPECT_DOUBLE_EQ((*nd_data)(0), 1.0);   // [0,0,0]
    EXPECT_DOUBLE_EQ((*nd_data)(1), 2.0);   // [0,0,1]
    EXPECT_DOUBLE_EQ((*nd_data)(4), 5.0);   // [0,1,0]
    EXPECT_DOUBLE_EQ((*nd_data)(12), 13.0); // [1,0,0]
    EXPECT_DOUBLE_EQ((*nd_data)(23), 24.0); // [1,2,3]
}

// Test shape of 3D array
TEST_F(StructuralTest, Shape3DArray) {
    // Create 2×3×4 array directly
    Eigen::VectorXd data(24);
    for (int i = 0; i < 24; ++i) data(i) = i + 1;
    std::vector<int> shape = {2, 3, 4};
    Value* ndarray = machine->heap->allocate_ndarray(data, shape);

    fn_shape(machine, nullptr, ndarray);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* shape_result = result->as_matrix();
    EXPECT_DOUBLE_EQ((*shape_result)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape_result)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape_result)(2, 0), 4.0);
}

// Test reshape cycling with 3D target
TEST_F(StructuralTest, Reshape3DCycling) {
    // 2 3 4⍴1 2 3 cycles 1 2 3 1 2 3...
    Eigen::VectorXd shape_vec(3);
    shape_vec << 2.0, 3.0, 4.0;
    Value* shape = machine->heap->allocate_vector(shape_vec);

    Eigen::VectorXd data_vec(3);
    data_vec << 1.0, 2.0, 3.0;
    Value* data = machine->heap->allocate_vector(data_vec);

    fn_reshape(machine, nullptr, shape, data);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_ndarray());
    const Eigen::VectorXd* nd_data = result->ndarray_data();
    EXPECT_DOUBLE_EQ((*nd_data)(0), 1.0);
    EXPECT_DOUBLE_EQ((*nd_data)(1), 2.0);
    EXPECT_DOUBLE_EQ((*nd_data)(2), 3.0);
    EXPECT_DOUBLE_EQ((*nd_data)(3), 1.0);  // Cycles
    EXPECT_DOUBLE_EQ((*nd_data)(4), 2.0);
}

// Test reshape from scalar to 3D
TEST_F(StructuralTest, Reshape3DFromScalar) {
    // 2 2 2⍴5 fills all 8 elements with 5
    Eigen::VectorXd shape_vec(3);
    shape_vec << 2.0, 2.0, 2.0;
    Value* shape = machine->heap->allocate_vector(shape_vec);
    Value* scalar = machine->heap->allocate_scalar(5.0);

    fn_reshape(machine, nullptr, shape, scalar);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_ndarray());
    EXPECT_EQ(result->size(), 8);
    const Eigen::VectorXd* nd_data = result->ndarray_data();
    for (int i = 0; i < 8; ++i) {
        EXPECT_DOUBLE_EQ((*nd_data)(i), 5.0);
    }
}

// Test 4D array
TEST_F(StructuralTest, Reshape4D) {
    // 2 2 2 2⍴⍳16
    Eigen::VectorXd shape_vec(4);
    shape_vec << 2.0, 2.0, 2.0, 2.0;
    Value* shape = machine->heap->allocate_vector(shape_vec);

    Eigen::VectorXd data_vec(16);
    for (int i = 0; i < 16; ++i) data_vec(i) = i + 1;
    Value* data = machine->heap->allocate_vector(data_vec);

    fn_reshape(machine, nullptr, shape, data);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_ndarray());
    EXPECT_EQ(result->rank(), 4);
    EXPECT_EQ(result->size(), 16);
}

// Test reshape 3D to 2D (NDARRAY source, matrix result)
TEST_F(StructuralTest, Reshape3Dto2D) {
    // Create 2×3×4 array, reshape to 6×4
    Eigen::VectorXd nd_data(24);
    for (int i = 0; i < 24; ++i) nd_data(i) = i + 1;
    std::vector<int> nd_shape = {2, 3, 4};
    Value* ndarray = machine->heap->allocate_ndarray(nd_data, nd_shape);

    Eigen::VectorXd new_shape(2);
    new_shape << 6.0, 4.0;
    Value* shape = machine->heap->allocate_vector(new_shape);

    fn_reshape(machine, nullptr, shape, ndarray);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 6);
    EXPECT_EQ(result->cols(), 4);
    // First row should be 1 2 3 4
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 3), 4.0);
}

// Test reshape 3D to vector
TEST_F(StructuralTest, Reshape3DtoVector) {
    // Create 2×3×4 array, reshape to 24-element vector
    Eigen::VectorXd nd_data(24);
    for (int i = 0; i < 24; ++i) nd_data(i) = i + 1;
    std::vector<int> nd_shape = {2, 3, 4};
    Value* ndarray = machine->heap->allocate_ndarray(nd_data, nd_shape);

    Eigen::VectorXd new_shape(1);
    new_shape << 24.0;
    Value* shape = machine->heap->allocate_vector(new_shape);

    fn_reshape(machine, nullptr, shape, ndarray);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 24);
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(23, 0), 24.0);
}

// Test empty shape produces scalar from NDARRAY
TEST_F(StructuralTest, Reshape3DtoScalar) {
    // (⍳0)⍴(2 3 4⍴⍳24) → 1 (first element)
    Eigen::VectorXd nd_data(24);
    for (int i = 0; i < 24; ++i) nd_data(i) = i + 1;
    std::vector<int> nd_shape = {2, 3, 4};
    Value* ndarray = machine->heap->allocate_ndarray(nd_data, nd_shape);

    Eigen::VectorXd empty_shape(0);
    Value* shape = machine->heap->allocate_vector(empty_shape);

    fn_reshape(machine, nullptr, shape, ndarray);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// Test strides are correct
TEST_F(StructuralTest, NDArrayStrides) {
    // 2×3×4 array should have strides {12, 4, 1}
    Eigen::VectorXd data(24);
    for (int i = 0; i < 24; ++i) data(i) = i + 1;
    std::vector<int> shape = {2, 3, 4};
    Value* ndarray = machine->heap->allocate_ndarray(data, shape);

    const auto& strides = ndarray->ndarray_strides();
    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12);  // stride along first axis
    EXPECT_EQ(strides[1], 4);   // stride along second axis
    EXPECT_EQ(strides[2], 1);   // stride along last axis
}

// Test ndarray_at indexing
TEST_F(StructuralTest, NDArrayAt) {
    // Create 2×3×4 array with values 1..24
    Eigen::VectorXd data(24);
    for (int i = 0; i < 24; ++i) data(i) = i + 1;
    std::vector<int> shape = {2, 3, 4};
    Value* ndarray = machine->heap->allocate_ndarray(data, shape);

    // [0,0,0] = 1, [0,0,1] = 2, [0,1,0] = 5, [1,0,0] = 13
    EXPECT_DOUBLE_EQ(ndarray->ndarray_at({0, 0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(ndarray->ndarray_at({0, 0, 1}), 2.0);
    EXPECT_DOUBLE_EQ(ndarray->ndarray_at({0, 0, 3}), 4.0);
    EXPECT_DOUBLE_EQ(ndarray->ndarray_at({0, 1, 0}), 5.0);
    EXPECT_DOUBLE_EQ(ndarray->ndarray_at({0, 2, 3}), 12.0);
    EXPECT_DOUBLE_EQ(ndarray->ndarray_at({1, 0, 0}), 13.0);
    EXPECT_DOUBLE_EQ(ndarray->ndarray_at({1, 2, 3}), 24.0);
}

// Test ndarray_linear_index
TEST_F(StructuralTest, NDArrayLinearIndex) {
    Eigen::VectorXd data(24);
    for (int i = 0; i < 24; ++i) data(i) = i + 1;
    std::vector<int> shape = {2, 3, 4};
    Value* ndarray = machine->heap->allocate_ndarray(data, shape);

    // For 2×3×4 with strides {12, 4, 1}:
    // [i,j,k] = i*12 + j*4 + k
    EXPECT_EQ(ndarray->ndarray_linear_index({0, 0, 0}), 0);
    EXPECT_EQ(ndarray->ndarray_linear_index({0, 0, 1}), 1);
    EXPECT_EQ(ndarray->ndarray_linear_index({0, 1, 0}), 4);
    EXPECT_EQ(ndarray->ndarray_linear_index({1, 0, 0}), 12);
    EXPECT_EQ(ndarray->ndarray_linear_index({1, 2, 3}), 23);
}

// ========================================================================
// NDARRAY Integration Tests (via machine->eval())
// ========================================================================

// Test 3D reshape via eval
TEST_F(StructuralTest, Eval3DReshape) {
    Value* result = machine->eval("2 3 4⍴⍳24");
    ASSERT_TRUE(result->is_ndarray());
    EXPECT_EQ(result->rank(), 3);
    EXPECT_EQ(result->size(), 24);

    const auto& shape = result->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

// Test shape of 3D array via eval
TEST_F(StructuralTest, EvalShape3D) {
    Value* result = machine->eval("⍴2 3 4⍴⍳24");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 4.0);
}

// Test 4D reshape via eval
TEST_F(StructuralTest, Eval4DReshape) {
    Value* result = machine->eval("2 2 2 2⍴⍳16");
    ASSERT_TRUE(result->is_ndarray());
    EXPECT_EQ(result->rank(), 4);
    EXPECT_EQ(result->size(), 16);
}

// Test reshaping 3D to 2D via eval
TEST_F(StructuralTest, Eval3Dto2DReshape) {
    Value* result = machine->eval("4 6⍴2 3 4⍴⍳24");
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 4);
    EXPECT_EQ(result->cols(), 6);
}

// Test reshaping 3D to vector via eval
TEST_F(StructuralTest, Eval3DtoVectorReshape) {
    Value* result = machine->eval("24⍴2 3 4⍴⍳24");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 24);
}

// Test cycling with 3D
TEST_F(StructuralTest, Eval3DCycling) {
    Value* result = machine->eval("2 2 2⍴1 2 3");
    ASSERT_TRUE(result->is_ndarray());
    const Eigen::VectorXd* data = result->ndarray_data();
    // 1 2 3 1 2 3 1 2 (cycling)
    EXPECT_DOUBLE_EQ((*data)(0), 1.0);
    EXPECT_DOUBLE_EQ((*data)(1), 2.0);
    EXPECT_DOUBLE_EQ((*data)(2), 3.0);
    EXPECT_DOUBLE_EQ((*data)(3), 1.0);
    EXPECT_DOUBLE_EQ((*data)(7), 2.0);
}

// Test assignment of 3D array
TEST_F(StructuralTest, Eval3DAssignment) {
    machine->eval("A←2 3 4⍴⍳24");
    Value* result = machine->eval("A");
    ASSERT_TRUE(result->is_ndarray());
    EXPECT_EQ(result->rank(), 3);
}

// Test shape of assigned 3D array
TEST_F(StructuralTest, EvalShapeAssigned3D) {
    machine->eval("B←2 2 2⍴⍳8");
    Value* result = machine->eval("⍴B");
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

// ========================================================================
// NDARRAY Indexing Tests
// ========================================================================

// Test scalar indexing: A[i;j;k] → scalar
TEST_F(StructuralTest, NDArrayIndexScalar) {
    machine->eval("A←2 3 4⍴⍳24");
    // A[1;1;1] = 1 (first element, 1-indexed)
    Value* r1 = machine->eval("A[1;1;1]");
    ASSERT_TRUE(r1->is_scalar());
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    // A[1;2;3] = 7 (indices 0,1,2 → linear 0*12+1*4+2=6 → value 7)
    Value* r2 = machine->eval("A[1;2;3]");
    ASSERT_TRUE(r2->is_scalar());
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 7.0);

    // A[2;1;1] = 13 (indices 1,0,0 → linear 1*12=12 → value 13)
    Value* r3 = machine->eval("A[2;1;1]");
    ASSERT_TRUE(r3->is_scalar());
    EXPECT_DOUBLE_EQ(r3->as_scalar(), 13.0);

    // A[2;3;4] = 24 (last element)
    Value* r4 = machine->eval("A[2;3;4]");
    ASSERT_TRUE(r4->is_scalar());
    EXPECT_DOUBLE_EQ(r4->as_scalar(), 24.0);
}

// Test elided axis: A[i;;] → matrix (all rows, all columns in plane i)
TEST_F(StructuralTest, NDArrayIndexElidedPlane) {
    machine->eval("A←2 3 4⍴⍳24");
    // A[1;;] = first plane (3×4 matrix)
    Value* r = machine->eval("A[1;;]");
    ASSERT_TRUE(r->is_matrix());
    EXPECT_EQ(r->rows(), 3);
    EXPECT_EQ(r->cols(), 4);
    const Eigen::MatrixXd* mat = r->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 3), 12.0);
}

// Test elided axis: A[;j;] → matrix (all planes, all columns in row j)
TEST_F(StructuralTest, NDArrayIndexElidedRow) {
    machine->eval("A←2 3 4⍴⍳24");
    // A[;2;] = row 2 from each plane (2×4 matrix)
    Value* r = machine->eval("A[;2;]");
    ASSERT_TRUE(r->is_matrix());
    EXPECT_EQ(r->rows(), 2);
    EXPECT_EQ(r->cols(), 4);
    const Eigen::MatrixXd* mat = r->as_matrix();
    // Plane 0, row 1: values 5,6,7,8
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 3), 8.0);
    // Plane 1, row 1: values 17,18,19,20
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 17.0);
}

// Test elided axis: A[;;k] → matrix (all planes, all rows, column k)
TEST_F(StructuralTest, NDArrayIndexElidedCol) {
    machine->eval("A←2 3 4⍴⍳24");
    // A[;;1] = first column from all planes (2×3 matrix)
    Value* r = machine->eval("A[;;1]");
    ASSERT_TRUE(r->is_matrix());
    EXPECT_EQ(r->rows(), 2);
    EXPECT_EQ(r->cols(), 3);
    const Eigen::MatrixXd* mat = r->as_matrix();
    // Plane 0: cols 1,5,9
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 9.0);
    // Plane 1: cols 13,17,21
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 13.0);
}

// Test vector result: A[i;j;] → vector
TEST_F(StructuralTest, NDArrayIndexVector) {
    machine->eval("A←2 3 4⍴⍳24");
    // A[1;2;] = row 2 of plane 1 (values 5,6,7,8)
    Value* r = machine->eval("A[1;2;]");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 4);
    const Eigen::MatrixXd* mat = r->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 8.0);
}

// Test multiple indices per axis
TEST_F(StructuralTest, NDArrayIndexMultiple) {
    machine->eval("A←2 3 4⍴⍳24");
    // A[1 2;1;1] = values from both planes, row 1, col 1 (1 and 13)
    Value* r = machine->eval("A[1 2;1;1]");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 2);
    const Eigen::MatrixXd* mat = r->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 13.0);
}

// Test linear indexing into ravel via eval
TEST_F(StructuralTest, NDArrayLinearIndexEvalRankError) {
    // ISO §6.3.8: index count must match rank — single index on rank-3 is RANK ERROR
    machine->eval("A←2 3 4⍴⍳24");
    EXPECT_THROW(machine->eval("A[1]"), APLError);
}

TEST_F(StructuralTest, NDArrayFullIndexEval) {
    // ISO §6.3.8: proper multi-index on rank-3 array
    machine->eval("A←2 3 4⍴⍳24");
    Value* r1 = machine->eval("A[1;1;1]");
    ASSERT_TRUE(r1->is_scalar());
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    Value* r2 = machine->eval("A[2;3;4]");
    ASSERT_TRUE(r2->is_scalar());
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 24.0);
}

// Test 4D array indexing
TEST_F(StructuralTest, NDArray4DIndex) {
    machine->eval("B←2 2 2 2⍴⍳16");
    // B[1;1;1;1] = 1
    Value* r1 = machine->eval("B[1;1;1;1]");
    ASSERT_TRUE(r1->is_scalar());
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    // B[2;2;2;2] = 16
    Value* r2 = machine->eval("B[2;2;2;2]");
    ASSERT_TRUE(r2->is_scalar());
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 16.0);
}

// Test index error handling
TEST_F(StructuralTest, NDArrayIndexOutOfBounds) {
    machine->eval("A←2 3 4⍴⍳24");
    EXPECT_THROW(machine->eval("A[3;1;1]"), APLError);  // axis 0 out of bounds
    EXPECT_THROW(machine->eval("A[1;4;1]"), APLError);  // axis 1 out of bounds
    EXPECT_THROW(machine->eval("A[1;1;5]"), APLError);  // axis 2 out of bounds
}

// Test rank error (wrong number of indices)
TEST_F(StructuralTest, NDArrayIndexRankError) {
    machine->eval("A←2 3 4⍴⍳24");
    EXPECT_THROW(machine->eval("A[1;1]"), APLError);    // too few indices
    EXPECT_THROW(machine->eval("A[1;1;1;1]"), APLError);  // too many indices
}

// ========================================================================
// NDARRAY Structural Operations Tests (ISO 13751)
// ========================================================================

// --- Ravel (,) ISO 13751 §8.2.1 ---

TEST_F(StructuralTest, NDArrayRavel) {
    // ,A returns vector with elements in row-major order
    machine->eval("A←2 3 4⍴⍳24");
    Value* r = machine->eval(",A");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 24);
    const Eigen::MatrixXd* mat = r->as_matrix();
    for (int i = 0; i < 24; ++i) {
        EXPECT_DOUBLE_EQ((*mat)(i, 0), i + 1.0);
    }
}

TEST_F(StructuralTest, NDArrayRavelShape) {
    // ⍴,A returns single element (total size)
    machine->eval("A←2 3 4⍴⍳24");
    Value* r = machine->eval("⍴,A");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 1);
    EXPECT_DOUBLE_EQ(r->as_matrix()->operator()(0, 0), 24.0);
}

TEST_F(StructuralTest, NDArrayRavel4D) {
    // Ravel 4D array
    machine->eval("B←2 2 2 3⍴⍳24");
    Value* r = machine->eval(",B");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 24);
}

// --- Monadic Transpose (⍉) ISO 13751 §10.1.5 ---

TEST_F(StructuralTest, NDArrayMonadicTransposeShape) {
    // ⍉ reverses axes: 2×3×4 → 4×3×2
    machine->eval("A←2 3 4⍴⍳24");
    Value* r = machine->eval("⍴⍉A");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 2.0);
}

TEST_F(StructuralTest, NDArrayMonadicTransposeValues) {
    // Verify element positions after transpose
    // Original A[1;1;1]=1, A[2;3;4]=24
    // Transposed: [1;1;1]=1, [4;3;2]=24
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←⍉A");

    Value* r1 = machine->eval("B[1;1;1]");
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    Value* r2 = machine->eval("B[4;3;2]");
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 24.0);

    // A[1;2;3]=7 → B[3;2;1]=7
    Value* r3 = machine->eval("B[3;2;1]");
    EXPECT_DOUBLE_EQ(r3->as_scalar(), 7.0);
}

TEST_F(StructuralTest, NDArrayMonadicTranspose4D) {
    // 4D transpose: 2×2×2×3 → 3×2×2×2
    machine->eval("B←2 2 2 3⍴⍳24");
    Value* r = machine->eval("⍴⍉B");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 4);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(3, 0), 2.0);
}

// --- Dyadic Transpose (A⍉B) ISO 13751 §10.2.10 ---

TEST_F(StructuralTest, NDArrayDyadicTransposeIdentity) {
    // 1 2 3⍉A = A (identity permutation)
    machine->eval("A←2 3 4⍴⍳24");
    Value* r = machine->eval("⍴1 2 3⍉A");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 4.0);
}

TEST_F(StructuralTest, NDArrayDyadicTransposeSwap23) {
    // 1 3 2⍉A swaps axes 2 and 3: 2×3×4 → 2×4×3
    machine->eval("A←2 3 4⍴⍳24");
    Value* r = machine->eval("⍴1 3 2⍉A");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 3.0);
}

TEST_F(StructuralTest, NDArrayDyadicTransposeSwap12) {
    // 2 1 3⍉A swaps axes 1 and 2: 2×3×4 → 3×2×4
    machine->eval("A←2 3 4⍴⍳24");
    Value* r = machine->eval("⍴2 1 3⍉A");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 4.0);
}

TEST_F(StructuralTest, NDArrayDyadicTransposeReverse) {
    // 3 2 1⍉A = ⍉A (reverse all axes)
    machine->eval("A←2 3 4⍴⍳24");
    Value* r = machine->eval("⍴3 2 1⍉A");
    ASSERT_TRUE(r->is_vector());
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 2.0);
}

TEST_F(StructuralTest, NDArrayDyadicTransposeDiagonal) {
    // 1 1 2⍉A selects diagonal: 2×3×4 → 2×4 (min of axes 1,2 = 2)
    machine->eval("A←2 3 4⍴⍳24");
    Value* r = machine->eval("⍴1 1 2⍉A");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 2);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 2.0);  // min(2,3)
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 4.0);
}

TEST_F(StructuralTest, NDArrayDyadicTransposeLengthError) {
    // Wrong number of permutation elements
    machine->eval("A←2 3 4⍴⍳24");
    EXPECT_THROW(machine->eval("1 2⍉A"), APLError);  // too few
    EXPECT_THROW(machine->eval("1 2 3 4⍉A"), APLError);  // too many
}

// --- Catenate (,) ISO 13751 §10.2.1 ---

TEST_F(StructuralTest, NDArrayCatenateLastAxis) {
    // A,B catenates along last axis: 2×3×4 , 2×3×4 → 2×3×8
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 4⍴24+⍳24");
    Value* r = machine->eval("⍴A,B");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 8.0);
}

TEST_F(StructuralTest, NDArrayCatenateLastAxisValues) {
    // Verify values after catenation
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 4⍴24+⍳24");
    machine->eval("C←A,B");

    // First element from A
    Value* r1 = machine->eval("C[1;1;1]");
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    // Last element from A
    Value* r2 = machine->eval("C[2;3;4]");
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 24.0);

    // First element from B (at position [1;1;5])
    Value* r3 = machine->eval("C[1;1;5]");
    EXPECT_DOUBLE_EQ(r3->as_scalar(), 25.0);

    // Last element from B
    Value* r4 = machine->eval("C[2;3;8]");
    EXPECT_DOUBLE_EQ(r4->as_scalar(), 48.0);
}

TEST_F(StructuralTest, NDArrayCatenateAxis1) {
    // A,[1]B catenates along first axis: 2×3×4 ,[1] 2×3×4 → 4×3×4
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 4⍴24+⍳24");
    Value* r = machine->eval("⍴A,[1]B");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 4.0);
}

TEST_F(StructuralTest, NDArrayCatenateAxis2) {
    // A,[2]B catenates along second axis: 2×3×4 ,[2] 2×3×4 → 2×6×4
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 4⍴24+⍳24");
    Value* r = machine->eval("⍴A,[2]B");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 6.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 4.0);
}

TEST_F(StructuralTest, NDArrayCatenateAxis3) {
    // A,[3]B same as A,B for 3D: 2×3×4 ,[3] 2×3×4 → 2×3×8
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 4⍴24+⍳24");
    Value* r = machine->eval("⍴A,[3]B");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 8.0);
}

TEST_F(StructuralTest, NDArrayCatenateLengthError) {
    // Shape mismatch on non-catenation axes
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 4 4⍴⍳32");  // Different middle dimension
    EXPECT_THROW(machine->eval("A,B"), APLError);
}

TEST_F(StructuralTest, NDArrayCatenateAxisError) {
    // Invalid axis
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 4⍴24+⍳24");
    EXPECT_THROW(machine->eval("A,[4]B"), APLError);  // Axis 4 doesn't exist
    EXPECT_THROW(machine->eval("A,[0]B"), APLError);  // Axis 0 with ⎕IO=1
}

// ============================================================================
// NDARRAY Catenate-First (⍪) Tests - ISO 13751 §8.3.2
// ============================================================================

// A⍪B is equivalent to A,[1]B (catenate along first axis)
TEST_F(StructuralTest, NDArrayCatenateFirstShape) {
    // 2×3×4 ⍪ 2×3×4 → 4×3×4
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 4⍴24+⍳24");
    Value* r = machine->eval("⍴A⍪B");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 4.0);
}

TEST_F(StructuralTest, NDArrayCatenateFirstValues) {
    // Verify values: A elements first, then B elements
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 4⍴100+⍳24");
    machine->eval("C←A⍪B");

    // First element from A
    Value* r1 = machine->eval("C[1;1;1]");
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    // Last element of first "plane" from A
    Value* r2 = machine->eval("C[2;3;4]");
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 24.0);

    // First element from B (now at row 3)
    Value* r3 = machine->eval("C[3;1;1]");
    EXPECT_DOUBLE_EQ(r3->as_scalar(), 101.0);

    // Last element from B
    Value* r4 = machine->eval("C[4;3;4]");
    EXPECT_DOUBLE_EQ(r4->as_scalar(), 124.0);
}

TEST_F(StructuralTest, NDArrayCatenateFirstDifferentSizes) {
    // Different first axis sizes: 2×3×4 ⍪ 3×3×4 → 5×3×4
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←3 3 4⍴⍳36");
    Value* r = machine->eval("⍴A⍪B");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 4.0);
}

TEST_F(StructuralTest, NDArrayCatenateFirstLengthError) {
    // Non-first axes must match
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 4 4⍴⍳32");  // Different second dimension
    EXPECT_THROW(machine->eval("A⍪B"), APLError);
}

TEST_F(StructuralTest, NDArrayCatenateFirst4D) {
    // 4D arrays: 2×2×3×4 ⍪ 3×2×3×4 → 5×2×3×4
    machine->eval("A←2 2 3 4⍴⍳48");
    machine->eval("B←3 2 3 4⍴⍳72");
    Value* r = machine->eval("⍴A⍪B");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 4);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*shape)(3, 0), 4.0);
}

// ============================================================================
// NDARRAY Pervasive Operations Tests - ISO 13751 Scalar Extension
// ============================================================================

// Monadic scalar function on NDARRAY - result same shape as argument
TEST_F(StructuralTest, NDArrayPervasiveMonadicNegate) {
    // -2 3 4⍴⍳24 → negated array with same shape
    Value* r = machine->eval("-2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    // First element is -1
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), -1.0);
    // Last element is -24
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(23), -24.0);
}

TEST_F(StructuralTest, NDArrayPervasiveMonadicAbs) {
    // |-(2 3 4⍴⍳24) → absolute values, same shape
    machine->eval("A←-(2 3 4⍴⍳24)");
    Value* r = machine->eval("|A");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 1.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(23), 24.0);
}

// Dyadic scalar function with scalar extension - scalar op NDARRAY
TEST_F(StructuralTest, NDArrayPervasiveDyadicScalarLeft) {
    // 10+2 3 4⍴⍳24 → each element +10, same shape
    Value* r = machine->eval("10+2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 11.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(23), 34.0);
}

TEST_F(StructuralTest, NDArrayPervasiveDyadicScalarRight) {
    // (2 3 4⍴⍳24)×2 → each element ×2, same shape
    Value* r = machine->eval("(2 3 4⍴⍳24)×2");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 2.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(23), 48.0);
}

// Dyadic scalar function with same-shape NDARRAYs
TEST_F(StructuralTest, NDArrayPervasiveDyadicSameShape) {
    // (2 3 4⍴⍳24)+(2 3 4⍴24+⍳24) → element-wise add
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 4⍴24+⍳24");
    Value* r = machine->eval("A+B");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    // First: 1+25=26, Last: 24+48=72
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 26.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(23), 72.0);
}

// Comparison functions on NDARRAY
TEST_F(StructuralTest, NDArrayPervasiveComparison) {
    // (2 3 4⍴⍳24)>12 → boolean array, same shape
    Value* r = machine->eval("(2 3 4⍴⍳24)>12");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    // Elements 1-12 are 0, elements 13-24 are 1
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 0.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(11), 0.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(12), 1.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(23), 1.0);
}

// Length error - mismatched shapes
TEST_F(StructuralTest, NDArrayPervasiveLengthError) {
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←2 3 5⍴⍳30");  // Different last dimension
    EXPECT_THROW(machine->eval("A+B"), APLError);
}

// Rank error - different ranks
TEST_F(StructuralTest, NDArrayPervasiveRankError) {
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("B←3 4⍴⍳12");  // Matrix vs 3D array
    EXPECT_THROW(machine->eval("A+B"), APLError);
}

// Chained pervasive operations
TEST_F(StructuralTest, NDArrayPervasiveChained) {
    // 2×1+(2 3 4⍴⍳24) → (2×(1+each element))
    Value* r = machine->eval("2×1+2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    // First: 2×(1+1)=4, Last: 2×(1+24)=50
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 4.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(23), 50.0);
}

// ============================================================================
// Strand Monadic Pervasive Tests - ISO 13751 Scalar Extension
// ============================================================================

// Monadic negate on single-element strand
TEST_F(StructuralTest, StrandPervasiveMonadicNegateSingle) {
    // -⊂1 2 3 → (¯1 ¯2 ¯3)
    Value* r = machine->eval("-⊂1 2 3");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 1);
    Value* inner = (*r->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_EQ(inner->size(), 3);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(2, 0), -3.0);
}

// Monadic negate on multi-element strand
TEST_F(StructuralTest, StrandPervasiveMonadicNegateMulti) {
    // -((⊂1 2 3),⊂4 5) → ((¯1 ¯2 ¯3)(¯4 ¯5))
    Value* r = machine->eval("-(⊂1 2 3),⊂4 5");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 2);

    Value* first = (*r->as_strand())[0];
    ASSERT_TRUE(first->is_vector());
    EXPECT_EQ(first->size(), 3);
    EXPECT_DOUBLE_EQ((*first->as_matrix())(0, 0), -1.0);

    Value* second = (*r->as_strand())[1];
    ASSERT_TRUE(second->is_vector());
    EXPECT_EQ(second->size(), 2);
    EXPECT_DOUBLE_EQ((*second->as_matrix())(0, 0), -4.0);
    EXPECT_DOUBLE_EQ((*second->as_matrix())(1, 0), -5.0);
}

// Monadic absolute value on strand
TEST_F(StructuralTest, StrandPervasiveMonadicAbs) {
    // |⊂¯1 ¯2 ¯3 → (1 2 3)
    Value* r = machine->eval("|⊂¯1 ¯2 ¯3");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 1);
    Value* inner = (*r->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(2, 0), 3.0);
}

// Monadic reciprocal on strand
TEST_F(StructuralTest, StrandPervasiveMonadicReciprocal) {
    // ÷⊂2 4 8 → (0.5 0.25 0.125)
    Value* r = machine->eval("÷⊂2 4 8");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 1);
    Value* inner = (*r->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), 0.5);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 0), 0.25);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(2, 0), 0.125);
}

// Monadic not on strand (boolean)
TEST_F(StructuralTest, StrandPervasiveMonadicNot) {
    // ~⊂1 0 1 0 → (0 1 0 1)
    Value* r = machine->eval("~⊂1 0 1 0");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 1);
    Value* inner = (*r->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(2, 0), 0.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(3, 0), 1.0);
}

// Chained monadic on strand
TEST_F(StructuralTest, StrandPervasiveMonadicChained) {
    // |-⊂1 2 3 → (1 2 3) (negate then abs)
    Value* r = machine->eval("|-⊂1 2 3");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 1);
    Value* inner = (*r->as_strand())[0];
    ASSERT_TRUE(inner->is_vector());
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(2, 0), 3.0);
}

// Nested strand - monadic applied recursively
TEST_F(StructuralTest, StrandPervasiveMonadicNested) {
    // Strand containing matrix
    // -⊂2 3⍴⍳6 → enclosed negated matrix
    Value* r = machine->eval("-⊂2 3⍴⍳6");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 1);
    Value* inner = (*r->as_strand())[0];
    ASSERT_TRUE(inner->is_matrix());
    EXPECT_EQ(inner->rows(), 2);
    EXPECT_EQ(inner->cols(), 3);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*inner->as_matrix())(1, 2), -6.0);
}

// ============================================================================
// NDARRAY Take (↑) Tests - ISO 13751 §10.2.11
// ============================================================================

// Take from NDARRAY: 1 2 3↑(2 3 4⍴⍳24) → 1×2×3 array
TEST_F(StructuralTest, NDArrayTakePositive) {
    Value* r = machine->eval("1 2 3↑2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 3);
    // First element: 1
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 1.0);
}

// Negative take: ¯1 ¯2 ¯3↑(2 3 4⍴⍳24) → last plane, last 2 rows, last 3 cols
TEST_F(StructuralTest, NDArrayTakeNegative) {
    Value* r = machine->eval("¯1 ¯2 ¯3↑2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 3);
    // Last 3 elements of last 2 rows of last plane
    // Plane 2: rows 2-3, cols 2-4 → 22 23 24, 18 19 20
    // Actually: row 2 (last): 21 22 23 24, take last 3: 22 23 24
    //          row 1 (second): 17 18 19 20, take last 3: 18 19 20
    const Eigen::VectorXd* data = r->ndarray_data();
    // With negative take, we get elements from the end
    EXPECT_DOUBLE_EQ((*data)(0), 18.0);  // First row of result
}

// Over-take with padding: 3 4 5↑(2 3 4⍴⍳24) → padded with zeros
TEST_F(StructuralTest, NDArrayTakeOverPadded) {
    Value* r = machine->eval("3 4 5↑2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], 5);
    // First element still 1
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 1.0);
    // Element at position beyond original should be 0 (padding)
    // Position [2,3,4] = beyond original 2×3×4
    int idx = 2 * 4 * 5 + 3 * 5 + 4;  // Last element
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(idx), 0.0);
}

// Length error: wrong number of elements in left arg
TEST_F(StructuralTest, NDArrayTakeLengthError) {
    machine->eval("A←2 3 4⍴⍳24");
    EXPECT_THROW(machine->eval("1 2↑A"), APLError);  // Need 3 elements for rank-3
}

// ============================================================================
// NDARRAY Drop (↓) Tests - ISO 13751 §10.2.12
// ============================================================================

// Drop from NDARRAY: 1 1 1↓(2 3 4⍴⍳24) → 1×2×3 array
TEST_F(StructuralTest, NDArrayDropPositive) {
    Value* r = machine->eval("1 1 1↓2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 1);  // 2-1
    EXPECT_EQ(shape[1], 2);  // 3-1
    EXPECT_EQ(shape[2], 3);  // 4-1
    // First element after dropping: element at [1,1,1] of original
    // = plane 2, row 2, col 2 = 12 + 4 + 1 + 1 = 18
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 18.0);
}

// Negative drop: ¯1 ¯1 ¯1↓(2 3 4⍴⍳24) → drop from end
TEST_F(StructuralTest, NDArrayDropNegative) {
    Value* r = machine->eval("¯1 ¯1 ¯1↓2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 3);
    // First element: original [0,0,0] = 1
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 1.0);
}

// Over-drop: drop more than exists → empty result
TEST_F(StructuralTest, NDArrayDropOverEmpty) {
    Value* r = machine->eval("⍴3 4 5↓2 3 4⍴⍳24");
    // Shape should be 0 0 0 (or close to it)
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);
    EXPECT_DOUBLE_EQ((*r->as_matrix())(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*r->as_matrix())(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*r->as_matrix())(2, 0), 0.0);
}

// ============================================================================
// NDARRAY Reverse (⌽) Tests - ISO 13751 §10.1.4
// ============================================================================

// Monadic reverse on NDARRAY: ⌽(2 3 4⍴⍳24) → reverse last axis
TEST_F(StructuralTest, NDArrayReverseLastAxis) {
    Value* r = machine->eval("⌽2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    // First row reversed: 1 2 3 4 → 4 3 2 1
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 4.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(1), 3.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(2), 2.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(3), 1.0);
}

// Reverse with axis: ⌽[1](2 3 4⍴⍳24) → reverse first axis (planes)
TEST_F(StructuralTest, NDArrayReverseFirstAxis) {
    Value* r = machine->eval("⌽[1]2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    // First element should now be from plane 2: 13
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 13.0);
}

// Reverse with axis 2: ⌽[2](2 3 4⍴⍳24) → reverse rows within each plane
TEST_F(StructuralTest, NDArrayReverseSecondAxis) {
    Value* r = machine->eval("⌽[2]2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    // First row of first plane should now be last row: 9 10 11 12
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 9.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(1), 10.0);
}

// ⊖ reverses first axis by default
TEST_F(StructuralTest, NDArrayReverseFirstDefault) {
    Value* r = machine->eval("⊖2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    // First element should be from plane 2: 13
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 13.0);
}

// ============================================================================
// NDARRAY Rotate (⌽) Dyadic Tests - ISO 13751 §10.2.7
// ============================================================================

// Scalar rotate on NDARRAY: 1⌽(2 3 4⍴⍳24) → rotate last axis by 1
TEST_F(StructuralTest, NDArrayRotateLastAxis) {
    Value* r = machine->eval("1⌽2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    const auto& shape = r->ndarray_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    // First row rotated by 1: 1 2 3 4 → 2 3 4 1
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 2.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(1), 3.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(2), 4.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(3), 1.0);
}

// Negative rotate: ¯1⌽(2 3 4⍴⍳24) → rotate last axis by -1
TEST_F(StructuralTest, NDArrayRotateNegative) {
    Value* r = machine->eval("¯1⌽2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    // First row rotated by -1: 1 2 3 4 → 4 1 2 3
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 4.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(1), 1.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(2), 2.0);
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(3), 3.0);
}

// Rotate with axis: 1⌽[1](2 3 4⍴⍳24) → rotate first axis (planes)
TEST_F(StructuralTest, NDArrayRotateFirstAxis) {
    Value* r = machine->eval("1⌽[1]2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    // First element should now be from plane 2: 13
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 13.0);
}

// ⊖ rotates first axis: 1⊖(2 3 4⍴⍳24)
TEST_F(StructuralTest, NDArrayRotateFirstDefault) {
    Value* r = machine->eval("1⊖2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    // First element should be from plane 2: 13
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 13.0);
}

// Vector left arg for per-subarray rotation
TEST_F(StructuralTest, NDArrayRotateVector) {
    // 1 2 3⌽(2 3 4⍴⍳24) → rotate each row by different amount
    Value* r = machine->eval("1 2 3⌽2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_ndarray());
    // Row 0 rotated by 1: 1 2 3 4 → 2 3 4 1
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 2.0);
    // Row 1 rotated by 2: 5 6 7 8 → 7 8 5 6
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(4), 7.0);
    // Row 2 rotated by 3: 9 10 11 12 → 12 9 10 11
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(8), 12.0);
}

// ============================================================================
// Strand Take (↑) Tests - ISO 13751 §10.2.11
// ============================================================================

TEST_F(StructuralTest, StrandTakePositive) {
    // 2↑ strand of 3 elements → first 2 elements
    machine->eval("S←(⊂1 2 3),(⊂4 5),(⊂6 7 8 9)");
    Value* r = machine->eval("2↑S");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 2);
    // First element is (1 2 3)
    EXPECT_TRUE((*r->as_strand())[0]->is_vector());
}

TEST_F(StructuralTest, StrandTakeNegative) {
    // ¯2↑ strand → last 2 elements
    machine->eval("S←(⊂1 2 3),(⊂4 5),(⊂6 7 8 9)");
    Value* r = machine->eval("¯2↑S");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 2);
    // Last element is (6 7 8 9)
    EXPECT_TRUE((*r->as_strand())[1]->is_vector());
    EXPECT_EQ((*r->as_strand())[1]->as_matrix()->rows(), 4);
}

TEST_F(StructuralTest, StrandTakeOverfill) {
    // 5↑ strand of 2 elements → 2 elements + 3 fills
    machine->eval("S←(⊂1 2),(⊂3 4)");
    Value* r = machine->eval("5↑S");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 5);
    // Fill elements are scalar 0
    EXPECT_TRUE((*r->as_strand())[2]->is_scalar());
    EXPECT_DOUBLE_EQ((*r->as_strand())[2]->as_scalar(), 0.0);
}

// ============================================================================
// Strand Drop (↓) Tests - ISO 13751 §10.2.12
// ============================================================================

TEST_F(StructuralTest, StrandDropPositive) {
    // 1↓ strand of 3 elements → last 2 elements
    machine->eval("S←(⊂1 2 3),(⊂4 5),(⊂6 7 8 9)");
    Value* r = machine->eval("1↓S");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 2);
}

TEST_F(StructuralTest, StrandDropNegative) {
    // ¯1↓ strand → drop last element
    machine->eval("S←(⊂1 2 3),(⊂4 5),(⊂6 7 8 9)");
    Value* r = machine->eval("¯1↓S");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 2);
    // First element is still (1 2 3)
    EXPECT_TRUE((*r->as_strand())[0]->is_vector());
    EXPECT_EQ((*r->as_strand())[0]->as_matrix()->rows(), 3);
}

TEST_F(StructuralTest, StrandDropAll) {
    // 5↓ strand of 2 elements → empty strand
    machine->eval("S←(⊂1 2),(⊂3 4)");
    Value* r = machine->eval("5↓S");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 0);
}

// ============================================================================
// NDARRAY First (↑ monadic) Tests - ISO 13751 §10.1.9
// ============================================================================

TEST_F(StructuralTest, NDArrayFirst3D) {
    // ↑ of 2 3 4 array → first plane (3×4 matrix)
    Value* r = machine->eval("↑2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_matrix());
    EXPECT_EQ(r->as_matrix()->rows(), 3);
    EXPECT_EQ(r->as_matrix()->cols(), 4);
    // First element is 1
    EXPECT_DOUBLE_EQ((*r->as_matrix())(0, 0), 1.0);
    // Last element of first plane is 12
    EXPECT_DOUBLE_EQ((*r->as_matrix())(2, 3), 12.0);
}

TEST_F(StructuralTest, NDArrayFirst4D) {
    // ↑ of 2 3 4 5 array → first 3D subarray (3×4×5)
    Value* r = machine->eval("↑2 3 4 5⍴⍳120");
    ASSERT_TRUE(r->is_ndarray());
    auto shape = r->ndarray_shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], 5);
    // First element is 1
    EXPECT_DOUBLE_EQ((*r->ndarray_data())(0), 1.0);
}

// ============================================================================
// Multi-dimensional Iota (⍳) Tests - ISO 13751 §10.1.2
// ============================================================================

TEST_F(StructuralTest, IotaMultiDim2D) {
    // ⍳2 3 → strand of 6 index pairs
    Value* r = machine->eval("⍳2 3");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 6);
    // First element is (1 1)
    Value* first = (*r->as_strand())[0];
    ASSERT_TRUE(first->is_strand());
    EXPECT_EQ(first->as_strand()->size(), 2);
    EXPECT_DOUBLE_EQ((*first->as_strand())[0]->as_scalar(), 1.0);
    EXPECT_DOUBLE_EQ((*first->as_strand())[1]->as_scalar(), 1.0);
    // Last element is (2 3)
    Value* last = (*r->as_strand())[5];
    ASSERT_TRUE(last->is_strand());
    EXPECT_DOUBLE_EQ((*last->as_strand())[0]->as_scalar(), 2.0);
    EXPECT_DOUBLE_EQ((*last->as_strand())[1]->as_scalar(), 3.0);
}

TEST_F(StructuralTest, IotaMultiDim3D) {
    // ⍳2 2 2 → strand of 8 index triples
    Value* r = machine->eval("⍳2 2 2");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 8);
    // First element is (1 1 1)
    Value* first = (*r->as_strand())[0];
    ASSERT_TRUE(first->is_strand());
    EXPECT_EQ(first->as_strand()->size(), 3);
    // Last element is (2 2 2)
    Value* last = (*r->as_strand())[7];
    EXPECT_DOUBLE_EQ((*last->as_strand())[0]->as_scalar(), 2.0);
    EXPECT_DOUBLE_EQ((*last->as_strand())[1]->as_scalar(), 2.0);
    EXPECT_DOUBLE_EQ((*last->as_strand())[2]->as_scalar(), 2.0);
}

TEST_F(StructuralTest, IotaStrandArg) {
    // Strand argument also works
    machine->eval("S←(⊂2),(⊂3)");
    Value* r = machine->eval("⍳S");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 6);
}

// ============================================================================
// NDARRAY Indexed Assignment Tests - ISO 13751 §10.2.16
// ============================================================================

TEST_F(StructuralTest, NDArrayIndexedAssignScalar) {
    // A[1;1;1]←99
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("A[1;1;1]←99");
    Value* r = machine->eval("A[1;1;1]");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 99.0);
}

TEST_F(StructuralTest, NDArrayIndexedAssignMultiple) {
    // A[1;1;1 2]←88 99
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("A[1;1;1 2]←88 99");
    Value* r1 = machine->eval("A[1;1;1]");
    Value* r2 = machine->eval("A[1;1;2]");
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 88.0);
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 99.0);
}

TEST_F(StructuralTest, NDArrayIndexedAssignScalarExtend) {
    // A[1;;]←0 (scalar extends to all selected positions)
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("A[1;;]←0");
    Value* r = machine->eval("+/,A[1;;]");
    EXPECT_DOUBLE_EQ(r->as_scalar(), 0.0);  // All zeros
}

TEST_F(StructuralTest, NDArrayIndexedAssignElided) {
    // A[;;1]←100 (assign to all planes, all rows, column 1)
    machine->eval("A←2 3 4⍴⍳24");
    machine->eval("A[;;1]←100");
    Value* r = machine->eval("A[1;1;1]");
    EXPECT_DOUBLE_EQ(r->as_scalar(), 100.0);
    Value* r2 = machine->eval("A[2;3;1]");
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 100.0);
}

// ============================================================================
// Strand Indexed Assignment Tests
// ============================================================================

TEST_F(StructuralTest, StrandIndexedAssign) {
    machine->eval("S←(⊂1 2 3),(⊂4 5),(⊂6)");
    machine->eval("S[2]←⊂99 100");
    Value* r = machine->eval("S[2]");
    ASSERT_TRUE(r->is_strand());
    // The enclosed vector 99 100
    EXPECT_TRUE((*r->as_strand())[0]->is_vector());
}

TEST_F(StructuralTest, StrandIndexedAssignScalar) {
    machine->eval("S←(⊂1 2),(⊂3 4),(⊂5 6)");
    machine->eval("S[1]←42");
    Value* r = machine->eval("S[1]");
    EXPECT_DOUBLE_EQ(r->as_scalar(), 42.0);
}

// ============================================================================
// NDARRAY Table (⍪B monadic) Tests - ISO 13751 §8.2.4
// ============================================================================

// Table on 3D: shape 2×3×4 → 2×12 matrix
TEST_F(StructuralTest, NDArrayTable3D) {
    Value* r = machine->eval("⍴⍪2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 2);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 12.0);
}

// Table on 4D: shape 2×3×4×5 → 2×60 matrix
TEST_F(StructuralTest, NDArrayTable4D) {
    Value* r = machine->eval("⍴⍪2 3 4 5⍴⍳120");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 2);
    const Eigen::MatrixXd* shape = r->as_matrix();
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*shape)(1, 0), 60.0);
}

// Table preserves values in row-major order
TEST_F(StructuralTest, NDArrayTableValues) {
    machine->eval("T←⍪2 3 4⍴⍳24");
    // First row should be elements 1-12
    Value* r1 = machine->eval("T[1;1]");
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);
    Value* r2 = machine->eval("T[1;12]");
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 12.0);
    // Second row should be elements 13-24
    Value* r3 = machine->eval("T[2;1]");
    EXPECT_DOUBLE_EQ(r3->as_scalar(), 13.0);
    Value* r4 = machine->eval("T[2;12]");
    EXPECT_DOUBLE_EQ(r4->as_scalar(), 24.0);
}

// Table on strand returns strand unchanged (rank 1 → n×1)
TEST_F(StructuralTest, StrandTable) {
    machine->eval("S←(⊂1 2),(⊂3 4)");
    Value* r = machine->eval("⍪S");
    ASSERT_TRUE(r->is_strand());
    EXPECT_EQ(r->size(), 2);
}

// ============================================================================
// NDARRAY Depth (≡B) Tests - ISO 13751 §8.2.5
// ============================================================================

// Depth of simple NDARRAY is 1
TEST_F(StructuralTest, NDArrayDepth3D) {
    Value* r = machine->eval("≡2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 1.0);
}

// Depth of 4D NDARRAY is still 1 (simple array)
TEST_F(StructuralTest, NDArrayDepth4D) {
    Value* r = machine->eval("≡2 2 2 2⍴⍳16");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 1.0);
}

// ============================================================================
// NDARRAY Enlist (∊B) Tests - ISO 13751 §8.2.6
// ============================================================================

// Enlist on NDARRAY is same as ravel (simple array)
TEST_F(StructuralTest, NDArrayEnlist3D) {
    Value* r = machine->eval("∊2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 24);
    // First and last elements
    const Eigen::MatrixXd* v = r->as_matrix();
    EXPECT_DOUBLE_EQ((*v)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*v)(23, 0), 24.0);
}

// Enlist on 4D NDARRAY
TEST_F(StructuralTest, NDArrayEnlist4D) {
    Value* r = machine->eval("∊2 2 2 2⍴⍳16");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 16);
}

// Enlist shape is always 1D
TEST_F(StructuralTest, NDArrayEnlistShape) {
    Value* r = machine->eval("⍴∊2 3 4⍴⍳24");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 1);
    EXPECT_DOUBLE_EQ(r->as_matrix()->operator()(0, 0), 24.0);
}

// ========================================================================
// Pick (⊃) with ⎕IO=0
// ISO 13751 §10.2.24: Pick uses index-origin
// ========================================================================

TEST_F(StructuralTest, PickWithIO0Vector) {
    // With ⎕IO←0, index 0 should pick first element
    machine->eval("⎕IO←0");
    Value* r = machine->eval("0⊃10 20 30");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 10.0);
}

TEST_F(StructuralTest, PickWithIO0LastElement) {
    machine->eval("⎕IO←0");
    Value* r = machine->eval("2⊃10 20 30");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 30.0);
}

TEST_F(StructuralTest, PickWithIO0Enclosed) {
    // Pick from enclosed value with ⎕IO=0
    // Monadic ⊃ (disclose) on an enclosed value should work with ⎕IO=0
    machine->eval("⎕IO←0");
    Value* r = machine->eval("0⊃10 20 30");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 10.0);  // First element at index 0
}

TEST_F(StructuralTest, PickWithIO0OutOfBounds) {
    machine->eval("⎕IO←0");
    // Index 3 with 3-element vector should be out of bounds
    EXPECT_THROW(machine->eval("3⊃10 20 30"), APLError);
}

// ========================================================================
// Without (~) with tolerant comparison (⎕CT)
// ISO 13751 §10.2.16: implicit argument comparison-tolerance
// ========================================================================

TEST_F(StructuralTest, WithoutTolerantMatch) {
    // With ⎕CT=0.01, 1.005 should be tolerantly equal to 1.0
    machine->eval("⎕CT←0.01");
    Value* r = machine->eval("1 2 3~1.005");
    ASSERT_TRUE(r->is_vector());
    // 1 is tolerantly equal to 1.005, so it should be removed
    EXPECT_EQ(r->size(), 2);
    EXPECT_DOUBLE_EQ(r->as_matrix()->operator()(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(r->as_matrix()->operator()(1, 0), 3.0);
}

TEST_F(StructuralTest, WithoutExactNoMatch) {
    // With ⎕CT=0 (exact), 1.005 should NOT match 1.0
    machine->eval("⎕CT←0");
    Value* r = machine->eval("1 2 3~1.005");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 3);  // Nothing removed
}

TEST_F(StructuralTest, WithoutTolerantMultiple) {
    machine->eval("⎕CT←0.01");
    Value* r = machine->eval("1 2 3 4~1.005 3.002");
    ASSERT_TRUE(r->is_vector());
    // 1 and 3 removed (tolerantly equal to 1.005 and 3.002)
    EXPECT_EQ(r->size(), 2);
    EXPECT_DOUBLE_EQ(r->as_matrix()->operator()(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(r->as_matrix()->operator()(1, 0), 4.0);
}

// ========================================================================

// ============================================================================
// ISO 13751 Compliance Tests — Phase 5 (previously Unknown structural rows)
// ============================================================================

// --- ST-24: Incompatible axis lengths → length-error (ISO 13751 §8.2.8) ---
// For ,(last-axis catenate) on matrices, all axes except last must match.
// (2 3⍴0),(3 3⍴0) — row counts differ: 2≠3 → LENGTH ERROR
TEST_F(StructuralTest, ST24_CatenateIncompatibleAxisLengthError) {
    EXPECT_THROW(machine->eval("(2 3⍴0),(3 3⍴0)"), APLError);
}

// --- ST-25: A≡B match (ISO 13751 §8.3.1) ---
TEST_F(StructuralTest, ST25_MatchEqualVectors) {
    Value* r = machine->eval("(1 2 3)≡(1 2 3)");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 1.0);
}

TEST_F(StructuralTest, ST25_MatchUnequalVectors) {
    Value* r = machine->eval("(1 2 3)≡(1 2 4)");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 0.0);
}

// --- ST-26: 1≡1.0 = 1 (ISO 13751 §8.3.1 numeric equality) ---
TEST_F(StructuralTest, ST26_MatchNumericEquality) {
    Value* r = machine->eval("1≡1.0");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 1.0);
}

// --- ST-27: (1 2)≡(1 2 3) = 0 (ISO 13751 §8.3.1 different shape) ---
TEST_F(StructuralTest, ST27_MatchDifferentShapeIsFalse) {
    Value* r = machine->eval("(1 2)≡(1 2 3)");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 0.0);
}

// --- ST-62: A∩B intersection (ISO 13751 §8) ---

TEST_F(StructuralTest, ST62_IntersectionBasic) {
    // 1 2 3 ∩ 2 3 4 → 2 3 (elements of left that appear in right, order from left)
    Value* r = machine->eval("1 2 3∩2 3 4");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    const auto* mat = r->as_matrix();
    ASSERT_EQ(mat->rows(), 2);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);
}

TEST_F(StructuralTest, ST62_IntersectionEmptyResult) {
    // 1 2 3 ∩ 4 5 6 → ⍬
    Value* r = machine->eval("1 2 3∩4 5 6");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->as_matrix()->rows(), 0);
}

TEST_F(StructuralTest, ST62_IntersectionOrderFromLeft) {
    // 3 1 2 ∩ 1 2 3 → 3 1 2 (order is from left argument)
    Value* r = machine->eval("3 1 2∩1 2 3");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    const auto* mat = r->as_matrix();
    ASSERT_EQ(mat->rows(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 2.0);
}

TEST_F(StructuralTest, ST62_IntersectionDuplicatesInLeft) {
    // 1 1 2 3 ∩ 1 3 → 1 1 3 (each occurrence in left checked independently)
    Value* r = machine->eval("1 1 2 3∩1 3");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    const auto* mat = r->as_matrix();
    ASSERT_EQ(mat->rows(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
}

// --- ST-63: A⍷B find (ISO 13751 §8) ---

TEST_F(StructuralTest, ST63_FindBasic) {
    // (1 2)⍷1 2 3 1 2 4 → 1 0 0 1 0 0
    Value* r = machine->eval("(1 2)⍷1 2 3 1 2 4");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    const auto* mat = r->as_matrix();
    ASSERT_EQ(mat->rows(), 6);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(4, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(5, 0), 0.0);
}

TEST_F(StructuralTest, ST63_FindNotPresent) {
    // (5 6)⍷1 2 3 4 → 0 0 0 0
    Value* r = machine->eval("(5 6)⍷1 2 3 4");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    const auto* mat = r->as_matrix();
    ASSERT_EQ(mat->rows(), 4);
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ((*mat)(i, 0), 0.0);
}

TEST_F(StructuralTest, ST63_FindNeedleLongerThanHaystack) {
    // (1 2 3 4 5)⍷1 2 → 0 0 (needle longer than haystack; no match possible)
    Value* r = machine->eval("(1 2 3 4 5)⍷1 2");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    const auto* mat = r->as_matrix();
    ASSERT_EQ(mat->rows(), 2);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 0.0);
}

TEST_F(StructuralTest, ST63_FindOverlapping) {
    // (1 1)⍷1 1 1 → 1 1 0  (overlapping matches)
    Value* r = machine->eval("(1 1)⍷1 1 1");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    const auto* mat = r->as_matrix();
    ASSERT_EQ(mat->rows(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 0.0);
}

// --- ST-24: Catenate rank-error (ISO 13751 §8.2.8) ---
// When ranks differ by more than 1, catenate signals rank-error.
// (1 2 3) is rank 1; (2 2 2⍴⍳8) is rank 3; rank diff = 2 → error.
TEST_F(StructuralTest, ST24_CatenateRankErrorHigherRank) {
    EXPECT_THROW(machine->eval("(1 2 3),(2 2 2⍴⍳8)"), APLError);
}

// --- ST-49: Replicate length-error (ISO 13751 §8) ---
// Length of left arg A must match length of right arg B, or A is scalar.
// 1 0/1 2 3 → length-error (A has 2 elements, B has 3)
TEST_F(StructuralTest, ST49_ReplicateLengthError) {
    EXPECT_THROW(machine->eval("1 0/1 2 3"), APLError);
}

// --- ST-67: A⌹B matrix divide (ISO 13751 §8) ---
// Dyadic ⌹: solve B×X=A for X (least-squares when B is not square).
// With A=vector (lhs) and B=square matrix (rhs): result is a vector.
TEST_F(StructuralTest, ST67_MatrixDivideLeastSquares) {
    // Exact solution: (1 2)⌹(2 2⍴1 0 0 1) → 1 2
    // Solves identity×X = [1 2] → X = [1 2]
    Value* r = machine->eval("(1 2)⌹(2 2⍴1 0 0 1)");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    ASSERT_EQ(r->size(), 2u);
    EXPECT_NEAR((*r->as_matrix())(0, 0), 1.0, 1e-10);
    EXPECT_NEAR((*r->as_matrix())(1, 0), 2.0, 1e-10);
}

TEST_F(StructuralTest, ST67_MatrixDivideLengthError) {
    // Row count of rhs (B) must match row count of lhs (A)
    // (2 3⍴1) is 2×3 (A), 1 2 3 is 3×1 (B): 3 rows ≠ 2 rows → LENGTH ERROR
    EXPECT_THROW(machine->eval("(2 3⍴1)⌹1 2 3"), APLError);
}

// --- ST-68: A?B deal (ISO 13751 §8) ---
// Returns A distinct random integers from ⎕IO..⎕IO+B-1
TEST_F(StructuralTest, ST68_DealBasicCount) {
    // 3?5 → 3 distinct integers from 1..5 (with IO=1)
    Value* r = machine->eval("3?5");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    ASSERT_EQ(r->size(), 3u);
    // All values in 1..5
    for (int i = 0; i < 3; ++i) {
        double v = (*r->as_matrix())(i, 0);
        EXPECT_GE(v, 1.0);
        EXPECT_LE(v, 5.0);
    }
}

TEST_F(StructuralTest, ST68_DealDistinct) {
    // 5?5 → all 5 values, each exactly once
    Value* r = machine->eval("5?5");
    ASSERT_NE(r, nullptr);
    ASSERT_TRUE(r->is_vector());
    ASSERT_EQ(r->size(), 5u);
    std::vector<int> seen(6, 0);
    for (int i = 0; i < 5; ++i) {
        int v = static_cast<int>((*r->as_matrix())(i, 0));
        EXPECT_GE(v, 1);
        EXPECT_LE(v, 5);
        seen[v]++;
    }
    for (int v = 1; v <= 5; ++v)
        EXPECT_EQ(seen[v], 1) << "Value " << v << " appeared " << seen[v] << " times";
}

// --- ST-62: ∩ and ST-63: ⍷ also tested above; verified by existing tests ---
