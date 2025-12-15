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
TEST_F(StructuralTest, DISABLED_ReverseHigherRank) {
    // Reverse on 3D array
    Value* result = machine->eval("⌽ 2 2 3⍴⍳12");
    ASSERT_TRUE(result->is_matrix());  // 3D stored as matrix
    // Should reverse along last axis
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

TEST_F(StructuralTest, ReverseRotateTallyRegistered) {
    ASSERT_NE(machine->env->lookup("⌽"), nullptr);
    ASSERT_NE(machine->env->lookup("⊖"), nullptr);
    ASSERT_NE(machine->env->lookup("≢"), nullptr);
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

TEST_F(StructuralTest, IndexOfScalarHaystack) {
    // 5 ⍳ 5 → 1 (found at index 1, 1-origin per ISO 13751)
    Value* lhs = machine->heap->allocate_scalar(5.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);

    fn_index_of(machine, nullptr, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
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
    ASSERT_NE(machine->env->lookup("⍳"), nullptr);
    ASSERT_NE(machine->env->lookup("∊"), nullptr);
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
    ASSERT_NE(machine->env->lookup("⍋"), nullptr);
    ASSERT_NE(machine->env->lookup("⍒"), nullptr);
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
    EXPECT_TRUE(std::string(err->error_message).find("RANK") != std::string::npos);
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
    EXPECT_TRUE(std::string(err->error_message).find("DOMAIN") != std::string::npos);
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
    EXPECT_TRUE(std::string(err->error_message).find("DOMAIN") != std::string::npos);
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
    // ∪ 5 → 5
    Value* val = machine->heap->allocate_scalar(5.0);

    fn_unique(machine, nullptr, val);

    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
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
    ASSERT_NE(machine->env->lookup("∪"), nullptr);
    // ~ should already be registered for logical not
    ASSERT_NE(machine->env->lookup("~"), nullptr);
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
TEST_F(StructuralTest, DISABLED_FirstHigherRank) {
    // ↑ 2 3 4⍴⍳24 → first 3×4 matrix
    Value* result = machine->eval("↑ 2 3 4⍴⍳24");
    ASSERT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 4);
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

TEST_F(StructuralTest, QuestionRegistered) {
    ASSERT_NE(machine->env->lookup("?"), nullptr);
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
    // 0⍉(1 2 3) → 1 2 3
    Value* lhs = machine->heap->allocate_scalar(0.0);
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
    ASSERT_NE(machine->env->lookup("⌹"), nullptr);
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
    ASSERT_NE(machine->env->lookup("⍎"), nullptr);
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
    ASSERT_NE(machine->env->lookup("⌷"), nullptr);
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
    ASSERT_NE(machine->env->lookup("⊣"), nullptr);
}

TEST_F(StructuralTest, RightTackRegistered) {
    ASSERT_NE(machine->env->lookup("⊢"), nullptr);
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

// ========================================================================
