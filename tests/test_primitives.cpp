// Tests for arithmetic primitives

#include <gtest/gtest.h>
#include "primitives.h"
#include "value.h"
#include <cmath>

using namespace apl;

// Main function for running tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ============================================================================
// Dyadic Arithmetic Tests
// ============================================================================

TEST(PrimitivesTest, AddScalarScalar) {
    Value* a = Value::from_scalar(3.0);
    Value* b = Value::from_scalar(4.0);
    Value* result = fn_add(a, b);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

    delete a;
    delete b;
    delete result;
}

TEST(PrimitivesTest, AddScalarVector) {
    Value* scalar = Value::from_scalar(5.0);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_add(scalar, vec);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 8.0);

    delete scalar;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, AddVectorScalar) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = Value::from_vector(v);
    Value* scalar = Value::from_scalar(10.0);

    Value* result = fn_add(vec, scalar);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 13.0);

    delete vec;
    delete scalar;
    delete result;
}

TEST(PrimitivesTest, AddVectorVector) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(3);
    v2 << 4.0, 5.0, 6.0;

    Value* vec1 = Value::from_vector(v1);
    Value* vec2 = Value::from_vector(v2);

    Value* result = fn_add(vec1, vec2);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 9.0);

    delete vec1;
    delete vec2;
    delete result;
}

TEST(PrimitivesTest, SubtractScalars) {
    Value* a = Value::from_scalar(10.0);
    Value* b = Value::from_scalar(3.0);
    Value* result = fn_subtract(a, b);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

    delete a;
    delete b;
    delete result;
}

TEST(PrimitivesTest, MultiplyScalars) {
    Value* a = Value::from_scalar(3.0);
    Value* b = Value::from_scalar(4.0);
    Value* result = fn_multiply(a, b);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);

    delete a;
    delete b;
    delete result;
}

TEST(PrimitivesTest, MultiplyScalarVector) {
    Value* scalar = Value::from_scalar(2.0);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_multiply(scalar, vec);

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 6.0);

    delete scalar;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, DivideScalars) {
    Value* a = Value::from_scalar(12.0);
    Value* b = Value::from_scalar(3.0);
    Value* result = fn_divide(a, b);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);

    delete a;
    delete b;
    delete result;
}

TEST(PrimitivesTest, DivideByZeroError) {
    Value* a = Value::from_scalar(12.0);
    Value* b = Value::from_scalar(0.0);

    EXPECT_THROW(fn_divide(a, b), std::runtime_error);

    delete a;
    delete b;
}

TEST(PrimitivesTest, PowerScalars) {
    Value* a = Value::from_scalar(2.0);
    Value* b = Value::from_scalar(3.0);
    Value* result = fn_power(a, b);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);

    delete a;
    delete b;
    delete result;
}

// ============================================================================
// Monadic Arithmetic Tests
// ============================================================================

TEST(PrimitivesTest, IdentityScalar) {
    Value* a = Value::from_scalar(5.0);
    Value* result = fn_conjugate(a);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);

    delete a;
    delete result;
}

TEST(PrimitivesTest, NegateScalar) {
    Value* a = Value::from_scalar(5.0);
    Value* result = fn_negate(a);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);

    delete a;
    delete result;
}

TEST(PrimitivesTest, NegateVector) {
    Eigen::VectorXd v(3);
    v << 1.0, -2.0, 3.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_negate(vec);

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), -3.0);

    delete vec;
    delete result;
}

TEST(PrimitivesTest, SignPositive) {
    Value* a = Value::from_scalar(5.0);
    Value* result = fn_signum(a);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);

    delete a;
    delete result;
}

TEST(PrimitivesTest, SignNegative) {
    Value* a = Value::from_scalar(-5.0);
    Value* result = fn_signum(a);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -1.0);

    delete a;
    delete result;
}

TEST(PrimitivesTest, SignZero) {
    Value* a = Value::from_scalar(0.0);
    Value* result = fn_signum(a);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);

    delete a;
    delete result;
}

TEST(PrimitivesTest, ReciprocalScalar) {
    Value* a = Value::from_scalar(4.0);
    Value* result = fn_reciprocal(a);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.25);

    delete a;
    delete result;
}

TEST(PrimitivesTest, ReciprocalZeroError) {
    Value* a = Value::from_scalar(0.0);

    EXPECT_THROW(fn_reciprocal(a), std::runtime_error);

    delete a;
}

TEST(PrimitivesTest, ExponentialScalar) {
    Value* a = Value::from_scalar(1.0);
    Value* result = fn_exponential(a);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), M_E, 1e-10);

    delete a;
    delete result;
}

TEST(PrimitivesTest, ExponentialZero) {
    Value* a = Value::from_scalar(0.0);
    Value* result = fn_exponential(a);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);

    delete a;
    delete result;
}

// ============================================================================
// Matrix Tests
// ============================================================================

TEST(PrimitivesTest, AddMatrices) {
    Eigen::MatrixXd m1(2, 2);
    m1 << 1.0, 2.0,
          3.0, 4.0;
    Eigen::MatrixXd m2(2, 2);
    m2 << 5.0, 6.0,
          7.0, 8.0;

    Value* mat1 = Value::from_matrix(m1);
    Value* mat2 = Value::from_matrix(m2);

    Value* result = fn_add(mat1, mat2);

    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 8.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 10.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 12.0);

    delete mat1;
    delete mat2;
    delete result;
}

TEST(PrimitivesTest, MismatchedShapeError) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(2);
    v2 << 4.0, 5.0;

    Value* vec1 = Value::from_vector(v1);
    Value* vec2 = Value::from_vector(v2);

    EXPECT_THROW(fn_add(vec1, vec2), std::runtime_error);

    delete vec1;
    delete vec2;
}

// ============================================================================
// Array Operation Tests
// ============================================================================

TEST(PrimitivesTest, ShapeScalar) {
    Value* scalar = Value::from_scalar(5.0);
    Value* result = fn_shape(scalar);

    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);  // Empty shape for scalar

    delete scalar;
    delete result;
}

TEST(PrimitivesTest, ShapeVector) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = Value::from_vector(v);
    Value* result = fn_shape(vec);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* shape = result->as_matrix();
    EXPECT_EQ(shape->rows(), 1);
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 5.0);

    delete vec;
    delete result;
}

TEST(PrimitivesTest, ReshapeVector) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = Value::from_vector(v);

    Eigen::VectorXd new_shape(2);
    new_shape << 2.0, 3.0;
    Value* shape = Value::from_vector(new_shape);

    Value* result = fn_reshape(shape, vec);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 3.0);

    delete vec;
    delete shape;
    delete result;
}

TEST(PrimitivesTest, Ravel) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = Value::from_matrix(m);

    Value* result = fn_ravel(mat);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 6);
    // Column-major order
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 2.0);

    delete mat;
    delete result;
}

TEST(PrimitivesTest, Catenate) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(2);
    v2 << 4.0, 5.0;

    Value* vec1 = Value::from_vector(v1);
    Value* vec2 = Value::from_vector(v2);

    Value* result = fn_catenate(vec1, vec2);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);

    delete vec1;
    delete vec2;
    delete result;
}

TEST(PrimitivesTest, Transpose) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = Value::from_matrix(m);

    Value* result = fn_transpose(mat);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_EQ(res->cols(), 2);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 4.0);

    delete mat;
    delete result;
}

TEST(PrimitivesTest, Iota) {
    Value* n = Value::from_scalar(5.0);
    Value* result = fn_iota(n);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 4.0);

    delete n;
    delete result;
}

TEST(PrimitivesTest, Take) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = Value::from_vector(v);

    Value* count = Value::from_scalar(3.0);
    Value* result = fn_take(count, vec);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);

    delete vec;
    delete count;
    delete result;
}

TEST(PrimitivesTest, Drop) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = Value::from_vector(v);

    Value* count = Value::from_scalar(2.0);
    Value* result = fn_drop(count, vec);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);

    delete vec;
    delete count;
    delete result;
}

// ============================================================================
// Reduction and Scan Tests
// ============================================================================

TEST(PrimitivesTest, ReduceVector) {
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = Value::from_vector(v);
    Value* plus_fn = Value::from_function(&prim_plus);

    Value* result = fn_reduce(plus_fn, vec);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);  // 1+2+3+4

    delete vec;
    delete plus_fn;
    delete result;
}

TEST(PrimitivesTest, ReduceWithMultiply) {
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = Value::from_vector(v);
    Value* times_fn = Value::from_function(&prim_times);

    Value* result = fn_reduce(times_fn, vec);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 24.0);  // 1*2*3*4

    delete vec;
    delete times_fn;
    delete result;
}

TEST(PrimitivesTest, ReduceMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = Value::from_matrix(m);
    Value* plus_fn = Value::from_function(&prim_plus);

    Value* result = fn_reduce(plus_fn, mat);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 2);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 15.0);  // 4+5+6

    delete mat;
    delete plus_fn;
    delete result;
}

TEST(PrimitivesTest, ReduceFirstMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = Value::from_matrix(m);
    Value* plus_fn = Value::from_function(&prim_plus);

    Value* result = fn_reduce_first(plus_fn, mat);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 3);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 5.0);  // 1+4
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 7.0);  // 2+5
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 9.0);  // 3+6

    delete mat;
    delete plus_fn;
    delete result;
}

TEST(PrimitivesTest, ScanVector) {
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = Value::from_vector(v);
    Value* plus_fn = Value::from_function(&prim_plus);

    Value* result = fn_scan(plus_fn, vec);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    // Right-to-left cumulative: [10, 9, 7, 4]
    // 1+(2+(3+4)) = 1+9 = 10
    // 2+(3+4) = 9
    // 3+4 = 7
    // 4 = 4
    EXPECT_DOUBLE_EQ((*res)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 9.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 7.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 4.0);

    delete vec;
    delete plus_fn;
    delete result;
}

TEST(PrimitivesTest, ScanMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = Value::from_matrix(m);
    Value* plus_fn = Value::from_function(&prim_plus);

    Value* result = fn_scan(plus_fn, mat);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 2);
    EXPECT_EQ(res->cols(), 3);
    // Row 0: [6, 5, 3] = [1+2+3, 2+3, 3]
    // Row 1: [15, 11, 6] = [4+5+6, 5+6, 6]
    EXPECT_DOUBLE_EQ((*res)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 5.0);
    EXPECT_DOUBLE_EQ((*res)(0, 2), 3.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 15.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 11.0);
    EXPECT_DOUBLE_EQ((*res)(1, 2), 6.0);

    delete mat;
    delete plus_fn;
    delete result;
}
