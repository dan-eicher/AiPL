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
