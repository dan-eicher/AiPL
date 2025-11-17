// Tests for arithmetic primitives

#include <gtest/gtest.h>
#include "primitives.h"
#include "value.h"
#include "environment.h"
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

// ============================================================================
// Environment and Binding Tests
// ============================================================================

TEST(PrimitivesTest, EnvironmentInit) {
    Environment env;
    init_global_environment(&env);

    // Verify arithmetic primitives are bound
    Value* plus = env.lookup("+");
    ASSERT_NE(plus, nullptr);
    ASSERT_TRUE(plus->is_function());
    EXPECT_EQ(plus->data.function, &prim_plus);

    Value* minus = env.lookup("-");
    ASSERT_NE(minus, nullptr);
    ASSERT_TRUE(minus->is_function());

    // Verify array operations are bound
    Value* rho = env.lookup("⍴");
    ASSERT_NE(rho, nullptr);
    ASSERT_TRUE(rho->is_function());

    Value* iota = env.lookup("⍳");
    ASSERT_NE(iota, nullptr);
    ASSERT_TRUE(iota->is_function());
}

TEST(PrimitivesTest, EnvironmentLookup) {
    Environment env;
    init_global_environment(&env);

    // Lookup and use a primitive from environment
    Value* plus = env.lookup("+");
    ASSERT_NE(plus, nullptr);

    // Use it to add two numbers
    Value* a = Value::from_scalar(3.0);
    Value* b = Value::from_scalar(4.0);
    PrimitiveFn* fn = plus->data.function;
    Value* result = fn->dyadic(a, b);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

    delete a;
    delete b;
    delete result;
}

TEST(PrimitivesTest, EnvironmentDefineAndUpdate) {
    Environment env;

    // Define a variable
    Value* x = Value::from_scalar(42.0);
    env.define("x", x);

    // Lookup the variable
    Value* lookup = env.lookup("x");
    ASSERT_NE(lookup, nullptr);
    EXPECT_DOUBLE_EQ(lookup->as_scalar(), 42.0);

    // Update the variable
    Value* y = Value::from_scalar(100.0);
    bool updated = env.update("x", y);
    ASSERT_TRUE(updated);

    Value* lookup2 = env.lookup("x");
    EXPECT_DOUBLE_EQ(lookup2->as_scalar(), 100.0);

    delete x;
    delete y;
}

TEST(PrimitivesTest, EnvironmentScoping) {
    Environment parent;
    Environment child(&parent);

    // Define in parent
    Value* x = Value::from_scalar(10.0);
    parent.define("x", x);

    // Define in child
    Value* y = Value::from_scalar(20.0);
    child.define("y", y);

    // Child can see both
    EXPECT_NE(child.lookup("x"), nullptr);
    EXPECT_NE(child.lookup("y"), nullptr);
    EXPECT_DOUBLE_EQ(child.lookup("x")->as_scalar(), 10.0);
    EXPECT_DOUBLE_EQ(child.lookup("y")->as_scalar(), 20.0);

    // Parent can only see its own
    EXPECT_NE(parent.lookup("x"), nullptr);
    EXPECT_EQ(parent.lookup("y"), nullptr);

    delete x;
    delete y;
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(PrimitivesTest, ErrorShapeMismatchVectorVector) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(5);
    v2 << 1.0, 2.0, 3.0, 4.0, 5.0;

    Value* vec1 = Value::from_vector(v1);
    Value* vec2 = Value::from_vector(v2);

    // Should throw on shape mismatch
    EXPECT_THROW(fn_add(vec1, vec2), std::runtime_error);
    EXPECT_THROW(fn_subtract(vec1, vec2), std::runtime_error);
    EXPECT_THROW(fn_multiply(vec1, vec2), std::runtime_error);
    EXPECT_THROW(fn_divide(vec1, vec2), std::runtime_error);

    delete vec1;
    delete vec2;
}

TEST(PrimitivesTest, ErrorShapeMismatchMatrixMatrix) {
    Eigen::MatrixXd m1(2, 3);
    m1.setConstant(1.0);
    Eigen::MatrixXd m2(3, 2);
    m2.setConstant(2.0);

    Value* mat1 = Value::from_matrix(m1);
    Value* mat2 = Value::from_matrix(m2);

    // Should throw on shape mismatch
    EXPECT_THROW(fn_add(mat1, mat2), std::runtime_error);
    EXPECT_THROW(fn_subtract(mat1, mat2), std::runtime_error);
    EXPECT_THROW(fn_multiply(mat1, mat2), std::runtime_error);

    delete mat1;
    delete mat2;
}

TEST(PrimitivesTest, ErrorDivideVectorByZeroVector) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(3);
    v2 << 1.0, 0.0, 2.0;  // Has a zero

    Value* vec1 = Value::from_vector(v1);
    Value* vec2 = Value::from_vector(v2);

    // Should throw on division by zero
    EXPECT_THROW(fn_divide(vec1, vec2), std::runtime_error);

    delete vec1;
    delete vec2;
}

TEST(PrimitivesTest, ErrorReciprocalVector) {
    Eigen::VectorXd v(3);
    v << 2.0, 0.0, 4.0;  // Has a zero

    Value* vec = Value::from_vector(v);

    // Should throw on reciprocal of zero
    EXPECT_THROW(fn_reciprocal(vec), std::runtime_error);

    delete vec;
}

TEST(PrimitivesTest, ErrorReshapeIncompatibleSize) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = Value::from_vector(v);

    // Try to reshape 6 elements into 2×4 (needs 8 elements)
    Eigen::VectorXd shape(2);
    shape << 2.0, 4.0;
    Value* shape_val = Value::from_vector(shape);

    EXPECT_THROW(fn_reshape(shape_val, vec), std::runtime_error);

    delete vec;
    delete shape_val;
}

TEST(PrimitivesTest, ErrorReshapeNonIntegerShape) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = Value::from_vector(v);

    // Shape with non-integer values
    Eigen::VectorXd shape(2);
    shape << 2.5, 3.0;
    Value* shape_val = Value::from_vector(shape);

    EXPECT_THROW(fn_reshape(shape_val, vec), std::runtime_error);

    delete vec;
    delete shape_val;
}

TEST(PrimitivesTest, ErrorReshapeNegativeShape) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = Value::from_vector(v);

    // Negative shape dimension
    Eigen::VectorXd shape(2);
    shape << -2.0, 3.0;
    Value* shape_val = Value::from_vector(shape);

    EXPECT_THROW(fn_reshape(shape_val, vec), std::runtime_error);

    delete vec;
    delete shape_val;
}

TEST(PrimitivesTest, ErrorIotaNonScalar) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = Value::from_vector(v);

    // Iota requires scalar argument
    EXPECT_THROW(fn_iota(vec), std::runtime_error);

    delete vec;
}

TEST(PrimitivesTest, ErrorIotaNegative) {
    Value* neg = Value::from_scalar(-5.0);

    // Negative iota doesn't make sense
    EXPECT_THROW(fn_iota(neg), std::runtime_error);

    delete neg;
}

TEST(PrimitivesTest, ErrorIotaNonInteger) {
    Value* frac = Value::from_scalar(3.5);

    // Fractional iota doesn't make sense
    EXPECT_THROW(fn_iota(frac), std::runtime_error);

    delete frac;
}

TEST(PrimitivesTest, ErrorTakeNonScalar) {
    Eigen::VectorXd n(2);
    n << 2.0, 3.0;
    Value* n_val = Value::from_vector(n);

    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = Value::from_vector(v);

    // Take requires scalar count
    EXPECT_THROW(fn_take(n_val, vec), std::runtime_error);

    delete n_val;
    delete vec;
}

TEST(PrimitivesTest, ErrorDropNonScalar) {
    Eigen::VectorXd n(2);
    n << 2.0, 3.0;
    Value* n_val = Value::from_vector(n);

    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = Value::from_vector(v);

    // Drop requires scalar count
    EXPECT_THROW(fn_drop(n_val, vec), std::runtime_error);

    delete n_val;
    delete vec;
}

TEST(PrimitivesTest, ErrorCatenateIncompatibleShapes) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Value* vec1 = Value::from_vector(v1);

    Eigen::MatrixXd m2(2, 2);
    m2.setConstant(5.0);
    Value* mat2 = Value::from_matrix(m2);

    // Cannot catenate vector with 2×2 matrix
    EXPECT_THROW(fn_catenate(vec1, mat2), std::runtime_error);

    delete vec1;
    delete mat2;
}

TEST(PrimitivesTest, ErrorReduceNonFunction) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = Value::from_vector(v);

    // First argument should be a function
    EXPECT_THROW(fn_reduce(vec, vec), std::runtime_error);

    delete vec;
}

TEST(PrimitivesTest, ErrorScanNonFunction) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = Value::from_vector(v);

    // First argument should be a function
    EXPECT_THROW(fn_scan(vec, vec), std::runtime_error);

    delete vec;
}

// ============================================================================
// Broadcasting Edge Cases
// ============================================================================

TEST(PrimitivesTest, BroadcastScalarMatrix) {
    Value* scalar = Value::from_scalar(5.0);
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = Value::from_matrix(m);

    Value* result = fn_add(scalar, mat);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 2);
    EXPECT_EQ(res_mat->cols(), 3);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 1), 7.0);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 2), 8.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 9.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 1), 10.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 2), 11.0);

    delete scalar;
    delete mat;
    delete result;
}

TEST(PrimitivesTest, BroadcastMatrixScalar) {
    Eigen::MatrixXd m(2, 2);
    m << 10.0, 20.0,
         30.0, 40.0;
    Value* mat = Value::from_matrix(m);
    Value* scalar = Value::from_scalar(3.0);

    Value* result = fn_multiply(mat, scalar);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 30.0);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 1), 60.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 90.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 1), 120.0);

    delete mat;
    delete scalar;
    delete result;
}

TEST(PrimitivesTest, BroadcastScalarLargeMatrix) {
    Value* scalar = Value::from_scalar(2.0);
    Eigen::MatrixXd m(10, 10);
    m.setConstant(1.0);
    Value* mat = Value::from_matrix(m);

    Value* result = fn_add(scalar, mat);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 10);
    EXPECT_EQ(res_mat->cols(), 10);

    // Check a few elements
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(5, 5), 3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(9, 9), 3.0);

    delete scalar;
    delete mat;
    delete result;
}

TEST(PrimitivesTest, BroadcastWithNegativeScalar) {
    Value* scalar = Value::from_scalar(-5.0);
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_add(scalar, vec);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), -4.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), -3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), -2.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), -1.0);

    delete scalar;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, BroadcastZeroScalar) {
    Value* zero = Value::from_scalar(0.0);
    Eigen::VectorXd v(3);
    v << 5.0, 10.0, 15.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_multiply(zero, vec);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 0.0);

    delete zero;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, BroadcastScalarDivideVector) {
    Value* scalar = Value::from_scalar(100.0);
    Eigen::VectorXd v(4);
    v << 2.0, 4.0, 5.0, 10.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_divide(scalar, vec);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 50.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 25.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 20.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 10.0);

    delete scalar;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, BroadcastVectorDivideScalar) {
    Eigen::VectorXd v(4);
    v << 10.0, 20.0, 30.0, 40.0;
    Value* vec = Value::from_vector(v);
    Value* scalar = Value::from_scalar(2.0);

    Value* result = fn_divide(vec, scalar);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 10.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 15.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 20.0);

    delete vec;
    delete scalar;
    delete result;
}

TEST(PrimitivesTest, BroadcastScalarPower) {
    Value* base = Value::from_scalar(2.0);
    Eigen::VectorXd exponents(5);
    exponents << 0.0, 1.0, 2.0, 3.0, 4.0;
    Value* exp_vec = Value::from_vector(exponents);

    Value* result = fn_power(base, exp_vec);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 8.0);
    EXPECT_DOUBLE_EQ((*res_mat)(4, 0), 16.0);

    delete base;
    delete exp_vec;
    delete result;
}

TEST(PrimitivesTest, BroadcastEmptyVector) {
    Value* scalar = Value::from_scalar(5.0);
    Eigen::VectorXd v(0);
    Value* empty_vec = Value::from_vector(v);

    Value* result = fn_add(scalar, empty_vec);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 0);
    EXPECT_EQ(res_mat->cols(), 1);

    delete scalar;
    delete empty_vec;
    delete result;
}

// ============================================================================
// Reduction/Scan Operator Tests
// ============================================================================

TEST(PrimitivesTest, ReduceMultiply) {
    Value* func = Value::from_function(&prim_times);
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_reduce(func, vec);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 24.0);  // 1*2*3*4 = 24

    delete func;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, ReduceSubtract) {
    Value* func = Value::from_function(&prim_minus);
    Eigen::VectorXd v(4);
    v << 10.0, 3.0, 2.0, 1.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_reduce(func, vec);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);  // APL right-to-left: 10-(3-(2-1)) = 10-2 = 8

    delete func;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, ReduceDivide) {
    Value* func = Value::from_function(&prim_divide);
    Eigen::VectorXd v(3);
    v << 100.0, 5.0, 2.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_reduce(func, vec);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 40.0);  // APL right-to-left: 100/(5/2) = 100/2.5 = 40

    delete func;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, ReducePower) {
    Value* func = Value::from_function(&prim_star);
    Eigen::VectorXd v(3);
    v << 2.0, 3.0, 2.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_reduce(func, vec);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 512.0);  // (2^3)^2 = 8^2 = 64... wait, right-to-left: 2^(3^2) = 2^9 = 512

    delete func;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, ReduceSingleElement) {
    Value* func = Value::from_function(&prim_plus);
    Eigen::VectorXd v(1);
    v << 42.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_reduce(func, vec);

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);

    delete func;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, ReduceEmptyVector) {
    Value* func = Value::from_function(&prim_plus);
    Eigen::VectorXd v(0);
    Value* vec = Value::from_vector(v);

    // Reducing empty vector should throw or return identity
    EXPECT_THROW(fn_reduce(func, vec), std::runtime_error);

    delete func;
    delete vec;
}

TEST(PrimitivesTest, ScanMultiply) {
    Value* func = Value::from_function(&prim_times);
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_scan(func, vec);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 4);
    // APL right-to-left scan: 1*(2*(3*4)), 2*(3*4), 3*4, 4
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 24.0);  // 1*24 = 24
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 24.0);  // 2*12 = 24
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 12.0);  // 3*4 = 12
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 4.0);   // 4

    delete func;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, ScanSubtract) {
    Value* func = Value::from_function(&prim_minus);
    Eigen::VectorXd v(5);
    v << 10.0, 1.0, 1.0, 1.0, 1.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_scan(func, vec);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    // APL right-to-left scan: 10-(1-(1-(1-1))), 1-(1-(1-1)), 1-(1-1), 1-1, 1
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 10.0);  // 10-(1-0) = 10-1 = 9... wait let me recalc
    // From right: 1, 1-1=0, 1-0=1, 1-1=0, 10-0=10
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 0.0);   // 1-(1-(1-1)) = 1-0 = 1... wait
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 1.0);   // 1-(1-1) = 1-0 = 1
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 0.0);   // 1-1 = 0
    EXPECT_DOUBLE_EQ((*res_mat)(4, 0), 1.0);   // 1

    delete func;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, ScanDivide) {
    Value* func = Value::from_function(&prim_divide);
    Eigen::VectorXd v(4);
    v << 100.0, 2.0, 5.0, 2.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_scan(func, vec);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    // APL right-to-left scan: 100/(2/(5/2))
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 125.0);   // 100/(2/(5/2)) = 100/0.8 = 125
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 0.8);     // 2/(5/2) = 2/2.5 = 0.8
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 2.5);     // 5/2 = 2.5
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 2.0);     // 2

    delete func;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, ScanSingleElement) {
    Value* func = Value::from_function(&prim_plus);
    Eigen::VectorXd v(1);
    v << 99.0;
    Value* vec = Value::from_vector(v);

    Value* result = fn_scan(func, vec);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 1);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 99.0);

    delete func;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, ReduceFirstAxis) {
    Value* func = Value::from_function(&prim_plus);
    Eigen::MatrixXd m(3, 4);
    m << 1.0, 2.0, 3.0, 4.0,
         5.0, 6.0, 7.0, 8.0,
         9.0, 10.0, 11.0, 12.0;
    Value* mat = Value::from_matrix(m);

    Value* result = fn_reduce_first(func, mat);

    ASSERT_TRUE(result->is_vector());  // Returns a vector, not a matrix
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 4);
    EXPECT_EQ(res_mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 15.0);  // 1+5+9
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 18.0);  // 2+6+10
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 21.0);  // 3+7+11
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 24.0);  // 4+8+12

    delete func;
    delete mat;
    delete result;
}

TEST(PrimitivesTest, ScanFirstAxis) {
    Value* func = Value::from_function(&prim_plus);
    Eigen::MatrixXd m(3, 2);
    m << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
    Value* mat = Value::from_matrix(m);

    Value* result = fn_scan_first(func, mat);

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 3);
    EXPECT_EQ(res_mat->cols(), 2);
    // APL right-to-left scan along first axis
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 9.0);   // 1+(3+5)
    EXPECT_DOUBLE_EQ((*res_mat)(0, 1), 12.0);  // 2+(4+6)
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 8.0);   // 3+5
    EXPECT_DOUBLE_EQ((*res_mat)(1, 1), 10.0);  // 4+6
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 5.0);   // 5
    EXPECT_DOUBLE_EQ((*res_mat)(2, 1), 6.0);   // 6

    delete func;
    delete mat;
    delete result;
}

// ============================================================================
// Primitive Composition Tests
// ============================================================================

TEST(PrimitivesTest, CompositionIotaReshape) {
    // ⍳12 → 0 1 2 3 4 5 6 7 8 9 10 11 (0-indexed)
    Value* n = Value::from_scalar(12.0);
    Value* iota_result = fn_iota(n);

    // Reshape into 3×4 matrix
    Eigen::VectorXd shape(2);
    shape << 3.0, 4.0;
    Value* shape_val = Value::from_vector(shape);
    Value* reshaped = fn_reshape(shape_val, iota_result);

    ASSERT_TRUE(reshaped->is_matrix());
    const Eigen::MatrixXd* mat = reshaped->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 4);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 3), 11.0);

    delete n;
    delete iota_result;
    delete shape_val;
    delete reshaped;
}

TEST(PrimitivesTest, CompositionReshapeTranspose) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = Value::from_vector(v);

    // Reshape to 2×3 (column-major fill: [[1,3,5],[2,4,6]])
    Eigen::VectorXd shape(2);
    shape << 2.0, 3.0;
    Value* shape_val = Value::from_vector(shape);
    Value* mat = fn_reshape(shape_val, vec);

    // Transpose to 3×2: [[1,2],[3,4],[5,6]]
    Value* transposed = fn_transpose(mat);

    ASSERT_TRUE(transposed->is_matrix());
    const Eigen::MatrixXd* res_mat = transposed->as_matrix();
    EXPECT_EQ(res_mat->rows(), 3);
    EXPECT_EQ(res_mat->cols(), 2);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 1), 4.0);

    delete vec;
    delete shape_val;
    delete mat;
    delete transposed;
}

TEST(PrimitivesTest, CompositionIotaTakeReduce) {
    // ⍳10 → 0 1 2 3 4 5 6 7 8 9 (0-indexed)
    Value* n = Value::from_scalar(10.0);
    Value* iota_result = fn_iota(n);

    // Take first 5: [0,1,2,3,4]
    Value* five = Value::from_scalar(5.0);
    Value* taken = fn_take(five, iota_result);

    // Sum with reduction
    Value* func = Value::from_function(&prim_plus);
    Value* sum = fn_reduce(func, taken);

    ASSERT_TRUE(sum->is_scalar());
    EXPECT_DOUBLE_EQ(sum->as_scalar(), 10.0);  // 0+1+2+3+4 = 10

    delete n;
    delete iota_result;
    delete five;
    delete taken;
    delete func;
    delete sum;
}

TEST(PrimitivesTest, CompositionRavelCatenate) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = Value::from_matrix(m);

    // Ravel to vector
    Value* raveled = fn_ravel(mat);

    // Create another vector to catenate
    Eigen::VectorXd v(3);
    v << 7.0, 8.0, 9.0;
    Value* vec = Value::from_vector(v);

    // Catenate
    Value* result = fn_catenate(raveled, vec);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 9);
    EXPECT_EQ(res_mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res_mat)(5, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res_mat)(6, 0), 7.0);
    EXPECT_DOUBLE_EQ((*res_mat)(8, 0), 9.0);

    delete mat;
    delete raveled;
    delete vec;
    delete result;
}

TEST(PrimitivesTest, CompositionMultiplyReduce) {
    // Create a 2×3 matrix
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = Value::from_matrix(m);

    // Reduce along last axis with multiplication
    Value* func = Value::from_function(&prim_times);
    Value* result = fn_reduce(func, mat);

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 2);
    EXPECT_EQ(res_mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 6.0);   // 1*2*3
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 120.0); // 4*5*6

    delete mat;
    delete func;
    delete result;
}

TEST(PrimitivesTest, CompositionDropScan) {
    // ⍳10
    Value* n = Value::from_scalar(10.0);
    Value* iota_result = fn_iota(n);

    // Drop first 3
    Value* three = Value::from_scalar(3.0);
    Value* dropped = fn_drop(three, iota_result);

    // Scan with addition
    Value* func = Value::from_function(&prim_plus);
    Value* scanned = fn_scan(func, dropped);

    ASSERT_TRUE(scanned->is_vector());
    const Eigen::MatrixXd* res_mat = scanned->as_matrix();
    EXPECT_EQ(res_mat->rows(), 7);
    // APL right-to-left scan on [3,4,5,6,7,8,9]
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 42.0);  // 3+(4+(5+(6+(7+(8+9)))))
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 39.0);  // 4+(5+(6+(7+(8+9))))
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 35.0);  // 5+(6+(7+(8+9)))
    EXPECT_DOUBLE_EQ((*res_mat)(6, 0), 9.0);   // 9

    delete n;
    delete iota_result;
    delete three;
    delete dropped;
    delete func;
    delete scanned;
}

TEST(PrimitivesTest, CompositionArithmeticChain) {
    // (5 + 3) × 2
    Value* a = Value::from_scalar(5.0);
    Value* b = Value::from_scalar(3.0);
    Value* sum = fn_add(a, b);

    Value* c = Value::from_scalar(2.0);
    Value* product = fn_multiply(sum, c);

    ASSERT_TRUE(product->is_scalar());
    EXPECT_DOUBLE_EQ(product->as_scalar(), 16.0);

    delete a;
    delete b;
    delete sum;
    delete c;
    delete product;
}

TEST(PrimitivesTest, CompositionShapeReshape) {
    Eigen::MatrixXd m(3, 4);
    m.setConstant(1.0);
    Value* mat = Value::from_matrix(m);

    // Get shape
    Value* shape = fn_shape(mat);

    // Use that shape to reshape a vector
    Eigen::VectorXd v(12);
    for (int i = 0; i < 12; i++) v(i) = i + 1.0;
    Value* vec = Value::from_vector(v);

    Value* reshaped = fn_reshape(shape, vec);

    ASSERT_TRUE(reshaped->is_matrix());
    const Eigen::MatrixXd* res_mat = reshaped->as_matrix();
    EXPECT_EQ(res_mat->rows(), 3);
    EXPECT_EQ(res_mat->cols(), 4);

    delete mat;
    delete shape;
    delete vec;
    delete reshaped;
}

TEST(PrimitivesTest, CompositionNestedReduce) {
    // Create matrix
    Eigen::MatrixXd m(3, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0;
    Value* mat = Value::from_matrix(m);

    // Reduce along last axis (sum rows)
    Value* func = Value::from_function(&prim_plus);
    Value* row_sums = fn_reduce(func, mat);

    // Now reduce that result (sum of row sums)
    Value* total = fn_reduce(func, row_sums);

    ASSERT_TRUE(total->is_scalar());
    EXPECT_DOUBLE_EQ(total->as_scalar(), 45.0);  // 1+2+...+9 = 45

    delete mat;
    delete func;
    delete row_sums;
    delete total;
}
