// Tests for arithmetic primitives

#include <gtest/gtest.h>
#include "primitives.h"
#include "operators.h"
#include "value.h"
#include "machine.h"
#include "continuation.h"
#include <cmath>
#include <stdexcept>

using namespace apl;

// Test fixture
class PrimitivesTest : public ::testing::Test {
protected:
    Machine* machine;
    void SetUp() override { machine = new Machine(); }
    void TearDown() override { delete machine; }
};


// Main function for running tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ============================================================================
// Dyadic Arithmetic Tests
// ============================================================================

TEST_F(PrimitivesTest, AddScalarScalar) {
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(4.0);
    fn_add(machine, a, b);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

}

TEST_F(PrimitivesTest, AddScalarVector) {
    Value* scalar = machine->heap->allocate_scalar(5.0);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_add(machine, scalar, vec);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 8.0);

}

TEST_F(PrimitivesTest, AddVectorScalar) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* scalar = machine->heap->allocate_scalar(10.0);

    fn_add(machine, vec, scalar);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 13.0);

}

TEST_F(PrimitivesTest, AddVectorVector) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(3);
    v2 << 4.0, 5.0, 6.0;

    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_add(machine, vec1, vec2);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 9.0);

}

TEST_F(PrimitivesTest, SubtractScalars) {
    Value* a = machine->heap->allocate_scalar(10.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_subtract(machine, a, b);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

}

TEST_F(PrimitivesTest, MultiplyScalars) {
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(4.0);
    fn_multiply(machine, a, b);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);

}

TEST_F(PrimitivesTest, MultiplyScalarVector) {
    Value* scalar = machine->heap->allocate_scalar(2.0);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_multiply(machine, scalar, vec);


    Value* result = machine->result;

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 6.0);

}

TEST_F(PrimitivesTest, DivideScalars) {
    Value* a = machine->heap->allocate_scalar(12.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_divide(machine, a, b);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);

}

TEST_F(PrimitivesTest, DivideByZeroError) {
    Value* a = machine->heap->allocate_scalar(12.0);
    Value* b = machine->heap->allocate_scalar(0.0);

    // Primitives now push ThrowErrorK instead of throwing C++ exceptions
    fn_divide(machine, a, b);

    // Should have pushed a ThrowErrorK
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, PowerScalars) {
    Value* a = machine->heap->allocate_scalar(2.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_power(machine, a, b);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);

}

TEST_F(PrimitivesTest, EqualScalars) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_equal(machine, a, b);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // True

}

TEST_F(PrimitivesTest, NotEqualScalars) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_equal(machine, a, b);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // False

}

TEST_F(PrimitivesTest, EqualVectors) {
    Eigen::VectorXd vec_a(3);
    vec_a << 1.0, 2.0, 3.0;
    Eigen::VectorXd vec_b(3);
    vec_b << 1.0, 5.0, 3.0;

    Value* a = machine->heap->allocate_vector(vec_a);
    Value* b = machine->heap->allocate_vector(vec_b);
    fn_equal(machine, a, b);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    ASSERT_EQ(res_mat->size(), 3);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);  // 1=1 is true
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 0.0);  // 2=5 is false
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 1.0);  // 3=3 is true

}

// ============================================================================
// Monadic Arithmetic Tests
// ============================================================================

TEST_F(PrimitivesTest, IdentityScalar) {
    Value* a = machine->heap->allocate_scalar(5.0);
    fn_conjugate(machine, a);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);

}

TEST_F(PrimitivesTest, IdentityVector) {
    // ISO 7.1.1: Conjugate (+B) returns B for real numbers
    Eigen::VectorXd v(4);
    v << 1.0, -2.0, 0.0, 3.5;
    Value* vec = machine->heap->allocate_vector(v);
    fn_conjugate(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 3.5);
}

TEST_F(PrimitivesTest, NegateScalar) {
    Value* a = machine->heap->allocate_scalar(5.0);
    fn_negate(machine, a);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);

}

TEST_F(PrimitivesTest, NegateVector) {
    Eigen::VectorXd v(3);
    v << 1.0, -2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_negate(machine, vec);


    Value* result = machine->result;

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), -3.0);

}

TEST_F(PrimitivesTest, SignPositive) {
    Value* a = machine->heap->allocate_scalar(5.0);
    fn_signum(machine, a);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);

}

TEST_F(PrimitivesTest, SignNegative) {
    Value* a = machine->heap->allocate_scalar(-5.0);
    fn_signum(machine, a);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -1.0);

}

TEST_F(PrimitivesTest, SignZero) {
    Value* a = machine->heap->allocate_scalar(0.0);
    fn_signum(machine, a);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);

}

TEST_F(PrimitivesTest, ReciprocalScalar) {
    Value* a = machine->heap->allocate_scalar(4.0);
    fn_reciprocal(machine, a);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.25);

}

TEST_F(PrimitivesTest, ReciprocalZeroError) {
    Value* a = machine->heap->allocate_scalar(0.0);

    fn_reciprocal(machine, a);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ExponentialScalar) {
    Value* a = machine->heap->allocate_scalar(1.0);
    fn_exponential(machine, a);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), M_E, 1e-10);

}

TEST_F(PrimitivesTest, ExponentialZero) {
    Value* a = machine->heap->allocate_scalar(0.0);
    fn_exponential(machine, a);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);

}

// ============================================================================
// Matrix Tests
// ============================================================================

TEST_F(PrimitivesTest, AddMatrices) {
    Eigen::MatrixXd m1(2, 2);
    m1 << 1.0, 2.0,
          3.0, 4.0;
    Eigen::MatrixXd m2(2, 2);
    m2 << 5.0, 6.0,
          7.0, 8.0;

    Value* mat1 = machine->heap->allocate_matrix(m1);
    Value* mat2 = machine->heap->allocate_matrix(m2);

    fn_add(machine, mat1, mat2);


    Value* result = machine->result;

    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 8.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 10.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 12.0);

}

TEST_F(PrimitivesTest, MismatchedShapeError) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(2);
    v2 << 4.0, 5.0;

    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_add(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

// ============================================================================
// Array Operation Tests
// ============================================================================

TEST_F(PrimitivesTest, ShapeScalar) {
    Value* scalar = machine->heap->allocate_scalar(5.0);
    fn_shape(machine, scalar);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);  // Empty shape for scalar

}

TEST_F(PrimitivesTest, ShapeVector) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_shape(machine, vec);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* shape = result->as_matrix();
    EXPECT_EQ(shape->rows(), 1);
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 5.0);

}

TEST_F(PrimitivesTest, ReshapeVector) {
    // 2 3⍴1 2 3 4 5 6 should produce row-major matrix:
    // 1 2 3
    // 4 5 6
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = machine->heap->allocate_vector(v);

    Eigen::VectorXd new_shape(2);
    new_shape << 2.0, 3.0;
    Value* shape = machine->heap->allocate_vector(new_shape);

    fn_reshape(machine, shape, vec);

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

TEST_F(PrimitivesTest, ReshapeRowMajorOrder) {
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

    fn_reshape(machine, shape, vec);

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

TEST_F(PrimitivesTest, ReshapeMatrixToMatrix) {
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

    fn_reshape(machine, shape, mat);

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

TEST_F(PrimitivesTest, Ravel) {
    // Ravel flattens in row-major order (APL standard)
    // Matrix:
    // 1 2 3
    // 4 5 6
    // Should become: 1 2 3 4 5 6
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_ravel(machine, mat);

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

TEST_F(PrimitivesTest, Catenate) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(2);
    v2 << 4.0, 5.0;

    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_catenate(machine, vec1, vec2);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);

}

TEST_F(PrimitivesTest, Transpose) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_transpose(machine, mat);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_EQ(res->cols(), 2);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 4.0);

}

TEST_F(PrimitivesTest, Iota) {
    Value* n = machine->heap->allocate_scalar(5.0);
    fn_iota(machine, n);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);  // 1-based per ISO 13751
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 5.0);

}

TEST_F(PrimitivesTest, Take) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    Value* count = machine->heap->allocate_scalar(3.0);
    fn_take(machine, count, vec);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);

}

TEST_F(PrimitivesTest, Drop) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    Value* count = machine->heap->allocate_scalar(2.0);
    fn_drop(machine, count, vec);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);

}

// ============================================================================
// Environment and Binding Tests
// ============================================================================

TEST_F(PrimitivesTest, EnvironmentInit) {
    // Verify arithmetic primitives are bound
    Value* plus = machine->env->lookup("+");
    ASSERT_NE(plus, nullptr);
    ASSERT_TRUE(plus->is_function());
    EXPECT_EQ(plus->data.primitive_fn, &prim_plus);

    Value* minus = machine->env->lookup("-");
    ASSERT_NE(minus, nullptr);
    ASSERT_TRUE(minus->is_function());

    // Verify array operations are bound
    Value* rho = machine->env->lookup("⍴");
    ASSERT_NE(rho, nullptr);
    ASSERT_TRUE(rho->is_function());

    Value* iota = machine->env->lookup("⍳");
    ASSERT_NE(iota, nullptr);
    ASSERT_TRUE(iota->is_function());
}

TEST_F(PrimitivesTest, PrimitiveLookupAndApply) {
    // Lookup primitive from environment and apply it
    Value* plus = machine->env->lookup("+");
    ASSERT_NE(plus, nullptr);

    // Use it to add two numbers
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(4.0);
    PrimitiveFn* fn = plus->data.primitive_fn;
    fn->dyadic(machine, a, b);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

}

TEST_F(PrimitivesTest, EnvironmentDefineAndUpdate) {
    Environment env;

    // Define a variable
    Value* x = machine->heap->allocate_scalar(42.0);
    env.define("x", x);

    // Lookup the variable
    Value* lookup = env.lookup("x");
    ASSERT_NE(lookup, nullptr);
    EXPECT_DOUBLE_EQ(lookup->as_scalar(), 42.0);

    // Update the variable
    Value* y = machine->heap->allocate_scalar(100.0);
    bool updated = env.update("x", y);
    ASSERT_TRUE(updated);

    Value* lookup2 = env.lookup("x");
    EXPECT_DOUBLE_EQ(lookup2->as_scalar(), 100.0);

}

TEST_F(PrimitivesTest, EnvironmentScoping) {
    Environment parent;
    Environment child(&parent);

    // Define in parent
    Value* x = machine->heap->allocate_scalar(10.0);
    parent.define("x", x);

    // Define in child
    Value* y = machine->heap->allocate_scalar(20.0);
    child.define("y", y);

    // Child can see both
    EXPECT_NE(child.lookup("x"), nullptr);
    EXPECT_NE(child.lookup("y"), nullptr);
    EXPECT_DOUBLE_EQ(child.lookup("x")->as_scalar(), 10.0);
    EXPECT_DOUBLE_EQ(child.lookup("y")->as_scalar(), 20.0);

    // Parent can only see its own
    EXPECT_NE(parent.lookup("x"), nullptr);
    EXPECT_EQ(parent.lookup("y"), nullptr);

}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(PrimitivesTest, ErrorShapeMismatchVectorVector) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(5);
    v2 << 1.0, 2.0, 3.0, 4.0, 5.0;

    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    // Should push ThrowErrorK on shape mismatch
    fn_add(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_subtract(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_multiply(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_divide(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorShapeMismatchMatrixMatrix) {
    Eigen::MatrixXd m1(2, 3);
    m1.setConstant(1.0);
    Eigen::MatrixXd m2(3, 2);
    m2.setConstant(2.0);

    Value* mat1 = machine->heap->allocate_matrix(m1);
    Value* mat2 = machine->heap->allocate_matrix(m2);

    // Should push ThrowErrorK on shape mismatch
    fn_add(machine, mat1, mat2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_subtract(machine, mat1, mat2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_multiply(machine, mat1, mat2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorDivideVectorByZeroVector) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(3);
    v2 << 1.0, 0.0, 2.0;  // Has a zero

    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    // Should throw on division by zero
    fn_divide(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorReciprocalVector) {
    Eigen::VectorXd v(3);
    v << 2.0, 0.0, 4.0;  // Has a zero

    Value* vec = machine->heap->allocate_vector(v);

    // Should throw on reciprocal of zero
    fn_reciprocal(machine, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ReshapeWithCycling) {
    // APL reshape cycles through source data when target is larger
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Reshape 3 elements into 2×3 (needs 6 elements) - should cycle: 1,2,3,1,2,3
    Eigen::VectorXd shape(2);
    shape << 2.0, 3.0;
    Value* shape_val = machine->heap->allocate_vector(shape);

    fn_reshape(machine, shape_val, vec);
    EXPECT_EQ(machine->kont_stack.size(), 0);  // No error
    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_matrix());
    const Eigen::MatrixXd* mat = machine->result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    // Row 0: 1, 2, 3
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 2), 3.0);
    // Row 1: 1, 2, 3 (cycled)
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 3.0);
}

TEST_F(PrimitivesTest, ErrorReshapeEmptyToNonEmpty) {
    // Cannot reshape empty array to non-empty shape
    Eigen::VectorXd v(0);  // Empty vector
    Value* vec = machine->heap->allocate_vector(v);

    Eigen::VectorXd shape(1);
    shape << 3.0;
    Value* shape_val = machine->heap->allocate_vector(shape);

    fn_reshape(machine, shape_val, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(PrimitivesTest, ErrorReshapeNonIntegerShape) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Shape with non-integer values
    Eigen::VectorXd shape(2);
    shape << 2.5, 3.0;
    Value* shape_val = machine->heap->allocate_vector(shape);

    fn_reshape(machine, shape_val, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorReshapeNegativeShape) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Negative shape dimension
    Eigen::VectorXd shape(2);
    shape << -2.0, 3.0;
    Value* shape_val = machine->heap->allocate_vector(shape);

    fn_reshape(machine, shape_val, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

// ISO 13751 8.3.1: If rank of A > 1, signal rank-error
TEST_F(PrimitivesTest, RankErrorReshapeMatrixShape) {
    // (2 2⍴1 2 3 4)⍴⍳6 → RANK ERROR (shape is a matrix, not vector)
    EXPECT_THROW(machine->eval("(2 2⍴1 2 3 4)⍴⍳6"), APLError);
}

TEST_F(PrimitivesTest, ErrorIotaNonScalar) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Iota requires scalar argument
    fn_iota(machine, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorIotaNegative) {
    Value* neg = machine->heap->allocate_scalar(-5.0);

    // Negative iota doesn't make sense
    fn_iota(machine, neg);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorIotaNonInteger) {
    Value* frac = machine->heap->allocate_scalar(3.5);

    // Fractional iota doesn't make sense
    fn_iota(machine, frac);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorTakeNonScalar) {
    Eigen::VectorXd n(2);
    n << 2.0, 3.0;
    Value* n_val = machine->heap->allocate_vector(n);

    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Take requires scalar count
    fn_take(machine, n_val, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorDropNonScalar) {
    Eigen::VectorXd n(2);
    n << 2.0, 3.0;
    Value* n_val = machine->heap->allocate_vector(n);

    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Drop requires scalar count
    fn_drop(machine, n_val, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorCatenateIncompatibleShapes) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Value* vec1 = machine->heap->allocate_vector(v1);

    Eigen::MatrixXd m2(2, 2);
    m2.setConstant(5.0);
    Value* mat2 = machine->heap->allocate_matrix(m2);

    // Cannot catenate vector with 2×2 matrix
    fn_catenate(machine, vec1, mat2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

// ============================================================================
// Broadcasting Edge Cases
// ============================================================================

TEST_F(PrimitivesTest, BroadcastScalarMatrix) {
    Value* scalar = machine->heap->allocate_scalar(5.0);
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_add(machine, scalar, mat);


    Value* result = machine->result;

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

}

TEST_F(PrimitivesTest, BroadcastMatrixScalar) {
    Eigen::MatrixXd m(2, 2);
    m << 10.0, 20.0,
         30.0, 40.0;
    Value* mat = machine->heap->allocate_matrix(m);
    Value* scalar = machine->heap->allocate_scalar(3.0);

    fn_multiply(machine, mat, scalar);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 30.0);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 1), 60.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 90.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 1), 120.0);

}

TEST_F(PrimitivesTest, BroadcastScalarLargeMatrix) {
    Value* scalar = machine->heap->allocate_scalar(2.0);
    Eigen::MatrixXd m(10, 10);
    m.setConstant(1.0);
    Value* mat = machine->heap->allocate_matrix(m);

    fn_add(machine, scalar, mat);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 10);
    EXPECT_EQ(res_mat->cols(), 10);

    // Check a few elements
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(5, 5), 3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(9, 9), 3.0);

}

TEST_F(PrimitivesTest, BroadcastWithNegativeScalar) {
    Value* scalar = machine->heap->allocate_scalar(-5.0);
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_add(machine, scalar, vec);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), -4.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), -3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), -2.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), -1.0);

}

TEST_F(PrimitivesTest, BroadcastZeroScalar) {
    Value* zero = machine->heap->allocate_scalar(0.0);
    Eigen::VectorXd v(3);
    v << 5.0, 10.0, 15.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_multiply(machine, zero, vec);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 0.0);

}

TEST_F(PrimitivesTest, BroadcastScalarDivideVector) {
    Value* scalar = machine->heap->allocate_scalar(100.0);
    Eigen::VectorXd v(4);
    v << 2.0, 4.0, 5.0, 10.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_divide(machine, scalar, vec);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 50.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 25.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 20.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 10.0);

}

TEST_F(PrimitivesTest, BroadcastVectorDivideScalar) {
    Eigen::VectorXd v(4);
    v << 10.0, 20.0, 30.0, 40.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* scalar = machine->heap->allocate_scalar(2.0);

    fn_divide(machine, vec, scalar);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 10.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 15.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 20.0);

}

TEST_F(PrimitivesTest, BroadcastScalarPower) {
    Value* base = machine->heap->allocate_scalar(2.0);
    Eigen::VectorXd exponents(5);
    exponents << 0.0, 1.0, 2.0, 3.0, 4.0;
    Value* exp_vec = machine->heap->allocate_vector(exponents);

    fn_power(machine, base, exp_vec);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 8.0);
    EXPECT_DOUBLE_EQ((*res_mat)(4, 0), 16.0);

}

TEST_F(PrimitivesTest, BroadcastEmptyVector) {
    Value* scalar = machine->heap->allocate_scalar(5.0);
    Eigen::VectorXd v(0);
    Value* empty_vec = machine->heap->allocate_vector(v);

    fn_add(machine, scalar, empty_vec);


    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 0);
    EXPECT_EQ(res_mat->cols(), 1);

}

// ============================================================================
// Primitive Composition Tests
// ============================================================================

TEST_F(PrimitivesTest, CompositionIotaReshape) {
    // ⍳12 → 1 2 3 4 5 6 7 8 9 10 11 12 (1-based per ISO 13751)
    Value* n = machine->heap->allocate_scalar(12.0);
    fn_iota(machine, n);

    Value* iota_result = machine->result;

    // Reshape into 3×4 matrix
    Eigen::VectorXd shape(2);
    shape << 3.0, 4.0;
    Value* shape_val = machine->heap->allocate_vector(shape);
    fn_reshape(machine, shape_val, iota_result);

    Value* reshaped = machine->result;

    ASSERT_TRUE(reshaped->is_matrix());
    const Eigen::MatrixXd* mat = reshaped->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 4);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 3), 12.0);

}

TEST_F(PrimitivesTest, CompositionReshapeTranspose) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Reshape to 2×3 (row-major fill):
    // 1 2 3
    // 4 5 6
    Eigen::VectorXd shape(2);
    shape << 2.0, 3.0;
    Value* shape_val = machine->heap->allocate_vector(shape);
    fn_reshape(machine, shape_val, vec);

    Value* mat = machine->result;

    // Transpose to 3×2:
    // 1 4
    // 2 5
    // 3 6
    fn_transpose(machine, mat);

    Value* transposed = machine->result;

    ASSERT_TRUE(transposed->is_matrix());
    const Eigen::MatrixXd* res_mat = transposed->as_matrix();
    EXPECT_EQ(res_mat->rows(), 3);
    EXPECT_EQ(res_mat->cols(), 2);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 1), 5.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 1), 6.0);
}

TEST_F(PrimitivesTest, CompositionRavelCatenate) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    // Ravel to vector
    fn_ravel(machine, mat);

    Value* raveled = machine->result;

    // Create another vector to catenate
    Eigen::VectorXd v(3);
    v << 7.0, 8.0, 9.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Catenate
    fn_catenate(machine, raveled, vec);

    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 9);
    EXPECT_EQ(res_mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res_mat)(5, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res_mat)(6, 0), 7.0);
    EXPECT_DOUBLE_EQ((*res_mat)(8, 0), 9.0);

}

TEST_F(PrimitivesTest, CompositionArithmeticChain) {
    // (5 + 3) × 2
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_add(machine, a, b);

    Value* sum = machine->result;

    Value* c = machine->heap->allocate_scalar(2.0);
    fn_multiply(machine, sum, c);

    Value* product = machine->result;

    ASSERT_TRUE(product->is_scalar());
    EXPECT_DOUBLE_EQ(product->as_scalar(), 16.0);

}

TEST_F(PrimitivesTest, CompositionShapeReshape) {
    Eigen::MatrixXd m(3, 4);
    m.setConstant(1.0);
    Value* mat = machine->heap->allocate_matrix(m);

    // Get shape
    fn_shape(machine, mat);

    Value* shape = machine->result;

    // Use that shape to reshape a vector
    Eigen::VectorXd v(12);
    for (int i = 0; i < 12; i++) v(i) = i + 1.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reshape(machine, shape, vec);


    Value* reshaped = machine->result;

    ASSERT_TRUE(reshaped->is_matrix());
    const Eigen::MatrixXd* res_mat = reshaped->as_matrix();
    EXPECT_EQ(res_mat->rows(), 3);
    EXPECT_EQ(res_mat->cols(), 4);

}

// ============================================================================
// Vector/Matrix Status Preservation Tests
// Primitives should preserve vector status: vector in -> vector out
// ============================================================================

TEST_F(PrimitivesTest, NegatePreservesVector) {
    Eigen::VectorXd v(3);
    v << 1, 2, 3;
    Value* vec = machine->heap->allocate_vector(v);

    fn_negate(machine, vec);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Negate should preserve vector status";
    EXPECT_EQ(result->rows(), 3);
}

TEST_F(PrimitivesTest, NegatePreservesMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1, 2, 3, 4, 5, 6;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_negate(machine, mat);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_matrix()) << "Negate should preserve matrix status";
    ASSERT_FALSE(result->is_vector()) << "Negate should not convert matrix to vector";
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 3);
}

TEST_F(PrimitivesTest, SignumPreservesVector) {
    Eigen::VectorXd v(3);
    v << -1, 0, 2;
    Value* vec = machine->heap->allocate_vector(v);

    fn_signum(machine, vec);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Signum should preserve vector status";
}

TEST_F(PrimitivesTest, ReciprocalPreservesVector) {
    Eigen::VectorXd v(3);
    v << 1, 2, 4;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reciprocal(machine, vec);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Reciprocal should preserve vector status";
}

TEST_F(PrimitivesTest, ExponentialPreservesVector) {
    Eigen::VectorXd v(3);
    v << 0, 1, 2;
    Value* vec = machine->heap->allocate_vector(v);

    fn_exponential(machine, vec);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Exponential should preserve vector status";
}

TEST_F(PrimitivesTest, AddScalarVectorPreservesVector) {
    Value* scalar = machine->heap->allocate_scalar(10.0);
    Eigen::VectorXd v(3);
    v << 1, 2, 3;
    Value* vec = machine->heap->allocate_vector(v);

    fn_add(machine, scalar, vec);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Scalar + Vector should produce vector";
}

TEST_F(PrimitivesTest, AddVectorScalarPreservesVector) {
    Eigen::VectorXd v(3);
    v << 1, 2, 3;
    Value* vec = machine->heap->allocate_vector(v);
    Value* scalar = machine->heap->allocate_scalar(10.0);

    fn_add(machine, vec, scalar);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Vector + Scalar should produce vector";
}

TEST_F(PrimitivesTest, AddVectorVectorPreservesVector) {
    Eigen::VectorXd v1(3), v2(3);
    v1 << 1, 2, 3;
    v2 << 10, 20, 30;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_add(machine, vec1, vec2);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Vector + Vector should produce vector";
}

TEST_F(PrimitivesTest, SubtractPreservesVector) {
    Eigen::VectorXd v1(3), v2(3);
    v1 << 10, 20, 30;
    v2 << 1, 2, 3;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_subtract(machine, vec1, vec2);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Vector - Vector should produce vector";
}

TEST_F(PrimitivesTest, MultiplyPreservesVector) {
    Eigen::VectorXd v1(3), v2(3);
    v1 << 1, 2, 3;
    v2 << 10, 20, 30;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_multiply(machine, vec1, vec2);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Vector * Vector should produce vector";
}

TEST_F(PrimitivesTest, DividePreservesVector) {
    Eigen::VectorXd v1(3), v2(3);
    v1 << 10, 20, 30;
    v2 << 2, 4, 5;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_divide(machine, vec1, vec2);
    Value* result = machine->result;

    ASSERT_TRUE(result->is_vector()) << "Vector ÷ Vector should produce vector";
}

// ============================================================================
// Comparison Tests (≠ < > ≤ ≥)
// ============================================================================

TEST_F(PrimitivesTest, FnNotEqualScalarsDifferent) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_not_equal(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5≠3 is true
}

TEST_F(PrimitivesTest, FnNotEqualScalarsSame) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_not_equal(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 5≠5 is false
}

TEST_F(PrimitivesTest, FnNotEqualVectors) {
    Eigen::VectorXd v1(3), v2(3);
    v1 << 1.0, 2.0, 3.0;
    v2 << 1.0, 5.0, 3.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_not_equal(machine, vec1, vec2);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);  // 1≠1 is false
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);  // 2≠5 is true
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 0.0);  // 3≠3 is false
}

TEST_F(PrimitivesTest, LessThanScalars) {
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_less(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 3<5 is true
}

TEST_F(PrimitivesTest, LessThanScalarsFalse) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_less(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 5<3 is false
}

TEST_F(PrimitivesTest, LessThanScalarsEqual) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_less(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 5<5 is false
}

TEST_F(PrimitivesTest, LessThanVectors) {
    Eigen::VectorXd v1(4), v2(4);
    v1 << 1.0, 5.0, 3.0, 3.0;
    v2 << 2.0, 3.0, 3.0, 4.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_less(machine, vec1, vec2);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);  // 1<2 is true
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 0.0);  // 5<3 is false
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 0.0);  // 3<3 is false
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 1.0);  // 3<4 is true
}

TEST_F(PrimitivesTest, GreaterThanScalars) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_greater(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5>3 is true
}

TEST_F(PrimitivesTest, GreaterThanScalarsFalse) {
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_greater(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 3>5 is false
}

TEST_F(PrimitivesTest, GreaterThanVectors) {
    Eigen::VectorXd v1(4), v2(4);
    v1 << 5.0, 2.0, 3.0, 4.0;
    v2 << 3.0, 4.0, 3.0, 2.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_greater(machine, vec1, vec2);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);  // 5>3 is true
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 0.0);  // 2>4 is false
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 0.0);  // 3>3 is false
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 1.0);  // 4>2 is true
}

TEST_F(PrimitivesTest, LessOrEqualScalarsLess) {
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_less_eq(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 3≤5 is true
}

TEST_F(PrimitivesTest, LessOrEqualScalarsEqual) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_less_eq(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5≤5 is true
}

TEST_F(PrimitivesTest, LessOrEqualScalarsFalse) {
    Value* a = machine->heap->allocate_scalar(7.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_less_eq(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 7≤5 is false
}

TEST_F(PrimitivesTest, LessOrEqualVectors) {
    Eigen::VectorXd v1(4), v2(4);
    v1 << 1.0, 5.0, 3.0, 4.0;
    v2 << 2.0, 3.0, 3.0, 5.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_less_eq(machine, vec1, vec2);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);  // 1≤2 is true
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 0.0);  // 5≤3 is false
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);  // 3≤3 is true
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 1.0);  // 4≤5 is true
}

TEST_F(PrimitivesTest, GreaterOrEqualScalarsGreater) {
    Value* a = machine->heap->allocate_scalar(7.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_greater_eq(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 7≥5 is true
}

TEST_F(PrimitivesTest, GreaterOrEqualScalarsEqual) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_greater_eq(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 5≥5 is true
}

TEST_F(PrimitivesTest, GreaterOrEqualScalarsFalse) {
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_greater_eq(machine, a, b);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // 3≥5 is false
}

TEST_F(PrimitivesTest, GreaterOrEqualVectors) {
    Eigen::VectorXd v1(4), v2(4);
    v1 << 5.0, 2.0, 3.0, 4.0;
    v2 << 3.0, 4.0, 3.0, 2.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_greater_eq(machine, vec1, vec2);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);  // 5≥3 is true
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 0.0);  // 2≥4 is false
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);  // 3≥3 is true
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 1.0);  // 4≥2 is true
}

// Scalar extension tests for comparisons
TEST_F(PrimitivesTest, LessThanScalarVector) {
    Value* scalar = machine->heap->allocate_scalar(3.0);
    Eigen::VectorXd v(4);
    v << 1.0, 3.0, 5.0, 2.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_less(machine, scalar, vec);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);  // 3<1 is false
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 0.0);  // 3<3 is false
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);  // 3<5 is true
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 0.0);  // 3<2 is false
}

TEST_F(PrimitivesTest, GreaterThanVectorScalar) {
    Eigen::VectorXd v(4);
    v << 1.0, 3.0, 5.0, 2.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* scalar = machine->heap->allocate_scalar(3.0);

    fn_greater(machine, vec, scalar);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);  // 1>3 is false
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 0.0);  // 3>3 is false
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);  // 5>3 is true
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 0.0);  // 2>3 is false
}

// Shape mismatch error tests for comparisons
TEST_F(PrimitivesTest, ComparisonShapeMismatchError) {
    Eigen::VectorXd v1(3), v2(4);
    v1 << 1.0, 2.0, 3.0;
    v2 << 1.0, 2.0, 3.0, 4.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_less(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_greater(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_less_eq(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_greater_eq(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_not_equal(machine, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// Vector status preservation tests for comparisons
TEST_F(PrimitivesTest, ComparisonPreservesVector) {
    Eigen::VectorXd v1(3), v2(3);
    v1 << 1.0, 2.0, 3.0;
    v2 << 2.0, 2.0, 2.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_less(machine, vec1, vec2);
    ASSERT_TRUE(machine->result->is_vector()) << "< should preserve vector status";

    fn_greater(machine, vec1, vec2);
    ASSERT_TRUE(machine->result->is_vector()) << "> should preserve vector status";

    fn_less_eq(machine, vec1, vec2);
    ASSERT_TRUE(machine->result->is_vector()) << "≤ should preserve vector status";

    fn_greater_eq(machine, vec1, vec2);
    ASSERT_TRUE(machine->result->is_vector()) << "≥ should preserve vector status";

    fn_not_equal(machine, vec1, vec2);
    ASSERT_TRUE(machine->result->is_vector()) << "≠ should preserve vector status";
}

// Environment binding tests for new comparisons
TEST_F(PrimitivesTest, ComparisonPrimitivesRegistered) {
    Value* neq = machine->env->lookup("≠");
    ASSERT_NE(neq, nullptr);
    ASSERT_TRUE(neq->is_function());

    Value* lt = machine->env->lookup("<");
    ASSERT_NE(lt, nullptr);
    ASSERT_TRUE(lt->is_function());

    Value* gt = machine->env->lookup(">");
    ASSERT_NE(gt, nullptr);
    ASSERT_TRUE(gt->is_function());

    Value* le = machine->env->lookup("≤");
    ASSERT_NE(le, nullptr);
    ASSERT_TRUE(le->is_function());

    Value* ge = machine->env->lookup("≥");
    ASSERT_NE(ge, nullptr);
    ASSERT_TRUE(ge->is_function());
}

// ============================================================================
// Min/Max Tests (⌈ ⌊)
// ============================================================================

TEST_F(PrimitivesTest, CeilingMonadicScalar) {
    Value* v = machine->heap->allocate_scalar(3.2);
    fn_ceiling(machine, v);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 4.0);
}

TEST_F(PrimitivesTest, CeilingMonadicNegative) {
    Value* v = machine->heap->allocate_scalar(-3.2);
    fn_ceiling(machine, v);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), -3.0);
}

TEST_F(PrimitivesTest, CeilingMonadicVector) {
    Eigen::VectorXd v(3);
    v << 1.2, 2.7, -1.5;
    Value* vec = machine->heap->allocate_vector(v);
    fn_ceiling(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* mat = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), -1.0);
}

TEST_F(PrimitivesTest, FloorMonadicScalar) {
    Value* v = machine->heap->allocate_scalar(3.7);
    fn_floor(machine, v);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(PrimitivesTest, FloorMonadicNegative) {
    Value* v = machine->heap->allocate_scalar(-3.2);
    fn_floor(machine, v);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), -4.0);
}

TEST_F(PrimitivesTest, FloorMonadicVector) {
    Eigen::VectorXd v(3);
    v << 1.2, 2.7, -1.5;
    Value* vec = machine->heap->allocate_vector(v);
    fn_floor(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* mat = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), -2.0);
}

TEST_F(PrimitivesTest, MaximumDyadicScalars) {
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_maximum(machine, a, b);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, MaximumDyadicVectors) {
    Eigen::VectorXd v1(3), v2(3);
    v1 << 1.0, 5.0, 3.0;
    v2 << 4.0, 2.0, 6.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);
    fn_maximum(machine, vec1, vec2);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* mat = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 6.0);
}

TEST_F(PrimitivesTest, MinimumDyadicScalars) {
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_minimum(machine, a, b);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(PrimitivesTest, MinimumDyadicVectors) {
    Eigen::VectorXd v1(3), v2(3);
    v1 << 1.0, 5.0, 3.0;
    v2 << 4.0, 2.0, 6.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);
    fn_minimum(machine, vec1, vec2);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* mat = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);
}

// ============================================================================
// Logical Function Tests (∧ ∨ ~ ⍲ ⍱)
// ============================================================================

TEST_F(PrimitivesTest, NotMonadicScalar) {
    Value* zero = machine->heap->allocate_scalar(0.0);
    fn_not(machine, zero);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);

    Value* one = machine->heap->allocate_scalar(1.0);
    fn_not(machine, one);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, NotMonadicVector) {
    Eigen::VectorXd v(4);
    v << 1.0, 0.0, 1.0, 0.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_not(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* mat = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 1.0);
}

TEST_F(PrimitivesTest, NotDomainErrorNonBoolean) {
    // ISO 13751 7.1.12: ~ requires near-boolean argument
    // ~0.5 → DOMAIN ERROR
    EXPECT_THROW(machine->eval("~0.5"), APLError);
}

TEST_F(PrimitivesTest, NotDomainErrorNonBooleanVector) {
    // ~1 2 3 → DOMAIN ERROR (2 and 3 are not near-boolean)
    EXPECT_THROW(machine->eval("~1 2 3"), APLError);
}

TEST_F(PrimitivesTest, NotNearBooleanAccepted) {
    // ISO 13751 7.1.12: Near-boolean values should be accepted
    // ~0.99999999999 → 0 (within 1E-11 of 1, tolerance is 1E-10)
    Value* result = machine->eval("~0.99999999999");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);

    // ~1E¯12 → 1 (within tolerance of 0)
    result = machine->eval("~1E¯12");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, AndDyadicScalars) {
    Value* one = machine->heap->allocate_scalar(1.0);
    Value* zero = machine->heap->allocate_scalar(0.0);

    fn_and(machine, one, one);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);

    fn_and(machine, one, zero);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);

    fn_and(machine, zero, zero);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, AndDyadicVectors) {
    Eigen::VectorXd v1(4), v2(4);
    v1 << 1.0, 0.0, 1.0, 0.0;
    v2 << 1.0, 1.0, 0.0, 0.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);
    fn_and(machine, vec1, vec2);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* mat = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);  // 1∧1
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 0.0);  // 0∧1
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 0.0);  // 1∧0
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 0.0);  // 0∧0
}

TEST_F(PrimitivesTest, OrDyadicScalars) {
    Value* one = machine->heap->allocate_scalar(1.0);
    Value* zero = machine->heap->allocate_scalar(0.0);

    fn_or(machine, one, one);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);

    fn_or(machine, one, zero);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);

    fn_or(machine, zero, zero);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, OrDyadicVectors) {
    Eigen::VectorXd v1(4), v2(4);
    v1 << 1.0, 0.0, 1.0, 0.0;
    v2 << 1.0, 1.0, 0.0, 0.0;
    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);
    fn_or(machine, vec1, vec2);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* mat = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);  // 1∨1
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);  // 0∨1
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);  // 1∨0
    EXPECT_DOUBLE_EQ((*mat)(3, 0), 0.0);  // 0∨0
}

// ISO 13751 7.2.12: ∧ is LCM for non-boolean values
TEST_F(PrimitivesTest, AndLCM) {
    // 30∧36 = 180 (LCM of 30 and 36)
    Value* result = machine->eval("30∧36");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 180.0);

    // 3∧3.6 = 18 (per spec example)
    result = machine->eval("3∧3.6");
    ASSERT_NE(result, nullptr);
    EXPECT_NEAR(result->as_scalar(), 18.0, 0.0001);

    // LCM with zero returns zero
    result = machine->eval("0∧5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// ISO 13751 7.2.13: ∨ is GCD for non-boolean values
TEST_F(PrimitivesTest, OrGCD) {
    // 30∨36 = 6 (GCD of 30 and 36)
    Value* result = machine->eval("30∨36");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);

    // 3∨3.6 = 0.6 (per spec example)
    result = machine->eval("3∨3.6");
    ASSERT_NE(result, nullptr);
    EXPECT_NEAR(result->as_scalar(), 0.6, 0.0001);

    // GCD with zero returns the other number
    result = machine->eval("0∨5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, NandDyadicScalars) {
    Value* one = machine->heap->allocate_scalar(1.0);
    Value* zero = machine->heap->allocate_scalar(0.0);

    fn_nand(machine, one, one);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);  // ~(1∧1)

    fn_nand(machine, one, zero);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);  // ~(1∧0)

    fn_nand(machine, zero, zero);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);  // ~(0∧0)
}

TEST_F(PrimitivesTest, NorDyadicScalars) {
    Value* one = machine->heap->allocate_scalar(1.0);
    Value* zero = machine->heap->allocate_scalar(0.0);

    fn_nor(machine, one, one);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);  // ~(1∨1)

    fn_nor(machine, one, zero);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);  // ~(1∨0)

    fn_nor(machine, zero, zero);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);  // ~(0∨0)
}

// ISO 13751 7.2.14: Nand requires boolean arguments
TEST_F(PrimitivesTest, NandDomainErrorNonBoolean) {
    EXPECT_THROW(machine->eval("0.5⍲0.5"), APLError);
    EXPECT_THROW(machine->eval("2⍲1"), APLError);
    EXPECT_THROW(machine->eval("0⍲¯1"), APLError);
}

// ISO 13751 7.2.15: Nor requires boolean arguments
TEST_F(PrimitivesTest, NorDomainErrorNonBoolean) {
    EXPECT_THROW(machine->eval("0.5⍱0.5"), APLError);
    EXPECT_THROW(machine->eval("2⍱1"), APLError);
    EXPECT_THROW(machine->eval("0⍱¯1"), APLError);
}

// Nand/Nor accept near-boolean values (tolerantly close to 0 or 1)
TEST_F(PrimitivesTest, NandNorNearBooleanAccepted) {
    // Near-1 values should be accepted and treated as 1
    Value* result = machine->eval("0.99999999999⍲0.99999999999");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // ~(1∧1) = 0

    result = machine->eval("0.99999999999⍱0.99999999999");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // ~(1∨1) = 0
}

TEST_F(PrimitivesTest, LogicalPrimitivesRegistered) {
    ASSERT_NE(machine->env->lookup("⌈"), nullptr);
    ASSERT_NE(machine->env->lookup("⌊"), nullptr);
    ASSERT_NE(machine->env->lookup("∧"), nullptr);
    ASSERT_NE(machine->env->lookup("∨"), nullptr);
    ASSERT_NE(machine->env->lookup("~"), nullptr);
    ASSERT_NE(machine->env->lookup("⍲"), nullptr);
    ASSERT_NE(machine->env->lookup("⍱"), nullptr);
}

// ============================================================================
// Arithmetic Extensions Tests (| ⍟ !)
// ============================================================================

// Magnitude/Absolute Value (| monadic)
TEST_F(PrimitivesTest, MagnitudePositive) {
    Value* val = machine->heap->allocate_scalar(5.0);
    fn_magnitude(machine, val);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, MagnitudeNegative) {
    Value* val = machine->heap->allocate_scalar(-5.0);
    fn_magnitude(machine, val);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, MagnitudeZero) {
    Value* val = machine->heap->allocate_scalar(0.0);
    fn_magnitude(machine, val);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, MagnitudeVector) {
    Eigen::VectorXd v(4);
    v << -3.0, 4.0, -5.0, 0.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_magnitude(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 0.0);
}

// Residue (| dyadic)
TEST_F(PrimitivesTest, ResidueBasic) {
    // 3 | 7 → 1 (7 mod 3)
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Value* rhs = machine->heap->allocate_scalar(7.0);
    fn_residue(machine, lhs, rhs);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, ResidueExact) {
    // 3 | 9 → 0
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Value* rhs = machine->heap->allocate_scalar(9.0);
    fn_residue(machine, lhs, rhs);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, ResidueNegative) {
    // 3 | -7 → 2 (APL residue always non-negative for positive divisor)
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Value* rhs = machine->heap->allocate_scalar(-7.0);
    fn_residue(machine, lhs, rhs);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 2.0);
}

TEST_F(PrimitivesTest, ResidueVector) {
    // 3 | 1 2 3 4 5 → 1 2 0 1 2
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* rhs = machine->heap->allocate_vector(v);
    fn_residue(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 2.0);
}

TEST_F(PrimitivesTest, ResidueZeroLeft) {
    // ISO 7.2.9: "If A is zero, return B"
    // 0 | 5 → 5
    Value* lhs = machine->heap->allocate_scalar(0.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);
    fn_residue(machine, lhs, rhs);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, ResidueZeroLeftNegative) {
    // 0 | ¯7 → ¯7
    Value* lhs = machine->heap->allocate_scalar(0.0);
    Value* rhs = machine->heap->allocate_scalar(-7.0);
    fn_residue(machine, lhs, rhs);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), -7.0);
}

TEST_F(PrimitivesTest, ResidueZeroLeftVector) {
    // 0 | 1 2 3 → 1 2 3
    Value* lhs = machine->heap->allocate_scalar(0.0);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* rhs = machine->heap->allocate_vector(v);
    fn_residue(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

// Natural Logarithm (⍟ monadic)
TEST_F(PrimitivesTest, NaturalLogE) {
    // ⍟ e → 1
    Value* val = machine->heap->allocate_scalar(std::exp(1.0));
    fn_natural_log(machine, val);
    EXPECT_NEAR(machine->result->as_scalar(), 1.0, 1e-10);
}

TEST_F(PrimitivesTest, NaturalLogOne) {
    // ⍟ 1 → 0
    Value* val = machine->heap->allocate_scalar(1.0);
    fn_natural_log(machine, val);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, NaturalLogVector) {
    Eigen::VectorXd v(3);
    v << 1.0, std::exp(1.0), std::exp(2.0);
    Value* vec = machine->heap->allocate_vector(v);
    fn_natural_log(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 0.0);
    EXPECT_NEAR((*res)(1, 0), 1.0, 1e-10);
    EXPECT_NEAR((*res)(2, 0), 2.0, 1e-10);
}

// Logarithm (⍟ dyadic)
TEST_F(PrimitivesTest, LogarithmBase10) {
    // 10 ⍟ 100 → 2
    Value* lhs = machine->heap->allocate_scalar(10.0);
    Value* rhs = machine->heap->allocate_scalar(100.0);
    fn_logarithm(machine, lhs, rhs);
    EXPECT_NEAR(machine->result->as_scalar(), 2.0, 1e-10);
}

TEST_F(PrimitivesTest, LogarithmBase2) {
    // 2 ⍟ 8 → 3
    Value* lhs = machine->heap->allocate_scalar(2.0);
    Value* rhs = machine->heap->allocate_scalar(8.0);
    fn_logarithm(machine, lhs, rhs);
    EXPECT_NEAR(machine->result->as_scalar(), 3.0, 1e-10);
}

TEST_F(PrimitivesTest, LogarithmVector) {
    // 2 ⍟ 1 2 4 8 → 0 1 2 3
    Value* lhs = machine->heap->allocate_scalar(2.0);
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 4.0, 8.0;
    Value* rhs = machine->heap->allocate_vector(v);
    fn_logarithm(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_NEAR((*res)(0, 0), 0.0, 1e-10);
    EXPECT_NEAR((*res)(1, 0), 1.0, 1e-10);
    EXPECT_NEAR((*res)(2, 0), 2.0, 1e-10);
    EXPECT_NEAR((*res)(3, 0), 3.0, 1e-10);
}

// Factorial (! monadic)
TEST_F(PrimitivesTest, FactorialZero) {
    // ! 0 → 1
    Value* val = machine->heap->allocate_scalar(0.0);
    fn_factorial(machine, val);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, FactorialOne) {
    // ! 1 → 1
    Value* val = machine->heap->allocate_scalar(1.0);
    fn_factorial(machine, val);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, FactorialFive) {
    // ! 5 → 120
    Value* val = machine->heap->allocate_scalar(5.0);
    fn_factorial(machine, val);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 120.0);
}

TEST_F(PrimitivesTest, FactorialVector) {
    // ! 0 1 2 3 4 5 → 1 1 2 6 24 120
    Eigen::VectorXd v(6);
    v << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_factorial(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 24.0);
    EXPECT_DOUBLE_EQ((*res)(5, 0), 120.0);
}

// Binomial (! dyadic)
TEST_F(PrimitivesTest, BinomialBasic) {
    // 2 ! 5 → 10 (5 choose 2)
    Value* lhs = machine->heap->allocate_scalar(2.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);
    fn_binomial(machine, lhs, rhs);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 10.0);
}

TEST_F(PrimitivesTest, BinomialZeroK) {
    // 0 ! 5 → 1
    Value* lhs = machine->heap->allocate_scalar(0.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);
    fn_binomial(machine, lhs, rhs);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, BinomialSame) {
    // 5 ! 5 → 1
    Value* lhs = machine->heap->allocate_scalar(5.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);
    fn_binomial(machine, lhs, rhs);
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, BinomialPascalsRow) {
    // 0 1 2 3 4 ! 4 → 1 4 6 4 1 (row of Pascal's triangle)
    Eigen::VectorXd v(5);
    v << 0.0, 1.0, 2.0, 3.0, 4.0;
    Value* lhs = machine->heap->allocate_vector(v);
    Value* rhs = machine->heap->allocate_scalar(4.0);
    fn_binomial(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 1.0);
}

TEST_F(PrimitivesTest, ArithmeticExtensionsRegistered) {
    ASSERT_NE(machine->env->lookup("|"), nullptr);
    ASSERT_NE(machine->env->lookup("⍟"), nullptr);
    ASSERT_NE(machine->env->lookup("!"), nullptr);
}

// ============================================================================
// Reverse/Rotate/Tally Tests
// ============================================================================

TEST_F(PrimitivesTest, ReverseVector) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reverse(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 1.0);
}

TEST_F(PrimitivesTest, ReverseScalar) {
    Value* scalar = machine->heap->allocate_scalar(42.0);

    fn_reverse(machine, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 42.0);
}

TEST_F(PrimitivesTest, ReverseMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_reverse(machine, mat);

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

TEST_F(PrimitivesTest, ReverseFirstMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_reverse_first(machine, mat);

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

TEST_F(PrimitivesTest, RotateVectorPositive) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* count = machine->heap->allocate_scalar(2.0);

    fn_rotate(machine, count, vec);

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

TEST_F(PrimitivesTest, RotateVectorNegative) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* count = machine->heap->allocate_scalar(-2.0);

    fn_rotate(machine, count, vec);

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

TEST_F(PrimitivesTest, RotateFirstMatrix) {
    Eigen::MatrixXd m(3, 2);
    m << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);
    Value* count = machine->heap->allocate_scalar(1.0);

    fn_rotate_first(machine, count, mat);

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

TEST_F(PrimitivesTest, RotateWrapAround) {
    // ISO 10.2.7: rotation wraps around
    // ¯7⌽'ABCDEF' → 'FABCDE' (¯7 mod 6 = ¯1)
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* count = machine->heap->allocate_scalar(7.0);  // 7 mod 5 = 2

    fn_rotate(machine, count, vec);

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

TEST_F(PrimitivesTest, RotateScalar) {
    // ISO 10.2.7: rotating a scalar returns it unchanged
    Value* scalar = machine->heap->allocate_scalar(42.0);
    Value* count = machine->heap->allocate_scalar(5.0);

    fn_rotate(machine, count, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 42.0);
}

TEST_F(PrimitivesTest, TallyVector) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_tally(machine, vec);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, TallyScalar) {
    Value* scalar = machine->heap->allocate_scalar(42.0);

    fn_tally(machine, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, TallyMatrix) {
    Eigen::MatrixXd m(3, 4);
    m.setZero();
    Value* mat = machine->heap->allocate_matrix(m);

    fn_tally(machine, mat);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(PrimitivesTest, ReverseRotateTallyRegistered) {
    ASSERT_NE(machine->env->lookup("⌽"), nullptr);
    ASSERT_NE(machine->env->lookup("⊖"), nullptr);
    ASSERT_NE(machine->env->lookup("≢"), nullptr);
}

// ============================================================================
// Search Functions (⍳ dyadic, ∊)
// ============================================================================

TEST_F(PrimitivesTest, IndexOfFound) {
    // 1 2 3 4 5 ⍳ 3 → 3 (1-origin index per ISO 13751)
    Eigen::VectorXd haystack(5);
    haystack << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* lhs = machine->heap->allocate_vector(haystack);
    Value* rhs = machine->heap->allocate_scalar(3.0);

    fn_index_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(PrimitivesTest, IndexOfNotFound) {
    // 1 2 3 ⍳ 7 → 4 (not found = 1 + length of haystack, per ISO 13751)
    Eigen::VectorXd haystack(3);
    haystack << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(haystack);
    Value* rhs = machine->heap->allocate_scalar(7.0);

    fn_index_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 4.0);
}

TEST_F(PrimitivesTest, IndexOfVector) {
    // 10 20 30 40 ⍳ 30 20 99 → 3 2 5 (1-origin per ISO 13751)
    Eigen::VectorXd haystack(4);
    haystack << 10.0, 20.0, 30.0, 40.0;
    Eigen::VectorXd needles(3);
    needles << 30.0, 20.0, 99.0;
    Value* lhs = machine->heap->allocate_vector(haystack);
    Value* rhs = machine->heap->allocate_vector(needles);

    fn_index_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);  // 30 found at index 3 (1-origin)
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);  // 20 found at index 2 (1-origin)
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);  // 99 not found → 5 (1+length)
}

TEST_F(PrimitivesTest, IndexOfScalarHaystack) {
    // 5 ⍳ 5 → 1 (found at index 1, 1-origin per ISO 13751)
    Value* lhs = machine->heap->allocate_scalar(5.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);

    fn_index_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, MemberOfFound) {
    // 3 ∊ 1 2 3 4 5 → 1
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Eigen::VectorXd set(5);
    set << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* rhs = machine->heap->allocate_vector(set);

    fn_member_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, MemberOfNotFound) {
    // 7 ∊ 1 2 3 → 0
    Value* lhs = machine->heap->allocate_scalar(7.0);
    Eigen::VectorXd set(3);
    set << 1.0, 2.0, 3.0;
    Value* rhs = machine->heap->allocate_vector(set);

    fn_member_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, MemberOfVector) {
    // 1 5 3 7 ∊ 1 2 3 → 1 0 1 0
    Eigen::VectorXd query(4);
    query << 1.0, 5.0, 3.0, 7.0;
    Eigen::VectorXd set(3);
    set << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(query);
    Value* rhs = machine->heap->allocate_vector(set);

    fn_member_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);  // 1 is in set
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);  // 5 is not in set
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);  // 3 is in set
    EXPECT_DOUBLE_EQ((*res)(3, 0), 0.0);  // 7 is not in set
}

TEST_F(PrimitivesTest, EnlistVector) {
    // ∊ 1 2 3 → 1 2 3 (same as ravel for simple arrays)
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_enlist(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(PrimitivesTest, EnlistScalar) {
    // ∊ 5 → 5 (1-element vector)
    Value* scalar = machine->heap->allocate_scalar(5.0);

    fn_enlist(machine, scalar);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 1);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);
}

TEST_F(PrimitivesTest, EnlistMatrix) {
    // ISO 8.2.6: ∊ (2 3⍴⍳6) → 1 2 3 4 5 6 (ravel for simple arrays)
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_enlist(machine, mat);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 6);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(5, 0), 6.0);
}

TEST_F(PrimitivesTest, EnlistEmptyVector) {
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

TEST_F(PrimitivesTest, RavelScalar) {
    // ISO 8.2.1: ,5 → 1-element vector containing 5
    Value* scalar = machine->heap->allocate_scalar(5.0);
    fn_ravel(machine, scalar);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 1);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);
}

TEST_F(PrimitivesTest, RavelVector) {
    // ISO 8.2.1: ,1 2 3 → same vector (identity for vectors)
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_ravel(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

// --- Shape Edge Cases (Section 8.2.2) ---

TEST_F(PrimitivesTest, ShapeMatrix) {
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

TEST_F(PrimitivesTest, DepthScalar) {
    // ISO 8.2.5: ≡5 → 0 (simple scalar)
    Value* scalar = machine->heap->allocate_scalar(5.0);
    fn_depth(machine, scalar);

    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, DepthVector) {
    // ISO 8.2.5: ≡1 2 3 → 1 (simple array)
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_depth(machine, vec);

    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, DepthMatrix) {
    // ISO 8.2.5: ≡ (2 3⍴⍳6) → 1 (simple array)
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);
    fn_depth(machine, mat);

    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, DepthEmptyVector) {
    // ISO 8.2.5: ≡⍳0 → 1 (empty array still has depth 1)
    Value* result = machine->eval("≡⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// --- Table Edge Cases (Section 8.2.4) ---

TEST_F(PrimitivesTest, TableEmptyVector) {
    // ISO 8.2.4: ⍪⍳0 → 0×1 matrix
    Value* result = machine->eval("⍪⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 0);
    EXPECT_EQ(mat->cols(), 1);
}

// --- Reshape Edge Cases (Section 8.3.1) ---

TEST_F(PrimitivesTest, ReshapeToScalar) {
    // ISO 8.3.1: (⍳0)⍴5 → scalar 5 (empty shape produces scalar)
    Value* result = machine->eval("(⍳0)⍴5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, ReshapeZeroLength) {
    // ISO 8.3.1: 0⍴5 → empty vector
    Value* result = machine->eval("0⍴5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, ReshapeZeroMatrix) {
    // ISO 8.3.1: 0 3⍴5 → 0×3 matrix (empty rows)
    Value* result = machine->eval("0 3⍴5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 0);
    EXPECT_EQ(mat->cols(), 3);
}

// --- Join/Catenate Edge Cases (Section 8.3.2) ---

TEST_F(PrimitivesTest, CatenateScalarScalar) {
    // ISO 8.3.2: 5,3 → 5 3 (two-element vector)
    Value* result = machine->eval("5,3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 3.0);
}

TEST_F(PrimitivesTest, CatenateScalarVector) {
    // ISO 8.3.2: 5,1 2 3 → 5 1 2 3
    Value* result = machine->eval("5,1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(3, 0), 3.0);
}

TEST_F(PrimitivesTest, CatenateVectorScalar) {
    // ISO 8.3.2: 1 2 3,5 → 1 2 3 5
    Value* result = machine->eval("1 2 3,5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(3, 0), 5.0);
}

TEST_F(PrimitivesTest, SearchFunctionsRegistered) {
    // ⍳ should already be registered (monadic iota)
    ASSERT_NE(machine->env->lookup("⍳"), nullptr);
    ASSERT_NE(machine->env->lookup("∊"), nullptr);
}

// ============================================================================
// Grade Functions (⍋ ⍒)
// ============================================================================

TEST_F(PrimitivesTest, GradeUpVector) {
    // ⍋ 3 1 4 1 5 → 2 4 1 3 5 (indices for ascending order, 1-origin per ISO 13751)
    Eigen::VectorXd v(5);
    v << 3.0, 1.0, 4.0, 1.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_up(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 2.0);  // index 2 (value 1)
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);  // index 4 (value 1)
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);  // index 1 (value 3)
    EXPECT_DOUBLE_EQ((*res)(3, 0), 3.0);  // index 3 (value 4)
    EXPECT_DOUBLE_EQ((*res)(4, 0), 5.0);  // index 5 (value 5)
}

TEST_F(PrimitivesTest, GradeDownVector) {
    // ⍒ 3 1 4 1 5 → 5 3 1 2 4 (indices for descending order, 1-origin per ISO 13751)
    Eigen::VectorXd v(5);
    v << 3.0, 1.0, 4.0, 1.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_down(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 5.0);  // index 5 (value 5)
    EXPECT_DOUBLE_EQ((*res)(1, 0), 3.0);  // index 3 (value 4)
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);  // index 1 (value 3)
    EXPECT_DOUBLE_EQ((*res)(3, 0), 2.0);  // index 2 (value 1)
    EXPECT_DOUBLE_EQ((*res)(4, 0), 4.0);  // index 4 (value 1)
}

TEST_F(PrimitivesTest, GradeUpScalarError) {
    // ⍋ 5 → RANK ERROR (grade requires array per ISO 13751)
    Value* scalar = machine->heap->allocate_scalar(5.0);

    fn_grade_up(machine, scalar);

    // Should have pushed ThrowErrorK
    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(PrimitivesTest, GradeDownScalarError) {
    // ⍒ 5 → RANK ERROR (grade requires array per ISO 13751)
    Value* scalar = machine->heap->allocate_scalar(5.0);

    fn_grade_down(machine, scalar);

    // Should have pushed ThrowErrorK
    ASSERT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(PrimitivesTest, GradeUpAlreadySorted) {
    // ⍋ 1 2 3 4 5 → 1 2 3 4 5 (1-origin)
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_up(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*res)(i, 0), static_cast<double>(i + 1));
    }
}

TEST_F(PrimitivesTest, GradeDownReversed) {
    // ⍒ 1 2 3 4 5 → 5 4 3 2 1 (1-origin)
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_down(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*res)(i, 0), static_cast<double>(5 - i));
    }
}

TEST_F(PrimitivesTest, GradeFunctionsRegistered) {
    ASSERT_NE(machine->env->lookup("⍋"), nullptr);
    ASSERT_NE(machine->env->lookup("⍒"), nullptr);
}

// --- ISO 10.1.2/10.1.3 Grade Stability Tests ---
// "The indices of identical elements of B occur in Z in ascending order"

TEST_F(PrimitivesTest, GradeUpStable) {
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

TEST_F(PrimitivesTest, GradeDownStable) {
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

TEST_F(PrimitivesTest, GradeUpAllEqual) {
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

// ============================================================================
// Replicate Function (/)
// ============================================================================

TEST_F(PrimitivesTest, ReplicateBasic) {
    // 2 0 3 / 1 2 3 → 1 1 3 3 3
    Eigen::VectorXd counts(3);
    counts << 2.0, 0.0, 3.0;
    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(counts);
    Value* rhs = machine->heap->allocate_vector(data);

    fn_replicate(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);  // 2+0+3 = 5
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 3.0);
}

TEST_F(PrimitivesTest, ReplicateCompress) {
    // 1 0 1 0 1 / 10 20 30 40 50 → 10 30 50 (filter)
    Eigen::VectorXd counts(5);
    counts << 1.0, 0.0, 1.0, 0.0, 1.0;
    Eigen::VectorXd data(5);
    data << 10.0, 20.0, 30.0, 40.0, 50.0;
    Value* lhs = machine->heap->allocate_vector(counts);
    Value* rhs = machine->heap->allocate_vector(data);

    fn_replicate(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 30.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 50.0);
}

TEST_F(PrimitivesTest, ReplicateAllZero) {
    // 0 0 0 / 1 2 3 → (empty)
    Eigen::VectorXd counts(3);
    counts << 0.0, 0.0, 0.0;
    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(counts);
    Value* rhs = machine->heap->allocate_vector(data);

    fn_replicate(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 0);
}

TEST_F(PrimitivesTest, ReplicateScalar) {
    // 3 / 5 → 5 5 5
    Value* lhs = machine->heap->allocate_scalar(3.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);

    fn_replicate(machine, lhs, rhs);

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

TEST_F(PrimitivesTest, UniqueVector) {
    // ∪ 1 2 2 3 1 4 → 1 2 3 4
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 2.0, 3.0, 1.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_unique(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 4.0);
}

TEST_F(PrimitivesTest, UniqueScalar) {
    // ∪ 5 → 5
    Value* val = machine->heap->allocate_scalar(5.0);

    fn_unique(machine, val);

    EXPECT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, UniqueAllSame) {
    // ∪ 3 3 3 3 → 3
    Eigen::VectorXd v(4);
    v << 3.0, 3.0, 3.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_unique(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 1);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
}

TEST_F(PrimitivesTest, UniqueAlreadyUnique) {
    // ∪ 1 2 3 4 → 1 2 3 4
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_unique(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
}

TEST_F(PrimitivesTest, UnionBasic) {
    // 1 2 3 ∪ 3 4 5 → 1 2 3 4 5
    Eigen::VectorXd left(3), right(3);
    left << 1.0, 2.0, 3.0;
    right << 3.0, 4.0, 5.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_union(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(4, 0), 5.0);
}

TEST_F(PrimitivesTest, UnionNoOverlap) {
    // 1 2 ∪ 3 4 → 1 2 3 4
    Eigen::VectorXd left(2), right(2);
    left << 1.0, 2.0;
    right << 3.0, 4.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_union(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
}

TEST_F(PrimitivesTest, UnionWithDuplicates) {
    // 1 1 2 ∪ 2 3 3 → 1 2 3
    Eigen::VectorXd left(3), right(3);
    left << 1.0, 1.0, 2.0;
    right << 2.0, 3.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_union(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(PrimitivesTest, WithoutBasic) {
    // 1 2 3 4 5 ~ 2 4 → 1 3 5
    Eigen::VectorXd left(5), right(2);
    left << 1.0, 2.0, 3.0, 4.0, 5.0;
    right << 2.0, 4.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_without(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);
}

TEST_F(PrimitivesTest, WithoutNoMatch) {
    // 1 2 3 ~ 4 5 6 → 1 2 3
    Eigen::VectorXd left(3), right(3);
    left << 1.0, 2.0, 3.0;
    right << 4.0, 5.0, 6.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_without(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
}

TEST_F(PrimitivesTest, WithoutAllMatch) {
    // 1 2 3 ~ 1 2 3 → (empty)
    Eigen::VectorXd left(3), right(3);
    left << 1.0, 2.0, 3.0;
    right << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_without(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 0);
}

TEST_F(PrimitivesTest, WithoutPreservesDuplicates) {
    // 1 2 2 3 3 3 ~ 2 → 1 3 3 3
    Eigen::VectorXd left(6), right(1);
    left << 1.0, 2.0, 2.0, 3.0, 3.0, 3.0;
    right << 2.0;
    Value* lhs = machine->heap->allocate_vector(left);
    Value* rhs = machine->heap->allocate_vector(right);

    fn_without(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 3.0);
}

TEST_F(PrimitivesTest, SetFunctionsRegistered) {
    ASSERT_NE(machine->env->lookup("∪"), nullptr);
    // ~ should already be registered for logical not
    ASSERT_NE(machine->env->lookup("~"), nullptr);
}

// ============================================================================
// First (↑ monadic) Tests
// ============================================================================

TEST_F(PrimitivesTest, FirstScalar) {
    Value* scalar = machine->heap->allocate_scalar(42.0);

    fn_first(machine, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 42.0);
}

TEST_F(PrimitivesTest, FirstVector) {
    Eigen::VectorXd v(4);
    v << 10.0, 20.0, 30.0, 40.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_first(machine, vec);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 10.0);
}

TEST_F(PrimitivesTest, FirstMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1, 2, 3,
         4, 5, 6;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_first(machine, mat);

    // First of matrix returns first row as vector
    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(PrimitivesTest, FirstEmptyVector) {
    Eigen::VectorXd v(0);
    Value* vec = machine->heap->allocate_vector(v);

    fn_first(machine, vec);

    // First of empty returns 0 (prototype)
    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, FirstSingleElementVector) {
    Eigen::VectorXd v(1);
    v << 99.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_first(machine, vec);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 99.0);
}

// ============================================================================
// Circular Functions (○) Tests
// ============================================================================

TEST_F(PrimitivesTest, PiTimesScalar) {
    Value* one = machine->heap->allocate_scalar(1.0);

    fn_pi_times(machine, one);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), M_PI, 1e-10);
}

TEST_F(PrimitivesTest, PiTimesVector) {
    Eigen::VectorXd v(3);
    v << 0.5, 1.0, 2.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_pi_times(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_NEAR((*res)(0, 0), M_PI * 0.5, 1e-10);
    EXPECT_NEAR((*res)(1, 0), M_PI, 1e-10);
    EXPECT_NEAR((*res)(2, 0), M_PI * 2.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularSin) {
    // 1○x = sin(x)
    Value* fn_code = machine->heap->allocate_scalar(1.0);
    Value* arg = machine->heap->allocate_scalar(M_PI / 2.0);  // sin(π/2) = 1

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 1.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularCos) {
    // 2○x = cos(x)
    Value* fn_code = machine->heap->allocate_scalar(2.0);
    Value* arg = machine->heap->allocate_scalar(0.0);  // cos(0) = 1

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 1.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularTan) {
    // 3○x = tan(x)
    Value* fn_code = machine->heap->allocate_scalar(3.0);
    Value* arg = machine->heap->allocate_scalar(M_PI / 4.0);  // tan(π/4) = 1

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 1.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularSqrt1MinusX2) {
    // 0○x = sqrt(1-x²)
    Value* fn_code = machine->heap->allocate_scalar(0.0);
    Value* arg = machine->heap->allocate_scalar(0.6);  // sqrt(1-0.36) = sqrt(0.64) = 0.8

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 0.8, 1e-10);
}

TEST_F(PrimitivesTest, CircularAsin) {
    // ¯1○x = asin(x)
    Value* fn_code = machine->heap->allocate_scalar(-1.0);
    Value* arg = machine->heap->allocate_scalar(1.0);  // asin(1) = π/2

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), M_PI / 2.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularAtan) {
    // ¯3○x = atan(x)
    Value* fn_code = machine->heap->allocate_scalar(-3.0);
    Value* arg = machine->heap->allocate_scalar(1.0);  // atan(1) = π/4

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), M_PI / 4.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularSinh) {
    // 5○x = sinh(x)
    Value* fn_code = machine->heap->allocate_scalar(5.0);
    Value* arg = machine->heap->allocate_scalar(0.0);  // sinh(0) = 0

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 0.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularCosh) {
    // 6○x = cosh(x)
    Value* fn_code = machine->heap->allocate_scalar(6.0);
    Value* arg = machine->heap->allocate_scalar(0.0);  // cosh(0) = 1

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 1.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularTanh) {
    // 7○x = tanh(x)
    Value* fn_code = machine->heap->allocate_scalar(7.0);
    Value* arg = machine->heap->allocate_scalar(0.0);  // tanh(0) = 0

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 0.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularSqrt1PlusX2) {
    // 4○x = sqrt(1+x²)
    Value* fn_code = machine->heap->allocate_scalar(4.0);
    Value* arg = machine->heap->allocate_scalar(2.0);  // sqrt(1+4) = sqrt(5) ≈ 2.236

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), std::sqrt(5.0), 1e-10);
}

TEST_F(PrimitivesTest, CircularAcos) {
    // ¯2○x = acos(x)
    Value* fn_code = machine->heap->allocate_scalar(-2.0);
    Value* arg = machine->heap->allocate_scalar(1.0);  // acos(1) = 0

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 0.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularAsinh) {
    // ¯5○x = asinh(x)
    Value* fn_code = machine->heap->allocate_scalar(-5.0);
    Value* arg = machine->heap->allocate_scalar(0.0);  // asinh(0) = 0

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 0.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularAcosh) {
    // ¯6○x = acosh(x)
    Value* fn_code = machine->heap->allocate_scalar(-6.0);
    Value* arg = machine->heap->allocate_scalar(1.0);  // acosh(1) = 0

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 0.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularAtanh) {
    // ¯7○x = atanh(x)
    Value* fn_code = machine->heap->allocate_scalar(-7.0);
    Value* arg = machine->heap->allocate_scalar(0.0);  // atanh(0) = 0

    fn_circular(machine, fn_code, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_NEAR(machine->result->as_scalar(), 0.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularVector) {
    // Apply sin to vector
    Value* fn_code = machine->heap->allocate_scalar(1.0);
    Eigen::VectorXd v(3);
    v << 0.0, M_PI / 6.0, M_PI / 2.0;  // sin: 0, 0.5, 1
    Value* vec = machine->heap->allocate_vector(v);

    fn_circular(machine, fn_code, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_NEAR((*res)(0, 0), 0.0, 1e-10);
    EXPECT_NEAR((*res)(1, 0), 0.5, 1e-10);
    EXPECT_NEAR((*res)(2, 0), 1.0, 1e-10);
}

TEST_F(PrimitivesTest, CircularRegistered) {
    ASSERT_NE(machine->env->lookup("○"), nullptr);
}

// ========================================================================
// Roll (? monadic) Tests
// ========================================================================

TEST_F(PrimitivesTest, RollScalar) {
    // ?6 returns random integer in [1,6] (1-based per ISO 13751)
    Value* arg = machine->heap->allocate_scalar(6.0);
    fn_roll(machine, arg);

    ASSERT_TRUE(machine->result->is_scalar());
    double result = machine->result->as_scalar();
    EXPECT_GE(result, 1.0);  // 1-based per ISO 13751 (⎕IO=1)
    EXPECT_LE(result, 6.0);
    EXPECT_EQ(result, std::floor(result));  // Should be integer
}

TEST_F(PrimitivesTest, RollVector) {
    // ?3 3 3 returns vector of random integers in [1,3] (1-based per ISO 13751)
    Eigen::VectorXd v(3);
    v << 3.0, 3.0, 3.0;
    Value* arg = machine->heap->allocate_vector(v);

    fn_roll(machine, arg);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE((*res)(i, 0), 1.0);  // 1-based per ISO 13751 (⎕IO=1)
        EXPECT_LE((*res)(i, 0), 3.0);
        EXPECT_EQ((*res)(i, 0), std::floor((*res)(i, 0)));
    }
}

TEST_F(PrimitivesTest, RollErrorZero) {
    // ?0 is an error
    Value* arg = machine->heap->allocate_scalar(0.0);
    fn_roll(machine, arg);

    // Should push error continuation
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(PrimitivesTest, RollErrorNegative) {
    // ?¯5 is an error
    Value* arg = machine->heap->allocate_scalar(-5.0);
    fn_roll(machine, arg);

    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// ========================================================================
// Deal (? dyadic) Tests
// ========================================================================

TEST_F(PrimitivesTest, DealBasic) {
    // 3?10 returns 3 unique random integers from [1,10] (1-based per ISO 13751)
    Value* count = machine->heap->allocate_scalar(3.0);
    Value* range = machine->heap->allocate_scalar(10.0);

    fn_deal(machine, count, range);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);

    // All values should be in range [1,10] (1-based per ISO 13751)
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE((*res)(i, 0), 1.0);  // 1-based per ISO 13751 (⎕IO=1)
        EXPECT_LE((*res)(i, 0), 10.0);
    }

    // All values should be unique
    double v0 = (*res)(0, 0);
    double v1 = (*res)(1, 0);
    double v2 = (*res)(2, 0);
    EXPECT_NE(v0, v1);
    EXPECT_NE(v0, v2);
    EXPECT_NE(v1, v2);
}

TEST_F(PrimitivesTest, DealEmpty) {
    // 0?5 returns empty vector
    Value* count = machine->heap->allocate_scalar(0.0);
    Value* range = machine->heap->allocate_scalar(5.0);

    fn_deal(machine, count, range);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 0);
}

TEST_F(PrimitivesTest, DealAll) {
    // 5?5 returns all 5 values (permutation of 1-5 per ISO 13751)
    Value* count = machine->heap->allocate_scalar(5.0);
    Value* range = machine->heap->allocate_scalar(5.0);

    fn_deal(machine, count, range);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);

    // All values 1-5 should appear exactly once (1-based per ISO 13751)
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(5);
    for (int i = 0; i < 5; ++i) {
        int val = static_cast<int>((*res)(i, 0));
        EXPECT_GE(val, 1);  // 1-based per ISO 13751 (⎕IO=1)
        EXPECT_LE(val, 5);
        counts(val - 1) += 1.0;  // Adjust for 1-based
    }
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(counts(i), 1.0);
    }
}

TEST_F(PrimitivesTest, DealErrorTooMany) {
    // 6?5 is an error (can't deal 6 from 5)
    Value* count = machine->heap->allocate_scalar(6.0);
    Value* range = machine->heap->allocate_scalar(5.0);

    fn_deal(machine, count, range);

    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

// ========================================================================
// Expand (\ dyadic) Tests
// ========================================================================

TEST_F(PrimitivesTest, ExpandBasic) {
    // 1 0 1 1 \ 1 2 3 → 1 0 2 3
    Eigen::VectorXd mask(4);
    mask << 1.0, 0.0, 1.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);  // Fill element
    EXPECT_DOUBLE_EQ((*res)(2, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 3.0);
}

TEST_F(PrimitivesTest, ExpandAllOnes) {
    // 1 1 1 \ 1 2 3 → 1 2 3 (identity)
    Eigen::VectorXd mask(3);
    mask << 1.0, 1.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(PrimitivesTest, ExpandLeadingZeros) {
    // 0 0 1 1 \ 1 2 → 0 0 1 2
    Eigen::VectorXd mask(4);
    mask << 0.0, 0.0, 1.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(2);
    data << 1.0, 2.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 2.0);
}

TEST_F(PrimitivesTest, ExpandScalar) {
    // 0 1 0 \ 5 → 0 5 0
    Eigen::VectorXd mask(3);
    mask << 0.0, 1.0, 0.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Value* data_val = machine->heap->allocate_scalar(5.0);

    fn_expand(machine, mask_val, data_val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 0.0);
}

TEST_F(PrimitivesTest, ExpandLengthError) {
    // 1 0 1 \ 1 2 3 is error (2 ones vs 3 elements)
    Eigen::VectorXd mask(3);
    mask << 1.0, 0.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, mask_val, data_val);

    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(PrimitivesTest, ExpandDomainError) {
    // 1 2 1 \ 1 2 is error (non-boolean mask)
    Eigen::VectorXd mask(3);
    mask << 1.0, 2.0, 1.0;
    Value* mask_val = machine->heap->allocate_vector(mask);

    Eigen::VectorXd data(2);
    data << 1.0, 2.0;
    Value* data_val = machine->heap->allocate_vector(data);

    fn_expand(machine, mask_val, data_val);

    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
}

TEST_F(PrimitivesTest, ExpandAllZeros) {
    // ISO 10.2.6 example: 0 0\5 → empty vector
    Eigen::VectorXd mask(2);
    mask << 0.0, 0.0;
    Value* mask_val = machine->heap->allocate_vector(mask);
    Value* data_val = machine->heap->allocate_scalar(5.0);

    fn_expand(machine, mask_val, data_val);

    // +/0 0 = 0, so B must have 0 elements (scalar 5 is treated as 0-element vector)
    // Result should be empty (all zeros filled)
    ASSERT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->size(), 2);
    // All elements should be fill value (0)
    EXPECT_DOUBLE_EQ(machine->result->as_matrix()->operator()(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(machine->result->as_matrix()->operator()(1, 0), 0.0);
}

TEST_F(PrimitivesTest, QuestionRegistered) {
    ASSERT_NE(machine->env->lookup("?"), nullptr);
}

// ========================================================================
// Decode (⊥ dyadic) Tests
// ========================================================================

TEST_F(PrimitivesTest, DecodeBinary) {
    // 2⊥1 0 1 1 → 11 (binary 1011 = 11)
    Value* radix = machine->heap->allocate_scalar(2.0);
    Eigen::VectorXd digits(4);
    digits << 1.0, 0.0, 1.0, 1.0;
    Value* digits_val = machine->heap->allocate_vector(digits);

    fn_decode(machine, radix, digits_val);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 11.0);
}

TEST_F(PrimitivesTest, DecodeDecimal) {
    // 10⊥1 2 3 → 123
    Value* radix = machine->heap->allocate_scalar(10.0);
    Eigen::VectorXd digits(3);
    digits << 1.0, 2.0, 3.0;
    Value* digits_val = machine->heap->allocate_vector(digits);

    fn_decode(machine, radix, digits_val);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 123.0);
}

TEST_F(PrimitivesTest, DecodeMixedRadix) {
    // 24 60 60⊥1 30 45 → 5445 (1h 30m 45s in seconds)
    Eigen::VectorXd radix(3);
    radix << 24.0, 60.0, 60.0;
    Value* radix_val = machine->heap->allocate_vector(radix);

    Eigen::VectorXd digits(3);
    digits << 1.0, 30.0, 45.0;
    Value* digits_val = machine->heap->allocate_vector(digits);

    fn_decode(machine, radix_val, digits_val);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5445.0);
}

TEST_F(PrimitivesTest, DecodeEmpty) {
    // Empty decode returns 0
    Value* radix = machine->heap->allocate_scalar(10.0);
    Value* digits_val = machine->heap->allocate_vector(Eigen::VectorXd(0));

    fn_decode(machine, radix, digits_val);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

// ========================================================================
// Encode (⊤ dyadic) Tests
// ========================================================================

TEST_F(PrimitivesTest, EncodeBinary) {
    // 2 2 2 2⊤11 → 1 0 1 1
    Eigen::VectorXd radix(4);
    radix << 2.0, 2.0, 2.0, 2.0;
    Value* radix_val = machine->heap->allocate_vector(radix);
    Value* val = machine->heap->allocate_scalar(11.0);

    fn_encode(machine, radix_val, val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 4);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(3, 0), 1.0);
}

TEST_F(PrimitivesTest, EncodeDecimal) {
    // 10 10 10⊤345 → 3 4 5
    Eigen::VectorXd radix(3);
    radix << 10.0, 10.0, 10.0;
    Value* radix_val = machine->heap->allocate_vector(radix);
    Value* val = machine->heap->allocate_scalar(345.0);

    fn_encode(machine, radix_val, val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);
}

TEST_F(PrimitivesTest, EncodeMixedRadix) {
    // 24 60 60⊤5445 → 1 30 45 (seconds to h:m:s)
    Eigen::VectorXd radix(3);
    radix << 24.0, 60.0, 60.0;
    Value* radix_val = machine->heap->allocate_vector(radix);
    Value* val = machine->heap->allocate_scalar(5445.0);

    fn_encode(machine, radix_val, val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 30.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 45.0);
}

TEST_F(PrimitivesTest, EncodeOverflow) {
    // 2 2 2⊤15 → 1 1 1 (only last 3 bits, overflow discarded)
    Eigen::VectorXd radix(3);
    radix << 2.0, 2.0, 2.0;
    Value* radix_val = machine->heap->allocate_vector(radix);
    Value* val = machine->heap->allocate_scalar(15.0);  // 1111 in binary

    fn_encode(machine, radix_val, val);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 1.0);
}

TEST_F(PrimitivesTest, DecodeEncodeRoundtrip) {
    // Encode then decode should give original value (within radix range)
    Eigen::VectorXd radix(4);
    radix << 2.0, 2.0, 2.0, 2.0;
    Value* radix_val = machine->heap->allocate_vector(radix);
    Value* original = machine->heap->allocate_scalar(13.0);

    // Encode: 13 → 1 1 0 1
    fn_encode(machine, radix_val, original);
    Value* encoded = machine->result;

    // Decode: 1 1 0 1 → 13
    fn_decode(machine, machine->heap->allocate_scalar(2.0), encoded);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 13.0);
}

TEST_F(PrimitivesTest, DecodeRegistered) {
    ASSERT_NE(machine->env->lookup("⊥"), nullptr);
}

TEST_F(PrimitivesTest, EncodeRegistered) {
    ASSERT_NE(machine->env->lookup("⊤"), nullptr);
}

// ============================================================================
// Matrix Inverse (⌹) monadic tests
// ============================================================================

TEST_F(PrimitivesTest, MatrixInverseScalar) {
    // ⌹4 → 0.25 (reciprocal)
    Value* val = machine->heap->allocate_scalar(4.0);
    fn_matrix_inverse(machine, val);
    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.25);
}

TEST_F(PrimitivesTest, MatrixInverseScalarZeroError) {
    // ⌹0 → DOMAIN ERROR
    Value* val = machine->heap->allocate_scalar(0.0);
    fn_matrix_inverse(machine, val);
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

TEST_F(PrimitivesTest, MatrixInverse2x2) {
    // Inverse of [[1,2],[3,4]]
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Value* val = machine->heap->allocate_matrix(mat);
    fn_matrix_inverse(machine, val);
    ASSERT_FALSE(machine->result->is_scalar());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 2);
    EXPECT_EQ(res->cols(), 2);
    // Check A * A^-1 ≈ I
    Eigen::MatrixXd product = mat * (*res);
    EXPECT_NEAR(product(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(product(0, 1), 0.0, 1e-10);
    EXPECT_NEAR(product(1, 0), 0.0, 1e-10);
    EXPECT_NEAR(product(1, 1), 1.0, 1e-10);
}

TEST_F(PrimitivesTest, MatrixInverseVector) {
    // Pseudoinverse of vector
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* val = machine->heap->allocate_vector(vec);
    fn_matrix_inverse(machine, val);
    // Should return a matrix (1x3 pseudoinverse)
    ASSERT_FALSE(machine->result->is_scalar());
}

TEST_F(PrimitivesTest, MatrixInverseSingular) {
    // Singular matrix: [[1,2],[2,4]] (rows are linearly dependent)
    // ISO 10.1.6 says "generalisation of matrix inverse" - pseudoinverse
    // Pseudoinverse is well-defined for singular matrices
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 2, 4;  // Row 2 = 2 * Row 1
    Value* val = machine->heap->allocate_matrix(mat);
    fn_matrix_inverse(machine, val);
    // Pseudoinverse should succeed (not error)
    ASSERT_NE(machine->result, nullptr);
    EXPECT_TRUE(machine->result->is_matrix());
}

// ============================================================================
// Matrix Divide (⌹) dyadic tests
// ============================================================================

TEST_F(PrimitivesTest, MatrixDivideScalarScalar) {
    // 6 ⌹ 2 → 3
    Value* lhs = machine->heap->allocate_scalar(6.0);
    Value* rhs = machine->heap->allocate_scalar(2.0);
    fn_matrix_divide(machine, lhs, rhs);
    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(PrimitivesTest, MatrixDivideByZeroError) {
    // 6 ⌹ 0 → DOMAIN ERROR
    Value* lhs = machine->heap->allocate_scalar(6.0);
    Value* rhs = machine->heap->allocate_scalar(0.0);
    fn_matrix_divide(machine, lhs, rhs);
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

TEST_F(PrimitivesTest, MatrixDivideVectorByScalar) {
    // (1 2 3) ⌹ 2 → 0.5 1 1.5
    Eigen::VectorXd vec(3);
    vec << 2, 4, 6;
    Value* lhs = machine->heap->allocate_vector(vec);
    Value* rhs = machine->heap->allocate_scalar(2.0);
    fn_matrix_divide(machine, lhs, rhs);
    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(PrimitivesTest, MatrixDivideLinearSystem) {
    // Solve A*x = b where A = [[1,0],[0,1]], b = [3,4]
    // Solution: x = [3,4]
    Eigen::MatrixXd A(2, 2);
    A << 1, 0, 0, 1;
    Eigen::VectorXd b(2);
    b << 3, 4;
    Value* lhs = machine->heap->allocate_vector(b);
    Value* rhs = machine->heap->allocate_matrix(A);
    fn_matrix_divide(machine, lhs, rhs);
    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_NEAR((*res)(0, 0), 3.0, 1e-10);
    EXPECT_NEAR((*res)(1, 0), 4.0, 1e-10);
}

// ============================================================================
// Dyadic Transpose (⍉) tests
// ============================================================================

TEST_F(PrimitivesTest, DyadicTransposeScalar) {
    // 0⍉5 → 5 (scalar unchanged)
    Value* lhs = machine->heap->allocate_scalar(0.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);
    fn_dyadic_transpose(machine, lhs, rhs);
    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, DyadicTransposeVectorIdentity) {
    // 0⍉(1 2 3) → 1 2 3
    Value* lhs = machine->heap->allocate_scalar(0.0);
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* rhs = machine->heap->allocate_vector(vec);
    fn_dyadic_transpose(machine, lhs, rhs);
    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
}

TEST_F(PrimitivesTest, DyadicTransposeMatrixIdentity) {
    // 0 1⍉M → M (identity permutation)
    Eigen::VectorXd perm(2);
    perm << 0, 1;
    Value* lhs = machine->heap->allocate_vector(perm);
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    fn_dyadic_transpose(machine, lhs, rhs);
    ASSERT_FALSE(machine->result->is_scalar());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 2);
    EXPECT_EQ(res->cols(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(1, 2), 6.0);
}

TEST_F(PrimitivesTest, DyadicTransposeMatrixSwap) {
    // 1 0⍉M → transpose
    Eigen::VectorXd perm(2);
    perm << 1, 0;
    Value* lhs = machine->heap->allocate_vector(perm);
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    fn_dyadic_transpose(machine, lhs, rhs);
    ASSERT_FALSE(machine->result->is_scalar());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_EQ(res->cols(), 2);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 4.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 1), 6.0);
}

TEST_F(PrimitivesTest, DyadicTransposeInvalidPermError) {
    // 2 2⍉M → DOMAIN ERROR
    Eigen::VectorXd perm(2);
    perm << 2, 2;
    Value* lhs = machine->heap->allocate_vector(perm);
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    fn_dyadic_transpose(machine, lhs, rhs);
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

TEST_F(PrimitivesTest, DominoRegistered) {
    ASSERT_NE(machine->env->lookup("⌹"), nullptr);
}

// ============================================================================
// Execute (⍎) tests
// ============================================================================

TEST_F(PrimitivesTest, ExecuteRequiresString) {
    // ⍎5 → DOMAIN ERROR (not a string)
    Value* val = machine->heap->allocate_scalar(5.0);
    fn_execute(machine, val);
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

TEST_F(PrimitivesTest, ExecutePushesContination) {
    // ⍎'42' should push a continuation
    Value* str = machine->heap->allocate_string("42");
    size_t stack_before = machine->kont_stack.size();
    fn_execute(machine, str);
    EXPECT_GT(machine->kont_stack.size(), stack_before);
}

TEST_F(PrimitivesTest, ExecuteRegistered) {
    ASSERT_NE(machine->env->lookup("⍎"), nullptr);
}

TEST_F(PrimitivesTest, ExecuteEmptyString) {
    // ⍎'' → zilde (empty numeric vector)
    Value* result = machine->eval("⍎''");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

// ============================================================================
// Squad (Indexing) Tests - ⌷
// ============================================================================

TEST_F(PrimitivesTest, SquadRegistered) {
    // ⌷ should be registered in the environment
    ASSERT_NE(machine->env->lookup("⌷"), nullptr);
}

TEST_F(PrimitivesTest, SquadVectorScalarIndex) {
    // (1 2 3 4 5)[3] → 3  (1-based indexing)
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(3.0);

    fn_squad(machine, arr, idx);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(PrimitivesTest, SquadVectorVectorIndex) {
    // (10 20 30 40 50)[2 4] → 20 40
    Eigen::VectorXd v(5);
    v << 10.0, 20.0, 30.0, 40.0, 50.0;
    Value* arr = machine->heap->allocate_vector(v);

    Eigen::VectorXd idx_v(2);
    idx_v << 2.0, 4.0;
    Value* idx = machine->heap->allocate_vector(idx_v);

    fn_squad(machine, arr, idx);

    ASSERT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->data.matrix->size(), 2);
    EXPECT_DOUBLE_EQ((*machine->result->data.matrix)(0, 0), 20.0);
    EXPECT_DOUBLE_EQ((*machine->result->data.matrix)(1, 0), 40.0);
}

TEST_F(PrimitivesTest, SquadVectorFirstElement) {
    // (5 6 7)[1] → 5  (first element, 1-based)
    Eigen::VectorXd v(3);
    v << 5.0, 6.0, 7.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(1.0);

    fn_squad(machine, arr, idx);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 5.0);
}

TEST_F(PrimitivesTest, SquadVectorLastElement) {
    // (5 6 7)[3] → 7  (last element)
    Eigen::VectorXd v(3);
    v << 5.0, 6.0, 7.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(3.0);

    fn_squad(machine, arr, idx);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 7.0);
}

TEST_F(PrimitivesTest, SquadOutOfBoundsError) {
    // (1 2 3)[5] → INDEX ERROR
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(5.0);

    fn_squad(machine, arr, idx);

    // Should push ThrowErrorK
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

TEST_F(PrimitivesTest, SquadZeroIndexError) {
    // (1 2 3)[0] → INDEX ERROR (APL is 1-based)
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* arr = machine->heap->allocate_vector(v);
    Value* idx = machine->heap->allocate_scalar(0.0);

    fn_squad(machine, arr, idx);

    // Should push ThrowErrorK
    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

// ============================================================================
// Squad String Indexing Tests
// ============================================================================

TEST_F(PrimitivesTest, SquadStringScalarIndex) {
    // 'hello'[2] → 101 (ASCII 'e')
    Value* str = machine->heap->allocate_string("hello");
    Value* idx = machine->heap->allocate_scalar(2.0);

    fn_squad(machine, str, idx);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 101.0);  // 'e'
}

TEST_F(PrimitivesTest, SquadStringFirstChar) {
    // 'hello'[1] → 104 (ASCII 'h')
    Value* str = machine->heap->allocate_string("hello");
    Value* idx = machine->heap->allocate_scalar(1.0);

    fn_squad(machine, str, idx);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 104.0);  // 'h'
}

TEST_F(PrimitivesTest, SquadStringLastChar) {
    // 'hello'[5] → 111 (ASCII 'o')
    Value* str = machine->heap->allocate_string("hello");
    Value* idx = machine->heap->allocate_scalar(5.0);

    fn_squad(machine, str, idx);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 111.0);  // 'o'
}

TEST_F(PrimitivesTest, SquadStringVectorIndex) {
    // 'hello'[1 3 5] → 104 108 111 (ASCII for 'hlo')
    Value* str = machine->heap->allocate_string("hello");
    Eigen::VectorXd idx_v(3);
    idx_v << 1.0, 3.0, 5.0;
    Value* idx = machine->heap->allocate_vector(idx_v);

    fn_squad(machine, str, idx);

    ASSERT_TRUE(machine->result->is_vector());
    EXPECT_EQ(machine->result->size(), 3);
    auto* mat = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 104.0);  // 'h'
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 108.0);  // 'l'
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 111.0);  // 'o'
}

TEST_F(PrimitivesTest, SquadStringOutOfBoundsError) {
    // 'hi'[5] → INDEX ERROR
    Value* str = machine->heap->allocate_string("hi");
    Value* idx = machine->heap->allocate_scalar(5.0);

    fn_squad(machine, str, idx);

    ASSERT_FALSE(machine->kont_stack.empty());
    auto* k = dynamic_cast<ThrowErrorK*>(machine->kont_stack.back());
    ASSERT_NE(k, nullptr);
}

// ============================================================================
// Bracket Indexing Syntax Tests (via parser)
// ============================================================================

TEST_F(PrimitivesTest, BracketIndexVectorScalar) {
    // (1 2 3 4 5)[3] → 3
    Value* result = machine->eval("(1 2 3 4 5)[3]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(PrimitivesTest, BracketIndexVectorVector) {
    // (10 20 30)[1 3] → 10 30
    Value* result = machine->eval("(10 20 30)[1 3]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->data.matrix->size(), 2);
    EXPECT_DOUBLE_EQ((*result->data.matrix)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*result->data.matrix)(1, 0), 30.0);
}

TEST_F(PrimitivesTest, BracketIndexIota) {
    // (⍳5)[3] → 3
    Value* result = machine->eval("(⍳5)[3]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(PrimitivesTest, BracketIndexVariable) {
    // x←1 2 3 4 5 ⋄ x[2]
    machine->eval("x←1 2 3 4 5");
    Value* result = machine->eval("x[2]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(PrimitivesTest, BracketIndexString) {
    // 'hello'[2] → 101 (ASCII 'e')
    Value* result = machine->eval("'hello'[2]");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 101.0);  // 'e'
}

TEST_F(PrimitivesTest, BracketIndexStringMultiple) {
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

TEST_F(PrimitivesTest, BracketIndexChained) {
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

TEST_F(PrimitivesTest, TableScalar) {
    // ⍪ 5 → 1×1 matrix containing 5
    Value* scalar = machine->heap->allocate_scalar(5.0);
    fn_table(machine, scalar);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_matrix());
    EXPECT_FALSE(result->is_vector());  // Must be matrix, not vector
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 1);
    EXPECT_EQ(mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 5.0);
}

TEST_F(PrimitivesTest, TableVector) {
    // ⍸ 1 2 3 4 → 4×1 matrix
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_table(machine, vec);

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

TEST_F(PrimitivesTest, TableMatrix) {
    // ⍸ (2 3⍴⍳6) → same 2×3 matrix (unchanged for 2D)
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat_val = machine->heap->allocate_matrix(m);
    fn_table(machine, mat_val);

    Value* result = machine->result;
    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 2), 6.0);
}

// ============================================================================
// Domain Error Tests (ISO 13751 Compliance)
// ============================================================================

// --- Division/Reciprocal Domain Errors ---

TEST_F(PrimitivesTest, DomainErrorReciprocalZero) {
    // ÷0 → DOMAIN ERROR
    EXPECT_THROW(machine->eval("÷0"), APLError);
}

TEST_F(PrimitivesTest, DomainErrorDivideByZero) {
    // 5÷0 → DOMAIN ERROR
    EXPECT_THROW(machine->eval("5÷0"), APLError);
}

TEST_F(PrimitivesTest, ZeroDivZeroReturnsOne) {
    // ISO 13751 7.2.4: 0÷0 returns 1 (the identity element)
    Value* result = machine->eval("0÷0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, ZeroPowerZeroReturnsOne) {
    // ISO 13751 7.2.7: 0*0 returns 1
    Value* result = machine->eval("0*0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// --- Logarithm Domain Errors ---

TEST_F(PrimitivesTest, DomainErrorLogZero) {
    // ⍟0 → DOMAIN ERROR
    EXPECT_THROW(machine->eval("⍟0"), APLError);
}

TEST_F(PrimitivesTest, DomainErrorLogBaseOne) {
    // 1⍟5 → DOMAIN ERROR (log base 1 undefined)
    EXPECT_THROW(machine->eval("1⍟5"), APLError);
}

TEST_F(PrimitivesTest, DomainErrorLogNegative) {
    // ⍟¯1 → DOMAIN ERROR (complex result)
    EXPECT_THROW(machine->eval("⍟¯1"), APLError);
}

// --- Factorial Domain Errors ---

TEST_F(PrimitivesTest, DomainErrorFactorialNegInt) {
    // !¯1 → DOMAIN ERROR (negative integer)
    EXPECT_THROW(machine->eval("!¯1"), APLError);
}

TEST_F(PrimitivesTest, DomainErrorFactorialNegInt2) {
    // !¯2 → DOMAIN ERROR
    EXPECT_THROW(machine->eval("!¯2"), APLError);
}

TEST_F(PrimitivesTest, FactorialNegHalfValid) {
    // !¯0.5 → valid (Gamma function: Γ(0.5) = √π)
    Value* result = machine->eval("!¯0.5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    // Γ(0.5) = √π ≈ 1.7724538509
    EXPECT_NEAR(result->as_scalar(), 1.7724538509, 0.0001);
}

// --- Iota Domain Errors ---

TEST_F(PrimitivesTest, DomainErrorIotaNegative) {
    // ⍳¯1 → DOMAIN ERROR
    EXPECT_THROW(machine->eval("⍳¯1"), APLError);
}

TEST_F(PrimitivesTest, DomainErrorIotaNonInteger) {
    // ⍳1.5 → DOMAIN ERROR (non-integer)
    EXPECT_THROW(machine->eval("⍳1.5"), APLError);
}

// ISO 13751 8.2.3: If rank of B > 1, signal rank-error
TEST_F(PrimitivesTest, RankErrorIotaMatrix) {
    // ⍳(2 2⍴1) → RANK ERROR (matrix argument)
    EXPECT_THROW(machine->eval("⍳2 2⍴1"), APLError);
}

// ISO 13751 8.2.3: If count of B ≠ 1, signal length-error
TEST_F(PrimitivesTest, LengthErrorIotaVector) {
    // ⍳1 2 3 → LENGTH ERROR (vector argument)
    EXPECT_THROW(machine->eval("⍳1 2 3"), APLError);
}

TEST_F(PrimitivesTest, IotaZeroValid) {
    // ⍳0 → empty vector (valid)
    Value* result = machine->eval("⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

// --- Roll Domain Errors ---

TEST_F(PrimitivesTest, DomainErrorRollZero) {
    // ?0 → DOMAIN ERROR
    EXPECT_THROW(machine->eval("?0"), APLError);
}

TEST_F(PrimitivesTest, DomainErrorRollNegative) {
    // ?¯1 → DOMAIN ERROR
    EXPECT_THROW(machine->eval("?¯1"), APLError);
}

// --- Grade Rank Errors ---

TEST_F(PrimitivesTest, RankErrorGradeUpScalar) {
    // ⍋5 → RANK ERROR (scalar)
    EXPECT_THROW(machine->eval("⍋5"), APLError);
}

TEST_F(PrimitivesTest, RankErrorGradeDownScalar) {
    // ⍒5 → RANK ERROR (scalar)
    EXPECT_THROW(machine->eval("⍒5"), APLError);
}

// ============================================================================
// Phase 3: Empty Array Handling Tests (ISO 13751)
// ============================================================================

// --- Structural Operations on Empty Arrays ---

TEST_F(PrimitivesTest, ShapeEmptyVector) {
    // ⍴⍳0 → 1-element vector containing 0
    Value* result = machine->eval("⍴⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 0.0);
}

TEST_F(PrimitivesTest, ShapeEmptyMatrix) {
    // ⍴0 3⍴0 → 0 3 (shape of 0×3 matrix)
    Value* result = machine->eval("⍴0 3⍴0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 3.0);
}

TEST_F(PrimitivesTest, RavelEmptyMatrix) {
    // ,0 3⍴0 → empty vector
    Value* result = machine->eval(",0 3⍴0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, CatenateEmptyLeft) {
    // (⍳0),1 2 3 → 1 2 3
    Value* result = machine->eval("(⍳0),1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 3.0);
}

TEST_F(PrimitivesTest, CatenateEmptyRight) {
    // 1 2 3,⍳0 → 1 2 3
    Value* result = machine->eval("1 2 3,⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 3.0);
}

TEST_F(PrimitivesTest, CatenateEmptyBoth) {
    // (⍳0),⍳0 → empty vector
    Value* result = machine->eval("(⍳0),⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, TallyEmpty) {
    // ≢⍳0 → 0
    Value* result = machine->eval("≢⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, ReverseEmpty) {
    // ⌽⍳0 → empty vector
    Value* result = machine->eval("⌽⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, TransposeEmpty) {
    // ⍉0 3⍴0 → 3 0 matrix
    Value* result = machine->eval("⍉0 3⍴0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 0);
}

// --- Arithmetic on Empty Arrays ---

TEST_F(PrimitivesTest, AddScalarEmpty) {
    // 5+⍳0 → empty vector (scalar extension)
    Value* result = machine->eval("5+⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, AddEmptyScalar) {
    // (⍳0)+5 → empty vector
    Value* result = machine->eval("(⍳0)+5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, DivideScalarEmpty) {
    // 5÷⍳0 → empty vector (no domain error!)
    Value* result = machine->eval("5÷⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, AddEmptyEmpty) {
    // (⍳0)+⍳0 → empty vector
    Value* result = machine->eval("(⍳0)+⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, TimesEmptyEmpty) {
    // (⍳0)×⍳0 → empty vector
    Value* result = machine->eval("(⍳0)×⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, NegateEmpty) {
    // -⍳0 → empty vector
    Value* result = machine->eval("-⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, ReciprocalEmpty) {
    // ÷⍳0 → empty vector (no domain error!)
    Value* result = machine->eval("÷⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

// --- Search Functions with Empty Arrays ---

TEST_F(PrimitivesTest, MembershipEmptyRight) {
    // 1 2 3∊⍳0 → 0 0 0 (nothing found in empty set)
    Value* result = machine->eval("1 2 3∊⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 0.0);
}

TEST_F(PrimitivesTest, MembershipEmptyLeft) {
    // (⍳0)∊1 2 3 → empty vector
    Value* result = machine->eval("(⍳0)∊1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, UniqueEmpty) {
    // ∪⍳0 → empty vector
    Value* result = machine->eval("∪⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, GradeUpEmpty) {
    // ⍋⍳0 → empty vector
    Value* result = machine->eval("⍋⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, GradeDownEmpty) {
    // ⍒⍳0 → empty vector
    Value* result = machine->eval("⍒⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

// --- Take/Drop with Empty Arrays ---

TEST_F(PrimitivesTest, TakeFromEmpty) {
    // 3↑⍳0 → 0 0 0 (take pads with zeros)
    Value* result = machine->eval("3↑⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->operator()(2, 0), 0.0);
}

TEST_F(PrimitivesTest, TakeZeroElements) {
    // 0↑1 2 3 → empty vector
    Value* result = machine->eval("0↑1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, DropToEmpty) {
    // 3↓1 2 3 → empty vector
    Value* result = machine->eval("3↓1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, DropFromEmpty) {
    // 3↓⍳0 → empty vector
    Value* result = machine->eval("3↓⍳0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);
}

TEST_F(PrimitivesTest, TakeNegativeOverextend) {
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
// Phase 4: Scalar Extension Systematic Tests (ISO 13751)
// ============================================================================

// Test all 9 argument combinations for all dyadic scalar functions:
// 1. scalar OP scalar     - valid
// 2. scalar OP vector     - valid (scalar extension)
// 3. scalar OP matrix     - valid (scalar extension)
// 4. vector OP scalar     - valid (scalar extension)
// 5. vector OP vector     - valid (same length)
// 6. vector OP matrix     - ERROR (shape mismatch)
// 7. matrix OP scalar     - valid (scalar extension)
// 8. matrix OP vector     - ERROR (shape mismatch)
// 9. matrix OP matrix     - valid (same shape)

TEST_F(PrimitivesTest, ScalarExtensionAllCombinations) {
    // All dyadic scalar functions to test (excluding boolean-only ⍲ ⍱)
    std::vector<std::string> ops = {
        "+", "-", "×", "÷", "*", "⌈", "⌊", "|",
        "=", "≠", "<", ">", "≤", "≥",
        "∧", "∨"
    };

    // Boolean-only operators (⍲ ⍱ require boolean domain per ISO 13751)
    std::vector<std::string> bool_ops = {"⍲", "⍱"};

    // Argument templates: {left, right, should_succeed}
    struct TestCase {
        std::string left;
        std::string right;
        bool should_succeed;
        std::string description;
    };

    // General numeric test cases
    std::vector<TestCase> cases = {
        {"5",           "3",           true,  "scalar-scalar"},
        {"5",           "1 2 3",       true,  "scalar-vector"},
        {"5",           "2 2⍴1 2 3 4", true,  "scalar-matrix"},
        {"1 2 3",       "5",           true,  "vector-scalar"},
        {"1 2 3",       "4 5 6",       true,  "vector-vector"},
        {"1 2 3",       "2 2⍴1 2 3 4", false, "vector-matrix"},
        {"2 2⍴1 2 3 4", "5",           true,  "matrix-scalar"},
        {"2 2⍴1 2 3 4", "1 2 3",       false, "matrix-vector"},
        {"2 2⍴1 2 3 4", "2 2⍴5 6 7 8", true,  "matrix-matrix"},
    };

    // Boolean test cases for ⍲ and ⍱
    std::vector<TestCase> bool_cases = {
        {"1",           "0",           true,  "scalar-scalar"},
        {"1",           "0 1 0",       true,  "scalar-vector"},
        {"1",           "2 2⍴1 0 0 1", true,  "scalar-matrix"},
        {"1 0 1",       "0",           true,  "vector-scalar"},
        {"1 0 1",       "0 1 0",       true,  "vector-vector"},
        {"1 0 1",       "2 2⍴1 0 0 1", false, "vector-matrix"},
        {"2 2⍴1 0 0 1", "1",           true,  "matrix-scalar"},
        {"2 2⍴1 0 0 1", "1 0 1",       false, "matrix-vector"},
        {"2 2⍴1 0 0 1", "2 2⍴0 1 1 0", true,  "matrix-matrix"},
    };

    int total_tests = 0;
    int passed_tests = 0;

    // Test non-boolean operators with general numeric values
    for (const auto& op : ops) {
        for (const auto& tc : cases) {
            total_tests++;
            std::string expr = "(" + tc.left + ")" + op + "(" + tc.right + ")";

            if (tc.should_succeed) {
                try {
                    Value* result = machine->eval(expr);
                    if (result != nullptr) {
                        passed_tests++;
                    } else {
                        ADD_FAILURE() << "NULL result for " << op << " " << tc.description
                                      << ": " << expr;
                    }
                } catch (const std::exception& e) {
                    ADD_FAILURE() << "Unexpected error for " << op << " " << tc.description
                                  << ": " << expr << " - " << e.what();
                }
            } else {
                try {
                    machine->eval(expr);
                    ADD_FAILURE() << "Expected error for " << op << " " << tc.description
                                  << ": " << expr;
                } catch (const APLError&) {
                    passed_tests++;  // Expected error occurred
                }
            }
        }
    }

    // Test boolean operators with boolean values
    for (const auto& op : bool_ops) {
        for (const auto& tc : bool_cases) {
            total_tests++;
            std::string expr = "(" + tc.left + ")" + op + "(" + tc.right + ")";

            if (tc.should_succeed) {
                try {
                    Value* result = machine->eval(expr);
                    if (result != nullptr) {
                        passed_tests++;
                    } else {
                        ADD_FAILURE() << "NULL result for " << op << " " << tc.description
                                      << ": " << expr;
                    }
                } catch (const std::exception& e) {
                    ADD_FAILURE() << "Unexpected error for " << op << " " << tc.description
                                  << ": " << expr << " - " << e.what();
                }
            } else {
                try {
                    machine->eval(expr);
                    ADD_FAILURE() << "Expected error for " << op << " " << tc.description
                                  << ": " << expr;
                } catch (const APLError&) {
                    passed_tests++;  // Expected error occurred
                }
            }
        }
    }

    // Summary: 16 ops × 9 combinations + 2 bool_ops × 9 combinations = 162 sub-tests
    EXPECT_EQ(passed_tests, total_tests)
        << "Failed " << (total_tests - passed_tests) << " of " << total_tests << " tests";
}

// Additional test for vector length mismatch
TEST_F(PrimitivesTest, VectorLengthMismatch) {
    std::vector<std::string> ops = {"+", "-", "×", "÷", "*", "⌈", "⌊", "|",
                                     "=", "≠", "<", ">", "≤", "≥", "∧", "∨"};

    for (const auto& op : ops) {
        std::string expr = "1 2 3" + op + "1 2";
        EXPECT_THROW(machine->eval(expr), APLError)
            << "Expected LENGTH ERROR for mismatched vectors with " << op;
    }
}

// Additional test for matrix shape mismatch
TEST_F(PrimitivesTest, MatrixShapeMismatch) {
    std::vector<std::string> ops = {"+", "-", "×", "÷", "*", "⌈", "⌊", "|",
                                     "=", "≠", "<", ">", "≤", "≥", "∧", "∨"};

    for (const auto& op : ops) {
        std::string expr = "(2 3⍴⍳6)" + op + "(3 2⍴⍳6)";
        EXPECT_THROW(machine->eval(expr), APLError)
            << "Expected LENGTH ERROR for mismatched matrices with " << op;
    }
}

// ============================================================================
// Structural Function Combinations: Catenate First (⍪)
// ============================================================================

TEST_F(PrimitivesTest, CatenateFirstAllCombinations) {
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

TEST_F(PrimitivesTest, IotaIO1) {
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

TEST_F(PrimitivesTest, IotaIO0) {
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

TEST_F(PrimitivesTest, GradeUpIO1) {
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

TEST_F(PrimitivesTest, GradeUpIO0) {
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

TEST_F(PrimitivesTest, GradeDownIO0) {
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

TEST_F(PrimitivesTest, IndexingIO1) {
    // Default ⎕IO=1: (1 2 3)[2] → 2
    EXPECT_EQ(machine->io, 1);
    Value* result = machine->eval("(1 2 3)[2]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

TEST_F(PrimitivesTest, IndexingIO0) {
    // ⎕IO=0: (1 2 3)[0] → 1
    machine->io = 0;
    Value* result = machine->eval("(1 2 3)[0]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

TEST_F(PrimitivesTest, RollIO0) {
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

TEST_F(PrimitivesTest, RollIO1) {
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
// Format (⍕) Primitive Tests - ISO 13751 Section 15.4
// ============================================================================

// Monadic Format - String Passthrough
TEST_F(PrimitivesTest, FormatMonadicStringPassthrough) {
    Value* str = machine->heap->allocate_string("hello");
    fn_format_monadic(machine, str);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "hello");
}

TEST_F(PrimitivesTest, FormatMonadicEmptyString) {
    Value* str = machine->heap->allocate_string("");
    fn_format_monadic(machine, str);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");
}

// Monadic Format - Scalar Formatting
TEST_F(PrimitivesTest, FormatMonadicIntegerScalar) {
    Value* num = machine->heap->allocate_scalar(42.0);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "42");
}

TEST_F(PrimitivesTest, FormatMonadicNegativeInteger) {
    Value* num = machine->heap->allocate_scalar(-5.0);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "¯5");
}

TEST_F(PrimitivesTest, FormatMonadicZero) {
    Value* num = machine->heap->allocate_scalar(0.0);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "0");
}

TEST_F(PrimitivesTest, FormatMonadicFloat) {
    Value* num = machine->heap->allocate_scalar(3.14);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("3.14") != std::string::npos);
}

TEST_F(PrimitivesTest, FormatMonadicNegativeFloat) {
    Value* num = machine->heap->allocate_scalar(-3.14);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("¯3.14") != std::string::npos);
}

// Monadic Format - Vector Formatting
TEST_F(PrimitivesTest, FormatMonadicIntegerVector) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_format_monadic(machine, vec);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "1 2 3");
}

TEST_F(PrimitivesTest, FormatMonadicVectorWithNegatives) {
    Eigen::VectorXd v(3);
    v << -1.0, 2.0, -3.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_format_monadic(machine, vec);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "¯1 2 ¯3");
}

// Monadic Format - Empty Vector
TEST_F(PrimitivesTest, FormatMonadicEmptyVector) {
    Eigen::VectorXd v(0);
    Value* vec = machine->heap->allocate_vector(v);
    fn_format_monadic(machine, vec);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");
}

// Monadic Format - Print Precision
TEST_F(PrimitivesTest, FormatMonadicPrintPrecision3) {
    machine->pp = 3;
    Value* num = machine->heap->allocate_scalar(3.14159265);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.length() <= 6);  // "3.14" or similar
}

TEST_F(PrimitivesTest, FormatMonadicPrintPrecision10) {
    machine->pp = 10;
    Value* num = machine->heap->allocate_scalar(3.14159265);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("3.14159") != std::string::npos);
}

// Monadic Format - Large/Small Numbers (Exponential)
TEST_F(PrimitivesTest, FormatMonadicLargeNumber) {
    Value* num = machine->heap->allocate_scalar(1e15);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

TEST_F(PrimitivesTest, FormatMonadicSmallNumber) {
    Value* num = machine->heap->allocate_scalar(1e-7);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

// Monadic Format - Infinity
TEST_F(PrimitivesTest, FormatMonadicInfinity) {
    Value* num = machine->heap->allocate_scalar(INFINITY);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("∞") != std::string::npos);
}

TEST_F(PrimitivesTest, FormatMonadicNegativeInfinity) {
    Value* num = machine->heap->allocate_scalar(-INFINITY);
    fn_format_monadic(machine, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("¯∞") != std::string::npos);
}

// Monadic Format - Matrix
TEST_F(PrimitivesTest, FormatMonadicMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);
    fn_format_monadic(machine, mat);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("\n") != std::string::npos);
}

// Dyadic Format - Fixed Decimal
TEST_F(PrimitivesTest, FormatDyadicFixedBasic) {
    Eigen::VectorXd spec(2);
    spec << 5.0, 2.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(3.14159);
    fn_format_dyadic(machine, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("3.14") != std::string::npos);
}

TEST_F(PrimitivesTest, FormatDyadicZeroDecimals) {
    Eigen::VectorXd spec(2);
    spec << 5.0, 0.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(42.7);
    fn_format_dyadic(machine, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("43") != std::string::npos);  // Rounds
}

TEST_F(PrimitivesTest, FormatDyadicNegative) {
    Eigen::VectorXd spec(2);
    spec << 6.0, 2.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(-3.14);
    fn_format_dyadic(machine, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 6);
    EXPECT_TRUE(s.find("¯3.14") != std::string::npos);
}

// Dyadic Format - Exponential (negative precision)
TEST_F(PrimitivesTest, FormatDyadicExponential) {
    Eigen::VectorXd spec(2);
    spec << 10.0, -3.0;  // Negative precision = exponential
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(3.14159);
    fn_format_dyadic(machine, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

// Dyadic Format - Vector
TEST_F(PrimitivesTest, FormatDyadicVector) {
    Eigen::VectorXd spec(2);
    spec << 6.0, 2.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* omega = machine->heap->allocate_vector(v);
    fn_format_dyadic(machine, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 18);  // 3 * 6
}
