// Tests for arithmetic primitives

#include <gtest/gtest.h>
#include "primitives.h"
#include "operators.h"
#include "value.h"
#include "machine.h"
#include <cmath>

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
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 4.0);

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
    // ⍳12 → 0 1 2 3 4 5 6 7 8 9 10 11 (0-indexed)
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
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 3), 11.0);

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
    // 1 2 3 4 5 ⍳ 3 → 2 (0-origin index)
    Eigen::VectorXd haystack(5);
    haystack << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* lhs = machine->heap->allocate_vector(haystack);
    Value* rhs = machine->heap->allocate_scalar(3.0);

    fn_index_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 2.0);
}

TEST_F(PrimitivesTest, IndexOfNotFound) {
    // 1 2 3 ⍳ 7 → 3 (not found = length of haystack)
    Eigen::VectorXd haystack(3);
    haystack << 1.0, 2.0, 3.0;
    Value* lhs = machine->heap->allocate_vector(haystack);
    Value* rhs = machine->heap->allocate_scalar(7.0);

    fn_index_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 3.0);
}

TEST_F(PrimitivesTest, IndexOfVector) {
    // 10 20 30 40 ⍳ 30 20 99 → 2 1 4
    Eigen::VectorXd haystack(4);
    haystack << 10.0, 20.0, 30.0, 40.0;
    Eigen::VectorXd needles(3);
    needles << 30.0, 20.0, 99.0;
    Value* lhs = machine->heap->allocate_vector(haystack);
    Value* rhs = machine->heap->allocate_vector(needles);

    fn_index_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 2.0);  // 30 found at index 2
    EXPECT_DOUBLE_EQ((*res)(1, 0), 1.0);  // 20 found at index 1
    EXPECT_DOUBLE_EQ((*res)(2, 0), 4.0);  // 99 not found → 4
}

TEST_F(PrimitivesTest, IndexOfScalarHaystack) {
    // 5 ⍳ 5 → 0 (found at index 0)
    Value* lhs = machine->heap->allocate_scalar(5.0);
    Value* rhs = machine->heap->allocate_scalar(5.0);

    fn_index_of(machine, lhs, rhs);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
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

TEST_F(PrimitivesTest, SearchFunctionsRegistered) {
    // ⍳ should already be registered (monadic iota)
    ASSERT_NE(machine->env->lookup("⍳"), nullptr);
    ASSERT_NE(machine->env->lookup("∊"), nullptr);
}

// ============================================================================
// Grade Functions (⍋ ⍒)
// ============================================================================

TEST_F(PrimitivesTest, GradeUpVector) {
    // ⍋ 3 1 4 1 5 → 1 3 0 2 4 (indices for ascending order, 0-origin)
    Eigen::VectorXd v(5);
    v << 3.0, 1.0, 4.0, 1.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_up(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);  // 1 (value 1)
    EXPECT_DOUBLE_EQ((*res)(1, 0), 3.0);  // 3 (value 1)
    EXPECT_DOUBLE_EQ((*res)(2, 0), 0.0);  // 0 (value 3)
    EXPECT_DOUBLE_EQ((*res)(3, 0), 2.0);  // 2 (value 4)
    EXPECT_DOUBLE_EQ((*res)(4, 0), 4.0);  // 4 (value 5)
}

TEST_F(PrimitivesTest, GradeDownVector) {
    // ⍒ 3 1 4 1 5 → 4 2 0 1 3 (indices for descending order, 0-origin)
    Eigen::VectorXd v(5);
    v << 3.0, 1.0, 4.0, 1.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_down(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    EXPECT_EQ(res->rows(), 5);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 4.0);  // 4 (value 5)
    EXPECT_DOUBLE_EQ((*res)(1, 0), 2.0);  // 2 (value 4)
    EXPECT_DOUBLE_EQ((*res)(2, 0), 0.0);  // 0 (value 3)
    EXPECT_DOUBLE_EQ((*res)(3, 0), 1.0);  // 1 (value 1)
    EXPECT_DOUBLE_EQ((*res)(4, 0), 3.0);  // 3 (value 1)
}

TEST_F(PrimitivesTest, GradeUpScalar) {
    // ⍋ 5 → 0 (single element)
    Value* scalar = machine->heap->allocate_scalar(5.0);

    fn_grade_up(machine, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, GradeDownScalar) {
    // ⍒ 5 → 0 (single element)
    Value* scalar = machine->heap->allocate_scalar(5.0);

    fn_grade_down(machine, scalar);

    ASSERT_TRUE(machine->result->is_scalar());
    EXPECT_DOUBLE_EQ(machine->result->as_scalar(), 0.0);
}

TEST_F(PrimitivesTest, GradeUpAlreadySorted) {
    // ⍋ 1 2 3 4 5 → 0 1 2 3 4
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_up(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*res)(i, 0), static_cast<double>(i));
    }
}

TEST_F(PrimitivesTest, GradeDownReversed) {
    // ⍒ 1 2 3 4 5 → 4 3 2 1 0
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_grade_down(machine, vec);

    ASSERT_TRUE(machine->result->is_vector());
    const Eigen::MatrixXd* res = machine->result->as_matrix();
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*res)(i, 0), static_cast<double>(4 - i));
    }
}

TEST_F(PrimitivesTest, GradeFunctionsRegistered) {
    ASSERT_NE(machine->env->lookup("⍋"), nullptr);
    ASSERT_NE(machine->env->lookup("⍒"), nullptr);
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
