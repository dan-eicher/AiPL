// Tests for arithmetic primitives

#include <gtest/gtest.h>
#include "primitives.h"
#include "operators.h"
#include "value.h"
#include "environment.h"
#include <cmath>

using namespace apl;
#include "machine.h"

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

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, AddScalarVector) {
    Value* scalar = machine->heap->allocate_scalar(5.0);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_add(machine, scalar, vec);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 8.0);

    // GC will clean up -     delete scalar;
    // GC will clean up -     delete vec;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, AddVectorScalar) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* scalar = machine->heap->allocate_scalar(10.0);

    fn_add(machine, vec, scalar);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 13.0);

    // GC will clean up -     delete vec;
    // GC will clean up -     delete scalar;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, AddVectorVector) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(3);
    v2 << 4.0, 5.0, 6.0;

    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_add(machine, vec1, vec2);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 9.0);

    // GC will clean up -     delete vec1;
    // GC will clean up -     delete vec2;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, SubtractScalars) {
    Value* a = machine->heap->allocate_scalar(10.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_subtract(machine, a, b);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, MultiplyScalars) {
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(4.0);
    fn_multiply(machine, a, b);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, MultiplyScalarVector) {
    Value* scalar = machine->heap->allocate_scalar(2.0);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_multiply(machine, scalar, vec);


    Value* result = machine->ctrl.value;

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 6.0);

    // GC will clean up -     delete scalar;
    // GC will clean up -     delete vec;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, DivideScalars) {
    Value* a = machine->heap->allocate_scalar(12.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_divide(machine, a, b);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, DivideByZeroError) {
    Value* a = machine->heap->allocate_scalar(12.0);
    Value* b = machine->heap->allocate_scalar(0.0);

    // Primitives now push ThrowErrorK instead of throwing C++ exceptions
    fn_divide(machine, a, b);

    // Should have pushed a ThrowErrorK
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
}

TEST_F(PrimitivesTest, PowerScalars) {
    Value* a = machine->heap->allocate_scalar(2.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_power(machine, a, b);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, EqualScalars) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(5.0);
    fn_equal(machine, a, b);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // True

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, NotEqualScalars) {
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_equal(machine, a, b);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // False

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, EqualVectors) {
    Eigen::VectorXd vec_a(3);
    vec_a << 1.0, 2.0, 3.0;
    Eigen::VectorXd vec_b(3);
    vec_b << 1.0, 5.0, 3.0;

    Value* a = machine->heap->allocate_vector(vec_a);
    Value* b = machine->heap->allocate_vector(vec_b);
    fn_equal(machine, a, b);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    ASSERT_EQ(res_mat->size(), 3);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);  // 1=1 is true
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 0.0);  // 2=5 is false
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 1.0);  // 3=3 is true

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete result;
}

// ============================================================================
// Monadic Arithmetic Tests
// ============================================================================

TEST_F(PrimitivesTest, IdentityScalar) {
    Value* a = machine->heap->allocate_scalar(5.0);
    fn_conjugate(machine, a);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, NegateScalar) {
    Value* a = machine->heap->allocate_scalar(5.0);
    fn_negate(machine, a);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, NegateVector) {
    Eigen::VectorXd v(3);
    v << 1.0, -2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_negate(machine, vec);


    Value* result = machine->ctrl.value;

    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 0), -3.0);

    // GC will clean up -     delete vec;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, SignPositive) {
    Value* a = machine->heap->allocate_scalar(5.0);
    fn_signum(machine, a);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, SignNegative) {
    Value* a = machine->heap->allocate_scalar(-5.0);
    fn_signum(machine, a);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -1.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, SignZero) {
    Value* a = machine->heap->allocate_scalar(0.0);
    fn_signum(machine, a);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, ReciprocalScalar) {
    Value* a = machine->heap->allocate_scalar(4.0);
    fn_reciprocal(machine, a);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.25);

    // GC will clean up -     delete a;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, ReciprocalZeroError) {
    Value* a = machine->heap->allocate_scalar(0.0);

    fn_reciprocal(machine, a);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

    // GC will clean up -     delete a;
}

TEST_F(PrimitivesTest, ExponentialScalar) {
    Value* a = machine->heap->allocate_scalar(1.0);
    fn_exponential(machine, a);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_NEAR(result->as_scalar(), M_E, 1e-10);

    // GC will clean up -     delete a;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, ExponentialZero) {
    Value* a = machine->heap->allocate_scalar(0.0);
    fn_exponential(machine, a);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete result;
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


    Value* result = machine->ctrl.value;

    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res)(0, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 8.0);
    EXPECT_DOUBLE_EQ((*res)(1, 0), 10.0);
    EXPECT_DOUBLE_EQ((*res)(1, 1), 12.0);

    // GC will clean up -     delete mat1;
    // GC will clean up -     delete mat2;
    // GC will clean up -     delete result;
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

    // GC will clean up -     delete vec1;
    // GC will clean up -     delete vec2;
}

// ============================================================================
// Array Operation Tests
// ============================================================================

TEST_F(PrimitivesTest, ShapeScalar) {
    Value* scalar = machine->heap->allocate_scalar(5.0);
    fn_shape(machine, scalar);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 0);  // Empty shape for scalar

    // GC will clean up -     delete scalar;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, ShapeVector) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_shape(machine, vec);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* shape = result->as_matrix();
    EXPECT_EQ(shape->rows(), 1);
    EXPECT_DOUBLE_EQ((*shape)(0, 0), 5.0);

    // GC will clean up -     delete vec;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, ReshapeVector) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = machine->heap->allocate_vector(v);

    Eigen::VectorXd new_shape(2);
    new_shape << 2.0, 3.0;
    Value* shape = machine->heap->allocate_vector(new_shape);

    fn_reshape(machine, shape, vec);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ(mat->rows(), 2);
    EXPECT_EQ(mat->cols(), 3);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(0, 1), 3.0);

    // GC will clean up -     delete vec;
    // GC will clean up -     delete shape;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, Ravel) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_ravel(machine, mat);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 6);
    // Column-major order
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*vec)(2, 0), 2.0);

    // GC will clean up -     delete mat;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, Catenate) {
    Eigen::VectorXd v1(3);
    v1 << 1.0, 2.0, 3.0;
    Eigen::VectorXd v2(2);
    v2 << 4.0, 5.0;

    Value* vec1 = machine->heap->allocate_vector(v1);
    Value* vec2 = machine->heap->allocate_vector(v2);

    fn_catenate(machine, vec1, vec2);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(3, 0), 4.0);

    // GC will clean up -     delete vec1;
    // GC will clean up -     delete vec2;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, Transpose) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    fn_transpose(machine, mat);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_EQ(res->cols(), 2);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(0, 1), 4.0);

    // GC will clean up -     delete mat;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, Iota) {
    Value* n = machine->heap->allocate_scalar(5.0);
    fn_iota(machine, n);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_EQ(vec->rows(), 5);
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(1, 0), 1.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 4.0);

    // GC will clean up -     delete n;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, Take) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    Value* count = machine->heap->allocate_scalar(3.0);
    fn_take(machine, count, vec);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 3.0);

    // GC will clean up -     delete vec;
    // GC will clean up -     delete count;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, Drop) {
    Eigen::VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Value* vec = machine->heap->allocate_vector(v);

    Value* count = machine->heap->allocate_scalar(2.0);
    fn_drop(machine, count, vec);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res = result->as_matrix();
    EXPECT_EQ(res->rows(), 3);
    EXPECT_DOUBLE_EQ((*res)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res)(2, 0), 5.0);

    // GC will clean up -     delete vec;
    // GC will clean up -     delete count;
    // GC will clean up -     delete result;
}

// ============================================================================
// Environment and Binding Tests
// ============================================================================

TEST_F(PrimitivesTest, EnvironmentInit) {
    
    init_global_environment(machine);

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

TEST_F(PrimitivesTest, EnvironmentLookup) {
    
    init_global_environment(machine);

    // Lookup and use a primitive from environment
    Value* plus = machine->env->lookup("+");
    ASSERT_NE(plus, nullptr);

    // Use it to add two numbers
    Value* a = machine->heap->allocate_scalar(3.0);
    Value* b = machine->heap->allocate_scalar(4.0);
    PrimitiveFn* fn = plus->data.primitive_fn;
    fn->dyadic(machine, a, b);
    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete result;
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

    // GC will clean up -     delete x;
    // GC will clean up -     delete y;
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

    // GC will clean up -     delete x;
    // GC will clean up -     delete y;
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

    // GC will clean up -     delete vec1;
    // GC will clean up -     delete vec2;
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

    // GC will clean up -     delete mat1;
    // GC will clean up -     delete mat2;
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

    // GC will clean up -     delete vec1;
    // GC will clean up -     delete vec2;
}

TEST_F(PrimitivesTest, ErrorReciprocalVector) {
    Eigen::VectorXd v(3);
    v << 2.0, 0.0, 4.0;  // Has a zero

    Value* vec = machine->heap->allocate_vector(v);

    // Should throw on reciprocal of zero
    fn_reciprocal(machine, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

    // GC will clean up -     delete vec;
}

TEST_F(PrimitivesTest, ErrorReshapeIncompatibleSize) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Try to reshape 6 elements into 2×4 (needs 8 elements)
    Eigen::VectorXd shape(2);
    shape << 2.0, 4.0;
    Value* shape_val = machine->heap->allocate_vector(shape);

    fn_reshape(machine, shape_val, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

    // GC will clean up -     delete vec;
    // GC will clean up -     delete shape_val;
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

    // GC will clean up -     delete vec;
    // GC will clean up -     delete shape_val;
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

    // GC will clean up -     delete vec;
    // GC will clean up -     delete shape_val;
}

TEST_F(PrimitivesTest, ErrorIotaNonScalar) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Iota requires scalar argument
    fn_iota(machine, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

    // GC will clean up -     delete vec;
}

TEST_F(PrimitivesTest, ErrorIotaNegative) {
    Value* neg = machine->heap->allocate_scalar(-5.0);

    // Negative iota doesn't make sense
    fn_iota(machine, neg);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

    // GC will clean up -     delete neg;
}

TEST_F(PrimitivesTest, ErrorIotaNonInteger) {
    Value* frac = machine->heap->allocate_scalar(3.5);

    // Fractional iota doesn't make sense
    fn_iota(machine, frac);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

    // GC will clean up -     delete frac;
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

    // GC will clean up -     delete n_val;
    // GC will clean up -     delete vec;
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

    // GC will clean up -     delete n_val;
    // GC will clean up -     delete vec;
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

    // GC will clean up -     delete vec1;
    // GC will clean up -     delete mat2;
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


    Value* result = machine->ctrl.value;

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

    // GC will clean up -     delete scalar;
    // GC will clean up -     delete mat;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, BroadcastMatrixScalar) {
    Eigen::MatrixXd m(2, 2);
    m << 10.0, 20.0,
         30.0, 40.0;
    Value* mat = machine->heap->allocate_matrix(m);
    Value* scalar = machine->heap->allocate_scalar(3.0);

    fn_multiply(machine, mat, scalar);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 30.0);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 1), 60.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 90.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 1), 120.0);

    // GC will clean up -     delete mat;
    // GC will clean up -     delete scalar;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, BroadcastScalarLargeMatrix) {
    Value* scalar = machine->heap->allocate_scalar(2.0);
    Eigen::MatrixXd m(10, 10);
    m.setConstant(1.0);
    Value* mat = machine->heap->allocate_matrix(m);

    fn_add(machine, scalar, mat);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 10);
    EXPECT_EQ(res_mat->cols(), 10);

    // Check a few elements
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(5, 5), 3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(9, 9), 3.0);

    // GC will clean up -     delete scalar;
    // GC will clean up -     delete mat;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, BroadcastWithNegativeScalar) {
    Value* scalar = machine->heap->allocate_scalar(-5.0);
    Eigen::VectorXd v(4);
    v << 1.0, 2.0, 3.0, 4.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_add(machine, scalar, vec);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), -4.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), -3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), -2.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), -1.0);

    // GC will clean up -     delete scalar;
    // GC will clean up -     delete vec;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, BroadcastZeroScalar) {
    Value* zero = machine->heap->allocate_scalar(0.0);
    Eigen::VectorXd v(3);
    v << 5.0, 10.0, 15.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_multiply(machine, zero, vec);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 0.0);

    // GC will clean up -     delete zero;
    // GC will clean up -     delete vec;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, BroadcastScalarDivideVector) {
    Value* scalar = machine->heap->allocate_scalar(100.0);
    Eigen::VectorXd v(4);
    v << 2.0, 4.0, 5.0, 10.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_divide(machine, scalar, vec);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 50.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 25.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 20.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 10.0);

    // GC will clean up -     delete scalar;
    // GC will clean up -     delete vec;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, BroadcastVectorDivideScalar) {
    Eigen::VectorXd v(4);
    v << 10.0, 20.0, 30.0, 40.0;
    Value* vec = machine->heap->allocate_vector(v);
    Value* scalar = machine->heap->allocate_scalar(2.0);

    fn_divide(machine, vec, scalar);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 10.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 15.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 20.0);

    // GC will clean up -     delete vec;
    // GC will clean up -     delete scalar;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, BroadcastScalarPower) {
    Value* base = machine->heap->allocate_scalar(2.0);
    Eigen::VectorXd exponents(5);
    exponents << 0.0, 1.0, 2.0, 3.0, 4.0;
    Value* exp_vec = machine->heap->allocate_vector(exponents);

    fn_power(machine, base, exp_vec);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*res_mat)(2, 0), 4.0);
    EXPECT_DOUBLE_EQ((*res_mat)(3, 0), 8.0);
    EXPECT_DOUBLE_EQ((*res_mat)(4, 0), 16.0);

    // GC will clean up -     delete base;
    // GC will clean up -     delete exp_vec;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, BroadcastEmptyVector) {
    Value* scalar = machine->heap->allocate_scalar(5.0);
    Eigen::VectorXd v(0);
    Value* empty_vec = machine->heap->allocate_vector(v);

    fn_add(machine, scalar, empty_vec);


    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 0);
    EXPECT_EQ(res_mat->cols(), 1);

    // GC will clean up -     delete scalar;
    // GC will clean up -     delete empty_vec;
    // GC will clean up -     delete result;
}

// ============================================================================
// Primitive Composition Tests
// ============================================================================

TEST_F(PrimitivesTest, CompositionIotaReshape) {
    // ⍳12 → 0 1 2 3 4 5 6 7 8 9 10 11 (0-indexed)
    Value* n = machine->heap->allocate_scalar(12.0);
    fn_iota(machine, n);

    Value* iota_result = machine->ctrl.value;

    // Reshape into 3×4 matrix
    Eigen::VectorXd shape(2);
    shape << 3.0, 4.0;
    Value* shape_val = machine->heap->allocate_vector(shape);
    fn_reshape(machine, shape_val, iota_result);

    Value* reshaped = machine->ctrl.value;

    ASSERT_TRUE(reshaped->is_matrix());
    const Eigen::MatrixXd* mat = reshaped->as_matrix();
    EXPECT_EQ(mat->rows(), 3);
    EXPECT_EQ(mat->cols(), 4);
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(2, 3), 11.0);

    // GC will clean up -     delete n;
    // GC will clean up -     delete iota_result;
    // GC will clean up -     delete shape_val;
    // GC will clean up -     delete reshaped;
}

TEST_F(PrimitivesTest, CompositionReshapeTranspose) {
    Eigen::VectorXd v(6);
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Reshape to 2×3 (column-major fill: [[1,3,5],[2,4,6]])
    Eigen::VectorXd shape(2);
    shape << 2.0, 3.0;
    Value* shape_val = machine->heap->allocate_vector(shape);
    fn_reshape(machine, shape_val, vec);

    Value* mat = machine->ctrl.value;

    // Transpose to 3×2: [[1,2],[3,4],[5,6]]
    fn_transpose(machine, mat);

    Value* transposed = machine->ctrl.value;

    ASSERT_TRUE(transposed->is_matrix());
    const Eigen::MatrixXd* res_mat = transposed->as_matrix();
    EXPECT_EQ(res_mat->rows(), 3);
    EXPECT_EQ(res_mat->cols(), 2);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*res_mat)(1, 1), 4.0);

    // GC will clean up -     delete vec;
    // GC will clean up -     delete shape_val;
    // GC will clean up -     delete mat;
    // GC will clean up -     delete transposed;
}

TEST_F(PrimitivesTest, CompositionRavelCatenate) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);

    // Ravel to vector
    fn_ravel(machine, mat);

    Value* raveled = machine->ctrl.value;

    // Create another vector to catenate
    Eigen::VectorXd v(3);
    v << 7.0, 8.0, 9.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Catenate
    fn_catenate(machine, raveled, vec);

    Value* result = machine->ctrl.value;

    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* res_mat = result->as_matrix();
    EXPECT_EQ(res_mat->rows(), 9);
    EXPECT_EQ(res_mat->cols(), 1);
    EXPECT_DOUBLE_EQ((*res_mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*res_mat)(5, 0), 6.0);
    EXPECT_DOUBLE_EQ((*res_mat)(6, 0), 7.0);
    EXPECT_DOUBLE_EQ((*res_mat)(8, 0), 9.0);

    // GC will clean up -     delete mat;
    // GC will clean up -     delete raveled;
    // GC will clean up -     delete vec;
    // GC will clean up -     delete result;
}

TEST_F(PrimitivesTest, CompositionArithmeticChain) {
    // (5 + 3) × 2
    Value* a = machine->heap->allocate_scalar(5.0);
    Value* b = machine->heap->allocate_scalar(3.0);
    fn_add(machine, a, b);

    Value* sum = machine->ctrl.value;

    Value* c = machine->heap->allocate_scalar(2.0);
    fn_multiply(machine, sum, c);

    Value* product = machine->ctrl.value;

    ASSERT_TRUE(product->is_scalar());
    EXPECT_DOUBLE_EQ(product->as_scalar(), 16.0);

    // GC will clean up -     delete a;
    // GC will clean up -     delete b;
    // GC will clean up -     delete sum;
    // GC will clean up -     delete c;
    // GC will clean up -     delete product;
}

TEST_F(PrimitivesTest, CompositionShapeReshape) {
    Eigen::MatrixXd m(3, 4);
    m.setConstant(1.0);
    Value* mat = machine->heap->allocate_matrix(m);

    // Get shape
    fn_shape(machine, mat);

    Value* shape = machine->ctrl.value;

    // Use that shape to reshape a vector
    Eigen::VectorXd v(12);
    for (int i = 0; i < 12; i++) v(i) = i + 1.0;
    Value* vec = machine->heap->allocate_vector(v);

    fn_reshape(machine, shape, vec);


    Value* reshaped = machine->ctrl.value;

    ASSERT_TRUE(reshaped->is_matrix());
    const Eigen::MatrixXd* res_mat = reshaped->as_matrix();
    EXPECT_EQ(res_mat->rows(), 3);
    EXPECT_EQ(res_mat->cols(), 4);

    // GC will clean up -     delete mat;
    // GC will clean up -     delete shape;
    // GC will clean up -     delete vec;
    // GC will clean up -     delete reshaped;
}

TEST_F(PrimitivesTest, CompositionNestedReduce) {
    // Create matrix
    Eigen::MatrixXd m(3, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0;
    Value* mat = machine->heap->allocate_matrix(m);

    // Reduce along last axis (sum rows)
    Value* func = machine->heap->allocate_primitive(&prim_plus);
    fn_reduce(machine, func, mat);

    Value* row_sums = machine->ctrl.value;

    // Now reduce that result (sum of row sums)
    fn_reduce(machine, func, row_sums);

    Value* total = machine->ctrl.value;

    ASSERT_TRUE(total->is_scalar());
    EXPECT_DOUBLE_EQ(total->as_scalar(), 45.0);  // 1+2+...+9 = 45

    // GC will clean up -     delete mat;
    // GC will clean up -     delete func;
    // GC will clean up -     delete row_sums;
    // GC will clean up -     delete total;
}
