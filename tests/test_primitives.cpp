// Miscellaneous Primitive Tests
// Covers: Environment bindings, error handling, domain errors
// Most primitive tests have been split into:
//   test_prim_arithmetic.cpp - arithmetic, comparison, logical functions
//   test_prim_structural.cpp - array manipulation functions
//   test_prim_format.cpp - format (⍕) function

#include <gtest/gtest.h>
#include "primitives.h"
#include "operators.h"
#include "value.h"
#include "machine.h"
#include "continuation.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

using namespace apl;

class PrimitivesTest : public ::testing::Test {
protected:
    Machine* machine;
    void SetUp() override { machine = new Machine(); }
    void TearDown() override { delete machine; }
};

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

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
    fn->dyadic(machine, nullptr, a, b);
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
    fn_add(machine, nullptr, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_subtract(machine, nullptr, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_multiply(machine, nullptr, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_divide(machine, nullptr, vec1, vec2);
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
    fn_add(machine, nullptr, mat1, mat2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_subtract(machine, nullptr, mat1, mat2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);
    machine->kont_stack.clear();

    fn_multiply(machine, nullptr, mat1, mat2);
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
    fn_divide(machine, nullptr, vec1, vec2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorReciprocalVector) {
    Eigen::VectorXd v(3);
    v << 2.0, 0.0, 4.0;  // Has a zero

    Value* vec = machine->heap->allocate_vector(v);

    // Should throw on reciprocal of zero
    fn_reciprocal(machine, nullptr, vec);
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

    fn_reshape(machine, nullptr, shape_val, vec);
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

    fn_reshape(machine, nullptr, shape_val, vec);
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

    fn_reshape(machine, nullptr, shape_val, vec);
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

    fn_reshape(machine, nullptr, shape_val, vec);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

// ISO 13751 8.3.1: If rank of A > 1, signal rank-error
TEST_F(PrimitivesTest, RankErrorReshapeMatrixShape) {
    // (2 2⍴1 2 3 4)⍴⍳6 → RANK ERROR (shape is a matrix, not vector)
    EXPECT_THROW(machine->eval("(2 2⍴1 2 3 4)⍴⍳6"), APLError);
}

TEST_F(PrimitivesTest, IotaMultiDimVector) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);

    // Multi-dimensional iota now accepts vector argument (ISO 13751 §10.1.2)
    fn_iota(machine, nullptr, vec);
    // Should succeed and produce a strand of index tuples
    ASSERT_TRUE(machine->result != nullptr);
    EXPECT_TRUE(machine->result->is_strand());
    // ⍳1 2 3 produces 1×2×3 = 6 index triples
    EXPECT_EQ(machine->result->as_strand()->size(), 6);
}

TEST_F(PrimitivesTest, ErrorIotaNegative) {
    Value* neg = machine->heap->allocate_scalar(-5.0);

    // Negative iota doesn't make sense
    fn_iota(machine, nullptr, neg);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

TEST_F(PrimitivesTest, ErrorIotaNonInteger) {
    Value* frac = machine->heap->allocate_scalar(3.5);

    // Fractional iota doesn't make sense
    fn_iota(machine, nullptr, frac);
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
    fn_take(machine, nullptr, n_val, vec);
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
    fn_drop(machine, nullptr, n_val, vec);
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
    fn_catenate(machine, nullptr, vec1, mat2);
    EXPECT_EQ(machine->kont_stack.size(), 1);
    EXPECT_NE(dynamic_cast<ThrowErrorK*>(machine->kont_stack.back()), nullptr);

}

// ============================================================================
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

// ISO 13751 §10.1.2: Multi-dimensional index generator
TEST_F(PrimitivesTest, IotaVectorMultiDim) {
    // ⍳1 2 3 → strand of 6 index triples (now valid)
    Value* r = machine->eval("⍳1 2 3");
    EXPECT_TRUE(r->is_strand());
    EXPECT_EQ(r->as_strand()->size(), 6);
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
