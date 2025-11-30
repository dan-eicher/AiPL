// Value system tests

#include <gtest/gtest.h>
#include "value.h"
#include "machine.h"
#include <Eigen/Dense>

using namespace apl;

class ValueTest : public ::testing::Test {
protected:
    Machine* machine;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
    }
};

// Test scalar creation and access
TEST_F(ValueTest, ScalarCreation) {
    Value* v = machine->heap->allocate_scalar(42.0);

    EXPECT_TRUE(v->is_scalar());
    EXPECT_FALSE(v->is_vector());
    EXPECT_FALSE(v->is_matrix());
    EXPECT_FALSE(v->is_array());
    EXPECT_FALSE(v->is_function());
    EXPECT_FALSE(v->is_operator());

    EXPECT_DOUBLE_EQ(v->as_scalar(), 42.0);
    EXPECT_EQ(v->rank(), 0);
    EXPECT_EQ(v->size(), 1);

    // GC will clean up v
}

// Test scalar with negative value
TEST_F(ValueTest, NegativeScalar) {
    Value* v = machine->heap->allocate_scalar(-3.14);

    EXPECT_TRUE(v->is_scalar());
    EXPECT_DOUBLE_EQ(v->as_scalar(), -3.14);

    // GC will clean up v
}

// Test scalar lazy promotion to matrix
TEST_F(ValueTest, ScalarLazyPromotion) {
    Value* v = machine->heap->allocate_scalar(5.0);

    // First call should create the promoted matrix
    Eigen::MatrixXd* m1 = v->as_matrix();
    ASSERT_NE(m1, nullptr);
    EXPECT_EQ(m1->rows(), 1);
    EXPECT_EQ(m1->cols(), 1);
    EXPECT_DOUBLE_EQ((*m1)(0, 0), 5.0);

    // Second call should return the same cached matrix
    Eigen::MatrixXd* m2 = v->as_matrix();
    EXPECT_EQ(m1, m2);  // Same pointer

    // GC will clean up v
}

// Test vector creation
TEST_F(ValueTest, VectorCreation) {
    Eigen::VectorXd vec(4);
    vec << 1.0, 2.0, 3.0, 4.0;

    Value* v = machine->heap->allocate_vector(vec);

    EXPECT_TRUE(v->is_vector());
    EXPECT_TRUE(v->is_array());
    EXPECT_FALSE(v->is_scalar());
    EXPECT_FALSE(v->is_matrix());

    EXPECT_EQ(v->rank(), 1);
    EXPECT_EQ(v->rows(), 4);
    EXPECT_EQ(v->cols(), 1);
    EXPECT_EQ(v->size(), 4);

    // GC will clean up v
}

// Test vector stored as n×1 matrix
TEST_F(ValueTest, VectorAsMatrix) {
    Eigen::VectorXd vec(3);
    vec << 10.0, 20.0, 30.0;

    Value* v = machine->heap->allocate_vector(vec);

    // Access as matrix (should be zero-copy)
    Eigen::MatrixXd* m = v->as_matrix();
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->rows(), 3);
    EXPECT_EQ(m->cols(), 1);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 20.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 30.0);

    // GC will clean up v
}

// Test matrix creation
TEST_F(ValueTest, MatrixCreation) {
    Eigen::MatrixXd mat(2, 3);
    mat << 1.0, 2.0, 3.0,
           4.0, 5.0, 6.0;

    Value* v = machine->heap->allocate_matrix(mat);

    EXPECT_TRUE(v->is_matrix());
    EXPECT_TRUE(v->is_array());
    EXPECT_FALSE(v->is_scalar());
    EXPECT_FALSE(v->is_vector());

    EXPECT_EQ(v->rank(), 2);
    EXPECT_EQ(v->rows(), 2);
    EXPECT_EQ(v->cols(), 3);
    EXPECT_EQ(v->size(), 6);

    // GC will clean up v
}

// Test matrix access
TEST_F(ValueTest, MatrixAccess) {
    Eigen::MatrixXd mat(2, 2);
    mat << 1.0, 2.0,
           3.0, 4.0;

    Value* v = machine->heap->allocate_matrix(mat);

    Eigen::MatrixXd* m = v->as_matrix();
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 4.0);

    // GC will clean up v
}

// Test empty vector
TEST_F(ValueTest, EmptyVector) {
    Eigen::VectorXd vec(0);

    Value* v = machine->heap->allocate_vector(vec);

    EXPECT_TRUE(v->is_vector());
    EXPECT_EQ(v->size(), 0);
    EXPECT_EQ(v->rows(), 0);
    EXPECT_EQ(v->cols(), 1);

    // GC will clean up v
}

// Test 1×1 matrix vs scalar
TEST_F(ValueTest, SingleElementMatrix) {
    Eigen::MatrixXd mat(1, 1);
    mat << 7.0;

    Value* v = machine->heap->allocate_matrix(mat);

    EXPECT_TRUE(v->is_matrix());
    EXPECT_FALSE(v->is_scalar());
    EXPECT_EQ(v->size(), 1);
    EXPECT_EQ(v->rows(), 1);
    EXPECT_EQ(v->cols(), 1);

    Eigen::MatrixXd* m = v->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 7.0);

    // GC will clean up v
}

// Test large vector
TEST_F(ValueTest, LargeVector) {
    const int size = 1000;
    Eigen::VectorXd vec(size);
    for (int i = 0; i < size; i++) {
        vec(i) = static_cast<double>(i);
    }

    Value* v = machine->heap->allocate_vector(vec);

    EXPECT_EQ(v->size(), size);
    EXPECT_EQ(v->rows(), size);

    Eigen::MatrixXd* m = v->as_matrix();
    for (int i = 0; i < size; i++) {
        EXPECT_DOUBLE_EQ((*m)(i, 0), static_cast<double>(i));
    }

    // GC will clean up v
}

// Test large matrix
TEST_F(ValueTest, LargeMatrix) {
    const int rows = 100;
    const int cols = 50;
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat(i, j) = static_cast<double>(i * cols + j);
        }
    }

    Value* v = machine->heap->allocate_matrix(mat);

    EXPECT_EQ(v->rows(), rows);
    EXPECT_EQ(v->cols(), cols);
    EXPECT_EQ(v->size(), rows * cols);

    Eigen::MatrixXd* m = v->as_matrix();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            EXPECT_DOUBLE_EQ((*m)(i, j), static_cast<double>(i * cols + j));
        }
    }

    // GC will clean up v
}

// Test error handling - as_scalar on vector
TEST_F(ValueTest, ErrorScalarOnVector) {
    Eigen::VectorXd vec(3);
    vec << 1.0, 2.0, 3.0;

    Value* v = machine->heap->allocate_vector(vec);

    EXPECT_THROW(v->as_scalar(), std::runtime_error);

    // GC will clean up v
}

// Test zero value
TEST_F(ValueTest, ZeroScalar) {
    Value* v = machine->heap->allocate_scalar(0.0);

    EXPECT_TRUE(v->is_scalar());
    EXPECT_DOUBLE_EQ(v->as_scalar(), 0.0);

    // GC will clean up v
}

// Test const as_matrix
TEST_F(ValueTest, ConstAsMatrix) {
    Eigen::VectorXd vec(2);
    vec << 5.0, 10.0;

    const Value* v = machine->heap->allocate_vector(vec);

    const Eigen::MatrixXd* m = v->as_matrix();
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 10.0);

    // GC will clean up v
}

// Test GC metadata initialization
TEST_F(ValueTest, GCMetadataInit) {
    Value* v = machine->heap->allocate_scalar(1.0);

    EXPECT_FALSE(v->marked);
    EXPECT_FALSE(v->in_old_generation);

    // GC will clean up v
}

// Test GC metadata on different types
TEST_F(ValueTest, GCMetadataAllTypes) {
    Value* s = machine->heap->allocate_scalar(1.0);
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* v = machine->heap->allocate_vector(vec);
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Value* m = machine->heap->allocate_matrix(mat);

    EXPECT_FALSE(s->marked);
    EXPECT_FALSE(v->marked);
    EXPECT_FALSE(m->marked);

    EXPECT_FALSE(s->in_old_generation);
    EXPECT_FALSE(v->in_old_generation);
    EXPECT_FALSE(m->in_old_generation);

    // GC will clean up -     delete s;
    // GC will clean up v
    // GC will clean up -     delete m;
}

// Test mark bit setting
TEST_F(ValueTest, MarkBit) {
    Value* v = machine->heap->allocate_scalar(5.0);

    EXPECT_FALSE(v->marked);

    v->marked = true;
    EXPECT_TRUE(v->marked);

    v->marked = false;
    EXPECT_FALSE(v->marked);

    // GC will clean up v
}

// Test old generation flag
TEST_F(ValueTest, OldGenerationFlag) {
    Value* v = machine->heap->allocate_scalar(10.0);

    EXPECT_FALSE(v->in_old_generation);

    v->in_old_generation = true;
    EXPECT_TRUE(v->in_old_generation);

    // GC will clean up v
}

// Test rank for all types
TEST_F(ValueTest, RankAllTypes) {
    Value* s = machine->heap->allocate_scalar(1.0);
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* v = machine->heap->allocate_vector(vec);
    Eigen::MatrixXd mat(2, 3);
    mat.setConstant(1.0);
    Value* m = machine->heap->allocate_matrix(mat);

    EXPECT_EQ(s->rank(), 0);
    EXPECT_EQ(v->rank(), 1);
    EXPECT_EQ(m->rank(), 2);

    // GC will clean up -     delete s;
    // GC will clean up v
    // GC will clean up -     delete m;
}

// Test size for all types
TEST_F(ValueTest, SizeAllTypes) {
    Value* s = machine->heap->allocate_scalar(1.0);
    Eigen::VectorXd vec(5);
    vec.setConstant(1.0);
    Value* v = machine->heap->allocate_vector(vec);
    Eigen::MatrixXd mat(3, 4);
    mat.setConstant(1.0);
    Value* m = machine->heap->allocate_matrix(mat);

    EXPECT_EQ(s->size(), 1);
    EXPECT_EQ(v->size(), 5);
    EXPECT_EQ(m->size(), 12);

    // GC will clean up -     delete s;
    // GC will clean up v
    // GC will clean up -     delete m;
}

// Test rows and cols
TEST_F(ValueTest, RowsAndCols) {
    Value* s = machine->heap->allocate_scalar(1.0);
    Eigen::VectorXd vec(7);
    vec.setConstant(1.0);
    Value* v = machine->heap->allocate_vector(vec);
    Eigen::MatrixXd mat(4, 5);
    mat.setConstant(1.0);
    Value* m = machine->heap->allocate_matrix(mat);

    EXPECT_EQ(s->rows(), 1);
    EXPECT_EQ(s->cols(), 1);

    EXPECT_EQ(v->rows(), 7);
    EXPECT_EQ(v->cols(), 1);

    EXPECT_EQ(m->rows(), 4);
    EXPECT_EQ(m->cols(), 5);

    // GC will clean up -     delete s;
    // GC will clean up v
    // GC will clean up -     delete m;
}

// Test vector with fractional values
TEST_F(ValueTest, VectorFractional) {
    Eigen::VectorXd vec(3);
    vec << 1.5, 2.7, 3.14;

    Value* v = machine->heap->allocate_vector(vec);

    Eigen::MatrixXd* m = v->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.5);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.7);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.14);

    // GC will clean up v
}

// Test matrix with negative values
TEST_F(ValueTest, MatrixNegative) {
    Eigen::MatrixXd mat(2, 2);
    mat << -1.0, -2.0, -3.0, -4.0;

    Value* v = machine->heap->allocate_matrix(mat);

    Eigen::MatrixXd* m = v->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), -2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), -4.0);

    // GC will clean up v
}

// Test very large scalar
TEST_F(ValueTest, VeryLargeScalar) {
    Value* v = machine->heap->allocate_scalar(1.7976931348623157e+308);  // Near max double

    EXPECT_DOUBLE_EQ(v->as_scalar(), 1.7976931348623157e+308);

    // GC will clean up v
}

// Test very small scalar
TEST_F(ValueTest, VerySmallScalar) {
    Value* v = machine->heap->allocate_scalar(2.2250738585072014e-308);  // Near min positive double

    EXPECT_DOUBLE_EQ(v->as_scalar(), 2.2250738585072014e-308);

    // GC will clean up v
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
