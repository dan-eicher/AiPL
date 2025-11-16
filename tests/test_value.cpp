// Value system tests

#include <gtest/gtest.h>
#include "value.h"
#include <Eigen/Dense>

using namespace apl;

class ValueTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Clean up any allocated values
    }
};

// Test scalar creation and access
TEST_F(ValueTest, ScalarCreation) {
    Value* v = Value::from_scalar(42.0);

    EXPECT_TRUE(v->is_scalar());
    EXPECT_FALSE(v->is_vector());
    EXPECT_FALSE(v->is_matrix());
    EXPECT_FALSE(v->is_array());
    EXPECT_FALSE(v->is_function());
    EXPECT_FALSE(v->is_operator());

    EXPECT_DOUBLE_EQ(v->as_scalar(), 42.0);
    EXPECT_EQ(v->rank(), 0);
    EXPECT_EQ(v->size(), 1);

    delete v;
}

// Test scalar with negative value
TEST_F(ValueTest, NegativeScalar) {
    Value* v = Value::from_scalar(-3.14);

    EXPECT_TRUE(v->is_scalar());
    EXPECT_DOUBLE_EQ(v->as_scalar(), -3.14);

    delete v;
}

// Test scalar lazy promotion to matrix
TEST_F(ValueTest, ScalarLazyPromotion) {
    Value* v = Value::from_scalar(5.0);

    // First call should create the promoted matrix
    Eigen::MatrixXd* m1 = v->as_matrix();
    ASSERT_NE(m1, nullptr);
    EXPECT_EQ(m1->rows(), 1);
    EXPECT_EQ(m1->cols(), 1);
    EXPECT_DOUBLE_EQ((*m1)(0, 0), 5.0);

    // Second call should return the same cached matrix
    Eigen::MatrixXd* m2 = v->as_matrix();
    EXPECT_EQ(m1, m2);  // Same pointer

    delete v;
}

// Test vector creation
TEST_F(ValueTest, VectorCreation) {
    Eigen::VectorXd vec(4);
    vec << 1.0, 2.0, 3.0, 4.0;

    Value* v = Value::from_vector(vec);

    EXPECT_TRUE(v->is_vector());
    EXPECT_TRUE(v->is_array());
    EXPECT_FALSE(v->is_scalar());
    EXPECT_FALSE(v->is_matrix());

    EXPECT_EQ(v->rank(), 1);
    EXPECT_EQ(v->rows(), 4);
    EXPECT_EQ(v->cols(), 1);
    EXPECT_EQ(v->size(), 4);

    delete v;
}

// Test vector stored as n×1 matrix
TEST_F(ValueTest, VectorAsMatrix) {
    Eigen::VectorXd vec(3);
    vec << 10.0, 20.0, 30.0;

    Value* v = Value::from_vector(vec);

    // Access as matrix (should be zero-copy)
    Eigen::MatrixXd* m = v->as_matrix();
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->rows(), 3);
    EXPECT_EQ(m->cols(), 1);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 20.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 30.0);

    delete v;
}

// Test matrix creation
TEST_F(ValueTest, MatrixCreation) {
    Eigen::MatrixXd mat(2, 3);
    mat << 1.0, 2.0, 3.0,
           4.0, 5.0, 6.0;

    Value* v = Value::from_matrix(mat);

    EXPECT_TRUE(v->is_matrix());
    EXPECT_TRUE(v->is_array());
    EXPECT_FALSE(v->is_scalar());
    EXPECT_FALSE(v->is_vector());

    EXPECT_EQ(v->rank(), 2);
    EXPECT_EQ(v->rows(), 2);
    EXPECT_EQ(v->cols(), 3);
    EXPECT_EQ(v->size(), 6);

    delete v;
}

// Test matrix access
TEST_F(ValueTest, MatrixAccess) {
    Eigen::MatrixXd mat(2, 2);
    mat << 1.0, 2.0,
           3.0, 4.0;

    Value* v = Value::from_matrix(mat);

    Eigen::MatrixXd* m = v->as_matrix();
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 4.0);

    delete v;
}

// Test empty vector
TEST_F(ValueTest, EmptyVector) {
    Eigen::VectorXd vec(0);

    Value* v = Value::from_vector(vec);

    EXPECT_TRUE(v->is_vector());
    EXPECT_EQ(v->size(), 0);
    EXPECT_EQ(v->rows(), 0);
    EXPECT_EQ(v->cols(), 1);

    delete v;
}

// Test 1×1 matrix vs scalar
TEST_F(ValueTest, SingleElementMatrix) {
    Eigen::MatrixXd mat(1, 1);
    mat << 7.0;

    Value* v = Value::from_matrix(mat);

    EXPECT_TRUE(v->is_matrix());
    EXPECT_FALSE(v->is_scalar());
    EXPECT_EQ(v->size(), 1);
    EXPECT_EQ(v->rows(), 1);
    EXPECT_EQ(v->cols(), 1);

    Eigen::MatrixXd* m = v->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 7.0);

    delete v;
}

// Test large vector
TEST_F(ValueTest, LargeVector) {
    const int size = 1000;
    Eigen::VectorXd vec(size);
    for (int i = 0; i < size; i++) {
        vec(i) = static_cast<double>(i);
    }

    Value* v = Value::from_vector(vec);

    EXPECT_EQ(v->size(), size);
    EXPECT_EQ(v->rows(), size);

    Eigen::MatrixXd* m = v->as_matrix();
    for (int i = 0; i < size; i++) {
        EXPECT_DOUBLE_EQ((*m)(i, 0), static_cast<double>(i));
    }

    delete v;
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

    Value* v = Value::from_matrix(mat);

    EXPECT_EQ(v->rows(), rows);
    EXPECT_EQ(v->cols(), cols);
    EXPECT_EQ(v->size(), rows * cols);

    Eigen::MatrixXd* m = v->as_matrix();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            EXPECT_DOUBLE_EQ((*m)(i, j), static_cast<double>(i * cols + j));
        }
    }

    delete v;
}

// Test error handling - as_scalar on vector
TEST_F(ValueTest, ErrorScalarOnVector) {
    Eigen::VectorXd vec(3);
    vec << 1.0, 2.0, 3.0;

    Value* v = Value::from_vector(vec);

    EXPECT_THROW(v->as_scalar(), std::runtime_error);

    delete v;
}

// Test zero value
TEST_F(ValueTest, ZeroScalar) {
    Value* v = Value::from_scalar(0.0);

    EXPECT_TRUE(v->is_scalar());
    EXPECT_DOUBLE_EQ(v->as_scalar(), 0.0);

    delete v;
}

// Test const as_matrix
TEST_F(ValueTest, ConstAsMatrix) {
    Eigen::VectorXd vec(2);
    vec << 5.0, 10.0;

    const Value* v = Value::from_vector(vec);

    const Eigen::MatrixXd* m = v->as_matrix();
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 10.0);

    delete v;
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
