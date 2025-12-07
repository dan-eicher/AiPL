// Value system tests

#include <gtest/gtest.h>
#include "value.h"
#include "machine.h"
#include "primitives.h"
#include "operators.h"
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

}

// Test scalar with negative value
TEST_F(ValueTest, NegativeScalar) {
    Value* v = machine->heap->allocate_scalar(-3.14);

    EXPECT_TRUE(v->is_scalar());
    EXPECT_DOUBLE_EQ(v->as_scalar(), -3.14);

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

}

// Test empty vector
TEST_F(ValueTest, EmptyVector) {
    Eigen::VectorXd vec(0);

    Value* v = machine->heap->allocate_vector(vec);

    EXPECT_TRUE(v->is_vector());
    EXPECT_EQ(v->size(), 0);
    EXPECT_EQ(v->rows(), 0);
    EXPECT_EQ(v->cols(), 1);

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

}

// Test error handling - as_scalar on vector
TEST_F(ValueTest, ErrorScalarOnVector) {
    Eigen::VectorXd vec(3);
    vec << 1.0, 2.0, 3.0;

    Value* v = machine->heap->allocate_vector(vec);

    EXPECT_THROW(v->as_scalar(), std::runtime_error);

}

// Test zero value
TEST_F(ValueTest, ZeroScalar) {
    Value* v = machine->heap->allocate_scalar(0.0);

    EXPECT_TRUE(v->is_scalar());
    EXPECT_DOUBLE_EQ(v->as_scalar(), 0.0);

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

}

// Test GC metadata initialization
TEST_F(ValueTest, GCMetadataInit) {
    Value* v = machine->heap->allocate_scalar(1.0);

    EXPECT_FALSE(v->marked);
    EXPECT_FALSE(v->in_old_generation);

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

}

// Test mark bit setting
TEST_F(ValueTest, MarkBit) {
    Value* v = machine->heap->allocate_scalar(5.0);

    EXPECT_FALSE(v->marked);

    v->marked = true;
    EXPECT_TRUE(v->marked);

    v->marked = false;
    EXPECT_FALSE(v->marked);

}

// Test old generation flag
TEST_F(ValueTest, OldGenerationFlag) {
    Value* v = machine->heap->allocate_scalar(10.0);

    EXPECT_FALSE(v->in_old_generation);

    v->in_old_generation = true;
    EXPECT_TRUE(v->in_old_generation);

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

}

// Test very large scalar
TEST_F(ValueTest, VeryLargeScalar) {
    Value* v = machine->heap->allocate_scalar(1.7976931348623157e+308);  // Near max double

    EXPECT_DOUBLE_EQ(v->as_scalar(), 1.7976931348623157e+308);

}

// Test very small scalar
TEST_F(ValueTest, VerySmallScalar) {
    Value* v = machine->heap->allocate_scalar(2.2250738585072014e-308);  // Near min positive double

    EXPECT_DOUBLE_EQ(v->as_scalar(), 2.2250738585072014e-308);

}

// ============================================================================
// G2 Grammar Value Types Tests
// ============================================================================

// Test DERIVED_OPERATOR value creation
TEST_F(ValueTest, DerivedOperatorCreation) {
    // Create a simple function value (operand)
    Value* operand = machine->heap->allocate_scalar(5.0);

    // Create a derived operator (e.g., +∘. where + is the operator and operand is first arg)
    Value* derived = machine->heap->allocate_derived_operator(&op_dot, operand);

    EXPECT_TRUE(derived->is_operator());
    EXPECT_TRUE(derived->is_derived_operator());
    // DERIVED_OPERATOR is also a function (can be applied to arguments, like +/)
    EXPECT_TRUE(derived->is_function());
    EXPECT_FALSE(derived->is_array());
    EXPECT_EQ(derived->tag, ValueType::DERIVED_OPERATOR);

    // Check the data is correctly stored
    EXPECT_EQ(derived->data.derived_op->op, &op_dot);
    EXPECT_EQ(derived->data.derived_op->first_operand, operand);

}

// Test CURRIED_FN value creation with dyadic curry
TEST_F(ValueTest, CurriedFnDyadicCreation) {
    // Create a function and argument
    Value* fn = machine->heap->allocate_primitive(&prim_plus);
    Value* first_arg = machine->heap->allocate_scalar(3.0);

    // Create curried function: (+3)
    Value* curried = machine->heap->allocate_curried_fn(fn, first_arg, Value::CurryType::DYADIC_CURRY);

    EXPECT_TRUE(curried->is_function());
    EXPECT_TRUE(curried->is_curried_fn());
    EXPECT_FALSE(curried->is_operator());
    EXPECT_FALSE(curried->is_array());
    EXPECT_EQ(curried->tag, ValueType::CURRIED_FN);

    // Check the data
    EXPECT_EQ(curried->data.curried_fn->fn, fn);
    EXPECT_EQ(curried->data.curried_fn->first_arg, first_arg);
    EXPECT_EQ(curried->data.curried_fn->curry_type, Value::CurryType::DYADIC_CURRY);

}

// Test CURRIED_FN value creation with g' transformation
TEST_F(ValueTest, CurriedFnGPrimeCreation) {
    // Create a function and argument
    Value* fn = machine->heap->allocate_primitive(&prim_minus);
    Value* first_arg = machine->heap->allocate_scalar(10.0);

    // Create curried function with g' transformation: (-10)
    Value* curried = machine->heap->allocate_curried_fn(fn, first_arg, Value::CurryType::G_PRIME);

    EXPECT_TRUE(curried->is_function());
    EXPECT_TRUE(curried->is_curried_fn());
    EXPECT_EQ(curried->tag, ValueType::CURRIED_FN);

    // Check the data
    EXPECT_EQ(curried->data.curried_fn->fn, fn);
    EXPECT_EQ(curried->data.curried_fn->first_arg, first_arg);
    EXPECT_EQ(curried->data.curried_fn->curry_type, Value::CurryType::G_PRIME);

}

// Test OPERATOR value creation
TEST_F(ValueTest, OperatorCreation) {
    // Create an operator value
    Value* op = machine->heap->allocate_operator(&op_diaeresis);

    EXPECT_TRUE(op->is_operator());
    EXPECT_FALSE(op->is_derived_operator());
    EXPECT_FALSE(op->is_function());
    EXPECT_FALSE(op->is_array());
    EXPECT_EQ(op->tag, ValueType::OPERATOR);

    // Check the data
    EXPECT_EQ(op->data.op, &op_diaeresis);

}

// Test is_basic_value for G2 grammar
TEST_F(ValueTest, IsBasicValue) {
    Value* scalar = machine->heap->allocate_scalar(1.0);
    Eigen::VectorXd vec(2);
    vec << 1, 2;
    Value* vector = machine->heap->allocate_vector(vec);
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Value* matrix = machine->heap->allocate_matrix(mat);
    Value* func = machine->heap->allocate_primitive(&prim_plus);
    Value* op = machine->heap->allocate_operator(&op_dot);

    // Basic values are scalars and arrays
    EXPECT_TRUE(scalar->is_basic_value());
    EXPECT_TRUE(vector->is_basic_value());
    EXPECT_TRUE(matrix->is_basic_value());

    // Functions and operators are not basic values
    EXPECT_FALSE(func->is_basic_value());
    EXPECT_FALSE(op->is_basic_value());

}

// Test CURRIED_FN with array argument
TEST_F(ValueTest, CurriedFnWithArrayArg) {
    Value* fn = machine->heap->allocate_primitive(&prim_rho);
    Eigen::VectorXd vec(3);
    vec << 2, 3, 4;
    Value* arr_arg = machine->heap->allocate_vector(vec);

    Value* curried = machine->heap->allocate_curried_fn(fn, arr_arg, Value::CurryType::DYADIC_CURRY);

    EXPECT_TRUE(curried->is_function());
    EXPECT_EQ(curried->data.curried_fn->fn, fn);
    EXPECT_EQ(curried->data.curried_fn->first_arg, arr_arg);
    EXPECT_TRUE(curried->data.curried_fn->first_arg->is_array());

}

// Test DERIVED_OPERATOR with function operand
TEST_F(ValueTest, DerivedOperatorWithFunction) {
    Value* func = machine->heap->allocate_primitive(&prim_times);
    Value* derived = machine->heap->allocate_derived_operator(&op_diaeresis, func);

    EXPECT_TRUE(derived->is_derived_operator());
    EXPECT_EQ(derived->data.derived_op->op, &op_diaeresis);
    EXPECT_EQ(derived->data.derived_op->first_operand, func);
    EXPECT_TRUE(derived->data.derived_op->first_operand->is_function());

}

// String Value tests
TEST_F(ValueTest, StringBasic) {
    Value* str = machine->heap->allocate_string("hello");
    EXPECT_TRUE(str->is_string());
    EXPECT_FALSE(str->is_scalar());
    EXPECT_FALSE(str->is_array());
    EXPECT_STREQ(str->as_string(), "hello");
}

TEST_F(ValueTest, StringIsBasicValue) {
    Value* str = machine->heap->allocate_string("test");
    EXPECT_TRUE(str->is_basic_value());
    EXPECT_FALSE(str->is_function());
}

TEST_F(ValueTest, StringInterning) {
    // Same string content should give same pointer (interned)
    Value* str1 = machine->heap->allocate_string("same");
    Value* str2 = machine->heap->allocate_string("same");
    EXPECT_EQ(str1->as_string(), str2->as_string());  // Same pointer
}

TEST_F(ValueTest, StringEmpty) {
    Value* str = machine->heap->allocate_string("");
    EXPECT_TRUE(str->is_string());
    EXPECT_STREQ(str->as_string(), "");
}

// ============================================================================
// String/Character Vector Conversion Tests
// ============================================================================

// Test to_char_vector with ASCII string
TEST_F(ValueTest, StringToCharVectorASCII) {
    Value* str = machine->heap->allocate_string("ABC");
    Value* vec = str->to_char_vector(machine->heap);

    EXPECT_TRUE(vec->is_vector());
    EXPECT_TRUE(vec->is_char_data());
    EXPECT_EQ(vec->size(), 3);

    Eigen::MatrixXd* m = vec->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 65.0);  // 'A'
    EXPECT_DOUBLE_EQ((*m)(1, 0), 66.0);  // 'B'
    EXPECT_DOUBLE_EQ((*m)(2, 0), 67.0);  // 'C'
}

// Test to_string_value with ASCII codepoints
TEST_F(ValueTest, CharVectorToStringASCII) {
    Eigen::VectorXd vec(3);
    vec << 72.0, 105.0, 33.0;  // 'H', 'i', '!'
    Value* charVec = machine->heap->allocate_vector(vec, true);  // is_char_data = true

    EXPECT_TRUE(charVec->is_char_data());

    Value* str = charVec->to_string_value(machine->heap);
    EXPECT_TRUE(str->is_string());
    EXPECT_STREQ(str->as_string(), "Hi!");
}

// Test round-trip conversion ASCII
TEST_F(ValueTest, StringRoundTripASCII) {
    Value* original = machine->heap->allocate_string("Hello, World!");
    Value* vec = original->to_char_vector(machine->heap);
    Value* back = vec->to_string_value(machine->heap);

    EXPECT_STREQ(back->as_string(), "Hello, World!");
}

// Test to_char_vector returns self if already array
TEST_F(ValueTest, ToCharVectorIdempotent) {
    Eigen::VectorXd vec(2);
    vec << 65.0, 66.0;
    Value* charVec = machine->heap->allocate_vector(vec, true);

    Value* result = charVec->to_char_vector(machine->heap);
    EXPECT_EQ(result, charVec);  // Same pointer - no conversion needed
}

// Test to_string_value returns self if already string
TEST_F(ValueTest, ToStringValueIdempotent) {
    Value* str = machine->heap->allocate_string("test");

    Value* result = str->to_string_value(machine->heap);
    EXPECT_EQ(result, str);  // Same pointer - no conversion needed
}

// Test empty string conversion
TEST_F(ValueTest, EmptyStringToCharVector) {
    Value* str = machine->heap->allocate_string("");
    Value* vec = str->to_char_vector(machine->heap);

    EXPECT_TRUE(vec->is_vector());
    EXPECT_TRUE(vec->is_char_data());
    EXPECT_EQ(vec->size(), 0);
}

// Test empty char vector conversion
TEST_F(ValueTest, EmptyCharVectorToString) {
    Eigen::VectorXd vec(0);
    Value* charVec = machine->heap->allocate_vector(vec, true);

    Value* str = charVec->to_string_value(machine->heap);
    EXPECT_TRUE(str->is_string());
    EXPECT_STREQ(str->as_string(), "");
}

// Test UTF-8 2-byte character (Greek letter alpha: α = U+03B1)
TEST_F(ValueTest, StringToCharVectorUTF8TwoByte) {
    Value* str = machine->heap->allocate_string("α");
    Value* vec = str->to_char_vector(machine->heap);

    EXPECT_TRUE(vec->is_vector());
    EXPECT_TRUE(vec->is_char_data());
    EXPECT_EQ(vec->size(), 1);  // One codepoint, not two bytes

    Eigen::MatrixXd* m = vec->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0x03B1);  // U+03B1 = 945
}

// Test UTF-8 3-byte character (Euro sign: € = U+20AC)
TEST_F(ValueTest, StringToCharVectorUTF8ThreeByte) {
    Value* str = machine->heap->allocate_string("€");
    Value* vec = str->to_char_vector(machine->heap);

    EXPECT_TRUE(vec->is_vector());
    EXPECT_EQ(vec->size(), 1);

    Eigen::MatrixXd* m = vec->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0x20AC);  // U+20AC = 8364
}

// Test UTF-8 4-byte character (Emoji: 😀 = U+1F600)
TEST_F(ValueTest, StringToCharVectorUTF8FourByte) {
    Value* str = machine->heap->allocate_string("😀");
    Value* vec = str->to_char_vector(machine->heap);

    EXPECT_TRUE(vec->is_vector());
    EXPECT_EQ(vec->size(), 1);

    Eigen::MatrixXd* m = vec->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 0x1F600);  // U+1F600 = 128512
}

// Test mixed ASCII and UTF-8
TEST_F(ValueTest, StringToCharVectorMixed) {
    Value* str = machine->heap->allocate_string("A⍳B");  // ASCII + APL iota + ASCII
    Value* vec = str->to_char_vector(machine->heap);

    EXPECT_EQ(vec->size(), 3);

    Eigen::MatrixXd* m = vec->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 65.0);     // 'A'
    EXPECT_DOUBLE_EQ((*m)(1, 0), 0x2373);   // APL iota = U+2373 = 9075
    EXPECT_DOUBLE_EQ((*m)(2, 0), 66.0);     // 'B'
}

// Test UTF-8 round-trip with multi-byte chars
TEST_F(ValueTest, StringRoundTripUTF8) {
    Value* original = machine->heap->allocate_string("Hello α€😀 World!");
    Value* vec = original->to_char_vector(machine->heap);
    Value* back = vec->to_string_value(machine->heap);

    EXPECT_STREQ(back->as_string(), "Hello α€😀 World!");
}

// Test codepoint to UTF-8 encoding
TEST_F(ValueTest, CharVectorToStringUTF8) {
    Eigen::VectorXd vec(4);
    vec << 65.0, 0x03B1, 0x20AC, 0x1F600;  // 'A', α, €, 😀
    Value* charVec = machine->heap->allocate_vector(vec, true);

    Value* str = charVec->to_string_value(machine->heap);
    EXPECT_STREQ(str->as_string(), "Aα€😀");
}

// Test is_char_data flag preserved
TEST_F(ValueTest, CharDataFlagPreserved) {
    // Numeric vector should not have char data flag
    Eigen::VectorXd numVec(3);
    numVec << 1.0, 2.0, 3.0;
    Value* numeric = machine->heap->allocate_vector(numVec);
    EXPECT_FALSE(numeric->is_char_data());

    // Char vector should have flag
    Eigen::VectorXd charVec(3);
    charVec << 65.0, 66.0, 67.0;
    Value* chars = machine->heap->allocate_vector(charVec, true);
    EXPECT_TRUE(chars->is_char_data());
}

// Test char data flag on matrix
TEST_F(ValueTest, CharDataFlagMatrix) {
    Eigen::MatrixXd mat(2, 2);
    mat << 65.0, 66.0, 67.0, 68.0;  // A, B, C, D

    Value* numMat = machine->heap->allocate_matrix(mat, false);
    EXPECT_FALSE(numMat->is_char_data());

    Value* charMat = machine->heap->allocate_matrix(mat, true);
    EXPECT_TRUE(charMat->is_char_data());
}

// Test error: to_char_vector on non-string non-array
TEST_F(ValueTest, ToCharVectorErrorOnFunction) {
    Value* fn = machine->heap->allocate_primitive(&prim_plus);
    EXPECT_THROW(fn->to_char_vector(machine->heap), std::runtime_error);
}

// Test error: to_string_value on non-array non-string
TEST_F(ValueTest, ToStringValueErrorOnFunction) {
    Value* fn = machine->heap->allocate_primitive(&prim_plus);
    EXPECT_THROW(fn->to_string_value(machine->heap), std::runtime_error);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
