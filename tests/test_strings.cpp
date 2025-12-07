// String operations tests
// Tests for STRING type, character vectors, and UTF-8 support

#include <gtest/gtest.h>
#include "machine.h"
#include "parser.h"
#include "value.h"

using namespace apl;

class StringTest : public ::testing::Test {
protected:
    Machine* machine = nullptr;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
    }

    Value* eval(const std::string& input) {
        Value* result = machine->eval(input);
        if (!result) {
            ADD_FAILURE() << "Eval failed: " << machine->parser->get_error();
        }
        return result;
    }
};

// ============================================================================
// Basic String Literals
// ============================================================================

TEST_F(StringTest, StringLiteral) {
    Value* result = eval("'hello'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "hello");
}

TEST_F(StringTest, EmptyString) {
    Value* result = eval("''");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");
}

TEST_F(StringTest, StringWithSpaces) {
    Value* result = eval("'hello world'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "hello world");
}

// ============================================================================
// String to Char Vector Conversion (via primitives)
// ============================================================================

TEST_F(StringTest, ShapeOfString) {
    // Shape of string should give its length (after converting to char vector)
    Value* result = eval("⍴'hello'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 5.0);
}

TEST_F(StringTest, TallyOfString) {
    // Tally of string should give its length
    Value* result = eval("≢'hello'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(StringTest, RavelString) {
    // Ravel of string converts to char vector
    Value* result = eval(",'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    // Should be codepoints for 'A' and 'B'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 66.0);
}

TEST_F(StringTest, ReverseString) {
    // Reverse of string
    Value* result = eval("⌽'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // Should be C, B, A (67, 66, 65)
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 67.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 66.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 65.0);
}

// ============================================================================
// String Concatenation
// ============================================================================

TEST_F(StringTest, CatenateStrings) {
    // Concatenating two strings
    Value* result = eval("'hello','world'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 10);
}

TEST_F(StringTest, CatenateStringAndVector) {
    // String with numeric vector (both become char vectors)
    Value* result = eval("'A',66");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);  // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 66.0);  // scalar 66
}

// ============================================================================
// String Comparison
// ============================================================================

TEST_F(StringTest, CompareStringElements) {
    // Compare string elements (as codepoints)
    Value* result = eval("'ABC'='ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // All should be 1 (equal)
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 1.0);
}

TEST_F(StringTest, CompareStringWithScalar) {
    // Compare string with scalar codepoint
    Value* result = eval("65='ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // Only first element (A=65) should match
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 0.0);
}

// ============================================================================
// Execute with Strings
// ============================================================================

TEST_F(StringTest, ExecuteString) {
    Value* result = eval("⍎'1+2'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(StringTest, ExecuteStringVariable) {
    // Assign string to variable then execute
    eval("code←'2×3'");
    Value* result = eval("⍎code");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// ============================================================================
// UTF-8 Support
// ============================================================================

TEST_F(StringTest, UTF8Greek) {
    // Greek letter alpha (2-byte UTF-8)
    Value* result = eval("⍴'α'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);  // 1 character
}

TEST_F(StringTest, UTF8APLSymbols) {
    // APL iota symbol (3-byte UTF-8)
    Value* result = eval(",'⍳'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 0x2373);  // U+2373
}

TEST_F(StringTest, UTF8Emoji) {
    // Emoji (4-byte UTF-8)
    Value* result = eval(",'\xF0\x9F\x98\x80'");  // U+1F600 grinning face
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 0x1F600);
}

TEST_F(StringTest, UTF8MixedLength) {
    // Mixed: ASCII + 2-byte + 3-byte
    Value* result = eval("⍴'Aα€'");
    ASSERT_NE(result, nullptr);
    // Should be 3 characters, not 6 bytes
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 3.0);
}

// ============================================================================
// Arithmetic on Strings (Character Codepoints)
// ============================================================================

TEST_F(StringTest, AddToString) {
    // Adding to string codepoints
    Value* result = eval("1+'A'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 66.0);  // 'A' + 1 = 'B'
}

TEST_F(StringTest, SubtractStrings) {
    // Subtracting strings gives codepoint differences
    Value* result = eval("'B'-'A'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);  // 66 - 65 = 1
}

// ============================================================================
// String with Operators
// ============================================================================

TEST_F(StringTest, ReduceOverString) {
    // Sum of codepoints
    Value* result = eval("+/'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 131.0);  // 65 + 66
}

TEST_F(StringTest, EachOverString) {
    // Negate each codepoint
    Value* result = eval("-¨'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), -65.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), -66.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
