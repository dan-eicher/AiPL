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

TEST_F(StringTest, ReduceAxisOverString) {
    // +/[1] on string should work same as +/ (strings are 1D)
    Value* result = eval("+/[1]'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 198.0);  // 65 + 66 + 67
}

TEST_F(StringTest, ScanAxisOverString) {
    // +\[1] on string gives running sum of codepoints
    Value* result = eval("+\\[1]'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);   // A
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 131.0);  // A+B
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 198.0);  // A+B+C
}

TEST_F(StringTest, EachOverString) {
    // Negate each codepoint
    Value* result = eval("-¨'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), -65.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), -66.0);
}

// ============================================================================
// String Indexing Tests (using char vector form)
// ============================================================================

TEST_F(StringTest, IndexStringScalar) {
    // S←,'ABC' ⋄ S[2] → 66 (character 'B')
    eval("S←,'ABC'");
    Value* result = eval("S[2]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 66.0);  // 'B' = 66
}

TEST_F(StringTest, IndexStringFirst) {
    // S←,'hello' ⋄ S[1] → 104 (character 'h')
    eval("S←,'hello'");
    Value* result = eval("S[1]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 104.0);  // 'h' = 104
}

TEST_F(StringTest, IndexStringLast) {
    // S←,'hello' ⋄ S[5] → 111 (character 'o')
    eval("S←,'hello'");
    Value* result = eval("S[5]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 111.0);  // 'o' = 111
}

TEST_F(StringTest, IndexStringVector) {
    // S←,'ABCDE' ⋄ S[2 4] → 66 68 (characters 'B' 'D')
    eval("S←,'ABCDE'");
    Value* result = eval("S[2 4]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 66.0);  // 'B'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 68.0);  // 'D'
}

// ============================================================================
// String Indexed Assignment Tests (using char vector form)
// ============================================================================

TEST_F(StringTest, IndexedAssignStringScalar) {
    // S←,'ABC' ⋄ S[2]←90 ⋄ S[2] → 90 (replace 'B' with 'Z')
    eval("S←,'ABC'");
    eval("S[2]←90");  // 'Z' = 90

    Value* result = eval("S[2]");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 90.0);
}

TEST_F(StringTest, IndexedAssignStringFirst) {
    // S←,'abc' ⋄ S[1]←65 ⋄ S → 65 98 99 (replace 'a' with 'A')
    eval("S←,'abc'");
    eval("S[1]←65");  // 'A' = 65

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);   // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 98.0);   // 'b'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 99.0);   // 'c'
}

TEST_F(StringTest, IndexedAssignStringLast) {
    // S←,'xyz' ⋄ S[3]←33 ⋄ S → 120 121 33 (replace 'z' with '!')
    eval("S←,'xyz'");
    eval("S[3]←33");  // '!' = 33

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 120.0);  // 'x'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 121.0);  // 'y'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 33.0);   // '!'
}

TEST_F(StringTest, IndexedAssignStringReturnsValue) {
    // The return value of S[I]←V is V
    eval("S←,'test'");
    Value* result = eval("S[2]←88");  // 'X' = 88
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 88.0);
}

TEST_F(StringTest, IndexedAssignStringChained) {
    // Multiple indexed assignments to build a different string
    eval("S←,'----'");
    eval("S[1]←65");   // 'A'
    eval("S[2]←66");   // 'B'
    eval("S[3]←67");   // 'C'
    eval("S[4]←68");   // 'D'

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);  // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 66.0);  // 'B'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 67.0);  // 'C'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(3, 0), 68.0);  // 'D'
}

TEST_F(StringTest, IndexedAssignStringDirectNoRavel) {
    // S←'ABC' ⋄ S[2]←90 → string converted to numeric, then modified
    // Without ravel, string stays STRING type until indexed assignment
    eval("S←'ABC'");
    eval("S[2]←90");  // 'Z' = 90, should convert string to numeric

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());  // Should now be numeric vector
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);   // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 90.0);   // 'Z' (was 'B')
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 67.0);   // 'C'
}

TEST_F(StringTest, IndexedAssignStringVectorIndices) {
    // S←'ABCDE' ⋄ S[2 4]←88 89 → modify positions 2 and 4
    eval("S←'ABCDE'");
    eval("S[2 4]←88 89");  // 'X'=88, 'Y'=89

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);   // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 88.0);   // 'X' (was 'B')
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 67.0);   // 'C'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(3, 0), 89.0);   // 'Y' (was 'D')
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(4, 0), 69.0);   // 'E'
}

// ============================================================================
// Character Grading (⍋ ⍒ on strings)
// ============================================================================

TEST_F(StringTest, GradeUpString) {
    // ⍋'cab' → indices that sort to 'abc': 2 3 1 (1-origin per ISO 13751)
    Value* result = eval("⍋'cab'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);  // 'a' at index 2
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);  // 'b' at index 3
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);  // 'c' at index 1
}

TEST_F(StringTest, GradeDownString) {
    // ⍒'cab' → indices that sort descending: 1 3 2 (1-origin per ISO 13751)
    Value* result = eval("⍒'cab'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);  // 'c' at index 1
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);  // 'b' at index 3
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 2.0);  // 'a' at index 2
}

TEST_F(StringTest, GradeUpSortString) {
    // Use grade to sort: 'cab'[⍋'cab'] → 'abc'
    Value* result = eval("'cab'[⍋'cab']");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 97.0);   // 'a'
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 98.0);   // 'b'
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 99.0);   // 'c'
}

// ============================================================================
// Format (⍕) - ISO 13751 Section 15.4
// ============================================================================

// Monadic Format - Character Passthrough
TEST_F(StringTest, FormatCharacterStringPassthrough) {
    Value* result = eval("⍕'hello'");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "hello");
}

TEST_F(StringTest, FormatEmptyStringPassthrough) {
    Value* result = eval("⍕''");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");
}

// Monadic Format - Scalar Formatting
TEST_F(StringTest, FormatIntegerScalar) {
    Value* result = eval("⍕42");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "42");
}

TEST_F(StringTest, FormatNegativeInteger) {
    Value* result = eval("⍕¯5");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "¯5");
}

TEST_F(StringTest, FormatZero) {
    Value* result = eval("⍕0");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "0");
}

TEST_F(StringTest, FormatFloatScalar) {
    Value* result = eval("⍕3.14");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("3.14") != std::string::npos);
}

TEST_F(StringTest, FormatNegativeFloat) {
    Value* result = eval("⍕¯3.14");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("¯3.14") != std::string::npos);
}

// Monadic Format - Vector Formatting
TEST_F(StringTest, FormatIntegerVector) {
    Value* result = eval("⍕1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "1 2 3");
}

TEST_F(StringTest, FormatVectorWithNegatives) {
    Value* result = eval("⍕¯1 2 ¯3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "¯1 2 ¯3");
}

// Monadic Format - Empty Arrays
TEST_F(StringTest, FormatEmptyVector) {
    Value* result = eval("⍕⍬");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");
}

// Monadic Format - Print Precision
TEST_F(StringTest, FormatPrintPrecision3) {
    machine->pp = 3;
    Value* result = eval("⍕3.14159265");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // With pp=3, should have at most 3 significant digits
    EXPECT_TRUE(s.length() <= 6);  // "3.14" or similar
}

TEST_F(StringTest, FormatPrintPrecision10) {
    machine->pp = 10;
    Value* result = eval("⍕3.14159265");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // With pp=10, should preserve more digits
    EXPECT_TRUE(s.find("3.14159") != std::string::npos);
}

// Monadic Format - Exponential Form
TEST_F(StringTest, FormatLargeNumber) {
    Value* result = eval("⍕1e15");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Should use exponential form for large numbers
    EXPECT_TRUE(s.find("E") != std::string::npos || s.find("e") != std::string::npos);
}

TEST_F(StringTest, FormatSmallNumber) {
    Value* result = eval("⍕0.0000001");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Should use exponential form for very small numbers
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

// Monadic Format - Matrix
TEST_F(StringTest, FormatMatrix) {
    Value* result = eval("⍕2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Should have newline between rows
    EXPECT_TRUE(s.find("\n") != std::string::npos);
}

// Note: Infinity format tests are in test_primitives.cpp where we can directly
// allocate INFINITY values. Per ISO 13751 Section 9.2.1.1, ÷0 throws a DOMAIN ERROR
// ("domain-error is returned...such as one divided-by zero"), not infinity.

// Dyadic Format - Fixed Decimal
TEST_F(StringTest, DyadicFormatFixedBasic) {
    Value* result = eval("5 2⍕3.14159");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Width 5, 2 decimal places: " 3.14"
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("3.14") != std::string::npos);
}

TEST_F(StringTest, DyadicFormatFixedZeroDecimals) {
    Value* result = eval("5 0⍕42.7");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Width 5, 0 decimals, rounds: "   43"
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("43") != std::string::npos);
}

TEST_F(StringTest, DyadicFormatFixedNegative) {
    Value* result = eval("6 2⍕¯3.14");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 6);
    EXPECT_TRUE(s.find("¯3.14") != std::string::npos);
}

// Dyadic Format - Exponential
TEST_F(StringTest, DyadicFormatExponentialBasic) {
    Value* result = eval("10 ¯3⍕3.14159");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Negative precision means exponential form
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

TEST_F(StringTest, DyadicFormatExponentialLarge) {
    Value* result = eval("10 ¯3⍕31415.9");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
    EXPECT_TRUE(s.find("4") != std::string::npos);  // exponent should be 4
}

TEST_F(StringTest, DyadicFormatExponentialSmall) {
    Value* result = eval("12 ¯3⍕0.00314159");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
    EXPECT_TRUE(s.find("¯") != std::string::npos);  // negative exponent
}

// Dyadic Format - Vector with Single Spec
TEST_F(StringTest, DyadicFormatVectorSingleSpec) {
    Value* result = eval("6 2⍕1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Each element formatted with width 6
    EXPECT_EQ(s.length(), 18);  // 3 * 6
}

// Dyadic Format - Error Cases
TEST_F(StringTest, DyadicFormatWidthTooNarrow) {
    // Width too narrow for the number
    EXPECT_THROW(eval("3 2⍕123.456"), APLError);
}

TEST_F(StringTest, DyadicFormatNonNumericRight) {
    // Right argument must be numeric
    EXPECT_THROW(eval("5 2⍕'abc'"), APLError);
}

TEST_F(StringTest, DyadicFormatOddLengthSpec) {
    // Spec must have even length (pairs)
    EXPECT_THROW(eval("5 2 3⍕1 2"), APLError);
}

TEST_F(StringTest, DyadicFormatZeroWidth) {
    // Width must be positive
    EXPECT_THROW(eval("0 2⍕42"), APLError);
}

// ============================================================================
// Character Data Preservation Tests
// Verify is_char_data() flag is preserved through array operations
// ============================================================================

TEST_F(StringTest, ReshapePreservesCharData) {
    Value* result = eval("3 3 ⍴ 'abcdefghi'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReshapePreservesCharDataCyclic) {
    // Cyclic fill should still preserve char data
    Value* result = eval("2 4 ⍴ 'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReshapeToVectorPreservesCharData) {
    Value* result = eval("5 ⍴ 'abc'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, RavelPreservesCharData) {
    Value* result = eval(",'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, RavelMatrixPreservesCharData) {
    Value* result = eval(",2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, TakePreservesCharData) {
    Value* result = eval("3↑'ABCDE'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, TakeOverfillPreservesCharData) {
    // Taking more than available fills with zeros but should preserve char flag
    Value* result = eval("5↑'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, DropPreservesCharData) {
    Value* result = eval("2↓'ABCDE'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReversePreservesCharData) {
    Value* result = eval("⌽'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, RotatePreservesCharData) {
    Value* result = eval("1⌽'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, TransposePreservesCharData) {
    Value* result = eval("⍉2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, CatenatePreservesCharData) {
    Value* result = eval("'ABC','DEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, CatenateFirstPreservesCharData) {
    Value* result = eval("(2 3⍴'ABCDEF')⍪(1 3⍴'GHI')");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, FirstPreservesCharData) {
    // ↑'ABC' returns scalar, scalars don't have char flag
    // but ↑ on matrix should preserve it
    Value* result = eval("↑2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    // First of matrix is first row (a vector)
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReverseFirstPreservesCharData) {
    Value* result = eval("⊖2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, RotateFirstPreservesCharData) {
    Value* result = eval("1⊖2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReplicatePreservesCharData) {
    Value* result = eval("1 0 1/'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ExpandPreservesCharData) {
    Value* result = eval("1 0 1 0 1\\'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, IndexingPreservesCharData) {
    Value* result = eval("'ABCDE'[2 4]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, EachPreservesCharData) {
    // ⊢¨'ABC' should return each character, preserving char flag
    Value* result = eval("⊢¨'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

// Round-trip tests (⍎⍕)
TEST_F(StringTest, FormatExecuteRoundTripInteger) {
    Value* result = eval("⍎⍕42");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(StringTest, FormatExecuteRoundTripNegative) {
    Value* result = eval("⍎⍕¯17");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -17.0);
}

TEST_F(StringTest, FormatExecuteRoundTripFloat) {
    machine->pp = 17;  // Full precision
    Value* result = eval("⍎⍕3.14");
    ASSERT_NE(result, nullptr);
    EXPECT_NEAR(result->as_scalar(), 3.14, 1e-10);
}

TEST_F(StringTest, FormatExecuteRoundTripVector) {
    Value* result = eval("⍎⍕1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

// ============================================================================
// Mixed Numeric/String Strands (Characters are just codepoints)
// ============================================================================

TEST_F(StringTest, MixedStrandSingleChar) {
    // 'a' is codepoint 97, so 1 2 'a' should be 1 2 97
    Value* result = eval("1 2 'a'");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ((*mat)(0), 1);
    EXPECT_EQ((*mat)(1), 2);
    EXPECT_EQ((*mat)(2), 97);  // 'a' = 97
}

TEST_F(StringTest, MixedStrandMultiChar) {
    // 'ab' expands to 97 98, so 1 'ab' 2 should be 1 97 98 2
    Value* result = eval("1 'ab' 2");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ((*mat)(0), 1);
    EXPECT_EQ((*mat)(1), 97);  // 'a'
    EXPECT_EQ((*mat)(2), 98);  // 'b'
    EXPECT_EQ((*mat)(3), 2);
}

TEST_F(StringTest, MixedStrandCharFirst) {
    // 'x' 1 2 should be 120 1 2
    Value* result = eval("'x' 1 2");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ((*mat)(0), 120);  // 'x'
    EXPECT_EQ((*mat)(1), 1);
    EXPECT_EQ((*mat)(2), 2);
}

TEST_F(StringTest, MixedStrandAllChars) {
    // 'a' 'b' 'c' should be 97 98 99
    Value* result = eval("'a' 'b' 'c'");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ((*mat)(0), 97);
    EXPECT_EQ((*mat)(1), 98);
    EXPECT_EQ((*mat)(2), 99);
}

TEST_F(StringTest, ReduceOverMixedStrand) {
    // +/1 2 'a' should be 1+2+97 = 100
    Value* result = eval("+/1 2 'a'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_EQ(result->as_scalar(), 100);
}

TEST_F(StringTest, JuxtaposeMixedBasicValues) {
    // Two basic values next to each other form a strand
    // 1 'a' should be 1 97
    Value* result = eval("1 'a'");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ((*mat)(0), 1);
    EXPECT_EQ((*mat)(1), 97);
}

TEST_F(StringTest, JuxtaposeStringAndVector) {
    // 'ab' 1 2 should be 97 98 1 2
    Value* result = eval("'ab' 1 2");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ((*mat)(0), 97);
    EXPECT_EQ((*mat)(1), 98);
    EXPECT_EQ((*mat)(2), 1);
    EXPECT_EQ((*mat)(3), 2);
}

TEST_F(StringTest, JuxtaposeVectorAndString) {
    // 1 2 'ab' should be 1 2 97 98
    Value* result = eval("1 2 'ab'");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_EQ((*mat)(0), 1);
    EXPECT_EQ((*mat)(1), 2);
    EXPECT_EQ((*mat)(2), 97);
    EXPECT_EQ((*mat)(3), 98);
}

// ============================================================================
// Typical Element Tests (ISO 13751 §5.3.2)
// Character arrays use blank ' ' (32), numeric arrays use 0
// ============================================================================

TEST_F(StringTest, FirstOfEmptyCharArrayReturnsBlank) {
    // First (↑) of empty character array returns typical element (blank)
    Value* result = eval("↑0↑'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 32.0);  // blank = ASCII 32
}

TEST_F(StringTest, FirstOfEmptyNumericArrayReturnsZero) {
    // First (↑) of empty numeric array returns typical element (zero)
    Value* result = eval("↑0↑1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(StringTest, TakeExtendCharArrayWithBlanks) {
    // Taking more than length pads with blanks for character data
    Value* result = eval("5↑'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0), 65.0);  // 'A'
    EXPECT_DOUBLE_EQ((*mat)(1), 66.0);  // 'B'
    EXPECT_DOUBLE_EQ((*mat)(2), 32.0);  // blank
    EXPECT_DOUBLE_EQ((*mat)(3), 32.0);  // blank
    EXPECT_DOUBLE_EQ((*mat)(4), 32.0);  // blank
}

TEST_F(StringTest, TakeExtendNumericArrayWithZeros) {
    // Taking more than length pads with zeros for numeric data
    Value* result = eval("5↑1 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_FALSE(result->is_char_data());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(3), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(4), 0.0);
}

TEST_F(StringTest, ExpandCharArrayWithBlanks) {
    // Expand (backslash) uses blank for character fill
    Value* result = eval("1 0 1 0 1\\'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0), 65.0);  // 'A'
    EXPECT_DOUBLE_EQ((*mat)(1), 32.0);  // blank
    EXPECT_DOUBLE_EQ((*mat)(2), 66.0);  // 'B'
    EXPECT_DOUBLE_EQ((*mat)(3), 32.0);  // blank
    EXPECT_DOUBLE_EQ((*mat)(4), 67.0);  // 'C'
}

TEST_F(StringTest, ExpandNumericArrayWithZeros) {
    // Expand (backslash) uses zero for numeric fill
    Value* result = eval("1 0 1 0 1\\1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_FALSE(result->is_char_data());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(2), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(3), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(4), 3.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
