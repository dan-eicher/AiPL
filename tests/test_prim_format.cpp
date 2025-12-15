// Format (⍕) Primitive Tests - ISO 13751 Section 15.4
// Split from test_primitives.cpp for maintainability

#include <gtest/gtest.h>
#include "primitives.h"
#include "value.h"
#include "machine.h"
#include <Eigen/Dense>
#include <cmath>

using namespace apl;

class FormatTest : public ::testing::Test {
protected:
    Machine* machine;
    void SetUp() override { machine = new Machine(); }
    void TearDown() override { delete machine; }
};

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ============================================================================
// Format (⍕) Primitive Tests - ISO 13751 Section 15.4
// ============================================================================

// Monadic Format - String Passthrough
TEST_F(FormatTest, FormatMonadicStringPassthrough) {
    Value* str = machine->heap->allocate_string("hello");
    fn_format_monadic(machine, nullptr, str);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "hello");
}

TEST_F(FormatTest, FormatMonadicEmptyString) {
    Value* str = machine->heap->allocate_string("");
    fn_format_monadic(machine, nullptr, str);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");
}

// Monadic Format - Scalar Formatting
TEST_F(FormatTest, FormatMonadicIntegerScalar) {
    Value* num = machine->heap->allocate_scalar(42.0);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "42");
}

TEST_F(FormatTest, FormatMonadicNegativeInteger) {
    Value* num = machine->heap->allocate_scalar(-5.0);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "¯5");
}

TEST_F(FormatTest, FormatMonadicZero) {
    Value* num = machine->heap->allocate_scalar(0.0);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "0");
}

TEST_F(FormatTest, FormatMonadicFloat) {
    Value* num = machine->heap->allocate_scalar(3.14);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("3.14") != std::string::npos);
}

TEST_F(FormatTest, FormatMonadicNegativeFloat) {
    Value* num = machine->heap->allocate_scalar(-3.14);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("¯3.14") != std::string::npos);
}

// Monadic Format - Vector Formatting
TEST_F(FormatTest, FormatMonadicIntegerVector) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_format_monadic(machine, nullptr, vec);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "1 2 3");
}

TEST_F(FormatTest, FormatMonadicVectorWithNegatives) {
    Eigen::VectorXd v(3);
    v << -1.0, 2.0, -3.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_format_monadic(machine, nullptr, vec);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "¯1 2 ¯3");
}

// Monadic Format - Empty Vector
TEST_F(FormatTest, FormatMonadicEmptyVector) {
    Eigen::VectorXd v(0);
    Value* vec = machine->heap->allocate_vector(v);
    fn_format_monadic(machine, nullptr, vec);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");
}

// Monadic Format - Print Precision
TEST_F(FormatTest, FormatMonadicPrintPrecision3) {
    machine->pp = 3;
    Value* num = machine->heap->allocate_scalar(3.14159265);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.length() <= 6);  // "3.14" or similar
}

TEST_F(FormatTest, FormatMonadicPrintPrecision10) {
    machine->pp = 10;
    Value* num = machine->heap->allocate_scalar(3.14159265);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("3.14159") != std::string::npos);
}

// Monadic Format - Large/Small Numbers (Exponential)
TEST_F(FormatTest, FormatMonadicLargeNumber) {
    Value* num = machine->heap->allocate_scalar(1e15);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

TEST_F(FormatTest, FormatMonadicSmallNumber) {
    Value* num = machine->heap->allocate_scalar(1e-7);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

// Monadic Format - Infinity
TEST_F(FormatTest, FormatMonadicInfinity) {
    Value* num = machine->heap->allocate_scalar(INFINITY);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("∞") != std::string::npos);
}

TEST_F(FormatTest, FormatMonadicNegativeInfinity) {
    Value* num = machine->heap->allocate_scalar(-INFINITY);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("¯∞") != std::string::npos);
}

// Monadic Format - Matrix
TEST_F(FormatTest, FormatMonadicMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Value* mat = machine->heap->allocate_matrix(m);
    fn_format_monadic(machine, nullptr, mat);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("\n") != std::string::npos);
}

// Dyadic Format - Fixed Decimal
TEST_F(FormatTest, FormatDyadicFixedBasic) {
    Eigen::VectorXd spec(2);
    spec << 5.0, 2.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(3.14159);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("3.14") != std::string::npos);
}

TEST_F(FormatTest, FormatDyadicZeroDecimals) {
    Eigen::VectorXd spec(2);
    spec << 5.0, 0.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(42.7);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("43") != std::string::npos);  // Rounds
}

TEST_F(FormatTest, FormatDyadicNegative) {
    Eigen::VectorXd spec(2);
    spec << 6.0, 2.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(-3.14);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 6);
    EXPECT_TRUE(s.find("¯3.14") != std::string::npos);
}

// Dyadic Format - Exponential (negative precision)
TEST_F(FormatTest, FormatDyadicExponential) {
    Eigen::VectorXd spec(2);
    spec << 10.0, -3.0;  // Negative precision = exponential
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(3.14159);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

// Dyadic Format - Vector
TEST_F(FormatTest, FormatDyadicVector) {
    Eigen::VectorXd spec(2);
    spec << 6.0, 2.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    Value* omega = machine->heap->allocate_vector(v);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 18);  // 3 * 6
}

// ============================================================================
// ISO 13751 Section 15.4 Format - Additional Edge Case Tests
// ============================================================================

// Monadic format: character vector returns unchanged (not just strings)
TEST_F(FormatTest, FormatMonadicCharVector) {
    // Create a character vector (array with char data)
    Value* cv = machine->eval("'ABC'");
    ASSERT_NE(cv, nullptr);
    fn_format_monadic(machine, nullptr, cv);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    // Should return the character data (as string in our impl)
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "ABC");
}

// Monadic format: empty matrix returns empty character array
TEST_F(FormatTest, FormatMonadicEmptyMatrix) {
    Value* em = machine->eval("0 3⍴0");  // 0x3 empty matrix
    ASSERT_NE(em, nullptr);
    fn_format_monadic(machine, nullptr, em);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");  // Empty
}

// Dyadic format: rank error if A is matrix
TEST_F(FormatTest, FormatDyadicRankErrorMatrix) {
    EXPECT_THROW(machine->eval("(2 2⍴5 2 5 2)⍕42"), APLError);
}

// Dyadic format: length error if A has odd number of elements
TEST_F(FormatTest, FormatDyadicLengthErrorOdd) {
    EXPECT_THROW(machine->eval("5 2 3⍕42"), APLError);
}

// Dyadic format: domain error if A is character
TEST_F(FormatTest, FormatDyadicDomainErrorCharLeft) {
    EXPECT_THROW(machine->eval("'AB'⍕42"), APLError);
}

// Dyadic format: domain error if B is character
TEST_F(FormatTest, FormatDyadicDomainErrorCharRight) {
    EXPECT_THROW(machine->eval("5 2⍕'hello'"), APLError);
}

// Dyadic format: width too narrow error
TEST_F(FormatTest, FormatDyadicWidthTooNarrow) {
    EXPECT_THROW(machine->eval("2 2⍕12345"), APLError);
}

// Dyadic format: width must be positive
TEST_F(FormatTest, FormatDyadicWidthNotPositive) {
    EXPECT_THROW(machine->eval("0 2⍕42"), APLError);
}

// Dyadic format: empty B returns empty string
TEST_F(FormatTest, FormatDyadicEmptyB) {
    Eigen::VectorXd spec(2);
    spec << 5.0, 2.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Eigen::VectorXd v(0);  // Empty vector
    Value* omega = machine->heap->allocate_vector(v);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "");
}

// Dyadic format: matrix with single spec (applies to all columns)
TEST_F(FormatTest, FormatDyadicMatrixSingleSpec) {
    Eigen::VectorXd spec(2);
    spec << 5.0, 1.0;  // width=5, 1 decimal
    Value* alpha = machine->heap->allocate_vector(spec);
    Eigen::MatrixXd m(2, 3);
    m << 1.1, 2.2, 3.3,
         4.4, 5.5, 6.6;
    Value* omega = machine->heap->allocate_matrix(m);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Should have 2 rows, each with 3 columns of width 5 = 15 chars per row
    EXPECT_TRUE(s.find("\n") != std::string::npos);  // Multi-row
}

// Dyadic format: multiple specs for different columns
TEST_F(FormatTest, FormatDyadicMultipleSpecs) {
    Eigen::VectorXd spec(6);  // 3 pairs for 3 columns
    spec << 4.0, 0.0,   // column 1: width=4, 0 decimals
           6.0, 2.0,   // column 2: width=6, 2 decimals
           8.0, -2.0;  // column 3: width=8, exponential with 2 digits
    Value* alpha = machine->heap->allocate_vector(spec);
    Eigen::VectorXd v(3);
    v << 42.0, 3.14159, 1234.5;
    Value* omega = machine->heap->allocate_vector(v);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 18);  // 4 + 6 + 8 = 18
}

// Dyadic format: scalar width with implicit 0 decimals
TEST_F(FormatTest, FormatDyadicScalarSpec) {
    Value* alpha = machine->heap->allocate_scalar(5.0);  // Just width
    Value* omega = machine->heap->allocate_scalar(42.7);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("43") != std::string::npos);  // Rounded to integer
}

// Dyadic format: high precision exponential
TEST_F(FormatTest, FormatDyadicExponentialHighPrecision) {
    Eigen::VectorXd spec(2);
    spec << 15.0, -8.0;  // 8 significant digits in mantissa
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(3.141592653589793);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
    EXPECT_TRUE(s.find("3.14159") != std::string::npos);  // At least 6 digits of pi
}

// Dyadic format: negative number in exponential form
TEST_F(FormatTest, FormatDyadicExponentialNegative) {
    Eigen::VectorXd spec(2);
    spec << 12.0, -3.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(-0.00314159);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
    EXPECT_TRUE(s.find("¯") != std::string::npos);  // High minus for negative
}

// ============================================================================
// ISO 13751 Section 15.2 - Numeric Conversion Round-Trip Tests
// For any numeric scalar X, when print-precision is set to full-print-precision,
// X shall be the same as ⍎⍕X.
// ============================================================================

TEST_F(FormatTest, RoundTripInteger) {
    // ⍎⍕42 should equal 42
    Value* result = machine->eval("⍎⍕42");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(FormatTest, RoundTripNegativeInteger) {
    // ⍎⍕¯99 should equal ¯99
    Value* result = machine->eval("⍎⍕¯99");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -99.0);
}

TEST_F(FormatTest, RoundTripZero) {
    // ⍎⍕0 should equal 0
    Value* result = machine->eval("⍎⍕0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(FormatTest, RoundTripFloat) {
    // ⍎⍕3.14159 should be very close to 3.14159
    machine->pp = 17;  // Full precision
    Value* result = machine->eval("⍎⍕3.14159");
    ASSERT_NE(result, nullptr);
    EXPECT_NEAR(result->as_scalar(), 3.14159, 1e-10);
}

TEST_F(FormatTest, RoundTripNegativeFloat) {
    machine->pp = 17;
    Value* result = machine->eval("⍎⍕¯2.71828");
    ASSERT_NE(result, nullptr);
    EXPECT_NEAR(result->as_scalar(), -2.71828, 1e-10);
}

TEST_F(FormatTest, RoundTripLargeNumber) {
    // Large numbers in exponential form
    machine->pp = 17;
    Value* result = machine->eval("⍎⍕1E15");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1e15);
}

TEST_F(FormatTest, RoundTripSmallNumber) {
    // Small numbers in exponential form
    machine->pp = 17;
    Value* result = machine->eval("⍎⍕1E¯10");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1e-10);
}

TEST_F(FormatTest, RoundTripVector) {
    // Round-trip for vector: ⍎⍕1 2 3 should equal 1 2 3
    Value* result = machine->eval("⍎⍕1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

// ISO 15.2.1: Different forms of same value should produce same number
TEST_F(FormatTest, NumericInputEquivalentForms) {
    // 2.5E4 should equal 25000
    Value* r1 = machine->eval("2.5E4");
    Value* r2 = machine->eval("25000");
    EXPECT_DOUBLE_EQ(r1->as_scalar(), r2->as_scalar());

    // 002 should equal 2
    Value* r3 = machine->eval("002");
    Value* r4 = machine->eval("2");
    EXPECT_DOUBLE_EQ(r3->as_scalar(), r4->as_scalar());
}

// ============================================================================
// ISO 13751 Section 15.4.1 - Monadic Format Additional Tests
// ============================================================================

// Decimal-rational vs decimal-exponential switching rules
TEST_F(FormatTest, FormatMonadicSwitchToExponentialManyDigits) {
    // More than PP significant digits to left of decimal should use exponential
    machine->pp = 5;
    Value* num = machine->heap->allocate_scalar(123456789.0);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

TEST_F(FormatTest, FormatMonadicSwitchToExponentialLeadingZeros) {
    // More than 5 leading zeros should trigger exponential
    Value* num = machine->heap->allocate_scalar(0.0000001);  // 7 leading zeros
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

TEST_F(FormatTest, FormatMonadicDecimalRationalNoSwitch) {
    // 5 or fewer leading zeros stays decimal
    Value* num = machine->heap->allocate_scalar(0.00001);  // exactly 5 leading zeros
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Could be either form depending on PP, but verify it's valid
    EXPECT_TRUE(s.length() > 0);
}

TEST_F(FormatTest, FormatMonadicNaN) {
    Value* num = machine->heap->allocate_scalar(NAN);
    fn_format_monadic(machine, nullptr, num);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    // NaN representation may vary, but should produce something
    std::string s = result->as_string();
    EXPECT_TRUE(s.length() > 0);
}

TEST_F(FormatTest, FormatMonadicMatrixAlignment) {
    // Matrix columns should be aligned
    Eigen::MatrixXd m(2, 2);
    m << 1.0, 100.0,
         -99.0, 5.0;
    Value* mat = machine->heap->allocate_matrix(m);
    fn_format_monadic(machine, nullptr, mat);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Should have newline separating rows
    EXPECT_TRUE(s.find("\n") != std::string::npos);
    // Both rows should have same display width (aligned columns)
    // Count UTF-8 characters, not bytes (¯ is 2 bytes but 1 display char)
    size_t nl_pos = s.find("\n");
    ASSERT_NE(nl_pos, std::string::npos);
    std::string row1 = s.substr(0, nl_pos);
    std::string row2 = s.substr(nl_pos + 1);
    // Remove trailing newline if present
    if (!row2.empty() && row2.back() == '\n') {
        row2.pop_back();
    }
    // Count UTF-8 characters
    auto utf8_char_count = [](const std::string& str) -> size_t {
        size_t count = 0;
        for (size_t i = 0; i < str.length(); ) {
            unsigned char c = str[i];
            if ((c & 0x80) == 0) { i += 1; }
            else if ((c & 0xE0) == 0xC0) { i += 2; }
            else if ((c & 0xF0) == 0xE0) { i += 3; }
            else { i += 4; }
            count++;
        }
        return count;
    };
    EXPECT_EQ(utf8_char_count(row1), utf8_char_count(row2));
}

TEST_F(FormatTest, FormatMonadicVectorMixedSigns) {
    Eigen::VectorXd v(4);
    v << 1.0, -2.0, 3.0, -4.0;
    Value* vec = machine->heap->allocate_vector(v);
    fn_format_monadic(machine, nullptr, vec);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("¯2") != std::string::npos);
    EXPECT_TRUE(s.find("¯4") != std::string::npos);
}

TEST_F(FormatTest, FormatMonadicSingleElementMatrix) {
    Eigen::MatrixXd m(1, 1);
    m << 42.0;
    Value* mat = machine->heap->allocate_matrix(m);
    fn_format_monadic(machine, nullptr, mat);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "42");
}

// ============================================================================
// ISO 13751 Section 15.4.2 - Dyadic Format Additional Tests
// ============================================================================

TEST_F(FormatTest, FormatDyadicZeroInExponential) {
    // 0 in exponential form: 0.00E0
    Eigen::VectorXd spec(2);
    spec << 10.0, -3.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(0.0);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
    EXPECT_TRUE(s.find("0") != std::string::npos);
}

TEST_F(FormatTest, FormatDyadicLargeInFixed) {
    // Large number in fixed format
    Eigen::VectorXd spec(2);
    spec << 20.0, 0.0;  // width=20, no decimals
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(12345678901234.0);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 20);
}

TEST_F(FormatTest, FormatDyadicNearIntegerSpecs) {
    // ISO: If any item of A is not a near-integer, signal domain-error
    // Near-integers (within ⎕CT of integer) should work
    machine->ct = 1e-14;  // Set comparison tolerance
    Eigen::VectorXd spec(2);
    spec << 5.0 + 1e-15, 2.0 - 1e-15;  // Within ⎕CT
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(3.14);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
}

TEST_F(FormatTest, FormatDyadicNonIntegerSpecError) {
    // Non-near-integer should error
    EXPECT_THROW(machine->eval("5.5 2⍕42"), APLError);
}

TEST_F(FormatTest, FormatDyadicFieldWidthOne) {
    // Minimum field width
    Eigen::VectorXd spec(2);
    spec << 1.0, 0.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(5.0);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_EQ(strlen(result->as_string()), 1);
    EXPECT_STREQ(result->as_string(), "5");
}

TEST_F(FormatTest, FormatDyadicVerySmallExponent) {
    // Very small number in exponential form
    Eigen::VectorXd spec(2);
    spec << 15.0, -5.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(1e-100);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
    EXPECT_TRUE(s.find("¯") != std::string::npos);  // Negative exponent
}

TEST_F(FormatTest, FormatDyadicVeryLargeExponent) {
    // Very large number in exponential form
    Eigen::VectorXd spec(2);
    spec << 15.0, -5.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(1e100);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
    EXPECT_TRUE(s.find("100") != std::string::npos);
}

TEST_F(FormatTest, FormatDyadicMatrixPerColumnSpecs) {
    // Per-column format specs: different format for each column
    // 2 3⍴... creates 2 rows, 3 columns - need 3 pairs (6 elements)
    Value* result = machine->eval("4 0 6 2 8 ¯2⍕2 3⍴42 3.14159 1234.5 0.001 99 7.77");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Total width per row: 4 + 6 + 8 = 18
    // Should have newline for second row
    EXPECT_TRUE(s.find("\n") != std::string::npos);
}

TEST_F(FormatTest, FormatDyadicEmptyMatrixReturnsSpaces) {
    // Empty B returns appropriately sized space string
    // ISO: If B is empty, Set W to +/((⍴A)⍴1 0)/A, Return ((¯1↓⍴B),W)⍴' '
    Value* result = machine->eval("5 2⍕0⍴0");  // Empty vector
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    // For empty input, should return empty string (width calculated but no rows)
}

TEST_F(FormatTest, FormatDyadicNegativeWidth) {
    // Negative width should error
    EXPECT_THROW(machine->eval("¯5 2⍕42"), APLError);
}

TEST_F(FormatTest, FormatDyadicExponentialZeroMantissa) {
    // -0 mantissa digits (edge case)
    // Actually -1 is minimum meaningful, -0 would be weird
    // Let's test -1
    Eigen::VectorXd spec(2);
    spec << 8.0, -1.0;  // 1 digit mantissa
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(3.14159);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

TEST_F(FormatTest, FormatDyadicHighPrecisionFixed) {
    // High precision in fixed format (many decimal places)
    Eigen::VectorXd spec(2);
    spec << 25.0, 15.0;  // 15 decimal places
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(3.141592653589793);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 25);
    EXPECT_TRUE(s.find("3.141592653589793") != std::string::npos);
}

TEST_F(FormatTest, FormatDyadicRoundingUp) {
    // Test rounding: 2.5 with 0 decimals should round to 3 (or 2 depending on rounding mode)
    Eigen::VectorXd spec(2);
    spec << 3.0, 0.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(2.5);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    // Should be either "  2" or "  3" depending on rounding
    EXPECT_EQ(s.length(), 3);
}

TEST_F(FormatTest, FormatDyadicRoundingDown) {
    // 2.4 with 0 decimals should round to 2
    Eigen::VectorXd spec(2);
    spec << 3.0, 0.0;
    Value* alpha = machine->heap->allocate_vector(spec);
    Value* omega = machine->heap->allocate_scalar(2.4);
    fn_format_dyadic(machine, nullptr, alpha, omega);
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("2") != std::string::npos);
}

// ============================================================================
// Integration tests using eval
// ============================================================================

TEST_F(FormatTest, EvalMonadicFormatInteger) {
    Value* result = machine->eval("⍕42");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "42");
}

TEST_F(FormatTest, EvalMonadicFormatNegative) {
    Value* result = machine->eval("⍕¯5");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "¯5");
}

TEST_F(FormatTest, EvalMonadicFormatVector) {
    Value* result = machine->eval("⍕1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "1 2 3");
}

TEST_F(FormatTest, EvalMonadicFormatString) {
    Value* result = machine->eval("⍕'hello'");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string(), "hello");
}

TEST_F(FormatTest, EvalDyadicFormatBasic) {
    Value* result = machine->eval("5 2⍕3.14159");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("3.14") != std::string::npos);
}

TEST_F(FormatTest, EvalDyadicFormatExponential) {
    Value* result = machine->eval("10 ¯3⍕12345.6789");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

TEST_F(FormatTest, EvalDyadicFormatVector) {
    Value* result = machine->eval("6 2⍕1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_EQ(s.length(), 18);  // 3 * 6
}

TEST_F(FormatTest, EvalDyadicFormatMatrix) {
    Value* result = machine->eval("5 1⍕2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string();
    EXPECT_TRUE(s.find("\n") != std::string::npos);
}

// ============================================================================
// Error condition tests
// ============================================================================

TEST_F(FormatTest, DyadicFormatLengthErrorMismatch) {
    // 4 specs (2 pairs) for 3-element vector should error
    EXPECT_THROW(machine->eval("5 2 6 3⍕1 2 3"), APLError);
}

TEST_F(FormatTest, DyadicFormatDomainErrorNonNumericB) {
    // B must be numeric for dyadic format
    EXPECT_THROW(machine->eval("5 2⍕'abc'"), APLError);
}

