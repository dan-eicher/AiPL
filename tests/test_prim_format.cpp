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

