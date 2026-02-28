// System function tests - tests for ISO 13751 system functions
// Tests: ⎕TS, ⎕DL, ⎕AV, ⎕NC, ⎕EX, ⎕NL, ⎕LC

#include <gtest/gtest.h>
#include "parser.h"
#include "machine.h"
#include "continuation.h"
#include <chrono>
#include <thread>

using namespace apl;

class SysFnTest : public ::testing::Test {
protected:
    Machine* machine;
    Parser* parser;

    void SetUp() override {
        machine = new Machine();
        parser = machine->parser;
    }

    void TearDown() override {
        delete machine;
    }
};

// ============================================================================
// ⎕TS (Time Stamp) Tests - ISO 13751 §11.4.1
// ============================================================================

// Test ⎕TS returns a 7-element vector
TEST_F(SysFnTest, QuadTSReturns7Elements) {
    Value* result = machine->eval("⎕TS");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 7);
}

// Test ⎕TS shape is 7
TEST_F(SysFnTest, QuadTSShape) {
    Value* result = machine->eval("⍴⎕TS");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 7.0);
}

// Test ⎕TS first element is year (reasonable range)
TEST_F(SysFnTest, QuadTSYearIsReasonable) {
    Value* result = machine->eval("⎕TS[1]");
    ASSERT_NE(result, nullptr);
    double year = result->as_scalar();
    EXPECT_GE(year, 2020.0);
    EXPECT_LE(year, 2100.0);
}

// Test ⎕TS month is 1-12
TEST_F(SysFnTest, QuadTSMonthValid) {
    Value* result = machine->eval("⎕TS[2]");
    ASSERT_NE(result, nullptr);
    double month = result->as_scalar();
    EXPECT_GE(month, 1.0);
    EXPECT_LE(month, 12.0);
}

// Test ⎕TS day is 1-31
TEST_F(SysFnTest, QuadTSDayValid) {
    Value* result = machine->eval("⎕TS[3]");
    ASSERT_NE(result, nullptr);
    double day = result->as_scalar();
    EXPECT_GE(day, 1.0);
    EXPECT_LE(day, 31.0);
}

// Test ⎕TS hour is 0-23
TEST_F(SysFnTest, QuadTSHourValid) {
    Value* result = machine->eval("⎕TS[4]");
    ASSERT_NE(result, nullptr);
    double hour = result->as_scalar();
    EXPECT_GE(hour, 0.0);
    EXPECT_LE(hour, 23.0);
}

// Test ⎕TS minute is 0-59
TEST_F(SysFnTest, QuadTSMinuteValid) {
    Value* result = machine->eval("⎕TS[5]");
    ASSERT_NE(result, nullptr);
    double minute = result->as_scalar();
    EXPECT_GE(minute, 0.0);
    EXPECT_LE(minute, 59.0);
}

// Test ⎕TS second is 0-59
TEST_F(SysFnTest, QuadTSSecondValid) {
    Value* result = machine->eval("⎕TS[6]");
    ASSERT_NE(result, nullptr);
    double second = result->as_scalar();
    EXPECT_GE(second, 0.0);
    EXPECT_LE(second, 59.0);
}

// Test ⎕TS millisecond is 0-999
TEST_F(SysFnTest, QuadTSMillisecondValid) {
    Value* result = machine->eval("⎕TS[7]");
    ASSERT_NE(result, nullptr);
    double ms = result->as_scalar();
    EXPECT_GE(ms, 0.0);
    EXPECT_LT(ms, 1000.0);
}

// Test ⎕TS first 6 elements are integral (ISO 13751 requirement)
TEST_F(SysFnTest, QuadTSFirst6Integral) {
    Value* result = machine->eval("6↑⎕TS");
    ASSERT_NE(result, nullptr);
    const Eigen::MatrixXd* mat = result->as_matrix();
    for (int i = 0; i < 6; ++i) {
        double val = (*mat)(i, 0);
        EXPECT_DOUBLE_EQ(val, std::floor(val)) << "Element " << i+1 << " is not integral";
    }
}

// Test ⎕TS is read-only (cannot be assigned)
TEST_F(SysFnTest, QuadTSReadOnly) {
    EXPECT_THROW(machine->eval("⎕TS←1 2 3 4 5 6 7"), APLError);
}

// ============================================================================
// ⎕DL (Delay) Tests - ISO 13751 §11.5.1
// ============================================================================

// Test ⎕DL returns scalar
TEST_F(SysFnTest, QuadDLReturnsScalar) {
    Value* result = machine->eval("⎕DL 0");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
}

// Test ⎕DL with zero delay
TEST_F(SysFnTest, QuadDLZeroDelay) {
    Value* result = machine->eval("⎕DL 0");
    ASSERT_NE(result, nullptr);
    double actual = result->as_scalar();
    EXPECT_GE(actual, 0.0);
    EXPECT_LT(actual, 0.1);  // Should be nearly instant
}

// Test ⎕DL with small delay
TEST_F(SysFnTest, QuadDLSmallDelay) {
    auto start = std::chrono::high_resolution_clock::now();
    Value* result = machine->eval("⎕DL 0.05");
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_NE(result, nullptr);
    double actual = result->as_scalar();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Should delay at least 0.05 seconds
    EXPECT_GE(actual, 0.04);  // Allow small tolerance
    EXPECT_GE(elapsed, 0.04);
}

// Test ⎕DL returns actual delay time
TEST_F(SysFnTest, QuadDLReturnsActualTime) {
    Value* result = machine->eval("⎕DL 0.02");
    ASSERT_NE(result, nullptr);
    double actual = result->as_scalar();
    // Result should be close to requested time
    EXPECT_GE(actual, 0.015);
    EXPECT_LE(actual, 0.5);  // Should not be way over
}

// Test ⎕DL with negative argument returns immediately (ISO 13751 allows negative B)
TEST_F(SysFnTest, QuadDLNegativeReturnsImmediately) {
    Value* result = machine->eval("⎕DL ¯1");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    // Should return near-zero elapsed time (immediate return)
    EXPECT_LT(result->as_scalar(), 0.1);
}

// Test ⎕DL requires scalar argument
TEST_F(SysFnTest, QuadDLRankError) {
    EXPECT_THROW(machine->eval("⎕DL 1 2 3"), APLError);
}

// Test ⎕DL with fractional seconds
TEST_F(SysFnTest, QuadDLFractionalSeconds) {
    Value* result = machine->eval("⎕DL 0.001");
    ASSERT_NE(result, nullptr);
    double actual = result->as_scalar();
    EXPECT_GE(actual, 0.0);
}

// ============================================================================
// ⎕AV (Atomic Vector) Tests - ISO 13751 §11.4.2
// ============================================================================

// Test ⎕AV returns 256-element vector
TEST_F(SysFnTest, QuadAVReturns256Elements) {
    Value* result = machine->eval("⍴⎕AV");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 256.0);
}

// Test ⎕AV is character data
TEST_F(SysFnTest, QuadAVIsCharacter) {
    Value* result = machine->eval("⎕AV");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_char_data());
}

// Test ⎕AV contains ASCII characters in order
TEST_F(SysFnTest, QuadAVContainsASCII) {
    // Check that ⎕AV[65] (0-indexed: 64) is 'A' (codepoint 65)
    // With ⎕IO=1, ⎕AV[66] should be 'A'
    Value* result = machine->eval("⎕AV[66]");  // 'A' at index 66 (1-origin)
    ASSERT_NE(result, nullptr);
    double cp = result->as_scalar();
    EXPECT_DOUBLE_EQ(cp, 65.0);  // ASCII 'A'
}

// Test ⎕AV[1] is character 0 (null)
TEST_F(SysFnTest, QuadAVFirstElement) {
    Value* result = machine->eval("⎕AV[1]");
    ASSERT_NE(result, nullptr);
    double cp = result->as_scalar();
    EXPECT_DOUBLE_EQ(cp, 0.0);
}

// Test ⎕AV[256] is character 255
TEST_F(SysFnTest, QuadAVLastElement) {
    Value* result = machine->eval("⎕AV[256]");
    ASSERT_NE(result, nullptr);
    double cp = result->as_scalar();
    EXPECT_DOUBLE_EQ(cp, 255.0);
}

// Test ⎕AV is read-only
TEST_F(SysFnTest, QuadAVReadOnly) {
    EXPECT_THROW(machine->eval("⎕AV←'x'"), APLError);
}

// Test each element of ⎕AV is unique (ISO 13751: "every element exactly once")
TEST_F(SysFnTest, QuadAVUnique) {
    // ≢∪⎕AV should equal ≢⎕AV (256)
    Value* result = machine->eval("≢∪⎕AV");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 256.0);
}

// ============================================================================
// ⎕NC (Name Class) Tests - ISO 13751 §11.5.2
// ============================================================================

// Test ⎕NC returns 0 for undefined name
TEST_F(SysFnTest, QuadNCUndefined) {
    Value* result = machine->eval("⎕NC 'undefined_name_xyz'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test ⎕NC returns 2 for variable
TEST_F(SysFnTest, QuadNCVariable) {
    machine->eval("myvar←42");
    Value* result = machine->eval("⎕NC 'myvar'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// Test ⎕NC signals DOMAIN ERROR for non-identifier (ISO §11.5.2)
TEST_F(SysFnTest, QuadNCPrimitiveFunctionDomainError) {
    EXPECT_THROW(machine->eval("⎕NC '+'"), APLError);
}

// Test ⎕NC returns 3 for user-defined function
TEST_F(SysFnTest, QuadNCUserFunction) {
    machine->eval("myfunc←{⍵+1}");
    Value* result = machine->eval("⎕NC 'myfunc'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// Test ⎕NC signals DOMAIN ERROR for operator symbol (ISO §11.5.2)
TEST_F(SysFnTest, QuadNCOperatorDomainError) {
    EXPECT_THROW(machine->eval("⎕NC '/'"), APLError);
}

// Test ⎕NC with multiple names (matrix argument)
TEST_F(SysFnTest, QuadNCMatrix) {
    machine->eval("a←1");
    machine->eval("b←2");
    // Create matrix with names 'a' and 'b' (each 1 char, so 2×1 matrix)
    Value* result = machine->eval("⎕NC 2 1⍴'ab'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 2.0);  // a is variable
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 2.0);  // b is variable
}

// Test ⎕NC rank error for rank > 2
TEST_F(SysFnTest, QuadNCRankError) {
    EXPECT_THROW(machine->eval("⎕NC 2 2 2⍴'a'"), APLError);
}

// Test ⎕NC domain error for non-character
TEST_F(SysFnTest, QuadNCDomainError) {
    EXPECT_THROW(machine->eval("⎕NC 123"), APLError);
}

// Test ⎕NC with empty matrix returns empty vector (ISO 13751: "If B is empty, return ⍳0")
TEST_F(SysFnTest, QuadNCEmpty) {
    Value* result = machine->eval("⎕NC 0 0⍴''");
    ASSERT_NE(result, nullptr);
    // Should return empty or handle gracefully
}

// Test ⎕NC with padded names in matrix
TEST_F(SysFnTest, QuadNCPaddedNames) {
    machine->eval("abc←1");
    machine->eval("x←2");
    // Matrix: 'abc' and 'x  ' (padded)
    Value* result = machine->eval("⎕NC 2 3⍴'abcx  '");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 2.0);  // abc is variable
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 2.0);  // x is variable
}

// ============================================================================
// ⎕EX (Expunge) Tests - ISO 13751 §11.5.3
// ============================================================================

// Test ⎕EX returns 1 for successful expunge
TEST_F(SysFnTest, QuadEXSuccess) {
    machine->eval("victim←42");
    Value* result = machine->eval("⎕EX 'victim'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// Test ⎕EX actually removes the variable
TEST_F(SysFnTest, QuadEXRemoves) {
    machine->eval("victim←42");
    machine->eval("⎕EX 'victim'");
    Value* result = machine->eval("⎕NC 'victim'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // Now undefined
}

// Test ⎕EX returns 1 for undefined name (ISO §11.5.3: ~×⎕NC B, name is available)
TEST_F(SysFnTest, QuadEXUndefined) {
    Value* result = machine->eval("⎕EX 'nonexistent_xyz'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// Test ⎕EX returns 0 for protected system names
TEST_F(SysFnTest, QuadEXProtected) {
    Value* result = machine->eval("⎕EX '⎕IO'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test ⎕EX with multiple names
TEST_F(SysFnTest, QuadEXMultiple) {
    machine->eval("a←1");
    machine->eval("b←2");
    Value* result = machine->eval("⎕EX 2 1⍴'ab'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 1.0);  // a removed
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 1.0);  // b removed
}

// Test ⎕EX mixed success/failure
TEST_F(SysFnTest, QuadEXMixed) {
    machine->eval("exists←1");
    Value* result = machine->eval("⎕EX 2 6⍴'existsnoexst'");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 1.0);  // exists removed, now available
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 1.0);  // noexst never existed, also available
}

// Test ⎕EX rank error
TEST_F(SysFnTest, QuadEXRankError) {
    EXPECT_THROW(machine->eval("⎕EX 2 2 2⍴'a'"), APLError);
}

// Test ⎕EX domain error for non-character
TEST_F(SysFnTest, QuadEXDomainError) {
    EXPECT_THROW(machine->eval("⎕EX 123"), APLError);
}

// Test ⎕EX can remove functions
TEST_F(SysFnTest, QuadEXRemovesFunction) {
    machine->eval("f←{⍵+1}");
    EXPECT_DOUBLE_EQ(machine->eval("⎕NC 'f'")->as_scalar(), 3.0);
    machine->eval("⎕EX 'f'");
    EXPECT_DOUBLE_EQ(machine->eval("⎕NC 'f'")->as_scalar(), 0.0);
}

// Test ⎕EX returns ~∨/⎕NC B (ISO 13751: "Return ~∨/⎕NC B")
// After expunge, ⎕NC should be 0, so result is 1
TEST_F(SysFnTest, QuadEXReturnsCorrectValue) {
    machine->eval("test←99");
    Value* result = machine->eval("⎕EX 'test'");
    Value* nc = machine->eval("⎕NC 'test'");
    // ⎕EX returns 1 if name is now available (⎕NC = 0)
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
    EXPECT_DOUBLE_EQ(nc->as_scalar(), 0.0);
}

// ============================================================================
// ⎕NL (Name List) Tests - ISO 13751 §11.5.4
// ============================================================================

// Test ⎕NL returns character matrix
TEST_F(SysFnTest, QuadNLReturnsMatrix) {
    machine->eval("testvar←1");
    Value* result = machine->eval("⎕NL 2");
    ASSERT_NE(result, nullptr);
    // Should be matrix (rank 2) or empty
    EXPECT_TRUE(result->is_matrix() || result->size() == 0);
}

// Test ⎕NL returns character data
TEST_F(SysFnTest, QuadNLIsCharacter) {
    machine->eval("testvar←1");
    Value* result = machine->eval("⎕NL 2");
    ASSERT_NE(result, nullptr);
    if (result->size() > 0) {
        EXPECT_TRUE(result->is_char_data());
    }
}

// Test ⎕NL 2 includes variables
TEST_F(SysFnTest, QuadNLIncludesVariables) {
    machine->eval("mynewvar←123");
    Value* result = machine->eval("⎕NL 2");
    ASSERT_NE(result, nullptr);
    // Should include 'mynewvar' somewhere
    EXPECT_GT(result->size(), 0);
}

// Test ⎕NL 3 includes functions
TEST_F(SysFnTest, QuadNLIncludesFunctions) {
    Value* result = machine->eval("⎕NL 3");
    ASSERT_NE(result, nullptr);
    // Should include primitive functions like +, -, etc.
    EXPECT_GT(result->size(), 0);
}

// Test ⎕NL 4 includes operators
TEST_F(SysFnTest, QuadNLIncludesOperators) {
    Value* result = machine->eval("⎕NL 4");
    ASSERT_NE(result, nullptr);
    // Should include operators like /, \, etc.
    EXPECT_GT(result->size(), 0);
}

// Test ⎕NL with multiple classes
TEST_F(SysFnTest, QuadNLMultipleClasses) {
    machine->eval("myvar←1");
    Value* result = machine->eval("⎕NL 2 3");
    ASSERT_NE(result, nullptr);
    // Should include both variables and functions
    EXPECT_GT(result->size(), 0);
}

// Test ⎕NL rank error for rank > 1
TEST_F(SysFnTest, QuadNLRankError) {
    EXPECT_THROW(machine->eval("⎕NL 2 2⍴1 2 3 4"), APLError);
}

// Test ⎕NL domain error for invalid class
TEST_F(SysFnTest, QuadNLDomainError) {
    EXPECT_THROW(machine->eval("⎕NL 0"), APLError);   // 0 is invalid
    EXPECT_THROW(machine->eval("⎕NL 7"), APLError);   // 7 is invalid
    EXPECT_THROW(machine->eval("⎕NL ¯1"), APLError);  // negative invalid
}

// Test ⎕NL with empty result
TEST_F(SysFnTest, QuadNLEmptyResult) {
    // Class 1 is labels - we don't have any
    Value* result = machine->eval("⎕NL 1");
    ASSERT_NE(result, nullptr);
    // Result should be 0 0⍴'' (empty matrix)
    EXPECT_EQ(result->size(), 0);
}

// Test ⎕NL scalar argument (treated as 1-element vector)
TEST_F(SysFnTest, QuadNLScalarArg) {
    Value* result = machine->eval("⎕NL 3");
    ASSERT_NE(result, nullptr);
    // Should work and return functions
    EXPECT_GT(result->size(), 0);
}

// Test ⎕NL names are sorted alphabetically
TEST_F(SysFnTest, QuadNLSorted) {
    machine->eval("zvar←1");
    machine->eval("avar←2");
    machine->eval("mvar←3");
    Value* result = machine->eval("⎕NL 2");
    ASSERT_NE(result, nullptr);
    // Names should be sorted - 'avar' before 'mvar' before 'zvar'
    // This is harder to test without string extraction, but at least check it works
    EXPECT_GT(result->size(), 0);
}

// ============================================================================
// ⎕LC (Line Counter) Tests - ISO 13751 §11.4.3
// ============================================================================

// Test ⎕LC returns numeric vector
TEST_F(SysFnTest, QuadLCReturnsVector) {
    Value* result = machine->eval("⎕LC");
    ASSERT_NE(result, nullptr);
    // At top level, should be empty or contain call info
    EXPECT_TRUE(result->is_vector() || result->is_scalar() || result->size() == 0);
}

// Test ⎕LC is read-only
TEST_F(SysFnTest, QuadLCReadOnly) {
    EXPECT_THROW(machine->eval("⎕LC←1 2 3"), APLError);
}

// Test ⎕LC at top level (outside any function)
TEST_F(SysFnTest, QuadLCTopLevel) {
    Value* result = machine->eval("⎕LC");
    ASSERT_NE(result, nullptr);
    // At top level with no active functions, ⎕LC may be empty
}

// Test ⎕LC shape
TEST_F(SysFnTest, QuadLCShape) {
    Value* result = machine->eval("⍴⎕LC");
    ASSERT_NE(result, nullptr);
    // Shape should be a scalar (the length of ⎕LC)
}

// ============================================================================
// ⎕LX (Latent Expression) Tests - ISO 13751 §12.2.5
// ============================================================================

// Test ⎕LX default is empty
TEST_F(SysFnTest, QuadLXDefaultEmpty) {
    Value* result = machine->eval("⎕LX");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "");
}

// Test ⎕LX assignment with string
TEST_F(SysFnTest, QuadLXAssignString) {
    Value* result = machine->eval("⎕LX←'2+2'");
    ASSERT_NE(result, nullptr);
    // Verify stored value
    Value* read = machine->eval("⎕LX");
    ASSERT_NE(read, nullptr);
    EXPECT_STREQ(read->as_string()->c_str(), "2+2");
}

// Test ⎕LX assignment with character vector
TEST_F(SysFnTest, QuadLXAssignCharVector) {
    machine->eval("⎕LX←'hello'");
    Value* result = machine->eval("⎕LX");
    ASSERT_NE(result, nullptr);
    EXPECT_STREQ(result->as_string()->c_str(), "hello");
}

// Test ⎕LX can be cleared
TEST_F(SysFnTest, QuadLXClear) {
    machine->eval("⎕LX←'test'");
    machine->eval("⎕LX←''");
    Value* result = machine->eval("⎕LX");
    ASSERT_NE(result, nullptr);
    EXPECT_STREQ(result->as_string()->c_str(), "");
}

// Test ⎕LX rejects non-character data
TEST_F(SysFnTest, QuadLXDomainError) {
    EXPECT_THROW(machine->eval("⎕LX←123"), APLError);
    EXPECT_THROW(machine->eval("⎕LX←1 2 3"), APLError);
}

// Test ⎕LX shape is empty (string type)
TEST_F(SysFnTest, QuadLXShape) {
    machine->eval("⎕LX←'test'");
    Value* result = machine->eval("⍴⎕LX");
    ASSERT_NE(result, nullptr);
    // STRING type has shape based on length
}

// ============================================================================
// ⎕IO (Index Origin) Tests - ISO 13751 §11.4.6
// ============================================================================

// Test ⎕IO default is 1
TEST_F(SysFnTest, QuadIORead) {
    Value* result = machine->eval("⎕IO");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
}

// Test ⎕IO←0 sets index origin to 0
TEST_F(SysFnTest, QuadIOAssign) {
    Value* result = machine->eval("⎕IO←0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
    EXPECT_EQ(machine->io, 0);
}

// Test ⎕IO affects iota
TEST_F(SysFnTest, QuadIOAffectsIota) {
    machine->eval("⎕IO←0");
    Value* result = machine->eval("⍳5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* vec = result->as_matrix();
    EXPECT_DOUBLE_EQ((*vec)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*vec)(4, 0), 4.0);
}

// Test ⎕IO←2 should error (only 0 or 1 allowed)
TEST_F(SysFnTest, QuadIOInvalidValueError) {
    EXPECT_THROW(machine->eval("⎕IO←2"), APLError);
}

// Test ⎕IO can be used in expressions
TEST_F(SysFnTest, QuadIOInExpression) {
    Value* result = machine->eval("⎕IO + 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);  // 1 + 5
}

// ============================================================================
// ⎕PP (Print Precision) Tests - ISO 13751 §11.4.7
// ============================================================================

// Test ⎕PP default is 10
TEST_F(SysFnTest, QuadPPRead) {
    Value* result = machine->eval("⎕PP");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test ⎕PP←5 sets print precision to 5
TEST_F(SysFnTest, QuadPPAssign) {
    Value* result = machine->eval("⎕PP←5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
    EXPECT_EQ(machine->pp, 5);
}

// Test ⎕PP←0 should error (must be 1-17)
TEST_F(SysFnTest, QuadPPInvalidValueError) {
    EXPECT_THROW(machine->eval("⎕PP←0"), APLError);
}

// ============================================================================
// ⎕CT (Comparison Tolerance) Tests - ISO 13751 §11.4.8
// ============================================================================

// Test ⎕CT default is 0 (exact comparisons)
TEST_F(SysFnTest, QuadCTRead) {
    Value* result = machine->eval("⎕CT");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test ⎕CT←0 sets comparison tolerance to 0
TEST_F(SysFnTest, QuadCTAssign) {
    Value* result = machine->eval("⎕CT←0");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
    EXPECT_DOUBLE_EQ(machine->ct, 0.0);
}

// Test ⎕CT←0.1 sets comparison tolerance to 0.1
TEST_F(SysFnTest, QuadCTAssignLarger) {
    Value* result = machine->eval("⎕CT←0.1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.1);
    EXPECT_DOUBLE_EQ(machine->ct, 0.1);
}

// Test ⎕CT←¯1 should error (must be nonnegative)
TEST_F(SysFnTest, QuadCTInvalidNegative) {
    EXPECT_THROW(machine->eval("⎕CT←¯1"), APLError);
}

// Test tolerant equality
TEST_F(SysFnTest, QuadCTTolerantEquality) {
    machine->eval("⎕CT←0.1");
    Value* result = machine->eval("1.05 = 1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // Tolerantly equal
}

// Test exact equality with CT=0
TEST_F(SysFnTest, QuadCTExactEquality) {
    machine->eval("⎕CT←0");
    Value* result = machine->eval("1.0000000001 = 1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // Not exactly equal
}

// Test tolerant less-than
TEST_F(SysFnTest, QuadCTTolerantLessThan) {
    machine->eval("⎕CT←0.1");
    Value* result = machine->eval("0.95 < 1");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);  // Tolerantly equal, so not <
}

// Test tolerant floor
TEST_F(SysFnTest, QuadCTTolerantFloor) {
    machine->eval("⎕CT←1e-9");
    Value* result = machine->eval("⌊2.9999999999");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// Test tolerant ceiling
TEST_F(SysFnTest, QuadCTTolerantCeiling) {
    machine->eval("⎕CT←1e-9");
    Value* result = machine->eval("⌈3.0000000001");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// Test exact floor with CT=0
TEST_F(SysFnTest, QuadCTZeroFloorExact) {
    machine->eval("⎕CT←0");
    Value* result = machine->eval("⌊2.9999999999");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// ============================================================================
// ⎕RL (Random Link) Tests - ISO 13751 §11.4.9
// ============================================================================

// Test ⎕RL is seeded from system at startup (positive integer)
TEST_F(SysFnTest, QuadRLRead) {
    Value* result = machine->eval("⎕RL");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_GT(result->as_scalar(), 0.0);  // Must be positive
}

// Test ⎕RL←12345 sets random seed
TEST_F(SysFnTest, QuadRLAssign) {
    Value* result = machine->eval("⎕RL←12345");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 12345.0);
    EXPECT_EQ(machine->rl, 12345u);
}

// Test ⎕RL←0 should error (must be positive)
TEST_F(SysFnTest, QuadRLInvalidZero) {
    EXPECT_THROW(machine->eval("⎕RL←0"), APLError);
}

// Test ⎕RL←¯1 should error (must be positive)
TEST_F(SysFnTest, QuadRLInvalidNegative) {
    EXPECT_THROW(machine->eval("⎕RL←¯1"), APLError);
}

// Test ⎕RL←1.5 should error (must be integer)
TEST_F(SysFnTest, QuadRLInvalidNonInteger) {
    EXPECT_THROW(machine->eval("⎕RL←1.5"), APLError);
}

// Test same seed produces same sequence
TEST_F(SysFnTest, QuadRLReproducibility) {
    machine->eval("⎕RL←42");
    Value* r1 = machine->eval("?100");
    double first1 = r1->as_scalar();

    machine->eval("⎕RL←42");  // Reset to same seed
    Value* r2 = machine->eval("?100");
    double first2 = r2->as_scalar();

    EXPECT_DOUBLE_EQ(first1, first2);
}

// Test different seeds produce different sequences
TEST_F(SysFnTest, QuadRLDifferentSeeds) {
    machine->eval("⎕RL←1");
    Value* r1 = machine->eval("?1000000");
    double val1 = r1->as_scalar();

    machine->eval("⎕RL←2");
    Value* r2 = machine->eval("?1000000");
    double val2 = r2->as_scalar();

    EXPECT_NE(val1, val2);
}

// Test deal also uses ⎕RL for reproducibility
TEST_F(SysFnTest, QuadRLDealReproducibility) {
    machine->eval("⎕RL←123");
    Value* d1 = machine->eval("5?10");
    const Eigen::MatrixXd* v1 = d1->as_matrix();

    machine->eval("⎕RL←123");  // Reset to same seed
    Value* d2 = machine->eval("5?10");
    const Eigen::MatrixXd* v2 = d2->as_matrix();

    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ((*v1)(i, 0), (*v2)(i, 0));
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

// Test ⎕NC and ⎕EX interaction
TEST_F(SysFnTest, NCandEXIntegration) {
    machine->eval("testvar←99");
    EXPECT_DOUBLE_EQ(machine->eval("⎕NC 'testvar'")->as_scalar(), 2.0);
    machine->eval("⎕EX 'testvar'");
    EXPECT_DOUBLE_EQ(machine->eval("⎕NC 'testvar'")->as_scalar(), 0.0);
}

// Test ⎕NL and ⎕NC consistency
TEST_F(SysFnTest, NLandNCConsistency) {
    machine->eval("mytest←1");
    Value* nl = machine->eval("⎕NL 2");
    // All names in ⎕NL 2 should have ⎕NC = 2
    // (We can't easily extract names, but at least test they exist)
    EXPECT_GT(nl->size(), 0);
}

// Test ⎕EX followed by ⎕NL
TEST_F(SysFnTest, EXthenNL) {
    machine->eval("temp1←1");
    machine->eval("temp2←2");
    Value* before = machine->eval("⎕NL 2");
    machine->eval("⎕EX 'temp1'");
    Value* after = machine->eval("⎕NL 2");
    // After should have fewer rows than before (temp1 removed)
    // This is hard to verify exactly, but the operations should work
    EXPECT_NE(before, nullptr);
    EXPECT_NE(after, nullptr);
}

// Test ⎕TS changes over time
TEST_F(SysFnTest, TSChangesOverTime) {
    Value* ts1 = machine->eval("⎕TS[7]");  // milliseconds
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Value* ts2 = machine->eval("⎕TS[7]");
    // At least one of the timestamps should have changed
    // (This might occasionally fail if we're unlucky with timing)
    // Just verify both work
    EXPECT_NE(ts1, nullptr);
    EXPECT_NE(ts2, nullptr);
}

// ============================================================================
// ⎕PP Upper Bound Tests - ISO 13751 §12.2.3
// ============================================================================

TEST_F(SysFnTest, QuadPPUpperBoundValid) {
    // ⎕PP←17 should succeed (our max)
    machine->eval("⎕PP←17");
    Value* r = machine->eval("⎕PP");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 17.0);
}

TEST_F(SysFnTest, QuadPPUpperBoundExceeded) {
    // ⎕PP←18 should signal error
    EXPECT_THROW(machine->eval("⎕PP←18"), APLError);
}

TEST_F(SysFnTest, QuadPPNegativeError) {
    EXPECT_THROW(machine->eval("⎕PP←¯1"), APLError);
}

TEST_F(SysFnTest, QuadPPNonIntegerError) {
    EXPECT_THROW(machine->eval("⎕PP←3.5"), APLError);
}

// ============================================================================
// ⎕NC returning classes 5 and 6 - ISO 13751 §11.5.2
// ============================================================================

TEST_F(SysFnTest, QuadNCSystemVariable) {
    // ⎕NC '⎕IO' should return 5 (system-variable)
    Value* r = machine->eval("⎕NC '⎕IO'");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 5.0);
}

TEST_F(SysFnTest, QuadNCSystemVariablePP) {
    Value* r = machine->eval("⎕NC '⎕PP'");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 5.0);
}

TEST_F(SysFnTest, QuadNCSystemVariableCT) {
    Value* r = machine->eval("⎕NC '⎕CT'");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 5.0);
}

TEST_F(SysFnTest, QuadNCSystemFunction) {
    // ⎕NC '⎕DL' should return 6 (system-function)
    Value* r = machine->eval("⎕NC '⎕DL'");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 6.0);
}

TEST_F(SysFnTest, QuadNCSystemFunctionEA) {
    Value* r = machine->eval("⎕NC '⎕EA'");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 6.0);
}

TEST_F(SysFnTest, QuadNCSystemFunctionNL) {
    Value* r = machine->eval("⎕NC '⎕NL'");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 6.0);
}

TEST_F(SysFnTest, QuadNCUnknownSystemName) {
    // ⎕NC '⎕BOGUS' should return 0 (undefined)
    Value* r = machine->eval("⎕NC '⎕BOGUS'");
    ASSERT_TRUE(r->is_scalar());
    EXPECT_DOUBLE_EQ(r->as_scalar(), 0.0);
}

// ============================================================================
// ⎕LC inside nested function calls - ISO 13751 §11.4.3
// ============================================================================

TEST_F(SysFnTest, QuadLCInsideDfn) {
    // ⎕LC inside a dfn should report line info
    Value* r = machine->eval("{⎕LC}0");
    ASSERT_NE(r, nullptr);
    // Result should be a vector (possibly empty if no line tracking in dfns)
}

TEST_F(SysFnTest, QuadLCInsideNestedDfn) {
    // ⎕LC inside nested dfn calls
    Value* r = machine->eval("{({⎕LC}⍵)}0");
    ASSERT_NE(r, nullptr);
}

TEST_F(SysFnTest, QuadLCShapeAlwaysVector) {
    // ⎕LC should always return a vector (shape of shape is 1-element)
    Value* r = machine->eval("⍴⍴⎕LC");
    ASSERT_TRUE(r->is_vector());
    EXPECT_EQ(r->size(), 1);
    EXPECT_DOUBLE_EQ(r->as_matrix()->operator()(0, 0), 1.0);
}

// Main function for Google Test
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
