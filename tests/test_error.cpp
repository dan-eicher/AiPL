// Error handling tests - tests for ISO 13751 error system functions
// Tests ⎕EA (Execute Alternate), ⎕ES (Error Signal), ⎕ET (Event Type), ⎕EM (Event Message)

#include <gtest/gtest.h>
#include "parser.h"
#include "machine.h"
#include "continuation.h"

using namespace apl;

class ErrorTest : public ::testing::Test {
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
// ⎕EA (Execute Alternate) Tests - ISO 13751 §11.6.4
// ============================================================================

// Test ⎕EA catches division by zero and executes alternate
TEST_F(ErrorTest, QuadEACatchesDivideByZero) {
    Value* result = machine->eval("'99' ⎕EA '1÷0'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 99.0);
}

// Test ⎕EA passes through successful expression
TEST_F(ErrorTest, QuadEASuccessPassThrough) {
    Value* result = machine->eval("'99' ⎕EA '1+2'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// Test ⎕EA with array result
TEST_F(ErrorTest, QuadEAArrayResult) {
    Value* result = machine->eval("'1 2 3' ⎕EA '1÷0'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_array());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(2, 0), 3.0);
}

// Test ⎕EA with expression in alternate
TEST_F(ErrorTest, QuadEAExpressionInAlternate) {
    Value* result = machine->eval("'10+5' ⎕EA '1÷0'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

// Test ⎕EA success path with complex expression
TEST_F(ErrorTest, QuadEAComplexSuccess) {
    Value* result = machine->eval("'0' ⎕EA '+/⍳5'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);  // 1+2+3+4+5
}

// Test ⎕EA requires character vectors
TEST_F(ErrorTest, QuadEADomainErrorOnNonString) {
    EXPECT_THROW(machine->eval("99 ⎕EA '1+2'"), APLError);
}

// ============================================================================
// ⎕ET (Event Type) Tests - ISO 13751 §11.4.4
// ============================================================================

// Test ⎕ET accessible in error handler
TEST_F(ErrorTest, QuadETInHandler) {
    Value* result = machine->eval("'⎕ET' ⎕EA '1÷0'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_array());
    EXPECT_EQ(result->size(), 2);
    // Division by zero should be DOMAIN ERROR class 11
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 11.0);
}

// Test ⎕ET shape in handler
TEST_F(ErrorTest, QuadETShape) {
    Value* result = machine->eval("'⍴⎕ET' ⎕EA '1÷0'");
    ASSERT_NE(result, nullptr);
    // ⍴ of a 2-element vector is 2
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 2.0);
}

// Test ⎕ET is 0 0 when no error
TEST_F(ErrorTest, QuadETZeroWhenNoError) {
    Value* result = machine->eval("⎕ET");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_array());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 0.0);
}

// ============================================================================
// ⎕EM (Event Message) Tests - ISO 13751 §11.4.5
// ============================================================================

// Test ⎕EM accessible in error handler
TEST_F(ErrorTest, QuadEMInHandler) {
    Value* result = machine->eval("'⎕EM' ⎕EA '1÷0'");
    ASSERT_NE(result, nullptr);
    // Should be a string containing the error message
    EXPECT_TRUE(result->is_string() || result->is_char_data());
}

// Test ⎕EM contains relevant text
TEST_F(ErrorTest, QuadEMContainsDomainError) {
    Value* result = machine->eval("'⎕EM' ⎕EA '1÷0'");
    ASSERT_NE(result, nullptr);
    std::string msg;
    if (result->is_string()) {
        msg = result->as_string();
    } else {
        msg = result->to_string_value(machine->heap)->as_string();
    }
    // Should mention DOMAIN ERROR
    EXPECT_TRUE(msg.find("DOMAIN") != std::string::npos || msg.find("divide") != std::string::npos);
}

// Test ⎕EM is empty when no error
TEST_F(ErrorTest, QuadEMEmptyWhenNoError) {
    Value* result = machine->eval("⎕EM");
    ASSERT_NE(result, nullptr);
    // Empty string
    if (result->is_string()) {
        EXPECT_STREQ(result->as_string(), "");
    } else if (result->is_array()) {
        EXPECT_EQ(result->size(), 0);
    }
}

// ============================================================================
// ⎕ES (Error Signal) Tests - ISO 13751 §11.5.7
// Note: ⎕ES is monadic - it takes a single argument
// ============================================================================

// Test ⎕ES signals an error with string message
TEST_F(ErrorTest, QuadESSignalsErrorWithString) {
    // ⎕ES 'message' signals an unclassified error
    Value* result = machine->eval("'42' ⎕EA '⎕ES ''test error'''");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test ⎕ES with 2-element error code
TEST_F(ErrorTest, QuadESWithErrorCode) {
    // ⎕ES 11 1 signals error with class 11, subclass 1
    Value* result = machine->eval("'⎕ET' ⎕EA '⎕ES 11 1'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_array());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 1.0);
}

// Test ⎕ES with 0 0 clears error state
TEST_F(ErrorTest, QuadESClearError) {
    // First trigger an error to set ⎕ET
    machine->eval("'0' ⎕EA '1÷0'");
    // ⎕ES 0 0 should clear the error state
    machine->eval("⎕ES 0 0");
    Value* result = machine->eval("⎕ET");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 0.0);
}

// Test ⎕ES with empty argument is no-op (ISO 13751 §11.5.7: conditional signalling)
TEST_F(ErrorTest, QuadESEmptyNoOp) {
    // Empty numeric vector - no action, no error
    Value* result = machine->eval("⎕ES ⍬");
    ASSERT_NE(result, nullptr);
}

// Test ⎕ES with empty string is no-op
TEST_F(ErrorTest, QuadESEmptyStringNoOp) {
    Value* result = machine->eval("⎕ES ''");
    ASSERT_NE(result, nullptr);
}

// ============================================================================
// Dyadic ⎕ES Tests - ISO 13751 §11.6.5
// ============================================================================

// Test dyadic ⎕ES with custom message and error code
TEST_F(ErrorTest, QuadESDyadicWithMessage) {
    Value* result = machine->eval("'⎕EM' ⎕EA '''my error'' ⎕ES 11 1'");
    ASSERT_NE(result, nullptr);
    std::string msg;
    if (result->is_string()) {
        msg = result->as_string();
    } else {
        msg = result->to_string_value(machine->heap)->as_string();
    }
    EXPECT_TRUE(msg.find("my error") != std::string::npos);
}

// Test dyadic ⎕ES sets correct error type
TEST_F(ErrorTest, QuadESDyadicErrorType) {
    Value* result = machine->eval("'⎕ET' ⎕EA '''test'' ⎕ES 3 5'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_array());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 5.0);
}

// Test dyadic ⎕ES with 0 0 clears error state
TEST_F(ErrorTest, QuadESDyadicClearError) {
    machine->eval("'0' ⎕EA '1÷0'");
    machine->eval("'clear' ⎕ES 0 0");
    Value* result = machine->eval("⎕ET");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 0.0);
}

// Test dyadic ⎕ES with empty right arg is no-op (conditional signalling)
TEST_F(ErrorTest, QuadESDyadicEmptyNoOp) {
    Value* result = machine->eval("'message' ⎕ES ⍬");
    ASSERT_NE(result, nullptr);
}

// Test dyadic ⎕ES requires character left argument
TEST_F(ErrorTest, QuadESDyadicNonCharLeftError) {
    EXPECT_THROW(machine->eval("99 ⎕ES 11 1"), APLError);
}

// Test dyadic ⎕ES requires 2-element right argument
TEST_F(ErrorTest, QuadESDyadicWrongLengthError) {
    EXPECT_THROW(machine->eval("'msg' ⎕ES 1 2 3"), APLError);
}

// ============================================================================
// Error Propagation Tests
// ============================================================================

// Test that uncaught errors propagate as APLError
TEST_F(ErrorTest, UncaughtErrorPropagates) {
    EXPECT_THROW(machine->eval("1÷0"), APLError);
}

// Test error in handler propagates
TEST_F(ErrorTest, ErrorInHandlerPropagates) {
    // Handler also throws, should propagate up
    EXPECT_THROW(machine->eval("'1÷0' ⎕EA '2÷0'"), APLError);
}

// ============================================================================
// Edge Cases
// ============================================================================

// Test ⎕EA with empty try expression
TEST_F(ErrorTest, QuadEAEmptyTry) {
    // Empty expression should return something (probably empty)
    Value* result = machine->eval("'fallback' ⎕EA ''");
    // Depends on how parser handles empty string
    ASSERT_NE(result, nullptr);
}

// Test ⎕EA with dfn string
TEST_F(ErrorTest, QuadEAWithDfnString) {
    Value* result = machine->eval("'{⍵+1}5' ⎕EA '{⍵÷0}1'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// Test multiple errors caught correctly
TEST_F(ErrorTest, MultipleErrorsCaught) {
    // First expression fails, second succeeds
    Value* r1 = machine->eval("'1' ⎕EA '1÷0'");
    EXPECT_DOUBLE_EQ(r1->as_scalar(), 1.0);

    // Try again with different values
    Value* r2 = machine->eval("'2' ⎕EA '2÷0'");
    EXPECT_DOUBLE_EQ(r2->as_scalar(), 2.0);
}

// Test INDEX ERROR has correct class (3)
TEST_F(ErrorTest, IndexErrorClass) {
    Value* result = machine->eval("'⎕ET' ⎕EA '(⍳3)[10]'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_array());
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 3.0);  // INDEX ERROR is class 3
}

// Test RANK ERROR has correct class (4)
TEST_F(ErrorTest, RankErrorClass) {
    Value* result = machine->eval("'⎕ET' ⎕EA '1 2 3 + 2 2⍴⍳4'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_array());
    // RANK ERROR is class 4, or LENGTH ERROR class 5
    double err_class = (*result->as_matrix())(0, 0);
    EXPECT_TRUE(err_class == 4.0 || err_class == 5.0);
}

// ============================================================================
// ISO 13751 Validation Tests
// ============================================================================

// Test ⎕ES requires rank ≤ 1 (ISO 13751 §11.5.7)
TEST_F(ErrorTest, QuadESMonadicRankError) {
    EXPECT_THROW(machine->eval("⎕ES 2 2⍴⍳4"), APLError);
}

// Test ⎕ES requires near-integer error codes (ISO 13751)
TEST_F(ErrorTest, QuadESMonadicNearIntegerError) {
    EXPECT_THROW(machine->eval("⎕ES 11.5 1.5"), APLError);
}

// Test ⎕ES dyadic left arg must be rank ≤ 1 (ISO 13751 §11.6.5)
TEST_F(ErrorTest, QuadESDyadicLeftRankError) {
    EXPECT_THROW(machine->eval("(2 2⍴'test') ⎕ES 11 1"), APLError);
}

// Test ⎕ES dyadic right arg must be rank ≤ 1 (ISO 13751 §11.6.5)
TEST_F(ErrorTest, QuadESDyadicRightRankError) {
    EXPECT_THROW(machine->eval("'msg' ⎕ES 2 2⍴⍳4"), APLError);
}

// Test ⎕ES dyadic requires near-integer error codes (ISO 13751)
TEST_F(ErrorTest, QuadESDyadicNearIntegerError) {
    EXPECT_THROW(machine->eval("'msg' ⎕ES 11.5 1.5"), APLError);
}

// Test ⎕EA left arg must be rank ≤ 1 (ISO 13751 §11.6.4)
TEST_F(ErrorTest, QuadEALeftRankError) {
    EXPECT_THROW(machine->eval("(2 2⍴'test') ⎕EA '1+2'"), APLError);
}

// Test ⎕EA right arg must be rank ≤ 1 (ISO 13751 §11.6.4)
TEST_F(ErrorTest, QuadEARightRankError) {
    EXPECT_THROW(machine->eval("'99' ⎕EA (2 2⍴'1+2 ')"), APLError);
}

// Test ⎕ET is cleared after ⎕EA handler completes (ISO 13751 context locality)
TEST_F(ErrorTest, QuadETClearedAfterHandler) {
    // First, catch an error
    machine->eval("'0' ⎕EA '1÷0'");
    // Now ⎕ET should be 0 0 (cleared after handler completed)
    Value* result = machine->eval("⎕ET");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_array());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*result->as_matrix())(1, 0), 0.0);
}

// Test ⎕EM is cleared after ⎕EA handler completes (ISO 13751 context locality)
TEST_F(ErrorTest, QuadEMClearedAfterHandler) {
    // First, catch an error
    machine->eval("'0' ⎕EA '1÷0'");
    // Now ⎕EM should be empty (cleared after handler completed)
    Value* result = machine->eval("⎕EM");
    ASSERT_NE(result, nullptr);
    if (result->is_string()) {
        EXPECT_STREQ(result->as_string(), "");
    } else if (result->is_array()) {
        EXPECT_EQ(result->size(), 0);
    }
}

// Main function for Google Test
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
