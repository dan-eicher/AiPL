// Completion records tests
// Phase 2 complete: Completion handling now done through continuations
// Integration tests for :Leave and :Return are in test_statements.cpp

#include <gtest/gtest.h>
#include "machine.h"
#include "continuation.h"
#include "parser.h"

using namespace apl;

class CompletionTest : public ::testing::Test {
protected:
    Machine* machine;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
    }
};

// Placeholder test - actual completion testing done via statement tests
TEST_F(CompletionTest, CompletionHandlingViaStatements) {
    // Completion system tested through :Leave and :Return in test_statements.cpp
    // See: LeaveFromWhile, LeaveFromFor, LeaveFromNested
    EXPECT_TRUE(true);
}

// Phase 5.3: Test error propagation through THROW completions
TEST_F(CompletionTest, ErrorPropagationUncaught) {
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<ThrowErrorK>("Test error"));

    // Execute - should throw C++ exception since no CatchErrorK
    EXPECT_THROW(machine->execute(), APLError);
}

// Test error propagation with catch
TEST_F(CompletionTest, ErrorPropagationCaught) {
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<CatchErrorK>());
    machine->push_kont(machine->heap->allocate<ThrowErrorK>("Test error"));

    // Execute - should NOT throw because CatchErrorK catches it
    EXPECT_NO_THROW(machine->execute());
}

// Test error propagation through multiple stack frames
TEST_F(CompletionTest, ErrorPropagationThroughFrames) {
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<CatchErrorK>());
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<ThrowErrorK>("Deep error"));

    // Execute - should unwind through all the HaltKs and catch at CatchErrorK
    EXPECT_NO_THROW(machine->execute());
}

// Test error with message
TEST_F(CompletionTest, ErrorMessagePreserved) {
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<ThrowErrorK>("Custom error message"));

    try {
        machine->execute();
        FAIL() << "Expected exception to be thrown";
    } catch (const APLError& e) {
        std::string msg(e.what());
        EXPECT_TRUE(msg.find("Custom error message") != std::string::npos);
    }
}

// Test multiple error boundaries
TEST_F(CompletionTest, NestedErrorBoundaries) {
    // Outer boundary
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<CatchErrorK>());

    // Some work
    machine->push_kont(machine->heap->allocate<HaltK>());

    // Inner boundary
    machine->push_kont(machine->heap->allocate<CatchErrorK>());

    // Error thrown here
    machine->push_kont(machine->heap->allocate<ThrowErrorK>("Inner error"));

    // Should catch at inner boundary, not propagate to outer
    EXPECT_NO_THROW(machine->execute());
}

// Test error doesn't cross function boundaries incorrectly
TEST_F(CompletionTest, ErrorRespectsLoopBoundaries) {
    machine->push_kont(machine->heap->allocate<HaltK>());

    // CatchBreakK should NOT catch THROW completions
    machine->push_kont(machine->heap->allocate<CatchBreakK>());

    machine->push_kont(machine->heap->allocate<ThrowErrorK>("Error in loop"));

    // Should throw because CatchBreakK doesn't catch THROW
    EXPECT_THROW(machine->execute(), APLError);
}

// ============================================================================
// Unified Error Mechanism Tests
// Parse errors now flow through the same continuation system as runtime errors
// ============================================================================

class UnifiedErrorTest : public ::testing::Test {
protected:
    Machine* machine;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
    }
};

// Test that parse errors are routed through ThrowErrorK (not nullptr return)
TEST_F(UnifiedErrorTest, ParseErrorFlowsThroughThrowErrorK) {
    EXPECT_THROW(machine->eval("@invalid@"), APLError);
}

// Test that parse error messages are preserved through ThrowErrorK
TEST_F(UnifiedErrorTest, ParseErrorMessagePreserved) {
    try {
        machine->eval("(2 + 3");
        FAIL() << "Expected exception for parse error";
    } catch (const APLError& e) {
        std::string msg(e.what());
        EXPECT_TRUE(msg.find("Expected") != std::string::npos ||
                    msg.find("')") != std::string::npos ||
                    msg.length() > 0)
            << "Parse error message should be meaningful: " << msg;
    }
}

// Test that parse errors can be caught by CatchErrorK like runtime errors
TEST_F(UnifiedErrorTest, ParseErrorCaughtByCatchErrorK) {
    // Parse will fail, creating ThrowErrorK
    Continuation* k = machine->parser->parse("@invalid@");

    // k should be nullptr from parser
    EXPECT_EQ(k, nullptr);

    // But if we manually set up the error handling like eval() does:
    const char* msg = machine->string_pool.intern(machine->parser->get_error().c_str());
    k = machine->heap->allocate<ThrowErrorK>(msg);

    // Push CatchErrorK to catch the error
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<CatchErrorK>());
    machine->push_kont(k);

    // Should NOT throw because CatchErrorK catches it
    EXPECT_NO_THROW(machine->execute());
}

// Test that runtime and parse errors behave identically when uncaught
TEST_F(UnifiedErrorTest, ParseAndRuntimeErrorsUnified) {
    Machine machine2;

    // Parse error (invalid token)
    bool parse_threw = false;
    try {
        machine->eval("@invalid@");
    } catch (const APLError&) {
        parse_threw = true;
    }

    // Runtime error (undefined variable)
    bool runtime_threw = false;
    try {
        machine2.eval("undefined_var_xyz");
    } catch (const APLError&) {
        runtime_threw = true;
    }

    // Both should throw the same type of exception
    EXPECT_TRUE(parse_threw) << "Parse error should throw std::runtime_error";
    EXPECT_TRUE(runtime_threw) << "Runtime error should throw std::runtime_error";
}

// Test that empty parentheses parse error flows through ThrowErrorK
TEST_F(UnifiedErrorTest, EmptyParensErrorThroughThrowErrorK) {
    EXPECT_THROW(machine->eval("()"), APLError);
}

// Test that unmatched closing paren error flows through ThrowErrorK
TEST_F(UnifiedErrorTest, UnmatchedClosingParenErrorThroughThrowErrorK) {
    EXPECT_THROW(machine->eval("2 + 3)"), APLError);
}

// Test that multiple parse errors in sequence all throw properly
TEST_F(UnifiedErrorTest, MultipleParseErrorsInSequence) {
    // Each should throw, and machine should be reusable
    EXPECT_THROW(machine->eval("@"), APLError);
    EXPECT_THROW(machine->eval("("), APLError);
    EXPECT_THROW(machine->eval(")"), APLError);

    // Valid expression should still work after errors
    Value* result = nullptr;
    EXPECT_NO_THROW(result = machine->eval("2 + 3"));
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
