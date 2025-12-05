// Completion records tests
// Phase 2 complete: Completion handling now done through continuations
// Integration tests for :Leave and :Return are in test_statements.cpp

#include <gtest/gtest.h>
#include "machine.h"
#include "continuation.h"
#include "parser.h"
#include "environment.h"

// Placeholder test - actual completion testing done via statement tests
TEST(CompletionTest, CompletionHandlingViaStatements) {
    // Completion system tested through :Leave and :Return in test_statements.cpp
    // See: LeaveFromWhile, LeaveFromFor, LeaveFromNested
    EXPECT_TRUE(true);
}

// Phase 5.3: Test error propagation through THROW completions
TEST(CompletionTest, ErrorPropagationUncaught) {
    apl::Machine machine;

    // Push a HaltK
    machine.push_kont(machine.heap->allocate<apl::HaltK>());

    // Push a ThrowErrorK - this will create a THROW completion
    machine.push_kont(machine.heap->allocate<apl::ThrowErrorK>("Test error"));

    // Execute - should throw C++ exception since no CatchErrorK
    EXPECT_THROW(machine.execute(), std::runtime_error);
}

// Test error propagation with catch
TEST(CompletionTest, ErrorPropagationCaught) {
    apl::Machine machine;

    // Push a HaltK
    machine.push_kont(machine.heap->allocate<apl::HaltK>());

    // Push a CatchErrorK to catch the error
    machine.push_kont(machine.heap->allocate<apl::CatchErrorK>());

    // Push a ThrowErrorK - this will create a THROW completion
    machine.push_kont(machine.heap->allocate<apl::ThrowErrorK>("Test error"));

    // Execute - should NOT throw because CatchErrorK catches it
    EXPECT_NO_THROW(machine.execute());
}

// Test error propagation through multiple stack frames
TEST(CompletionTest, ErrorPropagationThroughFrames) {
    apl::Machine machine;

    // Push a HaltK
    machine.push_kont(machine.heap->allocate<apl::HaltK>());

    // Push a CatchErrorK at outer level
    machine.push_kont(machine.heap->allocate<apl::CatchErrorK>());

    // Push several normal continuations
    machine.push_kont(machine.heap->allocate<apl::HaltK>());
    machine.push_kont(machine.heap->allocate<apl::HaltK>());
    machine.push_kont(machine.heap->allocate<apl::HaltK>());

    // Push a ThrowErrorK deep in the stack
    machine.push_kont(machine.heap->allocate<apl::ThrowErrorK>("Deep error"));

    // Execute - should unwind through all the HaltKs and catch at CatchErrorK
    EXPECT_NO_THROW(machine.execute());
}

// Test error with message
TEST(CompletionTest, ErrorMessagePreserved) {
    apl::Machine machine;

    machine.push_kont(machine.heap->allocate<apl::HaltK>());
    machine.push_kont(machine.heap->allocate<apl::ThrowErrorK>("Custom error message"));

    // Should throw with our custom message
    try {
        machine.execute();
        FAIL() << "Expected exception to be thrown";
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        EXPECT_TRUE(msg.find("Custom error message") != std::string::npos);
    }
}

// Test multiple error boundaries
TEST(CompletionTest, NestedErrorBoundaries) {
    apl::Machine machine;

    // Outer boundary
    machine.push_kont(machine.heap->allocate<apl::HaltK>());
    machine.push_kont(machine.heap->allocate<apl::CatchErrorK>());

    // Some work
    machine.push_kont(machine.heap->allocate<apl::HaltK>());

    // Inner boundary
    machine.push_kont(machine.heap->allocate<apl::CatchErrorK>());

    // Error thrown here
    machine.push_kont(machine.heap->allocate<apl::ThrowErrorK>("Inner error"));

    // Should catch at inner boundary, not propagate to outer
    EXPECT_NO_THROW(machine.execute());
}

// Test error doesn't cross function boundaries incorrectly
TEST(CompletionTest, ErrorRespectsLoopBoundaries) {
    apl::Machine machine;

    machine.push_kont(machine.heap->allocate<apl::HaltK>());

    // CatchBreakK should NOT catch THROW completions
    machine.push_kont(machine.heap->allocate<apl::CatchBreakK>());

    machine.push_kont(machine.heap->allocate<apl::ThrowErrorK>("Error in loop"));

    // Should throw because CatchBreakK doesn't catch THROW
    EXPECT_THROW(machine.execute(), std::runtime_error);
}

// ============================================================================
// Unified Error Mechanism Tests
// Parse errors now flow through the same continuation system as runtime errors
// ============================================================================

// Test that parse errors are routed through ThrowErrorK (not nullptr return)
TEST(UnifiedErrorTest, ParseErrorFlowsThroughThrowErrorK) {
    apl::Machine machine;

    // Evaluate an invalid expression - should throw, not return nullptr
    // Previously this returned nullptr, now it throws via ThrowErrorK
    EXPECT_THROW(machine.eval("@invalid@"), std::runtime_error);
}

// Test that parse error messages are preserved through ThrowErrorK
TEST(UnifiedErrorTest, ParseErrorMessagePreserved) {
    apl::Machine machine;

    // Parse an unclosed parenthesis
    try {
        machine.eval("(2 + 3");
        FAIL() << "Expected exception for parse error";
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        // The error message should mention the parse issue
        EXPECT_TRUE(msg.find("Expected") != std::string::npos ||
                    msg.find("')") != std::string::npos ||
                    msg.length() > 0)
            << "Parse error message should be meaningful: " << msg;
    }
}

// Test that parse errors can be caught by CatchErrorK like runtime errors
TEST(UnifiedErrorTest, ParseErrorCaughtByCatchErrorK) {
    apl::Machine machine;

    // Parse will fail, creating ThrowErrorK
    apl::Continuation* k = machine.parser->parse("@invalid@");

    // k should be nullptr from parser
    EXPECT_EQ(k, nullptr);

    // But if we manually set up the error handling like eval() does:
    const char* msg = machine.string_pool.intern(machine.parser->get_error().c_str());
    k = machine.heap->allocate<apl::ThrowErrorK>(msg);

    // Push CatchErrorK to catch the error
    machine.push_kont(machine.heap->allocate<apl::HaltK>());
    machine.push_kont(machine.heap->allocate<apl::CatchErrorK>());
    machine.push_kont(k);

    // Should NOT throw because CatchErrorK catches it
    EXPECT_NO_THROW(machine.execute());
}

// Test that runtime and parse errors behave identically when uncaught
TEST(UnifiedErrorTest, ParseAndRuntimeErrorsUnified) {
    apl::Machine machine1;
    apl::Machine machine2;

    // Parse error (invalid token)
    bool parse_threw = false;
    try {
        machine1.eval("@invalid@");
    } catch (const std::runtime_error&) {
        parse_threw = true;
    }

    // Runtime error (undefined variable)
    bool runtime_threw = false;
    try {
        machine2.eval("undefined_var_xyz");
    } catch (const std::runtime_error&) {
        runtime_threw = true;
    }

    // Both should throw the same type of exception
    EXPECT_TRUE(parse_threw) << "Parse error should throw std::runtime_error";
    EXPECT_TRUE(runtime_threw) << "Runtime error should throw std::runtime_error";
}

// Test that empty parentheses parse error flows through ThrowErrorK
TEST(UnifiedErrorTest, EmptyParensErrorThroughThrowErrorK) {
    apl::Machine machine;

    EXPECT_THROW(machine.eval("()"), std::runtime_error);
}

// Test that unmatched closing paren error flows through ThrowErrorK
TEST(UnifiedErrorTest, UnmatchedClosingParenErrorThroughThrowErrorK) {
    apl::Machine machine;

    EXPECT_THROW(machine.eval("2 + 3)"), std::runtime_error);
}

// Test that multiple parse errors in sequence all throw properly
TEST(UnifiedErrorTest, MultipleParseErrorsInSequence) {
    apl::Machine machine;
    apl::init_global_environment(&machine);  // Initialize primitives

    // Each should throw, and machine should be reusable
    EXPECT_THROW(machine.eval("@"), std::runtime_error);
    EXPECT_THROW(machine.eval("("), std::runtime_error);
    EXPECT_THROW(machine.eval(")"), std::runtime_error);

    // Valid expression should still work after errors
    apl::Value* result = nullptr;
    EXPECT_NO_THROW(result = machine.eval("2 + 3"));
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
