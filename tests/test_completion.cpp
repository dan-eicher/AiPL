// Completion records tests
// Phase 2 complete: Completion handling now done through continuations
// Integration tests for :Leave and :Return are in test_statements.cpp

#include <gtest/gtest.h>
#include "machine.h"
#include "continuation.h"

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

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
