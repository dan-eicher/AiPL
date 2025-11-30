// Continuation tests

#include <gtest/gtest.h>
#include "continuation.h"
#include "heap.h"
#include "machine.h"
#include "value.h"

using namespace apl;

class ContinuationTest : public ::testing::Test {
protected:
    Machine* machine;
    APLHeap* heap;  // Convenience pointer to machine->heap

    void SetUp() override {
        machine = new Machine();
        heap = machine->heap;  // Use machine's heap
        machine->ctrl.init_evaluating();
    }

    void TearDown() override {
        delete machine;
    }
};

// Test HaltK
TEST_F(ContinuationTest, HaltK) {
    HaltK* halt = heap->allocate<HaltK>();

    Value* v = machine->heap->allocate_scalar(42.0);
    machine->ctrl.set_value(v);

    machine->push_kont(halt);
    Value* result = machine->execute();

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    // GC will clean up halt
    // GC will clean up v
}

// Test HaltK mark (should do nothing)
TEST_F(ContinuationTest, HaltKMark) {
    HaltK* halt = heap->allocate<HaltK>();

    // Should not crash
    halt->mark(nullptr);

    // GC will clean up halt
}

// Test HaltK is not a boundary
TEST_F(ContinuationTest, HaltKNotBoundary) {
    HaltK* halt = heap->allocate<HaltK>();

    EXPECT_FALSE(halt->is_function_boundary());
    EXPECT_FALSE(halt->is_loop_boundary());
    EXPECT_FALSE(halt->matches_label("any"));

    // GC will clean up halt
}

// Test ArgK basic
TEST_F(ContinuationTest, ArgKBasic) {
    Value* arg = machine->heap->allocate_scalar(10.0);
    HaltK* halt = heap->allocate<HaltK>();
    ArgK* argk = heap->allocate<ArgK>(arg, halt);

    machine->push_kont(argk);
    Value* result = machine->execute();

    EXPECT_EQ(result, arg);
    EXPECT_EQ(machine->ctrl.value, arg);

    // GC will clean up -     delete argk;  // Will also delete halt
    // GC will clean up -     delete arg;
}

// Test ArgK without next
TEST_F(ContinuationTest, ArgKWithoutNext) {
    Value* arg = machine->heap->allocate_scalar(99.0);
    ArgK* argk = heap->allocate<ArgK>(arg, nullptr);

    machine->push_kont(argk);
    Value* result = machine->execute();

    EXPECT_EQ(result, arg);

    // GC will clean up argk
    // GC will clean up -     delete arg;
}

// Test ArgK chaining
TEST_F(ContinuationTest, ArgKChaining) {
    Value* arg1 = machine->heap->allocate_scalar(1.0);
    Value* arg2 = machine->heap->allocate_scalar(2.0);

    HaltK* halt = heap->allocate<HaltK>();
    ArgK* argk2 = heap->allocate<ArgK>(arg2, halt);
    ArgK* argk1 = heap->allocate<ArgK>(arg1, argk2);

    // Push first ArgK and execute via trampoline
    machine->push_kont(argk1);
    Value* result = machine->execute();

    // Should eventually return arg2 (last in chain)
    EXPECT_EQ(result, arg2);
    EXPECT_EQ(machine->ctrl.value, arg2);

    // GC will clean up -     delete argk1;  // Will cascade delete
    // GC will clean up -     delete arg1;
    // GC will clean up -     delete arg2;
}

// Test ArgK mark
TEST_F(ContinuationTest, ArgKMark) {
    Value* arg = machine->heap->allocate_scalar(5.0);
    ArgK* argk = heap->allocate<ArgK>(arg, nullptr);

    // Should not crash with nullptr heap
    argk->mark(nullptr);

    // GC will clean up argk
    // GC will clean up -     delete arg;
}

// Test ArgK is not a boundary
TEST_F(ContinuationTest, ArgKNotBoundary) {
    Value* arg = machine->heap->allocate_scalar(1.0);
    ArgK* argk = heap->allocate<ArgK>(arg, nullptr);

    EXPECT_FALSE(argk->is_function_boundary());
    EXPECT_FALSE(argk->is_loop_boundary());
    EXPECT_FALSE(argk->matches_label("label"));

    // GC will clean up argk
    // GC will clean up -     delete arg;
}

// Test FrameK basic
TEST_F(ContinuationTest, FrameKBasic) {
    HaltK* halt = heap->allocate<HaltK>();
    FrameK* frame = heap->allocate<FrameK>("test_func", halt);

    Value* v = machine->heap->allocate_scalar(7.0);
    machine->ctrl.set_value(v);

    // Push frame onto stack and execute via trampoline
    machine->push_kont(frame);
    Value* result = machine->execute();

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    // GC will clean up frame and halt
    // GC will clean up v
}

// Test FrameK is function boundary
TEST_F(ContinuationTest, FrameKIsFunctionBoundary) {
    FrameK* frame = heap->allocate<FrameK>("my_func", nullptr);

    EXPECT_TRUE(frame->is_function_boundary());
    EXPECT_FALSE(frame->is_loop_boundary());
    EXPECT_FALSE(frame->matches_label("label"));

    // GC will clean up frame
}

// Test FrameK without return continuation
TEST_F(ContinuationTest, FrameKWithoutReturn) {
    FrameK* frame = heap->allocate<FrameK>("func", nullptr);

    Value* v = machine->heap->allocate_scalar(88.0);
    machine->ctrl.set_value(v);

    // Push frame onto stack and execute via trampoline
    machine->push_kont(frame);
    Value* result = machine->execute();

    EXPECT_EQ(result, v);

    // GC will clean up frame
    // GC will clean up v
}

// Test FrameK mark
TEST_F(ContinuationTest, FrameKMark) {
    HaltK* halt = heap->allocate<HaltK>();
    FrameK* frame = heap->allocate<FrameK>("test", halt);

    // Should not crash
    frame->mark(nullptr);

    // GC will clean up frame
}

// Test FrameK function name
TEST_F(ContinuationTest, FrameKFunctionName) {
    const char* name = "my_function";
    FrameK* frame = heap->allocate<FrameK>(name, nullptr);

    EXPECT_EQ(frame->function_name, name);
    EXPECT_STREQ(frame->function_name, "my_function");

    // GC will clean up frame
}

// Test Machine push and execute
TEST_F(ContinuationTest, MachinePushPop) {
    Value* v = machine->heap->allocate_scalar(42.0);
    machine->ctrl.set_value(v);

    HaltK* halt = heap->allocate<HaltK>();
    machine->push_kont(halt);

    EXPECT_EQ(machine->kont_stack.size(), 1);

    Value* result = machine->execute();

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->kont_stack.size(), 0);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    // GC will clean up v
}

// Test Machine execute with empty stack
TEST_F(ContinuationTest, MachinePopEmpty) {
    Value* v = machine->heap->allocate_scalar(123.0);
    machine->ctrl.set_value(v);

    EXPECT_EQ(machine->kont_stack.size(), 0);

    Value* result = machine->execute();

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    // GC will clean up v
}

// Test Machine with multiple continuations
TEST_F(ContinuationTest, MachineMultipleContinuations) {
    Value* arg1 = machine->heap->allocate_scalar(10.0);
    Value* arg2 = machine->heap->allocate_scalar(20.0);

    HaltK* halt = heap->allocate<HaltK>();
    ArgK* argk2 = heap->allocate<ArgK>(arg2, halt);
    ArgK* argk1 = heap->allocate<ArgK>(arg1, argk2);

    machine->push_kont(argk1);

    EXPECT_EQ(machine->kont_stack.size(), 1);

    Value* result = machine->execute();

    EXPECT_EQ(result, arg2);  // Last arg in chain

    // GC will clean up -     delete arg1;
    // GC will clean up -     delete arg2;
}

// Test continuation chaining with FrameK
TEST_F(ContinuationTest, FrameKChaining) {
    Value* v = machine->heap->allocate_scalar(99.0);

    HaltK* halt = heap->allocate<HaltK>();
    FrameK* frame = heap->allocate<FrameK>("outer", halt);
    ArgK* argk = heap->allocate<ArgK>(v, frame);

    machine->ctrl.set_value(v);

    machine->push_kont(argk);
    Value* result = machine->execute();

    EXPECT_EQ(result, v);

    // GC will clean up argk
    // GC will clean up v
}

// ============================================================================
// LiteralK Tests - Parse-time safe continuation
// ============================================================================

// Test LiteralK basic invoke
TEST_F(ContinuationTest, LiteralKBasic) {
    auto lit = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(42.0));

    // Push onto stack and execute via trampoline
    machine->push_kont(lit);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
    EXPECT_EQ(machine->ctrl.value, result);
}

// Test LiteralK without next - now just tests basic execution
TEST_F(ContinuationTest, LiteralKWithoutNext) {
    auto lit = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(3.14));

    machine->push_kont(lit);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.14);
}

// Test LiteralK chaining via trampoline
TEST_F(ContinuationTest, LiteralKChaining) {
    auto lit1 = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(10.0));
    auto lit2 = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(20.0));

    // Push in reverse order (stack is LIFO)
    machine->push_kont(lit2);
    machine->push_kont(lit1);

    Value* result = machine->execute();

    // Last continuation wins - should have 20.0
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

// Test LiteralK uses scalar cache
TEST_F(ContinuationTest, LiteralKUsesScalarCache) {
    auto lit1 = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(42.0));
    auto lit2 = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(42.0));

    machine->push_kont(lit1);
    Value* result1 = machine->execute();

    machine->ctrl.init_evaluating();
    machine->push_kont(lit2);
    Value* result2 = machine->execute();

    EXPECT_EQ(result1, result2);
    EXPECT_DOUBLE_EQ(result1->as_scalar(), 42.0);
}

// Test LiteralK parse-time safety - stores double not Value*
TEST_F(ContinuationTest, LiteralKParseTimeSafety) {
    LiteralK* lit = heap->allocate<LiteralK>(123.0);

    EXPECT_DOUBLE_EQ(lit->literal_value, 123.0);

    // lit is already heap-allocated, no need to register again

    size_t values_before = machine->heap->total_size();
    machine->push_kont(lit);
    Value* result = machine->execute();
    size_t values_after = machine->heap->total_size();

    // Should have allocated a new Value for the literal
    EXPECT_GT(values_after, values_before);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 123.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
