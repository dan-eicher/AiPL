// Continuation tests

#include <gtest/gtest.h>
#include "continuation.h"
#include "machine.h"
#include "value.h"

using namespace apl;

class ContinuationTest : public ::testing::Test {
protected:
    Machine* machine;

    void SetUp() override {
        machine = new Machine();
        machine->ctrl.init_evaluating();
    }

    void TearDown() override {
        delete machine;
    }
};

// Test HaltK
TEST_F(ContinuationTest, HaltK) {
    HaltK* halt = new HaltK();

    Value* v = Value::from_scalar(42.0);
    machine->ctrl.set_value(v);

    Value* result = halt->invoke(machine);

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    delete halt;
    delete v;
}

// Test HaltK mark (should do nothing)
TEST_F(ContinuationTest, HaltKMark) {
    HaltK* halt = new HaltK();

    // Should not crash
    halt->mark(nullptr);

    delete halt;
}

// Test HaltK is not a boundary
TEST_F(ContinuationTest, HaltKNotBoundary) {
    HaltK* halt = new HaltK();

    EXPECT_FALSE(halt->is_function_boundary());
    EXPECT_FALSE(halt->is_loop_boundary());
    EXPECT_FALSE(halt->matches_label("any"));

    delete halt;
}

// Test ArgK basic
TEST_F(ContinuationTest, ArgKBasic) {
    Value* arg = Value::from_scalar(10.0);
    HaltK* halt = new HaltK();
    ArgK* argk = new ArgK(arg, halt);

    Value* result = argk->invoke(machine);

    EXPECT_EQ(result, arg);
    EXPECT_EQ(machine->ctrl.value, arg);

    delete argk;  // Will also delete halt
    delete arg;
}

// Test ArgK without next
TEST_F(ContinuationTest, ArgKWithoutNext) {
    Value* arg = Value::from_scalar(99.0);
    ArgK* argk = new ArgK(arg, nullptr);

    Value* result = argk->invoke(machine);

    EXPECT_EQ(result, arg);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    delete argk;
    delete arg;
}

// Test ArgK chaining
TEST_F(ContinuationTest, ArgKChaining) {
    Value* arg1 = Value::from_scalar(1.0);
    Value* arg2 = Value::from_scalar(2.0);

    HaltK* halt = new HaltK();
    ArgK* argk2 = new ArgK(arg2, halt);
    ArgK* argk1 = new ArgK(arg1, argk2);

    // Invoke first ArgK
    Value* result = argk1->invoke(machine);

    // Should eventually return arg2 (last in chain)
    EXPECT_EQ(result, arg2);
    EXPECT_EQ(machine->ctrl.value, arg2);

    delete argk1;  // Will cascade delete
    delete arg1;
    delete arg2;
}

// Test ArgK mark
TEST_F(ContinuationTest, ArgKMark) {
    Value* arg = Value::from_scalar(5.0);
    ArgK* argk = new ArgK(arg, nullptr);

    // Should not crash with nullptr heap
    argk->mark(nullptr);

    delete argk;
    delete arg;
}

// Test ArgK is not a boundary
TEST_F(ContinuationTest, ArgKNotBoundary) {
    Value* arg = Value::from_scalar(1.0);
    ArgK* argk = new ArgK(arg, nullptr);

    EXPECT_FALSE(argk->is_function_boundary());
    EXPECT_FALSE(argk->is_loop_boundary());
    EXPECT_FALSE(argk->matches_label("label"));

    delete argk;
    delete arg;
}

// Test FrameK basic
TEST_F(ContinuationTest, FrameKBasic) {
    HaltK* halt = new HaltK();
    FrameK* frame = new FrameK("test_func", halt);

    Value* v = Value::from_scalar(7.0);
    machine->ctrl.set_value(v);

    Value* result = frame->invoke(machine);

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    delete frame;  // Will also delete halt
    delete v;
}

// Test FrameK is function boundary
TEST_F(ContinuationTest, FrameKIsFunctionBoundary) {
    FrameK* frame = new FrameK("my_func", nullptr);

    EXPECT_TRUE(frame->is_function_boundary());
    EXPECT_FALSE(frame->is_loop_boundary());
    EXPECT_FALSE(frame->matches_label("label"));

    delete frame;
}

// Test FrameK without return continuation
TEST_F(ContinuationTest, FrameKWithoutReturn) {
    FrameK* frame = new FrameK("func", nullptr);

    Value* v = Value::from_scalar(88.0);
    machine->ctrl.set_value(v);

    Value* result = frame->invoke(machine);

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    delete frame;
    delete v;
}

// Test FrameK mark
TEST_F(ContinuationTest, FrameKMark) {
    HaltK* halt = new HaltK();
    FrameK* frame = new FrameK("test", halt);

    // Should not crash
    frame->mark(nullptr);

    delete frame;
}

// Test FrameK function name
TEST_F(ContinuationTest, FrameKFunctionName) {
    const char* name = "my_function";
    FrameK* frame = new FrameK(name, nullptr);

    EXPECT_EQ(frame->function_name, name);
    EXPECT_STREQ(frame->function_name, "my_function");

    delete frame;
}

// Test Machine push and pop
TEST_F(ContinuationTest, MachinePushPop) {
    Value* v = Value::from_scalar(42.0);
    machine->ctrl.set_value(v);

    HaltK* halt = new HaltK();
    machine->push_kont(halt);

    EXPECT_EQ(machine->kont_stack.size(), 1);

    Value* result = machine->pop_kont_and_invoke();

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->kont_stack.size(), 0);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    delete v;
}

// Test Machine pop on empty stack
TEST_F(ContinuationTest, MachinePopEmpty) {
    Value* v = Value::from_scalar(123.0);
    machine->ctrl.set_value(v);

    EXPECT_EQ(machine->kont_stack.size(), 0);

    Value* result = machine->pop_kont_and_invoke();

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    delete v;
}

// Test Machine with multiple continuations
TEST_F(ContinuationTest, MachineMultipleContinuations) {
    Value* arg1 = Value::from_scalar(10.0);
    Value* arg2 = Value::from_scalar(20.0);

    HaltK* halt = new HaltK();
    ArgK* argk2 = new ArgK(arg2, halt);
    ArgK* argk1 = new ArgK(arg1, argk2);

    machine->push_kont(argk1);

    EXPECT_EQ(machine->kont_stack.size(), 1);

    Value* result = machine->pop_kont_and_invoke();

    EXPECT_EQ(result, arg2);  // Last arg in chain
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    delete arg1;
    delete arg2;
}

// Test continuation chaining with FrameK
TEST_F(ContinuationTest, FrameKChaining) {
    Value* v = Value::from_scalar(99.0);

    HaltK* halt = new HaltK();
    FrameK* frame = new FrameK("outer", halt);
    ArgK* argk = new ArgK(v, frame);

    machine->ctrl.set_value(v);

    Value* result = argk->invoke(machine);

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);

    delete argk;
    delete v;
}

// ============================================================================
// LiteralK Tests - Parse-time safe continuation
// ============================================================================

// Test LiteralK basic invoke
TEST_F(ContinuationTest, LiteralKBasic) {
    auto halt = static_cast<HaltK*>(machine->heap->allocate_continuation(new HaltK()));
    auto lit = static_cast<LiteralK*>(machine->heap->allocate_continuation(new LiteralK(42.0, halt)));

    Value* result = lit->invoke(machine);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
    EXPECT_EQ(machine->ctrl.value, result);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);
}

// Test LiteralK without next
TEST_F(ContinuationTest, LiteralKWithoutNext) {
    auto lit = static_cast<LiteralK*>(machine->heap->allocate_continuation(new LiteralK(3.14, nullptr)));

    Value* result = lit->invoke(machine);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.14);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);
}

// Test LiteralK chaining
TEST_F(ContinuationTest, LiteralKChaining) {
    auto halt = static_cast<HaltK*>(machine->heap->allocate_continuation(new HaltK()));
    auto lit2 = static_cast<LiteralK*>(machine->heap->allocate_continuation(new LiteralK(20.0, halt)));
    auto lit1 = static_cast<LiteralK*>(machine->heap->allocate_continuation(new LiteralK(10.0, lit2)));

    Value* result = lit1->invoke(machine);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);
}

// Test LiteralK uses scalar cache
TEST_F(ContinuationTest, LiteralKUsesScalarCache) {
    auto halt1 = static_cast<HaltK*>(machine->heap->allocate_continuation(new HaltK()));
    auto halt2 = static_cast<HaltK*>(machine->heap->allocate_continuation(new HaltK()));

    auto lit1 = static_cast<LiteralK*>(machine->heap->allocate_continuation(new LiteralK(42.0, halt1)));
    auto lit2 = static_cast<LiteralK*>(machine->heap->allocate_continuation(new LiteralK(42.0, halt2)));

    machine->ctrl.init_evaluating();
    Value* result1 = lit1->invoke(machine);

    machine->ctrl.init_evaluating();
    Value* result2 = lit2->invoke(machine);

    EXPECT_EQ(result1, result2);
    EXPECT_DOUBLE_EQ(result1->as_scalar(), 42.0);
}

// Test LiteralK parse-time safety
TEST_F(ContinuationTest, LiteralKParseTimeSafety) {
    HaltK* halt = new HaltK();
    LiteralK* lit = new LiteralK(123.0, halt);

    EXPECT_DOUBLE_EQ(lit->literal_value, 123.0);
    EXPECT_EQ(lit->next, halt);

    machine->heap->allocate_continuation(halt);
    machine->heap->allocate_continuation(lit);

    size_t values_before = machine->heap->total_size();
    Value* result = lit->invoke(machine);
    size_t values_after = machine->heap->total_size();

    EXPECT_GT(values_after, values_before);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 123.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
