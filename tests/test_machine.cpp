// Machine execution tests

#include <gtest/gtest.h>
#include "machine.h"
#include "continuation.h"
#include "completion.h"
#include "value.h"

using namespace apl;

class MachineTest : public ::testing::Test {
protected:
    Machine* machine;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
    }
};

// Test machine initialization
TEST_F(MachineTest, Initialization) {
    EXPECT_NE(machine->heap, nullptr);
    EXPECT_NE(machine->env, nullptr);
    EXPECT_EQ(machine->kont_stack.size(), 0);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);
}

// Test halt
TEST_F(MachineTest, Halt) {
    machine->halt();
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);
    EXPECT_FALSE(machine->should_continue());
}

// Test should_continue
TEST_F(MachineTest, ShouldContinue) {
    // Initially halted with no completion
    EXPECT_FALSE(machine->should_continue());

    // Set to evaluating with normal completion
    machine->ctrl.init_evaluating();
    EXPECT_TRUE(machine->should_continue());

    // Set to halted
    machine->ctrl.mode = ExecMode::HALTED;
    EXPECT_FALSE(machine->should_continue());
}

// Test push and pop continuation
TEST_F(MachineTest, PushPopKont) {
    Continuation* k1 = machine->heap->allocate<HaltK>();
    Continuation* k2 = machine->heap->allocate<HaltK>();

    machine->push_kont(k1);
    EXPECT_EQ(machine->kont_stack.size(), 1);

    machine->push_kont(k2);
    EXPECT_EQ(machine->kont_stack.size(), 2);

    Continuation* popped = machine->pop_kont();
    EXPECT_EQ(popped, k2);
    EXPECT_EQ(machine->kont_stack.size(), 1);

    popped = machine->pop_kont();
    EXPECT_EQ(popped, k1);
    EXPECT_EQ(machine->kont_stack.size(), 0);

    // GC will clean up -     delete k1;
    // GC will clean up -     delete k2;
}

// Test pop from empty stack
TEST_F(MachineTest, PopEmptyStack) {
    Continuation* k = machine->pop_kont();
    EXPECT_EQ(k, nullptr);
}

// Test execute with HaltK
TEST_F(MachineTest, ExecuteWithHaltK) {
    Value* v = machine->heap->allocate_scalar(42.0);
    machine->ctrl.value = v;
    machine->push_kont(machine->heap->allocate<HaltK>());

    Value* result = machine->execute();

    EXPECT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);
}

// Test execute with empty stack
TEST_F(MachineTest, ExecuteEmptyStack) {
    Value* v = machine->heap->allocate_scalar(100.0);
    machine->ctrl.mode = ExecMode::EVALUATING;
    machine->ctrl.value = v;

    Value* result = machine->execute();

    // With empty stack, should halt and return the value
    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->ctrl.mode, ExecMode::HALTED);
}

// Test environment variable lookup
TEST_F(MachineTest, EnvironmentLookup) {
    Value* v = machine->heap->allocate_scalar(5.0);
    machine->env->define("x", v);

    Value* result = machine->env->lookup("x");
    EXPECT_EQ(result, v);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test environment lookup failure
TEST_F(MachineTest, EnvironmentLookupFailure) {
    Value* result = machine->env->lookup("nonexistent");
    EXPECT_EQ(result, nullptr);
}

// Test environment update
TEST_F(MachineTest, EnvironmentUpdate) {
    Value* v1 = machine->heap->allocate_scalar(10.0);
    machine->env->define("y", v1);

    Value* v2 = machine->heap->allocate_scalar(20.0);
    machine->env->define("y", v2);

    Value* result = machine->env->lookup("y");
    EXPECT_EQ(result, v2);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

// Test nested environment
TEST_F(MachineTest, NestedEnvironment) {
    Value* v1 = machine->heap->allocate_scalar(1.0);
    machine->env->define("a", v1);

    Environment* child = machine->heap->allocate<Environment>(machine->env);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    child->define("b", v2);

    // Child can see parent's binding
    Value* result = child->lookup("a");
    EXPECT_EQ(result, v1);

    // Child can see its own binding
    result = child->lookup("b");
    EXPECT_EQ(result, v2);

    // Parent cannot see child's binding
    result = machine->env->lookup("b");
    EXPECT_EQ(result, nullptr);

    // GC will clean up -     delete child;
}

// Test nested environment shadowing
TEST_F(MachineTest, NestedEnvironmentShadowing) {
    Value* v1 = machine->heap->allocate_scalar(10.0);
    machine->env->define("x", v1);

    Environment* child = machine->heap->allocate<Environment>(machine->env);
    Value* v2 = machine->heap->allocate_scalar(20.0);
    child->define("x", v2);

    // Child sees its own binding, not parent's
    Value* result = child->lookup("x");
    EXPECT_EQ(result, v2);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);

    // Parent still has original binding
    result = machine->env->lookup("x");
    EXPECT_EQ(result, v1);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);

    // GC will clean up -     delete child;
}

// Test normal completion handling
TEST_F(MachineTest, NormalCompletion) {
    Value* v = machine->heap->allocate_scalar(42.0);
    machine->ctrl.completion = nullptr;  // nullptr = NORMAL
    machine->ctrl.value = v;

    machine->handle_completion();

    EXPECT_EQ(machine->ctrl.value, v);
    EXPECT_EQ(machine->ctrl.completion, nullptr);
}

// Test return completion with FrameK
TEST_F(MachineTest, ReturnCompletion) {
    Value* v = machine->heap->allocate_scalar(99.0);

    // Push a FrameK (function boundary)
    machine->push_kont(machine->heap->allocate<FrameK>("test_func", nullptr));
    machine->push_kont(machine->heap->allocate<HaltK>());  // Some other continuation

    machine->ctrl.completion = machine->heap->allocate<APLCompletion>(CompletionType::RETURN, v, nullptr);

    machine->handle_completion();

    EXPECT_EQ(machine->ctrl.value, v);
    EXPECT_EQ(machine->ctrl.completion, nullptr);
    // Should have unwound to the FrameK
    EXPECT_EQ(machine->kont_stack.size(), 1);
}

// Test return outside of function
TEST_F(MachineTest, ReturnOutsideFunction) {
    Value* v = machine->heap->allocate_scalar(1.0);
    machine->ctrl.completion = machine->heap->allocate<APLCompletion>(CompletionType::RETURN, v, nullptr);

    EXPECT_THROW(machine->handle_completion(), std::runtime_error);
}

// Test function cache empty
TEST_F(MachineTest, FunctionCacheEmpty) {
    EXPECT_EQ(machine->function_cache.size(), 0);
}

// Test function cache insertion
TEST_F(MachineTest, FunctionCacheInsertion) {
    Continuation* k = machine->heap->allocate<HaltK>();
    machine->function_cache["test_func"] = k;

    EXPECT_EQ(machine->function_cache.size(), 1);
    EXPECT_EQ(machine->function_cache["test_func"], k);
}

// Test GC integration with environment
TEST_F(MachineTest, GCWithEnvironment) {
    // Allocate some values and store in environment
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    Value* v3 = machine->heap->allocate_scalar(3.0);

    machine->env->define("a", v1);
    machine->env->define("b", v2);
    // v3 is not stored anywhere

    // Force a GC
    machine->heap->collect(machine);

    // v1 and v2 should be marked (reachable from environment)
    EXPECT_TRUE(v1->marked);
    EXPECT_TRUE(v2->marked);
    // v3 might be collected or unmarked depending on when we check
}

// Test multiple continuations on stack
TEST_F(MachineTest, MultipleContinuations) {
    Continuation* k1 = machine->heap->allocate<HaltK>();
    Continuation* k2 = machine->heap->allocate<FrameK>("func", nullptr);
    Continuation* k3 = machine->heap->allocate<HaltK>();

    machine->push_kont(k1);
    machine->push_kont(k2);
    machine->push_kont(k3);

    EXPECT_EQ(machine->kont_stack.size(), 3);
    EXPECT_EQ(machine->kont_stack[0], k1);
    EXPECT_EQ(machine->kont_stack[1], k2);
    EXPECT_EQ(machine->kont_stack[2], k3);
}

// Test unwind to boundary
TEST_F(MachineTest, UnwindToBoundary) {
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<FrameK>("boundary", nullptr));  // Boundary
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<HaltK>());

    EXPECT_EQ(machine->kont_stack.size(), 4);

    bool found = machine->unwind_to_boundary(
        [](Continuation* k) { return k->is_function_boundary(); }
    );

    EXPECT_TRUE(found);
    // Should unwind to the FrameK
    EXPECT_EQ(machine->kont_stack.size(), 2);
    EXPECT_TRUE(machine->kont_stack.back()->is_function_boundary());
}

// Test unwind to boundary not found
TEST_F(MachineTest, UnwindToBoundaryNotFound) {
    machine->push_kont(machine->heap->allocate<HaltK>());
    machine->push_kont(machine->heap->allocate<HaltK>());

    bool found = machine->unwind_to_boundary(
        [](Continuation* k) { return k->is_function_boundary(); }
    );

    EXPECT_FALSE(found);
}

// Test string pool
TEST_F(MachineTest, StringPool) {
    const char* s1 = machine->string_pool.intern("hello");
    const char* s2 = machine->string_pool.intern("hello");
    const char* s3 = machine->string_pool.intern("world");

    // Same string should return same pointer
    EXPECT_EQ(s1, s2);
    // Different string should return different pointer
    EXPECT_NE(s1, s3);

    EXPECT_STREQ(s1, "hello");
    EXPECT_STREQ(s3, "world");
}

// Test environment mark for GC
TEST_F(MachineTest, EnvironmentMarkForGC) {
    Value* v1 = machine->heap->allocate_scalar(10.0);
    Value* v2 = machine->heap->allocate_scalar(20.0);

    machine->env->define("x", v1);
    machine->env->define("y", v2);

    // Clear marks
    machine->heap->clear_marks();
    EXPECT_FALSE(v1->marked);
    EXPECT_FALSE(v2->marked);

    // Mark environment
    machine->env->mark(machine->heap);

    // Both values should be marked
    EXPECT_TRUE(v1->marked);
    EXPECT_TRUE(v2->marked);
}

// Test nested environment mark
TEST_F(MachineTest, NestedEnvironmentMark) {
    Value* v1 = machine->heap->allocate_scalar(1.0);
    machine->env->define("a", v1);

    Environment* child = machine->heap->allocate<Environment>(machine->env);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    child->define("b", v2);

    machine->heap->clear_marks();
    child->mark(machine->heap);

    // Both parent and child values should be marked
    EXPECT_TRUE(v1->marked);
    EXPECT_TRUE(v2->marked);

    // GC will clean up -     delete child;
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
