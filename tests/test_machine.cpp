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
    // Phase 1: No more ctrl.mode - halted state is just empty stack
    EXPECT_FALSE(machine->should_continue());
}

// Test should_continue - now based on stack emptiness
TEST_F(MachineTest, ShouldContinue) {
    // Initially empty stack = halted
    EXPECT_FALSE(machine->should_continue());

    // Push a continuation = should continue
    machine->push_kont(machine->heap->allocate<HaltK>());
    EXPECT_TRUE(machine->should_continue());

    // Pop continuation = halted again
    machine->pop_kont();
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
    // Phase 1: Stack should be empty after execution
    EXPECT_FALSE(machine->should_continue());
}

// Test execute with empty stack
TEST_F(MachineTest, ExecuteEmptyStack) {
    Value* v = machine->heap->allocate_scalar(100.0);
    machine->ctrl.value = v;

    Value* result = machine->execute();

    // With empty stack, should just return the value
    EXPECT_EQ(result, v);
    EXPECT_FALSE(machine->should_continue());
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

}

// Phase 2 complete: Completion handling now done through continuations
// Old imperative completion tests removed - new system tested in test_statements.cpp
// See: LeaveFromWhile, LeaveFromFor, LeaveFromNested for :Leave tests

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

// Test function cache survives GC (Phase 4)
TEST_F(MachineTest, FunctionCacheSurvivesGC) {
    // Allocate a continuation and add to cache
    Continuation* cached_k = machine->heap->allocate<HaltK>();
    machine->function_cache["test_func"] = cached_k;

    // Allocate some other objects that won't be reachable
    for (int i = 0; i < 100; i++) {
        machine->heap->allocate<HaltK>();
    }

    // Force a GC - cached continuation should be marked as a root
    machine->heap->collect(machine);

    // Cached continuation should be marked (reachable from function_cache)
    EXPECT_TRUE(cached_k->marked);

    // Cache should still contain the continuation
    EXPECT_EQ(machine->function_cache.size(), 1);
    EXPECT_EQ(machine->function_cache["test_func"], cached_k);
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

// Phase 2.4: Removed unwind_to_boundary tests - unwinding now done by PropagateCompletionK
// Stack unwinding is tested through :Leave and :Return integration tests

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

}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
