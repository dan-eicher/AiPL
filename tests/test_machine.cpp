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
    machine->result = v;
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
    machine->result = v;

    Value* result = machine->execute();

    // With empty stack, should just return the value
    EXPECT_EQ(result, v);
    EXPECT_FALSE(machine->should_continue());
}

// Test environment variable lookup
TEST_F(MachineTest, EnvironmentLookup) {
    Value* v = machine->heap->allocate_scalar(5.0);
    machine->env->define(machine->string_pool.intern("x"), v);

    Value* result = machine->env->lookup(machine->string_pool.intern("x"));
    EXPECT_EQ(result, v);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test environment lookup failure
TEST_F(MachineTest, EnvironmentLookupFailure) {
    Value* result = machine->env->lookup(machine->string_pool.intern("nonexistent"));
    EXPECT_EQ(result, nullptr);
}

// Test environment update
TEST_F(MachineTest, EnvironmentUpdate) {
    Value* v1 = machine->heap->allocate_scalar(10.0);
    machine->env->define(machine->string_pool.intern("y"), v1);

    Value* v2 = machine->heap->allocate_scalar(20.0);
    machine->env->define(machine->string_pool.intern("y"), v2);

    Value* result = machine->env->lookup(machine->string_pool.intern("y"));
    EXPECT_EQ(result, v2);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

// Test nested environment
TEST_F(MachineTest, NestedEnvironment) {
    Value* v1 = machine->heap->allocate_scalar(1.0);
    machine->env->define(machine->string_pool.intern("a"), v1);

    Environment* child = machine->heap->allocate<Environment>(machine->env);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    String* b_name = machine->string_pool.intern("b");
    child->define(b_name, v2);

    // Child can see parent's binding
    Value* result = child->lookup(machine->string_pool.intern("a"));
    EXPECT_EQ(result, v1);

    // Child can see its own binding
    result = child->lookup(b_name);
    EXPECT_EQ(result, v2);

    // Parent cannot see child's binding
    result = machine->env->lookup(machine->string_pool.intern("b"));
    EXPECT_EQ(result, nullptr);

}

// Test nested environment shadowing
TEST_F(MachineTest, NestedEnvironmentShadowing) {
    Value* v1 = machine->heap->allocate_scalar(10.0);
    machine->env->define(machine->string_pool.intern("x"), v1);

    Environment* child = machine->heap->allocate<Environment>(machine->env);
    Value* v2 = machine->heap->allocate_scalar(20.0);
    String* x_name = machine->string_pool.intern("x");
    child->define(x_name, v2);

    // Child sees its own binding, not parent's
    Value* result = child->lookup(x_name);
    EXPECT_EQ(result, v2);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);

    // Parent still has original binding
    result = machine->env->lookup(machine->string_pool.intern("x"));
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

    machine->env->define(machine->string_pool.intern("a"), v1);
    machine->env->define(machine->string_pool.intern("b"), v2);
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
    Continuation* k2 = machine->heap->allocate<FrameK>(machine->string_pool.intern("func"), nullptr);
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
    String* s1 = machine->string_pool.intern("hello");
    String* s2 = machine->string_pool.intern("hello");
    String* s3 = machine->string_pool.intern("world");

    // Same string should return same pointer
    EXPECT_EQ(s1, s2);
    // Different string should return different pointer
    EXPECT_NE(s1, s3);

    EXPECT_STREQ(s1->c_str(), "hello");
    EXPECT_STREQ(s3->c_str(), "world");
}

// Test environment mark for GC
TEST_F(MachineTest, EnvironmentMarkForGC) {
    Value* v1 = machine->heap->allocate_scalar(10.0);
    Value* v2 = machine->heap->allocate_scalar(20.0);

    machine->env->define(machine->string_pool.intern("x"), v1);
    machine->env->define(machine->string_pool.intern("y"), v2);

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
    machine->env->define(machine->string_pool.intern("a"), v1);

    Environment* child = machine->heap->allocate<Environment>(machine->env);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    child->define(machine->string_pool.intern("b"), v2);

    machine->heap->clear_marks();
    child->mark(machine->heap);

    // Both parent and child values should be marked
    EXPECT_TRUE(v1->marked);
    EXPECT_TRUE(v2->marked);

}

// ============================================================================
// Dynamic Scoping Tests (ISO 13751 Section 13.4.1)
// ============================================================================
// APL uses dynamic scoping: called functions see the caller's local bindings,
// not the bindings from where the function was defined.

// Test: Called function sees caller's shadowed variable
TEST_F(MachineTest, DynamicScopingBasic) {
    // Set up:
    // X←10           (global X)
    // G←{⍵+X}        (G uses X)
    // F←{X←99 ⋄ G ⍵} (F shadows X, then calls G)
    // F 1            (should return 1+99=100, not 1+10=11)

    machine->eval("X←10");
    machine->eval("G←{⍵+X}");
    machine->eval("F←{X←99 ⋄ G ⍵}");

    Value* result = machine->eval("F 1");

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    // Dynamic scoping: G sees F's local X (99), not global X (10)
    EXPECT_DOUBLE_EQ(result->as_scalar(), 100.0);
}

// Test: Nested function calls maintain dynamic scope chain
TEST_F(MachineTest, DynamicScopingNested) {
    // X←1
    // H←{X}           (H returns X)
    // G←{X←3 ⋄ H 0}   (G shadows X=3, calls H)
    // F←{X←2 ⋄ G 0}   (F shadows X=2, calls G)
    // F 0             (should return 3 - H sees G's X)

    machine->eval("X←1");
    machine->eval("H←{X}");
    machine->eval("G←{X←3 ⋄ H 0}");
    machine->eval("F←{X←2 ⋄ G 0}");

    Value* result = machine->eval("F 0");

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    // H sees G's local X (3), the most recent shadow in call chain
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

// Test: Global is restored after function returns
TEST_F(MachineTest, DynamicScopingRestoresGlobal) {
    // X←10
    // F←{X←99 ⋄ X}  (F shadows X, returns it)
    // F 0           (should return 99)
    // X             (should return 10 - global is unchanged)

    machine->eval("X←10");
    machine->eval("F←{X←99 ⋄ X}");

    Value* result1 = machine->eval("F 0");
    ASSERT_NE(result1, nullptr);
    EXPECT_DOUBLE_EQ(result1->as_scalar(), 99.0);

    Value* result2 = machine->eval("X");
    ASSERT_NE(result2, nullptr);
    EXPECT_DOUBLE_EQ(result2->as_scalar(), 10.0);
}

// Test: Function without shadowing sees global
TEST_F(MachineTest, DynamicScopingSeesGlobal) {
    // X←42
    // G←{X+⍵}       (G uses global X)
    // G 8           (should return 42+8=50)

    machine->eval("X←42");
    machine->eval("G←{X+⍵}");

    Value* result = machine->eval("G 8");

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 50.0);
}

// Test: Intermediate function without shadowing passes through
TEST_F(MachineTest, DynamicScopingPassthrough) {
    // X←10
    // H←{X}           (H returns X)
    // G←{H 0}         (G just calls H, doesn't shadow X)
    // F←{X←99 ⋄ G 0}  (F shadows X, calls G)
    // F 0             (should return 99 - H sees F's X through G)

    machine->eval("X←10");
    machine->eval("H←{X}");
    machine->eval("G←{H 0}");
    machine->eval("F←{X←99 ⋄ G 0}");

    Value* result = machine->eval("F 0");

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    // H sees F's X (99) because G doesn't shadow it
    EXPECT_DOUBLE_EQ(result->as_scalar(), 99.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
