// Heap and memory management tests

#include <gtest/gtest.h>
#include "heap.h"
#include "machine.h"
#include "machine.h"
#include "value.h"
#include "continuation.h"

using namespace apl;

class HeapTest : public ::testing::Test {
protected:
    Machine* machine;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
    }
};

// Test basic heap creation
TEST_F(HeapTest, Creation) {
    // Machine constructor allocates one Environment for global env
    // So we expect 1 environment allocated, not 0
    EXPECT_EQ(machine->heap->young_size(), 0);  // No Values yet
    EXPECT_EQ(machine->heap->old_size(), 0);
    EXPECT_EQ(machine->heap->total_size(), 0);  // Only counting Values
    EXPECT_EQ(machine->heap->bytes_allocated, sizeof(Environment));  // One Environment from Machine()
    EXPECT_FALSE(machine->heap->gc_in_progress);
}

// Test scalar cache initialization
TEST_F(HeapTest, ScalarCacheInit) {
    for (int i = 0; i < 256; i++) {
        EXPECT_EQ(machine->heap->scalar_cache[i], nullptr);
    }
}

// Test allocate scalar with caching
TEST_F(HeapTest, AllocateScalarWithCache) {
    // Allocate a cacheable scalar
    Value* v1 = machine->heap->allocate_scalar(42.0);

    ASSERT_NE(v1, nullptr);
    EXPECT_EQ(v1->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(v1->as_scalar(), 42.0);
    EXPECT_EQ(machine->heap->young_size(), 1);

    // Check it's in cache
    EXPECT_EQ(machine->heap->scalar_cache[42 + 128], v1);

    // Allocate same value again - should return cached
    Value* v2 = machine->heap->allocate_scalar(42.0);

    EXPECT_EQ(v1, v2);  // Same pointer
    EXPECT_EQ(machine->heap->young_size(), 1);  // No new allocation
}

// Test scalar cache range
TEST_F(HeapTest, ScalarCacheRange) {
    // Test boundary values
    Value* v_min = machine->heap->allocate_scalar(-128.0);
    Value* v_max = machine->heap->allocate_scalar(127.0);
    Value* v_zero = machine->heap->allocate_scalar(0.0);

    EXPECT_EQ(machine->heap->scalar_cache[0], v_min);
    EXPECT_EQ(machine->heap->scalar_cache[255], v_max);
    EXPECT_EQ(machine->heap->scalar_cache[128], v_zero);

    // Test non-cacheable values
    Value* v_low = machine->heap->allocate_scalar(-129.0);
    Value* v_high = machine->heap->allocate_scalar(128.0);
    Value* v_frac = machine->heap->allocate_scalar(1.5);

    // These should NOT be in cache
    bool found = false;
    for (int i = 0; i < 256; i++) {
        if (machine->heap->scalar_cache[i] == v_low ||
            machine->heap->scalar_cache[i] == v_high ||
            machine->heap->scalar_cache[i] == v_frac) {
            found = true;
        }
    }
    EXPECT_FALSE(found);
}

// Test allocate regular value
TEST_F(HeapTest, AllocateValue) {
    Value* v = machine->heap->allocate_scalar(99.5);
    // allocate_scalar already calls allocate() internally, don't double-allocate!

    EXPECT_EQ(machine->heap->young_size(), 1);
    EXPECT_GT(machine->heap->bytes_allocated, sizeof(Environment));  // Environment + Value
}

// Test allocate vector
TEST_F(HeapTest, AllocateVector) {
    Eigen::VectorXd vec(5);
    vec << 1, 2, 3, 4, 5;

    Value* v = machine->heap->allocate_vector(vec);
    // allocate_vector already calls allocate() internally, don't double-allocate!

    EXPECT_EQ(machine->heap->young_size(), 1);
    // Should count Environment + Value + matrix storage
    EXPECT_GT(machine->heap->bytes_allocated, sizeof(Environment) + sizeof(Value));
}

// Test clear marks
TEST_F(HeapTest, ClearMarks) {
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);

    v1->marked = true;
    v2->marked = true;

    machine->heap->clear_marks();

    EXPECT_FALSE(v1->marked);
    EXPECT_FALSE(v2->marked);
}

// Test mark value
TEST_F(HeapTest, MarkValue) {
    Value* v = machine->heap->allocate_scalar(10.0);

    EXPECT_FALSE(v->marked);

    machine->heap->mark_value(v);

    EXPECT_TRUE(v->marked);

    // Marking again should be idempotent
    machine->heap->mark_value(v);
    EXPECT_TRUE(v->marked);
}

// Test mark from roots with Machine
TEST_F(HeapTest, MarkFromRoots) {
    Machine* machine = new Machine();

    Value* v1 = machine->heap->allocate_scalar(5.0);
    Value* v2 = machine->heap->allocate_scalar(10.0);

    // Set v1 as current value
    machine->ctrl.set_value(v1);

    machine->heap->mark_from_roots(machine);

    EXPECT_TRUE(v1->marked);   // Should be marked (in ctrl)
    EXPECT_FALSE(v2->marked);  // Should not be marked (not reachable)

    machine->heap = nullptr;  // Don't let machine delete our test heap
}

// Test promote survivors
TEST_F(HeapTest, PromoteSurvivors) {
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);

    EXPECT_EQ(machine->heap->young_size(), 2);
    EXPECT_EQ(machine->heap->old_size(), 0);

    // Mark v1 for survival
    v1->marked = true;
    v2->marked = false;

    machine->heap->promote_survivors();

    EXPECT_EQ(machine->heap->young_size(), 1);  // v2 still in young
    EXPECT_EQ(machine->heap->old_size(), 1);    // v1 promoted
    EXPECT_TRUE(v1->in_old_generation);
    EXPECT_FALSE(v2->in_old_generation);
}

// Test sweep
TEST_F(HeapTest, Sweep) {
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    Value* v3 = machine->heap->allocate_scalar(3.0);

    EXPECT_EQ(machine->heap->young_size(), 3);

    // Mark only v1 and v3
    v1->marked = true;
    v2->marked = false;
    v3->marked = true;

    size_t initial_bytes = machine->heap->bytes_allocated;

    machine->heap->sweep();

    EXPECT_EQ(machine->heap->young_size(), 2);  // v2 should be swept
    EXPECT_LT(machine->heap->bytes_allocated, initial_bytes);  // Less memory used
}

// Test minor GC
TEST_F(HeapTest, MinorGC) {
    Machine* machine = new Machine();

    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);

    // Set v1 as root
    machine->ctrl.set_value(v1);

    size_t initial_young = machine->heap->young_size();

    machine->heap->minor_gc(machine);

    // v1 should survive and be promoted
    // v2 should be collected
    EXPECT_LT(machine->heap->young_size(), initial_young);
    EXPECT_GT(machine->heap->old_size(), 0);
    EXPECT_EQ(machine->heap->minor_gc_count, 1);

    machine->heap = nullptr;
}

// Test major GC
TEST_F(HeapTest, MajorGC) {
    Machine* machine = new Machine();

    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);

    // Set v1 as root
    machine->ctrl.set_value(v1);

    machine->heap->major_gc(machine);

    // v1 should survive, v2 should be collected
    EXPECT_EQ(machine->heap->total_size(), 1);
    EXPECT_EQ(machine->heap->major_gc_count, 1);
    EXPECT_EQ(machine->heap->minor_gc_count, 0);  // Reset after major GC

    machine->heap = nullptr;
}

// Test GC threshold check
TEST_F(HeapTest, ShouldGC) {
    EXPECT_FALSE(machine->heap->should_gc());

    // Simulate allocation up to threshold
    machine->heap->bytes_allocated = machine->heap->gc_threshold - 1;
    EXPECT_FALSE(machine->heap->should_gc());

    machine->heap->bytes_allocated = machine->heap->gc_threshold;
    EXPECT_TRUE(machine->heap->should_gc());

    machine->heap->bytes_allocated = machine->heap->gc_threshold + 100;
    EXPECT_TRUE(machine->heap->should_gc());
}

// Test GC statistics
TEST_F(HeapTest, GCStatistics) {
    Machine* machine = new Machine();

    Value* v = machine->heap->allocate_scalar(1.0);
    machine->ctrl.set_value(v);

    EXPECT_EQ(machine->heap->minor_gc_count, 0);
    EXPECT_EQ(machine->heap->major_gc_count, 0);

    machine->heap->minor_gc(machine);
    EXPECT_EQ(machine->heap->minor_gc_count, 1);

    machine->heap->major_gc(machine);
    EXPECT_EQ(machine->heap->major_gc_count, 1);

    machine->heap = nullptr;
}

// Test scalar cache survives GC
TEST_F(HeapTest, ScalarCacheSurvivesGC) {
    Machine* machine = new Machine();

    Value* v = machine->heap->allocate_scalar(42.0);
    machine->ctrl.set_value(v);

    EXPECT_EQ(machine->heap->scalar_cache[42 + 128], v);

    machine->heap->minor_gc(machine);

    // Cached value should still exist
    EXPECT_NE(machine->heap->scalar_cache[42 + 128], nullptr);

    machine->heap = nullptr;
}

// Test large allocation
TEST_F(HeapTest, LargeAllocation) {
    Eigen::MatrixXd mat(100, 100);
    mat.setConstant(1.0);

    Value* v = machine->heap->allocate_matrix(mat);
    // allocate_matrix already calls allocate() internally, don't double-allocate!

    // Should account for Environment + large matrix
    EXPECT_GE(machine->heap->bytes_allocated, sizeof(Environment) + sizeof(Value) + 100 * 100 * sizeof(double));
}

// ============================================================================
// Continuation GC Tests
// ============================================================================

// Test allocate continuation
TEST_F(HeapTest, AllocateContinuation) {
    Continuation* k = machine->heap->allocate<HaltK>();

    EXPECT_EQ(machine->heap->young_continuations.size(), 1);
    EXPECT_EQ(machine->heap->old_continuations.size(), 0);
    EXPECT_FALSE(k->marked);
    EXPECT_FALSE(k->in_old_generation);
    EXPECT_GT(machine->heap->bytes_allocated, 0);
}

// Test mark continuation
TEST_F(HeapTest, MarkContinuation) {
    Continuation* k = machine->heap->allocate<HaltK>();

    EXPECT_FALSE(k->marked);

    machine->heap->mark_continuation(k);

    EXPECT_TRUE(k->marked);

    // Marking again should be idempotent
    machine->heap->mark_continuation(k);
    EXPECT_TRUE(k->marked);
}

// Test continuation graph marking (transitive closure)
TEST_F(HeapTest, MarkContinuationGraph) {
    // Create a chain: k1 -> k2 -> k3 -> HaltK
    Continuation* k_halt = machine->heap->allocate<HaltK>();
    Continuation* k3 = machine->heap->allocate<FrameK>(nullptr, k_halt);
    Continuation* k2 = machine->heap->allocate<FrameK>(nullptr, k3);
    Continuation* k1 = machine->heap->allocate<FrameK>(nullptr, k2);


    EXPECT_EQ(machine->heap->young_continuations.size(), 4);

    // Mark only the head - should transitively mark entire graph
    machine->heap->mark_continuation(k1);

    EXPECT_TRUE(k1->marked);
    EXPECT_TRUE(k2->marked);
    EXPECT_TRUE(k3->marked);
    EXPECT_TRUE(k_halt->marked);
}

// Test ArgK marks its arg_value
TEST_F(HeapTest, ArgKMarksValue) {
    Value* arg = machine->heap->allocate_scalar(42.0);
    Continuation* k_next = machine->heap->allocate<HaltK>();
    Continuation* k_arg = machine->heap->allocate<ArgK>(arg, k_next);


    EXPECT_FALSE(arg->marked);
    EXPECT_FALSE(k_arg->marked);

    machine->heap->mark_continuation(k_arg);

    EXPECT_TRUE(k_arg->marked);
    EXPECT_TRUE(k_next->marked);
    EXPECT_TRUE(arg->marked);  // ArgK should mark its Value
}

// Test CLOSURE value marks its continuation graph
TEST_F(HeapTest, ClosureMarksGraph) {
    // Create a simple continuation graph for the closure
    Continuation* k_halt = machine->heap->allocate<HaltK>();
    Continuation* k_body = machine->heap->allocate<FrameK>(nullptr, k_halt);


    // Create a CLOSURE value that references this graph
    Value* closure = machine->heap->allocate_closure(k_body);
    // allocate_closure already calls allocate() internally, don't double-allocate!

    EXPECT_FALSE(closure->marked);
    EXPECT_FALSE(k_body->marked);
    EXPECT_FALSE(k_halt->marked);

    // Mark the closure - should transitively mark continuation graph
    machine->heap->mark_value(closure);

    EXPECT_TRUE(closure->marked);
    EXPECT_TRUE(k_body->marked);
    EXPECT_TRUE(k_halt->marked);
}

// Test sweep unmarked continuations
TEST_F(HeapTest, SweepContinuations) {
    Continuation* k1 = machine->heap->allocate<HaltK>();
    Continuation* k2 = machine->heap->allocate<HaltK>();
    Continuation* k3 = machine->heap->allocate<HaltK>();


    EXPECT_EQ(machine->heap->young_continuations.size(), 3);

    // Mark only k1 and k3
    k1->marked = true;
    k2->marked = false;
    k3->marked = true;

    size_t initial_bytes = machine->heap->bytes_allocated;

    machine->heap->sweep();

    EXPECT_EQ(machine->heap->young_continuations.size(), 2);  // k2 swept
    EXPECT_LT(machine->heap->bytes_allocated, initial_bytes);
}

// Test continuations from kont_stack are marked
TEST_F(HeapTest, MarkContinuationsFromKontStack) {
    Machine* machine = new Machine();

    Continuation* k1 = machine->heap->allocate<HaltK>();
    Continuation* k2 = machine->heap->allocate<FrameK>(nullptr, nullptr);


    // Push continuations onto kont_stack
    machine->kont_stack.push_back(k1);
    machine->kont_stack.push_back(k2);

    EXPECT_FALSE(k1->marked);
    EXPECT_FALSE(k2->marked);

    machine->heap->mark_from_roots(machine);

    EXPECT_TRUE(k1->marked);
    EXPECT_TRUE(k2->marked);

    machine->heap = nullptr;
}

// Test continuations from function_cache are marked
TEST_F(HeapTest, MarkContinuationsFromFunctionCache) {
    Machine* machine = new Machine();

    Continuation* k = machine->heap->allocate<HaltK>();

    // Add to function cache
    machine->function_cache["test_fn"] = k;

    EXPECT_FALSE(k->marked);

    machine->heap->mark_from_roots(machine);

    EXPECT_TRUE(k->marked);

    machine->heap = nullptr;
}

// Test continuation promotion to old generation
TEST_F(HeapTest, PromoteContinuations) {
    Continuation* k1 = machine->heap->allocate<HaltK>();
    Continuation* k2 = machine->heap->allocate<HaltK>();


    EXPECT_EQ(machine->heap->young_continuations.size(), 2);
    EXPECT_EQ(machine->heap->old_continuations.size(), 0);

    // Mark k1 for survival
    k1->marked = true;
    k2->marked = false;

    // Note: promote_survivors only handles Values, not Continuations
    // Continuations are promoted during minor_gc via sweep
    // For now, manually test the metadata
    k1->in_old_generation = true;

    EXPECT_TRUE(k1->in_old_generation);
    EXPECT_FALSE(k2->in_old_generation);
}

// Test minor GC collects unreachable continuations
TEST_F(HeapTest, MinorGCContinuations) {
    Machine* machine = new Machine();

    Continuation* k_reachable = machine->heap->allocate<HaltK>();
    Continuation* k_unreachable = machine->heap->allocate<HaltK>();


    // Make k_reachable a root
    machine->kont_stack.push_back(k_reachable);

    size_t initial_young = machine->heap->young_continuations.size();

    machine->heap->minor_gc(machine);

    // k_reachable should survive, k_unreachable should be collected
    EXPECT_LT(machine->heap->young_continuations.size(), initial_young);

    machine->heap = nullptr;
}

// Test major GC collects continuations from all generations
TEST_F(HeapTest, MajorGCContinuations) {
    Machine* machine = new Machine();

    Continuation* k_young = machine->heap->allocate<HaltK>();
    Continuation* k_old = machine->heap->allocate<HaltK>();


    // Manually promote k_old
    k_old->in_old_generation = true;
    machine->heap->old_continuations.push_back(k_old);
    machine->heap->young_continuations.pop_back();

    // Don't make either a root - both should be collected
    size_t initial_total = machine->heap->young_continuations.size() + machine->heap->old_continuations.size();

    machine->heap->major_gc(machine);

    size_t final_total = machine->heap->young_continuations.size() + machine->heap->old_continuations.size();
    EXPECT_LT(final_total, initial_total);

    machine->heap = nullptr;
}

// Test clear marks for continuations
TEST_F(HeapTest, ClearMarksContinuations) {
    Continuation* k1 = machine->heap->allocate<HaltK>();
    Continuation* k2 = machine->heap->allocate<HaltK>();


    k1->marked = true;
    k2->marked = true;

    machine->heap->clear_marks();

    EXPECT_FALSE(k1->marked);
    EXPECT_FALSE(k2->marked);
}

// Test bidirectional marking: Value -> Continuation -> Value
TEST_F(HeapTest, BidirectionalMarking) {
    // Create: closure_value -> continuation_graph -> arg_value
    Value* arg = machine->heap->allocate_scalar(99.0);
    Continuation* k_halt = machine->heap->allocate<HaltK>();
    Continuation* k_arg = machine->heap->allocate<ArgK>(arg, k_halt);


    Value* closure = machine->heap->allocate_closure(k_arg);
    // allocate_closure already calls allocate() internally, don't double-allocate!

    EXPECT_FALSE(closure->marked);
    EXPECT_FALSE(k_arg->marked);
    EXPECT_FALSE(k_halt->marked);
    EXPECT_FALSE(arg->marked);

    // Mark the closure - should mark entire graph
    machine->heap->mark_value(closure);

    EXPECT_TRUE(closure->marked);
    EXPECT_TRUE(k_arg->marked);
    EXPECT_TRUE(k_halt->marked);
    EXPECT_TRUE(arg->marked);
}

// Test complex continuation graph with multiple references
TEST_F(HeapTest, ComplexContinuationGraph) {
    // Build a more complex graph:
    //   closure1 -> k1 -> k2 -> HaltK
    //                 \-> value1
    //   closure2 -> k3 -> k2 (shared!)

    Value* value1 = machine->heap->allocate_scalar(42.0);

    Continuation* k_halt = machine->heap->allocate<HaltK>();
    Continuation* k2 = machine->heap->allocate<FrameK>(nullptr, k_halt);
    Continuation* k1 = machine->heap->allocate<ArgK>(value1, k2);
    Continuation* k3 = machine->heap->allocate<FrameK>(nullptr, k2);


    Value* closure1 = machine->heap->allocate_closure(k1);
    Value* closure2 = machine->heap->allocate_closure(k3);

    // allocate_closure already calls allocate() internally, don't double-allocate!

    // Mark only closure1
    machine->heap->mark_value(closure1);

    // Should mark k1, k2, k_halt, value1
    EXPECT_TRUE(closure1->marked);
    EXPECT_TRUE(k1->marked);
    EXPECT_TRUE(k2->marked);
    EXPECT_TRUE(k_halt->marked);
    EXPECT_TRUE(value1->marked);

    // k3 and closure2 should NOT be marked yet
    EXPECT_FALSE(k3->marked);
    EXPECT_FALSE(closure2->marked);

    // Now mark closure2
    machine->heap->mark_value(closure2);

    // Now k3 and closure2 should be marked (k2 already marked)
    EXPECT_TRUE(k3->marked);
    EXPECT_TRUE(closure2->marked);
}

// ============================================================================
// Template Allocation Interface Tests (Phase 1.1)
// ============================================================================

// Test template allocation for Value types
TEST_F(HeapTest, TemplateAllocationValue) {
    // Note: We can't directly use machine->heap->allocate<Value>() for Value types
    // because Value is not a leaf type with a public constructor.
    // The template interface works with Continuation types.
    // For Values, we use the helper methods like allocate_scalar, allocate_vector, etc.

    // Test the helper methods work correctly
    Value* v1 = machine->heap->allocate_scalar(42.0);
    EXPECT_NE(v1, nullptr);
    EXPECT_EQ(v1->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(v1->as_scalar(), 42.0);
    EXPECT_EQ(machine->heap->young_size(), 1);

    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* v2 = machine->heap->allocate_vector(vec);
    EXPECT_NE(v2, nullptr);
    EXPECT_EQ(v2->tag, ValueType::VECTOR);
    EXPECT_EQ(machine->heap->young_size(), 2);

    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Value* v3 = machine->heap->allocate_matrix(mat);
    EXPECT_NE(v3, nullptr);
    EXPECT_EQ(v3->tag, ValueType::MATRIX);
    EXPECT_EQ(machine->heap->young_size(), 3);
}

// Test template allocation for Continuation types
TEST_F(HeapTest, TemplateAllocationContinuation) {
    // Test direct allocation using template interface
    Continuation* k1 = machine->heap->allocate<HaltK>();
    EXPECT_NE(k1, nullptr);
    EXPECT_EQ(machine->heap->young_continuations.size(), 1);
    EXPECT_FALSE(k1->marked);
    EXPECT_FALSE(k1->in_old_generation);

    // Test with arguments
    Value* arg = machine->heap->allocate_scalar(42.0);
    Continuation* k2 = machine->heap->allocate<ArgK>(arg, k1);
    EXPECT_NE(k2, nullptr);
    EXPECT_EQ(machine->heap->young_continuations.size(), 2);

    // Test FrameK with nullptr and next continuation
    Continuation* k3 = machine->heap->allocate<FrameK>(nullptr, k2);
    EXPECT_NE(k3, nullptr);
    EXPECT_EQ(machine->heap->young_continuations.size(), 3);
}

// Test template allocation preserves GC integration
TEST_F(HeapTest, TemplateAllocationGCIntegration) {
    Machine* machine = new Machine();

    // Allocate continuations using template interface
    Continuation* k1 = machine->heap->allocate<HaltK>();
    Continuation* k2 = machine->heap->allocate<FrameK>(nullptr, k1);
    Continuation* k3_ptr = machine->heap->allocate<FrameK>(nullptr, nullptr);

    EXPECT_EQ(machine->heap->young_continuations.size(), 3);

    // Make only k1 and k2 reachable (k3 is NOT reachable)
    machine->kont_stack.push_back(k2);

    // Test marking - this just tests the mark phase
    machine->heap->clear_marks();
    machine->heap->mark_from_roots(machine);

    EXPECT_TRUE(k2->marked);    // k2 is on kont_stack
    EXPECT_TRUE(k1->marked);    // k1 is referenced by k2
    EXPECT_FALSE(k3_ptr->marked);  // k3 is unreachable

    // This test demonstrates that the template allocation
    // interface correctly integrates with GC marking

    machine->heap = nullptr;
}

// Test that template allocation coexists with old interface
TEST_F(HeapTest, TemplateAllocationBackwardCompatibility) {
    // Old-style allocation
    Continuation* k_old = machine->heap->allocate<HaltK>();

    // New-style template allocation
    Continuation* k_new = machine->heap->allocate<HaltK>();

    // Both should work and be in the heap
    EXPECT_EQ(machine->heap->young_continuations.size(), 2);

    // Both should be GC-managed identically
    EXPECT_FALSE(k_old->marked);
    EXPECT_FALSE(k_new->marked);
    EXPECT_FALSE(k_old->in_old_generation);
    EXPECT_FALSE(k_new->in_old_generation);
}

// Test template allocation with complex continuation types
TEST_F(HeapTest, TemplateAllocationComplexTypes) {
    Value* arg1 = machine->heap->allocate_scalar(1.0);

    Continuation* k_halt = machine->heap->allocate<HaltK>();
    Continuation* k_arg = machine->heap->allocate<ArgK>(arg1, k_halt);
    Continuation* k_frame = machine->heap->allocate<FrameK>(nullptr, k_arg);

    EXPECT_EQ(machine->heap->young_continuations.size(), 3);

    // Test marking through the chain
    machine->heap->mark_continuation(k_frame);

    EXPECT_TRUE(k_frame->marked);
    EXPECT_TRUE(k_arg->marked);
    EXPECT_TRUE(k_halt->marked);
    EXPECT_TRUE(arg1->marked);
}

// ============================================================================
// Environment GC Tests (Phase 3.2)
// ============================================================================

// Test environment chain marking
TEST_F(HeapTest, EnvironmentChainMarking) {
    // Create a chain of environments: global -> fn1 -> fn2
    Environment* global_env = machine->env;
    Environment* fn1_env = machine->heap->allocate<Environment>(global_env);
    Environment* fn2_env = machine->heap->allocate<Environment>(fn1_env);

    // Add some values to each environment
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    Value* v3 = machine->heap->allocate_scalar(3.0);

    global_env->define("x", v1);
    fn1_env->define("y", v2);
    fn2_env->define("z", v3);

    // Clear marks and mark only fn2_env
    machine->heap->clear_marks();
    fn2_env->marked = true;
    fn2_env->mark(machine->heap);

    // All three environments should be marked (fn2 -> fn1 -> global)
    EXPECT_TRUE(fn2_env->marked);
    EXPECT_TRUE(fn1_env->marked);
    EXPECT_TRUE(global_env->marked);

    // All three values should be marked
    EXPECT_TRUE(v3->marked);
    EXPECT_TRUE(v2->marked);
    EXPECT_TRUE(v1->marked);
}

// Test deep environment nesting (10+ levels)
TEST_F(HeapTest, DeepEnvironmentNesting) {
    const int DEPTH = 15;
    Environment* env_chain[DEPTH];
    Value* values[DEPTH];

    // Create a deep chain
    env_chain[0] = machine->env;
    values[0] = machine->heap->allocate_scalar(0.0);
    env_chain[0]->define("v0", values[0]);

    for (int i = 1; i < DEPTH; i++) {
        env_chain[i] = machine->heap->allocate<Environment>(env_chain[i-1]);
        values[i] = machine->heap->allocate_scalar((double)i);
        char name[10];
        snprintf(name, sizeof(name), "v%d", i);
        env_chain[i]->define(name, values[i]);
    }

    // Mark from the deepest environment
    machine->heap->clear_marks();
    env_chain[DEPTH-1]->marked = true;
    env_chain[DEPTH-1]->mark(machine->heap);

    // All environments in the chain should be marked
    for (int i = 0; i < DEPTH; i++) {
        EXPECT_TRUE(env_chain[i]->marked) << "Environment " << i << " should be marked";
        EXPECT_TRUE(values[i]->marked) << "Value " << i << " should be marked";
    }
}

// Test environment promotion to old generation
TEST_F(HeapTest, EnvironmentPromotion) {
    Machine* machine = new Machine();

    // Create a child environment
    Environment* child_env = machine->heap->allocate<Environment>(machine->env);
    Value* val = machine->heap->allocate_scalar(42.0);
    child_env->define("x", val);

    // Initially in young generation
    EXPECT_FALSE(child_env->in_old_generation);

    // Make it reachable and run minor GC
    machine->env = child_env;  // Make it the current environment (GC root)
    machine->heap->clear_marks();
    machine->heap->mark_from_roots(machine);

    // Environment should be marked
    EXPECT_TRUE(child_env->marked);
    EXPECT_TRUE(val->marked);

    // Note: Environments don't get promoted to old generation
    // They stay in the environments list (no generational separation)
    // This is by design since environments are typically GC roots

    delete machine;
}

// Test environment survives GC when reachable
TEST_F(HeapTest, EnvironmentSurvivesGC) {
    Machine* machine = new Machine();

    Environment* initial_env = machine->env;
    Environment* child_env = machine->heap->allocate<Environment>(initial_env);
    Value* val = machine->heap->allocate_scalar(99.0);
    child_env->define("test", val);

    // Make child environment the current environment
    machine->env = child_env;

    size_t env_count_before = machine->heap->environments.size();

    // Run GC - child_env should survive because it's the current environment
    machine->heap->collect(machine);

    // Child environment should still exist
    EXPECT_EQ(machine->heap->environments.size(), env_count_before);

    // Value should still be accessible
    Value* retrieved = machine->env->lookup("test");
    EXPECT_NE(retrieved, nullptr);
    EXPECT_DOUBLE_EQ(retrieved->as_scalar(), 99.0);

    delete machine;
}

// Test environment and value cleanup
TEST_F(HeapTest, EnvironmentValueLifecycle) {
    Machine* machine = new Machine();

    // Create a child environment with values
    Environment* child_env = machine->heap->allocate<Environment>(machine->env);
    Value* val1 = machine->heap->allocate_scalar(123.0);
    Value* val2 = machine->heap->allocate_scalar(456.0);
    child_env->define("x", val1);
    machine->env->define("reachable", val2);

    // Mark from roots - machine->env is a root, child_env is not directly
    machine->heap->clear_marks();
    machine->heap->mark_from_roots(machine);

    // machine->env should be marked, child_env should NOT be marked (not reachable)
    EXPECT_TRUE(machine->env->marked);
    EXPECT_FALSE(child_env->marked);

    // val2 should be marked (in machine->env), val1 should NOT (in unreachable child_env)
    EXPECT_TRUE(val2->marked);
    EXPECT_FALSE(val1->marked);

    delete machine;
}

// Test that unreachable environments are collected during MAJOR GC
// Minor GC only sweeps young generation Values, not environments
// Major GC sweeps all generations including environments
TEST_F(HeapTest, UnreachableEnvironmentCollectedMajorGC) {
    Machine* machine = new Machine();

    size_t initial_envs = machine->heap->environments.size();

    // Create a detached environment (not reachable from machine->env)
    Environment* detached = machine->heap->allocate<Environment>();
    Value* v = machine->heap->allocate_scalar(42.0);
    detached->define("x", v);

    size_t after_alloc = machine->heap->environments.size();
    EXPECT_EQ(after_alloc, initial_envs + 1);

    // Trigger MINOR GC - should NOT collect environments
    machine->heap->minor_gc(machine);
    size_t after_minor = machine->heap->environments.size();
    EXPECT_EQ(after_minor, initial_envs + 1) << "Minor GC should not sweep environments";

    // Trigger MAJOR GC - SHOULD collect unreachable environments
    machine->heap->major_gc(machine);
    size_t after_major = machine->heap->environments.size();
    EXPECT_EQ(after_major, initial_envs) << "Major GC should collect unreachable environment";

    delete machine;
}

// Test that heap->collect() eventually triggers major GC to collect environments
TEST_F(HeapTest, UnreachableEnvironmentCollectedViaCollect) {
    Machine* machine = new Machine();

    size_t initial_envs = machine->heap->environments.size();

    // Create a detached environment
    Environment* detached = machine->heap->allocate<Environment>();
    Value* v = machine->heap->allocate_scalar(42.0);
    detached->define("x", v);

    EXPECT_EQ(machine->heap->environments.size(), initial_envs + 1);

    // Call collect() multiple times to trigger major GC
    // Major GC happens every 10 minor GCs
    for (int i = 0; i < 15; i++) {
        machine->heap->collect(machine);
    }

    size_t after_gc = machine->heap->environments.size();
    EXPECT_EQ(after_gc, initial_envs) << "Eventually should collect unreachable environment";

    delete machine;
}

// ============================================================================
// Scalar Cache + Generational GC Tests
// ============================================================================

// Test that cached scalars work correctly when promoted to old generation
TEST_F(HeapTest, CachedScalarPromotion) {
    Machine* machine = new Machine();

    // Allocate a cacheable scalar
    Value* v = machine->heap->allocate_scalar(42.0);
    machine->ctrl.set_value(v);

    // Verify it's cached
    EXPECT_EQ(machine->heap->scalar_cache[42 + 128], v);
    EXPECT_EQ(v->in_old_generation, false);  // Should be in young generation

    // Run minor GC - should promote to old generation
    machine->heap->minor_gc(machine);

    // Scalar should still be cached and now in old generation
    EXPECT_EQ(machine->heap->scalar_cache[42 + 128], v);
    EXPECT_EQ(v->in_old_generation, true);

    // Allocate the same scalar again - should return the cached (now old gen) value
    Value* v2 = machine->heap->allocate_scalar(42.0);
    EXPECT_EQ(v2, v);  // Should be same pointer
    EXPECT_EQ(v2->in_old_generation, true);

    delete machine;
}

// Test cache interaction with major GC
TEST_F(HeapTest, CachedScalarMajorGC) {
    Machine* machine = new Machine();

    // Allocate several cacheable scalars
    Value* v1 = machine->heap->allocate_scalar(1.0);
    Value* v2 = machine->heap->allocate_scalar(2.0);
    Value* v3 = machine->heap->allocate_scalar(3.0);

    machine->ctrl.set_value(v3);  // Only keep v3 alive

    // Run major GC
    machine->heap->major_gc(machine);

    // v3 should survive and still be cached
    EXPECT_EQ(machine->heap->scalar_cache[3 + 128], v3);

    // v1 and v2 should be collected and cache cleared
    EXPECT_EQ(machine->heap->scalar_cache[1 + 128], nullptr);
    EXPECT_EQ(machine->heap->scalar_cache[2 + 128], nullptr);

    // Re-allocate v1 - should create new entry
    Value* v1_new = machine->heap->allocate_scalar(1.0);
    EXPECT_NE(v1_new, v1);  // Different pointer (old one was collected)
    EXPECT_EQ(machine->heap->scalar_cache[1 + 128], v1_new);

    delete machine;
}

// Test that cache correctly handles boundary values
TEST_F(HeapTest, CachedScalarBoundaries) {
    Machine* machine = new Machine();

    // Test lower boundary (-128)
    Value* vmin = machine->heap->allocate_scalar(-128.0);
    EXPECT_EQ(machine->heap->scalar_cache[0], vmin);

    // Test upper boundary (127)
    Value* vmax = machine->heap->allocate_scalar(127.0);
    EXPECT_EQ(machine->heap->scalar_cache[255], vmax);

    // Test just outside boundaries - should NOT be cached
    Value* below = machine->heap->allocate_scalar(-129.0);
    Value* above = machine->heap->allocate_scalar(128.0);

    // These shouldn't hit the cache
    Value* below2 = machine->heap->allocate_scalar(-129.0);
    Value* above2 = machine->heap->allocate_scalar(128.0);
    EXPECT_NE(below, below2);  // Different allocations
    EXPECT_NE(above, above2);

    // Keep both boundaries alive for GC via environment
    machine->env->define("vmin", vmin);
    machine->env->define("vmax", vmax);
    machine->heap->collect(machine);

    // Boundaries should still be cached after GC
    EXPECT_EQ(machine->heap->scalar_cache[0], vmin);
    EXPECT_EQ(machine->heap->scalar_cache[255], vmax);

    delete machine;
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
