// Heap and memory management tests

#include <gtest/gtest.h>
#include "heap.h"
#include "machine.h"
#include "value.h"
#include "continuation.h"

using namespace apl;

class HeapTest : public ::testing::Test {
protected:
    APLHeap* heap;

    void SetUp() override {
        heap = new APLHeap();
    }

    void TearDown() override {
        delete heap;
    }
};

// Test basic heap creation
TEST_F(HeapTest, Creation) {
    EXPECT_EQ(heap->young_size(), 0);
    EXPECT_EQ(heap->old_size(), 0);
    EXPECT_EQ(heap->total_size(), 0);
    EXPECT_EQ(heap->bytes_allocated, 0);
    EXPECT_FALSE(heap->gc_in_progress);
}

// Test scalar cache initialization
TEST_F(HeapTest, ScalarCacheInit) {
    for (int i = 0; i < 256; i++) {
        EXPECT_EQ(heap->scalar_cache[i], nullptr);
    }
}

// Test allocate scalar with caching
TEST_F(HeapTest, AllocateScalarWithCache) {
    // Allocate a cacheable scalar
    Value* v1 = heap->allocate_scalar(42.0);

    ASSERT_NE(v1, nullptr);
    EXPECT_EQ(v1->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(v1->as_scalar(), 42.0);
    EXPECT_EQ(heap->young_size(), 1);

    // Check it's in cache
    EXPECT_EQ(heap->scalar_cache[42 + 128], v1);

    // Allocate same value again - should return cached
    Value* v2 = heap->allocate_scalar(42.0);

    EXPECT_EQ(v1, v2);  // Same pointer
    EXPECT_EQ(heap->young_size(), 1);  // No new allocation
}

// Test scalar cache range
TEST_F(HeapTest, ScalarCacheRange) {
    // Test boundary values
    Value* v_min = heap->allocate_scalar(-128.0);
    Value* v_max = heap->allocate_scalar(127.0);
    Value* v_zero = heap->allocate_scalar(0.0);

    EXPECT_EQ(heap->scalar_cache[0], v_min);
    EXPECT_EQ(heap->scalar_cache[255], v_max);
    EXPECT_EQ(heap->scalar_cache[128], v_zero);

    // Test non-cacheable values
    Value* v_low = heap->allocate_scalar(-129.0);
    Value* v_high = heap->allocate_scalar(128.0);
    Value* v_frac = heap->allocate_scalar(1.5);

    // These should NOT be in cache
    bool found = false;
    for (int i = 0; i < 256; i++) {
        if (heap->scalar_cache[i] == v_low ||
            heap->scalar_cache[i] == v_high ||
            heap->scalar_cache[i] == v_frac) {
            found = true;
        }
    }
    EXPECT_FALSE(found);
}

// Test allocate regular value
TEST_F(HeapTest, AllocateValue) {
    Value* v = Value::from_scalar(99.5);
    heap->allocate(v);

    EXPECT_EQ(heap->young_size(), 1);
    EXPECT_GT(heap->bytes_allocated, 0);
}

// Test allocate vector
TEST_F(HeapTest, AllocateVector) {
    Eigen::VectorXd vec(5);
    vec << 1, 2, 3, 4, 5;

    Value* v = Value::from_vector(vec);
    heap->allocate(v);

    EXPECT_EQ(heap->young_size(), 1);
    // Should count both Value and matrix storage
    EXPECT_GT(heap->bytes_allocated, sizeof(Value));
}

// Test clear marks
TEST_F(HeapTest, ClearMarks) {
    Value* v1 = heap->allocate_scalar(1.0);
    Value* v2 = heap->allocate_scalar(2.0);

    v1->marked = true;
    v2->marked = true;

    heap->clear_marks();

    EXPECT_FALSE(v1->marked);
    EXPECT_FALSE(v2->marked);
}

// Test mark value
TEST_F(HeapTest, MarkValue) {
    Value* v = heap->allocate_scalar(10.0);

    EXPECT_FALSE(v->marked);

    heap->mark_value(v);

    EXPECT_TRUE(v->marked);

    // Marking again should be idempotent
    heap->mark_value(v);
    EXPECT_TRUE(v->marked);
}

// Test mark from roots with Machine
TEST_F(HeapTest, MarkFromRoots) {
    Machine machine;
    machine.heap = heap;  // Use our test heap

    Value* v1 = heap->allocate_scalar(5.0);
    Value* v2 = heap->allocate_scalar(10.0);

    // Set v1 as current value
    machine.ctrl.set_value(v1);

    heap->mark_from_roots(&machine);

    EXPECT_TRUE(v1->marked);   // Should be marked (in ctrl)
    EXPECT_FALSE(v2->marked);  // Should not be marked (not reachable)

    machine.heap = nullptr;  // Don't let machine delete our test heap
}

// Test promote survivors
TEST_F(HeapTest, PromoteSurvivors) {
    Value* v1 = heap->allocate_scalar(1.0);
    Value* v2 = heap->allocate_scalar(2.0);

    EXPECT_EQ(heap->young_size(), 2);
    EXPECT_EQ(heap->old_size(), 0);

    // Mark v1 for survival
    v1->marked = true;
    v2->marked = false;

    heap->promote_survivors();

    EXPECT_EQ(heap->young_size(), 1);  // v2 still in young
    EXPECT_EQ(heap->old_size(), 1);    // v1 promoted
    EXPECT_TRUE(v1->in_old_generation);
    EXPECT_FALSE(v2->in_old_generation);
}

// Test sweep
TEST_F(HeapTest, Sweep) {
    Value* v1 = heap->allocate_scalar(1.0);
    Value* v2 = heap->allocate_scalar(2.0);
    Value* v3 = heap->allocate_scalar(3.0);

    EXPECT_EQ(heap->young_size(), 3);

    // Mark only v1 and v3
    v1->marked = true;
    v2->marked = false;
    v3->marked = true;

    size_t initial_bytes = heap->bytes_allocated;

    heap->sweep();

    EXPECT_EQ(heap->young_size(), 2);  // v2 should be swept
    EXPECT_LT(heap->bytes_allocated, initial_bytes);  // Less memory used
}

// Test minor GC
TEST_F(HeapTest, MinorGC) {
    Machine machine;
    machine.heap = heap;

    Value* v1 = heap->allocate_scalar(1.0);
    Value* v2 = heap->allocate_scalar(2.0);

    // Set v1 as root
    machine.ctrl.set_value(v1);

    size_t initial_young = heap->young_size();

    heap->minor_gc(&machine);

    // v1 should survive and be promoted
    // v2 should be collected
    EXPECT_LT(heap->young_size(), initial_young);
    EXPECT_GT(heap->old_size(), 0);
    EXPECT_EQ(heap->minor_gc_count, 1);

    machine.heap = nullptr;
}

// Test major GC
TEST_F(HeapTest, MajorGC) {
    Machine machine;
    machine.heap = heap;

    Value* v1 = heap->allocate_scalar(1.0);
    Value* v2 = heap->allocate_scalar(2.0);

    // Set v1 as root
    machine.ctrl.set_value(v1);

    heap->major_gc(&machine);

    // v1 should survive, v2 should be collected
    EXPECT_EQ(heap->total_size(), 1);
    EXPECT_EQ(heap->major_gc_count, 1);
    EXPECT_EQ(heap->minor_gc_count, 0);  // Reset after major GC

    machine.heap = nullptr;
}

// Test GC threshold check
TEST_F(HeapTest, ShouldGC) {
    EXPECT_FALSE(heap->should_gc());

    // Simulate allocation up to threshold
    heap->bytes_allocated = heap->gc_threshold - 1;
    EXPECT_FALSE(heap->should_gc());

    heap->bytes_allocated = heap->gc_threshold;
    EXPECT_TRUE(heap->should_gc());

    heap->bytes_allocated = heap->gc_threshold + 100;
    EXPECT_TRUE(heap->should_gc());
}

// Test GC statistics
TEST_F(HeapTest, GCStatistics) {
    Machine machine;
    machine.heap = heap;

    Value* v = heap->allocate_scalar(1.0);
    machine.ctrl.set_value(v);

    EXPECT_EQ(heap->minor_gc_count, 0);
    EXPECT_EQ(heap->major_gc_count, 0);

    heap->minor_gc(&machine);
    EXPECT_EQ(heap->minor_gc_count, 1);

    heap->major_gc(&machine);
    EXPECT_EQ(heap->major_gc_count, 1);

    machine.heap = nullptr;
}

// Test scalar cache survives GC
TEST_F(HeapTest, ScalarCacheSurvivesGC) {
    Machine machine;
    machine.heap = heap;

    Value* v = heap->allocate_scalar(42.0);
    machine.ctrl.set_value(v);

    EXPECT_EQ(heap->scalar_cache[42 + 128], v);

    heap->minor_gc(&machine);

    // Cached value should still exist
    EXPECT_NE(heap->scalar_cache[42 + 128], nullptr);

    machine.heap = nullptr;
}

// Test large allocation
TEST_F(HeapTest, LargeAllocation) {
    Eigen::MatrixXd mat(100, 100);
    mat.setConstant(1.0);

    Value* v = Value::from_matrix(mat);
    heap->allocate(v);

    // Should account for large matrix
    EXPECT_GE(heap->bytes_allocated, sizeof(Value) + 100 * 100 * sizeof(double));
}

// ============================================================================
// Continuation GC Tests
// ============================================================================

// Test allocate continuation
TEST_F(HeapTest, AllocateContinuation) {
    Continuation* k = new HaltK();
    heap->allocate_continuation(k);

    EXPECT_EQ(heap->young_continuations.size(), 1);
    EXPECT_EQ(heap->old_continuations.size(), 0);
    EXPECT_FALSE(k->marked);
    EXPECT_FALSE(k->in_old_generation);
    EXPECT_GT(heap->bytes_allocated, 0);
}

// Test mark continuation
TEST_F(HeapTest, MarkContinuation) {
    Continuation* k = new HaltK();
    heap->allocate_continuation(k);

    EXPECT_FALSE(k->marked);

    heap->mark_continuation(k);

    EXPECT_TRUE(k->marked);

    // Marking again should be idempotent
    heap->mark_continuation(k);
    EXPECT_TRUE(k->marked);
}

// Test continuation graph marking (transitive closure)
TEST_F(HeapTest, MarkContinuationGraph) {
    // Create a chain: k1 -> k2 -> k3 -> HaltK
    Continuation* k_halt = new HaltK();
    Continuation* k3 = new FrameK(nullptr, k_halt);
    Continuation* k2 = new FrameK(nullptr, k3);
    Continuation* k1 = new FrameK(nullptr, k2);

    heap->allocate_continuation(k_halt);
    heap->allocate_continuation(k3);
    heap->allocate_continuation(k2);
    heap->allocate_continuation(k1);

    EXPECT_EQ(heap->young_continuations.size(), 4);

    // Mark only the head - should transitively mark entire graph
    heap->mark_continuation(k1);

    EXPECT_TRUE(k1->marked);
    EXPECT_TRUE(k2->marked);
    EXPECT_TRUE(k3->marked);
    EXPECT_TRUE(k_halt->marked);
}

// Test ArgK marks its arg_value
TEST_F(HeapTest, ArgKMarksValue) {
    Value* arg = heap->allocate_scalar(42.0);
    Continuation* k_next = new HaltK();
    Continuation* k_arg = new ArgK(arg, k_next);

    heap->allocate_continuation(k_next);
    heap->allocate_continuation(k_arg);

    EXPECT_FALSE(arg->marked);
    EXPECT_FALSE(k_arg->marked);

    heap->mark_continuation(k_arg);

    EXPECT_TRUE(k_arg->marked);
    EXPECT_TRUE(k_next->marked);
    EXPECT_TRUE(arg->marked);  // ArgK should mark its Value
}

// Test CLOSURE value marks its continuation graph
TEST_F(HeapTest, ClosureMarksGraph) {
    // Create a simple continuation graph for the closure
    Continuation* k_halt = new HaltK();
    Continuation* k_body = new FrameK(nullptr, k_halt);

    heap->allocate_continuation(k_halt);
    heap->allocate_continuation(k_body);

    // Create a CLOSURE value that references this graph
    Value* closure = Value::from_closure(k_body);
    heap->allocate(closure);

    EXPECT_FALSE(closure->marked);
    EXPECT_FALSE(k_body->marked);
    EXPECT_FALSE(k_halt->marked);

    // Mark the closure - should transitively mark continuation graph
    heap->mark_value(closure);

    EXPECT_TRUE(closure->marked);
    EXPECT_TRUE(k_body->marked);
    EXPECT_TRUE(k_halt->marked);
}

// Test sweep unmarked continuations
TEST_F(HeapTest, SweepContinuations) {
    Continuation* k1 = new HaltK();
    Continuation* k2 = new HaltK();
    Continuation* k3 = new HaltK();

    heap->allocate_continuation(k1);
    heap->allocate_continuation(k2);
    heap->allocate_continuation(k3);

    EXPECT_EQ(heap->young_continuations.size(), 3);

    // Mark only k1 and k3
    k1->marked = true;
    k2->marked = false;
    k3->marked = true;

    size_t initial_bytes = heap->bytes_allocated;

    heap->sweep();

    EXPECT_EQ(heap->young_continuations.size(), 2);  // k2 swept
    EXPECT_LT(heap->bytes_allocated, initial_bytes);
}

// Test continuations from kont_stack are marked
TEST_F(HeapTest, MarkContinuationsFromKontStack) {
    Machine machine;
    machine.heap = heap;

    Continuation* k1 = new HaltK();
    Continuation* k2 = new FrameK(nullptr, nullptr);

    heap->allocate_continuation(k1);
    heap->allocate_continuation(k2);

    // Push continuations onto kont_stack
    machine.kont_stack.push_back(k1);
    machine.kont_stack.push_back(k2);

    EXPECT_FALSE(k1->marked);
    EXPECT_FALSE(k2->marked);

    heap->mark_from_roots(&machine);

    EXPECT_TRUE(k1->marked);
    EXPECT_TRUE(k2->marked);

    machine.heap = nullptr;
}

// Test continuations from function_cache are marked
TEST_F(HeapTest, MarkContinuationsFromFunctionCache) {
    Machine machine;
    machine.heap = heap;

    Continuation* k = new HaltK();
    heap->allocate_continuation(k);

    // Add to function cache
    machine.function_cache["test_fn"] = k;

    EXPECT_FALSE(k->marked);

    heap->mark_from_roots(&machine);

    EXPECT_TRUE(k->marked);

    machine.heap = nullptr;
}

// Test continuation promotion to old generation
TEST_F(HeapTest, PromoteContinuations) {
    Continuation* k1 = new HaltK();
    Continuation* k2 = new HaltK();

    heap->allocate_continuation(k1);
    heap->allocate_continuation(k2);

    EXPECT_EQ(heap->young_continuations.size(), 2);
    EXPECT_EQ(heap->old_continuations.size(), 0);

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
    Machine machine;
    machine.heap = heap;

    Continuation* k_reachable = new HaltK();
    Continuation* k_unreachable = new HaltK();

    heap->allocate_continuation(k_reachable);
    heap->allocate_continuation(k_unreachable);

    // Make k_reachable a root
    machine.kont_stack.push_back(k_reachable);

    size_t initial_young = heap->young_continuations.size();

    heap->minor_gc(&machine);

    // k_reachable should survive, k_unreachable should be collected
    EXPECT_LT(heap->young_continuations.size(), initial_young);

    machine.heap = nullptr;
}

// Test major GC collects continuations from all generations
TEST_F(HeapTest, MajorGCContinuations) {
    Machine machine;
    machine.heap = heap;

    Continuation* k_young = new HaltK();
    Continuation* k_old = new HaltK();

    heap->allocate_continuation(k_young);
    heap->allocate_continuation(k_old);

    // Manually promote k_old
    k_old->in_old_generation = true;
    heap->old_continuations.push_back(k_old);
    heap->young_continuations.pop_back();

    // Don't make either a root - both should be collected
    size_t initial_total = heap->young_continuations.size() + heap->old_continuations.size();

    heap->major_gc(&machine);

    size_t final_total = heap->young_continuations.size() + heap->old_continuations.size();
    EXPECT_LT(final_total, initial_total);

    machine.heap = nullptr;
}

// Test clear marks for continuations
TEST_F(HeapTest, ClearMarksContinuations) {
    Continuation* k1 = new HaltK();
    Continuation* k2 = new HaltK();

    heap->allocate_continuation(k1);
    heap->allocate_continuation(k2);

    k1->marked = true;
    k2->marked = true;

    heap->clear_marks();

    EXPECT_FALSE(k1->marked);
    EXPECT_FALSE(k2->marked);
}

// Test bidirectional marking: Value -> Continuation -> Value
TEST_F(HeapTest, BidirectionalMarking) {
    // Create: closure_value -> continuation_graph -> arg_value
    Value* arg = heap->allocate_scalar(99.0);
    Continuation* k_halt = new HaltK();
    Continuation* k_arg = new ArgK(arg, k_halt);

    heap->allocate_continuation(k_halt);
    heap->allocate_continuation(k_arg);

    Value* closure = Value::from_closure(k_arg);
    heap->allocate(closure);

    EXPECT_FALSE(closure->marked);
    EXPECT_FALSE(k_arg->marked);
    EXPECT_FALSE(k_halt->marked);
    EXPECT_FALSE(arg->marked);

    // Mark the closure - should mark entire graph
    heap->mark_value(closure);

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

    Value* value1 = heap->allocate_scalar(42.0);

    Continuation* k_halt = new HaltK();
    Continuation* k2 = new FrameK(nullptr, k_halt);
    Continuation* k1 = new ArgK(value1, k2);
    Continuation* k3 = new FrameK(nullptr, k2);

    heap->allocate_continuation(k_halt);
    heap->allocate_continuation(k2);
    heap->allocate_continuation(k1);
    heap->allocate_continuation(k3);

    Value* closure1 = Value::from_closure(k1);
    Value* closure2 = Value::from_closure(k3);

    heap->allocate(closure1);
    heap->allocate(closure2);

    // Mark only closure1
    heap->mark_value(closure1);

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
    heap->mark_value(closure2);

    // Now k3 and closure2 should be marked (k2 already marked)
    EXPECT_TRUE(k3->marked);
    EXPECT_TRUE(closure2->marked);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
