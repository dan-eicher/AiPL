// Heap and memory management tests

#include <gtest/gtest.h>
#include "heap.h"
#include "machine.h"
#include "value.h"

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

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
