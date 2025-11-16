// APLHeap - Generational garbage-collected heap for APL values

#pragma once

#include "value.h"
#include <vector>
#include <cstddef>

namespace apl {

// Forward declaration
class Machine;

// APLHeap - Generational heap with specialized zones
class APLHeap {
public:
    // Generational zones
    std::vector<Value*> young_objects;      // Short-lived allocations
    std::vector<Value*> old_objects;        // Long-lived objects

    // Scalar cache for common values (-128 to 127)
    Value* scalar_cache[256];

    // GC statistics
    size_t bytes_allocated;
    size_t gc_threshold;
    size_t minor_gc_count;
    size_t major_gc_count;
    bool gc_in_progress;

    // Capacity limits
    size_t young_capacity;
    size_t old_capacity;

    // Constructor
    APLHeap(size_t young_cap = 4096, size_t old_cap = 16384)
        : bytes_allocated(0),
          gc_threshold(young_cap * sizeof(Value)),
          minor_gc_count(0),
          major_gc_count(0),
          gc_in_progress(false),
          young_capacity(young_cap),
          old_capacity(old_cap) {

        young_objects.reserve(young_capacity);
        old_objects.reserve(old_capacity);

        // Initialize scalar cache to nullptr
        for (int i = 0; i < 256; i++) {
            scalar_cache[i] = nullptr;
        }
    }

    // Destructor
    ~APLHeap() {
        // Clean up all objects
        for (Value* v : young_objects) {
            delete v;
        }
        for (Value* v : old_objects) {
            delete v;
        }
        // Scalar cache shares pointers with young/old, so don't double-delete
    }

    // Allocate a new Value in the heap
    Value* allocate(Value* val);

    // Allocate a scalar (with cache checking)
    Value* allocate_scalar(double d);

    // Garbage collection
    void minor_gc(Machine* machine);    // Collect young generation
    void major_gc(Machine* machine);    // Collect all generations
    void collect(Machine* machine);     // Trigger appropriate GC

    // Mark phase - mark all reachable values
    void mark_from_roots(Machine* machine);
    void mark_value(Value* val);

    // Sweep phase - reclaim unmarked values
    void sweep();

    // Promote objects from young to old generation
    void promote_survivors();

    // Check if GC should be triggered
    bool should_gc() const {
        return bytes_allocated >= gc_threshold;
    }

    // Statistics
    size_t young_size() const { return young_objects.size(); }
    size_t old_size() const { return old_objects.size(); }
    size_t total_size() const { return young_size() + old_size(); }

    // Clear mark bits (public for testing)
    void clear_marks();

private:
    // Add to scalar cache
    void cache_scalar(Value* val);
};

} // namespace apl
