// Heap - Generational garbage-collected heap for APL values

#pragma once

#include "value.h"
#include <vector>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace apl {

// Forward declarations
class Machine;
class Continuation;
class Completion;
class Environment;

// Heap - Generational heap with specialized zones
class Heap {
private:
    Machine* machine;  // Back-pointer to owning machine (for GC)

public:
    // Generational zones for Values
    std::vector<Value*> young_objects;      // Short-lived allocations
    std::vector<Value*> old_objects;        // Long-lived objects

    // Generational zones for Continuations
    std::vector<Continuation*> young_continuations;  // Short-lived continuations
    std::vector<Continuation*> old_continuations;    // Long-lived continuations

    // Completions (always short-lived, no old generation needed)
    std::vector<Completion*> completions;

    // Environments (GC roots, no generational separation needed)
    std::vector<Environment*> environments;

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
    Heap(size_t young_cap = 4096, size_t old_cap = 16384)
        : machine(nullptr),
          bytes_allocated(0),
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

    // Set the owning machine (called by Machine constructor)
    void set_machine(Machine* m) { machine = m; }

    // Destructor (implemented in .cpp to avoid incomplete type issues)
    ~Heap();

    // --- Public Allocation API ---

    // Allocate a scalar (with cache checking)
    Value* allocate_scalar(double d);

    // Convenient Value allocation helpers
    Value* allocate_vector(const Eigen::VectorXd& v, bool is_char_data = false);
    Value* allocate_matrix(const Eigen::MatrixXd& m, bool is_char_data = false);
    Value* allocate_string(const char* s);  // s must be interned (stable pointer)
    Value* allocate_primitive(PrimitiveFn* fn);
    Value* allocate_operator(PrimitiveOp* op);
    Value* allocate_closure(Continuation* body, bool is_niladic = false);

    // Defined operator allocation
    Value* allocate_defined_operator(Value::DefinedOperatorData* op_data);

    // Strand allocation (nested arrays)
    Value* allocate_strand(const std::vector<Value*>& elements);
    Value* allocate_strand(std::vector<Value*>&& elements);
    Value* allocate_empty_strand();

    // NDARRAY allocation (N-dimensional arrays, rank 3+)
    // Shape vector defines dimensions; strides are computed automatically (row-major)
    Value* allocate_ndarray(const Eigen::VectorXd& data, const std::vector<int>& shape);
    Value* allocate_ndarray(Eigen::VectorXd&& data, std::vector<int>&& shape);

    // G2 grammar allocation helpers
    Value* allocate_derived_operator(PrimitiveOp* op, Value* first_operand);
    Value* allocate_derived_operator(Value::DefinedOperatorData* op, Value* first_operand, Value* operator_value = nullptr);
    Value* allocate_curried_fn(Value* fn, Value* first_arg, Value::CurryType curry_type, Value* axis = nullptr);

    // Template-based allocation interface (unified allocation)
    template<typename T, typename... Args>
    T* allocate(Args&&... args) {
        T* obj = new T(std::forward<Args>(args)...);

        // Register with appropriate GC list based on type
        if constexpr (std::is_base_of<Value, T>::value) {
            return static_cast<T*>(allocate(static_cast<Value*>(obj)));
        } else if constexpr (std::is_base_of<Continuation, T>::value) {
            return static_cast<T*>(allocate_continuation(static_cast<Continuation*>(obj)));
        } else if constexpr (std::is_base_of<Completion, T>::value) {
            return static_cast<T*>(allocate_completion(static_cast<Completion*>(obj)));
        } else if constexpr (std::is_base_of<Environment, T>::value) {
            return static_cast<T*>(allocate_environment(static_cast<Environment*>(obj)));
        } else {
            // Unknown type - just return it (will cause errors if not intended)
            return obj;
        }
    }

    // Garbage collection
    void minor_gc(Machine* machine);    // Collect young generation
    void major_gc(Machine* machine);    // Collect all generations
    void collect(Machine* machine);     // Trigger appropriate GC

    // Mark phase - mark all reachable objects
    void mark_from_roots(Machine* machine);
    void mark(GCObject* obj);  // Unified mark for any GC object

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
    // --- Internal Allocation Helpers ---

    // Allocate a new Value in the heap (called by template allocate() only)
    Value* allocate(Value* val);

    // Internal allocation helpers (called by template allocate() only)
    Continuation* allocate_continuation(Continuation* k);
    Completion* allocate_completion(Completion* comp);
    Environment* allocate_environment(Environment* env);

    // Add to scalar cache
    void cache_scalar(Value* val);
};

} // namespace apl
