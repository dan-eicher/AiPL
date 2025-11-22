// APLHeap implementation

#include "heap.h"
#include "machine.h"
#include "continuation.h"
#include <algorithm>

namespace apl {

// Destructor
APLHeap::~APLHeap() {
    // Clean up all Values
    for (Value* v : young_objects) {
        delete v;
    }
    for (Value* v : old_objects) {
        delete v;
    }
    // Scalar cache shares pointers with young/old, so don't double-delete

    // Clean up all Continuations
    for (Continuation* k : young_continuations) {
        delete k;
    }
    for (Continuation* k : old_continuations) {
        delete k;
    }
}

// Allocate a new Value in the heap
Value* APLHeap::allocate(Value* val) {
    if (!val) return nullptr;

    // Check if GC is needed
    if (should_gc() && !gc_in_progress) {
        collect(nullptr);  // Will implement with Machine later
    }

    // Add to young generation
    young_objects.push_back(val);
    bytes_allocated += sizeof(Value);

    // For arrays, count matrix storage
    if (val->is_array() && val->data.matrix) {
        bytes_allocated += val->data.matrix->size() * sizeof(double);
    }

    return val;
}

// Allocate a scalar with cache checking
Value* APLHeap::allocate_scalar(double d) {
    // Check if this is a cacheable scalar
    if (d >= -128.0 && d <= 127.0 && d == (int)d) {
        int idx = (int)d + 128;

        // Check cache
        if (scalar_cache[idx] != nullptr) {
            return scalar_cache[idx];
        }

        // Create and cache
        Value* val = Value::from_scalar(d);
        young_objects.push_back(val);
        bytes_allocated += sizeof(Value);
        scalar_cache[idx] = val;

        return val;
    }

    // Non-cacheable scalar - regular allocation
    Value* val = Value::from_scalar(d);
    return allocate(val);
}

// Allocate a continuation in the heap
Continuation* APLHeap::allocate_continuation(Continuation* k) {
    if (!k) return nullptr;

    // Check if GC is needed
    if (should_gc() && !gc_in_progress) {
        collect(nullptr);
    }

    // Add to young generation
    k->marked = false;
    k->in_old_generation = false;
    young_continuations.push_back(k);
    bytes_allocated += sizeof(Continuation);

    return k;
}

// Trigger appropriate garbage collection
void APLHeap::collect(Machine* machine) {
    if (gc_in_progress) return;

    gc_in_progress = true;

    // Decide between minor and major GC
    // Major GC every 10 minor GCs or if old generation is > 75% full
    bool need_major = (minor_gc_count >= 10) ||
                      (old_objects.size() > old_capacity * 3 / 4);

    if (need_major) {
        major_gc(machine);
    } else {
        minor_gc(machine);
    }

    gc_in_progress = false;
}

// Minor GC - collect young generation only
void APLHeap::minor_gc(Machine* machine) {
    minor_gc_count++;

    // Clear mark bits
    clear_marks();

    // Mark from roots
    mark_from_roots(machine);

    // Promote survivors to old generation
    promote_survivors();

    // Sweep young generation
    auto it = young_objects.begin();
    while (it != young_objects.end()) {
        Value* val = *it;
        if (!val->marked) {
            // Unmarked - reclaim
            bytes_allocated -= sizeof(Value);
            if (val->is_array() && val->data.matrix) {
                bytes_allocated -= val->data.matrix->size() * sizeof(double);
            }

            // Remove from cache if present
            if (val->is_scalar()) {
                double d = val->data.scalar;
                if (d >= -128.0 && d <= 127.0 && d == (int)d) {
                    int idx = (int)d + 128;
                    if (scalar_cache[idx] == val) {
                        scalar_cache[idx] = nullptr;
                    }
                }
            }

            delete val;
            it = young_objects.erase(it);
        } else {
            ++it;
        }
    }

    // Update GC threshold if needed
    if (young_objects.size() > young_capacity / 2) {
        young_capacity = (size_t)(young_capacity * 1.5);
        gc_threshold = young_capacity * sizeof(Value);
    }
}

// Major GC - collect all generations
void APLHeap::major_gc(Machine* machine) {
    major_gc_count++;
    minor_gc_count = 0;  // Reset minor GC counter

    // Clear mark bits
    clear_marks();

    // Mark from roots
    mark_from_roots(machine);

    // Sweep both generations
    sweep();

    // Update capacities if needed
    size_t total_live = total_size();
    if (total_live > young_capacity / 2) {
        young_capacity = (size_t)(young_capacity * 1.5);
    }
    if (old_objects.size() > old_capacity / 2) {
        old_capacity = (size_t)(old_capacity * 1.5);
    }

    gc_threshold = young_capacity * sizeof(Value);
}

// Mark from root set (Machine registers and stacks)
void APLHeap::mark_from_roots(Machine* machine) {
    if (!machine) return;

    // Mark value in control register
    if (machine->ctrl.value) {
        mark_value(machine->ctrl.value);
    }

    // Mark values in completion record
    if (machine->ctrl.completion && machine->ctrl.completion->value) {
        mark_value(machine->ctrl.completion->value);
    }

    // Mark continuations on kont_stack (they're GC objects now!)
    for (Continuation* k : machine->kont_stack) {
        if (k) {
            mark_continuation(k);
        }
    }

    // Mark continuations in function_cache
    for (auto& pair : machine->function_cache) {
        if (pair.second) {
            mark_continuation(pair.second);
        }
    }

    // Mark values in environment
    if (machine->env) {
        machine->env->mark(this);
    }
}

// Mark a value and its transitive references
void APLHeap::mark_value(Value* val) {
    if (!val) return;
    if (val->marked) return;  // Already marked

    val->marked = true;

    // Mark objects this value references (e.g., CLOSURE continuation graphs)
    val->mark_references(this);
}

// Mark a continuation and its transitive references
void APLHeap::mark_continuation(Continuation* k) {
    if (!k) return;
    if (k->marked) return;  // Already marked

    k->marked = true;

    // Mark Values and Continuations this continuation references
    k->mark(this);
}

// Sweep unmarked objects from both generations
void APLHeap::sweep() {
    // Sweep young generation
    auto it_young = young_objects.begin();
    while (it_young != young_objects.end()) {
        Value* val = *it_young;
        if (!val->marked) {
            bytes_allocated -= sizeof(Value);
            if (val->is_array() && val->data.matrix) {
                bytes_allocated -= val->data.matrix->size() * sizeof(double);
            }

            // Remove from cache
            if (val->is_scalar()) {
                double d = val->data.scalar;
                if (d >= -128.0 && d <= 127.0 && d == (int)d) {
                    int idx = (int)d + 128;
                    if (scalar_cache[idx] == val) {
                        scalar_cache[idx] = nullptr;
                    }
                }
            }

            delete val;
            it_young = young_objects.erase(it_young);
        } else {
            ++it_young;
        }
    }

    // Sweep old generation
    auto it_old = old_objects.begin();
    while (it_old != old_objects.end()) {
        Value* val = *it_old;
        if (!val->marked) {
            bytes_allocated -= sizeof(Value);
            if (val->is_array() && val->data.matrix) {
                bytes_allocated -= val->data.matrix->size() * sizeof(double);
            }

            delete val;
            it_old = old_objects.erase(it_old);
        } else {
            ++it_old;
        }
    }

    // Sweep young continuations
    auto it_young_cont = young_continuations.begin();
    while (it_young_cont != young_continuations.end()) {
        Continuation* k = *it_young_cont;
        if (!k->marked) {
            bytes_allocated -= sizeof(Continuation);
            delete k;
            it_young_cont = young_continuations.erase(it_young_cont);
        } else {
            ++it_young_cont;
        }
    }

    // Sweep old continuations
    auto it_old_cont = old_continuations.begin();
    while (it_old_cont != old_continuations.end()) {
        Continuation* k = *it_old_cont;
        if (!k->marked) {
            bytes_allocated -= sizeof(Continuation);
            delete k;
            it_old_cont = old_continuations.erase(it_old_cont);
        } else {
            ++it_old_cont;
        }
    }
}

// Promote survivors from young to old generation
void APLHeap::promote_survivors() {
    auto it = young_objects.begin();
    while (it != young_objects.end()) {
        Value* val = *it;
        if (val->marked && !val->in_old_generation) {
            // Promote to old generation
            val->in_old_generation = true;
            old_objects.push_back(val);
            it = young_objects.erase(it);
        } else {
            ++it;
        }
    }
}

// Clear all mark bits
void APLHeap::clear_marks() {
    for (Value* val : young_objects) {
        val->marked = false;
    }
    for (Value* val : old_objects) {
        val->marked = false;
    }
    for (Continuation* k : young_continuations) {
        k->marked = false;
    }
    for (Continuation* k : old_continuations) {
        k->marked = false;
    }
}

} // namespace apl
