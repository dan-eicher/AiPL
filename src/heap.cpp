// APLHeap implementation

#include "heap.h"
#include "machine.h"
#include "continuation.h"
#include "completion.h"
#include "environment.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

namespace apl {

// Destructor
APLHeap::~APLHeap() {
    // Clean up all Values
    for (Value* v : young_objects) {
        if (!v) {
            std::cerr << "GC ERROR: nullptr found in young_objects during heap destruction - indicates double-deletion or corruption" << std::endl;
            std::abort();
        }
        delete v;
    }
    for (Value* v : old_objects) {
        if (!v) {
            std::cerr << "GC ERROR: nullptr found in old_objects during heap destruction - indicates double-deletion or corruption" << std::endl;
            std::abort();
        }
        delete v;
    }
    // Scalar cache shares pointers with young/old, so don't double-delete

    // Clean up all Continuations
    for (Continuation* k : young_continuations) {
        if (!k) {
            std::cerr << "GC ERROR: nullptr found in young_continuations during heap destruction - indicates double-deletion or corruption" << std::endl;
            std::abort();
        }
        delete k;
    }
    for (Continuation* k : old_continuations) {
        if (!k) {
            std::cerr << "GC ERROR: nullptr found in old_continuations during heap destruction - indicates double-deletion or corruption" << std::endl;
            std::abort();
        }
        delete k;
    }

    // Clean up all Completions (no old generation)
    for (APLCompletion* c : completions) {
        if (!c) {
            std::cerr << "GC ERROR: nullptr found in completions during heap destruction - indicates double-deletion or corruption" << std::endl;
            std::abort();
        }
        delete c;
    }

    // Clean up all Environments (no old generation)
    for (Environment* e : environments) {
        if (!e) {
            std::cerr << "GC ERROR: nullptr found in environments during heap destruction - indicates double-deletion or corruption" << std::endl;
            std::abort();
        }
        delete e;
    }
}

// Allocate a new Value in the heap
Value* APLHeap::allocate(Value* val) {
    if (!val) return nullptr;

    // Check if GC is needed
    if (should_gc() && !gc_in_progress && machine_) {
        collect(machine_);
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
        Value* val = new Value();
        val->tag = ValueType::SCALAR;
        val->data.scalar = d;
        young_objects.push_back(val);
        bytes_allocated += sizeof(Value);
        scalar_cache[idx] = val;

        return val;
    }

    // Non-cacheable scalar - regular allocation
    Value* val = new Value();
    val->tag = ValueType::SCALAR;
    val->data.scalar = d;
    return allocate(val);
}

// Allocate a vector
Value* APLHeap::allocate_vector(const Eigen::VectorXd& v) {
    Value* val = new Value();
    val->tag = ValueType::VECTOR;
    // Store vector as n×1 matrix
    val->data.matrix = new Eigen::MatrixXd(v.size(), 1);
    val->data.matrix->col(0) = v;
    return allocate(val);
}

// Allocate a matrix
Value* APLHeap::allocate_matrix(const Eigen::MatrixXd& m) {
    Value* val = new Value();
    val->tag = ValueType::MATRIX;
    val->data.matrix = new Eigen::MatrixXd(m);
    return allocate(val);
}

// Allocate a primitive function value
Value* APLHeap::allocate_primitive(PrimitiveFn* fn) {
    Value* val = new Value();
    val->tag = ValueType::PRIMITIVE;
    val->data.primitive_fn = fn;
    return allocate(val);
}

// Allocate an operator value
Value* APLHeap::allocate_operator(PrimitiveOp* op) {
    Value* val = new Value();
    val->tag = ValueType::OPERATOR;
    val->data.op = op;
    return allocate(val);
}

// Allocate a closure (user-defined function)
Value* APLHeap::allocate_closure(Continuation* body) {
    Value* val = new Value();
    val->tag = ValueType::CLOSURE;
    val->data.closure = body;
    return allocate(val);
}

// G2 grammar: Allocate a derived operator (result of applying dyadic operator to first operand)
Value* APLHeap::allocate_derived_operator(PrimitiveOp* op, Value* first_operand) {
    Value* val = new Value();
    val->tag = ValueType::DERIVED_OPERATOR;
    val->data.derived_op = new Value::DerivedOperatorData();
    val->data.derived_op->op = op;
    val->data.derived_op->first_operand = first_operand;
    return allocate(val);
}

// G2 grammar: Allocate a curried function (result of applying function to first argument)
Value* APLHeap::allocate_curried_fn(Value* fn, Value* first_arg, Value::CurryType curry_type) {
    Value* val = new Value();
    val->tag = ValueType::CURRIED_FN;
    val->data.curried_fn = new Value::CurriedFnData();
    val->data.curried_fn->fn = fn;
    val->data.curried_fn->first_arg = first_arg;
    val->data.curried_fn->curry_type = curry_type;
    return allocate(val);
}

// Allocate a continuation in the heap (private - only called by template allocate)
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

// Allocate a completion in the heap (private - only called by template allocate)
APLCompletion* APLHeap::allocate_completion(APLCompletion* comp) {
    if (!comp) return nullptr;

    // Check if GC is needed
    if (should_gc() && !gc_in_progress) {
        collect(nullptr);
    }

    // Add to completions list (no generational separation)
    comp->marked = false;
    comp->in_old_generation = false;  // Always false for completions
    completions.push_back(comp);
    bytes_allocated += sizeof(APLCompletion);

    return comp;
}

// Allocate an environment in the heap (private - only called by template allocate)
Environment* APLHeap::allocate_environment(Environment* env) {
    if (!env) return nullptr;

    // Check if GC is needed
    if (should_gc() && !gc_in_progress) {
        collect(nullptr);
    }

    // Add to environments list (no generational separation - they're GC roots)
    env->marked = false;
    env->in_old_generation = false;  // Always false for environments
    environments.push_back(env);
    bytes_allocated += sizeof(Environment);

    return env;
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
        if (!val) {
            throw std::runtime_error("GC ERROR: nullptr found in young_objects during minor GC - indicates corruption");
        }
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

    // Phase 1: No more completion field in Control
    // Completions will be managed through continuation handlers in Phase 2

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

    // Mark environment (GC root - now GC-managed)
    if (machine->env) {
        if (!machine->env->marked) {
            machine->env->marked = true;
            machine->env->mark(this);  // Will recursively mark parent envs
        }
    }
}

// Mark a value and its transitive references
void APLHeap::mark_value(Value* val) {
    if (!val) return;
    if (val->marked) return;  // Already marked

    val->marked = true;

    // Mark objects this value references (e.g., CLOSURE continuation graphs)
    val->mark(this);
}

// Mark a continuation and its transitive references
void APLHeap::mark_continuation(Continuation* k) {
    if (!k) return;
    if (k->marked) return;  // Already marked

    k->marked = true;

    // Mark Values and Continuations this continuation references
    k->mark(this);
}

// Mark a completion and its transitive references
void APLHeap::mark_completion(APLCompletion* comp) {
    if (!comp) return;
    if (comp->marked) return;  // Already marked

    comp->marked = true;

    // Mark values this completion references
    comp->mark(this);
}

// Sweep unmarked objects from both generations
void APLHeap::sweep() {
    // Sweep young generation
    auto it_young = young_objects.begin();
    while (it_young != young_objects.end()) {
        Value* val = *it_young;
        if (!val) {
            throw std::runtime_error("GC ERROR: nullptr found in young_objects during sweep - indicates corruption");
        }
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
        if (!val) {
            throw std::runtime_error("GC ERROR: nullptr found in old_objects during sweep - indicates corruption");
        }
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
        if (!k) {
            throw std::runtime_error("GC ERROR: nullptr found in young_continuations during sweep - indicates corruption");
        }
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
        if (!k) {
            throw std::runtime_error("GC ERROR: nullptr found in old_continuations during sweep - indicates corruption");
        }
        if (!k->marked) {
            bytes_allocated -= sizeof(Continuation);
            delete k;
            it_old_cont = old_continuations.erase(it_old_cont);
        } else {
            ++it_old_cont;
        }
    }

    // Sweep completions (no generational separation)
    auto it_comp = completions.begin();
    while (it_comp != completions.end()) {
        APLCompletion* c = *it_comp;
        if (!c) {
            throw std::runtime_error("GC ERROR: nullptr found in completions during sweep - indicates corruption");
        }
        if (!c->marked) {
            bytes_allocated -= sizeof(APLCompletion);
            delete c;
            it_comp = completions.erase(it_comp);
        } else {
            ++it_comp;
        }
    }

    // Sweep environments (no generational separation)
    auto it_env = environments.begin();
    while (it_env != environments.end()) {
        Environment* e = *it_env;
        if (!e) {
            throw std::runtime_error("GC ERROR: nullptr found in environments during sweep - indicates corruption");
        }
        if (!e->marked) {
            bytes_allocated -= sizeof(Environment);
            delete e;
            it_env = environments.erase(it_env);
        } else {
            ++it_env;
        }
    }
}

// Promote survivors from young to old generation
void APLHeap::promote_survivors() {
    // Promote Values
    auto it = young_objects.begin();
    while (it != young_objects.end()) {
        Value* val = *it;
        if (!val) {
            throw std::runtime_error("GC ERROR: nullptr found in young_objects during promotion - indicates corruption");
        }
        if (val->marked && !val->in_old_generation) {
            // Promote to old generation
            val->in_old_generation = true;
            old_objects.push_back(val);
            it = young_objects.erase(it);
        } else {
            ++it;
        }
    }

    // Promote Continuations
    auto it_cont = young_continuations.begin();
    while (it_cont != young_continuations.end()) {
        Continuation* k = *it_cont;
        if (!k) {
            throw std::runtime_error("GC ERROR: nullptr found in young_continuations during promotion - indicates corruption");
        }
        if (k->marked && !k->in_old_generation) {
            // Promote to old generation
            k->in_old_generation = true;
            old_continuations.push_back(k);
            it_cont = young_continuations.erase(it_cont);
        } else {
            ++it_cont;
        }
    }

    // Completions are never promoted (always short-lived)
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
    for (APLCompletion* c : completions) {
        c->marked = false;
    }
    for (Environment* e : environments) {
        e->marked = false;
    }
}

} // namespace apl
