// Heap implementation

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
Heap::~Heap() {
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
    for (Completion* c : completions) {
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
// NOTE: We do NOT trigger GC here! GC is only safe at the trampoline's maybe_gc()
// call, after each continuation completes. Triggering GC during allocation could
// sweep Values that exist only in C local variables (not yet stored in roots).
Value* Heap::allocate(Value* val) {
    if (!val) return nullptr;

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
Value* Heap::allocate_scalar(double d) {
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
Value* Heap::allocate_vector(const Eigen::VectorXd& v, bool is_char_data) {
    Value* val = new Value();
    val->tag = ValueType::VECTOR;
    val->is_character_data_ = is_char_data;
    // Store vector as n×1 matrix
    val->data.matrix = new Eigen::MatrixXd(v.size(), 1);
    val->data.matrix->col(0) = v;
    return allocate(val);
}

// Allocate a matrix
Value* Heap::allocate_matrix(const Eigen::MatrixXd& m, bool is_char_data) {
    Value* val = new Value();
    val->tag = ValueType::MATRIX;
    val->is_character_data_ = is_char_data;
    val->data.matrix = new Eigen::MatrixXd(m);
    return allocate(val);
}

// Allocate a string value (re-intern to ensure stable pointer)
Value* Heap::allocate_string(const char* s) {
    Value* val = new Value();
    val->tag = ValueType::STRING;
    val->data.string = machine->string_pool.intern(s);
    return allocate(val);
}

// Allocate a primitive function value
Value* Heap::allocate_primitive(PrimitiveFn* fn) {
    Value* val = new Value();
    val->tag = ValueType::PRIMITIVE;
    val->data.primitive_fn = fn;
    return allocate(val);
}

// Allocate an operator value
Value* Heap::allocate_operator(PrimitiveOp* op) {
    Value* val = new Value();
    val->tag = ValueType::OPERATOR;
    val->data.op = op;
    return allocate(val);
}

// Allocate a closure (user-defined function)
Value* Heap::allocate_closure(Continuation* body) {
    Value* val = new Value();
    val->tag = ValueType::CLOSURE;
    val->data.closure = body;
    return allocate(val);
}

// G2 grammar: Allocate a derived operator (result of applying dyadic operator to first operand)
Value* Heap::allocate_derived_operator(PrimitiveOp* op, Value* first_operand) {
    Value* val = new Value();
    val->tag = ValueType::DERIVED_OPERATOR;
    val->data.derived_op = new Value::DerivedOperatorData();
    val->data.derived_op->op = op;
    val->data.derived_op->first_operand = first_operand;
    return allocate(val);
}

// G2 grammar: Allocate a curried function (result of applying function to first argument)
Value* Heap::allocate_curried_fn(Value* fn, Value* first_arg, Value::CurryType curry_type) {
    Value* val = new Value();
    val->tag = ValueType::CURRIED_FN;
    val->data.curried_fn = new Value::CurriedFnData();
    val->data.curried_fn->fn = fn;
    val->data.curried_fn->first_arg = first_arg;
    val->data.curried_fn->curry_type = curry_type;
    return allocate(val);
}

// Allocate a continuation in the heap (private - only called by template allocate)
// NOTE: No GC trigger here - same reason as allocate() above.
Continuation* Heap::allocate_continuation(Continuation* k) {
    if (!k) return nullptr;

    // Add to young generation
    k->marked = false;
    k->in_old_generation = false;
    young_continuations.push_back(k);
    bytes_allocated += sizeof(Continuation);

    return k;
}

// Allocate a completion in the heap (private - only called by template allocate)
// NOTE: No GC trigger here - same reason as allocate() above.
Completion* Heap::allocate_completion(Completion* comp) {
    if (!comp) return nullptr;

    // Add to completions list (no generational separation)
    comp->marked = false;
    comp->in_old_generation = false;  // Always false for completions
    completions.push_back(comp);
    bytes_allocated += sizeof(Completion);

    return comp;
}

// Allocate an environment in the heap (private - only called by template allocate)
// NOTE: No GC trigger here - same reason as allocate() above.
Environment* Heap::allocate_environment(Environment* env) {
    if (!env) return nullptr;

    // Add to environments list (no generational separation - they're GC roots)
    env->marked = false;
    env->in_old_generation = false;  // Always false for environments
    environments.push_back(env);
    bytes_allocated += sizeof(Environment);

    return env;
}

// Trigger appropriate garbage collection
void Heap::collect(Machine* machine) {
    if (gc_in_progress) return;
    if (!machine) return;  // Can't GC without machine - no roots to mark!

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
// Uses partition for O(n) instead of O(n²) from repeated erase()
void Heap::minor_gc(Machine* machine) {
    minor_gc_count++;

    // Clear mark bits
    clear_marks();

    // Mark from roots (includes scalar cache)
    mark_from_roots(machine);

    // Promote survivors to old generation
    promote_survivors();

    // Sweep young generation Values
    auto young_dead = std::partition(young_objects.begin(), young_objects.end(),
        [](Value* val) { return val && val->marked; });
    for (auto it = young_dead; it != young_objects.end(); ++it) {
        Value* val = *it;
        bytes_allocated -= sizeof(Value);
        if (val->is_array() && val->data.matrix) {
            bytes_allocated -= val->data.matrix->size() * sizeof(double);
        }
        delete val;
    }
    young_objects.erase(young_dead, young_objects.end());

    // Update GC threshold if needed
    if (young_objects.size() > young_capacity / 2) {
        young_capacity = (size_t)(young_capacity * 1.5);
        gc_threshold = young_capacity * sizeof(Value);
    }
}

// Major GC - collect all generations
void Heap::major_gc(Machine* machine) {
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
void Heap::mark_from_roots(Machine* machine) {
    if (!machine) return;

    // Mark cached scalars (they're roots - we want to keep common values alive)
    for (int i = 0; i < 256; i++) {
        if (scalar_cache[i]) {
            mark_value(scalar_cache[i]);
        }
    }

    // Mark value in control register
    if (machine->result) {
        mark_value(machine->result);
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
void Heap::mark_value(Value* val) {
    if (!val) return;
    if (val->marked) return;  // Already marked

    val->marked = true;

    // Mark objects this value references (e.g., CLOSURE continuation graphs)
    val->mark(this);
}

// Mark a continuation and its transitive references
void Heap::mark_continuation(Continuation* k) {
    if (!k) return;
    if (k->marked) return;  // Already marked

    k->marked = true;

    // Mark Values and Continuations this continuation references
    k->mark(this);
}

// Mark a completion and its transitive references
void Heap::mark_completion(Completion* comp) {
    if (!comp) return;
    if (comp->marked) return;  // Already marked

    comp->marked = true;

    // Mark values this completion references
    comp->mark(this);
}

// Sweep unmarked objects from both generations
// Uses partition for O(n) instead of O(n²) from repeated erase()
void Heap::sweep() {
    // Sweep young Values
    auto young_dead = std::partition(young_objects.begin(), young_objects.end(),
        [](Value* val) { return val && val->marked; });
    for (auto it = young_dead; it != young_objects.end(); ++it) {
        Value* val = *it;
        bytes_allocated -= sizeof(Value);
        if (val->is_array() && val->data.matrix) {
            bytes_allocated -= val->data.matrix->size() * sizeof(double);
        }
        delete val;
    }
    young_objects.erase(young_dead, young_objects.end());

    // Sweep old Values
    auto old_dead = std::partition(old_objects.begin(), old_objects.end(),
        [](Value* val) { return val && val->marked; });
    for (auto it = old_dead; it != old_objects.end(); ++it) {
        Value* val = *it;
        bytes_allocated -= sizeof(Value);
        if (val->is_array() && val->data.matrix) {
            bytes_allocated -= val->data.matrix->size() * sizeof(double);
        }
        delete val;
    }
    old_objects.erase(old_dead, old_objects.end());

    // Sweep young continuations
    auto young_cont_dead = std::partition(young_continuations.begin(), young_continuations.end(),
        [](Continuation* k) { return k && k->marked; });
    for (auto it = young_cont_dead; it != young_continuations.end(); ++it) {
        bytes_allocated -= sizeof(Continuation);
        delete *it;
    }
    young_continuations.erase(young_cont_dead, young_continuations.end());

    // Sweep old continuations
    auto old_cont_dead = std::partition(old_continuations.begin(), old_continuations.end(),
        [](Continuation* k) { return k && k->marked; });
    for (auto it = old_cont_dead; it != old_continuations.end(); ++it) {
        bytes_allocated -= sizeof(Continuation);
        delete *it;
    }
    old_continuations.erase(old_cont_dead, old_continuations.end());

    // Sweep completions
    auto comp_dead = std::partition(completions.begin(), completions.end(),
        [](Completion* c) { return c && c->marked; });
    for (auto it = comp_dead; it != completions.end(); ++it) {
        bytes_allocated -= sizeof(Completion);
        delete *it;
    }
    completions.erase(comp_dead, completions.end());

    // Sweep environments
    auto env_dead = std::partition(environments.begin(), environments.end(),
        [](Environment* e) { return e && e->marked; });
    for (auto it = env_dead; it != environments.end(); ++it) {
        bytes_allocated -= sizeof(Environment);
        delete *it;
    }
    environments.erase(env_dead, environments.end());
}

// Promote survivors from young to old generation
// Uses partition for O(n) instead of O(n²) from repeated erase()
void Heap::promote_survivors() {
    // Promote Values: partition into [to_promote | stay_young]
    auto promote_vals = std::partition(young_objects.begin(), young_objects.end(),
        [](Value* val) { return !(val && val->marked && !val->in_old_generation); });

    // Move promoted values to old generation
    for (auto it = promote_vals; it != young_objects.end(); ++it) {
        Value* val = *it;
        val->in_old_generation = true;
        old_objects.push_back(val);
    }
    young_objects.erase(promote_vals, young_objects.end());

    // Promote Continuations
    auto promote_conts = std::partition(young_continuations.begin(), young_continuations.end(),
        [](Continuation* k) { return !(k && k->marked && !k->in_old_generation); });

    for (auto it = promote_conts; it != young_continuations.end(); ++it) {
        Continuation* k = *it;
        k->in_old_generation = true;
        old_continuations.push_back(k);
    }
    young_continuations.erase(promote_conts, young_continuations.end());

    // Completions are never promoted (always short-lived)
}

// Clear all mark bits
void Heap::clear_marks() {
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
    for (Completion* c : completions) {
        c->marked = false;
    }
    for (Environment* e : environments) {
        e->marked = false;
    }
}

} // namespace apl
