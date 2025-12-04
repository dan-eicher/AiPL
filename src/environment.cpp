// Environment implementation

#include "environment.h"
#include "heap.h"
#include "primitives.h"
#include "operators.h"
#include "machine.h"

namespace apl {

void Environment::mark(APLHeap* heap) {
    if (!heap) return;

    // Mark all values in this environment
    for (auto& pair : bindings) {
        if (pair.second) {
            heap->mark_value(pair.second);
        }
    }

    // Mark parent environment (now GC-managed)
    if (parent && !parent->marked) {
        parent->marked = true;
        parent->mark(heap);
    }
}

// Initialize global environment with all built-in primitives
void init_global_environment(Machine* machine) {
    Environment* env = machine->env;
    APLHeap* heap = machine->heap;

    // Arithmetic primitives
    env->define("+", heap->allocate_primitive(&prim_plus));
    env->define("-", heap->allocate_primitive(&prim_minus));
    env->define("×", heap->allocate_primitive(&prim_times));
    env->define("÷", heap->allocate_primitive(&prim_divide));
    env->define("*", heap->allocate_primitive(&prim_star));
    env->define("=", heap->allocate_primitive(&prim_equal));

    // Array operations
    env->define("⍴", heap->allocate_primitive(&prim_rho));
    env->define(",", heap->allocate_primitive(&prim_comma));
    env->define("⍉", heap->allocate_primitive(&prim_transpose));
    env->define("⍳", heap->allocate_primitive(&prim_iota));
    env->define("↑", heap->allocate_primitive(&prim_uptack));
    env->define("↓", heap->allocate_primitive(&prim_downtack));

    // Operators (higher-order functions)
    env->define(".", heap->allocate_operator(&op_dot));
    env->define("∘.", heap->allocate_operator(&op_outer_dot));
    env->define("¨", heap->allocate_operator(&op_diaeresis));
    env->define("⍨", heap->allocate_operator(&op_tilde));
    env->define("/", heap->allocate_operator(&op_reduce));
    env->define("⌿", heap->allocate_operator(&op_reduce_first));
    env->define("\\", heap->allocate_operator(&op_scan));
    env->define("⍀", heap->allocate_operator(&op_scan_first));
    env->define("⍤", heap->allocate_operator(&op_rank_op));
}

} // namespace apl
