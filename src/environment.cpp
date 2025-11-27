// Environment implementation

#include "environment.h"
#include "heap.h"
#include "primitives.h"

namespace apl {

void Environment::mark(APLHeap* heap) {
    if (!heap) return;

    // Mark all values in this environment
    for (auto& pair : bindings) {
        if (pair.second) {
            heap->mark_value(pair.second);
        }
    }

    // Mark parent environment's values
    if (parent) {
        parent->mark(heap);
    }
}

// Initialize global environment with all built-in primitives
void init_global_environment(Environment* env) {
    // Arithmetic primitives
    env->define("+", Value::from_primitive(&prim_plus));
    env->define("-", Value::from_primitive(&prim_minus));
    env->define("×", Value::from_primitive(&prim_times));
    env->define("÷", Value::from_primitive(&prim_divide));
    env->define("*", Value::from_primitive(&prim_star));
    env->define("=", Value::from_primitive(&prim_equal));

    // Array operations
    env->define("⍴", Value::from_primitive(&prim_rho));
    env->define(",", Value::from_primitive(&prim_comma));
    env->define("⍉", Value::from_primitive(&prim_transpose));
    env->define("⍳", Value::from_primitive(&prim_iota));
    env->define("↑", Value::from_primitive(&prim_uptack));
    env->define("↓", Value::from_primitive(&prim_downtack));

    // Note: Reduction/scan operators will be added in Phase 5 when we
    // implement proper operator support. For now they're just functions
    // that take a function argument.
}

} // namespace apl
