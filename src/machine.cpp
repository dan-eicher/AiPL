// Machine implementation - CEK machine execution engine

#include "machine.h"
#include "continuation.h"
#include "completion.h"
#include "parser.h"
#include "primitives.h"
#include "operators.h"
#include <stdexcept>

namespace apl {

// Constructor
Machine::Machine() {
    result = nullptr;
    heap = new Heap();
    heap->set_machine(this);  // Give heap back-pointer for GC
    env = heap->allocate<Environment>();  // Global environment (GC-managed)
    parser = new Parser(this);  // Parser owned by machine
    init_globals();  // Populate with APL primitives and operators
}

// Initialize global environment with all built-in primitives and operators
void Machine::init_globals() {
    // Arithmetic primitives
    env->define("+", heap->allocate_primitive(&prim_plus));
    env->define("-", heap->allocate_primitive(&prim_minus));
    env->define("×", heap->allocate_primitive(&prim_times));
    env->define("÷", heap->allocate_primitive(&prim_divide));
    env->define("*", heap->allocate_primitive(&prim_star));
    env->define("=", heap->allocate_primitive(&prim_equal));
    env->define("≠", heap->allocate_primitive(&prim_not_equal));
    env->define("<", heap->allocate_primitive(&prim_less));
    env->define(">", heap->allocate_primitive(&prim_greater));
    env->define("≤", heap->allocate_primitive(&prim_less_eq));
    env->define("≥", heap->allocate_primitive(&prim_greater_eq));
    env->define("⌈", heap->allocate_primitive(&prim_ceiling));
    env->define("⌊", heap->allocate_primitive(&prim_floor));
    env->define("∧", heap->allocate_primitive(&prim_and));
    env->define("∨", heap->allocate_primitive(&prim_or));
    env->define("~", heap->allocate_primitive(&prim_not));
    env->define("⍲", heap->allocate_primitive(&prim_nand));
    env->define("⍱", heap->allocate_primitive(&prim_nor));
    env->define("|", heap->allocate_primitive(&prim_stile));
    env->define("⍟", heap->allocate_primitive(&prim_log));
    env->define("!", heap->allocate_primitive(&prim_factorial));

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

// Destructor
Machine::~Machine() {
    // Delete parser first (it doesn't own anything, just references machine)
    delete parser;

    // Clear continuation references (heap will delete them)
    kont_stack.clear();
    function_cache.clear();

    // Environment is GC-managed, heap will delete it
    // Clean up heap (deletes all GC objects: Values, Continuations, Completions, Environments)
    delete heap;
}

// High-level eval: parse and execute an expression
Value* Machine::eval(const std::string& input) {
    Continuation* k = parser->parse(input);
    if (!k) {
        // Parse error - route through the same error mechanism as runtime errors
        const char* msg = string_pool.intern(parser->get_error().c_str());
        k = heap->allocate<ThrowErrorK>(msg);
    }
    push_kont(k);
    return execute();
}

// Execute the machine until halt
// This is the main trampoline loop that drives the CEK machine
// Phase 3.3: Pure trampoline - just pop and invoke until stack empty
Value* Machine::execute() {
    while (!kont_stack.empty()) {
        // Pop next continuation from stack
        Continuation* k = kont_stack.back();
        kont_stack.pop_back();

        // Phase 3.1: invoke() now returns void
        k->invoke(this);

        // Check for GC periodically
        maybe_gc();
    }

    // Stack empty - finalize curried functions at top level
    // Per paper: "At the top level of an expression, y can also be null"
    // g' semantics: if null(y) then g1(x) else if bas(y) then g2(x,y) else y(g1(x))
    // When we reach top level with a curried function, y is null, so apply monadically
    if (result && result->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* curried_data = result->data.curried_fn;
        if (curried_data->curry_type == Value::CurryType::G_PRIME) {
            // This is a g' transformation curried function - finalize it
            // Apply monadically: g1(x) where x is first_arg
            Value* fn = curried_data->fn;
            Value* arg = curried_data->first_arg;

            if (fn->is_primitive()) {
                PrimitiveFn* prim_fn = fn->data.primitive_fn;
                if (prim_fn->monadic) {
                    prim_fn->monadic(this, arg);
                    // Result is now in result
                }
            }
        } else if (curried_data->curry_type == Value::CurryType::DYADIC_CURRY) {
            // Check if this is an operator-derived curry (f⍤k B) at top level
            // In that case, apply monadically since no left argument
            Value* inner_fn = curried_data->fn;
            Value* right_arr = curried_data->first_arg;

            if (inner_fn->tag == ValueType::CURRIED_FN) {
                Value::CurriedFnData* inner_curried = inner_fn->data.curried_fn;
                if (inner_curried->curry_type == Value::CurryType::OPERATOR_CURRY) {
                    // This is (f⍤k) B at top level - apply monadically
                    Value* derived_op = inner_curried->fn;
                    Value* second_operand = inner_curried->first_arg;

                    if (derived_op->tag == ValueType::DERIVED_OPERATOR) {
                        Value::DerivedOperatorData* derived_data = derived_op->data.derived_op;
                        PrimitiveOp* op = derived_data->op;
                        Value* first_operand = derived_data->first_operand;

                        if (op->dyadic) {
                            // Apply monadically: lhs=nullptr
                            op->dyadic(this, nullptr, first_operand, second_operand, right_arr);
                            // Continue running to get the result
                            while (!kont_stack.empty()) {
                                Continuation* k = kont_stack.back();
                                kont_stack.pop_back();
                                k->invoke(this);
                                maybe_gc();
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

// Phase 2 complete: Completion handling now done through continuations
// - PropagateCompletionK handles stack unwinding
// - CatchReturnK/CatchBreakK handle boundaries
// - No imperative completion handling needed

} // namespace apl
