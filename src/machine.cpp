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
    // Seed RNG from system random device (standard APL behavior)
    std::random_device rd;
    rl = rd();
    if (rl == 0) rl = 1;  // Ensure positive
    rng.seed(rl);
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
    env->define("⌽", heap->allocate_primitive(&prim_reverse));
    env->define("⊖", heap->allocate_primitive(&prim_reverse_first));
    env->define("≢", heap->allocate_primitive(&prim_tally));
    env->define("≡", heap->allocate_primitive(&prim_depth));
    env->define("∊", heap->allocate_primitive(&prim_member));
    env->define("⍋", heap->allocate_primitive(&prim_grade_up));
    env->define("⍒", heap->allocate_primitive(&prim_grade_down));
    env->define("∪", heap->allocate_primitive(&prim_union));
    env->define("○", heap->allocate_primitive(&prim_circle));
    env->define("?", heap->allocate_primitive(&prim_question));
    env->define("⊥", heap->allocate_primitive(&prim_decode));
    env->define("⊤", heap->allocate_primitive(&prim_encode));
    env->define("⌹", heap->allocate_primitive(&prim_domino));
    env->define("⍎", heap->allocate_primitive(&prim_execute));
    env->define("⍕", heap->allocate_primitive(&prim_format));
    env->define("⌷", heap->allocate_primitive(&prim_squad));
    env->define("⍪", heap->allocate_primitive(&prim_table));
    env->define("⊣", heap->allocate_primitive(&prim_left));
    env->define("⊢", heap->allocate_primitive(&prim_right));

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
    env->define(",⌷", heap->allocate_operator(&op_catenate_axis));
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
// Main trampoline loop - pure continuation-based execution
Value* Machine::execute() {
    // Main evaluation loop
    while (!kont_stack.empty()) {
        Continuation* k = kont_stack.back();
        kont_stack.pop_back();

        if (!k) {
            throw std::runtime_error("VM BUG: null continuation on stack");
        }

        k->invoke(this);
        maybe_gc();
    }

    // Finalize curries at top level via continuation graph
    while (result && result->tag == ValueType::CURRIED_FN) {
        Value* result_before = result;

        push_kont(heap->allocate<PerformFinalizeK>());

        // Run the finalization
        while (!kont_stack.empty()) {
            Continuation* k = kont_stack.back();
            kont_stack.pop_back();
            k->invoke(this);
            maybe_gc();
        }

        // If result unchanged, curry can't be finalized (valid partial application)
        if (result == result_before) {
            break;
        }
    }

    return result;
}

// Phase 2 complete: Completion handling now done through continuations
// - PropagateCompletionK handles stack unwinding
// - CatchReturnK/CatchBreakK handle boundaries
// - No imperative completion handling needed

} // namespace apl
