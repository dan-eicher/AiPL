// Machine implementation - CEK machine execution engine

#include "machine.h"
#include "continuation.h"
#include "completion.h"
#include "parser.h"
#include "primitives.h"
#include "operators.h"
#include "kont_print.h"
#include <stdexcept>
#include <sstream>

namespace apl {

// Constructor
Machine::Machine() {
    control = nullptr;
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
    env->define("⊂", heap->allocate_primitive(&prim_enclose));
    env->define("⊃", heap->allocate_primitive(&prim_disclose));

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
        throw_error(parser->get_error().c_str());
    } else {
        push_kont(k);
    }
    return execute();
}

// Execute the machine until halt
// Main trampoline loop - pure continuation-based execution
Value* Machine::execute() {
    // Main evaluation loop
    while (!kont_stack.empty()) {
        Continuation* prev = control;
        control = kont_stack.back();
        kont_stack.pop_back();

        if (!control) {
            throw std::runtime_error("VM BUG: null continuation on stack");
        }

        // Propagate source location from parser continuations to runtime continuations
        if (!control->has_location() && prev && prev->has_location()) {
            control->set_location(prev->line(), prev->column());
        }

        control->invoke(this);
        maybe_gc();
    }
    control = nullptr;

    // Finalize curries at top level via continuation graph
    while (result && result->tag == ValueType::CURRIED_FN) {
        Value* result_before = result;

        push_kont(heap->allocate<PerformFinalizeK>());

        // Run the finalization
        while (!kont_stack.empty()) {
            Continuation* prev = control;
            control = kont_stack.back();
            kont_stack.pop_back();

            // Propagate source location here too
            if (!control->has_location() && prev && prev->has_location()) {
                control->set_location(prev->line(), prev->column());
            }

            control->invoke(this);
            maybe_gc();
        }
        control = nullptr;

        // If result unchanged, curry can't be finalized (valid partial application)
        if (result == result_before) {
            break;
        }
    }

    // Auto-invoke niladic closures at top level
    // A niladic function (one that doesn't reference ⍵) should execute when referenced
    while (result && result->tag == ValueType::CLOSURE &&
           result->data.closure && result->data.closure->is_niladic) {
        Value* result_before = result;

        // Invoke the niladic closure with no arguments
        // Also finalize any CURRIED_FN that results (e.g., {-5} returns G_PRIME curry)
        push_kont(heap->allocate<PerformFinalizeK>());
        push_kont(heap->allocate<FunctionCallK>(result, nullptr, nullptr));

        // Run the invocation
        while (!kont_stack.empty()) {
            Continuation* prev = control;
            control = kont_stack.back();
            kont_stack.pop_back();

            if (!control->has_location() && prev && prev->has_location()) {
                control->set_location(prev->line(), prev->column());
            }

            control->invoke(this);
            maybe_gc();
        }
        control = nullptr;

        // If result unchanged (shouldn't happen for niladic), break to avoid infinite loop
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

// Format the error stack trace for display
std::string Machine::format_stack_trace() const {
    if (error_stack.empty()) {
        return "";
    }

    ContinuationPrinter printer;
    return "Stack trace (most recent first):\n" + printer.print_stack(error_stack);
}

// Throw an error: captures stack trace, creates ThrowErrorK, and pushes it
void Machine::throw_error(const char* msg, Continuation* source) {
    // Use control (the currently executing continuation) as fallback
    // This allows primitives to get location info from their calling continuation
    Continuation* error_source = source ? source : control;

    // Capture the current stack for error traces
    error_stack = kont_stack;

    // If we have a source, add it to the error stack (it has the error location)
    if (error_source) {
        error_stack.push_back(error_source);
    }

    // Create and push ThrowErrorK
    const char* interned_msg = string_pool.intern(msg);
    ThrowErrorK* err = heap->allocate<ThrowErrorK>(interned_msg);

    // Copy source location to ThrowErrorK if available
    if (error_source && error_source->has_location()) {
        err->set_location(error_source->line(), error_source->column());
    }

    push_kont(err);
}

} // namespace apl
