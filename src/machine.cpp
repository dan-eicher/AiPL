// Machine implementation - CEK machine execution engine

#include "machine.h"
#include "continuation.h"
#include "completion.h"
#include "parser.h"
#include "primitives.h"
#include "operators.h"
#include "optimizer.h"
#include "dir.h"
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
    dir_backend = new CloningBackend();
    init_globals();  // Populate with APL primitives and operators
}

// Initialize global environment with all built-in primitives and operators
void Machine::init_globals() {
    // Helper lambda to intern and define
    auto def = [this](const char* name, Value* val) {
        env->define(string_pool.intern(name), val);
    };

    // Arithmetic primitives
    def("+", heap->allocate_primitive(&prim_plus));
    def("-", heap->allocate_primitive(&prim_minus));
    def("×", heap->allocate_primitive(&prim_times));
    def("÷", heap->allocate_primitive(&prim_divide));
    def("*", heap->allocate_primitive(&prim_star));
    def("=", heap->allocate_primitive(&prim_equal));
    def("≠", heap->allocate_primitive(&prim_not_equal));
    def("<", heap->allocate_primitive(&prim_less));
    def(">", heap->allocate_primitive(&prim_greater));
    def("≤", heap->allocate_primitive(&prim_less_eq));
    def("≥", heap->allocate_primitive(&prim_greater_eq));
    def("⌈", heap->allocate_primitive(&prim_ceiling));
    def("⌊", heap->allocate_primitive(&prim_floor));
    def("∧", heap->allocate_primitive(&prim_and));
    def("∨", heap->allocate_primitive(&prim_or));
    def("~", heap->allocate_primitive(&prim_not));
    def("⍲", heap->allocate_primitive(&prim_nand));
    def("⍱", heap->allocate_primitive(&prim_nor));
    def("|", heap->allocate_primitive(&prim_stile));
    def("⍟", heap->allocate_primitive(&prim_log));
    def("!", heap->allocate_primitive(&prim_factorial));

    // Array operations
    def("⍴", heap->allocate_primitive(&prim_rho));
    def(",", heap->allocate_primitive(&prim_comma));
    def("⍉", heap->allocate_primitive(&prim_transpose));
    def("⍳", heap->allocate_primitive(&prim_iota));
    def("↑", heap->allocate_primitive(&prim_uptack));
    def("↓", heap->allocate_primitive(&prim_downtack));
    def("⌽", heap->allocate_primitive(&prim_reverse));
    def("⊖", heap->allocate_primitive(&prim_reverse_first));
    def("≢", heap->allocate_primitive(&prim_tally));
    def("≡", heap->allocate_primitive(&prim_depth));
    def("∊", heap->allocate_primitive(&prim_member));
    def("⍋", heap->allocate_primitive(&prim_grade_up));
    def("⍒", heap->allocate_primitive(&prim_grade_down));
    def("∪", heap->allocate_primitive(&prim_union));
    def("∩", heap->allocate_primitive(&prim_intersect));
    def("⍷", heap->allocate_primitive(&prim_find));
    def("○", heap->allocate_primitive(&prim_circle));
    def("?", heap->allocate_primitive(&prim_question));
    def("⊥", heap->allocate_primitive(&prim_decode));
    def("⊤", heap->allocate_primitive(&prim_encode));
    def("⌹", heap->allocate_primitive(&prim_domino));
    def("⍎", heap->allocate_primitive(&prim_execute));
    def("⍕", heap->allocate_primitive(&prim_format));
    def("⌷", heap->allocate_primitive(&prim_squad));
    def("⍪", heap->allocate_primitive(&prim_table));
    def("⊣", heap->allocate_primitive(&prim_left));
    def("⊢", heap->allocate_primitive(&prim_right));
    def("⊂", heap->allocate_primitive(&prim_enclose));
    def("⊃", heap->allocate_primitive(&prim_disclose));

    // Error handling system functions (ISO 13751 §11.5.7-11.6.5)
    // Note: ⎕ET and ⎕EM are system variables (read-only), not primitives
    def("⎕ES", heap->allocate_primitive(&prim_quad_es));
    def("⎕EA", heap->allocate_primitive(&prim_quad_ea));

    // Other system functions (ISO 13751 §11.5)
    def("⎕DL", heap->allocate_primitive(&prim_quad_dl));
    def("⎕NC", heap->allocate_primitive(&prim_quad_nc));
    def("⎕EX", heap->allocate_primitive(&prim_quad_ex));
    def("⎕NL", heap->allocate_primitive(&prim_quad_nl));

    // Operators (higher-order functions)
    def(".", heap->allocate_operator(&op_dot));
    def("∘.", heap->allocate_operator(&op_outer_dot));
    def("¨", heap->allocate_operator(&op_diaeresis));
    def("⍨", heap->allocate_operator(&op_tilde));
    def("/", heap->allocate_operator(&op_reduce));
    def("⌿", heap->allocate_operator(&op_reduce_first));
    def("\\", heap->allocate_operator(&op_scan));
    def("⍀", heap->allocate_operator(&op_scan_first));
    def("⍤", heap->allocate_operator(&op_rank_op));
    def(",⌷", heap->allocate_operator(&op_catenate_axis));
}

// Destructor
Machine::~Machine() {
    // Delete DIR backend
    delete dir_backend;
    dir_backend = nullptr;

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
        throw_error(parser->get_error().c_str(), nullptr, 1, 0);
    } else {
        // Static optimisation pass (wBurg single-pass rewrite)
        if (optimizer_enabled) {
            StaticOptimizer opt;
            AbsEnv abs_env = build_abs_env(env);
            k = opt.run(k, heap, abs_env);
        }
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

        push_kont(heap->allocate_ephemeral<PerformFinalizeK>());

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
        push_kont(heap->allocate_ephemeral<PerformFinalizeK>());
        push_kont(heap->allocate_ephemeral<FunctionCallK>(result, nullptr, nullptr));

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

    // Reset the ephemeral continuation arena after expression completes
    // All scaffolding continuations are now dead - O(1) cleanup
    heap->reset_arena();

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
void Machine::throw_error(const char* msg, Continuation* source,
                          int error_class, int error_subclass) {
    // Use control (the currently executing continuation) as fallback
    // This allows primitives to get location info from their calling continuation
    Continuation* error_source = source ? source : control;

    // Set error state (ISO 13751 §11.4.4-11.4.5)
    event_type[0] = error_class;
    event_type[1] = error_subclass;
    event_message = string_pool.intern(msg);

    // Capture the current stack for error traces
    error_stack = kont_stack;

    // If we have a source, add it to the error stack (it has the error location)
    if (error_source) {
        error_stack.push_back(error_source);
    }

    // Create and push ThrowErrorK
    ThrowErrorK* err = heap->allocate_ephemeral<ThrowErrorK>(event_message);

    // Copy source location to ThrowErrorK if available
    if (error_source && error_source->has_location()) {
        err->set_location(error_source->line(), error_source->column());
    }

    push_kont(err);
}

} // namespace apl
