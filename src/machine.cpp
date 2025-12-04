// Machine implementation - CEK machine execution engine

#include "machine.h"
#include "continuation.h"
#include "completion.h"
#include "parser.h"
#include <stdexcept>

namespace apl {

// Constructor
Machine::Machine() {
    heap = new APLHeap();
    heap->set_machine(this);  // Give heap back-pointer for GC
    env = heap->allocate<Environment>();  // Global environment (GC-managed)
    parser = new Parser(this);  // Parser owned by machine
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
        // Parse error - could throw or return nullptr
        // For now, return nullptr (caller can check parser->get_error())
        return nullptr;
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

    // Stack empty - finalize g' curried functions at top level
    // Per paper: "At the top level of an expression, y can also be null"
    // g' semantics: if null(y) then g1(x) else if bas(y) then g2(x,y) else y(g1(x))
    // When we reach top level with a g' curried function, y is null, so apply g1(x)
    if (ctrl.value && ctrl.value->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* curried_data = ctrl.value->data.curried_fn;
        if (curried_data->curry_type == Value::CurryType::G_PRIME) {
            // This is a g' transformation curried function - finalize it
            // Apply monadically: g1(x) where x is first_arg
            Value* fn = curried_data->fn;
            Value* arg = curried_data->first_arg;

            if (fn->is_primitive()) {
                PrimitiveFn* prim_fn = fn->data.primitive_fn;
                if (prim_fn->monadic) {
                    prim_fn->monadic(this, arg);
                    // Result is now in ctrl.value
                }
            }
        }
    }

    return ctrl.value;
}

// Phase 2 complete: Completion handling now done through continuations
// - PropagateCompletionK handles stack unwinding
// - CatchReturnK/CatchBreakK handle boundaries
// - No imperative completion handling needed

} // namespace apl
