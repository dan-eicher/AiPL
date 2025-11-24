// Machine implementation - CEK machine execution engine

#include "machine.h"
#include "continuation.h"
#include "completion.h"
#include <stdexcept>

namespace apl {

// Execute the machine until halt
// This is the main trampoline loop that drives the CEK machine
Value* Machine::execute() {
    while (!kont_stack.empty()) {
        // Pop next continuation from stack
        Continuation* k = kont_stack.back();
        kont_stack.pop_back();

        // Invoke continuation
        // If it returns non-nullptr, that's the final result
        Value* result = k->invoke(this);

        if (result != nullptr) {
            // Continuation returned final result - halt
            ctrl.halt();
            ctrl.value = result;
            return result;
        }

        // nullptr means continue - check for completions
        if (ctrl.completion && !ctrl.completion->is_normal()) {
            handle_completion();
            continue;
        }

        // Check for GC periodically
        maybe_gc();
    }

    // Stack empty - halt and return current value
    ctrl.halt();
    return ctrl.value;
}

// Handle completion records (RETURN, BREAK, CONTINUE, THROW)
void Machine::handle_completion() {
    if (!ctrl.completion) {
        return;
    }

    APLCompletion* comp = ctrl.completion;

    switch (comp->type) {
        case CompletionType::NORMAL:
            // Normal completion - just extract the value and continue
            ctrl.value = comp->value;
            ctrl.completion = nullptr;
            break;

        case CompletionType::RETURN: {
            // Unwind to function boundary
            bool found = unwind_to_boundary(
                [](Continuation* k) { return k->is_function_boundary(); },
                comp->target
            );

            if (!found) {
                throw std::runtime_error("RETURN outside of function");
            }

            // Set value and clear completion
            ctrl.value = comp->value;
            ctrl.completion = nullptr;
            break;
        }

        case CompletionType::BREAK: {
            // Unwind to loop boundary
            bool found = unwind_to_boundary(
                [](Continuation* k) { return k->is_loop_boundary(); },
                comp->target
            );

            if (!found) {
                throw std::runtime_error("BREAK outside of loop");
            }

            // For BREAK, we discard the loop and continue with the break value
            pop_kont();  // Remove the loop continuation
            ctrl.value = comp->value;
            ctrl.completion = nullptr;
            break;
        }

        case CompletionType::CONTINUE: {
            // Unwind to loop boundary
            bool found = unwind_to_boundary(
                [](Continuation* k) { return k->is_loop_boundary(); },
                comp->target
            );

            if (!found) {
                throw std::runtime_error("CONTINUE outside of loop");
            }

            // For CONTINUE, we re-invoke the loop continuation
            ctrl.value = comp->value;
            ctrl.completion = nullptr;

            // The loop continuation will handle the continue
            Continuation* loop_k = kont_stack.back();
            if (loop_k) {
                loop_k->invoke(this);
            }
            break;
        }

        case CompletionType::THROW:
            // For now, just throw a C++ exception
            // Later phases will implement proper APL error handling
            throw std::runtime_error(
                std::string("APL Error: ") +
                (comp->target ? comp->target : "unspecified")
            );
    }
}

// Unwind stack to a boundary (for RETURN/BREAK/CONTINUE)
bool Machine::unwind_to_boundary(bool (*predicate)(Continuation*), const char* label) {
    // Scan backwards through continuation stack
    for (auto it = kont_stack.rbegin(); it != kont_stack.rend(); ++it) {
        Continuation* k = *it;

        // Check if this continuation matches the predicate
        if (predicate(k)) {
            // If we need a label, check for match
            // For now, we don't have labeled continuations, so just succeed
            if (label) {
                // TODO: Implement label matching in later phases
                // For now, just match the first boundary
            }

            // Unwind everything after this point
            while (kont_stack.back() != k) {
                Continuation* to_remove = pop_kont();
                delete to_remove;
            }

            return true;
        }
    }

    return false;  // No matching boundary found
}

} // namespace apl
