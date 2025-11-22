// Machine implementation - CEK machine execution engine

#include "machine.h"
#include "continuation.h"
#include "completion.h"
#include <stdexcept>

namespace apl {

// Execute the machine until halt
// This is the main trampoline loop that drives the CEK machine
Value* Machine::execute() {
    while (ctrl.mode != ExecMode::HALTED) {
        // If we have an abrupt completion, handle it
        if (ctrl.completion && !ctrl.completion->is_normal()) {
            handle_completion();
            continue;
        }

        // If we have a normal completion, extract the value
        if (ctrl.completion && ctrl.completion->is_normal()) {
            ctrl.value = ctrl.completion->value;
            delete ctrl.completion;
            ctrl.completion = nullptr;
        }

        // If control has a value, pop continuation and invoke it
        if (ctrl.value) {
            Continuation* k = pop_kont();
            if (!k) {
                // No more continuations, halt with current value
                ctrl.halt();
                return ctrl.value;
            }

            // Invoke continuation (may modify ctrl.value or ctrl.completion)
            k->invoke(this);
            // Don't delete k - it's GC-managed now

            // Check for GC periodically
            maybe_gc();
            continue;
        }

        // If we reach here with no value and no completion, we're done
        if (!ctrl.value && !ctrl.completion) {
            ctrl.halt();
            return nullptr;
        }
    }

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
