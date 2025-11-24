// Continuation implementations

#include "continuation.h"
#include "machine.h"

namespace apl {

// Forward declaration of APLHeap for now
// Will be implemented in Phase 1.6
class APLHeap;

// HaltK implementation
Value* HaltK::invoke(Machine* machine) {
    // Terminal continuation - halt the machine
    machine->halt();
    return machine->ctrl.value;
}

void HaltK::mark(APLHeap* heap) {
    // HaltK has no references to mark
    (void)heap;  // Unused
}

// LiteralK implementation
Value* LiteralK::invoke(Machine* machine) {
    // Convert the literal double to a Value* at runtime
    Value* val = machine->heap->allocate_scalar(literal_value);
    machine->ctrl.set_value(val);

    return nullptr;  // Continue trampoline
}

void LiteralK::mark(APLHeap* heap) {
    // LiteralK only has a double, nothing to mark
    (void)heap;  // Unused
}

// LookupK implementation
Value* LookupK::invoke(Machine* machine) {
    // Look up the variable in the environment
    Value* val = machine->env->lookup(var_name.c_str());

    if (!val) {
        // Variable not found - error
        machine->halt();
        return nullptr;
    }

    machine->ctrl.set_value(val);
    return nullptr;  // Continue trampoline
}

void LookupK::mark(APLHeap* heap) {
    // var_name is std::string, doesn't need GC marking
    (void)heap;  // Unused
}

// StrandK implementation
Value* StrandK::invoke(Machine* machine) {
    // TODO: Implement with collector continuation pattern
    // Needs auxiliary continuations to evaluate each element and collect results
    // For now, throw error
    (void)machine;
    throw std::runtime_error("StrandK not yet implemented with trampoline");
}

void StrandK::mark(APLHeap* heap) {
    // Mark all element continuations
    for (Continuation* elem : elements) {
        if (elem) {
            heap->mark_continuation(elem);
        }
    }
}

// MonadicK implementation
Value* MonadicK::invoke(Machine* machine) {
    // TODO: Implement with auxiliary continuation pattern like DyadicK
    // For now, throw error
    (void)machine;
    throw std::runtime_error("MonadicK not yet implemented with trampoline");
}

void MonadicK::mark(APLHeap* heap) {
    if (operand) {
        heap->mark_continuation(operand);
    }
}

// DyadicK implementation
Value* DyadicK::invoke(Machine* machine) {
    // APL evaluates right-to-left: right operand first, then left, then apply
    // Use auxiliary continuations to manage the multi-step process

    // Allocate auxiliary continuation to evaluate left after right completes
    EvalDyadicLeftK* eval_left = new EvalDyadicLeftK(prim_fn, left, nullptr);
    machine->heap->allocate_continuation(eval_left);

    // Push work in REVERSE order (stack is LIFO)
    machine->push_kont(eval_left);  // Will execute after right
    machine->push_kont(right);       // Will execute now

    return nullptr;  // Continue trampoline
}

void DyadicK::mark(APLHeap* heap) {
    if (left) {
        heap->mark_continuation(left);
    }
    if (right) {
        heap->mark_continuation(right);
    }
}

// EvalDyadicLeftK implementation
Value* EvalDyadicLeftK::invoke(Machine* machine) {
    // Right operand has been evaluated - its value is in ctrl.value
    // Save the right value and set up left evaluation
    right_val = machine->ctrl.value;

    // Allocate auxiliary continuation to apply function after left evaluates
    ApplyDyadicK* apply = new ApplyDyadicK(prim_fn, right_val);
    machine->heap->allocate_continuation(apply);

    // Push work in reverse order
    machine->push_kont(apply);   // Will execute after left
    machine->push_kont(left);     // Will execute now

    return nullptr;  // Continue trampoline
}

void EvalDyadicLeftK::mark(APLHeap* heap) {
    if (left) {
        heap->mark_continuation(left);
    }
    if (right_val) {
        heap->mark_value(right_val);
    }
}


// ArgK implementation
Value* ArgK::invoke(Machine* machine) {
    // Set the argument value and continue with next continuation
    machine->ctrl.set_value(arg_value);

    if (next) {
        machine->push_kont(next);
    }

    return nullptr;  // Continue trampoline
}

void ArgK::mark(APLHeap* heap) {
    // Mark the argument Value
    if (arg_value) {
        heap->mark_value(arg_value);
    }

    // Mark next continuation
    if (next) {
        heap->mark_continuation(next);
    }
}

// ApplyDyadicK implementation
Value* ApplyDyadicK::invoke(Machine* machine) {
    // Both operands have been evaluated
    // Right value is saved in right_val
    // Left value is in ctrl.value
    Value* left_val = machine->ctrl.value;

    if (!prim_fn->dyadic) {
        throw std::runtime_error("Operator has no dyadic form");
    }

    // Apply the dyadic function
    Value* result = prim_fn->dyadic(left_val, right_val);
    machine->ctrl.set_value(result);

    return nullptr;  // Continue trampoline
}

void ApplyDyadicK::mark(APLHeap* heap) {
    // Mark the saved right value
    if (right_val) {
        heap->mark_value(right_val);
    }
}

// FrameK implementation
Value* FrameK::invoke(Machine* machine) {
    // Function frame - push return continuation onto stack

    if (return_k) {
        machine->push_kont(return_k);
    }

    return nullptr;  // Continue trampoline
}

void FrameK::mark(APLHeap* heap) {
    // Mark return continuation
    if (return_k) {
        heap->mark_continuation(return_k);
    }
}

} // namespace apl
