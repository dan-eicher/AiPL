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
    // Strand: evaluate elements right-to-left and collect into vector
    // Strategy: use auxiliary continuations to maintain accumulator

    if (elements.empty()) {
        // Empty strand - create empty vector
        Eigen::VectorXd empty_vec(0);
        Value* val = Value::from_vector(empty_vec);
        val = machine->heap->allocate(val);
        machine->ctrl.set_value(val);
        return nullptr;
    }

    if (elements.size() == 1) {
        // Single element - just evaluate it directly (no need for vector wrapper)
        machine->push_kont(elements[0]);
        return nullptr;
    }

    // Multiple elements: evaluate right-to-left using auxiliary continuation
    // Start by evaluating the rightmost element
    // The remaining elements will be evaluated by EvalStrandElementK

    std::vector<Continuation*> remaining(elements.begin(), elements.end() - 1);
    std::vector<Value*> evaluated;  // Empty accumulator

    EvalStrandElementK* eval_next = new EvalStrandElementK(remaining, evaluated);
    machine->heap->allocate_continuation(eval_next);

    // Push in reverse order (stack is LIFO)
    machine->push_kont(eval_next);         // Will execute after rightmost element
    machine->push_kont(elements.back());   // Evaluate rightmost element now

    return nullptr;  // Continue trampoline
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

// EvalStrandElementK implementation
Value* EvalStrandElementK::invoke(Machine* machine) {
    // An element has just been evaluated - its value is in ctrl.value
    // Add it to the FRONT of evaluated_values (we're going right-to-left)
    Value* current_val = machine->ctrl.value;
    evaluated_values.insert(evaluated_values.begin(), current_val);

    if (remaining_elements.empty()) {
        // No more elements to evaluate - build the final strand
        BuildStrandK* build = new BuildStrandK(evaluated_values);
        machine->heap->allocate_continuation(build);
        machine->push_kont(build);
        return nullptr;
    }

    // More elements to evaluate - take the rightmost remaining element
    Continuation* next_elem = remaining_elements.back();
    std::vector<Continuation*> new_remaining(remaining_elements.begin(), remaining_elements.end() - 1);

    // Create new EvalStrandElementK for the next iteration
    EvalStrandElementK* eval_next = new EvalStrandElementK(new_remaining, evaluated_values);
    machine->heap->allocate_continuation(eval_next);

    // Push in reverse order
    machine->push_kont(eval_next);   // Will execute after next element
    machine->push_kont(next_elem);   // Evaluate next element now

    return nullptr;  // Continue trampoline
}

void EvalStrandElementK::mark(APLHeap* heap) {
    // Mark remaining continuations
    for (Continuation* elem : remaining_elements) {
        if (elem) {
            heap->mark_continuation(elem);
        }
    }

    // Mark evaluated values
    for (Value* val : evaluated_values) {
        if (val) {
            heap->mark_value(val);
        }
    }
}

// BuildStrandK implementation
Value* BuildStrandK::invoke(Machine* machine) {
    // All elements have been evaluated - build the vector
    // values are already in left-to-right order

    if (values.empty()) {
        Eigen::VectorXd empty_vec(0);
        Value* val = Value::from_vector(empty_vec);
        val = machine->heap->allocate(val);
        machine->ctrl.set_value(val);
        return nullptr;
    }

    // Create a vector to hold the values
    size_t count = values.size();
    Eigen::VectorXd vec(count);

    for (size_t i = 0; i < count; i++) {
        Value* val = values[i];
        if (val->is_scalar()) {
            vec(i) = val->as_scalar();
        } else {
            // For now, if element is not a scalar, we have a problem
            // APL allows nested arrays, but we haven't implemented that yet
            throw std::runtime_error("Strand elements must be scalars (nested arrays not yet implemented)");
        }
    }

    Value* result = Value::from_vector(vec);
    result = machine->heap->allocate(result);
    machine->ctrl.set_value(result);

    return nullptr;  // Continue trampoline
}

void BuildStrandK::mark(APLHeap* heap) {
    // Mark all values
    for (Value* val : values) {
        if (val) {
            heap->mark_value(val);
        }
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
