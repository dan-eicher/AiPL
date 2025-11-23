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
    // This is safe because we're no longer in parse-time
    Value* val = machine->heap->allocate_scalar(literal_value);
    machine->ctrl.set_value(val);

    if (next) {
        return next->invoke(machine);
    }

    // No next continuation - halt
    machine->halt();
    return val;
}

void LiteralK::mark(APLHeap* heap) {
    // LiteralK only has a double (not a Value*), so nothing to mark there
    // Just mark the next continuation
    if (next) {
        heap->mark_continuation(next);
    }
}

// StrandK implementation
Value* StrandK::invoke(Machine* machine) {
    // Evaluate each element continuation and collect the results
    std::vector<double> data;

    for (Continuation* elem : elements) {
        Value* elem_val = elem->invoke(machine);

        // Each element should evaluate to a scalar (for now)
        // TODO: In full APL, strands can nest and need flattening
        if (elem_val->tag == ValueType::SCALAR) {
            data.push_back(elem_val->as_scalar());
        } else {
            // For now, error on non-scalar strand elements
            // Full APL would flatten nested arrays
            machine->halt();
            return nullptr;
        }
    }

    // Create Eigen vector from collected data
    Eigen::VectorXd vec(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        vec(i) = data[i];
    }

    Value* val = Value::from_vector(vec);
    machine->ctrl.set_value(val);

    if (next) {
        return next->invoke(machine);
    }

    // No next continuation - halt
    machine->halt();
    return val;
}

void StrandK::mark(APLHeap* heap) {
    // Mark all element continuations
    for (Continuation* elem : elements) {
        if (elem) {
            heap->mark_continuation(elem);
        }
    }

    // Mark next continuation
    if (next) {
        heap->mark_continuation(next);
    }
}

// BinOpK implementation
Value* BinOpK::invoke(Machine* machine) {
    // APL evaluates right-to-left, so:
    // 1. Evaluate right operand
    // 2. Evaluate left operand
    // 3. Look up operator and apply it

    // Evaluate right operand
    Value* right_val = right->invoke(machine);

    // Evaluate left operand
    Value* left_val = left->invoke(machine);

    // Look up the operator from the environment
    Value* op_val = machine->env->lookup(op_name);
    if (!op_val || op_val->tag != ValueType::PRIMITIVE) {
        throw std::runtime_error(std::string("Unknown operator: ") + op_name);
    }

    // Apply the dyadic form of the primitive function
    PrimitiveFn* prim_fn = op_val->data.primitive_fn;
    if (!prim_fn->dyadic) {
        throw std::runtime_error(std::string("Operator has no dyadic form: ") + op_name);
    }

    Value* result = prim_fn->dyadic(left_val, right_val);

    machine->ctrl.set_value(result);
    return result;
}

void BinOpK::mark(APLHeap* heap) {
    // Mark both operand continuations
    if (left) {
        heap->mark_continuation(left);
    }
    if (right) {
        heap->mark_continuation(right);
    }
}

// ArgK implementation
Value* ArgK::invoke(Machine* machine) {
    // Set the argument value and continue with next continuation
    machine->ctrl.set_value(arg_value);

    if (next) {
        return next->invoke(machine);
    }

    // No next continuation - halt
    machine->halt();
    return arg_value;
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

// FrameK implementation
Value* FrameK::invoke(Machine* machine) {
    // Function frame - set the current value and return
    // The return continuation will handle what happens next

    if (return_k) {
        return return_k->invoke(machine);
    }

    // No return continuation - halt
    machine->halt();
    return machine->ctrl.value;
}

void FrameK::mark(APLHeap* heap) {
    // Mark return continuation
    if (return_k) {
        heap->mark_continuation(return_k);
    }
}

} // namespace apl
