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

// AssignK implementation
Value* AssignK::invoke(Machine* machine) {
    // Assignment: evaluate expression, then bind to variable
    // Use auxiliary continuation to capture the result

    PerformAssignK* perform = new PerformAssignK(var_name.c_str());
    machine->heap->allocate_continuation(perform);

    machine->push_kont(perform);
    machine->push_kont(expr);

    return nullptr;  // Continue trampoline
}

void AssignK::mark(APLHeap* heap) {
    if (expr) {
        heap->mark_continuation(expr);
    }
}

// PerformAssignK implementation
Value* PerformAssignK::invoke(Machine* machine) {
    // Expression has been evaluated - result is in ctrl.value
    // Bind it to the variable name
    Value* val = machine->ctrl.value;

    machine->env->define(var_name.c_str(), val);

    // Assignment expression returns the assigned value
    machine->ctrl.set_value(val);

    return nullptr;  // Continue trampoline
}

void PerformAssignK::mark(APLHeap* heap) {
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
    // Monadic function application: evaluate operand, then apply function
    // Strategy: push operand continuation, then push auxiliary to apply function

    // Create auxiliary continuation to apply function after operand evaluates
    ApplyMonadicK* apply = new ApplyMonadicK(prim_fn);
    machine->heap->allocate_continuation(apply);

    // Push in reverse order (stack is LIFO)
    machine->push_kont(apply);    // Will execute after operand
    machine->push_kont(operand);  // Evaluate operand now

    return nullptr;  // Continue trampoline
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


// ApplyMonadicK implementation
Value* ApplyMonadicK::invoke(Machine* machine) {
    // Operand has been evaluated - its value is in ctrl.value
    Value* operand_val = machine->ctrl.value;

    if (!prim_fn->monadic) {
        throw std::runtime_error("Operator has no monadic form");
    }

    // Apply the monadic function
    Value* result = prim_fn->monadic(operand_val);
    machine->ctrl.set_value(result);

    return nullptr;  // Continue trampoline
}

void ApplyMonadicK::mark(APLHeap* heap) {
    // ApplyMonadicK has no Values to mark, only the function pointer
    (void)heap;  // Unused
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

    // CURRYING TRANSFORMATION (Georgeff et al. paper, page 121)
    // Check if any element is a function - if so, apply it with proper permutation
    // g' = λx. λy. if null(y) then g1(x) else if bas(y) then g2(x,y) else y(g1(x))
    //
    // Pattern matching for function application:
    // [f x] -> monadic: f(x)
    // [f x y] -> dyadic: x f y  (note: function is first in strand)
    // [x f y] -> dyadic: x f y  (note: function is second in strand - "permute" reorders)
    //
    // The paper's "permute" function handles the reordering based on positions

    // First, check if there's a function in the strand
    PrimitiveFn* prim = nullptr;
    int fn_index = -1;

    for (size_t i = 0; i < values.size(); i++) {
        if (values[i]->tag == ValueType::PRIMITIVE) {
            prim = values[i]->data.primitive_fn;
            fn_index = i;
            break;
        }
    }

    if (prim != nullptr) {
        // Pattern: [f x] - monadic application
        if (values.size() == 2 && fn_index == 0) {
            Value* arg = values[1];

            if (!prim->monadic) {
                throw std::runtime_error("Function has no monadic form");
            }

            // Apply monadic function
            Value* result = prim->monadic(arg);
            result = machine->heap->allocate(result);
            machine->ctrl.set_value(result);
            return nullptr;
        }

        // Pattern: [f x y] - dyadic application (function first)
        if (values.size() == 3 && fn_index == 0) {
            Value* left_arg = values[1];
            Value* right_arg = values[2];

            if (!prim->dyadic) {
                throw std::runtime_error("Function has no dyadic form");
            }

            // Apply dyadic function: x f y
            Value* result = prim->dyadic(left_arg, right_arg);
            result = machine->heap->allocate(result);
            machine->ctrl.set_value(result);
            return nullptr;
        }

        // Pattern: [x f y] - dyadic application (function in middle - APL infix notation)
        if (values.size() == 3 && fn_index == 1) {
            Value* left_arg = values[0];
            Value* right_arg = values[2];

            if (!prim->dyadic) {
                throw std::runtime_error("Function has no dyadic form");
            }

            // Apply dyadic function: x f y
            Value* result = prim->dyadic(left_arg, right_arg);
            result = machine->heap->allocate(result);
            machine->ctrl.set_value(result);
            return nullptr;
        }

        // TODO: Higher-order operators and other patterns
        // For now, fall through to regular vector formation (or error)
        throw std::runtime_error("Unsupported function application pattern in strand");
    }

    // Regular vector formation (no function application)
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

// ApplyFunctionK implementation
// Implements runtime dispatch for function application (currying transformation)
Value* ApplyFunctionK::invoke(Machine* machine) {
    // Strategy: Evaluate all components right-to-left, then dispatch based on what we got
    // 1. Evaluate right_arg
    // 2. Evaluate left_arg (if present)
    // 3. Evaluate fn_cont to get the function value
    // 4. Dispatch: if left_arg is null → monadic, else → dyadic

    // Use auxiliary continuations to manage multi-step evaluation
    // Similar to DyadicK but with runtime type checking

    if (left_arg) {
        // Dyadic case: evaluate right, then left, then function, then apply
        EvalApplyFunctionLeftK* eval_left = new EvalApplyFunctionLeftK(fn_cont, left_arg, nullptr);
        machine->heap->allocate_continuation(eval_left);

        machine->push_kont(eval_left);
        machine->push_kont(right_arg);
    } else {
        // Monadic case: evaluate right, then function, then apply
        EvalApplyFunctionMonadicK* eval_fn = new EvalApplyFunctionMonadicK(fn_cont, nullptr);
        machine->heap->allocate_continuation(eval_fn);

        machine->push_kont(eval_fn);
        machine->push_kont(right_arg);
    }

    return nullptr;
}

void ApplyFunctionK::mark(APLHeap* heap) {
    if (fn_cont) {
        heap->mark_continuation(fn_cont);
    }
    if (left_arg) {
        heap->mark_continuation(left_arg);
    }
    if (right_arg) {
        heap->mark_continuation(right_arg);
    }
}

// EvalApplyFunctionLeftK implementation
Value* EvalApplyFunctionLeftK::invoke(Machine* machine) {
    // Right argument has been evaluated - save it
    right_val = machine->ctrl.value;

    // Now evaluate left argument, then function, then dispatch
    // Create continuation that will evaluate function after left arg
    EvalApplyFunctionDyadicK* eval_fn = new EvalApplyFunctionDyadicK(fn_cont, nullptr, right_val);
    machine->heap->allocate_continuation(eval_fn);

    machine->push_kont(eval_fn);
    machine->push_kont(left_arg);

    return nullptr;
}

void EvalApplyFunctionLeftK::mark(APLHeap* heap) {
    if (fn_cont) {
        heap->mark_continuation(fn_cont);
    }
    if (left_arg) {
        heap->mark_continuation(left_arg);
    }
    if (right_val) {
        heap->mark_value(right_val);
    }
}

// EvalApplyFunctionMonadicK implementation
Value* EvalApplyFunctionMonadicK::invoke(Machine* machine) {
    // Argument has been evaluated - save it
    arg_val = machine->ctrl.value;

    // Now evaluate the function continuation, then dispatch (monadic case)
    DispatchFunctionK* dispatch = new DispatchFunctionK(nullptr, nullptr, arg_val);
    machine->heap->allocate_continuation(dispatch);

    machine->push_kont(dispatch);
    machine->push_kont(fn_cont);

    return nullptr;
}

void EvalApplyFunctionMonadicK::mark(APLHeap* heap) {
    if (fn_cont) {
        heap->mark_continuation(fn_cont);
    }
    if (arg_val) {
        heap->mark_value(arg_val);
    }
}

// EvalApplyFunctionDyadicK implementation
Value* EvalApplyFunctionDyadicK::invoke(Machine* machine) {
    // Left argument has been evaluated - save it
    left_val = machine->ctrl.value;

    // Now evaluate the function continuation, then dispatch (dyadic case)
    DispatchFunctionK* dispatch = new DispatchFunctionK(nullptr, left_val, right_val);
    machine->heap->allocate_continuation(dispatch);

    machine->push_kont(dispatch);
    machine->push_kont(fn_cont);

    return nullptr;
}

void EvalApplyFunctionDyadicK::mark(APLHeap* heap) {
    if (fn_cont) {
        heap->mark_continuation(fn_cont);
    }
    if (left_val) {
        heap->mark_value(left_val);
    }
    if (right_val) {
        heap->mark_value(right_val);
    }
}

// DispatchFunctionK implementation
// This is where the actual currying transformation happens:
// g' = λx. λy. if null(y) then g1(x) else if bas(y) then g2(x,y) else y(g1(x))
Value* DispatchFunctionK::invoke(Machine* machine) {
    // Function has been evaluated - it's in ctrl.value
    fn_val = machine->ctrl.value;

    // Check that fn_val is actually a function
    if (!fn_val->is_primitive()) {
        // TODO: Handle closures and other function types
        throw std::runtime_error("ApplyFunctionK: expected function value");
    }

    PrimitiveFn* prim_fn = fn_val->data.primitive_fn;

    // Determine monadic vs dyadic based on what arguments we have
    if (left_val == nullptr) {
        // Monadic case: only right argument
        if (!prim_fn->monadic) {
            throw std::runtime_error("Function has no monadic form");
        }

        Value* result = prim_fn->monadic(right_val);
        machine->ctrl.set_value(result);
    } else {
        // Dyadic case: both arguments
        if (!prim_fn->dyadic) {
            throw std::runtime_error("Function has no dyadic form");
        }

        Value* result = prim_fn->dyadic(left_val, right_val);
        machine->ctrl.set_value(result);
    }

    return nullptr;
}

void DispatchFunctionK::mark(APLHeap* heap) {
    if (fn_val) {
        heap->mark_value(fn_val);
    }
    if (left_val) {
        heap->mark_value(left_val);
    }
    if (right_val) {
        heap->mark_value(right_val);
    }
}

} // namespace apl
