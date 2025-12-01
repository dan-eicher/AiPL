// Continuation implementations

#include "continuation.h"
#include "machine.h"
#include "completion.h"

namespace apl {

// Forward declaration of APLHeap for now
// Will be implemented in Phase 1.6
class APLHeap;

// HaltK implementation
void HaltK::invoke(Machine* machine) {
    // Phase 3.2: Terminal continuation - clear the stack to signal termination
    // The value is already in ctrl.value
    machine->kont_stack.clear();
}

void HaltK::mark(APLHeap* heap) {
    // HaltK has no references to mark
    (void)heap;  // Unused
}

// Completion handler implementations - Phase 2
// Completions are handled through the continuation stack, not through Machine state

// PropagateCompletionK - Propagates completion up the stack
// This continuation unwinds the stack until it hits a boundary (CatchReturnK, CatchBreakK, etc.)
void PropagateCompletionK::invoke(Machine* machine) {
    // Set the completion value in ctrl
    if (completion && completion->value) {
        machine->ctrl.set_value(completion->value);
    }

    // Unwind the stack until we hit a boundary continuation
    // Pop continuations until we find one that can handle this completion type
    while (!machine->kont_stack.empty()) {
        Continuation* k = machine->kont_stack.back();

        // Check if this is a boundary that can catch our completion
        if (completion->is_return() && k->is_function_boundary()) {
            // Found a function boundary - pop it and we're done unwinding
            // The completion value is already in ctrl.value
            machine->pop_kont();
            return;
        }

        if (completion->is_break() && k->is_loop_boundary()) {
            // Found a loop boundary - pop it and we're done unwinding
            // The :Leave exits the loop, value is in ctrl.value
            machine->pop_kont();
            return;
        }

        if (completion->is_continue() && k->is_loop_boundary()) {
            // Found a loop boundary for continue - need to re-execute loop
            // For now, just pop and return (continue not fully implemented)
            machine->pop_kont();
            return;
        }

        // Phase 5: Check for error boundaries
        if (completion->is_throw()) {
            // Check if this is a CatchErrorK
            CatchErrorK* catch_err = dynamic_cast<CatchErrorK*>(k);
            if (catch_err) {
                // Found an error boundary - pop it and we're done unwinding
                // The error is "caught" - execution continues normally
                machine->pop_kont();
                return;
            }
        }

        // Not a boundary for our completion type - pop and continue unwinding
        machine->pop_kont();
    }

    // No boundary found - this is an error (unhandled completion)
    // For THROW completions, convert to C++ exception as last resort
    if (completion->is_throw()) {
        const char* msg = completion->target ? completion->target : "Unknown error";
        throw std::runtime_error(std::string("Uncaught APL error: ") + msg);
    }
    throw std::runtime_error("Unhandled completion: no matching boundary found");
}

void PropagateCompletionK::mark(APLHeap* heap) {
    if (completion) {
        heap->mark_completion(completion);
    }
}

// CatchReturnK - Catches RETURN completions at function boundaries
void CatchReturnK::invoke(Machine* machine) {
    // This is invoked in two cases:
    // 1. Function body completed normally - just return the value in ctrl
    // 2. PropagateCompletionK pushed us back - check if there's a completion on stack

    // Check if next item on stack is a PropagateCompletionK with RETURN
    if (!machine->kont_stack.empty()) {
        Continuation* next = machine->kont_stack.back();
        PropagateCompletionK* prop = dynamic_cast<PropagateCompletionK*>(next);

        if (prop && prop->completion && prop->completion->is_return()) {
            // Pop the PropagateCompletionK - we're handling the return
            machine->pop_kont();
            // The return value is already in ctrl.value (set by PropagateCompletionK)
            // Just continue normally - completion is handled
            return;
        }
    }

    // Normal function completion - value already in ctrl, just continue
    (void)function_name;  // Unused for now (could be used for debugging)
}

void CatchReturnK::mark(APLHeap* heap) {
    // No GC references to mark (function_name is static)
    (void)heap;
}

// CatchBreakK - Catches BREAK completions at loop boundaries
void CatchBreakK::invoke(Machine* machine) {
    // Check if next item on stack is a PropagateCompletionK with BREAK
    if (!machine->kont_stack.empty()) {
        Continuation* next = machine->kont_stack.back();
        PropagateCompletionK* prop = dynamic_cast<PropagateCompletionK*>(next);

        if (prop && prop->completion && prop->completion->is_break()) {
            // Pop the PropagateCompletionK - we're handling the break
            machine->pop_kont();
            // For :Leave, we typically return the last value or a default
            // The value is already in ctrl.value
            // Just continue normally - loop is exited
            return;
        }
    }

    // This shouldn't normally be invoked without a break completion
    // If we get here, just continue (shouldn't happen in normal execution)
    (void)label;  // Unused for now (could be used for labeled breaks)
}

void CatchBreakK::mark(APLHeap* heap) {
    // No GC references to mark (label is static)
    (void)heap;
}

// CatchContinueK - Catches CONTINUE completions at loop boundaries
void CatchContinueK::invoke(Machine* machine) {
    // Check if next item on stack is a PropagateCompletionK with CONTINUE
    if (!machine->kont_stack.empty()) {
        Continuation* next = machine->kont_stack.back();
        PropagateCompletionK* prop = dynamic_cast<PropagateCompletionK*>(next);

        if (prop && prop->completion && prop->completion->is_continue()) {
            // Pop the PropagateCompletionK - we're handling the continue
            machine->pop_kont();
            // Re-push the loop continuation to restart the loop
            if (loop_cont) {
                machine->push_kont(loop_cont);
            }
            return;
        }
    }

    // This shouldn't normally be invoked without a continue completion
    (void)loop_cont;  // Used above
}

void CatchContinueK::mark(APLHeap* heap) {
    if (loop_cont) {
        heap->mark_continuation(loop_cont);
    }
}

// CatchErrorK - Catches THROW completions (Phase 5)
void CatchErrorK::invoke(Machine* machine) {
    // This is invoked when an error boundary is reached
    // For now, just continue normally (error boundaries not yet fully implemented)
    // In the future, this would check for THROW completions and handle them
    (void)machine;
}

void CatchErrorK::mark(APLHeap* heap) {
    // No GC references to mark
    (void)heap;
}

// ThrowErrorK - Creates and propagates THROW completion (Phase 5.2)
void ThrowErrorK::invoke(Machine* machine) {
    // Create a THROW completion with the error message
    APLCompletion* throw_comp = machine->heap->allocate<APLCompletion>(
        CompletionType::THROW,
        nullptr,  // No value for errors
        error_message  // Error message in target field
    );

    // Push PropagateCompletionK to unwind the stack
    PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(throw_comp);
    machine->push_kont(prop);
}

void ThrowErrorK::mark(APLHeap* heap) {
    // No GC references to mark (error_message is static or pooled)
    (void)heap;
}

// LiteralK implementation
void LiteralK::invoke(Machine* machine) {
    // Convert the literal double to a Value* at runtime
    Value* val = machine->heap->allocate_scalar(literal_value);
    machine->ctrl.set_value(val);

    // Phase 3.1: No return needed, trampoline continues
}

void LiteralK::mark(APLHeap* heap) {
    // LiteralK only has a double, nothing to mark
    (void)heap;  // Unused
}

// ClosureLiteralK implementation
void ClosureLiteralK::invoke(Machine* machine) {
    // Convert the continuation body to a CLOSURE Value* at runtime
    Value* heap_closure = machine->heap->allocate_closure(body);
    machine->ctrl.set_value(heap_closure);

    // Phase 3.1: No return needed, trampoline continues
}

void ClosureLiteralK::mark(APLHeap* heap) {
    // Mark the body continuation graph
    if (body) {
        heap->mark_continuation(body);
    }
}

// LookupK implementation
void LookupK::invoke(Machine* machine) {
    // Look up the variable in the environment
    Value* val = machine->env->lookup(var_name);

    if (!val) {
        // Variable not found - push THROW completion
        std::string msg = std::string("VALUE ERROR: Undefined variable: ") + var_name;
        const char* interned_msg = machine->string_pool.intern(msg.c_str());
        machine->push_kont(machine->heap->allocate<ThrowErrorK>(interned_msg));
        return;
    }

    machine->ctrl.set_value(val);
    // Phase 3.1: No return needed, trampoline continues
}

void LookupK::mark(APLHeap* heap) {
    // var_name is interned const char*, doesn't need GC marking
    (void)heap;  // Unused
}

// AssignK implementation
void AssignK::invoke(Machine* machine) {
    // Assignment: evaluate expression, then bind to variable
    // Use auxiliary continuation to capture the result

    PerformAssignK* perform = machine->heap->allocate<PerformAssignK>(var_name);

    machine->push_kont(perform);
    machine->push_kont(expr);

    // Phase 3.1: No return needed, trampoline continues
}

void AssignK::mark(APLHeap* heap) {
    if (expr) {
        heap->mark_continuation(expr);
    }
}

// PerformAssignK implementation
void PerformAssignK::invoke(Machine* machine) {
    // Expression has been evaluated - result is in ctrl.value
    // Bind it to the variable name
    Value* val = machine->ctrl.value;

    machine->env->define(var_name, val);

    // Assignment expression returns the assigned value
    machine->ctrl.set_value(val);

    // Phase 3.1: No return needed, trampoline continues
}

void PerformAssignK::mark(APLHeap* heap) {
    // var_name is interned const char*, doesn't need GC marking
    (void)heap;  // Unused
}

// StrandK implementation
void StrandK::invoke(Machine* machine) {
    // Strand: evaluate elements right-to-left and collect into vector
    // Strategy: use auxiliary continuations to maintain accumulator

    if (elements.empty()) {
        // Empty strand - create empty vector
        Eigen::VectorXd empty_vec(0);
        Value* val = machine->heap->allocate_vector(empty_vec);
        machine->ctrl.set_value(val);
        return;  // Early exit for empty case
    }

    if (elements.size() == 1) {
        // Single element - just evaluate it directly (no need for vector wrapper)
        machine->push_kont(elements[0]);
        return;  // Early exit for single element
    }

    // Multiple elements: evaluate right-to-left using auxiliary continuation
    // Start by evaluating the rightmost element
    // The remaining elements will be evaluated by EvalStrandElementK

    std::vector<Continuation*> remaining(elements.begin(), elements.end() - 1);
    std::vector<Value*> evaluated;  // Empty accumulator

    EvalStrandElementK* eval_next = machine->heap->allocate<EvalStrandElementK>(remaining, evaluated);

    // Push in reverse order (stack is LIFO)
    machine->push_kont(eval_next);         // Will execute after rightmost element
    machine->push_kont(elements.back());   // Evaluate rightmost element now

    // Phase 3.1: No return needed, trampoline continues
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
void MonadicK::invoke(Machine* machine) {
    // Monadic function application: evaluate operand, then apply function
    // Strategy: push operand continuation, then push auxiliary to apply function

    // Create auxiliary continuation to apply function after operand evaluates
    ApplyMonadicK* apply = machine->heap->allocate<ApplyMonadicK>(op_name);

    // Push in reverse order (stack is LIFO)
    machine->push_kont(apply);    // Will execute after operand
    machine->push_kont(operand);  // Evaluate operand now

    // Phase 3.1: No return needed, trampoline continues
}

void MonadicK::mark(APLHeap* heap) {
    if (operand) {
        heap->mark_continuation(operand);
    }
}

// DyadicK implementation
void DyadicK::invoke(Machine* machine) {
    // APL evaluates right-to-left: right operand first, then left, then apply
    // Use auxiliary continuations to manage the multi-step process

    // Allocate auxiliary continuation to evaluate left after right completes
    EvalDyadicLeftK* eval_left = machine->heap->allocate<EvalDyadicLeftK>(op_name, left, nullptr);

    // Push work in REVERSE order (stack is LIFO)
    machine->push_kont(eval_left);  // Will execute after right
    machine->push_kont(right);       // Will execute now

    // Phase 3.1: No return needed, trampoline continues
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
void EvalDyadicLeftK::invoke(Machine* machine) {
    // Right operand has been evaluated - its value is in ctrl.value
    // Save the right value and set up left evaluation
    right_val = machine->ctrl.value;

    // Allocate auxiliary continuation to apply function after left evaluates
    ApplyDyadicK* apply = machine->heap->allocate<ApplyDyadicK>(op_name, right_val);

    // Push work in reverse order
    machine->push_kont(apply);   // Will execute after left
    machine->push_kont(left);     // Will execute now

    // Phase 3.1: No return needed, trampoline continues
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
void ApplyMonadicK::invoke(Machine* machine) {
    // Operand has been evaluated - its value is in ctrl.value
    Value* operand_val = machine->ctrl.value;

    // Look up the operator at evaluation time
    Value* op_val = machine->env->lookup(op_name);
    if (!op_val || op_val->tag != ValueType::PRIMITIVE) {
        std::string msg = std::string("VALUE ERROR: Unknown operator: ") + op_name;
        const char* interned_msg = machine->string_pool.intern(msg.c_str());
        machine->push_kont(machine->heap->allocate<ThrowErrorK>(interned_msg));
        return;
    }

    PrimitiveFn* prim_fn = op_val->data.primitive_fn;

    if (!prim_fn->monadic) {
        std::string msg = std::string("SYNTAX ERROR: Operator has no monadic form: ") + op_name;
        const char* interned_msg = machine->string_pool.intern(msg.c_str());
        machine->push_kont(machine->heap->allocate<ThrowErrorK>(interned_msg));
        return;
    }

    // Apply the monadic function (sets machine->ctrl.value directly or pushes ThrowErrorK)
    prim_fn->monadic(machine, operand_val);

    // Phase 3.1: No return needed, trampoline continues
}

void ApplyMonadicK::mark(APLHeap* heap) {
    // ApplyMonadicK has no Values to mark, only the function pointer
    (void)heap;  // Unused
}

// ArgK implementation
void ArgK::invoke(Machine* machine) {
    // Set the argument value and continue with next continuation
    machine->ctrl.set_value(arg_value);

    if (next) {
        machine->push_kont(next);
    }

    // Phase 3.1: No return needed, trampoline continues
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
void ApplyDyadicK::invoke(Machine* machine) {
    // Both operands have been evaluated
    // Right value is saved in right_val
    // Left value is in ctrl.value
    Value* left_val = machine->ctrl.value;

    // Look up the operator at evaluation time
    Value* op_val = machine->env->lookup(op_name);
    if (!op_val || op_val->tag != ValueType::PRIMITIVE) {
        std::string msg = std::string("VALUE ERROR: Unknown operator: ") + op_name;
        const char* interned_msg = machine->string_pool.intern(msg.c_str());
        machine->push_kont(machine->heap->allocate<ThrowErrorK>(interned_msg));
        return;
    }

    PrimitiveFn* prim_fn = op_val->data.primitive_fn;

    if (!prim_fn->dyadic) {
        std::string msg = std::string("SYNTAX ERROR: Operator has no dyadic form: ") + op_name;
        const char* interned_msg = machine->string_pool.intern(msg.c_str());
        machine->push_kont(machine->heap->allocate<ThrowErrorK>(interned_msg));
        return;
    }

    // Apply the dyadic function (sets machine->ctrl.value directly or pushes ThrowErrorK)
    prim_fn->dyadic(machine, left_val, right_val);

    // Phase 3.1: No return needed, trampoline continues
}

void ApplyDyadicK::mark(APLHeap* heap) {
    // Mark the saved right value
    if (right_val) {
        heap->mark_value(right_val);
    }
}

// EvalStrandElementK implementation
void EvalStrandElementK::invoke(Machine* machine) {
    // An element has just been evaluated - its value is in ctrl.value
    // Add it to the FRONT of evaluated_values (we're going right-to-left)
    Value* current_val = machine->ctrl.value;
    evaluated_values.insert(evaluated_values.begin(), current_val);

    if (remaining_elements.empty()) {
        // No more elements to evaluate - build the final strand
        BuildStrandK* build = machine->heap->allocate<BuildStrandK>(evaluated_values);
        machine->push_kont(build);
        return;  // Early exit - done evaluating elements
    }

    // More elements to evaluate - take the rightmost remaining element
    Continuation* next_elem = remaining_elements.back();
    std::vector<Continuation*> new_remaining(remaining_elements.begin(), remaining_elements.end() - 1);

    // Create new EvalStrandElementK for the next iteration
    EvalStrandElementK* eval_next = machine->heap->allocate<EvalStrandElementK>(new_remaining, evaluated_values);

    // Push in reverse order
    machine->push_kont(eval_next);   // Will execute after next element
    machine->push_kont(next_elem);   // Evaluate next element now

    // Phase 3.1: No return needed, trampoline continues
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
void BuildStrandK::invoke(Machine* machine) {
    // All elements have been evaluated - build the vector
    // values are already in left-to-right order

    if (values.empty()) {
        Eigen::VectorXd empty_vec(0);
        Value* val = machine->heap->allocate_vector(empty_vec);
        machine->ctrl.set_value(val);
        return;  // Early exit for empty case
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
                machine->push_kont(machine->heap->allocate<ThrowErrorK>("SYNTAX ERROR: Function has no monadic form"));
                return;
            }

            // Apply monadic function (sets machine->ctrl.value directly or pushes ThrowErrorK)
            prim->monadic(machine, arg);
            return;  // Early exit after monadic application
        }

        // Pattern: [f x y] - dyadic application (function first)
        if (values.size() == 3 && fn_index == 0) {
            Value* left_arg = values[1];
            Value* right_arg = values[2];

            if (!prim->dyadic) {
                machine->push_kont(machine->heap->allocate<ThrowErrorK>("SYNTAX ERROR: Function has no dyadic form"));
                return;
            }

            // Apply dyadic function: x f y (sets machine->ctrl.value directly or pushes ThrowErrorK)
            prim->dyadic(machine, left_arg, right_arg);
            return;  // Early exit after dyadic application
        }

        // Pattern: [x f y] - dyadic application (function in middle - APL infix notation)
        if (values.size() == 3 && fn_index == 1) {
            Value* left_arg = values[0];
            Value* right_arg = values[2];

            if (!prim->dyadic) {
                machine->push_kont(machine->heap->allocate<ThrowErrorK>("SYNTAX ERROR: Function has no dyadic form"));
                return;
            }

            // Apply dyadic function: x f y (sets machine->ctrl.value directly or pushes ThrowErrorK)
            prim->dyadic(machine, left_arg, right_arg);
            return;  // Early exit after dyadic application
        }

        // TODO: Higher-order operators and other patterns
        // For now, fall through to regular vector formation (or error)
        machine->push_kont(machine->heap->allocate<ThrowErrorK>("SYNTAX ERROR: Unsupported function application pattern in strand"));
        return;
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
            machine->push_kont(machine->heap->allocate<ThrowErrorK>("RANK ERROR: Strand elements must be scalars (nested arrays not yet implemented)"));
            return;
        }
    }

    Value* result = machine->heap->allocate_vector(vec);
    machine->ctrl.set_value(result);

    // Phase 3.1: No return needed, trampoline continues
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
void FrameK::invoke(Machine* machine) {
    // Function frame - push return continuation onto stack
    // Phase 2.2: Also push CatchReturnK to establish function boundary

    // Push the catch handler first (it will be invoked after function body completes)
    CatchReturnK* catch_k = machine->heap->allocate<CatchReturnK>(function_name);
    machine->push_kont(catch_k);

    // Then push the function body
    if (return_k) {
        machine->push_kont(return_k);
    }

    // Phase 3.1: No return needed, trampoline continues
}

void FrameK::mark(APLHeap* heap) {
    // Mark return continuation
    if (return_k) {
        heap->mark_continuation(return_k);
    }
}

// ApplyFunctionK implementation
// Implements runtime dispatch for function application (currying transformation)
void ApplyFunctionK::invoke(Machine* machine) {
    // Strategy: Evaluate all components right-to-left, then dispatch based on what we got
    // 1. Evaluate right_arg
    // 2. Evaluate left_arg (if present)
    // 3. Evaluate fn_cont to get the function value
    // 4. Dispatch: if left_arg is null → monadic, else → dyadic

    // Use auxiliary continuations to manage multi-step evaluation
    // Similar to DyadicK but with runtime type checking

    if (left_arg) {
        // Dyadic case: evaluate right, then left, then function, then apply
        EvalApplyFunctionLeftK* eval_left = machine->heap->allocate<EvalApplyFunctionLeftK>(fn_cont, left_arg, nullptr);

        machine->push_kont(eval_left);
        machine->push_kont(right_arg);
    } else {
        // Monadic case: evaluate right, then function, then apply
        EvalApplyFunctionMonadicK* eval_fn = machine->heap->allocate<EvalApplyFunctionMonadicK>(fn_cont, nullptr);

        machine->push_kont(eval_fn);
        machine->push_kont(right_arg);
    }

    // Phase 3.1: No return needed
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
void EvalApplyFunctionLeftK::invoke(Machine* machine) {
    // Right argument has been evaluated - save it
    right_val = machine->ctrl.value;

    // Now evaluate left argument, then function, then dispatch
    // Create continuation that will evaluate function after left arg
    EvalApplyFunctionDyadicK* eval_fn = machine->heap->allocate<EvalApplyFunctionDyadicK>(fn_cont, nullptr, right_val);

    machine->push_kont(eval_fn);
    machine->push_kont(left_arg);

    // Phase 3.1: No return needed
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
void EvalApplyFunctionMonadicK::invoke(Machine* machine) {
    // Argument has been evaluated - save it
    arg_val = machine->ctrl.value;

    // Now evaluate the function continuation, then dispatch (monadic case)
    DispatchFunctionK* dispatch = machine->heap->allocate<DispatchFunctionK>(nullptr, nullptr, arg_val);

    machine->push_kont(dispatch);
    machine->push_kont(fn_cont);

    // Phase 3.1: No return needed
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
void EvalApplyFunctionDyadicK::invoke(Machine* machine) {
    // Left argument has been evaluated - save it
    left_val = machine->ctrl.value;

    // Now evaluate the function continuation, then dispatch (dyadic case)
    DispatchFunctionK* dispatch = machine->heap->allocate<DispatchFunctionK>(nullptr, left_val, right_val);

    machine->push_kont(dispatch);
    machine->push_kont(fn_cont);

    // Phase 3.1: No return needed
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
void DispatchFunctionK::invoke(Machine* machine) {
    // Function has been evaluated - it's in ctrl.value
    fn_val = machine->ctrl.value;

    // Handle CLOSURE values (dfns)
    if (fn_val->tag == ValueType::CLOSURE) {
        // Call the closure using FunctionCallK
        FunctionCallK* call_k = machine->heap->allocate<FunctionCallK>(fn_val, left_val, right_val);
        machine->push_kont(call_k);
        return;  // Early exit for closure case
    }

    // Check that fn_val is actually a primitive function
    if (!fn_val->is_primitive()) {
        machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: expected function value"));
        return;
    }

    PrimitiveFn* prim_fn = fn_val->data.primitive_fn;

    // Determine monadic vs dyadic based on what arguments we have
    if (left_val == nullptr) {
        // Monadic case: only right argument
        if (!prim_fn->monadic) {
            machine->push_kont(machine->heap->allocate<ThrowErrorK>("SYNTAX ERROR: Function has no monadic form"));
            return;
        }

        // Apply monadic function (sets machine->ctrl.value directly or pushes ThrowErrorK)
        prim_fn->monadic(machine, right_val);
    } else {
        // Dyadic case: both arguments
        if (!prim_fn->dyadic) {
            machine->push_kont(machine->heap->allocate<ThrowErrorK>("SYNTAX ERROR: Function has no dyadic form"));
            return;
        }

        // Apply dyadic function (sets machine->ctrl.value directly or pushes ThrowErrorK)
        prim_fn->dyadic(machine, left_val, right_val);
    }

    // Phase 3.1: No return needed
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

// SeqK implementation - execute statements in sequence
void SeqK::invoke(Machine* machine) {
    if (statements.empty()) {
        // Empty sequence returns null/unit value (scalar 0)
        Value* val = machine->heap->allocate_scalar(0.0);
        machine->ctrl.set_value(val);
        return;  // Early exit for empty case
    }

    if (statements.size() == 1) {
        // Single statement - just push it directly
        machine->push_kont(statements[0]);
        return;  // Early exit for single statement
    }

    // Multiple statements - push auxiliary continuation and first statement
    // ExecNextStatementK will handle the remaining statements
    auto* next_k = machine->heap->allocate<ExecNextStatementK>(statements, 1);
    machine->push_kont(next_k);
    machine->push_kont(statements[0]);

    // Phase 3.1: No return needed
}

void SeqK::mark(APLHeap* heap) {
    for (Continuation* stmt : statements) {
        if (stmt) {
            heap->mark_continuation(stmt);
        }
    }
}

// ExecNextStatementK implementation - execute remaining statements
void ExecNextStatementK::invoke(Machine* machine) {
    // The previous statement has been executed and its result is in machine->ctrl.value
    // We discard that result (unless it's the last statement)

    if (next_index >= statements.size()) {
        // All statements executed - current value is the result
        return;  // Early exit - done
    }

    if (next_index == statements.size() - 1) {
        // Last statement - just push it
        machine->push_kont(statements[next_index]);
        return;  // Early exit for last statement
    }

    // More statements to execute - push continuation for next iteration
    auto* next_k = machine->heap->allocate<ExecNextStatementK>(statements, next_index + 1);
    machine->push_kont(next_k);
    machine->push_kont(statements[next_index]);

    // Phase 3.1: No return needed
}

void ExecNextStatementK::mark(APLHeap* heap) {
    for (Continuation* stmt : statements) {
        if (stmt) {
            heap->mark_continuation(stmt);
        }
    }
}

// ============================================================================
// Control Flow Continuations (Phase 3.3.2)
// ============================================================================

// IfK implementation - evaluate condition, then select branch
void IfK::invoke(Machine* machine) {
    // Push auxiliary continuation to select branch after condition is evaluated
    auto* select_k = machine->heap->allocate<SelectBranchK>(then_branch, else_branch);
    machine->push_kont(select_k);

    // Push condition to evaluate
    machine->push_kont(condition);

    // Phase 3.1: No return needed
}

void IfK::mark(APLHeap* heap) {
    if (condition) {
        heap->mark_continuation(condition);
    }
    if (then_branch) {
        heap->mark_continuation(then_branch);
    }
    if (else_branch) {
        heap->mark_continuation(else_branch);
    }
}

// SelectBranchK implementation - select branch based on condition result
void SelectBranchK::invoke(Machine* machine) {
    // Condition result is in machine->ctrl.value
    Value* cond_val = machine->ctrl.value;

    if (!cond_val) {
        // Error: no condition value
        machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: If condition evaluated to null"));
        return;
    }

    // APL convention: 0 is false, non-zero is true
    // For arrays, we'll use the first element
    bool is_true = false;

    if (cond_val->is_scalar()) {
        is_true = (cond_val->as_scalar() != 0.0);
    } else {
        // For arrays, use first element
        const Eigen::MatrixXd* mat = cond_val->as_matrix();
        if (mat->size() > 0) {
            is_true = ((*mat)(0, 0) != 0.0);
        }
    }

    // Select and push the appropriate branch
    if (is_true) {
        if (then_branch) {
            machine->push_kont(then_branch);
        }
    } else {
        if (else_branch) {
            machine->push_kont(else_branch);
        }
    }

    // If no branch was selected (e.g., false with no else), just continue
    // The result remains whatever the condition evaluated to
    // Phase 3.1: No return needed
}

void SelectBranchK::mark(APLHeap* heap) {
    if (then_branch) {
        heap->mark_continuation(then_branch);
    }
    if (else_branch) {
        heap->mark_continuation(else_branch);
    }
}

// WhileK implementation - check condition and loop
void WhileK::invoke(Machine* machine) {
    // Phase 2.2: Push CatchBreakK to establish loop boundary for :Leave
    CatchBreakK* catch_k = machine->heap->allocate<CatchBreakK>();
    machine->push_kont(catch_k);

    // Push auxiliary continuation to check condition
    auto* check_k = machine->heap->allocate<CheckWhileCondK>(condition, body);
    machine->push_kont(check_k);

    // Push condition to evaluate first
    machine->push_kont(condition);

    // Phase 3.1: No return needed
}

void WhileK::mark(APLHeap* heap) {
    if (condition) {
        heap->mark_continuation(condition);
    }
    if (body) {
        heap->mark_continuation(body);
    }
}

// CheckWhileCondK implementation - check condition and decide whether to loop
void CheckWhileCondK::invoke(Machine* machine) {
    // Condition result is in machine->ctrl.value
    Value* cond_val = machine->ctrl.value;

    if (!cond_val) {
        // Error: no condition value
        machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: While condition evaluated to null"));
        return;
    }

    // APL convention: 0 is false, non-zero is true
    bool is_true = false;

    if (cond_val->is_scalar()) {
        is_true = (cond_val->as_scalar() != 0.0);
    } else {
        // For arrays, use first element
        const Eigen::MatrixXd* mat = cond_val->as_matrix();
        if (mat->size() > 0) {
            is_true = ((*mat)(0, 0) != 0.0);
        }
    }

    if (is_true) {
        // Condition is true - execute body then check again
        // Push ourselves back to check after body executes
        auto* check_k = machine->heap->allocate<CheckWhileCondK>(condition, body);
        machine->push_kont(check_k);

        // Push condition to evaluate after body
        machine->push_kont(condition);

        // Push body to execute now
        if (body) {
            machine->push_kont(body);
        }
    }
    // If false, just exit - loop is done
    // Result remains the condition value

    // Phase 3.1: No return needed
}

void CheckWhileCondK::mark(APLHeap* heap) {
    if (condition) {
        heap->mark_continuation(condition);
    }
    if (body) {
        heap->mark_continuation(body);
    }
}

// ForK implementation - evaluate array and start iteration
void ForK::invoke(Machine* machine) {
    // Phase 2.2: Push CatchBreakK to establish loop boundary for :Leave
    CatchBreakK* catch_k = machine->heap->allocate<CatchBreakK>();
    machine->push_kont(catch_k);

    // Push auxiliary continuation to start iteration after array is evaluated
    auto* iterate_k = machine->heap->allocate<ForIterateK>(var_name, nullptr, body, 0);
    machine->push_kont(iterate_k);

    // Push array expression to evaluate
    machine->push_kont(array_expr);

    // Phase 3.1: No return needed
}

void ForK::mark(APLHeap* heap) {
    if (array_expr) {
        heap->mark_continuation(array_expr);
    }
    if (body) {
        heap->mark_continuation(body);
    }
}

// ForIterateK implementation - iterate over array elements
void ForIterateK::invoke(Machine* machine) {
    // First call: array is in machine->ctrl.value
    if (array == nullptr) {
        array = machine->ctrl.value;
        if (!array) {
            machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: For loop array evaluated to null"));
            return;
        }
    }

    // Get array dimensions
    size_t total_elements = 0;
    const Eigen::MatrixXd* mat = nullptr;

    if (array->is_scalar()) {
        // Scalar: iterate once
        total_elements = 1;
    } else {
        mat = array->as_matrix();
        total_elements = mat->size();
    }

    // Check if we're done iterating
    if (index >= total_elements) {
        // Loop finished - result is the last iteration's value (or scalar 0 if empty)
        if (total_elements == 0) {
            Value* zero = machine->heap->allocate_scalar(0.0);
            machine->ctrl.set_value(zero);
        }
        return;  // Early exit - loop done
    }

    // Get current element
    Value* element = nullptr;
    if (array->is_scalar()) {
        element = array;
    } else {
        // Arrays stored column-major in Eigen
        size_t row = index % mat->rows();
        size_t col = index / mat->rows();
        double val = (*mat)(row, col);
        element = machine->heap->allocate_scalar(val);
    }

    // Bind iterator variable to current element
    machine->env->define(var_name, element);

    // Push continuation for next iteration
    auto* next_k = machine->heap->allocate<ForIterateK>(var_name, array, body, index + 1);
    machine->push_kont(next_k);

    // Push body to execute
    if (body) {
        machine->push_kont(body);
    }

    // Phase 3.1: No return needed
}

void ForIterateK::mark(APLHeap* heap) {
    if (array) {
        heap->mark_value(array);
    }
    if (body) {
        heap->mark_continuation(body);
    }
}

// LeaveK implementation - exit from loop
void LeaveK::invoke(Machine* machine) {
    // Phase 2.3: Create BREAK completion and propagate it up the stack
    // This will unwind until we hit a CatchBreakK at a loop boundary

    // The current value in ctrl is the result of the :Leave statement (usually the last value)
    APLCompletion* break_comp = machine->heap->allocate<APLCompletion>(
        CompletionType::BREAK,
        machine->ctrl.value,  // The value to return from the loop
        nullptr  // No label for now
    );

    // Push PropagateCompletionK to unwind the stack
    PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(break_comp);
    machine->push_kont(prop);
}

void LeaveK::mark(APLHeap* heap) {
    // LeaveK has no references
    (void)heap;
}

// ReturnK implementation - return from function
void ReturnK::invoke(Machine* machine) {
    // Phase 2.3: Evaluate the return value, then create RETURN completion

    if (value_expr) {
        // Need to evaluate the value expression first
        // Push CreateReturnK to handle the result
        CreateReturnK* create_k = machine->heap->allocate<CreateReturnK>();
        machine->push_kont(create_k);

        // Evaluate the value expression
        machine->push_kont(value_expr);
    } else {
        // No value expression - return unit/zero
        Value* zero = machine->heap->allocate_scalar(0.0);
        machine->ctrl.set_value(zero);

        // Create RETURN completion with zero value
        APLCompletion* return_comp = machine->heap->allocate<APLCompletion>(
            CompletionType::RETURN,
            zero,
            nullptr
        );

        // Push PropagateCompletionK to unwind the stack
        PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(return_comp);
        machine->push_kont(prop);
    }
}

void ReturnK::mark(APLHeap* heap) {
    if (value_expr) {
        heap->mark_continuation(value_expr);
    }
}

// CreateReturnK implementation - create RETURN completion from evaluated value
void CreateReturnK::invoke(Machine* machine) {
    // Phase 2.3: Value has been evaluated, create RETURN completion
    // The value is in ctrl.value

    APLCompletion* return_comp = machine->heap->allocate<APLCompletion>(
        CompletionType::RETURN,
        machine->ctrl.value,
        nullptr
    );

    // Push PropagateCompletionK to unwind the stack
    PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(return_comp);
    machine->push_kont(prop);
}

void CreateReturnK::mark(APLHeap* heap) {
    // CreateReturnK has no references
    (void)heap;
}

// ============================================================================
// Function Call Continuations (Phase 4.3)
// ============================================================================

// FunctionCallK implementation - apply function to arguments
void FunctionCallK::invoke(Machine* machine) {
    // fn_value should be a CLOSURE
    if (!fn_value || fn_value->tag != ValueType::CLOSURE) {
        machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: Attempted to call non-function value"));
        return;
    }

    // Get the function body continuation graph
    Continuation* body = fn_value->data.closure;
    if (!body) {
        machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: Function has no body"));
        return;
    }

    // Create new environment for function scope (GC-managed)
    Environment* call_env = machine->heap->allocate<Environment>(machine->env);  // Parent for closures

    // Bind arguments in function environment
    if (right_arg) {
        call_env->define("⍵", right_arg);
    }
    if (left_arg) {
        call_env->define("⍺", left_arg);
    }

    // Save current environment
    Environment* saved_env = machine->env;

    // Switch to function environment
    machine->env = call_env;

    // Push restore environment continuation (executes after function returns)
    RestoreEnvK* restore_k = machine->heap->allocate<RestoreEnvK>(saved_env);
    machine->push_kont(restore_k);

    // Execute function body
    machine->push_kont(body);

    // Phase 3.1: No return needed
}

void FunctionCallK::mark(APLHeap* heap) {
    if (fn_value) {
        heap->mark_value(fn_value);
    }
    if (left_arg) {
        heap->mark_value(left_arg);
    }
    if (right_arg) {
        heap->mark_value(right_arg);
    }
}

// RestoreEnvK implementation - restore environment after function call
void RestoreEnvK::invoke(Machine* machine) {
    // Restore the saved environment
    machine->env = saved_env;

    // Result value is already in machine->ctrl.value
    // Phase 3.1: No return needed
}

void RestoreEnvK::mark(APLHeap* heap) {
    // saved_env will be marked by machine's environment chain
    (void)heap;
}

} // namespace apl
