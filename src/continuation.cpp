// Continuation implementations

#include "continuation.h"
#include "machine.h"
#include "completion.h"
#include <algorithm>

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
    // Lexical strand: just return the pre-computed vector Value
    machine->ctrl.set_value(vector_value);
}

void StrandK::mark(APLHeap* heap) {
    // Mark the vector Value
    if (vector_value) {
        heap->mark_value(vector_value);
    }
}

// JuxtaposeK implementation
// G2 Grammar: fbn-term ::= fb-term fbn-term
// Semantics: if type(x₁) = bas then x₂(x₁) else x₁(x₂)
void JuxtaposeK::invoke(Machine* machine) {
    // Evaluate right-to-left (APL evaluation order)
    // After right is evaluated, we'll evaluate left and then apply

    EvalJuxtaposeLeftK* eval_left = machine->heap->allocate<EvalJuxtaposeLeftK>(left, nullptr);

    // Push in reverse order (stack is LIFO)
    machine->push_kont(eval_left);  // Will execute after right
    machine->push_kont(right);       // Evaluate right now
}

void JuxtaposeK::mark(APLHeap* heap) {
    if (left) {
        heap->mark_continuation(left);
    }
    if (right) {
        heap->mark_continuation(right);
    }
}

// EvalJuxtaposeLeftK implementation
// After right is evaluated, save it and evaluate left
void EvalJuxtaposeLeftK::invoke(Machine* machine) {
    // Right has been evaluated - save it
    right_val = machine->ctrl.value;

    // Push continuation to perform juxtaposition after left is evaluated
    PerformJuxtaposeK* perform = machine->heap->allocate<PerformJuxtaposeK>(right_val);

    // Push in reverse order
    machine->push_kont(perform);  // Will execute after left
    machine->push_kont(left);      // Evaluate left now
}

void EvalJuxtaposeLeftK::mark(APLHeap* heap) {
    if (left) {
        heap->mark_continuation(left);
    }
    if (right_val) {
        heap->mark_value(right_val);
    }
}

// PerformJuxtaposeK implementation
// Both left and right are evaluated - apply G2 juxtaposition rule
// Rule: if type(x₁) = bas then x₂(x₁) else x₁(x₂)
void PerformJuxtaposeK::invoke(Machine* machine) {
    Value* left_val = machine->ctrl.value;

    // G2 Rule: if type(left) = bas then right(left) else left(right)
    if (left_val->is_basic_value()) {
        // Left is a basic value (scalar, vector, or matrix)
        // Apply right to left: right(left)
        // right must be a function

        // DispatchFunctionK expects the function in ctrl.value, so set it there
        machine->ctrl.set_value(right_val);
        // Use DispatchFunctionK to apply right_val as function to left_val as argument
        machine->push_kont(machine->heap->allocate<DispatchFunctionK>(nullptr, nullptr, left_val));
    } else {
        // Left is a function (or curried function, or derived operator)
        // Apply left to right: left(right)

        // DispatchFunctionK expects the function in ctrl.value, so set it there
        machine->ctrl.set_value(left_val);
        // Use DispatchFunctionK to apply left_val as function to right_val as argument
        machine->push_kont(machine->heap->allocate<DispatchFunctionK>(nullptr, nullptr, right_val));
    }
}

void PerformJuxtaposeK::mark(APLHeap* heap) {
    if (right_val) {
        heap->mark_value(right_val);
    }
}
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

    // G2 g' finalization: If operand is a g' curried function, unwrap it first
    // Per paper: g' x = λy. if null(y) then g1(x) else ...
    // When used as an argument (y is not null), we apply g1(x) first
    if (operand_val && operand_val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* curried_data = operand_val->data.curried_fn;
        if (curried_data->curry_type == Value::CurryType::G_PRIME) {
            // This is a g' curried function being used as an argument - finalize it
            Value* fn = curried_data->fn;
            Value* arg = curried_data->first_arg;
            if (fn->is_primitive()) {
                PrimitiveFn* prim_fn = fn->data.primitive_fn;
                if (prim_fn->monadic) {
                    prim_fn->monadic(machine, arg);
                    operand_val = machine->ctrl.value;  // Use the finalized value
                }
            }
        }
    }

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

    // G2 g' transformation: If function is overloaded (has both monadic and dyadic forms),
    // create a curried function to defer the monadic/dyadic decision to runtime
    if (prim_fn->monadic && prim_fn->dyadic) {
        // Overloaded function - create CURRIED_FN with G_PRIME (g' transformation)
        // This allows the function to be applied monadically now, or dyadically if another arg appears
        Value* curried = machine->heap->allocate_curried_fn(op_val, operand_val, Value::CurryType::G_PRIME);
        machine->ctrl.value = curried;
    } else {
        // Monadic-only function - apply immediately
        prim_fn->monadic(machine, operand_val);
    }

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

    // G2 g' finalization: Unwrap g' curried functions before adding to strand
    if (current_val && current_val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* curried_data = current_val->data.curried_fn;
        if (curried_data->curry_type == Value::CurryType::G_PRIME) {
            // Unwrap g' curried function
            Value* fn = curried_data->fn;
            Value* arg = curried_data->first_arg;
            if (fn->is_primitive()) {
                PrimitiveFn* prim_fn = fn->data.primitive_fn;
                if (prim_fn->monadic) {
                    prim_fn->monadic(machine, arg);
                    current_val = machine->ctrl.value;  // Use the finalized value
                }
            }
        }
    }

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
    // If fn_val wasn't provided in constructor, get it from ctrl.value
    // (for cases where function was just evaluated)
    if (fn_val == nullptr) {
        fn_val = machine->ctrl.value;
    }

    // Handle CLOSURE values (dfns)
    if (fn_val->tag == ValueType::CLOSURE) {
        // Call the closure using FunctionCallK
        FunctionCallK* call_k = machine->heap->allocate<FunctionCallK>(fn_val, left_val, right_val);
        machine->push_kont(call_k);
        return;  // Early exit for closure case
    }

    // G2 Grammar: Handle CURRIED_FN values
    if (fn_val->tag == ValueType::CURRIED_FN) {
        // A curried function was applied
        // The curried function already has one argument captured
        // Now we're applying it to the remaining argument(s)

        Value::CurriedFnData* curried_data = fn_val->data.curried_fn;
        Value* inner_fn = curried_data->fn;
        Value* first_arg = curried_data->first_arg;

        if (curried_data->curry_type == Value::CurryType::DYADIC_CURRY) {
            // Simple dyadic curry: fn was curried with first_arg (RIGHT/omega)
            // Now apply right_val as the LEFT/alpha argument
            // In G2 semantics: captured arg is omega, new arg is alpha
            if (right_val == nullptr) {
                machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: curried function expects argument"));
                return;
            }

            // Update state: swap arguments like G_PRIME does
            // right_val (newly applied) becomes LEFT (alpha)
            // first_arg (captured) becomes RIGHT (omega)
            fn_val = inner_fn;
            left_val = right_val;   // New argument is LEFT (alpha)
            right_val = first_arg;  // Captured argument is RIGHT (omega)
            // Fall through to check fn_val type below
        } else if (curried_data->curry_type == Value::CurryType::OPERATOR_CURRY) {
            // Operator curry: inner_fn is DERIVED_OPERATOR, first_arg is second operand
            // For inner product: first_arg is second function operand (×)
            // For rank: first_arg is rank specification (a value)
            // inner_fn = DERIVED_OPERATOR(op, first_operand)
            if (inner_fn->tag != ValueType::DERIVED_OPERATOR) {
                machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: OPERATOR_CURRY expected DERIVED_OPERATOR"));
                return;
            }
            Value::DerivedOperatorData* derived_data = inner_fn->data.derived_op;
            PrimitiveOp* op = derived_data->op;
            Value* first_operand = derived_data->first_operand;
            Value* second_operand = first_arg;  // Second operand (function for ., value for ⍤)

            if (left_val && right_val) {
                // Have both array arguments - call dyadic operator
                op->dyadic(machine, left_val, first_operand, second_operand, right_val);
            } else if (right_val) {
                // Only have one array argument - curry to wait for second (like g' for functions)
                // Monadic vs dyadic decision happens at top level via g' finalization
                Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::DYADIC_CURRY);
                machine->ctrl.value = curried;
            } else {
                machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: operator curry expects array argument"));
            }
            return;
        } else {
            // G_PRIME transformation: more complex logic for composition
            // For now, just apply like dyadic
            // TODO: Implement full g' semantics (check if right_val is bas or function)
            if (right_val == nullptr) {
                // No second argument - apply monadically to first_arg
                fn_val = inner_fn;
                left_val = nullptr;
                right_val = first_arg;
                // Fall through
            } else {
                // Has second argument - apply dyadically
                // In G2 juxtaposition, first_arg is the RIGHT operand (captured)
                // and right_val is the LEFT operand (newly applied)
                // So: left=right_val, right=first_arg
                fn_val = inner_fn;
                left_val = right_val;  // New argument is LEFT
                right_val = first_arg; // Captured argument is RIGHT
                // Fall through
            }
        }
        if (fn_val->tag == ValueType::CURRIED_FN) {
            // Update ctrl.value before recursing, since invoke() reads from it
            machine->ctrl.value = fn_val;
            this->invoke(machine);
            return;
        }
        if (fn_val->tag == ValueType::CLOSURE) {
            FunctionCallK* call_k = machine->heap->allocate<FunctionCallK>(fn_val, left_val, right_val);
            machine->push_kont(call_k);
            return;
        }
        if (fn_val->tag == ValueType::DERIVED_OPERATOR) {
            Value::DerivedOperatorData* derived_data = fn_val->data.derived_op;
            PrimitiveOp* op = derived_data->op;
            Value* first_operand = derived_data->first_operand;

            // For DERIVED_OPERATOR from dyadic operator (like inner product f.g):
            // first_arg contains the second function operand (g)
            // But if first_arg is a CURRIED_FN(g, array), we need to unwrap it:
            // - Extract g as the second function operand
            // - Extract array as the right array argument
            // - Use right_val as the left array argument
            Value* second_func_operand = nullptr;
            Value* actual_left_array = nullptr;
            Value* actual_right_array = nullptr;

            if (op->dyadic && first_arg) {
                if (first_arg->tag == ValueType::CURRIED_FN) {
                    // Unwrap CURRIED_FN to extract second function operand and right array
                    Value::CurriedFnData* inner_curried = first_arg->data.curried_fn;
                    if (inner_curried->fn->is_function()) {
                        // first_arg = CURRIED_FN(g, right_array)
                        // Extract: g is second operand, first_arg of curried is right array
                        second_func_operand = inner_curried->fn;
                        actual_right_array = inner_curried->first_arg;
                        actual_left_array = right_val;  // The value we were applied to
                    }
                } else if (first_arg->is_function()) {
                    // first_arg is already a plain function
                    second_func_operand = first_arg;
                    // In this case left_val should be left array, right_val is right array
                    actual_left_array = left_val;
                    actual_right_array = right_val;
                }
            }

            // G2 Universal Currying: curry all dyadic objects when applied with one argument
            if (op->dyadic && actual_left_array && actual_right_array) {
                // Have both array arguments - apply dyadic form with both function operands
                op->dyadic(machine, actual_left_array, first_operand, second_func_operand, actual_right_array);
            } else if (op->dyadic && left_val) {
                // Have left array but arrays weren't extracted - use original values
                op->dyadic(machine, left_val, first_operand, second_func_operand, right_val);
            } else if (op->dyadic && !left_val) {
                // Only have right argument - this is the second operator operand
                // Use OPERATOR_CURRY to capture it properly (not DYADIC_CURRY which is for array args)
                Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::OPERATOR_CURRY);
                machine->ctrl.value = curried;
            } else if (op->monadic && !left_val) {
                // Pure monadic operator - apply immediately
                op->monadic(machine, first_operand, right_val);
            } else {
                machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: operator requires operands"));
            }
            return;
        }
    }

    // G2 g' finalization: Unwrap any g' curried functions in arguments
    // Per paper: when a g' curried function is used as an argument (not at top level),
    // it should be unwrapped by applying g1(x)
    if (left_val && left_val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* curried_data = left_val->data.curried_fn;
        if (curried_data->curry_type == Value::CurryType::G_PRIME) {
            // Unwrap g' curried function
            Value* fn = curried_data->fn;
            Value* arg = curried_data->first_arg;
            if (fn->is_primitive()) {
                PrimitiveFn* prim_fn = fn->data.primitive_fn;
                if (prim_fn->monadic) {
                    prim_fn->monadic(machine, arg);
                    left_val = machine->ctrl.value;  // Use the finalized value
                }
            }
        }
    }
    if (right_val && right_val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* curried_data = right_val->data.curried_fn;
        if (curried_data->curry_type == Value::CurryType::G_PRIME) {
            // Unwrap g' curried function
            Value* fn = curried_data->fn;
            Value* arg = curried_data->first_arg;
            if (fn->is_primitive()) {
                PrimitiveFn* prim_fn = fn->data.primitive_fn;
                if (prim_fn->monadic) {
                    prim_fn->monadic(machine, arg);
                    right_val = machine->ctrl.value;  // Use the finalized value
                }
            }
        }
    }

    if (fn_val->tag == ValueType::DERIVED_OPERATOR) {
        Value::DerivedOperatorData* derived_data = fn_val->data.derived_op;
        PrimitiveOp* op = derived_data->op;
        Value* first_operand = derived_data->first_operand;

        // For operators with BOTH monadic and dyadic forms (like commute ⍨),
        // curry with G_PRIME when given one argument. This allows `2 +⍨ 3` to work correctly
        // by deferring until we know if there's a left argument.
        if (op->monadic && op->dyadic && !left_val && right_val) {
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::G_PRIME);
            machine->ctrl.value = curried;
            return;
        }

        // Monadic-only operators apply immediately when given one argument
        if (op->monadic && !op->dyadic && !left_val && right_val) {
            op->monadic(machine, first_operand, right_val);
            return;
        }

        // Dyadic operators with both arguments
        if (op->dyadic && left_val && right_val) {
            op->dyadic(machine, left_val, first_operand, nullptr, right_val);
            return;
        }

        // G2: Dyadic-only operators with one argument should curry
        // Inner product "." takes two function operands, uses OPERATOR_CURRY to store second function
        // Outer product "∘." takes one function operand, uses DYADIC_CURRY to store array argument
        if (op->dyadic && !op->monadic && !left_val && right_val) {
            if (strcmp(op->name, ".") == 0 || strcmp(op->name, "⍤") == 0) {
                // Inner product: first_arg stores the second function operand (for "+." applied to "×")
                // Rank operator: first_arg stores the rank specification (for "-⍤" applied to "2")
                Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::OPERATOR_CURRY);
                machine->ctrl.value = curried;
            } else {
                // Outer product and similar: first_arg stores the array argument
                Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::DYADIC_CURRY);
                machine->ctrl.value = curried;
            }
            return;
        }

        machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: operator requires operands"));
        return;
    }

    if (!fn_val->is_primitive()) {
        machine->push_kont(machine->heap->allocate<ThrowErrorK>("VALUE ERROR: expected function value"));
        return;
    }

    PrimitiveFn* prim_fn = fn_val->data.primitive_fn;

    // Determine monadic vs dyadic based on what arguments we have
    if (left_val == nullptr) {
        // Monadic case: only right argument

        if (prim_fn->monadic && prim_fn->dyadic && !force_monadic) {
            // Overloaded function: use g' transformation (complex currying)
            // Unless force_monadic is set, in which case apply monadic form directly
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::G_PRIME);
            machine->ctrl.value = curried;
        } else if (prim_fn->monadic && force_monadic) {
            // Force immediate monadic evaluation (used by operators like each, rank)
            prim_fn->monadic(machine, right_val);
        } else if (prim_fn->dyadic) {
            // Pure dyadic function: simple currying (right arg captured, waiting for left)
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::DYADIC_CURRY);
            machine->ctrl.value = curried;
        } else if (prim_fn->monadic) {
            // Pure monadic function - apply directly
            prim_fn->monadic(machine, right_val);
        } else {
            // No forms available
            machine->push_kont(machine->heap->allocate<ThrowErrorK>("SYNTAX ERROR: Function has no forms"));
            return;
        }
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

// ============================================================================
// G2 Grammar Continuations (Operator Support)
// ============================================================================

// DerivedOperatorK implementation - partially apply dyadic operator
void DerivedOperatorK::invoke(Machine* machine) {
    // Push continuation to apply operator after operand is evaluated
    machine->push_kont(machine->heap->allocate<ApplyDerivedOperatorK>(op_name));
    machine->push_kont(operand_cont);
}

void DerivedOperatorK::mark(APLHeap* heap) {
    if (operand_cont) {
        heap->mark_continuation(operand_cont);
    }
}

// ApplyDerivedOperatorK implementation - create DERIVED_OPERATOR value
void ApplyDerivedOperatorK::invoke(Machine* machine) {
    Value* first_operand = machine->ctrl.value;

    // Look up the operator by name from environment
    Value* op_val = machine->env->lookup(op_name);
    if (!op_val) {
        std::string msg = std::string("VALUE ERROR: Unknown operator: ") + op_name;
        const char* interned_msg = machine->string_pool.intern(msg.c_str());
        machine->push_kont(machine->heap->allocate<ThrowErrorK>(interned_msg));
        return;
    }

    // The operator should be stored as an OPERATOR ValueType
    if (op_val->tag != ValueType::OPERATOR) {
        std::string msg = std::string("VALUE ERROR: Not an operator: ") + op_name;
        const char* interned_msg = machine->string_pool.intern(msg.c_str());
        machine->push_kont(machine->heap->allocate<ThrowErrorK>(interned_msg));
        return;
    }

    PrimitiveOp* op = op_val->data.op;

    // Both monadic and dyadic operators create a DERIVED_OPERATOR value
    // For dyadic operators (like .): stores operator and first operand, waits for second
    // For monadic operators (like ¨): stores operator and operand (the function), waits for omega
    if (op->dyadic || op->monadic) {
        // Create a DERIVED_OPERATOR value that captures:
        //   - The operator (dyadic or monadic)
        //   - The first operand
        // When this derived operator is applied, it will call op->monadic() or op->dyadic()
        Value* derived = machine->heap->allocate_derived_operator(op, first_operand);
        machine->ctrl.value = derived;
    } else {
        machine->push_kont(machine->heap->allocate<ThrowErrorK>("SYNTAX ERROR: Operator has neither monadic nor dyadic form"));
    }
}

void ApplyDerivedOperatorK::mark(APLHeap* heap) {
    // op_name is an interned string, not GC-managed
    (void)heap;
}

// ============================================================================
// CellIterK - General-purpose cell iterator
// ============================================================================

// Helper: get the rank of a value (0=scalar, 1=vector, 2=matrix)
static int get_value_rank(Value* v) {
    if (v->is_scalar()) return 0;
    if (v->is_vector()) return 1;
    return 2;
}

// Helper function for cell counting
static int count_cells_for_rank(Value* arr, int k) {
    if (!arr) return 0;
    int arr_rank = get_value_rank(arr);
    if (k >= arr_rank) return 1;

    if (arr->is_scalar()) return 1;

    const Eigen::MatrixXd* mat = arr->as_matrix();

    if (arr->is_vector()) {
        return (k == 0) ? mat->rows() : 1;
    }

    // Matrix
    if (k == 0) {
        return mat->rows() * mat->cols();
    } else if (k == 1) {
        return mat->rows();
    }
    return 1;
}

// Helper: extract a k-cell from an array
static Value* extract_cell(Machine* m, Value* arr, int k, int cell_index) {
    if (!arr) return nullptr;

    int arr_rank = get_value_rank(arr);
    if (k >= arr_rank) {
        // Full rank: return whole array
        return arr;
    }

    if (arr->is_scalar()) {
        return arr;
    }

    const Eigen::MatrixXd* mat = arr->as_matrix();

    if (arr->is_vector()) {
        if (k == 0) {
            // 0-cell of vector: individual scalar
            if (cell_index >= mat->rows()) return nullptr;
            return m->heap->allocate_scalar((*mat)(cell_index, 0));
        }
        return arr;
    }

    // Matrix
    if (k == 0) {
        // 0-cell: scalar at linear index (row-major)
        int rows = mat->rows();
        int cols = mat->cols();
        int r = cell_index / cols;
        int c = cell_index % cols;
        if (r >= rows) return nullptr;
        return m->heap->allocate_scalar((*mat)(r, c));
    } else if (k == 1) {
        // 1-cell: row vector
        if (cell_index >= mat->rows()) return nullptr;
        Eigen::VectorXd row = mat->row(cell_index).transpose();
        return m->heap->allocate_vector(row);
    }

    return arr;
}

void CellIterK::invoke(Machine* machine) {
    if (mode == CellIterMode::COLLECT) {
        // Forward iteration: process cell at current_cell
        if (current_cell >= total_cells) {
            // Done - assemble results
            if (results.empty()) {
                // No results - return empty (shouldn't happen)
                machine->ctrl.set_value(machine->heap->allocate_scalar(0));
                return;
            }

            // Check if all results are scalars
            bool all_scalars = true;
            for (Value* v : results) {
                if (!v->is_scalar()) {
                    all_scalars = false;
                    break;
                }
            }

            if (all_scalars) {
                // Reassemble into array based on number of results
                // When function changes cell shape (like reduce), results.size() determines output shape
                if (results.size() == 1) {
                    // Single scalar result - return as scalar
                    machine->ctrl.set_value(results[0]);
                } else if (results.size() == (size_t)(orig_rows * orig_cols) && !orig_is_vector && orig_cols > 1) {
                    // Same number of results as input elements AND input was matrix - preserve matrix shape
                    // This handles rank-0 operations that preserve element count
                    Eigen::MatrixXd mat(orig_rows, orig_cols);
                    for (size_t i = 0; i < results.size(); i++) {
                        mat(i / orig_cols, i % orig_cols) = results[i]->as_scalar();
                    }
                    machine->ctrl.set_value(machine->heap->allocate_matrix(mat));
                } else {
                    // Otherwise (including reduction), return vector of results
                    Eigen::VectorXd vec(results.size());
                    for (size_t i = 0; i < results.size(); i++) {
                        vec(i) = results[i]->as_scalar();
                    }
                    machine->ctrl.set_value(machine->heap->allocate_vector(vec));
                }
            } else {
                // Results are vectors - try to assemble into matrix
                bool all_same_len = true;
                int vec_len = -1;
                for (Value* v : results) {
                    if (v->is_vector()) {
                        int len = v->rows();
                        if (vec_len < 0) vec_len = len;
                        else if (len != vec_len) all_same_len = false;
                    } else {
                        all_same_len = false;
                    }
                }

                if (all_same_len && vec_len > 0) {
                    Eigen::MatrixXd mat(results.size(), vec_len);
                    for (size_t i = 0; i < results.size(); i++) {
                        const Eigen::MatrixXd* v = results[i]->as_matrix();
                        mat.row(i) = v->col(0).transpose();
                    }
                    machine->ctrl.set_value(machine->heap->allocate_matrix(mat));
                } else {
                    // Mixed results - return last (TODO: nested arrays)
                    machine->ctrl.set_value(results.back());
                }
            }
            return;
        }

        // Extract cells
        Value* left_cell = extract_cell(machine, lhs, left_rank,
            (lhs && count_cells_for_rank(lhs, left_rank) == 1) ? 0 : current_cell);
        Value* right_cell = extract_cell(machine, rhs, right_rank, current_cell);

        if (!right_cell) {
            machine->push_kont(machine->heap->allocate<ThrowErrorK>("INDEX ERROR: cell extraction failed"));
            return;
        }

        // Push collector continuation, then dispatch function
        // Use force_monadic=true when applying monadically to get immediate result
        machine->push_kont(machine->heap->allocate<CellCollectK>(this));
        machine->push_kont(machine->heap->allocate<DispatchFunctionK>(fn, left_cell, right_cell, left_cell == nullptr));

    } else if (mode == CellIterMode::FOLD_RIGHT) {
        // Backward iteration for right-fold
        if (current_cell < 0) {
            // Done - accumulator has final result
            machine->ctrl.set_value(accumulator);
            return;
        }

        if (!accumulator) {
            // First iteration - set accumulator to last cell
            accumulator = extract_cell(machine, rhs, right_rank, current_cell);
            current_cell--;
            // Continue to next iteration
            machine->push_kont(this);
            return;
        }

        // Apply: element f accumulator
        Value* element = extract_cell(machine, rhs, right_rank, current_cell);

        machine->push_kont(machine->heap->allocate<CellCollectK>(this));
        machine->push_kont(machine->heap->allocate<DispatchFunctionK>(fn, element, accumulator));

    } else if (mode == CellIterMode::SCAN_RIGHT) {
        // Backward iteration for scan
        if (current_cell < 0) {
            // Done - reverse results and assemble
            std::reverse(results.begin(), results.end());

            if (orig_is_vector || orig_cols == 1) {
                Eigen::VectorXd vec(results.size());
                for (size_t i = 0; i < results.size(); i++) {
                    vec(i) = results[i]->as_scalar();
                }
                machine->ctrl.set_value(machine->heap->allocate_vector(vec));
            } else {
                // For matrix scan, each row is scanned independently
                // This simplified version assumes vector input
                Eigen::VectorXd vec(results.size());
                for (size_t i = 0; i < results.size(); i++) {
                    vec(i) = results[i]->as_scalar();
                }
                machine->ctrl.set_value(machine->heap->allocate_vector(vec));
            }
            return;
        }

        if (!accumulator) {
            // First iteration - last element is its own scan result
            accumulator = extract_cell(machine, rhs, right_rank, current_cell);
            results.push_back(accumulator);
            current_cell--;
            machine->push_kont(this);
            return;
        }

        // Apply: element f accumulator
        Value* element = extract_cell(machine, rhs, right_rank, current_cell);

        machine->push_kont(machine->heap->allocate<CellCollectK>(this));
        machine->push_kont(machine->heap->allocate<DispatchFunctionK>(fn, element, accumulator));
    }
}

void CellIterK::mark(APLHeap* heap) {
    if (fn) heap->mark_value(fn);
    if (lhs) heap->mark_value(lhs);
    if (rhs) heap->mark_value(rhs);
    if (accumulator) heap->mark_value(accumulator);
    for (Value* v : results) {
        if (v) heap->mark_value(v);
    }
}

void CellCollectK::invoke(Machine* machine) {
    Value* result = machine->ctrl.value;

    if (iter->mode == CellIterMode::COLLECT) {
        iter->results.push_back(result);
        iter->current_cell++;
    } else if (iter->mode == CellIterMode::FOLD_RIGHT) {
        iter->accumulator = result;
        iter->current_cell--;
    } else if (iter->mode == CellIterMode::SCAN_RIGHT) {
        iter->accumulator = result;
        iter->results.push_back(result);
        iter->current_cell--;
    }

    // Continue iteration
    machine->push_kont(iter);
}

void CellCollectK::mark(APLHeap* heap) {
    // iter is on the continuation stack, will be marked separately
    (void)heap;
}

} // namespace apl
