// Continuation implementations

#include "continuation.h"
#include "machine.h"
#include "completion.h"
#include "operators.h"
#include <algorithm>
#include <stdexcept>
#include <typeinfo>

namespace apl {

// Forward declaration of Heap for now
// Will be implemented in Phase 1.6
class Heap;

// ============================================================================
// Terminal and Completion Continuations
// ============================================================================

void HaltK::invoke(Machine* machine) {
    // Phase 3.2: Terminal continuation - clear the stack to signal termination
    // The value is already in result
    machine->kont_stack.clear();
}

void HaltK::mark(Heap* heap) {
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
        machine->result = completion->value;
    }

    // Note: error_stack is captured in ThrowErrorK before unwinding starts

    // Unwind the stack until we hit a boundary continuation
    // Pop continuations until we find one that can handle this completion type
    while (!machine->kont_stack.empty()) {
        Continuation* k = machine->kont_stack.back();

        // Check if this is a boundary that can catch our completion
        if (completion->is_return() && k->is_function_boundary()) {
            // Found a function boundary - pop it and we're done unwinding
            // The completion value is already in result
            machine->pop_kont();
            return;
        }

        if (completion->is_break() && k->is_break_boundary()) {
            // Found a break boundary (CatchBreakK) - pop it and we're done unwinding
            // The :Leave exits the loop, value is in result
            machine->pop_kont();
            return;
        }

        if (completion->is_continue() && k->is_continue_boundary()) {
            // Found a continue boundary (CatchContinueK) - pop it and we're done unwinding
            // Execution continues with what's next on stack (condition re-evaluation)
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
                machine->error_stack.clear();  // Discard trace, error was handled
                machine->pop_kont();
                return;
            }
        }

        // Not a boundary for our completion type - pop and continue unwinding
        machine->pop_kont();
    }

    // No boundary found - this is an error (unhandled completion)
    // For THROW completions, throw APLError (user-visible error)
    if (completion->is_throw()) {
        const char* msg = completion->target ? completion->target : "Unknown error";
        throw APLError(msg);
    }
    // Other unhandled completions are VM bugs
    throw std::runtime_error("Unhandled completion: no matching boundary found");
}

void PropagateCompletionK::mark(Heap* heap) {
    heap->mark(completion);
}

// CatchReturnK - Catches RETURN completions at function boundaries
void CatchReturnK::invoke(Machine* machine) {
    // This is invoked in two cases:
    // 1. Function body completed normally - just return the value in ctrl
    // 2. PropagateCompletionK pushed us back - check if there's a completion on stack

    // Check if next item on stack is propagating a RETURN completion
    if (!machine->kont_stack.empty()) {
        Completion* comp = machine->kont_stack.back()->get_propagating_completion();
        if (comp && comp->is_return()) {
            // Pop the propagating continuation - we're handling the return
            machine->pop_kont();
            // The return value is already in result
            return;
        }
    }

    // Normal function completion - value already in ctrl, just continue
    (void)function_name;  // Unused for now (could be used for debugging)
}

void CatchReturnK::mark(Heap* heap) {
    // No GC references to mark (function_name is static)
    (void)heap;
}

// CatchBreakK - Catches BREAK completions at loop boundaries
void CatchBreakK::invoke(Machine* machine) {
    // Check if next item on stack is propagating a BREAK completion
    if (!machine->kont_stack.empty()) {
        Completion* comp = machine->kont_stack.back()->get_propagating_completion();
        if (comp && comp->is_break()) {
            // Pop the propagating continuation - we're handling the break
            machine->pop_kont();
            // The value is already in result - loop is exited
            return;
        }
    }

    // Normal loop termination (condition became false) - just continue
}

void CatchBreakK::mark(Heap* heap) {
    // No GC references to mark
    (void)heap;
}

// CatchContinueK - Catches CONTINUE completions at loop boundaries
void CatchContinueK::invoke(Machine* machine) {
    // Check if next item on stack is propagating a CONTINUE completion
    if (!machine->kont_stack.empty()) {
        Continuation* next = machine->kont_stack.back();
        Completion* comp = next->get_propagating_completion();
        if (comp && comp->is_continue()) {
            // Pop the propagating continuation - we're handling the continue
            machine->pop_kont();
            // Re-push the loop continuation to restart the loop condition check
            if (loop_cont) {
                machine->push_kont(loop_cont);
            }
            return;
        }
    }

    // Normal body completion - just continue to next iteration (already set up on stack)
}

void CatchContinueK::mark(Heap* heap) {
    heap->mark(loop_cont);
}

// CatchErrorK - Catches THROW completions (Phase 5)
void CatchErrorK::invoke(Machine* machine) {
    // This is invoked when an error boundary is reached
    // For now, just continue normally (error boundaries not yet fully implemented)
    // In the future, this would check for THROW completions and handle them
    (void)machine;
}

void CatchErrorK::mark(Heap* heap) {
    // No GC references to mark
    (void)heap;
}

// ThrowErrorK - Creates and propagates THROW completion (Phase 5.2)
// Note: error_stack is captured by Machine::throw_error() before ThrowErrorK is created
void ThrowErrorK::invoke(Machine* machine) {
    // Create a THROW completion with the error message
    Completion* throw_comp = machine->heap->allocate<Completion>(
        CompletionType::THROW,
        nullptr,  // No value for errors
        error_message  // Error message in target field
    );

    // Push PropagateCompletionK to unwind the stack
    PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(throw_comp);
    machine->push_kont(prop);
}

void ThrowErrorK::mark(Heap* heap) {
    // No GC references to mark (error_message is static or pooled)
    (void)heap;
}

// ============================================================================
// Value Continuations (Literals, Lookup, Assignment)
// ============================================================================

void LiteralK::invoke(Machine* machine) {
    // Convert the literal double to a Value* at runtime
    Value* val = machine->heap->allocate_scalar(literal_value);
    machine->result = val;

    // Phase 3.1: No return needed, trampoline continues
}

void LiteralK::mark(Heap* heap) {
    // LiteralK only has a double, nothing to mark
    (void)heap;  // Unused
}

// ClosureLiteralK implementation
void ClosureLiteralK::invoke(Machine* machine) {
    // Convert the continuation body to a CLOSURE Value* at runtime
    Value* heap_closure = machine->heap->allocate_closure(body, is_niladic);
    machine->result = heap_closure;

    // Phase 3.1: No return needed, trampoline continues
}

void ClosureLiteralK::mark(Heap* heap) {
    // Mark the body continuation graph
    heap->mark(body);
}

// DefinedOperatorLiteralK implementation
void DefinedOperatorLiteralK::invoke(Machine* machine) {
    // Create DEFINED_OPERATOR value with captured environment
    Value::DefinedOperatorData* op_data = new Value::DefinedOperatorData();
    op_data->body = body;
    op_data->name = operator_name;
    op_data->is_dyadic_operator = (right_operand_name != nullptr);
    op_data->is_ambivalent = true;  // dfn-style operators are always ambivalent
    op_data->left_operand_name = left_operand_name;
    op_data->right_operand_name = right_operand_name;
    op_data->left_arg_name = "⍺";   // dfn convention
    op_data->right_arg_name = "⍵";  // dfn convention
    op_data->result_name = nullptr; // dfn-style doesn't name result
    op_data->lexical_env = machine->env;  // Capture current environment

    Value* op_val = machine->heap->allocate_defined_operator(op_data);

    // Assign to operator name in environment
    machine->env->define(operator_name, op_val);
    machine->result = op_val;
}

void DefinedOperatorLiteralK::mark(Heap* heap) {
    heap->mark(body);
    // operator_name, left_operand_name, right_operand_name are interned strings
}

// InvokeDefinedOperatorK implementation
// Invokes a user-defined operator with bound operands and arguments
void InvokeDefinedOperatorK::invoke(Machine* machine) {
    // Create new environment extending the operator's lexical environment
    Environment* env = machine->heap->allocate<Environment>(op->lexical_env);

    // Bind the left operand to named parameter (always present)
    env->define(op->left_operand_name, left_operand);
    // Also bind to ⍺⍺ for APL compatibility
    env->define("⍺⍺", left_operand);

    // Bind the right operand for dyadic operators
    if (op->is_dyadic_operator && right_operand && op->right_operand_name) {
        env->define(op->right_operand_name, right_operand);
        // Also bind to ⍵⍵ for APL compatibility
        env->define("⍵⍵", right_operand);
    }

    // Bind ∇ for recursive self-reference to the operator
    if (operator_value) {
        env->define("∇", operator_value);
    }

    // Bind arguments using dfn conventions (⍺ and ⍵)
    if (left_arg && op->left_arg_name) {
        env->define(op->left_arg_name, left_arg);
    }
    env->define(op->right_arg_name, right_arg);

    // Save current environment and set up for body execution
    Environment* saved_env = machine->env;
    machine->env = env;

    // Push continuation to restore environment after body completes
    machine->push_kont(machine->heap->allocate<RestoreEnvK>(saved_env));

    // Push continuation to catch RETURN completions
    machine->push_kont(machine->heap->allocate<CatchReturnK>(op->name));

    // Push the operator body for execution
    machine->push_kont(op->body);
}

void InvokeDefinedOperatorK::mark(Heap* heap) {
    // Note: op->body is GC-managed
    if (op && op->body) {
        heap->mark(op->body);
    }
    heap->mark(operator_value);
    heap->mark(left_operand);
    heap->mark(right_operand);
    heap->mark(left_arg);
    heap->mark(right_arg);
}

// LookupK implementation
void LookupK::invoke(Machine* machine) {
    // Look up the variable in the environment
    Value* val = machine->env->lookup(var_name);

    if (!val) {
        // Variable not found - throw error with our location
        std::string msg = std::string("VALUE ERROR: Undefined variable: ") + var_name;
        machine->throw_error(msg.c_str(), this);
        return;
    }

    machine->result = val;
    // Phase 3.1: No return needed, trampoline continues
}

void LookupK::mark(Heap* heap) {
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

void AssignK::mark(Heap* heap) {
    heap->mark(expr);
}

// PerformAssignK implementation
void PerformAssignK::invoke(Machine* machine) {
    // Expression has been evaluated - result is in result
    // Bind it to the variable name
    Value* val = machine->result;

    // Finalize curried functions before assignment (A←⍳5, A←+/1 2 3)
    if (val && val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* cd = val->data.curried_fn;
        if (cd->curry_type == Value::CurryType::G_PRIME ||
            cd->curry_type == Value::CurryType::DYADIC_CURRY) {
            machine->push_kont(this);
            machine->push_kont(machine->heap->allocate<PerformFinalizeK>());
            return;
        }
    }

    // Special handling for ⍺←value (alpha default): only assign if ⍺ not already defined
    // This implements APL's conditional default argument syntax
    if (strcmp(var_name, "⍺") == 0 && machine->env->lookup(var_name) != nullptr) {
        // ⍺ already defined (left arg was passed) - skip assignment
        // result stays as-is for the expression value
    } else {
        machine->env->define(var_name, val);
    }

    // Assignment expression returns the assigned value
    machine->result = val;

    // Phase 3.1: No return needed, trampoline continues
}

void PerformAssignK::mark(Heap* heap) {
    // var_name is interned const char*, doesn't need GC marking
    (void)heap;  // Unused
}

// SysVarReadK implementation - read a system variable
void SysVarReadK::invoke(Machine* machine) {
    switch (var_id) {
        case SysVarId::IO:
            machine->result = machine->heap->allocate_scalar(static_cast<double>(machine->io));
            break;
        case SysVarId::PP:
            machine->result = machine->heap->allocate_scalar(static_cast<double>(machine->pp));
            break;
        case SysVarId::CT:
            machine->result = machine->heap->allocate_scalar(machine->ct);
            break;
        case SysVarId::RL:
            machine->result = machine->heap->allocate_scalar(static_cast<double>(machine->rl));
            break;
        default:
            machine->throw_error("SYSTEM ERROR: unknown system variable", this);
            break;
    }
}

void SysVarReadK::mark(Heap* heap) {
    (void)heap;  // var_id is an enum, nothing to mark
}

// SysVarAssignK implementation - evaluate expression then assign to system variable
void SysVarAssignK::invoke(Machine* machine) {
    // Push continuation to perform the assignment after expression is evaluated
    machine->push_kont(machine->heap->allocate<PerformSysVarAssignK>(var_id));
    // Push the expression to evaluate
    machine->push_kont(expr);
}

void SysVarAssignK::mark(Heap* heap) {
    heap->mark(expr);
}

// PerformSysVarAssignK implementation - perform actual system variable assignment
void PerformSysVarAssignK::invoke(Machine* machine) {
    Value* val = machine->result;

    // System variables require scalar values
    if (!val || !val->is_scalar()) {
        machine->throw_error("DOMAIN ERROR: system variable requires scalar value", this);
        return;
    }

    double dbl_val = val->as_scalar();

    switch (var_id) {
        case SysVarId::IO: {
            int int_val = static_cast<int>(dbl_val);
            if (dbl_val != int_val || (int_val != 0 && int_val != 1)) {
                machine->throw_error("DOMAIN ERROR: ⎕IO must be 0 or 1", this);
                return;
            }
            machine->io = int_val;
            break;
        }
        case SysVarId::PP: {
            int int_val = static_cast<int>(dbl_val);
            if (dbl_val != int_val || int_val < 1 || int_val > 17) {
                machine->throw_error("DOMAIN ERROR: ⎕PP must be 1-17", this);
                return;
            }
            machine->pp = int_val;
            break;
        }
        case SysVarId::CT:
            if (dbl_val < 0) {
                machine->throw_error("DOMAIN ERROR: ⎕CT must be nonnegative", this);
                return;
            }
            machine->ct = dbl_val;
            break;
        case SysVarId::RL: {
            // RL must be a positive integer
            if (dbl_val < 1 || dbl_val != static_cast<double>(static_cast<uint64_t>(dbl_val))) {
                machine->throw_error("DOMAIN ERROR: ⎕RL must be a positive integer", this);
                return;
            }
            machine->rl = static_cast<uint64_t>(dbl_val);
            machine->rng.seed(machine->rl);
            break;
        }
        default:
            machine->throw_error("SYSTEM ERROR: unknown system variable", this);
            return;
    }

    // Assignment returns the assigned value
    machine->result = val;
}

void PerformSysVarAssignK::mark(Heap* heap) {
    (void)heap;  // var_id is an enum, nothing to mark
}

// StrandK implementation
void StrandK::invoke(Machine* machine) {
    // Lexical strand: just return the pre-computed vector Value
    machine->result = vector_value;
}

void StrandK::mark(Heap* heap) {
    // Mark the vector Value
    heap->mark(vector_value);
}

// ============================================================================
// Juxtaposition and Application Continuations
// ============================================================================

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

void JuxtaposeK::mark(Heap* heap) {
    heap->mark(left);
    heap->mark(right);
}

// EvalJuxtaposeLeftK implementation
// After right is evaluated, save it and evaluate left
void EvalJuxtaposeLeftK::invoke(Machine* machine) {
    // Right has been evaluated - save it
    right_val = machine->result;

    // Push continuation to perform juxtaposition after left is evaluated
    PerformJuxtaposeK* perform = machine->heap->allocate<PerformJuxtaposeK>(right_val);

    // Push in reverse order
    machine->push_kont(perform);  // Will execute after left
    machine->push_kont(left);      // Evaluate left now
}

void EvalJuxtaposeLeftK::mark(Heap* heap) {
    heap->mark(left);
    heap->mark(right_val);
}

// PerformJuxtaposeK implementation
// Both left and right are evaluated - apply G2 juxtaposition rule
// Extended rule: if both basic then strand, else if type(x₁) = bas then x₂(x₁) else x₁(x₂)
void PerformJuxtaposeK::invoke(Machine* machine) {
    Value* left_val = machine->result;

    // Extension to G2: when both values are basic, form a strand (vector)
    // This handles cases like {⍵ ⍵}5 which should return 5 5
    //
    // ISO 13751 NOTE: The ISO spec has NO stranding rule in the Phrase Table -
    // there's no "A B" pattern for adjacent values. Stranding like "1 2 3" works
    // because it's parsed as a single numeric-literal token at the lexer level.
    // But "1 (2 3) 4" cannot work that way since (2 3) is an evaluated expression.
    //
    // APL2-style stranding would create NESTED arrays here:
    //   "1 (2 3) 4"    → 3-element nested vector: 1, (2 3), 4
    //   "{⍵ ⍵}(1 2 3)" → 2-element nested vector: (1 2 3), (1 2 3)
    //
    // TODO: NESTED ARRAYS NOT YET IMPLEMENTED
    // Until we have nested arrays, we only allow scalar stranding (which produces
    // simple vectors) and reject non-scalar stranding with NONCE ERROR.
    // When nested arrays are implemented, this code should create boxed/enclosed
    // values instead of throwing an error.
    //
    // Related: fn_enlist (∊) in primitives.cpp will need to recursively flatten
    // nested structures once they exist.
    //
    if (left_val->is_basic_value() && right_val->is_basic_value()) {
        // Convert strings to character vectors for uniform handling
        if (left_val->is_string()) {
            left_val = left_val->to_char_vector(machine->heap);
        }
        if (right_val->is_string()) {
            right_val = right_val->to_char_vector(machine->heap);
        }

        // Concatenate left and right into a flat vector
        // NOTE: This flattening is technically incorrect for APL2-style nested arrays.
        // E.g., "1 (2 3) 4" SHOULD create a 3-element nested vector, but we produce
        // a flat 4-element vector "1 2 3 4". This is acceptable until nested arrays
        // are implemented, at which point this code should create boxed/enclosed
        // values for non-scalar strand elements.
        size_t left_size = left_val->is_scalar() ? 1 : left_val->size();
        size_t right_size = right_val->is_scalar() ? 1 : right_val->size();
        size_t total = left_size + right_size;

        Eigen::VectorXd result(total);

        // Copy left elements
        if (left_val->is_scalar()) {
            result(0) = left_val->as_scalar();
        } else {
            const Eigen::MatrixXd* left_mat = left_val->as_matrix();
            for (size_t i = 0; i < left_size; i++) {
                result(i) = (*left_mat)(i, 0);
            }
        }

        // Copy right elements
        if (right_val->is_scalar()) {
            result(left_size) = right_val->as_scalar();
        } else {
            const Eigen::MatrixXd* right_mat = right_val->as_matrix();
            for (size_t i = 0; i < right_size; i++) {
                result(left_size + i) = (*right_mat)(i, 0);
            }
        }

        machine->result = machine->heap->allocate_vector(result);
        return;
    }

    // G2 Rule: if type(left) = bas then right(left) else left(right)
    if (left_val->is_basic_value()) {
        // Left is a basic value (scalar, vector, or matrix)
        // Apply right to left: right(left)
        // right must be a function

        // DispatchFunctionK expects the function in result, so set it there
        machine->result = right_val;
        // Use DispatchFunctionK to apply right_val as function to left_val as argument
        machine->push_kont(machine->heap->allocate<DispatchFunctionK>(nullptr, nullptr, left_val));
    } else if (right_val->is_defined_operator() && left_val->is_function()) {
        // Operator application: left is function operand, right is DEFINED_OPERATOR
        // Create a DERIVED_OPERATOR that captures the operand
        Value::DefinedOperatorData* def_op = right_val->data.defined_op_data;
        Value* derived = machine->heap->allocate_derived_operator(def_op, left_val, right_val);
        machine->result = derived;
    } else if (left_val->is_defined_operator() && right_val->is_function()) {
        // Operator application: right is function operand, left is DEFINED_OPERATOR
        Value::DefinedOperatorData* def_op = left_val->data.defined_op_data;
        if (def_op->is_dyadic_operator) {
            // Dyadic operator: right is second operand, curry to wait for first
            Value* curried = machine->heap->allocate_curried_fn(left_val, right_val, Value::CurryType::OPERATOR_CURRY);
            machine->result = curried;
        } else {
            // Monadic operator: right is the operand, create derived function
            Value* derived = machine->heap->allocate_derived_operator(def_op, right_val, left_val);
            machine->result = derived;
        }
    } else if (left_val->is_defined_operator() && right_val->is_array()) {
        // Dyadic operator with value as second operand (e.g., F POW N where N is a number)
        Value::DefinedOperatorData* def_op = left_val->data.defined_op_data;
        if (def_op->is_dyadic_operator) {
            // Dyadic operator: right is second operand (value), curry to wait for first operand (function)
            Value* curried = machine->heap->allocate_curried_fn(left_val, right_val, Value::CurryType::OPERATOR_CURRY);
            machine->result = curried;
        } else {
            // Monadic operator doesn't take a value as operand - this is an error
            machine->throw_error("SYNTAX ERROR: Monadic operator cannot take value as operand", this);
            return;
        }
    } else if (left_val->is_function() && right_val->tag == ValueType::CURRIED_FN &&
               right_val->data.curried_fn->curry_type == Value::CurryType::OPERATOR_CURRY) {
        // Complete dyadic operator application: F (OP N) where OP is curried with second operand N
        Value::CurriedFnData* curry = right_val->data.curried_fn;
        Value* op_or_derived = curry->fn;
        Value* second_operand = curry->first_arg;

        if (op_or_derived->is_defined_operator()) {
            // OPERATOR_CURRY(DEFINED_OPERATOR, second_operand) + function
            // → create DERIVED_OPERATOR with first_operand=left_val, then apply second_operand
            Value::DefinedOperatorData* def_op = op_or_derived->data.defined_op_data;
            Value* derived = machine->heap->allocate_derived_operator(def_op, left_val, op_or_derived);
            // Now apply second operand via OPERATOR_CURRY
            Value* final_curry = machine->heap->allocate_curried_fn(derived, second_operand, Value::CurryType::OPERATOR_CURRY);
            machine->result = final_curry;
        } else if (op_or_derived->tag == ValueType::DERIVED_OPERATOR) {
            // Standard case: OPERATOR_CURRY(DERIVED_OPERATOR, second_operand) + function
            // Just pass through - this is handled elsewhere
            machine->result = left_val;
            machine->push_kont(machine->heap->allocate<DispatchFunctionK>(nullptr, nullptr, right_val));
        } else {
            machine->throw_error("VALUE ERROR: Invalid OPERATOR_CURRY structure", this);
            return;
        }
    } else {
        // Left is a function (or curried function, or derived operator)
        // Apply left to right: left(right)

        // DispatchFunctionK expects the function in result, so set it there
        machine->result = left_val;
        // Use DispatchFunctionK to apply left_val as function to right_val as argument
        machine->push_kont(machine->heap->allocate<DispatchFunctionK>(nullptr, nullptr, right_val));
    }
}

void PerformJuxtaposeK::mark(Heap* heap) {
    heap->mark(right_val);
}

// FinalizeK implementation
// Wraps parenthesized expressions to force g' finalization
void FinalizeK::invoke(Machine* machine) {
    // Push auxiliary to check/finalize result after inner evaluates
    // Pass finalize_gprime flag to control whether G_PRIME gets finalized
    machine->push_kont(machine->heap->allocate<PerformFinalizeK>(finalize_gprime));
    // Push inner expression to evaluate
    machine->push_kont(inner);
}

void FinalizeK::mark(Heap* heap) {
    heap->mark(inner);
}

// PerformFinalizeK - g' null(y) case: finalize curry to value via continuation graph
void PerformFinalizeK::invoke(Machine* machine) {
    Value* val = machine->result;

    if (val && val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* cd = val->data.curried_fn;

        // G_PRIME: always has monadic form (that's why it's G_PRIME)
        // Only finalize G_PRIME if finalize_gprime flag is true
        // Parentheses set this to false to preserve partial applications like (2×)
        if (cd->curry_type == Value::CurryType::G_PRIME && finalize_gprime) {
            // If curry has axis and is primitive, apply directly with axis
            if (cd->axis != nullptr && cd->fn->is_primitive()) {
                PrimitiveFn* prim_fn = cd->fn->data.primitive_fn;
                if (prim_fn->monadic) {
                    prim_fn->monadic(machine, cd->axis, cd->first_arg);
                    return;
                }
            }
            DispatchFunctionK* dispatch = machine->heap->allocate<DispatchFunctionK>(
                cd->fn, nullptr, cd->first_arg);
            dispatch->force_monadic = true;
            machine->push_kont(dispatch);
            return;
        }

        // DYADIC_CURRY: finalize based on inner function type
        if (cd->curry_type == Value::CurryType::DYADIC_CURRY) {
            Value* fn = cd->fn;
            Value* arg = cd->first_arg;

            // Special case: DYADIC_CURRY wrapping OPERATOR_CURRY (axis-based reduce/scan)
            // DYADIC_CURRY(OPERATOR_CURRY(DERIVED_OP(op, f), axis), array)
            // → call op->dyadic(null, f, axis, array)
            if (fn->tag == ValueType::CURRIED_FN &&
                fn->data.curried_fn->curry_type == Value::CurryType::OPERATOR_CURRY) {
                Value::CurriedFnData* oc = fn->data.curried_fn;
                Value* derived = oc->fn;  // DERIVED_OPERATOR
                Value* axis = oc->first_arg;  // axis specification

                if (derived->tag == ValueType::DERIVED_OPERATOR) {
                    PrimitiveOp* op = derived->data.derived_op->primitive_op;
                    Value::DefinedOperatorData* def_op = derived->data.derived_op->defined_op;
                    Value* first_operand = derived->data.derived_op->first_operand;
                    Value* op_value = derived->data.derived_op->operator_value;

                    if (def_op) {
                        // User-defined dyadic operator with axis as second operand
                        machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                            def_op, op_value, first_operand, axis, nullptr, arg));
                        return;
                    } else if (op) {
                        op->dyadic(machine, nullptr, first_operand, axis, arg);
                        return;
                    }
                }
            }

            // Standard case: check if inner function has monadic form
            bool has_monadic = false;
            if (fn->tag == ValueType::DERIVED_OPERATOR) {
                // For primitive ops, check monadic; defined ops always have monadic form
                auto* prim_op = fn->data.derived_op->primitive_op;
                has_monadic = prim_op ? (prim_op->monadic != nullptr) : true;
            } else if (fn->tag == ValueType::PRIMITIVE) {
                has_monadic = fn->data.primitive_fn->monadic != nullptr;
            } else if (fn->tag == ValueType::CLOSURE) {
                has_monadic = true;  // Closures always have monadic form
            }

            if (has_monadic) {
                DispatchFunctionK* dispatch = machine->heap->allocate<DispatchFunctionK>(
                    fn, nullptr, arg);
                dispatch->force_monadic = true;
                machine->push_kont(dispatch);
                return;
            }
            // No monadic form - leave as valid partial application
        }
    }
}

void PerformFinalizeK::mark(Heap* heap) {
    (void)heap;  // No Values or Continuations to mark
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

void MonadicK::mark(Heap* heap) {
    heap->mark(operand);
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

void DyadicK::mark(Heap* heap) {
    heap->mark(left);
    heap->mark(right);
}

// EvalDyadicLeftK implementation
void EvalDyadicLeftK::invoke(Machine* machine) {
    // Right operand has been evaluated - its value is in result
    // Save the right value and set up left evaluation
    right_val = machine->result;

    // Allocate auxiliary continuation to apply function after left evaluates
    ApplyDyadicK* apply = machine->heap->allocate<ApplyDyadicK>(op_name, right_val);

    // Push work in reverse order
    machine->push_kont(apply);   // Will execute after left
    machine->push_kont(left);     // Will execute now

    // Phase 3.1: No return needed, trampoline continues
}

void EvalDyadicLeftK::mark(Heap* heap) {
    heap->mark(left);
    heap->mark(right_val);
}


// ApplyMonadicK implementation
void ApplyMonadicK::invoke(Machine* machine) {
    // Operand has been evaluated - its value is in result
    Value* operand_val = machine->result;

    // g' finalization: If operand is a curry, finalize it first
    if (operand_val && operand_val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* cd = operand_val->data.curried_fn;
        if (cd->curry_type == Value::CurryType::G_PRIME ||
            cd->curry_type == Value::CurryType::DYADIC_CURRY) {
            machine->push_kont(this);
            machine->push_kont(machine->heap->allocate<PerformFinalizeK>());
            return;
        }
    }

    // Look up the operator at evaluation time
    Value* op_val = machine->env->lookup(op_name);
    if (!op_val || op_val->tag != ValueType::PRIMITIVE) {
        std::string msg = std::string("VALUE ERROR: Unknown operator: ") + op_name;
        machine->throw_error(msg.c_str(), this);
        return;
    }

    PrimitiveFn* prim_fn = op_val->data.primitive_fn;

    if (!prim_fn->monadic) {
        std::string msg = std::string("SYNTAX ERROR: Operator has no monadic form: ") + op_name;
        machine->throw_error(msg.c_str(), this);
        return;
    }

    // G2 g' transformation: If function is overloaded (has both monadic and dyadic forms),
    // create a curried function to defer the monadic/dyadic decision to runtime
    if (prim_fn->monadic && prim_fn->dyadic) {
        // Overloaded function - create CURRIED_FN with G_PRIME (g' transformation)
        // This allows the function to be applied monadically now, or dyadically if another arg appears
        Value* curried = machine->heap->allocate_curried_fn(op_val, operand_val, Value::CurryType::G_PRIME);
        machine->result = curried;
    } else {
        // Monadic-only function - apply immediately (no axis from this path)
        prim_fn->monadic(machine, nullptr, operand_val);
    }

    // Phase 3.1: No return needed, trampoline continues
}

void ApplyMonadicK::mark(Heap* heap) {
    // ApplyMonadicK has no Values to mark, only the function pointer
    (void)heap;  // Unused
}

// ArgK implementation
void ArgK::invoke(Machine* machine) {
    // Set the argument value and continue with next continuation
    machine->result = arg_value;

    if (next) {
        machine->push_kont(next);
    }

    // Phase 3.1: No return needed, trampoline continues
}

void ArgK::mark(Heap* heap) {
    // Mark the argument Value
    heap->mark(arg_value);

    // Mark next continuation
    heap->mark(next);
}

// ApplyDyadicK implementation
void ApplyDyadicK::invoke(Machine* machine) {
    // Both operands have been evaluated
    // Right value is saved in right_val
    // Left value is in result
    Value* left_val = machine->result;

    // Look up the operator at evaluation time
    Value* op_val = machine->env->lookup(op_name);
    if (!op_val || op_val->tag != ValueType::PRIMITIVE) {
        std::string msg = std::string("VALUE ERROR: Unknown operator: ") + op_name;
        machine->throw_error(msg.c_str(), this);
        return;
    }

    PrimitiveFn* prim_fn = op_val->data.primitive_fn;

    if (!prim_fn->dyadic) {
        std::string msg = std::string("SYNTAX ERROR: Operator has no dyadic form: ") + op_name;
        machine->throw_error(msg.c_str(), this);
        return;
    }

    // Apply the dyadic function (no axis from this path)
    prim_fn->dyadic(machine, nullptr, left_val, right_val);

    // Phase 3.1: No return needed, trampoline continues
}

void ApplyDyadicK::mark(Heap* heap) {
    // Mark the saved right value
    heap->mark(right_val);
}

// ============================================================================
// Strand Building Continuations
// ============================================================================

void EvalStrandElementK::invoke(Machine* machine) {
    // An element has just been evaluated - its value is in result
    // Add it to the FRONT of evaluated_values (we're going right-to-left)
    Value* current_val = machine->result;

    // g' finalization: Finalize curries before adding to strand
    if (current_val && current_val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* cd = current_val->data.curried_fn;
        if (cd->curry_type == Value::CurryType::G_PRIME ||
            cd->curry_type == Value::CurryType::DYADIC_CURRY) {
            machine->push_kont(this);
            machine->push_kont(machine->heap->allocate<PerformFinalizeK>());
            return;
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

void EvalStrandElementK::mark(Heap* heap) {
    // Mark remaining continuations
    for (Continuation* elem : remaining_elements) {
        heap->mark(elem);
    }

    // Mark evaluated values
    for (Value* val : evaluated_values) {
        heap->mark(val);
    }
}

// BuildStrandK implementation
void BuildStrandK::invoke(Machine* machine) {
    // All elements have been evaluated - build the vector
    // values are already in left-to-right order

    if (values.empty()) {
        Eigen::VectorXd empty_vec(0);
        Value* val = machine->heap->allocate_vector(empty_vec);
        machine->result = val;
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
                machine->throw_error("SYNTAX ERROR: Function has no monadic form", this);
                return;
            }

            // Apply monadic function (sets machine->result directly or pushes ThrowErrorK)
            prim->monadic(machine, nullptr, arg);
            return;  // Early exit after monadic application
        }

        // Pattern: [f x y] - dyadic application (function first)
        if (values.size() == 3 && fn_index == 0) {
            Value* left_arg = values[1];
            Value* right_arg = values[2];

            if (!prim->dyadic) {
                machine->throw_error("SYNTAX ERROR: Function has no dyadic form", this);
                return;
            }

            // Apply dyadic function: x f y (sets machine->result directly or pushes ThrowErrorK)
            prim->dyadic(machine, nullptr, left_arg, right_arg);
            return;  // Early exit after dyadic application
        }

        // Pattern: [x f y] - dyadic application (function in middle - APL infix notation)
        if (values.size() == 3 && fn_index == 1) {
            Value* left_arg = values[0];
            Value* right_arg = values[2];

            if (!prim->dyadic) {
                machine->throw_error("SYNTAX ERROR: Function has no dyadic form", this);
                return;
            }

            // Apply dyadic function: x f y (sets machine->result directly or pushes ThrowErrorK)
            prim->dyadic(machine, nullptr, left_arg, right_arg);
            return;  // Early exit after dyadic application
        }

        // TODO: Higher-order operators and other patterns
        // For now, fall through to regular vector formation (or error)
        machine->throw_error("SYNTAX ERROR: Unsupported function application pattern in strand", this);
        return;
    }

    // Regular vector formation (no function application)
    // First pass: convert strings to char vectors and calculate total size
    size_t total_size = 0;
    for (size_t i = 0; i < values.size(); i++) {
        Value* val = values[i];
        if (val->is_string()) {
            // Convert string to char vector (lazy conversion)
            values[i] = val->to_char_vector(machine->heap);
            val = values[i];
        }
        if (val->is_scalar()) {
            total_size += 1;
        } else if (val->is_array()) {
            total_size += val->size();
        } else {
            machine->throw_error("RANK ERROR: Strand elements must be scalars or arrays", this);
            return;
        }
    }

    // Second pass: build the result vector
    Eigen::VectorXd vec(total_size);
    size_t idx = 0;
    for (size_t i = 0; i < values.size(); i++) {
        Value* val = values[i];
        if (val->is_scalar()) {
            vec(idx++) = val->as_scalar();
        } else {
            // Array - copy all elements
            const Eigen::MatrixXd* mat = val->as_matrix();
            for (int j = 0; j < mat->size(); j++) {
                vec(idx++) = (*mat)(j);
            }
        }
    }

    Value* result = machine->heap->allocate_vector(vec);
    machine->result = result;

    // Phase 3.1: No return needed, trampoline continues
}

void BuildStrandK::mark(Heap* heap) {
    // Mark all values
    for (Value* val : values) {
        heap->mark(val);
    }
}

// ============================================================================
// Function Application and Dispatch Continuations
// ============================================================================

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

void FrameK::mark(Heap* heap) {
    // Mark return continuation
    heap->mark(return_k);
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

void ApplyFunctionK::mark(Heap* heap) {
    heap->mark(fn_cont);
    heap->mark(left_arg);
    heap->mark(right_arg);
}

// EvalApplyFunctionLeftK implementation
void EvalApplyFunctionLeftK::invoke(Machine* machine) {
    // Right argument has been evaluated - save it
    right_val = machine->result;

    // Now evaluate left argument, then function, then dispatch
    // Create continuation that will evaluate function after left arg
    EvalApplyFunctionDyadicK* eval_fn = machine->heap->allocate<EvalApplyFunctionDyadicK>(fn_cont, nullptr, right_val);

    machine->push_kont(eval_fn);
    machine->push_kont(left_arg);

    // Phase 3.1: No return needed
}

void EvalApplyFunctionLeftK::mark(Heap* heap) {
    heap->mark(fn_cont);
    heap->mark(left_arg);
    heap->mark(right_val);
}

// EvalApplyFunctionMonadicK implementation
void EvalApplyFunctionMonadicK::invoke(Machine* machine) {
    // Argument has been evaluated - save it
    arg_val = machine->result;

    // Now evaluate the function continuation, then dispatch (monadic case)
    DispatchFunctionK* dispatch = machine->heap->allocate<DispatchFunctionK>(nullptr, nullptr, arg_val);

    machine->push_kont(dispatch);
    machine->push_kont(fn_cont);

    // Phase 3.1: No return needed
}

void EvalApplyFunctionMonadicK::mark(Heap* heap) {
    heap->mark(fn_cont);
    heap->mark(arg_val);
}

// EvalApplyFunctionDyadicK implementation
void EvalApplyFunctionDyadicK::invoke(Machine* machine) {
    // Left argument has been evaluated - save it
    left_val = machine->result;

    // Now evaluate the function continuation, then dispatch (dyadic case)
    DispatchFunctionK* dispatch = machine->heap->allocate<DispatchFunctionK>(nullptr, left_val, right_val);

    machine->push_kont(dispatch);
    machine->push_kont(fn_cont);

    // Phase 3.1: No return needed
}

void EvalApplyFunctionDyadicK::mark(Heap* heap) {
    heap->mark(fn_cont);
    heap->mark(left_val);
    heap->mark(right_val);
}

// DispatchFunctionK implementation
// This is where the actual currying transformation happens:
// g' = λx. λy. if null(y) then g1(x) else if bas(y) then g2(x,y) else y(g1(x))
void DispatchFunctionK::invoke(Machine* machine) {
    // If fn_val wasn't provided in constructor, get it from result
    // (for cases where function was just evaluated)
    if (fn_val == nullptr) {
        fn_val = machine->result;
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
            // DYADIC_CURRY g' transformation:
            // null(y) → g1(x), bas(y) → g2(x,y), else → y(g1(x))
            if (right_val == nullptr) {
                // null(y): finalize monadically via DispatchFunctionK
                DispatchFunctionK* dispatch = machine->heap->allocate<DispatchFunctionK>(
                    inner_fn, nullptr, first_arg);
                dispatch->force_monadic = true;
                machine->push_kont(dispatch);
                return;
            } else if (right_val->is_basic_value()) {
                // bas(y): apply dyadically with swapped args
                fn_val = inner_fn;
                left_val = right_val;
                right_val = first_arg;
                // Fall through to dispatch
            } else {
                // y is a function: first finalize g1(x), then apply y to result
                machine->push_kont(machine->heap->allocate<PerformJuxtaposeK>(right_val));
                DispatchFunctionK* dispatch = machine->heap->allocate<DispatchFunctionK>(
                    inner_fn, nullptr, first_arg);
                dispatch->force_monadic = true;
                machine->push_kont(dispatch);
                return;
            }
        } else if (curried_data->curry_type == Value::CurryType::OPERATOR_CURRY) {
            // Operator curry: inner_fn is DERIVED_OPERATOR, first_arg is second operand
            // For inner product: first_arg is second function operand (×)
            // For rank: first_arg is rank specification (a value)
            // inner_fn = DERIVED_OPERATOR(op, first_operand)
            if (inner_fn->tag != ValueType::DERIVED_OPERATOR) {
                machine->throw_error("VALUE ERROR: OPERATOR_CURRY expected DERIVED_OPERATOR", this);
                return;
            }
            Value::DerivedOperatorData* derived_data = inner_fn->data.derived_op;
            PrimitiveOp* op = derived_data->primitive_op;
            Value::DefinedOperatorData* def_op = derived_data->defined_op;
            Value* first_operand = derived_data->first_operand;
            Value* op_value = derived_data->operator_value;
            Value* second_operand = first_arg;  // Second operand (function for ., value for ⍤)

            if (left_val && right_val) {
                // Have both array arguments - call dyadic operator
                // For reduce/scan: left_val is N, second_operand is axis

                // Finalize any G_PRIME curried functions in arguments first
                if (left_val->tag == ValueType::CURRIED_FN) {
                    Value::CurriedFnData* lcd = left_val->data.curried_fn;
                    if (lcd->curry_type == Value::CurryType::G_PRIME) {
                        Value* fn = lcd->fn;
                        Value* arg = lcd->first_arg;
                        Value* axis = lcd->axis;
                        if (fn->is_primitive() && fn->data.primitive_fn->monadic) {
                            fn->data.primitive_fn->monadic(machine, axis, arg);
                            left_val = machine->result;
                        }
                    }
                }
                if (right_val->tag == ValueType::CURRIED_FN) {
                    Value::CurriedFnData* rcd = right_val->data.curried_fn;
                    if (rcd->curry_type == Value::CurryType::G_PRIME) {
                        Value* fn = rcd->fn;
                        Value* arg = rcd->first_arg;
                        Value* axis = rcd->axis;
                        if (fn->is_primitive() && fn->data.primitive_fn->monadic) {
                            fn->data.primitive_fn->monadic(machine, axis, arg);
                            right_val = machine->result;
                        }
                    }
                }

                if (def_op) {
                    // User-defined dyadic operator with both operands and both args
                    machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                        def_op, op_value, first_operand, second_operand, left_val, right_val));
                } else {
                    op->dyadic(machine, left_val, first_operand, second_operand, right_val);
                }
            } else if (right_val) {
                // Only have right array argument - curry to wait for potential left
                // This enables N-wise reduction with axis: "2 +/[1] matrix"

                // Finalize any G_PRIME curried function in right_val first
                if (right_val->tag == ValueType::CURRIED_FN) {
                    Value::CurriedFnData* rcd = right_val->data.curried_fn;
                    if (rcd->curry_type == Value::CurryType::G_PRIME) {
                        Value* fn = rcd->fn;
                        Value* arg = rcd->first_arg;
                        Value* axis = rcd->axis;
                        if (fn->is_primitive() && fn->data.primitive_fn->monadic) {
                            fn->data.primitive_fn->monadic(machine, axis, arg);
                            right_val = machine->result;
                        }
                    }
                }

                if (def_op) {
                    // User-defined dyadic operator with both operands, monadic application
                    machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                        def_op, op_value, first_operand, second_operand, nullptr, right_val));
                } else {
                    Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::DYADIC_CURRY);
                    machine->result = curried;
                }
            } else {
                machine->throw_error("VALUE ERROR: operator curry expects array argument", this);
            }
            return;
        } else {
            // G_PRIME transformation per Georgeff et al. "Parsing and Evaluation of APL with Operators"
            // g' = λx . λy . if null(y) then g1(x)
            //                else if bas(y) then g2(x,y)
            //                else y(g1(x))
            //
            // Special case: axis-only curry (first_arg is nullptr, axis is set)
            // This represents F[k] waiting for its first operand
            // Always curry the value - monadic vs dyadic decided at finalization
            if (first_arg == nullptr && curried_data->axis != nullptr && right_val != nullptr && right_val->is_basic_value()) {
                if (inner_fn->is_primitive()) {
                    // Create G_PRIME curry of inner_fn with first_arg=B, preserving axis
                    Value* curried = machine->heap->allocate_curried_fn(inner_fn, right_val, Value::CurryType::G_PRIME, curried_data->axis);
                    machine->result = curried;
                    return;
                } else {
                    // For closures, dispatch normally (closures don't support axis per ISO spec)
                    machine->throw_error("SYNTAX ERROR: axis specification requires primitive function", this);
                    return;
                }
            } else if (right_val == nullptr) {
                // null(y): No second argument - apply monadically to first_arg
                // For axis curries, apply monadically with axis now
                if (curried_data->axis != nullptr && inner_fn->is_primitive()) {
                    PrimitiveFn* prim_fn = inner_fn->data.primitive_fn;
                    if (prim_fn->monadic) {
                        prim_fn->monadic(machine, curried_data->axis, first_arg);
                        return;
                    } else {
                        machine->throw_error("SYNTAX ERROR: Function has no monadic form", this);
                        return;
                    }
                }
                fn_val = inner_fn;
                left_val = nullptr;
                right_val = first_arg;
                first_arg = nullptr;  // Clear to prevent misuse in DERIVED_OPERATOR handling
                // Fall through to apply monadic
            } else if (right_val->is_basic_value()) {
                // bas(y): Second argument is a basic value - apply dyadically
                // In G2 juxtaposition, first_arg is the RIGHT operand (captured)
                // and right_val is the LEFT operand (newly applied)
                // For axis curries (e.g., 2↑[1]M), call dyadic with axis now
                if (curried_data->axis != nullptr && inner_fn->is_primitive()) {
                    PrimitiveFn* prim_fn = inner_fn->data.primitive_fn;
                    if (prim_fn->dyadic) {
                        prim_fn->dyadic(machine, curried_data->axis, right_val, first_arg);
                        return;
                    } else {
                        machine->throw_error("SYNTAX ERROR: Function has no dyadic form", this);
                        return;
                    }
                }
                fn_val = inner_fn;
                left_val = right_val;  // New argument is LEFT (alpha)
                right_val = first_arg; // Captured argument is RIGHT (omega)
                first_arg = nullptr;  // Clear to prevent misuse in DERIVED_OPERATOR handling
                // Fall through to apply dyadic
            } else {
                // y is a function: apply y(g1(x))
                // First, apply monadic form g1 to captured argument x
                if (inner_fn->is_primitive()) {
                    PrimitiveFn* prim_fn = inner_fn->data.primitive_fn;
                    if (prim_fn->monadic) {
                        // Pass axis from curried function if present
                        prim_fn->monadic(machine, curried_data->axis, first_arg);
                        // Now apply the function y to g1(x)
                        // right_val is the function y, machine->result is g1(x)
                        fn_val = right_val;
                        left_val = nullptr;
                        right_val = machine->result;
                        // Fall through to apply y to the result
                    } else {
                        machine->throw_error("VALUE ERROR: G_PRIME requires monadic form", this);
                        return;
                    }
                } else {
                    // For closures, need to evaluate g1(x) first then apply y
                    // Push continuation to apply y after g1(x) evaluates
                    machine->push_kont(machine->heap->allocate<DispatchFunctionK>(right_val, nullptr, nullptr));
                    machine->push_kont(machine->heap->allocate<DispatchFunctionK>(inner_fn, nullptr, first_arg));
                    return;
                }
            }
        }
        if (fn_val->tag == ValueType::CURRIED_FN) {
            // Update result before recursing, since invoke() reads from it
            machine->result = fn_val;
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
            PrimitiveOp* op = derived_data->primitive_op;
            Value::DefinedOperatorData* def_op = derived_data->defined_op;
            Value* first_operand = derived_data->first_operand;
            Value* op_value = derived_data->operator_value;

            // Handle user-defined operators (G_PRIME finalization path)
            if (def_op) {
                // Invoke the defined operator with both arguments
                machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                    def_op, op_value, first_operand, nullptr, left_val, right_val));
                return;
            }

            // For DERIVED_OPERATOR from dyadic operator (like inner product f.g):
            // first_arg contains the second function operand (g)
            // But if first_arg is a CURRIED_FN(g, array), we need to unwrap it:
            // - Extract g as the second function operand
            // - Extract array as the right array argument
            // - Use right_val as the left array argument
            Value* second_func_operand = nullptr;
            Value* actual_left_array = nullptr;
            Value* actual_right_array = nullptr;

            if (op && op->dyadic && first_arg) {
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
            } else if (op->monadic && !left_val) {
                // Monadic operator application - prefer this over curry when available
                // This handles cases like +/1 2 3 (reduce with no N argument)
                op->monadic(machine, first_operand, right_val);
            } else if (op->dyadic && !left_val) {
                // Only have right argument and no monadic form - curry for second operand
                // Use OPERATOR_CURRY to capture it properly (not DYADIC_CURRY which is for array args)
                Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::OPERATOR_CURRY);
                machine->result = curried;
            } else {
                machine->throw_error("VALUE ERROR: operator requires operands", this);
            }
            return;
        }
    }

    // G2 g' finalization: Unwrap any g' curried functions in arguments
    // Per paper: when a g' curried function is used as an argument (not at top level),
    // it should be unwrapped by applying g1(x)
    // Use continuation machinery to handle async evaluation (e.g., ⍎)
    if (right_val && right_val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* curried_data = right_val->data.curried_fn;
        if (curried_data->curry_type == Value::CurryType::G_PRIME) {
            // Unwrap g' curried function via continuation machinery
            Value* inner_fn = curried_data->fn;
            Value* inner_arg = curried_data->first_arg;
            // Push DeferredDispatchK to continue with unwrapped right_val
            machine->push_kont(machine->heap->allocate<DeferredDispatchK>(fn_val, left_val, force_monadic));
            // Dispatch inner function with force_monadic=true to actually evaluate
            DispatchFunctionK* inner = machine->heap->allocate<DispatchFunctionK>(inner_fn, nullptr, inner_arg);
            inner->force_monadic = true;
            machine->push_kont(inner);
            return;
        } else if (curried_data->curry_type == Value::CurryType::DYADIC_CURRY) {
            // Finalize DYADIC_CURRY from reduce/scan when used as argument
            // This handles nested reductions like "+/×/1 2 3 4"
            Value* inner_fn = curried_data->fn;
            Value* arg = curried_data->first_arg;
            if (inner_fn->tag == ValueType::DERIVED_OPERATOR) {
                Value::DerivedOperatorData* derived_data = inner_fn->data.derived_op;
                PrimitiveOp* op = derived_data->primitive_op;
                Value* first_operand = derived_data->first_operand;
                if (op->monadic) {
                    // Push DeferredDispatchK to continue after inner reduction completes
                    // It will read machine->result as the new right_val
                    machine->push_kont(machine->heap->allocate<DeferredDispatchK>(fn_val, left_val, force_monadic));
                    // Evaluate inner reduction - result will become new right_val
                    op->monadic(machine, first_operand, arg);
                    return;
                }
            }
        }
    }

    if (fn_val->tag == ValueType::DERIVED_OPERATOR) {
        Value::DerivedOperatorData* derived_data = fn_val->data.derived_op;
        PrimitiveOp* op = derived_data->primitive_op;
        Value::DefinedOperatorData* def_op = derived_data->defined_op;
        Value* first_operand = derived_data->first_operand;
        Value* op_value = derived_data->operator_value;

        // User-defined operators
        if (def_op) {
            if (right_val) {
                if (def_op->is_dyadic_operator) {
                    // Dyadic operator needs second operand - curry to wait for it
                    Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::OPERATOR_CURRY);
                    machine->result = curried;
                } else if (left_val) {
                    // Monadic operator with both arguments - invoke dyadically
                    machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                        def_op, op_value, first_operand, nullptr, left_val, right_val));
                } else if (force_monadic || !def_op->is_ambivalent) {
                    // Force monadic (from finalization) or non-ambivalent - invoke immediately
                    machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                        def_op, op_value, first_operand, nullptr, nullptr, right_val));
                } else {
                    // Ambivalent operator with only right arg - curry to wait for potential left arg
                    Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::G_PRIME);
                    machine->result = curried;
                }
            } else {
                machine->throw_error("VALUE ERROR: operator requires argument", this);
            }
            return;
        }

        // Force monadic: apply monadic form directly (used by each, rank operators)
        if (force_monadic && op->monadic && !left_val && right_val) {
            op->monadic(machine, first_operand, right_val);
            return;
        }

        // For operators with BOTH monadic and dyadic forms (like commute ⍨),
        // curry with G_PRIME when given one argument. This allows `2 +⍨ 3` to work correctly
        // by deferring until we know if there's a left argument.
        // Exception: reduce/scan with array operand (replicate) should apply immediately
        if (op->monadic && op->dyadic && !left_val && right_val) {
            // Reduce/scan operators: check if operand is a function or array
            if (strcmp(op->name, "/") == 0 || strcmp(op->name, "\\") == 0 ||
                strcmp(op->name, "⌿") == 0 || strcmp(op->name, "⍀") == 0) {
                // If operand is array (not function), this is replicate - apply immediately
                if (!first_operand->is_function()) {
                    op->monadic(machine, first_operand, right_val);
                    return;
                }
                // Function operand: curry with DYADIC_CURRY to wait for potential N (N-wise reduction)
                // The curry will be finalized at top level if no N is provided
                Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::DYADIC_CURRY);
                machine->result = curried;
                return;
            }
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::G_PRIME);
            machine->result = curried;
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
                machine->result = curried;
            } else {
                // Outer product and similar: first_arg stores the array argument
                Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::DYADIC_CURRY);
                machine->result = curried;
            }
            return;
        }

        machine->throw_error("VALUE ERROR: operator requires operands", this);
        return;
    }

    // Handle DEFINED_OPERATOR being "applied" to a value
    // This happens with right-to-left eval: "TWICE 5" before "-TWICE 5"
    // The operator needs an operand first, so curry to wait for it
    if (fn_val->is_defined_operator()) {
        if (right_val) {
            // Curry: store the value, wait for operand (function) to arrive
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::OPERATOR_CURRY);
            machine->result = curried;
        } else {
            machine->throw_error("VALUE ERROR: operator requires argument", this);
        }
        return;
    }

    if (!fn_val->is_primitive()) {
        machine->throw_error("VALUE ERROR: expected function value", this);
        return;
    }

    PrimitiveFn* prim_fn = fn_val->data.primitive_fn;

    // Determine monadic vs dyadic based on what arguments we have
    if (left_val == nullptr) {
        // Monadic case: only right argument

        if (prim_fn->monadic && !force_monadic) {
            // G_PRIME curry for all monadic functions (not just overloaded)
            // This allows proper error handling if used in dyadic context
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::G_PRIME);
            machine->result = curried;
        } else if (prim_fn->monadic && force_monadic) {
            // Force immediate monadic evaluation (used by operators like each, rank)
            prim_fn->monadic(machine, nullptr, right_val);
        } else if (prim_fn->dyadic) {
            // Pure dyadic function: simple currying (right arg captured, waiting for left)
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::DYADIC_CURRY);
            machine->result = curried;
        } else {
            // No forms available
            machine->throw_error("SYNTAX ERROR: Function has no forms", this);
            return;
        }
    } else {
        // Dyadic case: both arguments
        if (!prim_fn->dyadic) {
            machine->throw_error("SYNTAX ERROR: Function has no dyadic form", this);
            return;
        }

        // Apply dyadic function (sets machine->result directly or pushes ThrowErrorK)
        prim_fn->dyadic(machine, nullptr, left_val, right_val);
    }

    // Phase 3.1: No return needed
}

void DispatchFunctionK::mark(Heap* heap) {
    heap->mark(fn_val);
    heap->mark(left_val);
    heap->mark(right_val);
}

// DeferredDispatchK implementation - continues dispatch with result as right_val
void DeferredDispatchK::invoke(Machine* machine) {
    // The subcomputation completed, result is now the new right_val
    Value* right_val = machine->result;

    // Create and push DispatchFunctionK to continue the dispatch
    machine->push_kont(machine->heap->allocate<DispatchFunctionK>(fn_val, left_val, right_val, force_monadic));
}

void DeferredDispatchK::mark(Heap* heap) {
    heap->mark(fn_val);
    heap->mark(left_val);
}

// ============================================================================
// Statement Sequencing Continuations
// ============================================================================

void SeqK::invoke(Machine* machine) {
    if (statements.empty()) {
        // Empty sequence returns null/unit value (scalar 0)
        Value* val = machine->heap->allocate_scalar(0.0);
        machine->result = val;
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

void SeqK::mark(Heap* heap) {
    for (Continuation* stmt : statements) {
        heap->mark(stmt);
    }
}

// ExecNextStatementK implementation - execute remaining statements
void ExecNextStatementK::invoke(Machine* machine) {
    // The previous statement has been executed and its result is in machine->result
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

void ExecNextStatementK::mark(Heap* heap) {
    for (Continuation* stmt : statements) {
        heap->mark(stmt);
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

void IfK::mark(Heap* heap) {
    heap->mark(condition);
    heap->mark(then_branch);
    heap->mark(else_branch);
}

// SelectBranchK implementation - select branch based on condition result
void SelectBranchK::invoke(Machine* machine) {
    // Condition result is in machine->result
    Value* cond_val = machine->result;

    if (!cond_val) {
        // Error: no condition value
        machine->throw_error("VALUE ERROR: If condition evaluated to null", this);
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

void SelectBranchK::mark(Heap* heap) {
    heap->mark(then_branch);
    heap->mark(else_branch);
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

void WhileK::mark(Heap* heap) {
    heap->mark(condition);
    heap->mark(body);
}

// CheckWhileCondK implementation - check condition and decide whether to loop
void CheckWhileCondK::invoke(Machine* machine) {
    // Condition result is in machine->result
    Value* cond_val = machine->result;

    if (!cond_val) {
        // Error: no condition value
        machine->throw_error("VALUE ERROR: While condition evaluated to null", this);
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

        // Push CatchContinueK to handle :Continue - it will restart the loop
        auto* catch_continue = machine->heap->allocate<CatchContinueK>(check_k);
        machine->push_kont(catch_continue);

        // Push body to execute now
        if (body) {
            machine->push_kont(body);
        }
    }
    // If false, just exit - loop is done
    // Result remains the condition value
}

void CheckWhileCondK::mark(Heap* heap) {
    heap->mark(condition);
    heap->mark(body);
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

void ForK::mark(Heap* heap) {
    heap->mark(array_expr);
    heap->mark(body);
}

// ForIterateK implementation - iterate over array elements
void ForIterateK::invoke(Machine* machine) {
    // First call: array is in machine->result
    if (array == nullptr) {
        array = machine->result;
        if (!array) {
            machine->throw_error("VALUE ERROR: For loop array evaluated to null", this);
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
            machine->result = zero;
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

    // Push CatchContinueK to handle :Continue - it will skip to next iteration
    auto* catch_continue = machine->heap->allocate<CatchContinueK>(next_k);
    machine->push_kont(catch_continue);

    // Push body to execute
    if (body) {
        machine->push_kont(body);
    }
}

void ForIterateK::mark(Heap* heap) {
    heap->mark(array);
    heap->mark(body);
}

// LeaveK implementation - exit from loop
void LeaveK::invoke(Machine* machine) {
    // Phase 2.3: Create BREAK completion and propagate it up the stack
    // This will unwind until we hit a CatchBreakK at a loop boundary

    // The current value in ctrl is the result of the :Leave statement (usually the last value)
    Completion* break_comp = machine->heap->allocate<Completion>(
        CompletionType::BREAK,
        machine->result,  // The value to return from the loop
        nullptr  // No label for now
    );

    // Push PropagateCompletionK to unwind the stack
    PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(break_comp);
    machine->push_kont(prop);
}

void LeaveK::mark(Heap* heap) {
    // LeaveK has no references
    (void)heap;
}

// ContinueK implementation - skip to next loop iteration
void ContinueK::invoke(Machine* machine) {
    // Create CONTINUE completion and propagate it up the stack
    // This will unwind until we hit a CatchContinueK at a loop boundary

    Completion* continue_comp = machine->heap->allocate<Completion>(
        CompletionType::CONTINUE,
        machine->result,  // Preserve current value
        nullptr  // No label for now
    );

    // Push PropagateCompletionK to unwind the stack
    PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(continue_comp);
    machine->push_kont(prop);
}

void ContinueK::mark(Heap* heap) {
    // ContinueK has no references
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
        machine->result = zero;

        // Create RETURN completion with zero value
        Completion* return_comp = machine->heap->allocate<Completion>(
            CompletionType::RETURN,
            zero,
            nullptr
        );

        // Push PropagateCompletionK to unwind the stack
        PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(return_comp);
        machine->push_kont(prop);
    }
}

void ReturnK::mark(Heap* heap) {
    heap->mark(value_expr);
}

// CreateReturnK implementation - create RETURN completion from evaluated value
void CreateReturnK::invoke(Machine* machine) {
    // Phase 2.3: Value has been evaluated, create RETURN completion
    // The value is in result

    Completion* return_comp = machine->heap->allocate<Completion>(
        CompletionType::RETURN,
        machine->result,
        nullptr
    );

    // Push PropagateCompletionK to unwind the stack
    PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(return_comp);
    machine->push_kont(prop);
}

void CreateReturnK::mark(Heap* heap) {
    // CreateReturnK has no references
    (void)heap;
}

// BranchK implementation - evaluate target, then check if we should exit
void BranchK::invoke(Machine* machine) {
    // Save current result before evaluating branch target
    // This will be the return value if we exit (→0 returns the last computed value)
    Value* saved_result = machine->result;

    // Push CheckBranchK to process the result
    CheckBranchK* check_k = machine->heap->allocate<CheckBranchK>(saved_result);
    machine->push_kont(check_k);

    // Evaluate the target expression
    machine->push_kont(target_expr);
}

void BranchK::mark(Heap* heap) {
    heap->mark(target_expr);
}

// CheckBranchK implementation - check branch target and exit if 0 or empty
void CheckBranchK::invoke(Machine* machine) {
    Value* target = machine->result;

    if (!target) {
        machine->throw_error("VALUE ERROR: Branch target evaluated to null", this);
        return;
    }

    // Check if target is 0 or empty - these mean "exit function"
    bool should_exit = false;

    if (target->is_scalar()) {
        // →0 means exit
        should_exit = (target->as_scalar() == 0.0);
    } else if (target->is_vector() || target->is_matrix()) {
        // →⍬ (empty array) means exit
        const Eigen::MatrixXd* mat = target->as_matrix();
        should_exit = (mat->size() == 0);
    }

    if (should_exit) {
        // Exit function - create RETURN completion with saved result (not the branch target)
        // Use saved_result if available, otherwise use a default scalar 0
        Value* return_value = saved_result ? saved_result : machine->heap->allocate_scalar(0.0);

        Completion* return_comp = machine->heap->allocate<Completion>(
            CompletionType::RETURN,
            return_value,
            nullptr
        );

        PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(return_comp);
        machine->push_kont(prop);
    } else {
        // Non-zero, non-empty target - this would be a line number branch
        // We don't support line numbers, so report an error
        machine->throw_error("DOMAIN ERROR: Branch to line numbers not supported (use →0 or →⍬ to exit)", this);
    }
}

void CheckBranchK::mark(Heap* heap) {
    heap->mark(saved_result);
}

// ============================================================================
// Function Call Continuations (Phase 4.3)
// ============================================================================

// FunctionCallK implementation - apply function to arguments
void FunctionCallK::invoke(Machine* machine) {
    // fn_value should be a CLOSURE
    if (!fn_value || fn_value->tag != ValueType::CLOSURE) {
        machine->throw_error("VALUE ERROR: Attempted to call non-function value", this);
        return;
    }

    // Get the function body continuation graph
    Continuation* body = fn_value->data.closure->body;
    if (!body) {
        machine->throw_error("VALUE ERROR: Function has no body", this);
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
    // Bind ∇ for recursive self-reference
    call_env->define("∇", fn_value);

    // Save current environment
    Environment* saved_env = machine->env;

    // Switch to function environment
    machine->env = call_env;

    // Push restore environment continuation (executes after function returns)
    RestoreEnvK* restore_k = machine->heap->allocate<RestoreEnvK>(saved_env);
    machine->push_kont(restore_k);

    // Push CatchReturnK to establish function boundary for →0 and :Return
    CatchReturnK* catch_k = machine->heap->allocate<CatchReturnK>("dfn");
    machine->push_kont(catch_k);

    // Execute function body
    machine->push_kont(body);
}

void FunctionCallK::mark(Heap* heap) {
    heap->mark(fn_value);
    heap->mark(left_arg);
    heap->mark(right_arg);
}

// RestoreEnvK implementation - restore environment after function call
void RestoreEnvK::invoke(Machine* machine) {
    // Restore the saved environment
    machine->env = saved_env;

    // Result value is already in machine->result
    // Phase 3.1: No return needed
}

void RestoreEnvK::mark(Heap* heap) {
    // saved_env will be marked by machine's environment chain
    (void)heap;
}

// ============================================================================
// G2 Grammar Continuations (Operator Support)
// ============================================================================

// DerivedOperatorK implementation - partially apply dyadic operator
void DerivedOperatorK::invoke(Machine* machine) {
    // Push continuation to apply operator after operand is evaluated
    // Pass axis_cont if present (for f/[k] syntax)
    machine->push_kont(machine->heap->allocate<ApplyDerivedOperatorK>(op_name, axis_cont));
    machine->push_kont(operand_cont);
}

void DerivedOperatorK::mark(Heap* heap) {
    heap->mark(operand_cont);
    heap->mark(axis_cont);
}

// ApplyDerivedOperatorK implementation - create DERIVED_OPERATOR value
void ApplyDerivedOperatorK::invoke(Machine* machine) {
    Value* first_operand = machine->result;

    // Look up the operator by name from environment
    Value* op_val = machine->env->lookup(op_name);
    if (!op_val) {
        std::string msg = std::string("VALUE ERROR: Unknown operator: ") + op_name;
        machine->throw_error(msg.c_str(), this);
        return;
    }

    // Handle both primitive operators (OPERATOR) and defined operators (DEFINED_OPERATOR)
    if (op_val->tag == ValueType::DEFINED_OPERATOR) {
        // User-defined operator
        Value::DefinedOperatorData* def_op = op_val->data.defined_op_data;
        Value* derived = machine->heap->allocate_derived_operator(def_op, first_operand, op_val);

        // If axis is specified (f OP[k] syntax), evaluate it and create OPERATOR_CURRY
        if (axis_cont) {
            machine->push_kont(machine->heap->allocate<ApplyAxisK>(derived));
            machine->push_kont(axis_cont);
        } else {
            machine->result = derived;
        }
        return;
    }

    if (op_val->tag != ValueType::OPERATOR) {
        std::string msg = std::string("VALUE ERROR: Not an operator: ") + op_name;
        machine->throw_error(msg.c_str(), this);
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

        // If axis is specified (f/[k] syntax), evaluate it and create OPERATOR_CURRY
        if (axis_cont) {
            machine->push_kont(machine->heap->allocate<ApplyAxisK>(derived));
            machine->push_kont(axis_cont);
        } else {
            machine->result = derived;
        }
    } else {
        machine->throw_error("SYNTAX ERROR: Operator has neither monadic nor dyadic form", this);
    }
}

void ApplyDerivedOperatorK::mark(Heap* heap) {
    heap->mark(axis_cont);
}

// ApplyAxisK implementation - apply axis to derived operator
void ApplyAxisK::invoke(Machine* machine) {
    Value* axis = machine->result;

    // Create OPERATOR_CURRY: derived_op curried with axis as second operand
    // When this is applied to omega, DispatchFunctionK will call op->dyadic
    Value* curried = machine->heap->allocate_curried_fn(
        derived_op, axis, Value::CurryType::OPERATOR_CURRY);
    machine->result = curried;
}

void ApplyAxisK::mark(Heap* heap) {
    heap->mark(derived_op);
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
                // Empty array input - return empty with same shape
                if (orig_is_vector) {
                    machine->result = machine->heap->allocate_vector(Eigen::VectorXd(0), orig_is_char);
                } else {
                    machine->result = machine->heap->allocate_matrix(Eigen::MatrixXd(orig_rows, orig_cols), orig_is_char);
                }
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
                    machine->result = results[0];
                } else if (results.size() == (size_t)(orig_rows * orig_cols) && !orig_is_vector && orig_cols > 1) {
                    // Same number of results as input elements AND input was matrix - preserve matrix shape
                    // This handles rank-0 operations that preserve element count
                    Eigen::MatrixXd mat(orig_rows, orig_cols);
                    for (size_t i = 0; i < results.size(); i++) {
                        mat(i / orig_cols, i % orig_cols) = results[i]->as_scalar();
                    }
                    machine->result = machine->heap->allocate_matrix(mat, orig_is_char);
                } else {
                    // Otherwise (including reduction), return vector of results
                    Eigen::VectorXd vec(results.size());
                    for (size_t i = 0; i < results.size(); i++) {
                        vec(i) = results[i]->as_scalar();
                    }
                    machine->result = machine->heap->allocate_vector(vec, orig_is_char);
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
                    machine->result = machine->heap->allocate_matrix(mat, orig_is_char);
                } else {
                    // Mixed results - return last (TODO: nested arrays)
                    machine->result = results.back();
                }
            }
            return;
        }

        // Extract cells
        Value* left_cell = extract_cell(machine, lhs, left_rank,
            (lhs && count_cells_for_rank(lhs, left_rank) == 1) ? 0 : current_cell);
        Value* right_cell = extract_cell(machine, rhs, right_rank, current_cell);

        if (!right_cell) {
            machine->throw_error("INDEX ERROR: cell extraction failed", this);
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
            machine->result = accumulator;
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
                machine->result = machine->heap->allocate_vector(vec, orig_is_char);
            } else {
                // For matrix scan, each row is scanned independently
                // This simplified version assumes vector input
                Eigen::VectorXd vec(results.size());
                for (size_t i = 0; i < results.size(); i++) {
                    vec(i) = results[i]->as_scalar();
                }
                machine->result = machine->heap->allocate_vector(vec, orig_is_char);
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

    } else if (mode == CellIterMode::OUTER) {
        // Cartesian product iteration for outer product
        if (current_cell >= total_cells) {
            // Done - assemble results into matrix
            // Outer product ALWAYS returns a matrix (shape: lhs_shape × rhs_shape)
            // Even with scalars: scalar∘.f vector → 1×N matrix, vector∘.f scalar → N×1 matrix
            if (lhs_total == 0 || rhs_total == 0) {
                // Empty result - return empty matrix with correct shape
                machine->result = machine->heap->allocate_matrix(Eigen::MatrixXd(lhs_total, rhs_total));
                return;
            }
            if (lhs_total == 1 && rhs_total == 1) {
                // Scalar result (both sides scalar)
                machine->result = results[0];
            } else {
                // Matrix result (including N×1 and 1×N cases)
                Eigen::MatrixXd mat(lhs_total, rhs_total);
                for (int i = 0; i < lhs_total; i++) {
                    for (int j = 0; j < rhs_total; j++) {
                        mat(i, j) = results[i * rhs_total + j]->as_scalar();
                    }
                }
                machine->result = machine->heap->allocate_matrix(mat);
            }
            return;
        }

        // Compute (i, j) from linear index
        int i = current_cell / rhs_total;
        int j = current_cell % rhs_total;

        // Extract lhs[i] and rhs[j]
        Value* left_cell;
        Value* right_cell;

        if (lhs->is_scalar()) {
            left_cell = lhs;
        } else {
            const Eigen::MatrixXd* lhs_mat = lhs->as_matrix();
            int li = i / lhs_cols;
            int lj = i % lhs_cols;
            if (lhs->is_vector()) {
                left_cell = machine->heap->allocate_scalar((*lhs_mat)(i, 0));
            } else {
                left_cell = machine->heap->allocate_scalar((*lhs_mat)(li, lj));
            }
        }

        if (rhs->is_scalar()) {
            right_cell = rhs;
        } else {
            const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();
            int ri = j / rhs_cols;
            int rj = j % rhs_cols;
            if (rhs->is_vector()) {
                right_cell = machine->heap->allocate_scalar((*rhs_mat)(j, 0));
            } else {
                right_cell = machine->heap->allocate_scalar((*rhs_mat)(ri, rj));
            }
        }

        // Push collector and dispatch function dyadically
        machine->push_kont(machine->heap->allocate<CellCollectK>(this));
        machine->push_kont(machine->heap->allocate<DispatchFunctionK>(fn, left_cell, right_cell));
    }
}

void CellIterK::mark(Heap* heap) {
    heap->mark(fn);
    heap->mark(lhs);
    heap->mark(rhs);
    heap->mark(accumulator);
    for (Value* v : results) {
        heap->mark(v);
    }
}

void CellCollectK::invoke(Machine* machine) {
    Value* result = machine->result;

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
    } else if (iter->mode == CellIterMode::OUTER) {
        iter->results.push_back(result);
        iter->current_cell++;
    }

    // Continue iteration
    machine->push_kont(iter);
}

void CellCollectK::mark(Heap* heap) {
    // iter holds Values (fn, lhs, rhs, results, accumulator) that must be marked
    heap->mark(iter);
}

// ============================================================================
// RowReduceK - Implementation
// ============================================================================

void RowReduceK::invoke(Machine* machine) {
    if (current_row >= total_rows) {
        // Done - assemble results into vector
        Eigen::VectorXd vec(results.size());
        for (size_t i = 0; i < results.size(); i++) {
            vec(i) = results[i]->as_scalar();
        }
        machine->result = machine->heap->allocate_vector(vec);
        return;
    }

    // Extract current row/column as a vector and reduce it
    const Eigen::MatrixXd* mat = matrix->as_matrix();

    if (!reduce_first_axis) {
        // Regular reduce (/): reduce each row
        Eigen::VectorXd row = mat->row(current_row).transpose();
        Value* row_vec = machine->heap->allocate_vector(row);

        // Push collector, then CellIterK FOLD_RIGHT for this row
        machine->push_kont(machine->heap->allocate<RowReduceCollectK>(this));
        int row_len = row.rows();
        machine->push_kont(machine->heap->allocate<CellIterK>(
            fn, nullptr, row_vec, 0, 0, row_len,
            CellIterMode::FOLD_RIGHT, row_len, 1, true));
    } else {
        // Reduce-first (⌿): reduce each column
        Eigen::VectorXd col = mat->col(current_row);
        Value* col_vec = machine->heap->allocate_vector(col);

        // Push collector, then CellIterK FOLD_RIGHT for this column
        machine->push_kont(machine->heap->allocate<RowReduceCollectK>(this));
        int col_len = col.rows();
        machine->push_kont(machine->heap->allocate<CellIterK>(
            fn, nullptr, col_vec, 0, 0, col_len,
            CellIterMode::FOLD_RIGHT, col_len, 1, true));
    }
}

void RowReduceK::mark(Heap* heap) {
    heap->mark(fn);
    heap->mark(matrix);
    for (Value* v : results) {
        heap->mark(v);
    }
}

void RowReduceCollectK::invoke(Machine* machine) {
    Value* result = machine->result;
    iter->results.push_back(result);
    iter->current_row++;
    machine->push_kont(iter);
}

void RowReduceCollectK::mark(Heap* heap) {
    // iter holds Values (fn, matrix, results) that must be marked
    heap->mark(iter);
}

// ============================================================================
// PrefixScanK - Implementation
// ============================================================================

void PrefixScanK::invoke(Machine* machine) {
    if (current_prefix > total_len) {
        // Done - assemble results into vector
        Eigen::VectorXd result_vec(results.size());
        for (size_t i = 0; i < results.size(); i++) {
            result_vec(i) = results[i]->as_scalar();
        }
        machine->result = machine->heap->allocate_vector(result_vec);
        return;
    }

    const Eigen::MatrixXd* mat = vec->as_matrix();

    if (current_prefix == 1) {
        // First element is just itself
        Value* first = machine->heap->allocate_scalar((*mat)(0, 0));
        results.push_back(first);
        current_prefix++;
        machine->push_kont(this);
        return;
    }

    // Create a prefix vector of length current_prefix
    Eigen::VectorXd prefix(current_prefix);
    for (int i = 0; i < current_prefix; i++) {
        prefix(i) = (*mat)(i, 0);
    }
    Value* prefix_vec = machine->heap->allocate_vector(prefix);

    // Push collector, then CellIterK FOLD_RIGHT to reduce this prefix
    machine->push_kont(machine->heap->allocate<PrefixScanCollectK>(this));
    machine->push_kont(machine->heap->allocate<CellIterK>(
        fn, nullptr, prefix_vec, 0, 0, current_prefix,
        CellIterMode::FOLD_RIGHT, current_prefix, 1, true));
}

void PrefixScanK::mark(Heap* heap) {
    heap->mark(fn);
    heap->mark(vec);
    for (Value* v : results) {
        heap->mark(v);
    }
}

void PrefixScanCollectK::invoke(Machine* machine) {
    Value* result = machine->result;
    iter->results.push_back(result);
    iter->current_prefix++;
    machine->push_kont(iter);
}

void PrefixScanCollectK::mark(Heap* heap) {
    // iter holds Values (fn, vec, results) that must be marked
    heap->mark(iter);
}

// ============================================================================
// RowScanK - Implementation
// ============================================================================

void RowScanK::invoke(Machine* machine) {
    if (current_row >= total_rows) {
        // Done - assemble results into matrix
        if (results.empty()) {
            machine->result = machine->heap->allocate_scalar(0);
            return;
        }

        if (!scan_first_axis) {
            // Regular scan: results are row vectors, assemble into matrix
            int result_cols = results[0]->rows();  // Each result is a vector
            Eigen::MatrixXd mat(total_rows, result_cols);
            for (int r = 0; r < total_rows; r++) {
                const Eigen::MatrixXd* row_vec = results[r]->as_matrix();
                mat.row(r) = row_vec->col(0).transpose();
            }
            machine->result = machine->heap->allocate_matrix(mat);
        } else {
            // Scan-first: results are column vectors, assemble into matrix
            int result_rows = results[0]->rows();  // Each result is a vector
            Eigen::MatrixXd mat(result_rows, total_rows);
            for (int c = 0; c < total_rows; c++) {
                const Eigen::MatrixXd* col_vec = results[c]->as_matrix();
                mat.col(c) = col_vec->col(0);
            }
            machine->result = machine->heap->allocate_matrix(mat);
        }
        return;
    }

    // Extract current row/column as a vector and scan it
    const Eigen::MatrixXd* mat = matrix->as_matrix();

    if (!scan_first_axis) {
        // Regular scan (\): scan each row
        Eigen::VectorXd row = mat->row(current_row).transpose();
        Value* row_vec = machine->heap->allocate_vector(row);

        // Push collector, then PrefixScanK for this row
        machine->push_kont(machine->heap->allocate<RowScanCollectK>(this));
        int row_len = row.rows();
        if (row_len <= 1) {
            // Single element or empty: just return as-is
            machine->result = row_vec;
            machine->push_kont(machine->heap->allocate<RowScanCollectK>(this));
            return;
        }
        machine->push_kont(machine->heap->allocate<PrefixScanK>(fn, row_vec, row_len));
    } else {
        // Scan-first (⍀): scan each column
        Eigen::VectorXd col = mat->col(current_row);
        Value* col_vec = machine->heap->allocate_vector(col);

        // Push collector, then PrefixScanK for this column
        machine->push_kont(machine->heap->allocate<RowScanCollectK>(this));
        int col_len = col.rows();
        if (col_len <= 1) {
            // Single element or empty: just return as-is
            machine->result = col_vec;
            machine->push_kont(machine->heap->allocate<RowScanCollectK>(this));
            return;
        }
        machine->push_kont(machine->heap->allocate<PrefixScanK>(fn, col_vec, col_len));
    }
}

void RowScanK::mark(Heap* heap) {
    heap->mark(fn);
    heap->mark(matrix);
    for (Value* v : results) {
        heap->mark(v);
    }
}

void RowScanCollectK::invoke(Machine* machine) {
    Value* result = machine->result;
    iter->results.push_back(result);
    iter->current_row++;
    machine->push_kont(iter);
}

void RowScanCollectK::mark(Heap* heap) {
    // iter holds Values (fn, matrix, results) that must be marked
    heap->mark(iter);
}

// ============================================================================
// ReduceResultK - Implementation
// ============================================================================
// Takes the vector in result and reduces it with fn

void ReduceResultK::invoke(Machine* machine) {
    Value* vec = machine->result;

    // Handle scalar - just return it
    if (vec->is_scalar()) {
        // Already a scalar, nothing to reduce
        return;
    }

    // Handle empty vector - would need identity element, for now error
    int len = vec->rows();
    if (len == 0) {
        machine->throw_error("DOMAIN ERROR: cannot reduce empty vector", this);
        return;
    }

    // Single element - return as-is
    if (len == 1) {
        const Eigen::MatrixXd* mat = vec->as_matrix();
        machine->result = machine->heap->allocate_scalar((*mat)(0, 0));
        return;
    }

    // Multiple elements - use CellIterK FOLD_RIGHT
    machine->push_kont(machine->heap->allocate<CellIterK>(
        fn, nullptr, vec, 0, 0, len,
        CellIterMode::FOLD_RIGHT, len, 1, true));
}

void ReduceResultK::mark(Heap* heap) {
    heap->mark(fn);
}

// ============================================================================
// InnerProductIterK - Implementation
// ============================================================================
// Iterates over output cells for matrix inner product

void InnerProductIterK::invoke(Machine* machine) {
    int total_cells = lhs_rows * rhs_cols;

    if (current_i * rhs_cols + current_j >= total_cells) {
        // Done - assemble results
        if (lhs_rows == 1 && rhs_cols == 1) {
            // Scalar result
            machine->result = results[0];
        } else if (rhs_cols == 1) {
            // Column vector result (matrix × vector)
            Eigen::VectorXd vec(lhs_rows);
            for (int i = 0; i < lhs_rows; i++) {
                vec(i) = results[i]->as_scalar();
            }
            machine->result = machine->heap->allocate_vector(vec);
        } else if (lhs_rows == 1) {
            // Row vector result (vector × matrix)
            Eigen::VectorXd vec(rhs_cols);
            for (int j = 0; j < rhs_cols; j++) {
                vec(j) = results[j]->as_scalar();
            }
            machine->result = machine->heap->allocate_vector(vec);
        } else {
            // Matrix result
            Eigen::MatrixXd mat(lhs_rows, rhs_cols);
            for (int i = 0; i < lhs_rows; i++) {
                for (int j = 0; j < rhs_cols; j++) {
                    mat(i, j) = results[i * rhs_cols + j]->as_scalar();
                }
            }
            machine->result = machine->heap->allocate_matrix(mat);
        }
        return;
    }

    // Extract row current_i from lhs and column current_j from rhs
    // Handle vectors specially: vectors are stored as column vectors (N×1)
    const Eigen::MatrixXd* lhs_mat = lhs->as_matrix();
    const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();

    Eigen::VectorXd row_vec;
    Eigen::VectorXd col_vec;

    if (lhs->is_vector()) {
        // Vector × matrix: use whole vector as the "row"
        row_vec = lhs_mat->col(0);
    } else {
        row_vec = lhs_mat->row(current_i).transpose();
    }

    if (rhs->is_vector()) {
        // Matrix × vector: use whole vector as the "column"
        col_vec = rhs_mat->col(0);
    } else {
        col_vec = rhs_mat->col(current_j);
    }

    Value* row = machine->heap->allocate_vector(row_vec);
    Value* col = machine->heap->allocate_vector(col_vec);

    // For vector inner product: first apply g element-wise, then reduce with f
    // Push: collector -> ReduceResultK(f) -> dyadic CellIterK COLLECT(g)
    machine->push_kont(machine->heap->allocate<InnerProductCollectK>(this));
    machine->push_kont(machine->heap->allocate<ReduceResultK>(f_fn));
    machine->push_kont(machine->heap->allocate<CellIterK>(
        g_fn, row, col, 0, 0, lhs_cols,
        CellIterMode::COLLECT, lhs_cols, 1, true));
}

void InnerProductIterK::mark(Heap* heap) {
    heap->mark(f_fn);
    heap->mark(g_fn);
    heap->mark(lhs);
    heap->mark(rhs);
    for (Value* v : results) {
        heap->mark(v);
    }
}

void InnerProductCollectK::invoke(Machine* machine) {
    Value* result = machine->result;
    iter->results.push_back(result);

    // Advance to next cell
    iter->current_j++;
    if (iter->current_j >= iter->rhs_cols) {
        iter->current_j = 0;
        iter->current_i++;
    }

    // Continue iteration
    machine->push_kont(iter);
}

void InnerProductCollectK::mark(Heap* heap) {
    // iter holds Values (f_fn, g_fn, lhs, rhs, results) that must be marked
    heap->mark(iter);
}

// ============================================================================
// NwiseReduceK - Implementation
// ============================================================================

void NwiseReduceK::invoke(Machine* machine) {
    if (current_window >= total_windows) {
        // Done - assemble results into vector
        Eigen::VectorXd result_vec(results.size());
        if (reverse) {
            // Negative N: reverse the order of results
            for (size_t i = 0; i < results.size(); i++) {
                result_vec(i) = results[results.size() - 1 - i]->as_scalar();
            }
        } else {
            for (size_t i = 0; i < results.size(); i++) {
                result_vec(i) = results[i]->as_scalar();
            }
        }
        machine->result = machine->heap->allocate_vector(result_vec);
        return;
    }

    const Eigen::MatrixXd* mat = vec->as_matrix();

    // Extract window of size window_size starting at current_window
    Eigen::VectorXd window(window_size);
    for (int i = 0; i < window_size; i++) {
        window(i) = (*mat)(current_window + i, 0);
    }
    Value* window_vec = machine->heap->allocate_vector(window);

    // Push collector, then CellIterK FOLD_RIGHT to reduce this window
    machine->push_kont(machine->heap->allocate<NwiseCollectK>(this));
    machine->push_kont(machine->heap->allocate<CellIterK>(
        fn, nullptr, window_vec, 0, 0, window_size,
        CellIterMode::FOLD_RIGHT, window_size, 1, true));
}

void NwiseReduceK::mark(Heap* heap) {
    heap->mark(fn);
    heap->mark(vec);
    for (Value* v : results) {
        heap->mark(v);
    }
}

void NwiseCollectK::invoke(Machine* machine) {
    Value* result = machine->result;
    iter->results.push_back(result);
    iter->current_window++;
    machine->push_kont(iter);
}

void NwiseCollectK::mark(Heap* heap) {
    // iter holds Values (fn, vec, results) that must be marked
    heap->mark(iter);
}

// ============================================================================
// NwiseMatrixReduceK - Implementation
// ============================================================================

void NwiseMatrixReduceK::invoke(Machine* machine) {
    const Eigen::MatrixXd* mat = matrix->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (current_slice >= total_slices) {
        // Done - assemble results into matrix
        // Each result is a vector from N-wise reduction
        // Result shape depends on axis:
        //   axis 1 (first_axis): (rows - N + 1) x cols
        //   axis 2 (!first_axis): rows x (cols - N + 1)
        int axis_len = first_axis ? rows : cols;
        int result_axis_len = axis_len - window_size + 1;

        if (first_axis) {
            // Results are column vectors, stack horizontally
            Eigen::MatrixXd result_mat(result_axis_len, cols);
            for (int j = 0; j < cols; j++) {
                const Eigen::MatrixXd* col_result = results[j]->as_matrix();
                for (int i = 0; i < result_axis_len; i++) {
                    result_mat(i, j) = (*col_result)(i, 0);
                }
            }
            machine->result = machine->heap->allocate_matrix(result_mat);
        } else {
            // Results are row vectors, stack vertically
            Eigen::MatrixXd result_mat(rows, result_axis_len);
            for (int i = 0; i < rows; i++) {
                const Eigen::MatrixXd* row_result = results[i]->as_matrix();
                for (int j = 0; j < result_axis_len; j++) {
                    result_mat(i, j) = (*row_result)(j, 0);
                }
            }
            machine->result = machine->heap->allocate_matrix(result_mat);
        }
        return;
    }

    // Extract current slice (row or column) and apply N-wise reduction
    Eigen::VectorXd slice;
    if (first_axis) {
        // Axis 1: extract column, apply N-wise along rows
        slice = mat->col(current_slice);
    } else {
        // Axis 2: extract row, apply N-wise along columns
        slice = mat->row(current_slice).transpose();
    }
    Value* slice_vec = machine->heap->allocate_vector(slice);

    // Push collector, then NwiseReduceK for this slice
    machine->push_kont(machine->heap->allocate<NwiseMatrixCollectK>(this));
    machine->push_kont(machine->heap->allocate<NwiseReduceK>(fn, slice_vec, window_size, reverse));
}

void NwiseMatrixReduceK::mark(Heap* heap) {
    heap->mark(fn);
    heap->mark(matrix);
    for (Value* v : results) {
        heap->mark(v);
    }
}

void NwiseMatrixCollectK::invoke(Machine* machine) {
    Value* result = machine->result;
    iter->results.push_back(result);
    iter->current_slice++;
    machine->push_kont(iter);
}

void NwiseMatrixCollectK::mark(Heap* heap) {
    // iter holds Values (fn, matrix, results) that must be marked
    heap->mark(iter);
}

// ============================================================================
// Indexed Assignment Continuations
// ============================================================================

void IndexedAssignK::invoke(Machine* machine) {
    // Evaluate value first (APL right-to-left), then index
    machine->push_kont(machine->heap->allocate<IndexedAssignIndexK>(var_name, nullptr, index_cont));
    machine->push_kont(value_cont);
}

void IndexedAssignK::mark(Heap* heap) {
    heap->mark(index_cont);
    heap->mark(value_cont);
}

void IndexedAssignIndexK::invoke(Machine* machine) {
    // Value just evaluated, save it and evaluate index
    value_val = machine->result;
    machine->push_kont(machine->heap->allocate<PerformIndexedAssignK>(var_name, value_val, nullptr));
    machine->push_kont(index_cont);
}

void IndexedAssignIndexK::mark(Heap* heap) {
    heap->mark(value_val);
    heap->mark(index_cont);
}

void PerformIndexedAssignK::invoke(Machine* machine) {
    // Index just evaluated
    index_val = machine->result;

    // g' finalization: If index is a curry, finalize it first
    if (index_val && index_val->tag == ValueType::CURRIED_FN) {
        Value::CurriedFnData* cd = index_val->data.curried_fn;
        if (cd->curry_type == Value::CurryType::G_PRIME ||
            cd->curry_type == Value::CurryType::DYADIC_CURRY) {
            machine->push_kont(this);
            machine->push_kont(machine->heap->allocate<PerformFinalizeK>());
            return;
        }
    }

    // Lookup the array variable
    Value* arr = machine->env->lookup(var_name);
    if (!arr) {
        machine->throw_error("VALUE ERROR: undefined variable in indexed assignment", this);
        return;
    }

    // Convert string to char vector if needed (strings become numeric on modification)
    if (arr->is_string()) {
        arr = arr->to_char_vector(machine->heap);
        machine->env->define(var_name, arr);  // Update binding to converted array
    }

    // Numeric array indexed assignment
    if (!arr->is_array()) {
        machine->throw_error("INDEX ERROR: cannot index non-array value", this);
        return;
    }

    const Eigen::MatrixXd* mat = arr->as_matrix();
    int size = static_cast<int>(mat->size());

    // Create modified copy
    Eigen::MatrixXd new_mat = *mat;

    if (index_val->is_scalar()) {
        // Single index assignment
        int idx = static_cast<int>(index_val->as_scalar()) - machine->io;  // ⎕IO

        if (idx < 0 || idx >= size) {
            machine->throw_error("INDEX ERROR: index out of bounds", this);
            return;
        }

        if (!value_val->is_scalar()) {
            machine->throw_error("LENGTH ERROR: scalar index requires scalar value", this);
            return;
        }
        double new_val = value_val->as_scalar();

        // Use row-major linear indexing
        int row = idx / new_mat.cols();
        int col = idx % new_mat.cols();
        new_mat(row, col) = new_val;
    } else if (index_val->is_vector()) {
        // Vector index assignment: A[2 4]←99 88 or A[2 4]←0 (scalar extension)
        const Eigen::MatrixXd* idx_mat = index_val->as_matrix();
        int num_indices = static_cast<int>(idx_mat->size());

        // Empty index is a no-op (ISO 13751)
        if (num_indices == 0) {
            machine->result = value_val;
            return;
        }

        // Check value compatibility
        bool scalar_value = value_val->is_scalar();
        const Eigen::MatrixXd* val_mat = nullptr;
        if (!scalar_value) {
            if (!value_val->is_vector()) {
                machine->throw_error("RANK ERROR: value must be scalar or vector for vector index", this);
                return;
            }
            val_mat = value_val->as_matrix();
            if (static_cast<int>(val_mat->size()) != num_indices) {
                machine->throw_error("LENGTH ERROR: index and value lengths must match", this);
                return;
            }
        }

        // Assign each index
        for (int i = 0; i < num_indices; i++) {
            int idx = static_cast<int>((*idx_mat)(i, 0)) - machine->io;  // ⎕IO
            if (idx < 0 || idx >= size) {
                machine->throw_error("INDEX ERROR: index out of bounds", this);
                return;
            }

            double new_val = scalar_value ? value_val->as_scalar() : (*val_mat)(i, 0);

            // Use row-major linear indexing
            int row = idx / new_mat.cols();
            int col = idx % new_mat.cols();
            new_mat(row, col) = new_val;
        }
    } else {
        machine->throw_error("RANK ERROR: index must be scalar or vector", this);
        return;
    }

    Value* result;
    if (arr->is_vector()) {
        result = machine->heap->allocate_vector(new_mat.col(0));
    } else {
        result = machine->heap->allocate_matrix(new_mat);
    }
    machine->env->define(var_name, result);
    machine->result = value_val;
}

void PerformIndexedAssignK::mark(Heap* heap) {
    heap->mark(value_val);
    heap->mark(index_val);
}

} // namespace apl
