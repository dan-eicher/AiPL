// Continuation implementations

#include "continuation.h"
#include "machine.h"
#include "completion.h"
#include "operators.h"
#include <algorithm>
#include <stdexcept>
#include <typeinfo>
#include <chrono>
#include <ctime>

namespace apl {

// Forward declaration of Heap for now
// Will be implemented in Phase 1.6
class Heap;

// Forward declarations for helper functions used by apply_function_immediate
static int count_cells_for_rank(Value* arr, int k);

// ============================================================================
// Finalization Helpers
// ============================================================================
// Consolidates the logic for checking and performing finalization of curried
// values (G_PRIME, DYADIC_CURRY) used throughout DispatchFunctionK.

// Check if a value needs finalization
static bool needs_finalization(Value* val, bool finalize_gprime = true) {
    if (!val || val->tag != ValueType::CURRIED_FN) {
        return false;
    }

    Value::CurriedFnData* cd = val->data.curried_fn;

    // G_PRIME: only finalize if finalize_gprime is true
    // Exception: CLOSURE curries should always be finalized because they're
    // immediate expressions, not partial applications meant to be preserved
    if (cd->curry_type == Value::CurryType::G_PRIME) {
        bool should_finalize = finalize_gprime || cd->fn->is_closure();
        return should_finalize;
    }

    // DYADIC_CURRY: always needs finalization
    if (cd->curry_type == Value::CurryType::DYADIC_CURRY) {
        return true;
    }

    // OPERATOR_CURRY: does not need finalization (it's waiting for operand, not array)
    return false;
}

// Try to finalize a G_PRIME curry synchronously (without pushing continuations).
// Only works for primitive functions with monadic forms.
// Returns: val unchanged if not a curry, machine->result if finalized, nullptr if async needed.
static Value* try_finalize_sync(Machine* m, Value* val, bool finalize_gprime = true) {
    if (!val || val->tag != ValueType::CURRIED_FN) {
        return val;  // Not a curry, nothing to finalize
    }

    Value::CurriedFnData* cd = val->data.curried_fn;

    // Only handle G_PRIME curries
    if (cd->curry_type != Value::CurryType::G_PRIME) {
        return val;  // Not G_PRIME, can't sync finalize here
    }

    // Check finalize_gprime flag (unless it's a closure which always finalizes)
    bool should_finalize = finalize_gprime || cd->fn->is_closure();
    if (!should_finalize) {
        return val;  // Shouldn't finalize, return unchanged
    }

    // Only sync finalize if it's a primitive with monadic form
    if (!cd->fn->is_primitive()) {
        return nullptr;  // Closure - needs async finalization
    }

    PrimitiveFn* prim_fn = cd->fn->data.primitive_fn;
    if (!prim_fn->monadic) {
        return nullptr;  // No monadic form - can't finalize
    }

    // Call the primitive's monadic form directly
    prim_fn->monadic(m, cd->axis, cd->first_arg);
    return m->result;
}

// Push PerformFinalizeK then the caller's continuation
// After finalization completes, then_kont will be invoked with machine->result
static void push_finalize_then(Machine* m, Continuation* then_kont, bool finalize_gprime = true) {
    m->push_kont(then_kont);
    m->push_kont(m->heap->allocate<PerformFinalizeK>(finalize_gprime));
}

// Combined helper: check if finalization is needed, and if so, push it
// Returns true if finalization was pushed (caller should return immediately)
// Returns false if no finalization needed (caller should continue)
static bool maybe_push_finalize(Machine* m, Value* val, Continuation* then_kont,
                                bool finalize_gprime = true) {
    if (!needs_finalization(val, finalize_gprime)) {
        return false;
    }
    push_finalize_then(m, then_kont, finalize_gprime);
    return true;
}

// ============================================================================
// Function Application Helpers
// ============================================================================

// Apply a function immediately without creating curries.
// For primitives: calls the monadic or dyadic form directly.
// For closures: pushes FunctionCallK.
// For derived operators: invokes the operator's form.
// Returns true if result is ready in machine->result, false if continuation pushed.
bool apply_function_immediate(Machine* m, Value* fn_val, Value* left_val,
                              Value* right_val, Value* axis) {
    // Handle CLOSURE values (dfns)
    if (fn_val->tag == ValueType::CLOSURE) {
        FunctionCallK* call_k = m->heap->allocate<FunctionCallK>(fn_val, left_val, right_val);
        m->push_kont(call_k);
        return false;  // Continuation pushed
    }

    // Handle DERIVED_OPERATOR
    if (fn_val->tag == ValueType::DERIVED_OPERATOR) {
        Value::DerivedOperatorData* derived = fn_val->data.derived_op;
        PrimitiveOp* op = derived->primitive_op;
        Value::DefinedOperatorData* def_op = derived->defined_op;
        Value* first_operand = derived->first_operand;
        Value* op_value = derived->operator_value;

        if (def_op) {
            // User-defined operator
            m->push_kont(m->heap->allocate<InvokeDefinedOperatorK>(
                def_op, op_value, first_operand, nullptr, left_val, right_val));
            return false;  // Continuation pushed
        } else if (op) {
            // Primitive operator
            if (left_val && right_val && op->dyadic) {
                op->dyadic(m, axis, left_val, first_operand, nullptr, right_val);
            } else if (right_val && op->monadic) {
                op->monadic(m, axis, first_operand, right_val);
            } else {
                m->throw_error("SYNTAX ERROR: operator form not available", nullptr, 1, 0);
                return false;
            }
            return true;  // Result set synchronously
        }
    }

    // Handle PRIMITIVE functions
    if (fn_val->tag == ValueType::PRIMITIVE) {
        PrimitiveFn* prim_fn = fn_val->data.primitive_fn;

        if (left_val && right_val) {
            // Dyadic application
            if (!prim_fn->dyadic) {
                m->throw_error("SYNTAX ERROR: Function has no dyadic form", nullptr, 1, 0);
                return false;
            }

            // Pervasive strand handling: if either arg is a strand, apply element-wise
            bool left_strand = left_val->is_strand();
            bool right_strand = right_val->is_strand();
            if (prim_fn->is_pervasive && (left_strand || right_strand)) {
                // Get sizes
                int left_size = left_strand ? static_cast<int>(left_val->as_strand()->size())
                                           : (left_val->is_scalar() ? 1 : left_val->size());
                int right_size = right_strand ? static_cast<int>(right_val->as_strand()->size())
                                             : (right_val->is_scalar() ? 1 : right_val->size());

                // Scalar extension for strands
                if (left_val->is_scalar() && right_strand) {
                    // Scalar op strand: extend scalar to match strand
                    std::vector<Value*> extended(right_size, left_val);
                    left_val = m->heap->allocate_strand(std::move(extended));
                    left_size = right_size;
                } else if (left_strand && right_val->is_scalar()) {
                    // Strand op scalar: extend scalar to match strand
                    std::vector<Value*> extended(left_size, right_val);
                    right_val = m->heap->allocate_strand(std::move(extended));
                    right_size = left_size;
                }

                // Check lengths match
                if (left_size != right_size) {
                    m->throw_error("LENGTH ERROR: mismatched shapes in pervasive operation", nullptr, 5, 0);
                    return false;
                }

                // Use CellIterK with COLLECT mode to apply element-wise
                m->push_kont(m->heap->allocate<CellIterK>(
                    fn_val, left_val, right_val, 0, 0, left_size,
                    CellIterMode::COLLECT, left_size, 1, true, false, true));
                return false;  // Continuation pushed
            }

            prim_fn->dyadic(m, axis, left_val, right_val);
        } else if (right_val) {
            // Monadic application
            if (!prim_fn->monadic) {
                m->throw_error("SYNTAX ERROR: Function has no monadic form", nullptr, 1, 0);
                return false;
            }

            // Pervasive handling for monadic on strand/NDARRAY
            bool right_strand = right_val->is_strand();
            bool right_ndarray = right_val->is_ndarray();
            if (prim_fn->is_pervasive && (right_strand || right_ndarray)) {
                int total = count_cells_for_rank(right_val, 0);

                if (right_ndarray) {
                    // NDARRAY: preserve shape
                    CellIterK* iter = m->heap->allocate<CellIterK>(
                        fn_val, nullptr, right_val, 0, 0, total,
                        CellIterMode::COLLECT, total, 1, false, false, false);
                    iter->orig_ndarray_shape = right_val->ndarray_shape();
                    m->push_kont(iter);
                } else {
                    // Strand: preserve strand structure
                    m->push_kont(m->heap->allocate<CellIterK>(
                        fn_val, nullptr, right_val, 0, 0, total,
                        CellIterMode::COLLECT, total, 1, true, false, true));
                }
                return false;  // Continuation pushed
            }

            prim_fn->monadic(m, axis, right_val);
        } else {
            m->throw_error("VALUE ERROR: function requires argument", nullptr, 2, 0);
            return false;
        }
        return true;  // Result set synchronously
    }

    m->throw_error("VALUE ERROR: expected function value", nullptr, 2, 0);
    return false;
}

// ============================================================================
// Cell Iterator Helpers
// ============================================================================
// Helper functions for CellIterK that are also used by DispatchFunctionK
// for pervasive strand dispatch.

// Helper: get the rank of a value (0=scalar, 1=vector/strand, 2=matrix, N=NDARRAY)
static int get_value_rank(Value* v) {
    if (v->is_scalar()) return 0;
    if (v->is_vector() || v->is_strand()) return 1;
    if (v->is_ndarray()) return static_cast<int>(v->ndarray_shape().size());
    return 2;  // Matrix
}

// Helper function for cell counting
static int count_cells_for_rank(Value* arr, int k) {
    if (!arr) return 0;
    int arr_rank = get_value_rank(arr);
    if (k >= arr_rank) return 1;

    if (arr->is_scalar()) return 1;

    // Handle strands - 0-cells are the elements
    if (arr->is_strand()) {
        return (k == 0) ? static_cast<int>(arr->as_strand()->size()) : 1;
    }

    // Handle NDARRAY - k-cells are subarrays of rank (arr_rank - k)
    // Number of k-cells = product of first k dimensions
    if (arr->is_ndarray()) {
        const std::vector<int>& shape = arr->ndarray_shape();
        int count = 1;
        for (int i = 0; i < std::min(k, static_cast<int>(shape.size())); ++i) {
            count *= shape[i];
        }
        // For k=0, we want total elements = product of all dimensions
        if (k == 0) {
            count = 1;
            for (int d : shape) count *= d;
        }
        return count;
    }

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

    // Handle strands - 0-cells are the elements (already Value*)
    if (arr->is_strand()) {
        if (k == 0) {
            std::vector<Value*>* elems = arr->as_strand();
            if (cell_index >= static_cast<int>(elems->size())) return nullptr;
            return (*elems)[cell_index];
        }
        return arr;  // k >= 1: return whole strand
    }

    // Handle NDARRAY
    if (arr->is_ndarray()) {
        const std::vector<int>& shape = arr->ndarray_shape();
        const Eigen::VectorXd* data = arr->ndarray_data();
        int ndrank = static_cast<int>(shape.size());

        if (k == 0) {
            // 0-cell: scalar at linear index (row-major order)
            int total = 1;
            for (int d : shape) total *= d;
            if (cell_index >= total) return nullptr;
            return m->heap->allocate_scalar((*data)(cell_index));
        }

        // k-cell: subarray formed by LAST k dimensions
        // Frame = first (rank - k) dimensions
        // Cell shape = {shape[rank-k], shape[rank-k+1], ..., shape[rank-1]}
        int frame_rank = ndrank - k;

        // Cell size = product of last k dimensions
        int cell_size = 1;
        for (int i = frame_rank; i < ndrank; ++i) {
            cell_size *= shape[i];
        }

        int start = cell_index * cell_size;
        int total = 1;
        for (int d : shape) total *= d;
        if (start + cell_size > total) return nullptr;

        // Extract the cell data
        Eigen::VectorXd cell_data = data->segment(start, cell_size);

        // Build result shape from last k dimensions
        std::vector<int> cell_shape(shape.begin() + frame_rank, shape.end());

        if (cell_shape.size() == 1) {
            // Result is a vector
            return m->heap->allocate_vector(cell_data);
        } else if (cell_shape.size() == 2) {
            // Result is a matrix
            Eigen::MatrixXd mat(cell_shape[0], cell_shape[1]);
            for (int i = 0; i < cell_shape[0]; ++i) {
                for (int j = 0; j < cell_shape[1]; ++j) {
                    mat(i, j) = cell_data(i * cell_shape[1] + j);
                }
            }
            return m->heap->allocate_matrix(mat);
        } else {
            // Result is still NDARRAY
            return m->heap->allocate_ndarray(cell_data, cell_shape);
        }
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
                // Found an error boundary - pop it
                machine->pop_kont();

                // If there's a handler (from ⎕EA), execute it
                if (catch_err->handler) {
                    // Push ClearErrorStateK FIRST (runs AFTER handler completes)
                    // ISO 13751: ⎕ET/⎕EM reflect current state, not historical
                    ClearErrorStateK* clear_k = machine->heap->allocate<ClearErrorStateK>();
                    machine->push_kont(clear_k);

                    // Push handler for execution
                    machine->push_kont(catch_err->handler);
                } else {
                    // No handler - just discard error and clear state
                    machine->event_type[0] = 0;
                    machine->event_type[1] = 0;
                    machine->event_message = nullptr;
                }

                machine->error_stack.clear();  // Discard trace, error was handled
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
    heap->mark(handler);
}

// ClearErrorStateK - Clears error state after handler completes successfully
void ClearErrorStateK::invoke(Machine* machine) {
    // Clear the error state (ISO 13751: ⎕ET/⎕EM reflect current state)
    machine->event_type[0] = 0;
    machine->event_type[1] = 0;
    machine->event_message = nullptr;
    // Result unchanged - handler's result passes through
}

void ClearErrorStateK::mark(Heap* heap) {
    (void)heap;  // Nothing to mark
}

// ThrowErrorK - Creates and propagates THROW completion (Phase 5.2)
// Note: error_stack is captured by Machine::throw_error() before ThrowErrorK is created
void ThrowErrorK::invoke(Machine* machine) {
    // Create a THROW completion with the error message
    Completion* throw_comp = machine->heap->allocate<Completion>(
        CompletionType::THROW,
        nullptr,  // No value for errors
        error_message->c_str()  // Error message in target field
    );

    // Push PropagateCompletionK to unwind the stack
    PropagateCompletionK* prop = machine->heap->allocate<PropagateCompletionK>(throw_comp);
    machine->push_kont(prop);
}

void ThrowErrorK::mark(Heap* heap) {
    heap->mark(error_message);
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
    op_data->left_arg_name = machine->string_pool.intern("⍺");   // dfn convention
    op_data->right_arg_name = machine->string_pool.intern("⍵");  // dfn convention
    op_data->result_name = nullptr; // dfn-style doesn't name result
    op_data->lexical_env = machine->env;  // Capture current environment

    Value* op_val = machine->heap->allocate_defined_operator(op_data);

    // Assign to operator name in environment
    machine->env->define(operator_name, op_val);
    machine->result = op_val;
}

void DefinedOperatorLiteralK::mark(Heap* heap) {
    heap->mark(body);
    heap->mark(operator_name);
    heap->mark(left_operand_name);
    heap->mark(right_operand_name);
}

// InvokeDefinedOperatorK implementation
// Invokes a user-defined operator with bound operands and arguments
void InvokeDefinedOperatorK::invoke(Machine* machine) {
    // Create new environment extending the operator's lexical environment
    Environment* env = machine->heap->allocate<Environment>(op->lexical_env);

    // Bind the left operand to named parameter (always present)
    env->define(op->left_operand_name, left_operand);
    // Also bind to ⍺⍺ for APL compatibility
    env->define(machine->string_pool.intern("⍺⍺"), left_operand);

    // Bind the right operand for dyadic operators
    if (op->is_dyadic_operator && right_operand && op->right_operand_name) {
        env->define(op->right_operand_name, right_operand);
        // Also bind to ⍵⍵ for APL compatibility
        env->define(machine->string_pool.intern("⍵⍵"), right_operand);
    }

    // Bind ∇∇ for recursive self-reference to the operator
    // (∇ is for functions, ∇∇ is for operators)
    if (operator_value) {
        env->define(machine->string_pool.intern("∇∇"), operator_value);
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
        std::string msg = std::string("VALUE ERROR: Undefined variable: ") + var_name->c_str();
        machine->throw_error(msg.c_str(), this, 2, 0);
        return;
    }

    machine->result = val;
    // Phase 3.1: No return needed, trampoline continues
}

void LookupK::mark(Heap* heap) {
    heap->mark(var_name);
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
    heap->mark(var_name);
    heap->mark(expr);
}

// PerformAssignK implementation
void PerformAssignK::invoke(Machine* machine) {
    // Expression has been evaluated - result is in result
    // Bind it to the variable name
    Value* val = machine->result;

    // Finalize curried functions before assignment (A←⍳5, A←+/1 2 3)
    if (maybe_push_finalize(machine, val, this)) {
        return;
    }
    // After finalization, use machine->result (may have been updated by sync finalization)
    val = machine->result;

    // Special handling for ⍺←value (alpha default): only assign if ⍺ not already defined
    // This implements APL's conditional default argument syntax
    if (*var_name == "⍺" && machine->env->lookup(var_name) != nullptr) {
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
    heap->mark(var_name);
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
        case SysVarId::ET: {
            // Event type: 2-element vector {class, subclass}
            Eigen::VectorXd et(2);
            et << machine->event_type[0], machine->event_type[1];
            machine->result = machine->heap->allocate_vector(et);
            break;
        }
        case SysVarId::EM:
            // Event message: character vector
            if (machine->event_message) {
                machine->result = machine->heap->allocate_string(machine->event_message);
            } else {
                machine->result = machine->heap->allocate_string("");
            }
            break;
        case SysVarId::TS: {
            // Time stamp: 7-element vector {year, month, day, hour, minute, second, millisecond}
            // ISO 13751 §11.4.1
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
            std::tm* tm = std::localtime(&now_time_t);

            Eigen::VectorXd ts(7);
            ts << static_cast<double>(tm->tm_year + 1900),  // Year
                  static_cast<double>(tm->tm_mon + 1),       // Month (1-12)
                  static_cast<double>(tm->tm_mday),          // Day (1-31)
                  static_cast<double>(tm->tm_hour),          // Hour (0-23)
                  static_cast<double>(tm->tm_min),           // Minute (0-59)
                  static_cast<double>(tm->tm_sec),           // Second (0-59)
                  static_cast<double>(now_ms.count());       // Millisecond (0-999)
            machine->result = machine->heap->allocate_vector(ts);
            break;
        }
        case SysVarId::AV: {
            // Atomic vector: 256-element character vector (codepoints 0-255)
            // ISO 13751 §11.4.2
            Eigen::VectorXd av(256);
            for (int i = 0; i < 256; ++i) {
                av(i) = static_cast<double>(i);
            }
            machine->result = machine->heap->allocate_vector(av, true);  // true = character data
            break;
        }
        case SysVarId::LC: {
            // Line counter: vector of line numbers in call stack
            // ISO 13751 §11.4.3 - innermost to outermost
            std::vector<double> lines;
            // Traverse continuation stack for line numbers
            for (auto it = machine->kont_stack.rbegin(); it != machine->kont_stack.rend(); ++it) {
                Continuation* k = *it;
                if (k && k->has_location() && k->line() > 0) {
                    lines.push_back(static_cast<double>(k->line()));
                }
            }
            // Add current control if it has location
            if (machine->control && machine->control->has_location() && machine->control->line() > 0) {
                lines.insert(lines.begin(), static_cast<double>(machine->control->line()));
            }

            if (lines.empty()) {
                // Empty vector if no line info
                Eigen::VectorXd empty(0);
                machine->result = machine->heap->allocate_vector(empty);
            } else {
                Eigen::VectorXd lc(lines.size());
                for (size_t i = 0; i < lines.size(); ++i) {
                    lc(i) = lines[i];
                }
                machine->result = machine->heap->allocate_vector(lc);
            }
            break;
        }
        case SysVarId::LX:
            // Latent expression: character vector (ISO 13751 §12.2.5)
            if (machine->lx && !machine->lx->empty()) {
                machine->result = machine->heap->allocate_string(machine->lx);
            } else {
                // Empty string
                machine->result = machine->heap->allocate_string("");
            }
            break;
        default:
            machine->throw_error("SYSTEM ERROR: unknown system variable", this, 0, 0);
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

    // Handle ⎕LX specially - it accepts character data, not scalars
    if (var_id == SysVarId::LX) {
        if (!val) {
            machine->throw_error("VALUE ERROR: no value for ⎕LX assignment", this, 2, 0);
            return;
        }
        if (!val->is_char_data() && !val->is_string()) {
            machine->throw_error("DOMAIN ERROR: ⎕LX must be character data", this, 11, 0);
            return;
        }
        // Convert to STRING if needed - as_string() returns interned pointer
        machine->lx = val->to_string_value(machine->heap)->as_string();
        machine->result = val;
        return;
    }

    // Other system variables require scalar values
    if (!val || !val->is_scalar()) {
        machine->throw_error("DOMAIN ERROR: system variable requires scalar value", this, 11, 0);
        return;
    }

    double dbl_val = val->as_scalar();

    switch (var_id) {
        case SysVarId::IO: {
            int int_val = static_cast<int>(dbl_val);
            if (dbl_val != int_val || (int_val != 0 && int_val != 1)) {
                machine->throw_error("DOMAIN ERROR: ⎕IO must be 0 or 1", this, 11, 0);
                return;
            }
            machine->io = int_val;
            break;
        }
        case SysVarId::PP: {
            int int_val = static_cast<int>(dbl_val);
            if (dbl_val != int_val || int_val < 1 || int_val > 17) {
                machine->throw_error("DOMAIN ERROR: ⎕PP must be 1-17", this, 11, 0);
                return;
            }
            machine->pp = int_val;
            break;
        }
        case SysVarId::CT:
            if (dbl_val < 0) {
                machine->throw_error("DOMAIN ERROR: ⎕CT must be nonnegative", this, 11, 0);
                return;
            }
            machine->ct = dbl_val;
            break;
        case SysVarId::RL: {
            // RL must be a positive integer
            if (dbl_val < 1 || dbl_val != static_cast<double>(static_cast<uint64_t>(dbl_val))) {
                machine->throw_error("DOMAIN ERROR: ⎕RL must be a positive integer", this, 11, 0);
                return;
            }
            machine->rl = static_cast<uint64_t>(dbl_val);
            machine->rng.seed(machine->rl);
            break;
        }
        default:
            machine->throw_error("SYSTEM ERROR: unknown system variable", this, 0, 0);
            return;
    }

    // Assignment returns the assigned value
    machine->result = val;
}

void PerformSysVarAssignK::mark(Heap* heap) {
    (void)heap;  // var_id is an enum, nothing to mark
}

// LiteralStrandK implementation - lexer-level strands
void LiteralStrandK::invoke(Machine* machine) {
    // Lexical strand: just return the pre-computed vector Value
    machine->result = vector_value;
}

void LiteralStrandK::mark(Heap* heap) {
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

    // ISO 13751 compliance: value-value juxtaposition is a SYNTAX ERROR
    // The Phrase Table (Table 3) has no A B pattern - adjacent values with no
    // function between them are not valid. Nested arrays must be created using
    // ⊂ (enclose), not by juxtaposing values like (1 2)(3 4).
    // Note: Lexer-level strands (e.g., 1 2 3) are handled earlier as LiteralStrandK.
    if (left_val->is_basic_value() && right_val->is_basic_value()) {
        machine->throw_error("SYNTAX ERROR: Adjacent values require a function between them", this, 1, 0);
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
            machine->throw_error("SYNTAX ERROR: Monadic operator cannot take value as operand", this, 1, 0);
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
            machine->throw_error("VALUE ERROR: Invalid OPERATOR_CURRY structure", this, 2, 0);
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
        // Exception: CLOSURE curries should always be finalized because they're
        // immediate expressions, not partial applications meant to be preserved
        bool should_finalize = finalize_gprime || cd->fn->is_closure();
        if (cd->curry_type == Value::CurryType::G_PRIME && should_finalize) {
            apply_function_immediate(machine, cd->fn, nullptr, cd->first_arg, cd->axis);
            return;
        }

        // DYADIC_CURRY: finalize based on inner function type
        if (cd->curry_type == Value::CurryType::DYADIC_CURRY) {
            Value* fn = cd->fn;
            Value* arg = cd->first_arg;

            // DYADIC_CURRY wrapping OPERATOR_CURRY - finalize with axis passed through
            if (fn->tag == ValueType::CURRIED_FN &&
                fn->data.curried_fn->curry_type == Value::CurryType::OPERATOR_CURRY) {
                Value::CurriedFnData* oc = fn->data.curried_fn;
                Value* derived = oc->fn;
                Value* second_operand = oc->first_arg;
                Value* axis = oc->axis;  // Just pass it through

                if (derived->tag == ValueType::DERIVED_OPERATOR) {
                    PrimitiveOp* op = derived->data.derived_op->primitive_op;
                    Value::DefinedOperatorData* def_op = derived->data.derived_op->defined_op;
                    Value* first_operand = derived->data.derived_op->first_operand;
                    Value* op_value = derived->data.derived_op->operator_value;

                    if (def_op) {
                        machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                            def_op, op_value, first_operand, second_operand, nullptr, arg));
                        return;
                    } else if (op && op->monadic) {
                        op->monadic(machine, axis, first_operand, arg);
                        return;
                    } else if (op && op->dyadic) {
                        op->dyadic(machine, axis, nullptr, first_operand, second_operand, arg);
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
                apply_function_immediate(machine, fn, nullptr, arg);
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
    heap->mark(op_name);
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
    heap->mark(op_name);
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
    heap->mark(op_name);
    heap->mark(left);
    heap->mark(right_val);
}


// ApplyMonadicK implementation
void ApplyMonadicK::invoke(Machine* machine) {
    // Operand has been evaluated - its value is in result
    Value* operand_val = machine->result;

    // g' finalization: If operand is a curry, finalize it first
    if (maybe_push_finalize(machine, operand_val, this)) {
        return;
    }
    // After finalization, use machine->result (may have been updated by sync finalization)
    operand_val = machine->result;

    // Look up the operator at evaluation time
    Value* op_val = machine->env->lookup(op_name);
    if (!op_val || op_val->tag != ValueType::PRIMITIVE) {
        std::string msg = std::string("VALUE ERROR: Unknown operator: ") + op_name->c_str();
        machine->throw_error(msg.c_str(), this, 2, 0);
        return;
    }

    PrimitiveFn* prim_fn = op_val->data.primitive_fn;

    if (!prim_fn->monadic) {
        std::string msg = std::string("SYNTAX ERROR: Operator has no monadic form: ") + op_name->c_str();
        machine->throw_error(msg.c_str(), this, 2, 0);
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
    heap->mark(op_name);
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
        std::string msg = std::string("VALUE ERROR: Unknown operator: ") + op_name->c_str();
        machine->throw_error(msg.c_str(), this, 2, 0);
        return;
    }

    PrimitiveFn* prim_fn = op_val->data.primitive_fn;

    if (!prim_fn->dyadic) {
        std::string msg = std::string("SYNTAX ERROR: Operator has no dyadic form: ") + op_name->c_str();
        machine->throw_error(msg.c_str(), this, 2, 0);
        return;
    }

    // Pervasive strand handling: if either arg is a strand, apply element-wise
    bool left_strand = left_val->is_strand();
    bool right_strand = right_val->is_strand();
    if (prim_fn->is_pervasive && (left_strand || right_strand)) {
        int left_size = left_strand ? static_cast<int>(left_val->as_strand()->size())
                                   : (left_val->is_scalar() ? 1 : left_val->size());
        int right_size = right_strand ? static_cast<int>(right_val->as_strand()->size())
                                     : (right_val->is_scalar() ? 1 : right_val->size());

        if (left_val->is_scalar() && right_strand) {
            std::vector<Value*> extended(right_size, left_val);
            left_val = machine->heap->allocate_strand(std::move(extended));
            left_size = right_size;
        } else if (left_strand && right_val->is_scalar()) {
            std::vector<Value*> extended(left_size, right_val);
            right_val = machine->heap->allocate_strand(std::move(extended));
            right_size = left_size;
        }

        if (left_size != right_size) {
            machine->throw_error("LENGTH ERROR: mismatched shapes in pervasive operation", this, 5, 0);
            return;
        }

        machine->push_kont(machine->heap->allocate<CellIterK>(
            op_val, left_val, right_val, 0, 0, left_size,
            CellIterMode::COLLECT, left_size, 1, true, false, true));
        return;
    }

    // Apply the dyadic function (no axis from this path)
    prim_fn->dyadic(machine, nullptr, left_val, right_val);

    // Phase 3.1: No return needed, trampoline continues
}

void ApplyDyadicK::mark(Heap* heap) {
    heap->mark(op_name);
    heap->mark(right_val);
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
        if (left_val == nullptr) {
            // Only right argument - create G_PRIME curry to defer monadic/dyadic decision
            // This enables proper dyadic calls like "3 F 5" where F is a named closure
            // Per Georgeff et al. "Parsing and Evaluation of APL with Operators":
            // g' = λx . λy . if null(y) then g1(x)
            //                else if bas(y) then g2(x,y)
            //                else y(g1(x))
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::G_PRIME);
            machine->result = curried;
            return;
        }
        // Both arguments (dyadic) - call immediately
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
                // null(y): finalize monadically
                apply_function_immediate(machine, inner_fn, nullptr, first_arg);
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
                apply_function_immediate(machine, inner_fn, nullptr, first_arg);
                return;
            }
        } else if (curried_data->curry_type == Value::CurryType::OPERATOR_CURRY) {
            // Operator curry: inner_fn is DERIVED_OPERATOR
            // - first_arg is second operand (function for ., value for ⍤)
            // - axis is axis spec from [k] syntax (or nullptr)
            if (inner_fn->tag != ValueType::DERIVED_OPERATOR) {
                machine->throw_error("VALUE ERROR: OPERATOR_CURRY expected DERIVED_OPERATOR", this, 2, 0);
                return;
            }
            Value::DerivedOperatorData* derived_data = inner_fn->data.derived_op;
            PrimitiveOp* op = derived_data->primitive_op;
            Value::DefinedOperatorData* def_op = derived_data->defined_op;
            Value* first_operand = derived_data->first_operand;
            Value* op_value = derived_data->operator_value;
            Value* second_operand = first_arg;
            Value* axis = curried_data->axis;  // Just pass it through

            if (left_val && right_val) {
                // Finalize any G_PRIME curried functions in arguments first
                Value* finalized_left = try_finalize_sync(machine, left_val, true);
                if (finalized_left != nullptr) {
                    left_val = finalized_left;
                }
                Value* finalized_right = try_finalize_sync(machine, right_val, true);
                if (finalized_right != nullptr) {
                    right_val = finalized_right;
                }

                if (def_op) {
                    machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                        def_op, op_value, first_operand, second_operand, left_val, right_val));
                } else {
                    op->dyadic(machine, axis, left_val, first_operand, second_operand, right_val);
                }
            } else if (right_val) {
                // Only have right array argument - curry to wait for potential left
                // This enables N-wise reduction with axis: "2 +/[1] matrix"

                // Finalize any G_PRIME curried function in right_val first
                Value* finalized = try_finalize_sync(machine, right_val, true);
                if (finalized != nullptr) {
                    right_val = finalized;
                }

                if (def_op) {
                    // User-defined dyadic operator with both operands, monadic application
                    machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                        def_op, op_value, first_operand, second_operand, nullptr, right_val));
                } else {
                    // Curry to wait for potential left array argument
                    // When finalized, applies with lhs=nullptr (monadic) or the left arg (dyadic)
                    Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::DYADIC_CURRY);
                    machine->result = curried;
                }
            } else {
                machine->throw_error("VALUE ERROR: operator curry expects array argument", this, 2, 0);
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
            if (first_arg == nullptr && curried_data->axis != nullptr && right_val != nullptr) {
                // Per Georgeff et al.: if right_val is a G_PRIME curry, finalize it first
                // This is the y(g1(x)) case where y=F[k] and we need to compute g1(x) first
                Value* finalized = try_finalize_sync(machine, right_val, true);
                if (finalized != nullptr) {
                    right_val = finalized;
                }

                if (right_val->is_basic_value()) {
                    if (inner_fn->is_primitive()) {
                        // Create G_PRIME curry of inner_fn with first_arg=B, preserving axis
                        Value* curried = machine->heap->allocate_curried_fn(inner_fn, right_val, Value::CurryType::G_PRIME, curried_data->axis);
                        machine->result = curried;
                        return;
                    } else {
                        // For closures, dispatch normally (closures don't support axis per ISO spec)
                        machine->throw_error("SYNTAX ERROR: axis specification requires primitive function", this, 1, 0);
                        return;
                    }
                } else {
                    // right_val is a function - can't apply function-with-axis to another function
                    machine->throw_error("DOMAIN ERROR: function with axis requires array argument", this, 11, 0);
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
                        machine->throw_error("SYNTAX ERROR: Function has no monadic form", this, 1, 0);
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
                        machine->throw_error("SYNTAX ERROR: Function has no dyadic form", this, 1, 0);
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
                        machine->throw_error("VALUE ERROR: G_PRIME requires monadic form", this, 2, 0);
                        return;
                    }
                } else {
                    // For closures, need to evaluate g1(x) first then apply y
                    // Push DeferredDispatchK to apply y after g1(x) evaluates
                    // DeferredDispatchK will read machine->result as the new right_val
                    machine->push_kont(machine->heap->allocate<DeferredDispatchK>(right_val, nullptr));
                    // Apply inner closure monadically
                    apply_function_immediate(machine, inner_fn, nullptr, first_arg);
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
            // No axis from this dispatch path (axis would come through curry system)
            if (op->dyadic && actual_left_array && actual_right_array) {
                // Have both array arguments - apply dyadic form with both function operands
                op->dyadic(machine, nullptr, actual_left_array, first_operand, second_func_operand, actual_right_array);
            } else if (op->dyadic && left_val) {
                // Have left array but arrays weren't extracted - use original values
                op->dyadic(machine, nullptr, left_val, first_operand, second_func_operand, right_val);
            } else if (op->monadic && !left_val) {
                // Monadic operator application - prefer this over curry when available
                // This handles cases like +/1 2 3 (reduce with no N argument)
                op->monadic(machine, nullptr, first_operand, right_val);
            } else if (op->dyadic && !left_val) {
                // Only have right argument and no monadic form - curry for second operand
                // Use OPERATOR_CURRY to capture it properly (not DYADIC_CURRY which is for array args)
                Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::OPERATOR_CURRY);
                machine->result = curried;
            } else {
                machine->throw_error("VALUE ERROR: operator requires operands", this, 2, 0);
            }
            return;
        }
    }

    // G2 g' finalization: Unwrap any curried functions in arguments
    // Per paper: when a g' curried function is used as an argument (not at top level),
    // it should be unwrapped. Use consolidated finalization via PerformFinalizeK.
    if (needs_finalization(right_val, true)) {
        machine->result = right_val;
        DeferredDispatchK* then_k = machine->heap->allocate<DeferredDispatchK>(fn_val, left_val);
        push_finalize_then(machine, then_k, true);
        return;
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
                } else if (!def_op->is_ambivalent) {
                    // Non-ambivalent - invoke immediately
                    machine->push_kont(machine->heap->allocate<InvokeDefinedOperatorK>(
                        def_op, op_value, first_operand, nullptr, nullptr, right_val));
                } else {
                    // Ambivalent operator with only right arg - curry to wait for potential left arg
                    Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::G_PRIME);
                    machine->result = curried;
                }
            } else {
                machine->throw_error("VALUE ERROR: operator requires argument", this, 2, 0);
            }
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
                    op->monadic(machine, nullptr, first_operand, right_val);
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
            op->monadic(machine, nullptr, first_operand, right_val);
            return;
        }

        // Dyadic operators with both arguments
        if (op->dyadic && left_val && right_val) {
            op->dyadic(machine, nullptr, left_val, first_operand, nullptr, right_val);
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

        machine->throw_error("VALUE ERROR: operator requires operands", this, 2, 0);
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
            machine->throw_error("VALUE ERROR: operator requires argument", this, 2, 0);
        }
        return;
    }

    if (!fn_val->is_primitive()) {
        machine->throw_error("VALUE ERROR: expected function value", this, 2, 0);
        return;
    }

    PrimitiveFn* prim_fn = fn_val->data.primitive_fn;

    // Pervasive dispatch for strands and NDARRAYs: use CellIterK to iterate elements
    // Only applies to scalar (pervasive) functions, not structural ones
    // Note: monadic case goes through G_PRIME curry first, so only handle dyadic here
    bool left_strand = left_val && left_val->is_strand();
    bool right_strand = right_val && right_val->is_strand();
    bool left_ndarray = left_val && left_val->is_ndarray();
    bool right_ndarray = right_val && right_val->is_ndarray();

    if (prim_fn->is_pervasive && left_val && (left_strand || right_strand || left_ndarray || right_ndarray)) {
        int left_cells = left_val ? count_cells_for_rank(left_val, 0) : 0;
        int right_cells = count_cells_for_rank(right_val, 0);
        int total = std::max(left_cells, right_cells);

        // Scalar extension check: mismatched lengths are an error (unless one is scalar)
        if (left_cells > 1 && right_cells > 1 && left_cells != right_cells) {
            machine->throw_error("LENGTH ERROR: mismatched array lengths", this, 5, 0);
            return;
        }

        // Use CellIterK with rank 0 (iterate 0-cells = elements)
        // For NDARRAY: preserve shape in result
        if (left_ndarray || right_ndarray) {
            // Get shape from the NDARRAY argument (prefer non-scalar)
            const std::vector<int>* shape = nullptr;
            if (right_ndarray && right_cells > 1) {
                shape = &right_val->ndarray_shape();
            } else if (left_ndarray && left_cells > 1) {
                shape = &left_val->ndarray_shape();
            }

            CellIterK* iter = machine->heap->allocate<CellIterK>(
                fn_val, left_val, right_val, 0, 0, total,
                CellIterMode::COLLECT, total, 1, false, false, false);
            if (shape) {
                iter->orig_ndarray_shape = *shape;
            }
            machine->push_kont(iter);
        } else {
            // Strand case: preserve strand structure
            machine->push_kont(machine->heap->allocate<CellIterK>(
                fn_val, left_val, right_val, 0, 0, total,
                CellIterMode::COLLECT, total, 1, true, false, true));
        }
        return;
    }

    // Determine monadic vs dyadic based on what arguments we have
    if (left_val == nullptr) {
        // Monadic case: only right argument
        if (prim_fn->monadic && prim_fn->dyadic) {
            // Overloaded function: G_PRIME curry to defer monadic/dyadic decision
            // This allows dyadic application if a left arg appears later
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::G_PRIME);
            machine->result = curried;
        } else if (prim_fn->monadic) {
            // Monadic-only function: apply immediately
            // This ensures errors are thrown in the correct context for ⎕EA to catch
            prim_fn->monadic(machine, nullptr, right_val);
        } else if (prim_fn->dyadic) {
            // Pure dyadic function: simple currying (right arg captured, waiting for left)
            Value* curried = machine->heap->allocate_curried_fn(fn_val, right_val, Value::CurryType::DYADIC_CURRY);
            machine->result = curried;
        } else {
            // No forms available
            machine->throw_error("SYNTAX ERROR: Function has no forms", this, 1, 0);
            return;
        }
    } else {
        // Dyadic case: both arguments
        if (!prim_fn->dyadic) {
            machine->throw_error("SYNTAX ERROR: Function has no dyadic form", this, 1, 0);
            return;
        }

        // Pervasive strand handling: if either arg is a strand, apply element-wise
        bool left_strand = left_val->is_strand();
        bool right_strand = right_val->is_strand();
        if (prim_fn->is_pervasive && (left_strand || right_strand)) {
            // Get sizes
            int left_size = left_strand ? static_cast<int>(left_val->as_strand()->size())
                                       : (left_val->is_scalar() ? 1 : left_val->size());
            int right_size = right_strand ? static_cast<int>(right_val->as_strand()->size())
                                         : (right_val->is_scalar() ? 1 : right_val->size());

            // Scalar extension for strands
            if (left_val->is_scalar() && right_strand) {
                std::vector<Value*> extended(right_size, left_val);
                left_val = machine->heap->allocate_strand(std::move(extended));
                left_size = right_size;
            } else if (left_strand && right_val->is_scalar()) {
                std::vector<Value*> extended(left_size, right_val);
                right_val = machine->heap->allocate_strand(std::move(extended));
                right_size = left_size;
            }

            // Check lengths match
            if (left_size != right_size) {
                machine->throw_error("LENGTH ERROR: mismatched shapes in pervasive operation", this, 5, 0);
                return;
            }

            // Use CellIterK with COLLECT mode to apply element-wise
            machine->push_kont(machine->heap->allocate<CellIterK>(
                fn_val, left_val, right_val, 0, 0, left_size,
                CellIterMode::COLLECT, left_size, 1, true, false, true));
            return;
        }

        // Apply dyadic function (sets machine->result directly or pushes ThrowErrorK)
        prim_fn->dyadic(machine, nullptr, left_val, right_val);
    }
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
    machine->push_kont(machine->heap->allocate<DispatchFunctionK>(fn_val, left_val, right_val));
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
        machine->throw_error("VALUE ERROR: If condition evaluated to null", this, 2, 0);
        return;
    }

    // APL convention: 0 is false, non-zero is true
    // Per ISO 13751, guard conditions must be scalar boolean
    bool is_true = false;

    if (cond_val->is_scalar()) {
        is_true = (cond_val->as_scalar() != 0.0);
    } else {
        // Non-scalar guard condition is a DOMAIN ERROR per ISO 13751
        machine->throw_error("DOMAIN ERROR: Guard condition must be scalar", this, 11, 0);
        return;
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
        machine->throw_error("VALUE ERROR: While condition evaluated to null", this, 2, 0);
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
    heap->mark(var_name);
    heap->mark(array_expr);
    heap->mark(body);
}

// ForIterateK implementation - iterate over array elements
void ForIterateK::invoke(Machine* machine) {
    // First call: array is in machine->result
    if (array == nullptr) {
        array = machine->result;
        if (!array) {
            machine->throw_error("VALUE ERROR: For loop array evaluated to null", this, 2, 0);
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
    heap->mark(var_name);
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
        machine->throw_error("VALUE ERROR: Branch target evaluated to null", this, 2, 0);
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
        machine->throw_error("DOMAIN ERROR: Branch to line numbers not supported (use →0 or →⍬ to exit)", this, 11, 0);
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
        machine->throw_error("VALUE ERROR: Attempted to call non-function value", this, 2, 0);
        return;
    }

    // Get the function body continuation graph
    Continuation* body = fn_value->data.closure->body;
    if (!body) {
        machine->throw_error("VALUE ERROR: Function has no body", this, 2, 0);
        return;
    }

    // Create new environment for function scope (GC-managed)
    Environment* call_env = machine->heap->allocate<Environment>(machine->env);  // Parent for closures

    // Bind arguments in function environment
    if (right_arg) {
        call_env->define(machine->string_pool.intern("⍵"), right_arg);
    }
    if (left_arg) {
        call_env->define(machine->string_pool.intern("⍺"), left_arg);
    }
    // Bind ∇ for recursive self-reference
    call_env->define(machine->string_pool.intern("∇"), fn_value);

    // Save current environment
    Environment* saved_env = machine->env;

    // Switch to function environment
    machine->env = call_env;

    // Push restore environment continuation (executes after function returns)
    RestoreEnvK* restore_k = machine->heap->allocate<RestoreEnvK>(saved_env);
    machine->push_kont(restore_k);

    // Push finalization to resolve any G_PRIME curries BEFORE environment restoration
    // This ensures closures created inside the dfn execute while their environment is active
    PerformFinalizeK* finalize_k = machine->heap->allocate<PerformFinalizeK>(true);
    machine->push_kont(finalize_k);

    // Push CatchReturnK to establish function boundary for →0 and :Return
    CatchReturnK* catch_k = machine->heap->allocate<CatchReturnK>(machine->string_pool.intern("dfn"));
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
    heap->mark(op_name);
    heap->mark(operand_cont);
    heap->mark(axis_cont);
}

// ApplyDerivedOperatorK implementation - create DERIVED_OPERATOR value
void ApplyDerivedOperatorK::invoke(Machine* machine) {
    Value* first_operand = machine->result;

    // Look up the operator by name from environment
    Value* op_val = machine->env->lookup(op_name);
    if (!op_val) {
        std::string msg = std::string("VALUE ERROR: Unknown operator: ") + op_name->c_str();
        machine->throw_error(msg.c_str(), this, 2, 0);
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
        std::string msg = std::string("VALUE ERROR: Not an operator: ") + op_name->c_str();
        machine->throw_error(msg.c_str(), this, 2, 0);
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
        machine->throw_error("SYNTAX ERROR: Operator has neither monadic nor dyadic form", this, 1, 0);
    }
}

void ApplyDerivedOperatorK::mark(Heap* heap) {
    heap->mark(op_name);
    heap->mark(axis_cont);
}

// ApplyAxisK implementation - apply axis to derived operator
void ApplyAxisK::invoke(Machine* machine) {
    Value* axis = machine->result;

    // Store axis in the axis field, not first_arg
    // This keeps axis separate from operands, so dispatch doesn't need special cases
    Value* curried = machine->heap->allocate_curried_fn(
        derived_op, nullptr, Value::CurryType::OPERATOR_CURRY, axis);
    machine->result = curried;
}

void ApplyAxisK::mark(Heap* heap) {
    heap->mark(derived_op);
}

// ============================================================================
// CellIterK - General-purpose cell iterator
// ============================================================================
// Note: Helper functions (get_value_rank, count_cells_for_rank, extract_cell)
// are defined earlier in the file under "Cell Iterator Helpers".

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
                if (orig_is_strand) {
                    // Strand input: preserve strand structure even for single/scalar results
                    machine->result = machine->heap->allocate_strand(std::move(results));
                } else if (!orig_ndarray_shape.empty()) {
                    // NDARRAY input: preserve NDARRAY shape
                    Eigen::VectorXd data(results.size());
                    for (size_t i = 0; i < results.size(); i++) {
                        data(i) = results[i]->as_scalar();
                    }
                    machine->result = machine->heap->allocate_ndarray(data, orig_ndarray_shape);
                } else if (results.size() == 1) {
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
                // Results are non-scalars
                // If input was a strand, preserve strand structure
                if (orig_is_strand) {
                    machine->result = machine->heap->allocate_strand(std::move(results));
                } else {
                    // Check if all results have the same shape
                    bool all_same_shape = true;
                    std::vector<int> cell_shape;

                    // Determine cell shape from first result
                    if (results[0]->is_vector()) {
                        cell_shape = {results[0]->rows()};
                    } else if (results[0]->is_matrix()) {
                        cell_shape = {results[0]->rows(), results[0]->cols()};
                    } else if (results[0]->is_ndarray()) {
                        cell_shape = results[0]->ndarray_shape();
                    } else {
                        all_same_shape = false;
                    }

                    // Check all results have same shape
                    if (all_same_shape) {
                        for (size_t i = 1; i < results.size(); i++) {
                            std::vector<int> this_shape;
                            if (results[i]->is_vector()) {
                                this_shape = {results[i]->rows()};
                            } else if (results[i]->is_matrix()) {
                                this_shape = {results[i]->rows(), results[i]->cols()};
                            } else if (results[i]->is_ndarray()) {
                                this_shape = results[i]->ndarray_shape();
                            } else {
                                all_same_shape = false;
                                break;
                            }
                            if (this_shape != cell_shape) {
                                all_same_shape = false;
                                break;
                            }
                        }
                    }

                    if (all_same_shape && !cell_shape.empty()) {
                        // Combine frame_shape + cell_shape into result NDARRAY
                        std::vector<int> result_shape;

                        // Build frame shape
                        if (!orig_ndarray_shape.empty()) {
                            result_shape = orig_ndarray_shape;
                        } else if (orig_is_vector) {
                            result_shape = {orig_rows};
                        } else {
                            result_shape = {orig_rows, orig_cols};
                        }

                        // Append cell shape
                        result_shape.insert(result_shape.end(), cell_shape.begin(), cell_shape.end());

                        // Calculate total size
                        int total_size = 1;
                        for (int d : result_shape) total_size *= d;

                        // Flatten all results into data vector
                        Eigen::VectorXd data(total_size);
                        int pos = 0;
                        for (Value* v : results) {
                            if (v->is_vector()) {
                                const Eigen::MatrixXd* mat = v->as_matrix();
                                for (int j = 0; j < mat->rows(); j++) {
                                    data(pos++) = (*mat)(j, 0);
                                }
                            } else if (v->is_matrix()) {
                                const Eigen::MatrixXd* mat = v->as_matrix();
                                for (int r = 0; r < mat->rows(); r++) {
                                    for (int c = 0; c < mat->cols(); c++) {
                                        data(pos++) = (*mat)(r, c);
                                    }
                                }
                            } else if (v->is_ndarray()) {
                                const Eigen::VectorXd* nd = v->ndarray_data();
                                for (int j = 0; j < nd->size(); j++) {
                                    data(pos++) = (*nd)(j);
                                }
                            }
                        }

                        // Create result based on final shape
                        if (result_shape.size() == 1) {
                            machine->result = machine->heap->allocate_vector(data, orig_is_char);
                        } else if (result_shape.size() == 2) {
                            Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
                            for (int i = 0; i < result_shape[0]; i++) {
                                for (int j = 0; j < result_shape[1]; j++) {
                                    mat(i, j) = data(i * result_shape[1] + j);
                                }
                            }
                            machine->result = machine->heap->allocate_matrix(mat, orig_is_char);
                        } else {
                            machine->result = machine->heap->allocate_ndarray(data, result_shape);
                        }
                    } else {
                        // Mixed results - create STRAND
                        machine->result = machine->heap->allocate_strand(std::move(results));
                    }
                }
            }
            return;
        }

        // Extract cells
        Value* left_cell = extract_cell(machine, lhs, left_rank,
            (lhs && count_cells_for_rank(lhs, left_rank) == 1) ? 0 : current_cell);
        Value* right_cell = extract_cell(machine, rhs, right_rank, current_cell);

        if (!right_cell) {
            machine->throw_error("INDEX ERROR: cell extraction failed", this, 3, 0);
            return;
        }

        // Push collector continuation, then dispatch function
        machine->push_kont(machine->heap->allocate<CellCollectK>(this));
        if (left_cell == nullptr) {
            // Monadic - apply immediately without currying
            apply_function_immediate(machine, fn, nullptr, right_cell);
        } else {
            // Dyadic - use dispatch (may curry if needed)
            machine->push_kont(machine->heap->allocate<DispatchFunctionK>(fn, left_cell, right_cell));
        }

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

            // Strand scan: return results as strand
            if (orig_is_strand) {
                machine->result = machine->heap->allocate_strand(std::move(results));
                return;
            }

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

    } else if (mode == CellIterMode::SCAN_LEFT) {
        // Forward iteration for left-to-right scan (strand scan)
        if (current_cell >= total_cells) {
            // Done - results are already in order
            if (orig_is_strand) {
                machine->result = machine->heap->allocate_strand(std::move(results));
                return;
            }
            // For non-strand, assemble as vector
            Eigen::VectorXd vec(results.size());
            for (size_t i = 0; i < results.size(); i++) {
                vec(i) = results[i]->as_scalar();
            }
            machine->result = machine->heap->allocate_vector(vec, orig_is_char);
            return;
        }

        if (!accumulator) {
            // First iteration - first element is its own scan result
            accumulator = extract_cell(machine, rhs, right_rank, current_cell);
            results.push_back(accumulator);
            current_cell++;
            machine->push_kont(this);
            return;
        }

        // Apply: accumulator f element (left-to-right)
        Value* element = extract_cell(machine, rhs, right_rank, current_cell);

        machine->push_kont(machine->heap->allocate<CellCollectK>(this));
        machine->push_kont(machine->heap->allocate<DispatchFunctionK>(fn, accumulator, element));

    } else if (mode == CellIterMode::OUTER) {
        // Cartesian product iteration for outer product (ISO 9.3.1)
        if (current_cell >= total_cells) {
            // Done - assemble results
            if (lhs_total == 0 || rhs_total == 0) {
                // Empty result - return empty matrix with correct shape
                machine->result = machine->heap->allocate_matrix(Eigen::MatrixXd(lhs_total, rhs_total));
                return;
            }
            if (lhs_total == 1 && rhs_total == 1) {
                // Scalar result (both sides scalar)
                machine->result = results[0];
                return;
            }

            // Check if all results are scalars - if so, build a matrix
            // Otherwise build a strand of strands (nested array)
            bool all_scalars = true;
            for (Value* r : results) {
                if (!r->is_scalar()) {
                    all_scalars = false;
                    break;
                }
            }

            if (all_scalars) {
                // Check if result should be NDARRAY (rank > 2)
                if (!orig_ndarray_shape.empty()) {
                    // NDARRAY result
                    Eigen::VectorXd data(results.size());
                    for (size_t i = 0; i < results.size(); i++) {
                        data(i) = results[i]->as_scalar();
                    }
                    machine->result = machine->heap->allocate_ndarray(data, orig_ndarray_shape);
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
            } else {
                // Nested array result: build strand of strands (lhs_total rows, each with rhs_total elements)
                std::vector<Value*> outer;
                outer.reserve(lhs_total);
                for (int i = 0; i < lhs_total; i++) {
                    std::vector<Value*> inner;
                    inner.reserve(rhs_total);
                    for (int j = 0; j < rhs_total; j++) {
                        inner.push_back(results[i * rhs_total + j]);
                    }
                    if (rhs_total == 1) {
                        // Single-element row: don't wrap in strand
                        outer.push_back(inner[0]);
                    } else {
                        outer.push_back(machine->heap->allocate_strand(std::move(inner)));
                    }
                }
                if (lhs_total == 1) {
                    // Single row: return the inner strand directly
                    machine->result = outer[0];
                } else {
                    machine->result = machine->heap->allocate_strand(std::move(outer));
                }
            }
            return;
        }

        // Compute (i, j) from linear index
        int i = current_cell / rhs_total;
        int j = current_cell % rhs_total;

        // Extract lhs[i] and rhs[j], handling strands and NDARRAY
        Value* left_cell;
        Value* right_cell;

        if (lhs->is_scalar()) {
            left_cell = lhs;
        } else if (lhs->is_strand()) {
            const std::vector<Value*>* strand = lhs->as_strand();
            left_cell = (*strand)[i];
        } else if (lhs->is_ndarray()) {
            const Value::NDArrayData* nd = lhs->as_ndarray();
            left_cell = machine->heap->allocate_scalar((*nd->data)(i));
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
        } else if (rhs->is_strand()) {
            const std::vector<Value*>* strand = rhs->as_strand();
            right_cell = (*strand)[j];
        } else if (rhs->is_ndarray()) {
            const Value::NDArrayData* nd = rhs->as_ndarray();
            right_cell = machine->heap->allocate_scalar((*nd->data)(j));
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
    } else if (mode == CellIterMode::INNER) {
        // Inner product: extract fibers, apply g element-wise, reduce with f
        // Result shape is (¯1↓⍴A),1↓⍴B - stored in orig_rows/orig_cols or orig_ndarray_shape
        if (current_cell >= total_cells) {
            // Done - assemble results using orig_rows/orig_cols (set by op_inner_product)
            if (results.size() == 1) {
                // Scalar result
                machine->result = results[0];
            } else if (!orig_ndarray_shape.empty()) {
                // NDARRAY result (rank > 2)
                Eigen::VectorXd data(results.size());
                for (size_t k = 0; k < results.size(); k++) {
                    data(k) = results[k]->as_scalar();
                }
                machine->result = machine->heap->allocate_ndarray(data, orig_ndarray_shape);
            } else if (orig_is_vector) {
                // Vector result
                Eigen::VectorXd vec(results.size());
                for (size_t k = 0; k < results.size(); k++) {
                    vec(k) = results[k]->as_scalar();
                }
                machine->result = machine->heap->allocate_vector(vec);
            } else {
                // Matrix result - use orig_rows × orig_cols
                Eigen::MatrixXd mat(orig_rows, orig_cols);
                for (int ii = 0; ii < orig_rows; ii++) {
                    for (int jj = 0; jj < orig_cols; jj++) {
                        mat(ii, jj) = results[ii * orig_cols + jj]->as_scalar();
                    }
                }
                machine->result = machine->heap->allocate_matrix(mat);
            }
            return;
        }

        // Compute (i, j) from linear index
        int i = current_cell / rhs_total;
        int j = current_cell % rhs_total;

        // Extract fiber i from lhs (along last axis) and fiber j from rhs (along first axis)
        Value* lhs_fiber;
        Value* rhs_fiber;

        if (lhs->is_vector()) {
            // Vector: the whole vector is the fiber
            lhs_fiber = lhs;
        } else if (lhs->is_ndarray()) {
            // NDARRAY: extract fiber along last axis at frame position i
            const Value::NDArrayData* nd = lhs->as_ndarray();
            int rank = nd->shape.size();
            int last_dim = nd->shape[rank - 1];
            // Compute strides
            std::vector<int> strides(rank);
            strides[rank - 1] = 1;
            for (int k = rank - 2; k >= 0; k--) {
                strides[k] = strides[k + 1] * nd->shape[k + 1];
            }
            // Frame position i -> indices in all but last axis
            int frame_size = static_cast<int>(nd->data->size()) / last_dim;
            int pos = i;
            int base = 0;
            for (int k = 0; k < rank - 1; k++) {
                int dim_stride = frame_size / nd->shape[k];
                int idx = pos / dim_stride;
                pos = pos % dim_stride;
                base += idx * strides[k];
                frame_size = dim_stride;
            }
            Eigen::VectorXd fiber(last_dim);
            for (int k = 0; k < last_dim; k++) {
                fiber(k) = (*nd->data)(base + k);
            }
            lhs_fiber = machine->heap->allocate_vector(fiber);
        } else {
            // Matrix: extract row i
            const Eigen::MatrixXd* mat = lhs->as_matrix();
            Eigen::VectorXd row = mat->row(i).transpose();
            lhs_fiber = machine->heap->allocate_vector(row);
        }

        if (rhs->is_vector()) {
            // Vector: the whole vector is the fiber
            rhs_fiber = rhs;
        } else if (rhs->is_ndarray()) {
            // NDARRAY: extract fiber along first axis at frame position j
            const Value::NDArrayData* nd = rhs->as_ndarray();
            int rank = nd->shape.size();
            int first_dim = nd->shape[0];
            // Stride along first axis
            int first_stride = static_cast<int>(nd->data->size()) / first_dim;
            // Frame position j -> indices in all but first axis
            int frame_size = first_stride;
            int pos = j;
            int base = 0;
            for (int k = 1; k < rank; k++) {
                int dim_stride = frame_size / nd->shape[k];
                int idx = pos / dim_stride;
                pos = pos % dim_stride;
                base += idx;
                if (k < rank - 1) {
                    base *= nd->shape[k + 1];
                }
                frame_size = dim_stride;
            }
            // Re-compute base correctly
            std::vector<int> strides(rank);
            strides[rank - 1] = 1;
            for (int k = rank - 2; k >= 0; k--) {
                strides[k] = strides[k + 1] * nd->shape[k + 1];
            }
            // Convert j to multi-index for axes 1..rank-1
            pos = j;
            base = 0;
            frame_size = first_stride;
            for (int k = 1; k < rank; k++) {
                frame_size = frame_size / nd->shape[k];
                int idx = pos / frame_size;
                pos = pos % frame_size;
                base += idx * strides[k];
            }
            Eigen::VectorXd fiber(first_dim);
            for (int k = 0; k < first_dim; k++) {
                fiber(k) = (*nd->data)(base + k * strides[0]);
            }
            rhs_fiber = machine->heap->allocate_vector(fiber);
        } else {
            // Matrix: extract column j
            const Eigen::MatrixXd* mat = rhs->as_matrix();
            Eigen::VectorXd col = mat->col(j);
            rhs_fiber = machine->heap->allocate_vector(col);
        }

        // Push: collector -> ReduceResultK(f) -> CellIterK COLLECT(g, lhs_fiber, rhs_fiber)
        machine->push_kont(machine->heap->allocate<CellCollectK>(this));
        machine->push_kont(machine->heap->allocate<ReduceResultK>(fn));
        machine->push_kont(machine->heap->allocate<CellIterK>(
            g_fn, lhs_fiber, rhs_fiber, 0, 0, common_dim,
            CellIterMode::COLLECT, common_dim, 1, true));
    }
}

void CellIterK::mark(Heap* heap) {
    heap->mark(fn);
    heap->mark(lhs);
    heap->mark(rhs);
    heap->mark(accumulator);
    heap->mark(g_fn);  // For INNER mode
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
    } else if (iter->mode == CellIterMode::SCAN_LEFT) {
        iter->accumulator = result;
        iter->results.push_back(result);
        iter->current_cell++;
    } else if (iter->mode == CellIterMode::OUTER) {
        iter->results.push_back(result);
        iter->current_cell++;
    } else if (iter->mode == CellIterMode::INNER) {
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
// FiberReduceK - Unified reduction along array fibers
// ============================================================================

// Constructor: computes all iteration parameters from source array
FiberReduceK::FiberReduceK(Value* f, Value* src, int ax, int window, bool rev)
    : fn(f), source(src), axis(ax), window_size(window), reverse(rev),
      current_result(0), is_strand(src->is_strand()) {

    // Compute source shape and fiber parameters based on array type
    if (src->is_scalar()) {
        // Scalar: 1 fiber of length 1
        source_shape = {};
        fiber_length = 1;
        total_fibers = 1;
    } else if (src->is_strand()) {
        // Strand: 1 fiber, length = strand size
        auto* strand = src->as_strand();
        fiber_length = static_cast<int>(strand->size());
        total_fibers = 1;
        source_shape = {fiber_length};
    } else if (src->is_vector()) {
        // Vector: 1 fiber, length = vector size
        const Eigen::MatrixXd* mat = src->as_matrix();
        fiber_length = mat->rows();
        total_fibers = 1;
        source_shape = {fiber_length};
    } else if (src->is_ndarray()) {
        // NDARRAY: compute from shape
        const Value::NDArrayData* nd = src->as_ndarray();
        source_shape = nd->shape;
        fiber_length = source_shape[axis];
        total_fibers = 1;
        for (size_t d = 0; d < source_shape.size(); ++d) {
            if (static_cast<int>(d) != axis) {
                total_fibers *= source_shape[d];
            }
        }
        // Compute strides for element access
        int rank = static_cast<int>(source_shape.size());
        source_strides.resize(rank);
        source_strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            source_strides[d] = source_strides[d + 1] * source_shape[d + 1];
        }
    } else {
        // Matrix: rows or columns as fibers
        const Eigen::MatrixXd* mat = src->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();
        source_shape = {rows, cols};
        if (axis == 0) {
            // Reduce along rows (columns are fibers)
            fiber_length = rows;
            total_fibers = cols;
        } else {
            // Reduce along columns (rows are fibers)
            fiber_length = cols;
            total_fibers = rows;
        }
    }

    // Compute windows per fiber
    if (window_size == 0) {
        // Full reduce: one window covering entire fiber
        windows_per_fiber = 1;
    } else {
        // N-wise: sliding windows
        windows_per_fiber = fiber_length - window_size + 1;
        if (windows_per_fiber < 0) windows_per_fiber = 0;
    }

    total_results = total_fibers * windows_per_fiber;
    results.reserve(total_results);

    // Compute result shape
    if (total_results == 0) {
        result_shape = {};
    } else if (src->is_scalar() || src->is_vector() || src->is_strand()) {
        // 1D input
        if (window_size == 0) {
            // Full reduce → scalar (no shape needed, single result)
            result_shape = {};
        } else {
            // N-wise → vector of length windows_per_fiber
            result_shape = {windows_per_fiber};
        }
    } else if (src->is_ndarray()) {
        // N-D input: result shape excludes axis (full) or shrinks axis (N-wise)
        if (window_size == 0 || windows_per_fiber == 1) {
            // Full reduce OR N equals axis length: remove axis from shape
            // Per ISO 9.2.3: when N equals length of axis, result is like full reduce
            for (size_t d = 0; d < source_shape.size(); ++d) {
                if (static_cast<int>(d) != axis) {
                    result_shape.push_back(source_shape[d]);
                }
            }
        } else {
            // N-wise: axis dimension becomes windows_per_fiber
            for (size_t d = 0; d < source_shape.size(); ++d) {
                if (static_cast<int>(d) == axis) {
                    result_shape.push_back(windows_per_fiber);
                } else {
                    result_shape.push_back(source_shape[d]);
                }
            }
        }
    } else {
        // Matrix
        if (window_size == 0) {
            // Full reduce: vector of length total_fibers
            result_shape = {total_fibers};
        } else {
            // N-wise: matrix with one dimension shrunk
            if (axis == 0) {
                result_shape = {windows_per_fiber, total_fibers};
            } else {
                result_shape = {total_fibers, windows_per_fiber};
            }
        }
    }
}

void FiberReduceK::invoke(Machine* machine) {
    // Done: assemble results
    if (current_result >= total_results) {
        if (results.empty()) {
            // Empty result (e.g., N > fiber_length)
            if (is_strand) {
                machine->result = machine->heap->allocate_strand(std::vector<Value*>());
            } else {
                Eigen::VectorXd empty(0);
                machine->result = machine->heap->allocate_vector(empty);
            }
            return;
        }

        // Single scalar result (full reduce, or N-wise where N equals fiber length)
        // Per ISO 9.2.3: when N equals length of B, return f/B directly (unwrapped)
        if (results.size() == 1 && (result_shape.empty() ||
            (result_shape.size() == 1 && result_shape[0] == 1))) {
            machine->result = results[0];
            return;
        }

        // Strand result
        if (is_strand && result_shape.size() <= 1) {
            if (results.size() == 1) {
                machine->result = results[0];
            } else {
                machine->result = machine->heap->allocate_strand(std::move(results));
            }
            return;
        }

        // Assemble based on result shape
        if (result_shape.size() == 1) {
            // Vector result
            Eigen::VectorXd vec(results.size());
            for (size_t i = 0; i < results.size(); i++) {
                vec(i) = results[i]->as_scalar();
            }
            machine->result = machine->heap->allocate_vector(vec);
        } else if (result_shape.size() == 2) {
            // Matrix result
            Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
            for (size_t i = 0; i < results.size(); i++) {
                int row = i / result_shape[1];
                int col = i % result_shape[1];
                mat(row, col) = results[i]->as_scalar();
            }
            machine->result = machine->heap->allocate_matrix(mat);
        } else {
            // NDARRAY result
            Eigen::VectorXd data(results.size());
            for (size_t i = 0; i < results.size(); i++) {
                data(i) = results[i]->as_scalar();
            }
            machine->result = machine->heap->allocate_ndarray(data, result_shape);
        }
        return;
    }

    // Compute which fiber and window we're processing
    // For multi-dimensional results, we iterate in result row-major order
    // which means window varies slowest, fiber varies fastest (for matrices with axis=0)
    // For axis=1 (rows are fibers), result is (fibers, windows), so fiber varies slowest
    int fiber_idx, window_idx;
    if (source->is_matrix() && axis == 0) {
        // axis=0: result is (windows, fibers), iterate window first then fiber
        window_idx = current_result / total_fibers;
        fiber_idx = current_result % total_fibers;
    } else {
        // Default: fiber first, then window (for vectors, strands, axis=1 matrices, NDARRAY)
        fiber_idx = current_result / windows_per_fiber;
        window_idx = current_result % windows_per_fiber;
    }

    // Determine window length (full fiber or window_size)
    int win_len = (window_size == 0) ? fiber_length : window_size;

    // Extract window elements as a Value*
    Value* window_val;

    if (source->is_strand()) {
        // Extract from strand
        auto* strand = source->as_strand();
        std::vector<Value*> elements;
        elements.reserve(win_len);
        for (int i = 0; i < win_len; i++) {
            int idx = reverse ? (window_idx + win_len - 1 - i) : (window_idx + i);
            elements.push_back((*strand)[idx]);
        }
        window_val = machine->heap->allocate_strand(std::move(elements));

        // Reduce this window
        machine->push_kont(machine->heap->allocate<FiberReduceCollectK>(this));
        machine->push_kont(machine->heap->allocate<CellIterK>(
            fn, nullptr, window_val, 0, 0, win_len,
            CellIterMode::FOLD_RIGHT, win_len, 1, true, false, true));

    } else if (source->is_scalar()) {
        // Scalar: just return it
        results.push_back(source);
        current_result++;
        machine->push_kont(this);

    } else if (source->is_vector()) {
        // Extract window from vector
        const Eigen::MatrixXd* mat = source->as_matrix();
        Eigen::VectorXd window(win_len);
        for (int i = 0; i < win_len; i++) {
            int idx = reverse ? (window_idx + win_len - 1 - i) : (window_idx + i);
            window(i) = (*mat)(idx, 0);
        }
        window_val = machine->heap->allocate_vector(window);

        machine->push_kont(machine->heap->allocate<FiberReduceCollectK>(this));
        machine->push_kont(machine->heap->allocate<CellIterK>(
            fn, nullptr, window_val, 0, 0, win_len,
            CellIterMode::FOLD_RIGHT, win_len, 1, true));

    } else if (source->is_ndarray()) {
        // Extract fiber from NDARRAY, then window from fiber
        const Value::NDArrayData* nd = source->as_ndarray();
        int rank = static_cast<int>(source_shape.size());

        // Convert fiber_idx to indices for non-axis dimensions
        std::vector<int> src_idx(rank, 0);
        int fidx = fiber_idx;
        for (int d = rank - 1; d >= 0; --d) {
            if (d != axis) {
                src_idx[d] = fidx % source_shape[d];
                fidx /= source_shape[d];
            }
        }

        // Extract window elements from fiber
        Eigen::VectorXd window(win_len);
        for (int i = 0; i < win_len; i++) {
            int ax_pos = reverse ? (window_idx + win_len - 1 - i) : (window_idx + i);
            src_idx[axis] = ax_pos;
            int flat = 0;
            for (int d = 0; d < rank; ++d) {
                flat += src_idx[d] * source_strides[d];
            }
            window(i) = (*nd->data)(flat);
        }
        window_val = machine->heap->allocate_vector(window);

        machine->push_kont(machine->heap->allocate<FiberReduceCollectK>(this));
        machine->push_kont(machine->heap->allocate<CellIterK>(
            fn, nullptr, window_val, 0, 0, win_len,
            CellIterMode::FOLD_RIGHT, win_len, 1, true));

    } else {
        // Matrix: extract row or column, then window
        const Eigen::MatrixXd* mat = source->as_matrix();
        Eigen::VectorXd window(win_len);

        if (axis == 0) {
            // Fibers are columns
            for (int i = 0; i < win_len; i++) {
                int row = reverse ? (window_idx + win_len - 1 - i) : (window_idx + i);
                window(i) = (*mat)(row, fiber_idx);
            }
        } else {
            // Fibers are rows
            for (int i = 0; i < win_len; i++) {
                int col = reverse ? (window_idx + win_len - 1 - i) : (window_idx + i);
                window(i) = (*mat)(fiber_idx, col);
            }
        }
        window_val = machine->heap->allocate_vector(window);

        machine->push_kont(machine->heap->allocate<FiberReduceCollectK>(this));
        machine->push_kont(machine->heap->allocate<CellIterK>(
            fn, nullptr, window_val, 0, 0, win_len,
            CellIterMode::FOLD_RIGHT, win_len, 1, true));
    }
}

void FiberReduceK::mark(Heap* heap) {
    heap->mark(fn);
    heap->mark(source);
    for (Value* v : results) {
        heap->mark(v);
    }
}

void FiberReduceCollectK::invoke(Machine* machine) {
    iter->results.push_back(machine->result);
    iter->current_result++;
    machine->push_kont(iter);
}

void FiberReduceCollectK::mark(Heap* heap) {
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
    if (current_pos >= total_positions) {
        // Done - assemble results based on result shape
        if (results.empty()) {
            machine->result = machine->heap->allocate_scalar(0);
            return;
        }

        if (!result_shape.empty()) {
            // NDARRAY scan: assemble results into NDARRAY
            int ax_len = result_shape[scan_axis];
            int total_size = 1;
            for (int s : result_shape) total_size *= s;

            Eigen::VectorXd data(total_size);

            // Compute strides for result
            int rank = static_cast<int>(result_shape.size());
            std::vector<int> strides(rank);
            strides[rank - 1] = 1;
            for (int d = rank - 2; d >= 0; --d) {
                strides[d] = strides[d + 1] * result_shape[d + 1];
            }

            // Each result is a vector of length ax_len
            // Map result[pos][k] → data[flat_index]
            for (int pos = 0; pos < total_positions; pos++) {
                const Eigen::MatrixXd* vec = results[pos]->as_matrix();

                // Compute base indices (excluding scan axis)
                std::vector<int> base_idx(rank, 0);
                int p = pos;
                int axis_skip = 0;
                for (int d = rank - 1; d >= 0; --d) {
                    if (d == scan_axis) {
                        axis_skip = 1;
                        continue;
                    }
                    int dim_size = result_shape[d];
                    base_idx[d] = p % dim_size;
                    p /= dim_size;
                }

                // Copy scanned values
                for (int k = 0; k < ax_len; k++) {
                    base_idx[scan_axis] = k;
                    int flat = 0;
                    for (int d = 0; d < rank; ++d) {
                        flat += base_idx[d] * strides[d];
                    }
                    data(flat) = (*vec)(k, 0);
                }
            }

            if (result_shape.size() == 2) {
                Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
                for (int i = 0; i < result_shape[0]; i++) {
                    for (int j = 0; j < result_shape[1]; j++) {
                        mat(i, j) = data(i * result_shape[1] + j);
                    }
                }
                machine->result = machine->heap->allocate_matrix(mat);
            } else {
                machine->result = machine->heap->allocate_ndarray(data, result_shape);
            }
            return;
        }

        // Matrix scan
        if (scan_axis == 1) {
            // Regular scan: results are row vectors, assemble into matrix
            int result_cols = results[0]->rows();  // Each result is a vector
            Eigen::MatrixXd mat(total_positions, result_cols);
            for (int r = 0; r < total_positions; r++) {
                const Eigen::MatrixXd* row_vec = results[r]->as_matrix();
                mat.row(r) = row_vec->col(0).transpose();
            }
            machine->result = machine->heap->allocate_matrix(mat);
        } else {
            // Scan-first: results are column vectors, assemble into matrix
            int result_rows = results[0]->rows();  // Each result is a vector
            Eigen::MatrixXd mat(result_rows, total_positions);
            for (int c = 0; c < total_positions; c++) {
                const Eigen::MatrixXd* col_vec = results[c]->as_matrix();
                mat.col(c) = col_vec->col(0);
            }
            machine->result = machine->heap->allocate_matrix(mat);
        }
        return;
    }

    // NDARRAY scan
    if (source->is_ndarray()) {
        const Value::NDArrayData* nd = source->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());
        int ax_len = shape[scan_axis];

        // Compute strides for source array
        std::vector<int> strides(rank);
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // Compute base indices for current_pos (excluding scan axis)
        std::vector<int> base_idx(rank, 0);
        int pos = current_pos;
        for (int d = rank - 1; d >= 0; --d) {
            if (d == scan_axis) continue;
            int dim_size = shape[d];
            base_idx[d] = pos % dim_size;
            pos /= dim_size;
        }

        // Extract fiber along scan axis as vector
        Eigen::VectorXd fiber(ax_len);
        for (int k = 0; k < ax_len; ++k) {
            base_idx[scan_axis] = k;
            int flat = 0;
            for (int d = 0; d < rank; ++d) {
                flat += base_idx[d] * strides[d];
            }
            fiber(k) = (*nd->data)(flat);
        }
        Value* fiber_vec = machine->heap->allocate_vector(fiber);

        // Push collector, then PrefixScanK for this fiber
        machine->push_kont(machine->heap->allocate<RowScanCollectK>(this));
        if (ax_len <= 1) {
            machine->result = fiber_vec;
            return;
        }
        machine->push_kont(machine->heap->allocate<PrefixScanK>(fn, fiber_vec, ax_len));
        return;
    }

    // Matrix scan (original behavior)
    const Eigen::MatrixXd* mat = source->as_matrix();

    if (scan_axis == 1) {
        // Regular scan (\): scan each row
        Eigen::VectorXd row = mat->row(current_pos).transpose();
        Value* row_vec = machine->heap->allocate_vector(row);

        machine->push_kont(machine->heap->allocate<RowScanCollectK>(this));
        int row_len = row.rows();
        if (row_len <= 1) {
            machine->result = row_vec;
            return;
        }
        machine->push_kont(machine->heap->allocate<PrefixScanK>(fn, row_vec, row_len));
    } else {
        // Scan-first (⍀): scan each column
        Eigen::VectorXd col = mat->col(current_pos);
        Value* col_vec = machine->heap->allocate_vector(col);

        machine->push_kont(machine->heap->allocate<RowScanCollectK>(this));
        int col_len = col.rows();
        if (col_len <= 1) {
            machine->result = col_vec;
            return;
        }
        machine->push_kont(machine->heap->allocate<PrefixScanK>(fn, col_vec, col_len));
    }
}

void RowScanK::mark(Heap* heap) {
    heap->mark(fn);
    heap->mark(source);
    for (Value* v : results) {
        heap->mark(v);
    }
}

void RowScanCollectK::invoke(Machine* machine) {
    Value* result = machine->result;
    iter->results.push_back(result);
    iter->current_pos++;
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
        machine->throw_error("DOMAIN ERROR: cannot reduce empty vector", this, 11, 0);
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
// Indexed Assignment Continuations
// ============================================================================

void IndexedAssignK::invoke(Machine* machine) {
    // Evaluate value first (APL right-to-left), then index
    machine->push_kont(machine->heap->allocate<IndexedAssignIndexK>(var_name, nullptr, index_cont));
    machine->push_kont(value_cont);
}

void IndexedAssignK::mark(Heap* heap) {
    heap->mark(var_name);
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
    heap->mark(var_name);
    heap->mark(value_val);
    heap->mark(index_cont);
}

void PerformIndexedAssignK::invoke(Machine* machine) {
    // Index just evaluated
    index_val = machine->result;

    // g' finalization: If index is a curry, finalize it first
    if (maybe_push_finalize(machine, index_val, this)) {
        return;
    }
    // After finalization, use machine->result (may have been updated by sync finalization)
    index_val = machine->result;

    // Lookup the array variable
    Value* arr = machine->env->lookup(var_name);
    if (!arr) {
        machine->throw_error("VALUE ERROR: undefined variable in indexed assignment", this, 2, 0);
        return;
    }

    // Convert string to char vector if needed (strings become numeric on modification)
    if (arr->is_string()) {
        arr = arr->to_char_vector(machine->heap);
        machine->env->define(var_name, arr);  // Update binding to converted array
    }

    // NDARRAY indexed assignment: A[I;J;K]←V
    if (arr->is_ndarray()) {
        if (!index_val->is_strand()) {
            machine->throw_error("RANK ERROR: NDARRAY requires multi-axis index", this, 4, 0);
            return;
        }
        auto* idx_strand = index_val->as_strand();
        const Value::NDArrayData* nd = arr->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        if (static_cast<int>(idx_strand->size()) != rank) {
            machine->throw_error("RANK ERROR: index count must match array rank", this, 4, 0);
            return;
        }

        // Helper to check if index is elided
        auto is_elided = [](Value* idx) -> bool {
            return idx->is_vector() && idx->size() == 0;
        };

        // Helper to validate index is near-integer
        auto validate_index = [machine, this](double val) -> bool {
            double rounded = std::round(val);
            if (std::abs(val - rounded) > 1e-10) {
                machine->throw_error("DOMAIN ERROR: index must be integer", this, 11, 0);
                return false;
            }
            return true;
        };

        // Gather indices for each axis
        std::vector<std::vector<int>> axis_indices(rank);
        for (int d = 0; d < rank; ++d) {
            Value* idx = (*idx_strand)[d];
            if (is_elided(idx)) {
                for (int i = 0; i < shape[d]; ++i) {
                    axis_indices[d].push_back(i);
                }
            } else if (idx->is_scalar()) {
                double val = idx->as_scalar();
                if (!validate_index(val)) return;
                int i = static_cast<int>(std::round(val)) - machine->io;
                if (i < 0 || i >= shape[d]) {
                    machine->throw_error("INDEX ERROR: index out of bounds", this, 3, 0);
                    return;
                }
                axis_indices[d].push_back(i);
            } else if (idx->is_vector()) {
                const Eigen::MatrixXd* m = idx->as_matrix();
                for (int j = 0; j < m->rows(); ++j) {
                    double val = (*m)(j, 0);
                    if (!validate_index(val)) return;
                    int i = static_cast<int>(std::round(val)) - machine->io;
                    if (i < 0 || i >= shape[d]) {
                        machine->throw_error("INDEX ERROR: index out of bounds", this, 3, 0);
                        return;
                    }
                    axis_indices[d].push_back(i);
                }
            } else {
                machine->throw_error("DOMAIN ERROR: index must be numeric", this, 11, 0);
                return;
            }
        }

        // Compute strides
        std::vector<int> strides(rank);
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // Create modified copy
        Eigen::VectorXd new_data = *nd->data;

        // Compute number of positions to assign
        int num_positions = 1;
        for (int d = 0; d < rank; ++d) {
            num_positions *= static_cast<int>(axis_indices[d].size());
        }

        if (value_val->is_scalar()) {
            // Scalar extends to all selected positions
            double v = value_val->as_scalar();
            std::vector<int> counters(rank, 0);
            for (int pos = 0; pos < num_positions; ++pos) {
                // Compute linear index from counters
                int linear = 0;
                for (int d = 0; d < rank; ++d) {
                    linear += axis_indices[d][counters[d]] * strides[d];
                }
                new_data(linear) = v;

                // Increment counters
                for (int d = rank - 1; d >= 0; --d) {
                    counters[d]++;
                    if (counters[d] < static_cast<int>(axis_indices[d].size())) break;
                    counters[d] = 0;
                }
            }
        } else if (value_val->is_vector() || value_val->is_matrix() || value_val->is_ndarray()) {
            // Value must match selection shape
            int val_size = value_val->size();
            if (val_size != num_positions) {
                machine->throw_error("LENGTH ERROR: value shape doesn't match index", this, 5, 0);
                return;
            }

            // Get value data
            const double* val_data;
            if (value_val->is_ndarray()) {
                val_data = value_val->as_ndarray()->data->data();
            } else {
                val_data = value_val->as_matrix()->data();
            }

            std::vector<int> counters(rank, 0);
            for (int pos = 0; pos < num_positions; ++pos) {
                int linear = 0;
                for (int d = 0; d < rank; ++d) {
                    linear += axis_indices[d][counters[d]] * strides[d];
                }
                new_data(linear) = val_data[pos];

                for (int d = rank - 1; d >= 0; --d) {
                    counters[d]++;
                    if (counters[d] < static_cast<int>(axis_indices[d].size())) break;
                    counters[d] = 0;
                }
            }
        } else {
            machine->throw_error("DOMAIN ERROR: value must be scalar or array", this, 11, 0);
            return;
        }

        Value* result = machine->heap->allocate_ndarray(new_data, shape);
        machine->env->define(var_name, result);
        machine->result = value_val;
        return;
    }

    // Strand indexed assignment (nested arrays)
    if (arr->is_strand()) {
        if (!index_val->is_scalar()) {
            machine->throw_error("RANK ERROR: strand requires scalar index", this, 4, 0);
            return;
        }
        double val = index_val->as_scalar();
        if (val != std::floor(val)) {
            machine->throw_error("DOMAIN ERROR: index must be integer", this, 11, 0);
            return;
        }
        int idx = static_cast<int>(val) - machine->io;
        const std::vector<Value*>* strand = arr->as_strand();
        int len = static_cast<int>(strand->size());
        if (idx < 0 || idx >= len) {
            machine->throw_error("INDEX ERROR: index out of bounds", this, 3, 0);
            return;
        }

        // Create modified copy
        std::vector<Value*> new_strand(*strand);
        new_strand[idx] = value_val;
        Value* result = machine->heap->allocate_strand(std::move(new_strand));
        machine->env->define(var_name, result);
        machine->result = value_val;
        return;
    }

    // Numeric array indexed assignment
    if (!arr->is_array()) {
        machine->throw_error("INDEX ERROR: cannot index non-array value", this, 3, 0);
        return;
    }

    const Eigen::MatrixXd* mat = arr->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();
    int size = static_cast<int>(mat->size());

    // Helper to validate index is near-integer
    auto validate_index = [machine, this](double val) -> bool {
        double rounded = std::round(val);
        if (std::abs(val - rounded) > 1e-10) {
            machine->throw_error("DOMAIN ERROR: index must be integer", this, 11, 0);
            return false;
        }
        return true;
    };

    // Helper to check if index is elided (empty vector = select all)
    auto is_elided = [](Value* idx) -> bool {
        return idx->is_vector() && idx->size() == 0;
    };

    // Multi-axis indexed assignment: M[I;J]←V
    if (index_val->is_strand()) {
        auto* idx_strand = index_val->as_strand();
        if (idx_strand->size() != 2) {
            machine->throw_error("RANK ERROR: matrix requires exactly 2 indices", this, 4, 0);
            return;
        }

        Value* row_idx = (*idx_strand)[0];
        Value* col_idx = (*idx_strand)[1];

        // Get row indices
        std::vector<int> row_indices;
        bool had_error = false;
        if (is_elided(row_idx)) {
            for (int i = 0; i < rows; i++) row_indices.push_back(i);
        } else if (row_idx->is_scalar()) {
            double val = row_idx->as_scalar();
            if (!validate_index(val)) return;
            int i = static_cast<int>(std::round(val)) - machine->io;
            if (i < 0 || i >= rows) {
                machine->throw_error("INDEX ERROR: row index out of bounds", this, 3, 0);
                return;
            }
            row_indices.push_back(i);
        } else if (row_idx->is_array()) {
            const Eigen::MatrixXd* m = row_idx->as_matrix();
            for (int j = 0; j < m->rows(); j++) {
                double val = (*m)(j, 0);
                if (!validate_index(val)) return;
                int i = static_cast<int>(std::round(val)) - machine->io;
                if (i < 0 || i >= rows) {
                    machine->throw_error("INDEX ERROR: row index out of bounds", this, 3, 0);
                    return;
                }
                row_indices.push_back(i);
            }
        } else {
            machine->throw_error("DOMAIN ERROR: row index must be numeric", this, 11, 0);
            return;
        }

        // Get column indices
        std::vector<int> col_indices;
        if (is_elided(col_idx)) {
            for (int i = 0; i < cols; i++) col_indices.push_back(i);
        } else if (col_idx->is_scalar()) {
            double val = col_idx->as_scalar();
            if (!validate_index(val)) return;
            int i = static_cast<int>(std::round(val)) - machine->io;
            if (i < 0 || i >= cols) {
                machine->throw_error("INDEX ERROR: column index out of bounds", this, 3, 0);
                return;
            }
            col_indices.push_back(i);
        } else if (col_idx->is_array()) {
            const Eigen::MatrixXd* m = col_idx->as_matrix();
            for (int j = 0; j < m->rows(); j++) {
                double val = (*m)(j, 0);
                if (!validate_index(val)) return;
                int i = static_cast<int>(std::round(val)) - machine->io;
                if (i < 0 || i >= cols) {
                    machine->throw_error("INDEX ERROR: column index out of bounds", this, 3, 0);
                    return;
                }
                col_indices.push_back(i);
            }
        } else {
            machine->throw_error("DOMAIN ERROR: column index must be numeric", this, 11, 0);
            return;
        }

        int result_rows = static_cast<int>(row_indices.size());
        int result_cols = static_cast<int>(col_indices.size());

        // Create modified copy
        Eigen::MatrixXd new_mat = *mat;

        // Assign value(s)
        if (value_val->is_scalar()) {
            // Scalar extends to all selected positions
            double v = value_val->as_scalar();
            for (int r : row_indices) {
                for (int c : col_indices) {
                    new_mat(r, c) = v;
                }
            }
        } else if (result_rows == 1 && result_cols == 1) {
            // Single position, scalar value required
            machine->throw_error("LENGTH ERROR: scalar index requires scalar value", this, 5, 0);
            return;
        } else if (value_val->is_array()) {
            const Eigen::MatrixXd* val_mat = value_val->as_matrix();
            // Check shape compatibility
            if (result_rows == 1) {
                // Single row: value should be vector of length result_cols
                if (val_mat->size() != result_cols) {
                    machine->throw_error("LENGTH ERROR: value shape doesn't match index", this, 5, 0);
                    return;
                }
                for (int c = 0; c < result_cols; c++) {
                    new_mat(row_indices[0], col_indices[c]) = (*val_mat)(c, 0);
                }
            } else if (result_cols == 1) {
                // Single column: value should be vector of length result_rows
                if (val_mat->size() != result_rows) {
                    machine->throw_error("LENGTH ERROR: value shape doesn't match index", this, 5, 0);
                    return;
                }
                for (int r = 0; r < result_rows; r++) {
                    new_mat(row_indices[r], col_indices[0]) = (*val_mat)(r, 0);
                }
            } else {
                // Matrix: value should be result_rows × result_cols
                if (val_mat->rows() != result_rows || val_mat->cols() != result_cols) {
                    machine->throw_error("LENGTH ERROR: value shape doesn't match index", this, 5, 0);
                    return;
                }
                for (int r = 0; r < result_rows; r++) {
                    for (int c = 0; c < result_cols; c++) {
                        new_mat(row_indices[r], col_indices[c]) = (*val_mat)(r, c);
                    }
                }
            }
        } else {
            machine->throw_error("DOMAIN ERROR: value must be scalar or array", this, 11, 0);
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
        return;
    }

    // Create modified copy for single-axis indexing
    Eigen::MatrixXd new_mat = *mat;

    if (index_val->is_scalar()) {
        // Single index assignment
        int idx = static_cast<int>(index_val->as_scalar()) - machine->io;  // ⎕IO

        if (idx < 0 || idx >= size) {
            machine->throw_error("INDEX ERROR: index out of bounds", this, 3, 0);
            return;
        }

        if (!value_val->is_scalar()) {
            machine->throw_error("LENGTH ERROR: scalar index requires scalar value", this, 5, 0);
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
                machine->throw_error("RANK ERROR: value must be scalar or vector for vector index", this, 4, 0);
                return;
            }
            val_mat = value_val->as_matrix();
            if (static_cast<int>(val_mat->size()) != num_indices) {
                machine->throw_error("LENGTH ERROR: index and value lengths must match", this, 5, 0);
                return;
            }
        }

        // Assign each index
        for (int i = 0; i < num_indices; i++) {
            int idx = static_cast<int>((*idx_mat)(i, 0)) - machine->io;  // ⎕IO
            if (idx < 0 || idx >= size) {
                machine->throw_error("INDEX ERROR: index out of bounds", this, 3, 0);
                return;
            }

            double new_val = scalar_value ? value_val->as_scalar() : (*val_mat)(i, 0);

            // Use row-major linear indexing
            int row = idx / new_mat.cols();
            int col = idx % new_mat.cols();
            new_mat(row, col) = new_val;
        }
    } else {
        machine->throw_error("RANK ERROR: index must be scalar or vector", this, 4, 0);
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
    heap->mark(var_name);
    heap->mark(value_val);
    heap->mark(index_val);
}

// ============================================================================
// IndexListK - Multi-axis index evaluation
// ============================================================================

void IndexListK::invoke(Machine* machine) {
    // Start evaluating the first index expression
    // Push collector to gather result, then evaluate first index
    std::vector<Value*> empty_results;
    empty_results.reserve(indices.size());
    machine->push_kont(machine->heap->allocate<IndexListCollectK>(
        indices, 1, std::move(empty_results)));
    machine->push_kont(indices[0]);
}

void IndexListK::mark(Heap* heap) {
    for (Continuation* idx : indices) {
        heap->mark(idx);
    }
}

void IndexListCollectK::invoke(Machine* machine) {
    // Collect the result from the previous index evaluation
    results.push_back(machine->result);

    if (current >= indices.size()) {
        // All indices evaluated - create strand from results
        machine->result = machine->heap->allocate_strand(std::move(results));
        return;
    }

    // More indices to evaluate
    machine->push_kont(machine->heap->allocate<IndexListCollectK>(
        indices, current + 1, std::move(results)));
    machine->push_kont(indices[current]);
}

void IndexListCollectK::mark(Heap* heap) {
    for (Continuation* idx : indices) {
        heap->mark(idx);
    }
    for (Value* v : results) {
        heap->mark(v);
    }
}

} // namespace apl
