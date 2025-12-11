// Continuation - Abstract base class for CEK machine continuations

#pragma once

#include <vector>
#include <string>
#include "value.h"

namespace apl {

// Forward declarations
class Machine;
class Heap;
class Environment;
struct Completion;  // Forward declaration for completion records

// Abstract Continuation base class
// Represents "what to do next" in the CEK machine
class Continuation : public GCObject {
private:
    // Only Heap can allocate Continuation objects
    friend class Heap;

    // Private new operator enforces heap-only allocation
    void* operator new(size_t size) { return ::operator new(size); }

protected:
    // Protected delete allows derived class destructors to work
    void operator delete(void* ptr) { ::operator delete(ptr); }

    Continuation() : GCObject() {}
    virtual ~Continuation() {}

public:
    // Mark all Values and Continuations referenced by this continuation for GC
    virtual void mark(Heap* heap) = 0;

    // Query methods for control flow handling

    // Is this continuation a function boundary?
    // Used by RETURN to find where to stop unwinding
    virtual bool is_function_boundary() const { return false; }

    // Is this continuation a loop boundary?
    // Used by BREAK/CONTINUE to find loop context
    virtual bool is_loop_boundary() const { return false; }

    // Does this continuation match a target label?
    // Used by labeled BREAK/CONTINUE
    virtual bool matches_label(const char* label) const {
        (void)label;  // Unused
        return false;
    }

protected:
    // Execute this continuation
    // Phase 3.1: Now returns void, result goes in machine->ctrl.value
    // PROTECTED: Only Machine should call this via the trampoline
    virtual void invoke(Machine* machine) = 0;

    // Grant Machine access to invoke()
    friend class Machine;
};

// HaltK - Terminal continuation that stops execution
class HaltK : public Continuation {
public:
    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Completion handler continuations - Phase 2
// These continuations handle APL completion records (RETURN, BREAK, CONTINUE, THROW)
// in a purely functional way, replacing the imperative handle_completion() approach

// PropagateCompletionK - Default handler that propagates completions up the stack
// This is pushed automatically when an abrupt completion occurs
class PropagateCompletionK : public Continuation {
public:
    Completion* completion;  // The completion to propagate

    PropagateCompletionK(Completion* comp) : completion(comp) {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// CatchReturnK - Catches RETURN completions at function boundaries
// Pushed by FrameK to establish function call boundaries
class CatchReturnK : public Continuation {
public:
    const char* function_name;  // For debugging (not GC-managed, assumed static)

    CatchReturnK(const char* name = nullptr) : function_name(name) {}

    void mark(Heap* heap) override;
    bool is_function_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// CatchBreakK - Catches BREAK completions at loop boundaries
// Pushed by WhileK and ForK to establish loop boundaries for :Leave
class CatchBreakK : public Continuation {
public:
    const char* label;  // Optional label for labeled breaks (not GC-managed, assumed static)

    CatchBreakK(const char* lbl = nullptr) : label(lbl) {}

    void mark(Heap* heap) override;
    bool is_loop_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// CatchContinueK - Catches CONTINUE completions at loop boundaries
// Pushed by WhileK and ForK to handle loop continuation
class CatchContinueK : public Continuation {
public:
    Continuation* loop_cont;  // The loop to re-execute (GC-managed)

    CatchContinueK(Continuation* loop) : loop_cont(loop) {}

    void mark(Heap* heap) override;
    bool is_loop_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// CatchErrorK - Catches THROW completions for error handling (Phase 5)
// Can be pushed at any point to establish an error boundary
class CatchErrorK : public Continuation {
public:
    CatchErrorK() {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ThrowErrorK - Creates and propagates a THROW completion (Phase 5.2)
// Used by primitives and other code to signal errors through completions
class ThrowErrorK : public Continuation {
public:
    const char* error_message;  // Error message (not GC-managed, assumed static or pooled)

    ThrowErrorK(const char* msg) : error_message(msg) {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// LiteralK - Parse-time continuation for literal values
// Stores a double directly (not a Value*) for GC safety during parsing
// At runtime, this gets converted to a Value* by the Machine
class LiteralK : public Continuation {
public:
    double literal_value;       // The literal number

    LiteralK(double val)
        : literal_value(val) {}

    ~LiteralK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ClosureLiteralK - Parse-time continuation for closure literals (dfns)
// Stores a Continuation* (the function body) directly
// At runtime, this gets converted to a CLOSURE Value* by the Machine
class ClosureLiteralK : public Continuation {
public:
    Continuation* body;         // The function body continuation graph

    ClosureLiteralK(Continuation* b)
        : body(b) {}

    ~ClosureLiteralK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// LookupK - Parse-time continuation for variable lookups
// Stores the variable name (interned pointer from StringPool)
// At runtime, looks up the variable in the environment
class LookupK : public Continuation {
public:
    const char* var_name;       // Variable name (interned pointer)

    LookupK(const char* name)
        : var_name(name) {}

    ~LookupK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// AssignK - Assignment continuation for variable definition
// Evaluates the expression, then binds the result to a variable name
// Syntax: name ← expression
class AssignK : public Continuation {
public:
    const char* var_name;       // Variable name to assign to (interned pointer)
    Continuation* expr;         // Expression to evaluate

    AssignK(const char* name, Continuation* e)
        : var_name(name), expr(e) {}

    ~AssignK() override {
        // Don't delete expr - it's GC-managed
    }

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for AssignK - performs the actual binding after expression is evaluated
class PerformAssignK : public Continuation {
public:
    const char* var_name;       // Variable name to assign to (interned pointer)

    PerformAssignK(const char* name)
        : var_name(name) {}

    ~PerformAssignK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// StrandK - Lexical strand continuation for numeric vector literals (ISO 13751)
// Stores a pre-computed vector Value* from the lexer
// At runtime, just returns this Value
// Example: "1 2 3" → StrandK(vector_value)
class StrandK : public Continuation {
public:
    Value* vector_value;  // Pre-allocated vector Value

    StrandK(Value* val)
        : vector_value(val) {}

    ~StrandK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// JuxtaposeK - G2 Grammar juxtaposition: fbn-term ::= fb-term fbn-term
// Implements: if type(x₁) = bas then x₂(x₁) else x₁(x₂)
// This is different from StrandK - it performs function application based on types
class JuxtaposeK : public Continuation {
public:
    Continuation* left;   // Left fb-term
    Continuation* right;  // Right fbn-term

    JuxtaposeK(Continuation* l, Continuation* r)
        : left(l), right(r) {}

    ~JuxtaposeK() override {
        // Don't delete - GC-managed
    }

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for JuxtaposeK - evaluates left after right is done
class EvalJuxtaposeLeftK : public Continuation {
public:
    Continuation* left;  // Left continuation to evaluate
    Value* right_val;    // Right value (will be set when right completes)

    EvalJuxtaposeLeftK(Continuation* l, Value* r)
        : left(l), right_val(r) {}

    ~EvalJuxtaposeLeftK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// PerformJuxtaposeK - applies G2 juxtaposition rule after both sides are evaluated
class PerformJuxtaposeK : public Continuation {
public:
    Value* right_val;  // Right value (evaluated first)

    PerformJuxtaposeK(Value* right)
        : right_val(right) {}

    ~PerformJuxtaposeK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// FinalizeK - Forces g' finalization of parenthesized expressions
// When a parenthesized expression evaluates to a curry, finalize it to a value
// This ensures (f/x) becomes a value before being used in strand context
class FinalizeK : public Continuation {
public:
    Continuation* inner;  // The expression to evaluate and finalize

    FinalizeK(Continuation* expr) : inner(expr) {}

    ~FinalizeK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// PerformFinalizeK - Auxiliary that checks result after inner evaluates
class PerformFinalizeK : public Continuation {
public:
    PerformFinalizeK() {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// MonadicK - Monadic function application (e.g., -x, ⍳x)
// Evaluates operand, then applies monadic function
class MonadicK : public Continuation {
public:
    const char* op_name;        // Operator name (interned pointer, e.g., "+", "-", "⍳")
    Continuation* operand;      // Operand to evaluate

    MonadicK(const char* name, Continuation* op)
        : op_name(name), operand(op) {}

    ~MonadicK() override {
        // Don't delete operand - it's GC-managed
    }

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// DyadicK - Dyadic function application (e.g., x+y, x×y)
// Evaluates operands right-to-left, then applies dyadic function
class DyadicK : public Continuation {
public:
    const char* op_name;        // Operator name (interned pointer, e.g., "+", "-", "×")
    Continuation* left;         // Left operand
    Continuation* right;        // Right operand

    DyadicK(const char* name, Continuation* l, Continuation* r)
        : op_name(name), left(l), right(r) {}

    ~DyadicK() override {
        // Don't delete operands - they're GC-managed
    }

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for DyadicK - evaluates left after right is done
class EvalDyadicLeftK : public Continuation {
public:
    const char* op_name;        // Operator name (interned pointer)
    Continuation* left;
    Value* right_val;           // Saved right value (set at runtime)

    EvalDyadicLeftK(const char* name, Continuation* l, Value* r)
        : op_name(name), left(l), right_val(r) {}

    ~EvalDyadicLeftK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation to apply monadic function after operand evaluated
class ApplyMonadicK : public Continuation {
public:
    const char* op_name;        // Operator name (interned pointer)

    ApplyMonadicK(const char* name)
        : op_name(name) {}

    ~ApplyMonadicK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation to apply dyadic function after both operands evaluated
class ApplyDyadicK : public Continuation {
public:
    const char* op_name;        // Operator name (interned pointer)
    Value* right_val;           // Saved right value

    ApplyDyadicK(const char* name, Value* r)
        : op_name(name), right_val(r) {}

    ~ApplyDyadicK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};


// ArgK - Continuation for function arguments
// Saves an argument value and continues with next continuation
class ArgK : public Continuation {
public:
    Value* arg_value;           // The argument value
    Continuation* next;         // Next continuation

    ArgK(Value* arg, Continuation* k)
        : arg_value(arg), next(k) {}

    ~ArgK() override {
        // Don't delete next - it's GC-managed
    }

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for StrandK - evaluates remaining strand elements right-to-left
// Maintains an accumulator of already-evaluated values (in left-to-right order)
// Evaluates next element from the right, prepends to accumulator, repeats
class EvalStrandElementK : public Continuation {
public:
    std::vector<Continuation*> remaining_elements;  // Elements left to evaluate (left-to-right order)
    std::vector<Value*> evaluated_values;           // Values evaluated so far (left-to-right order)

    EvalStrandElementK(const std::vector<Continuation*>& remaining, const std::vector<Value*>& evaluated)
        : remaining_elements(remaining), evaluated_values(evaluated) {}

    ~EvalStrandElementK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation to build the final strand vector from collected values
// All elements have been evaluated, now construct the vector Value
class BuildStrandK : public Continuation {
public:
    std::vector<Value*> values;  // All evaluated strand elements (left-to-right order)

    BuildStrandK(const std::vector<Value*>& vals)
        : values(vals) {}

    ~BuildStrandK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// FrameK - Stack frame continuation for function calls
// Marks function boundaries and saves return continuation
class FrameK : public Continuation {
public:
    const char* function_name;  // Name of function (for debugging)
    Continuation* return_k;     // Where to return to

    FrameK(const char* name, Continuation* ret)
        : function_name(name), return_k(ret) {}

    ~FrameK() override {
        // Don't delete return_k - it's GC-managed
    }

    void mark(Heap* heap) override;

    // FrameK marks function boundaries
    bool is_function_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// ApplyFunctionK - Apply a function value (from a variable) to arguments
// This implements the currying transformation: determines at runtime whether to use monadic or dyadic form
// Parser creates this when it sees a function reference (variable) being applied but doesn't know which form
//
// Structure for "f x":     ApplyFunctionK(LookupK("f"), nullptr, LiteralK(x))  → monadic
// Structure for "x f y":   ApplyFunctionK(LookupK("f"), LookupK("x"), LiteralK(y)) → dyadic
//
// At eval time, evaluates the function reference and arguments, then dispatches based on what's available
class ApplyFunctionK : public Continuation {
public:
    Continuation* fn_cont;      // Continuation to get the function value (e.g., LookupK)
    Continuation* left_arg;     // Left argument (nullptr for monadic)
    Continuation* right_arg;    // Right argument (always present)

    ApplyFunctionK(Continuation* fn, Continuation* left, Continuation* right)
        : fn_cont(fn), left_arg(left), right_arg(right) {}

    ~ApplyFunctionK() override {
        // Don't delete - all are GC-managed
    }

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for ApplyFunctionK - evaluates left arg after right is done (dyadic case)
class EvalApplyFunctionLeftK : public Continuation {
public:
    Continuation* fn_cont;
    Continuation* left_arg;
    Value* right_val;           // Saved right value (set at runtime)

    EvalApplyFunctionLeftK(Continuation* fn, Continuation* left, Value* right)
        : fn_cont(fn), left_arg(left), right_val(right) {}

    ~EvalApplyFunctionLeftK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for ApplyFunctionK - evaluates function after arg is done (monadic case)
class EvalApplyFunctionMonadicK : public Continuation {
public:
    Continuation* fn_cont;
    Value* arg_val;             // Saved argument value (set at runtime)

    EvalApplyFunctionMonadicK(Continuation* fn, Value* arg)
        : fn_cont(fn), arg_val(arg) {}

    ~EvalApplyFunctionMonadicK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for ApplyFunctionK - evaluates function after both args are done (dyadic case)
class EvalApplyFunctionDyadicK : public Continuation {
public:
    Continuation* fn_cont;
    Value* left_val;            // Saved left argument value (set at runtime)
    Value* right_val;           // Saved right argument value

    EvalApplyFunctionDyadicK(Continuation* fn, Value* left, Value* right)
        : fn_cont(fn), left_val(left), right_val(right) {}

    ~EvalApplyFunctionDyadicK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation to apply function after both args and function are evaluated
class DispatchFunctionK : public Continuation {
public:
    Value* fn_val;              // The function value
    Value* left_val;            // Left argument (nullptr for monadic)
    Value* right_val;           // Right argument
    bool force_monadic;         // When true, apply monadic form immediately (skip G_PRIME currying)

    DispatchFunctionK(Value* fn, Value* left, Value* right, bool force_mon = false)
        : fn_val(fn), left_val(left), right_val(right), force_monadic(force_mon) {}

    ~DispatchFunctionK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// DeferredDispatchK - Continues dispatch after a subcomputation completes
// Used when a curried function argument needs to be finalized before dispatch
// Reads machine->result as the new right_val
class DeferredDispatchK : public Continuation {
public:
    Value* fn_val;              // The function to dispatch
    Value* left_val;            // Left argument (nullptr for monadic)
    bool force_monadic;         // Force monadic application

    DeferredDispatchK(Value* fn, Value* left, bool force_mon = false)
        : fn_val(fn), left_val(left), force_monadic(force_mon) {}

    ~DeferredDispatchK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// SeqK - Sequence continuation for executing multiple statements
// Executes statements in order, left-to-right
// The result of the last statement becomes the final result
class SeqK : public Continuation {
public:
    std::vector<Continuation*> statements;  // Statements to execute in order

    SeqK(const std::vector<Continuation*>& stmts)
        : statements(stmts) {}

    ~SeqK() override {
        // Don't delete statements - they're GC-managed
    }

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for SeqK - executes remaining statements
// Maintains index of next statement to execute
class ExecNextStatementK : public Continuation {
public:
    std::vector<Continuation*> statements;  // All statements
    size_t next_index;                      // Index of next statement to execute

    ExecNextStatementK(const std::vector<Continuation*>& stmts, size_t idx)
        : statements(stmts), next_index(idx) {}

    ~ExecNextStatementK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// Control Flow Continuations (Phase 3.3.2)
// ============================================================================

// IfK - Conditional execution
// Evaluates condition, then executes either then_branch or else_branch
// Syntax: :If condition ... :Else ... :EndIf
class IfK : public Continuation {
public:
    Continuation* condition;     // Condition to evaluate
    Continuation* then_branch;   // Execute if condition is true (non-zero)
    Continuation* else_branch;   // Execute if condition is false (zero) - can be nullptr

    IfK(Continuation* cond, Continuation* then_b, Continuation* else_b)
        : condition(cond), then_branch(then_b), else_branch(else_b) {}

    ~IfK() override {
        // Don't delete - all are GC-managed
    }

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for IfK - selects branch after condition is evaluated
class SelectBranchK : public Continuation {
public:
    Continuation* then_branch;   // Execute if condition was true
    Continuation* else_branch;   // Execute if condition was false (can be nullptr)

    SelectBranchK(Continuation* then_b, Continuation* else_b)
        : then_branch(then_b), else_branch(else_b) {}

    ~SelectBranchK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// WhileK - Loop execution (Phase 3.3.3)
// Syntax: :While condition ... :EndWhile
// Marks loop boundary for :Leave support
class WhileK : public Continuation {
public:
    Continuation* condition;     // Condition to evaluate before each iteration
    Continuation* body;          // Loop body to execute

    WhileK(Continuation* cond, Continuation* loop_body)
        : condition(cond), body(loop_body) {}

    ~WhileK() override {
        // Don't delete - all are GC-managed
    }

    void mark(Heap* heap) override;

    // WhileK marks loop boundaries for :Leave
    bool is_loop_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for WhileK - checks condition before iteration
class CheckWhileCondK : public Continuation {
public:
    Continuation* condition;     // Condition to re-evaluate
    Continuation* body;          // Body to execute if true

    CheckWhileCondK(Continuation* cond, Continuation* loop_body)
        : condition(cond), body(loop_body) {}

    ~CheckWhileCondK() override {}

    void mark(Heap* heap) override;

    // CheckWhileCondK also marks loop boundaries for :Leave
    bool is_loop_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// ForK - For loop iteration over array elements (Phase 3.3.4)
// Syntax: :For var :In array ... :EndFor
// Marks loop boundary for :Leave support
class ForK : public Continuation {
public:
    const char* var_name;        // Iterator variable name (interned pointer)
    Continuation* array_expr;    // Expression that produces the array
    Continuation* body;          // Loop body to execute

    ForK(const char* var, Continuation* arr, Continuation* loop_body)
        : var_name(var), array_expr(arr), body(loop_body) {}

    ~ForK() override {
        // Don't delete - all are GC-managed
    }

    void mark(Heap* heap) override;

    // ForK marks loop boundaries for :Leave
    bool is_loop_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for ForK - iterates over array elements
class ForIterateK : public Continuation {
public:
    const char* var_name;        // Iterator variable name (interned pointer)
    Value* array;                // Array to iterate over
    Continuation* body;          // Loop body
    size_t index;                // Current iteration index

    ForIterateK(const char* var, Value* arr, Continuation* loop_body, size_t idx)
        : var_name(var), array(arr), body(loop_body), index(idx) {}

    ~ForIterateK() override {}

    void mark(Heap* heap) override;

    // ForIterateK also marks loop boundaries for :Leave
    bool is_loop_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// LeaveK - Exit from loop (Phase 3.3.5)
// Syntax: :Leave
// Creates BREAK completion record to unwind to loop boundary
class LeaveK : public Continuation {
public:
    LeaveK() {}

    ~LeaveK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ReturnK - Return from function (Phase 3.3.5)
// Syntax: :Return [value]
// Creates RETURN completion record to unwind to function boundary
class ReturnK : public Continuation {
public:
    Continuation* value_expr;    // Optional value to return (nullptr for unit)

    ReturnK(Continuation* val = nullptr)
        : value_expr(val) {}

    ~ReturnK() override {
        // Don't delete value_expr - it's GC-managed
    }

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for ReturnK - creates RETURN completion after value is evaluated
class CreateReturnK : public Continuation {
public:
    CreateReturnK() {}

    ~CreateReturnK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// Function Call Continuations (Phase 4.3)
// ============================================================================

// FunctionCallK - Apply a function (CLOSURE) to arguments
// Marks function boundary for :Return support
class FunctionCallK : public Continuation {
public:
    Value* fn_value;             // CLOSURE value to call
    Value* left_arg;             // Left argument (⍺), nullptr for monadic
    Value* right_arg;            // Right argument (⍵)

    FunctionCallK(Value* fn, Value* left, Value* right)
        : fn_value(fn), left_arg(left), right_arg(right) {}

    ~FunctionCallK() override {}

    void mark(Heap* heap) override;

    // FunctionCallK marks function boundaries for :Return
    bool is_function_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// RestoreEnvK - Restore environment after function call
class RestoreEnvK : public Continuation {
public:
    Environment* saved_env;

    RestoreEnvK(Environment* env) : saved_env(env) {}

    ~RestoreEnvK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// G2 Grammar Continuations (Operator Support)
// ============================================================================

// DerivedOperatorK - Represents a partially applied dyadic operator (G2 grammar)
// Grammar: derived-operator ::= fb-term dyadic-operator
// Semantics: x₂(x₁) where x₁ is fb-term, x₂ is operator
// Result is a new operator (monadic)
// Optional axis_cont for axis specification: f/[k] where k is axis
class DerivedOperatorK : public Continuation {
public:
    Continuation* operand_cont;   // The fb-term to evaluate
    Continuation* axis_cont;      // Optional axis expression (for f/[k] syntax)
    const char* op_name;          // The dyadic operator name

    DerivedOperatorK(Continuation* operand, const char* operator_name,
                     Continuation* axis = nullptr)
        : operand_cont(operand), axis_cont(axis), op_name(operator_name) {}

    ~DerivedOperatorK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ApplyDerivedOperatorK - Apply dyadic operator to its first operand
// Creates a DERIVED_OPERATOR value (or OPERATOR_CURRY if axis is specified)
class ApplyDerivedOperatorK : public Continuation {
public:
    const char* op_name;
    Continuation* axis_cont;      // Optional axis (for f/[k] syntax)

    ApplyDerivedOperatorK(const char* operator_name, Continuation* axis = nullptr)
        : op_name(operator_name), axis_cont(axis) {}

    ~ApplyDerivedOperatorK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ApplyAxisK - Apply axis specification to a derived operator
// Creates OPERATOR_CURRY with axis as second operand
class ApplyAxisK : public Continuation {
public:
    Value* derived_op;            // The DERIVED_OPERATOR value

    ApplyAxisK(Value* derived) : derived_op(derived) {}

    ~ApplyAxisK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// CellIterK - General-purpose cell iterator for operators
// ============================================================================
// Handles iteration patterns for: each, reduce, scan, rank
// Dispatches function via DispatchFunctionK (works with any function type)

enum class CellIterMode {
    COLLECT,      // Gather all results (each, rank)
    FOLD_RIGHT,   // Accumulate right-to-left, single result (reduce)
    SCAN_RIGHT,   // Accumulate right-to-left, keep all intermediates (scan)
    OUTER         // Cartesian product iteration (outer product)
};

class CellIterK : public Continuation {
public:
    Value* fn;              // Function to apply
    Value* lhs;             // Left array (nullptr for monadic)
    Value* rhs;             // Right array
    int left_rank;          // Cell rank for left arg (0=scalars, 1=rows, etc.)
    int right_rank;         // Cell rank for right arg
    int total_cells;        // Total cells to process
    int current_cell;       // Current cell index (counts from end for fold/scan)
    CellIterMode mode;      // How to combine results
    std::vector<Value*> results;  // Collected results
    Value* accumulator;     // For FOLD_RIGHT and SCAN_RIGHT modes

    // Original array shape info for reassembly
    int orig_rows;
    int orig_cols;
    bool orig_is_vector;
    bool orig_is_char;     // Preserve character data flag

    // For OUTER mode: dimensions for Cartesian product
    int lhs_total;          // Total elements in lhs (for OUTER)
    int rhs_total;          // Total elements in rhs (for OUTER)
    int lhs_cols;           // Columns in lhs (for extracting elements)
    int rhs_cols;           // Columns in rhs (for extracting elements)

    CellIterK(Value* f, Value* l, Value* r, int lk, int rk, int total,
              CellIterMode m, int rows, int cols, bool is_vec, bool is_char = false)
        : fn(f), lhs(l), rhs(r), left_rank(lk), right_rank(rk),
          total_cells(total), current_cell(0), mode(m), accumulator(nullptr),
          orig_rows(rows), orig_cols(cols), orig_is_vector(is_vec), orig_is_char(is_char),
          lhs_total(0), rhs_total(0), lhs_cols(1), rhs_cols(1) {
        if (mode == CellIterMode::COLLECT || mode == CellIterMode::SCAN_RIGHT || mode == CellIterMode::OUTER) {
            results.reserve(total);
        }
        // For fold/scan modes, start from the last cell
        if (mode == CellIterMode::FOLD_RIGHT || mode == CellIterMode::SCAN_RIGHT) {
            current_cell = total - 1;
        }
    }

    // Constructor for OUTER mode with explicit dimensions
    CellIterK(Value* f, Value* l, Value* r, int l_total, int r_total, int l_cols, int r_cols)
        : fn(f), lhs(l), rhs(r), left_rank(0), right_rank(0),
          total_cells(l_total * r_total), current_cell(0), mode(CellIterMode::OUTER),
          accumulator(nullptr), orig_rows(l_total), orig_cols(r_total), orig_is_vector(false), orig_is_char(false),
          lhs_total(l_total), rhs_total(r_total), lhs_cols(l_cols), rhs_cols(r_cols) {
        results.reserve(total_cells);
    }

    ~CellIterK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// CellCollectK - Collects result and continues iteration
class CellCollectK : public Continuation {
public:
    CellIterK* iter;  // Parent iterator (owned by continuation stack)

    explicit CellCollectK(CellIterK* i) : iter(i) {}

    ~CellCollectK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// RowReduceK - Reduces each row of a matrix independently
// ============================================================================
// For f/ on matrix: reduces each row, returns vector of results

class RowReduceK : public Continuation {
public:
    Value* fn;              // Function to reduce with
    Value* matrix;          // Matrix to reduce
    int current_row;        // Current row being processed
    int total_rows;         // Total rows to process
    int cols;               // Number of columns per row
    std::vector<Value*> results;  // Collected row reduction results
    bool reduce_first_axis; // True for ⌿ (reduce columns), false for / (reduce rows)

    RowReduceK(Value* f, Value* m, int rows, int c, bool first_axis = false)
        : fn(f), matrix(m), current_row(0), total_rows(rows), cols(c),
          reduce_first_axis(first_axis) {
        results.reserve(rows);
    }

    ~RowReduceK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// RowReduceCollectK - Collects row reduction result and continues
class RowReduceCollectK : public Continuation {
public:
    RowReduceK* iter;

    explicit RowReduceCollectK(RowReduceK* i) : iter(i) {}

    ~RowReduceCollectK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// NwiseReduceK - N-wise reduction on vectors
// ============================================================================
// ISO-13751 §9.2.3: N f/ B applies f between successive N-element windows
// Example: 2 +/ 1 2 3 4 5 → (1+2) (2+3) (3+4) (4+5) = 3 5 7 9

class NwiseReduceK : public Continuation {
public:
    Value* fn;              // Function to reduce with
    Value* vec;             // Vector to reduce
    int window_size;        // Size of each window (N)
    bool reverse;           // True if N was negative (reverse windows)
    int current_window;     // Current window being processed
    int total_windows;      // Total number of windows
    std::vector<Value*> results;  // Collected window reduction results

    NwiseReduceK(Value* f, Value* v, int n, bool rev)
        : fn(f), vec(v), window_size(n), reverse(rev), current_window(0) {
        int len = v->as_matrix()->rows();
        total_windows = len - n + 1;
        results.reserve(total_windows);
    }

    ~NwiseReduceK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// NwiseCollectK - Collects N-wise reduction result and continues
class NwiseCollectK : public Continuation {
public:
    NwiseReduceK* iter;

    explicit NwiseCollectK(NwiseReduceK* i) : iter(i) {}

    ~NwiseCollectK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// NwiseMatrixReduceK - N-wise reduction on matrices along an axis
// ============================================================================

class NwiseMatrixReduceK : public Continuation {
public:
    Value* fn;              // Function to reduce with
    Value* matrix;          // Matrix to reduce
    int window_size;        // Size of each window (N)
    bool first_axis;        // True for axis 1, false for axis 2
    bool reverse;           // True if N was negative
    int current_slice;      // Current row/column being processed
    int total_slices;       // Total number of rows/columns
    std::vector<Value*> results;  // Collected slice results (each is a vector)

    NwiseMatrixReduceK(Value* f, Value* m, int n, bool first, bool rev)
        : fn(f), matrix(m), window_size(n), first_axis(first), reverse(rev),
          current_slice(0) {
        const Eigen::MatrixXd* mat = m->as_matrix();
        // We iterate over the non-reduced axis
        total_slices = first_axis ? mat->cols() : mat->rows();
        results.reserve(total_slices);
    }

    ~NwiseMatrixReduceK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// NwiseMatrixCollectK - Collects N-wise matrix reduction result
class NwiseMatrixCollectK : public Continuation {
public:
    NwiseMatrixReduceK* iter;

    explicit NwiseMatrixCollectK(NwiseMatrixReduceK* i) : iter(i) {}

    ~NwiseMatrixCollectK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// PrefixScanK - Computes prefix reductions for scan operator
// ============================================================================
// ISO-13751: Each element I of result is f/B[⍳I] (reduction of first I+1 elements)
// This requires O(n²) work since each prefix must be reduced independently

class PrefixScanK : public Continuation {
public:
    Value* fn;              // Function to reduce with
    Value* vec;             // Vector to scan
    int current_prefix;     // Current prefix length being computed (1 to n)
    int total_len;          // Total vector length
    std::vector<Value*> results;  // Collected prefix reduction results

    PrefixScanK(Value* f, Value* v, int len)
        : fn(f), vec(v), current_prefix(1), total_len(len) {
        results.reserve(len);
    }

    ~PrefixScanK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// PrefixScanCollectK - Collects prefix reduction result and continues
class PrefixScanCollectK : public Continuation {
public:
    PrefixScanK* iter;

    explicit PrefixScanCollectK(PrefixScanK* i) : iter(i) {}

    ~PrefixScanCollectK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// RowScanK - Scans each row of a matrix independently
// ============================================================================
// For f\ on matrix: scans each row, returns matrix of results

class RowScanK : public Continuation {
public:
    Value* fn;              // Function to scan with
    Value* matrix;          // Matrix to scan
    int current_row;        // Current row being processed
    int total_rows;         // Total rows to process
    int cols;               // Number of columns per row
    std::vector<Value*> results;  // Collected row scan results (vectors)
    bool scan_first_axis;   // True for ⍀ (scan columns), false for \ (scan rows)

    RowScanK(Value* f, Value* m, int rows, int c, bool first_axis = false)
        : fn(f), matrix(m), current_row(0), total_rows(rows), cols(c),
          scan_first_axis(first_axis) {
        results.reserve(rows);
    }

    ~RowScanK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// RowScanCollectK - Collects row scan result and continues
class RowScanCollectK : public Continuation {
public:
    RowScanK* iter;

    explicit RowScanCollectK(RowScanK* i) : iter(i) {}

    ~RowScanCollectK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// ReduceResultK - Reduce the vector in ctrl.value with a function
// ============================================================================
// Used for chaining: after computing a vector result, reduce it
// Example: inner product applies g element-wise, then reduces with f

class ReduceResultK : public Continuation {
public:
    Value* fn;  // Function to reduce with

    explicit ReduceResultK(Value* f) : fn(f) {}

    ~ReduceResultK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// InnerProductIterK - Iterates over output cells for matrix inner product
// ============================================================================
// For each output position (i,j), extracts row i from lhs and col j from rhs,
// then computes vector inner product (g element-wise, f reduce)

class InnerProductIterK : public Continuation {
public:
    Value* f_fn;            // Function for reduction
    Value* g_fn;            // Function for element-wise
    Value* lhs;             // Left matrix
    Value* rhs;             // Right matrix
    int lhs_rows;           // Rows in lhs
    int lhs_cols;           // Cols in lhs (= common dimension)
    int rhs_cols;           // Cols in rhs
    int current_i;          // Current output row
    int current_j;          // Current output col
    std::vector<Value*> results;  // Collected results

    InnerProductIterK(Value* f, Value* g, Value* l, Value* r,
                      int lr, int lc, int rc)
        : f_fn(f), g_fn(g), lhs(l), rhs(r),
          lhs_rows(lr), lhs_cols(lc), rhs_cols(rc),
          current_i(0), current_j(0) {
        results.reserve(lr * rc);
    }

    ~InnerProductIterK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// InnerProductCollectK - Collects inner product cell result
class InnerProductCollectK : public Continuation {
public:
    InnerProductIterK* iter;

    explicit InnerProductCollectK(InnerProductIterK* i) : iter(i) {}

    ~InnerProductCollectK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// Indexed Assignment Continuations (ISO 13751 §10.3.3)
// ============================================================================
// A[I]←V - evaluates right-to-left: value first, then index, then assign

// IndexedAssignK - Start indexed assignment: evaluate value first
class IndexedAssignK : public Continuation {
public:
    const char* var_name;         // Variable name (interned string)
    Continuation* index_cont;     // Index expression
    Continuation* value_cont;     // Value expression

    IndexedAssignK(const char* var, Continuation* idx, Continuation* val)
        : var_name(var), index_cont(idx), value_cont(val) {}

    ~IndexedAssignK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// IndexedAssignIndexK - Value evaluated, now evaluate index
class IndexedAssignIndexK : public Continuation {
public:
    const char* var_name;         // Variable name (interned string)
    Value* value_val;             // Already-evaluated value
    Continuation* index_cont;     // Index expression to evaluate

    IndexedAssignIndexK(const char* var, Value* val, Continuation* idx)
        : var_name(var), value_val(val), index_cont(idx) {}

    ~IndexedAssignIndexK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// PerformIndexedAssignK - Value and index evaluated, perform the assignment
class PerformIndexedAssignK : public Continuation {
public:
    const char* var_name;         // Variable name (interned string)
    Value* value_val;             // Value to assign
    Value* index_val;             // Index value(s)

    PerformIndexedAssignK(const char* var, Value* val, Value* idx)
        : var_name(var), value_val(val), index_val(idx) {}

    ~PerformIndexedAssignK() override {}

    void mark(Heap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

} // namespace apl
