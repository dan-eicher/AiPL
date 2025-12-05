// Continuation - Abstract base class for CEK machine continuations

#pragma once

#include <vector>
#include <string>
#include "value.h"

namespace apl {

// Forward declarations
class Machine;
class APLHeap;
class Environment;
struct APLCompletion;  // Forward declaration for completion records

// Abstract Continuation base class
// Represents "what to do next" in the CEK machine
class Continuation : public GCObject {
private:
    // Only APLHeap can allocate Continuation objects
    friend class APLHeap;

    // Private new operator enforces heap-only allocation
    void* operator new(size_t size) { return ::operator new(size); }

protected:
    // Protected delete allows derived class destructors to work
    void operator delete(void* ptr) { ::operator delete(ptr); }

    Continuation() : GCObject() {}
    virtual ~Continuation() {}

public:
    // Mark all Values and Continuations referenced by this continuation for GC
    virtual void mark(APLHeap* heap) = 0;

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
    void mark(APLHeap* heap) override;

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
    APLCompletion* completion;  // The completion to propagate

    PropagateCompletionK(APLCompletion* comp) : completion(comp) {}

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// CatchReturnK - Catches RETURN completions at function boundaries
// Pushed by FrameK to establish function call boundaries
class CatchReturnK : public Continuation {
public:
    const char* function_name;  // For debugging (not GC-managed, assumed static)

    CatchReturnK(const char* name = nullptr) : function_name(name) {}

    void mark(APLHeap* heap) override;
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

    void mark(APLHeap* heap) override;
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

    void mark(APLHeap* heap) override;
    bool is_loop_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// CatchErrorK - Catches THROW completions for error handling (Phase 5)
// Can be pushed at any point to establish an error boundary
class CatchErrorK : public Continuation {
public:
    CatchErrorK() {}

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ThrowErrorK - Creates and propagates a THROW completion (Phase 5.2)
// Used by primitives and other code to signal errors through completions
class ThrowErrorK : public Continuation {
public:
    const char* error_message;  // Error message (not GC-managed, assumed static or pooled)

    ThrowErrorK(const char* msg) : error_message(msg) {}

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for ReturnK - creates RETURN completion after value is evaluated
class CreateReturnK : public Continuation {
public:
    CreateReturnK() {}

    ~CreateReturnK() override {}

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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
class DerivedOperatorK : public Continuation {
public:
    Continuation* operand_cont;   // The fb-term to evaluate
    const char* op_name;          // The dyadic operator name

    DerivedOperatorK(Continuation* operand, const char* operator_name)
        : operand_cont(operand), op_name(operator_name) {}

    ~DerivedOperatorK() override {}

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// ApplyDerivedOperatorK - Apply dyadic operator to its first operand
// Creates a DERIVED_OPERATOR value
class ApplyDerivedOperatorK : public Continuation {
public:
    const char* op_name;

    ApplyDerivedOperatorK(const char* operator_name) : op_name(operator_name) {}

    ~ApplyDerivedOperatorK() override {}

    void mark(APLHeap* heap) override;

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

    // For OUTER mode: dimensions for Cartesian product
    int lhs_total;          // Total elements in lhs (for OUTER)
    int rhs_total;          // Total elements in rhs (for OUTER)
    int lhs_cols;           // Columns in lhs (for extracting elements)
    int rhs_cols;           // Columns in rhs (for extracting elements)

    CellIterK(Value* f, Value* l, Value* r, int lk, int rk, int total,
              CellIterMode m, int rows, int cols, bool is_vec)
        : fn(f), lhs(l), rhs(r), left_rank(lk), right_rank(rk),
          total_cells(total), current_cell(0), mode(m), accumulator(nullptr),
          orig_rows(rows), orig_cols(cols), orig_is_vector(is_vec),
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
          accumulator(nullptr), orig_rows(l_total), orig_cols(r_total), orig_is_vector(false),
          lhs_total(l_total), rhs_total(r_total), lhs_cols(l_cols), rhs_cols(r_cols) {
        results.reserve(total_cells);
    }

    ~CellIterK() override {}

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// CellCollectK - Collects result and continues iteration
class CellCollectK : public Continuation {
public:
    CellIterK* iter;  // Parent iterator (owned by continuation stack)

    explicit CellCollectK(CellIterK* i) : iter(i) {}

    ~CellCollectK() override {}

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// RowReduceCollectK - Collects row reduction result and continues
class RowReduceCollectK : public Continuation {
public:
    RowReduceK* iter;

    explicit RowReduceCollectK(RowReduceK* i) : iter(i) {}

    ~RowReduceCollectK() override {}

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// PrefixScanCollectK - Collects prefix reduction result and continues
class PrefixScanCollectK : public Continuation {
public:
    PrefixScanK* iter;

    explicit PrefixScanCollectK(PrefixScanK* i) : iter(i) {}

    ~PrefixScanCollectK() override {}

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// RowScanCollectK - Collects row scan result and continues
class RowScanCollectK : public Continuation {
public:
    RowScanK* iter;

    explicit RowScanCollectK(RowScanK* i) : iter(i) {}

    ~RowScanCollectK() override {}

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

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

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

// InnerProductCollectK - Collects inner product cell result
class InnerProductCollectK : public Continuation {
public:
    InnerProductIterK* iter;

    explicit InnerProductCollectK(InnerProductIterK* i) : iter(i) {}

    ~InnerProductCollectK() override {}

    void mark(APLHeap* heap) override;

protected:
    void invoke(Machine* machine) override;
};

} // namespace apl
