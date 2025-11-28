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

// Abstract Continuation base class
// Represents "what to do next" in the CEK machine
class Continuation {
public:
    // GC metadata (same pattern as Value)
    bool marked;
    bool in_old_generation;

    Continuation() : marked(false), in_old_generation(false) {}
    virtual ~Continuation() {}

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
    // Returns the result value (or nullptr if execution should continue)
    // PROTECTED: Only Machine should call this via the trampoline
    virtual Value* invoke(Machine* machine) = 0;

    // Grant Machine access to invoke()
    friend class Machine;
};

// HaltK - Terminal continuation that stops execution
class HaltK : public Continuation {
public:
    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
};

// LookupK - Parse-time continuation for variable lookups
// Stores the variable name (owned std::string for safety)
// At runtime, looks up the variable in the environment
class LookupK : public Continuation {
public:
    std::string var_name;       // Variable name (owned copy)

    LookupK(const char* name)
        : var_name(name) {}

    ~LookupK() override {}

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
};

// AssignK - Assignment continuation for variable definition
// Evaluates the expression, then binds the result to a variable name
// Syntax: name ← expression
class AssignK : public Continuation {
public:
    std::string var_name;       // Variable name to assign to (owned copy)
    Continuation* expr;         // Expression to evaluate

    AssignK(const char* name, Continuation* e)
        : var_name(name), expr(e) {}

    ~AssignK() override {
        // Don't delete expr - it's GC-managed
    }

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
};

// Auxiliary continuation for AssignK - performs the actual binding after expression is evaluated
class PerformAssignK : public Continuation {
public:
    std::string var_name;       // Variable name to assign to

    PerformAssignK(const char* name)
        : var_name(name) {}

    ~PerformAssignK() override {}

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
};

// StrandK - Parse-time continuation for array strands
// Stores a vector of continuations representing the strand elements
// At runtime, evaluates each element and combines them into a vector Value*
// Examples: "1 2 3" or "var1 var2 3" or "(1+2) 5 x"
class StrandK : public Continuation {
public:
    std::vector<Continuation*> elements;  // Continuation for each element

    StrandK(const std::vector<Continuation*>& elems)
        : elements(elems) {}

    ~StrandK() override {
        // Don't delete elements - they're GC-managed
    }

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
};

// MonadicK - Monadic function application (e.g., -x, ⍳x)
// Evaluates operand, then applies monadic function
class MonadicK : public Continuation {
public:
    PrimitiveFn* prim_fn;       // The function to apply
    Continuation* operand;      // Operand to evaluate

    MonadicK(PrimitiveFn* fn, Continuation* op)
        : prim_fn(fn), operand(op) {}

    ~MonadicK() override {
        // Don't delete operand - it's GC-managed
    }

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
};

// DyadicK - Dyadic function application (e.g., x+y, x×y)
// Evaluates operands right-to-left, then applies dyadic function
class DyadicK : public Continuation {
public:
    PrimitiveFn* prim_fn;       // The function to apply
    Continuation* left;         // Left operand
    Continuation* right;        // Right operand

    DyadicK(PrimitiveFn* fn, Continuation* l, Continuation* r)
        : prim_fn(fn), left(l), right(r) {}

    ~DyadicK() override {
        // Don't delete operands - they're GC-managed
    }

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
};

// Auxiliary continuation for DyadicK - evaluates left after right is done
class EvalDyadicLeftK : public Continuation {
public:
    PrimitiveFn* prim_fn;
    Continuation* left;
    Value* right_val;           // Saved right value (set at runtime)

    EvalDyadicLeftK(PrimitiveFn* fn, Continuation* l, Value* r)
        : prim_fn(fn), left(l), right_val(r) {}

    ~EvalDyadicLeftK() override {}

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
};

// Auxiliary continuation to apply monadic function after operand evaluated
class ApplyMonadicK : public Continuation {
public:
    PrimitiveFn* prim_fn;

    ApplyMonadicK(PrimitiveFn* fn)
        : prim_fn(fn) {}

    ~ApplyMonadicK() override {}

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
};

// Auxiliary continuation to apply dyadic function after both operands evaluated
class ApplyDyadicK : public Continuation {
public:
    PrimitiveFn* prim_fn;
    Value* right_val;           // Saved right value

    ApplyDyadicK(PrimitiveFn* fn, Value* r)
        : prim_fn(fn), right_val(r) {}

    ~ApplyDyadicK() override {}

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
};

// Auxiliary continuation to apply function after both args and function are evaluated
class DispatchFunctionK : public Continuation {
public:
    Value* fn_val;              // The function value
    Value* left_val;            // Left argument (nullptr for monadic)
    Value* right_val;           // Right argument

    DispatchFunctionK(Value* fn, Value* left, Value* right)
        : fn_val(fn), left_val(left), right_val(right) {}

    ~DispatchFunctionK() override {}

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
};

// ForK - For loop iteration over array elements (Phase 3.3.4)
// Syntax: :For var :In array ... :EndFor
// Marks loop boundary for :Leave support
class ForK : public Continuation {
public:
    std::string var_name;        // Iterator variable name
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
    Value* invoke(Machine* machine) override;
};

// Auxiliary continuation for ForK - iterates over array elements
class ForIterateK : public Continuation {
public:
    std::string var_name;        // Iterator variable name
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
};

// Auxiliary continuation for ReturnK - creates RETURN completion after value is evaluated
class CreateReturnK : public Continuation {
public:
    CreateReturnK() {}

    ~CreateReturnK() override {}

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
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
    Value* invoke(Machine* machine) override;
};

// RestoreEnvK - Restore environment after function call
class RestoreEnvK : public Continuation {
public:
    Environment* saved_env;

    RestoreEnvK(Environment* env) : saved_env(env) {}

    ~RestoreEnvK() override {}

    void mark(APLHeap* heap) override;

protected:
    Value* invoke(Machine* machine) override;
};

} // namespace apl
