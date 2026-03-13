// Continuation - Abstract base class for CEK machine continuations

#pragma once

#include <vector>
#include <string>
#include "value.h"
#include "sysvar.h"

namespace apl {

// Forward declarations
class Machine;
class Heap;
class Environment;
struct Completion;

// Forward declare all continuation types for visitor pattern
class Continuation;
class HaltK;
class PropagateCompletionK;
class CatchReturnK;
class CatchBreakK;
class CatchContinueK;
class CatchErrorK;
class ClearErrorStateK;
class ThrowErrorK;
class LiteralK;
class ClosureLiteralK;
class DefinedOperatorLiteralK;
class LookupK;
class AssignK;
class PerformAssignK;
class SysVarReadK;
class SysVarAssignK;
class PerformSysVarAssignK;
class LiteralStrandK;
class JuxtaposeK;
class EvalJuxtaposeLeftK;
class PerformJuxtaposeK;
class FinalizeK;
class PerformFinalizeK;
class MonadicK;
class DyadicK;
class EvalDyadicLeftK;
class ApplyMonadicK;
class ApplyDyadicK;
class ArgK;
class FrameK;
class ApplyFunctionK;
class EvalApplyFunctionLeftK;
class EvalApplyFunctionMonadicK;
class EvalApplyFunctionDyadicK;
class DispatchFunctionK;
class DeferredDispatchK;
class SeqK;
class ExecNextStatementK;
class IfK;
class SelectBranchK;
class WhileK;
class CheckWhileCondK;
class ForK;
class ForIterateK;
class LeaveK;
class ContinueK;
class ReturnK;
class CreateReturnK;
class BranchK;
class CheckBranchK;
class FunctionCallK;
class RestoreEnvK;
class DerivedOperatorK;
class ApplyDerivedOperatorK;
class ApplyAxisK;
class CellIterK;
class CellCollectK;
class FiberReduceK;
class FiberReduceCollectK;
class PrefixScanK;
class PrefixScanCollectK;
class RowScanK;
class RowScanCollectK;
class ReduceResultK;
class IndexedAssignK;
class IndexedAssignIndexK;
class PerformIndexedAssignK;
class IndexListK;
class IndexListCollectK;
class InvokeDefinedOperatorK;
class ValueK;
class MonadicCallK;
class EvalMonadicCallFnK;
class PerformMonadicCallK;
class DyadicCallK;
class EvalDyadicCallLeftK;
class EvalDyadicCallFnK;
class PerformDyadicCallK;
class EigenReduceK;
class PerformEigenReduceK;
class EigenProductK;
class EvalEigenProductLeftK;
class PerformEigenProductK;
class EigenOuterK;
class EvalEigenOuterLeftK;
class PerformEigenOuterK;
class EigenScanK;
class PerformEigenScanK;
class EigenReduceFirstK;
class PerformEigenReduceFirstK;
class EigenSortK;
class PerformEigenSortK;
class TypeDirectedK;
class ReturnTypeRecordK;

// ============================================================================
// Function Application Helper
// ============================================================================

// Apply a function immediately without creating curries.
// For primitives: calls the monadic or dyadic form directly.
// For closures: pushes FunctionCallK.
// For derived operators: invokes the operator's form.
// Returns true if result is ready in machine->result, false if continuation pushed.
bool apply_function_immediate(Machine* m, Value* fn_val, Value* left_val,
                              Value* right_val, Value* axis = nullptr);

// ============================================================================
// ContinuationVisitor - Visitor pattern for operations on continuations
// ============================================================================
// Enables external operations (printing, optimization, etc.) without
// modifying continuation classes.

class ContinuationVisitor {
public:
    virtual ~ContinuationVisitor() = default;

    virtual void visit(HaltK*) = 0;
    virtual void visit(PropagateCompletionK*) = 0;
    virtual void visit(CatchReturnK*) = 0;
    virtual void visit(CatchBreakK*) = 0;
    virtual void visit(CatchContinueK*) = 0;
    virtual void visit(CatchErrorK*) = 0;
    virtual void visit(ClearErrorStateK*) = 0;
    virtual void visit(ThrowErrorK*) = 0;
    virtual void visit(LiteralK*) = 0;
    virtual void visit(ClosureLiteralK*) = 0;
    virtual void visit(DefinedOperatorLiteralK*) = 0;
    virtual void visit(LookupK*) = 0;
    virtual void visit(AssignK*) = 0;
    virtual void visit(PerformAssignK*) = 0;
    virtual void visit(SysVarReadK*) = 0;
    virtual void visit(SysVarAssignK*) = 0;
    virtual void visit(PerformSysVarAssignK*) = 0;
    virtual void visit(LiteralStrandK*) = 0;
    virtual void visit(JuxtaposeK*) = 0;
    virtual void visit(EvalJuxtaposeLeftK*) = 0;
    virtual void visit(PerformJuxtaposeK*) = 0;
    virtual void visit(FinalizeK*) = 0;
    virtual void visit(PerformFinalizeK*) = 0;
    virtual void visit(MonadicK*) = 0;
    virtual void visit(DyadicK*) = 0;
    virtual void visit(EvalDyadicLeftK*) = 0;
    virtual void visit(ApplyMonadicK*) = 0;
    virtual void visit(ApplyDyadicK*) = 0;
    virtual void visit(ArgK*) = 0;
    virtual void visit(FrameK*) = 0;
    virtual void visit(ApplyFunctionK*) = 0;
    virtual void visit(EvalApplyFunctionLeftK*) = 0;
    virtual void visit(EvalApplyFunctionMonadicK*) = 0;
    virtual void visit(EvalApplyFunctionDyadicK*) = 0;
    virtual void visit(DispatchFunctionK*) = 0;
    virtual void visit(DeferredDispatchK*) = 0;
    virtual void visit(SeqK*) = 0;
    virtual void visit(ExecNextStatementK*) = 0;
    virtual void visit(IfK*) = 0;
    virtual void visit(SelectBranchK*) = 0;
    virtual void visit(WhileK*) = 0;
    virtual void visit(CheckWhileCondK*) = 0;
    virtual void visit(ForK*) = 0;
    virtual void visit(ForIterateK*) = 0;
    virtual void visit(LeaveK*) = 0;
    virtual void visit(ContinueK*) = 0;
    virtual void visit(ReturnK*) = 0;
    virtual void visit(CreateReturnK*) = 0;
    virtual void visit(BranchK*) = 0;
    virtual void visit(CheckBranchK*) = 0;
    virtual void visit(FunctionCallK*) = 0;
    virtual void visit(RestoreEnvK*) = 0;
    virtual void visit(DerivedOperatorK*) = 0;
    virtual void visit(ApplyDerivedOperatorK*) = 0;
    virtual void visit(ApplyAxisK*) = 0;
    virtual void visit(CellIterK*) = 0;
    virtual void visit(CellCollectK*) = 0;
    virtual void visit(FiberReduceK*) = 0;
    virtual void visit(FiberReduceCollectK*) = 0;
    virtual void visit(PrefixScanK*) = 0;
    virtual void visit(PrefixScanCollectK*) = 0;
    virtual void visit(RowScanK*) = 0;
    virtual void visit(RowScanCollectK*) = 0;
    virtual void visit(ReduceResultK*) = 0;
    virtual void visit(IndexedAssignK*) = 0;
    virtual void visit(IndexedAssignIndexK*) = 0;
    virtual void visit(PerformIndexedAssignK*) = 0;
    virtual void visit(IndexListK*) = 0;
    virtual void visit(IndexListCollectK*) = 0;
    virtual void visit(InvokeDefinedOperatorK*) = 0;
    virtual void visit(ValueK*) = 0;
    virtual void visit(MonadicCallK*) = 0;
    virtual void visit(EvalMonadicCallFnK*) = 0;
    virtual void visit(PerformMonadicCallK*) = 0;
    virtual void visit(DyadicCallK*) = 0;
    virtual void visit(EvalDyadicCallLeftK*) = 0;
    virtual void visit(EvalDyadicCallFnK*) = 0;
    virtual void visit(PerformDyadicCallK*) = 0;
    virtual void visit(EigenReduceK*) = 0;
    virtual void visit(PerformEigenReduceK*) = 0;
    virtual void visit(EigenProductK*) = 0;
    virtual void visit(EvalEigenProductLeftK*) = 0;
    virtual void visit(PerformEigenProductK*) = 0;
    virtual void visit(EigenOuterK*) = 0;
    virtual void visit(EvalEigenOuterLeftK*) = 0;
    virtual void visit(PerformEigenOuterK*) = 0;
    virtual void visit(EigenScanK*) = 0;
    virtual void visit(PerformEigenScanK*) = 0;
    virtual void visit(EigenReduceFirstK*) = 0;
    virtual void visit(PerformEigenReduceFirstK*) = 0;
    virtual void visit(EigenSortK*) = 0;
    virtual void visit(PerformEigenSortK*) = 0;
    virtual void visit(TypeDirectedK*) = 0;
    virtual void visit(ReturnTypeRecordK*) = 0;
};

// Abstract Continuation base class
// Represents "what to do next" in the CEK machine
class Continuation : public GCObject {
private:
    // Only Heap can allocate Continuation objects
    friend class Heap;

    // Private new operator enforces heap-only allocation
    void* operator new(size_t size) { return ::operator new(size); }

    // Placement new for arena allocation (used by Heap::allocate_ephemeral)
    void* operator new(size_t /*size*/, void* ptr) { return ptr; }

protected:
    // Protected delete allows derived class destructors to work
    void operator delete(void* ptr) { ::operator delete(ptr); }

    // Source location for error reporting (0,0 = no location)
    int src_line = 0;
    int src_column = 0;

    Continuation() : GCObject() {}
    Continuation(int line, int col) : GCObject(), src_line(line), src_column(col) {}
    virtual ~Continuation() {}

public:
    // Source location accessors
    int line() const { return src_line; }
    int column() const { return src_column; }
    bool has_location() const { return src_line > 0 || src_column > 0; }
    void set_location(int line, int col) { src_line = line; src_column = col; }

public:

    // Mark all Values and Continuations referenced by this continuation for GC
    virtual void mark(Heap* heap) = 0;

    // Visitor pattern - enables external operations on continuations
    virtual void accept(ContinuationVisitor& visitor) = 0;

    // Query methods for control flow handling

    // Is this continuation a function boundary?
    // Used by RETURN to find where to stop unwinding
    virtual bool is_function_boundary() const { return false; }

    // Is this continuation a loop boundary?
    // Used by BREAK/CONTINUE to find loop context
    virtual bool is_loop_boundary() const { return false; }

    // Is this a boundary that catches BREAK (:Leave)?
    virtual bool is_break_boundary() const { return false; }

    // Is this a boundary that catches CONTINUE (:Continue)?
    virtual bool is_continue_boundary() const { return false; }

    // Get completion being propagated (if any)
    // Used by catch handlers to check for completions without dynamic_cast
    virtual Completion* get_propagating_completion() const { return nullptr; }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }
    Completion* get_propagating_completion() const override { return completion; }

protected:
    void invoke(Machine* machine) override;
};

// CatchReturnK - Catches RETURN completions at function boundaries
// Pushed by FrameK to establish function call boundaries
class CatchReturnK : public Continuation {
public:
    String* function_name;  // For debugging (GC-managed interned string)

    CatchReturnK(String* name = nullptr) : function_name(name) {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }
    bool is_function_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// CatchBreakK - Catches BREAK completions at loop boundaries
// Pushed by WhileK and ForK to establish loop boundaries for :Leave
class CatchBreakK : public Continuation {
public:
    CatchBreakK() = default;

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }
    bool is_loop_boundary() const override { return true; }
    bool is_break_boundary() const override { return true; }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }
    bool is_continue_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// CatchErrorK - Catches THROW completions for error handling (Phase 5)
// Can be pushed at any point to establish an error boundary
// For ⎕EA: holds handler continuation to execute when error is caught
class CatchErrorK : public Continuation {
public:
    Continuation* handler;  // Handler to execute on error (nullptr = discard error)

    CatchErrorK(Continuation* h = nullptr) : handler(h) {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ClearErrorStateK - Clears ⎕ET and ⎕EM after error handler completes successfully
// ISO 13751: Error state should be current state, not historical
class ClearErrorStateK : public Continuation {
public:
    ClearErrorStateK() {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ThrowErrorK - Creates and propagates a THROW completion (Phase 5.2)
// Used by primitives and other code to signal errors through completions
class ThrowErrorK : public Continuation {
public:
    String* error_message;      // Error message (interned)

    ThrowErrorK(String* msg) : error_message(msg) {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ClosureLiteralK - Parse-time continuation for closure literals (dfns)
// Stores a Continuation* (the function body) directly
// At runtime, this gets converted to a CLOSURE Value* by the Machine
class ClosureLiteralK : public Continuation {
public:
    Continuation* body;         // The function body continuation graph
    bool is_niladic;            // True if dfn doesn't use ⍵ (auto-invokes when called without args)

    ClosureLiteralK(Continuation* b, bool niladic = false)
        : body(b), is_niladic(niladic) {}

    ~ClosureLiteralK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// DefinedOperatorLiteralK - Parse-time continuation for operator definitions
// Created from: (FF OP) ← {body} or (FF OP GG) ← {body}
// At runtime, creates DEFINED_OPERATOR value and assigns to operator name
class DefinedOperatorLiteralK : public Continuation {
public:
    Continuation* body;              // The operator body
    String* operator_name;           // OP - the name being defined
    String* left_operand_name;       // FF - left operand parameter
    String* right_operand_name;      // GG - right operand (nullptr for monadic)

    DefinedOperatorLiteralK(Continuation* b, String* op_name,
                            String* left_op, String* right_op = nullptr)
        : body(b), operator_name(op_name),
          left_operand_name(left_op), right_operand_name(right_op) {}

    ~DefinedOperatorLiteralK() override {}

    bool is_dyadic_operator() const { return right_operand_name != nullptr; }

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// LookupK - Parse-time continuation for variable lookups
// Stores the variable name (interned pointer from StringPool)
// At runtime, looks up the variable in the environment
class LookupK : public Continuation {
public:
    String* var_name;           // Variable name (interned)

    LookupK(String* name)
        : var_name(name) {}

    ~LookupK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// AssignK - Assignment continuation for variable definition
// Evaluates the expression, then binds the result to a variable name
// Syntax: name ← expression
class AssignK : public Continuation {
public:
    String* var_name;           // Variable name to assign to (interned)
    Continuation* expr;         // Expression to evaluate

    AssignK(String* name, Continuation* e)
        : var_name(name), expr(e) {}

    ~AssignK() override {
        // Don't delete expr - it's GC-managed
    }

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for AssignK - performs the actual binding after expression is evaluated
class PerformAssignK : public Continuation {
public:
    String* var_name;           // Variable name to assign to (interned)

    PerformAssignK(String* name)
        : var_name(name) {}

    ~PerformAssignK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// SysVarReadK - Read a system variable (⎕IO, ⎕PP, etc.)
class SysVarReadK : public Continuation {
public:
    SysVarId var_id;

    SysVarReadK(SysVarId id) : var_id(id) {}
    ~SysVarReadK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// SysVarAssignK - Assignment to a system variable
class SysVarAssignK : public Continuation {
public:
    SysVarId var_id;
    Continuation* expr;

    SysVarAssignK(SysVarId id, Continuation* e) : var_id(id), expr(e) {}
    ~SysVarAssignK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// PerformSysVarAssignK - Performs actual system variable assignment after expression evaluation
class PerformSysVarAssignK : public Continuation {
public:
    SysVarId var_id;

    PerformSysVarAssignK(SysVarId id) : var_id(id) {}
    ~PerformSysVarAssignK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// StrandK - Lexical strand continuation for numeric vector literals (ISO 13751)
// LiteralStrandK - Stores a pre-computed vector Value* from the lexer
// At runtime, just returns this Value
// Example: "1 2 3" → LiteralStrandK(vector_value)
class LiteralStrandK : public Continuation {
public:
    Value* vector_value;  // Pre-allocated vector Value

    LiteralStrandK(Value* val)
        : vector_value(val) {}

    ~LiteralStrandK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// JuxtaposeK - G2 Grammar juxtaposition: fbn-term ::= fb-term fbn-term
// Implements: if type(x₁) = bas then x₂(x₁) else x₁(x₂)
// ISO 13751: Adjacent values (both basic) are SYNTAX ERROR - no runtime stranding
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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// FinalizeK - Forces g' finalization of parenthesized expressions
// When a parenthesized expression evaluates to a curry, finalize it to a value
// This ensures (f/x) becomes a value before being used in strand context
class FinalizeK : public Continuation {
public:
    Continuation* inner;  // The expression to evaluate and finalize
    bool finalize_gprime;  // If false, only finalize DYADIC_CURRY, not G_PRIME (for parentheses)

    FinalizeK(Continuation* expr, bool gprime = true) : inner(expr), finalize_gprime(gprime) {}

    ~FinalizeK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// PerformFinalizeK - Auxiliary that checks result after inner evaluates
class PerformFinalizeK : public Continuation {
public:
    bool finalize_gprime;  // If false, only finalize DYADIC_CURRY, not G_PRIME

    PerformFinalizeK(bool gprime = true) : finalize_gprime(gprime) {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// MonadicK - Monadic function application (e.g., -x, ⍳x)
// Evaluates operand, then applies monadic function
class MonadicK : public Continuation {
public:
    String* op_name;            // Operator name (interned)
    Continuation* operand;      // Operand to evaluate

    MonadicK(String* name, Continuation* op)
        : op_name(name), operand(op) {}

    ~MonadicK() override {
        // Don't delete operand - it's GC-managed
    }

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// DyadicK - Dyadic function application (e.g., x+y, x×y)
// Evaluates operands right-to-left, then applies dyadic function
class DyadicK : public Continuation {
public:
    String* op_name;            // Operator name (interned)
    Continuation* left;         // Left operand
    Continuation* right;        // Right operand

    DyadicK(String* name, Continuation* l, Continuation* r)
        : op_name(name), left(l), right(r) {}

    ~DyadicK() override {
        // Don't delete operands - they're GC-managed
    }

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for DyadicK - evaluates left after right is done
class EvalDyadicLeftK : public Continuation {
public:
    String* op_name;            // Operator name (interned)
    Continuation* left;
    Value* right_val;           // Saved right value (set at runtime)

    EvalDyadicLeftK(String* name, Continuation* l, Value* r)
        : op_name(name), left(l), right_val(r) {}

    ~EvalDyadicLeftK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation to apply monadic function after operand evaluated
class ApplyMonadicK : public Continuation {
public:
    String* op_name;            // Operator name (interned)

    ApplyMonadicK(String* name)
        : op_name(name) {}

    ~ApplyMonadicK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation to apply dyadic function after both operands evaluated
class ApplyDyadicK : public Continuation {
public:
    String* op_name;            // Operator name (interned)
    Value* right_val;           // Saved right value

    ApplyDyadicK(String* name, Value* r)
        : op_name(name), right_val(r) {}

    ~ApplyDyadicK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// FrameK - Stack frame continuation for function calls
// Marks function boundaries and saves return continuation
class FrameK : public Continuation {
public:
    String* function_name;      // Name of function (for debugging)
    Continuation* return_k;     // Where to return to

    FrameK(String* name, Continuation* ret)
        : function_name(name), return_k(ret) {}

    ~FrameK() override {
        // Don't delete return_k - it's GC-managed
    }

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
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

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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

    DeferredDispatchK(Value* fn, Value* left)
        : fn_val(fn), left_val(left) {}

    ~DeferredDispatchK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    String* var_name;            // Iterator variable name (interned)
    Continuation* array_expr;    // Expression that produces the array
    Continuation* body;          // Loop body to execute

    ForK(String* var, Continuation* arr, Continuation* loop_body)
        : var_name(var), array_expr(arr), body(loop_body) {}

    ~ForK() override {
        // Don't delete - all are GC-managed
    }

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

    // ForK marks loop boundaries for :Leave
    bool is_loop_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for ForK - iterates over array elements
class ForIterateK : public Continuation {
public:
    String* var_name;            // Iterator variable name (interned)
    Value* array;                // Array to iterate over
    Continuation* body;          // Loop body
    size_t index;                // Current iteration index

    ForIterateK(String* var, Value* arr, Continuation* loop_body, size_t idx)
        : var_name(var), array(arr), body(loop_body), index(idx) {}

    ~ForIterateK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ContinueK - Skip to next iteration of loop (Phase 3.3.5)
// Syntax: :Continue
// Creates CONTINUE completion record to jump to next loop iteration
class ContinueK : public Continuation {
public:
    ContinueK() {}

    ~ContinueK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for ReturnK - creates RETURN completion after value is evaluated
class CreateReturnK : public Continuation {
public:
    CreateReturnK() {}

    ~CreateReturnK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// BranchK - Traditional APL branch (→)
// Syntax: →target
// If target is 0 or empty (⍬), exit function (like :Return)
// Non-zero targets are not supported (would require line numbers)
class BranchK : public Continuation {
public:
    Continuation* target_expr;    // Expression that produces the branch target

    BranchK(Continuation* target)
        : target_expr(target) {}

    ~BranchK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary continuation for BranchK - checks target and decides whether to exit
class CheckBranchK : public Continuation {
public:
    Value* saved_result;  // Result value from before branch target was evaluated

    CheckBranchK(Value* result) : saved_result(result) {}

    ~CheckBranchK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    String* op_name;              // The dyadic operator name (interned)

    DerivedOperatorK(Continuation* operand, String* operator_name,
                     Continuation* axis = nullptr)
        : operand_cont(operand), axis_cont(axis), op_name(operator_name) {}

    ~DerivedOperatorK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ApplyDerivedOperatorK - Apply dyadic operator to its first operand
// Creates a DERIVED_OPERATOR value (or OPERATOR_CURRY if axis is specified)
class ApplyDerivedOperatorK : public Continuation {
public:
    String* op_name;              // Operator name (interned)
    Continuation* axis_cont;      // Optional axis (for f/[k] syntax)

    ApplyDerivedOperatorK(String* operator_name, Continuation* axis = nullptr)
        : op_name(operator_name), axis_cont(axis) {}

    ~ApplyDerivedOperatorK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    SCAN_LEFT,    // Accumulate left-to-right, keep all intermediates (strand scan)
    OUTER,        // Cartesian product iteration (outer product)
    INNER         // Inner product: extract fibers, apply g element-wise, reduce with f
};

class CellIterK : public Continuation {
public:
    Value* fn;              // Function to apply (f for reduce in INNER mode)
    Value* lhs;             // Left array (nullptr for monadic)
    Value* rhs;             // Right array
    int left_rank;          // Cell rank for left arg (0=scalars, 1=rows, etc.)
    int right_rank;         // Cell rank for right arg
    int total_cells;        // Total cells to process
    int current_cell;       // Current cell index (counts from end for fold/scan)
    CellIterMode mode;      // How to combine results
    std::vector<Value*> results;  // Collected results
    Value* accumulator;     // For FOLD_RIGHT, SCAN_RIGHT, and SCAN_LEFT modes

    // Original array shape info for reassembly
    int orig_rows;
    int orig_cols;
    bool orig_is_vector;
    bool orig_is_char;     // Preserve character data flag
    bool orig_is_strand;   // Preserve strand structure for pervasive dispatch
    std::vector<int> orig_ndarray_shape;  // For NDARRAY: preserve shape for reassembly

    // For OUTER mode: dimensions for Cartesian product
    int lhs_total;          // Total elements in lhs (for OUTER/INNER)
    int rhs_total;          // Total elements in rhs (for OUTER/INNER)
    int lhs_cols;           // Columns in lhs (for extracting elements)
    int rhs_cols;           // Columns in rhs (for extracting elements)

    // For INNER mode: element-wise function and common dimension
    Value* g_fn;            // Element-wise function g (fn is reduce function f)
    int common_dim;         // Length of fibers (last dim of lhs = first dim of rhs)

    CellIterK(Value* f, Value* l, Value* r, int lk, int rk, int total,
              CellIterMode m, int rows, int cols, bool is_vec, bool is_char = false,
              bool is_strand = false)
        : fn(f), lhs(l), rhs(r), left_rank(lk), right_rank(rk),
          total_cells(total), current_cell(0), mode(m), accumulator(nullptr),
          orig_rows(rows), orig_cols(cols), orig_is_vector(is_vec), orig_is_char(is_char),
          orig_is_strand(is_strand), lhs_total(0), rhs_total(0), lhs_cols(1), rhs_cols(1),
          g_fn(nullptr), common_dim(0) {
        if (mode == CellIterMode::COLLECT || mode == CellIterMode::SCAN_RIGHT ||
            mode == CellIterMode::SCAN_LEFT || mode == CellIterMode::OUTER) {
            results.reserve(total);
        }
        // For right-to-left modes, start from the last cell
        if (mode == CellIterMode::FOLD_RIGHT || mode == CellIterMode::SCAN_RIGHT) {
            current_cell = total - 1;
        }
        // SCAN_LEFT starts at 0 (default), iterates forward
    }

    // Constructor for OUTER mode with explicit dimensions
    CellIterK(Value* f, Value* l, Value* r, int l_total, int r_total, int l_cols, int r_cols)
        : fn(f), lhs(l), rhs(r), left_rank(0), right_rank(0),
          total_cells(l_total * r_total), current_cell(0), mode(CellIterMode::OUTER),
          accumulator(nullptr), orig_rows(l_total), orig_cols(r_total), orig_is_vector(false), orig_is_char(false),
          orig_is_strand(false), lhs_total(l_total), rhs_total(r_total), lhs_cols(l_cols), rhs_cols(r_cols),
          g_fn(nullptr), common_dim(0) {
        results.reserve(total_cells);
    }

    // Constructor for INNER mode (inner product)
    // f = reduce function, g = element-wise function
    // lhs_frame = positions in lhs frame, rhs_frame = positions in rhs frame
    // common = fiber length (last dim of lhs = first dim of rhs)
    CellIterK(Value* f, Value* g, Value* l, Value* r,
              int lhs_frame, int rhs_frame, int common,
              int l_cols, int r_cols)
        : fn(f), lhs(l), rhs(r), left_rank(0), right_rank(0),
          total_cells(lhs_frame * rhs_frame), current_cell(0), mode(CellIterMode::INNER),
          accumulator(nullptr), orig_rows(lhs_frame), orig_cols(rhs_frame),
          orig_is_vector(false), orig_is_char(false), orig_is_strand(false),
          lhs_total(lhs_frame), rhs_total(rhs_frame), lhs_cols(l_cols), rhs_cols(r_cols),
          g_fn(g), common_dim(common) {
        results.reserve(total_cells);
    }

    ~CellIterK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// FiberReduceK - Unified reduction along array fibers
// ============================================================================
// Handles all reduction operations: f/ and N f/ on any array type.
// Iterates over fibers (1D slices) of an array along a specified axis,
// optionally using sliding windows for N-wise reduction.
//
// Examples:
//   f/ vector        → 1 fiber, full reduce → scalar
//   f/ strand        → 1 fiber, full reduce → scalar
//   f/ matrix        → rows fibers (axis=1), full reduce → vector
//   f/[1] matrix     → cols fibers (axis=0), full reduce → vector
//   2 f/ vector      → 1 fiber, 2-wise windows → vector
//   2 f/ matrix      → rows fibers, 2-wise each → matrix
//   f/ ndarray       → fibers along last axis → shape with axis removed
//   2 f/[k] ndarray  → fibers along axis k, 2-wise → shape with axis[k] reduced

class FiberReduceK : public Continuation {
public:
    Value* fn;              // Function to reduce with
    Value* source;          // Source array (scalar, vector, strand, matrix, NDARRAY)
    int axis;               // 0-indexed axis to reduce along
    int window_size;        // 0 = full reduce, N = N-wise windows
    bool reverse;           // True if N was negative (reverse window elements)

    // Iteration state - flattened over (fiber, window) pairs
    int current_result;     // Current position in results
    int total_results;      // total_fibers * windows_per_fiber
    int total_fibers;       // Product of all dims except axis
    int fiber_length;       // Length of axis dimension
    int windows_per_fiber;  // 1 for full reduce, fiber_length - window_size + 1 for N-wise

    // For source indexing
    std::vector<int> source_shape;   // Shape of source (empty for 1D)
    std::vector<int> source_strides; // Strides for element access

    // Results and output shape
    std::vector<Value*> results;
    std::vector<int> result_shape;   // Shape of output array
    bool is_strand;                  // True if source is strand (results may be non-scalar)

    // Unified constructor
    FiberReduceK(Value* f, Value* src, int ax, int window, bool rev);

    ~FiberReduceK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// FiberReduceCollectK - Collects reduction result and continues
class FiberReduceCollectK : public Continuation {
public:
    FiberReduceK* iter;

    explicit FiberReduceCollectK(FiberReduceK* i) : iter(i) {}

    ~FiberReduceCollectK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    Value* source;          // Array to scan (matrix or ndarray)
    int current_pos;        // Current result position being processed
    int total_positions;    // Total result positions to process
    int slice_len;          // Length of each slice to scan
    std::vector<Value*> results;  // Collected scan results (vectors)
    int scan_axis;          // Axis to scan (0-indexed)
    std::vector<int> result_shape;  // Shape of result (for NDARRAY, same as input)

    // Constructor for matrix scan (backward compatible)
    RowScanK(Value* f, Value* m, int rows, int c, bool first_axis = false)
        : fn(f), source(m), current_pos(0), total_positions(rows), slice_len(c),
          scan_axis(first_axis ? 0 : 1) {
        results.reserve(rows);
    }

    // Constructor for NDARRAY scan
    RowScanK(Value* f, Value* arr, int axis, const std::vector<int>& shape, int positions)
        : fn(f), source(arr), current_pos(0), total_positions(positions), slice_len(0),
          scan_axis(axis), result_shape(shape) {
        results.reserve(positions);
        // slice_len is extracted from source shape in invoke()
    }

    ~RowScanK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    void accept(ContinuationVisitor& v) override { v.visit(this); }

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
    String* var_name;             // Variable name (interned)
    Continuation* index_cont;     // Index expression
    Continuation* value_cont;     // Value expression

    IndexedAssignK(String* var, Continuation* idx, Continuation* val)
        : var_name(var), index_cont(idx), value_cont(val) {}

    ~IndexedAssignK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// IndexedAssignIndexK - Value evaluated, now evaluate index
class IndexedAssignIndexK : public Continuation {
public:
    String* var_name;             // Variable name (interned)
    Value* value_val;             // Already-evaluated value
    Continuation* index_cont;     // Index expression to evaluate

    IndexedAssignIndexK(String* var, Value* val, Continuation* idx)
        : var_name(var), value_val(val), index_cont(idx) {}

    ~IndexedAssignIndexK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// PerformIndexedAssignK - Value and index evaluated, perform the assignment
class PerformIndexedAssignK : public Continuation {
public:
    String* var_name;             // Variable name (interned)
    Value* value_val;             // Value to assign
    Value* index_val;             // Index value(s)

    PerformIndexedAssignK(String* var, Value* val, Value* idx)
        : var_name(var), value_val(val), index_val(idx) {}

    ~PerformIndexedAssignK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// IndexListK - Multi-axis index list for M[I;J;K] syntax (ISO 13751 10.2.14)
// ============================================================================
// Evaluates a list of index expressions and collects results into a strand.
// Empty vectors (zilde) represent elided indices meaning "all elements".

class IndexListK : public Continuation {
public:
    std::vector<Continuation*> indices;  // Index expressions to evaluate
    size_t current;                       // Current index being evaluated

    explicit IndexListK(std::vector<Continuation*>&& idx)
        : indices(std::move(idx)), current(0) {}

    ~IndexListK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// IndexListCollectK - Collects evaluated index values into a strand
class IndexListCollectK : public Continuation {
public:
    std::vector<Continuation*> indices;   // Remaining index expressions
    size_t current;                        // Next index to evaluate
    std::vector<Value*> results;           // Collected index values

    IndexListCollectK(std::vector<Continuation*> idx, size_t cur, std::vector<Value*> res)
        : indices(std::move(idx)), current(cur), results(std::move(res)) {}

    ~IndexListCollectK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// InvokeDefinedOperatorK - Invoke a user-defined operator
// ============================================================================
// Binds operands and arguments in a new environment, then executes the body
// Used when a DERIVED_OPERATOR (with defined_op) is applied to arguments

class InvokeDefinedOperatorK : public Continuation {
public:
    Value::DefinedOperatorData* op;  // The defined operator
    Value* operator_value;            // The DEFINED_OPERATOR Value (for ∇ binding)
    Value* left_operand;              // Always present (FF)
    Value* right_operand;             // For dyadic operators (GG), or axis value
    Value* left_arg;                  // For dyadic application (⍺)
    Value* right_arg;                 // Always present (⍵)

    InvokeDefinedOperatorK(Value::DefinedOperatorData* defined_op,
                           Value* op_value,
                           Value* left_op, Value* right_op,
                           Value* left, Value* right)
        : op(defined_op), operator_value(op_value), left_operand(left_op), right_operand(right_op),
          left_arg(left), right_arg(right) {}

    ~InvokeDefinedOperatorK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }
    bool is_function_boundary() const override { return true; }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// ValueK - Pre-computed value node (optimizer result)
// ============================================================================
// Stores a Value* that was computed at optimization time (constant folding,
// operator pre-building, etc.). When invoked, simply returns the stored value.
// Created only by StaticOptimizer; never emitted by the parser.
class ValueK : public Continuation {
public:
    Value* value;

    explicit ValueK(Value* v) : value(v) {}
    ~ValueK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// MonadicCallK - Direct monadic function call (optimizer D1 pattern)
// Evaluates fn_cont and arg_cont, then calls apply_function_immediate(fn, nullptr, arg).
// Bypasses the G_PRIME curry mechanism entirely.
// Created only by StaticOptimizer; never emitted by the parser.
class MonadicCallK : public Continuation {
public:
    Continuation* fn_cont;   // Continuation that produces the function value
    Continuation* arg_cont;  // Continuation that produces the argument value

    MonadicCallK(Continuation* fn, Continuation* arg)
        : fn_cont(fn), arg_cont(arg) {}
    ~MonadicCallK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary: evaluates fn after arg is ready
class EvalMonadicCallFnK : public Continuation {
public:
    Continuation* fn_cont;
    Value* arg_val;

    EvalMonadicCallFnK(Continuation* fn, Value* arg)
        : fn_cont(fn), arg_val(arg) {}
    ~EvalMonadicCallFnK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary: dispatches the monadic call after both fn and arg are ready
class PerformMonadicCallK : public Continuation {
public:
    Value* arg_val;

    explicit PerformMonadicCallK(Value* arg) : arg_val(arg) {}
    ~PerformMonadicCallK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// DyadicCallK - Direct dyadic function call (optimizer D2 pattern)
// Evaluates fn_cont, left_cont, right_cont, then calls
// apply_function_immediate(fn, left, right).
// Bypasses the G_PRIME curry mechanism entirely.
// Created only by StaticOptimizer; never emitted by the parser.
class DyadicCallK : public Continuation {
public:
    Continuation* fn_cont;    // Continuation that produces the function value
    Continuation* left_cont;  // Continuation that produces the left argument
    Continuation* right_cont; // Continuation that produces the right argument

    DyadicCallK(Continuation* fn, Continuation* left, Continuation* right)
        : fn_cont(fn), left_cont(left), right_cont(right) {}
    ~DyadicCallK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary: evaluates left after right is ready
class EvalDyadicCallLeftK : public Continuation {
public:
    Continuation* fn_cont;
    Continuation* left_cont;
    Value* right_val;

    EvalDyadicCallLeftK(Continuation* fn, Continuation* left, Value* right)
        : fn_cont(fn), left_cont(left), right_val(right) {}
    ~EvalDyadicCallLeftK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary: evaluates fn after left and right are ready
class EvalDyadicCallFnK : public Continuation {
public:
    Continuation* fn_cont;
    Value* left_val;
    Value* right_val;

    EvalDyadicCallFnK(Continuation* fn, Value* left, Value* right)
        : fn_cont(fn), left_val(left), right_val(right) {}
    ~EvalDyadicCallFnK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// Auxiliary: dispatches the dyadic call after fn, left, and right are ready
class PerformDyadicCallK : public Continuation {
public:
    Value* left_val;
    Value* right_val;

    PerformDyadicCallK(Value* left, Value* right) : left_val(left), right_val(right) {}
    ~PerformDyadicCallK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// EigenReduceK - Direct Eigen reduction (optimizer E3 pattern)
// For type-proven vector reductions: +/vec → sum(), ×/vec → prod(), etc.
// Bypasses the entire operator dispatch chain.
// Created only by StaticOptimizer; never emitted by the parser.
// ============================================================================

enum class EigenReduceOp { SUM, PROD, MAX, MIN };

class EigenReduceK : public Continuation {
public:
    EigenReduceOp reduce_op;
    Continuation* arg_cont;
    Value* derived_op;  // for fallback if runtime type doesn't match

    EigenReduceK(EigenReduceOp op, Continuation* arg, Value* derived)
        : reduce_op(op), arg_cont(arg), derived_op(derived) {}
    ~EigenReduceK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

class PerformEigenReduceK : public Continuation {
public:
    EigenReduceOp reduce_op;
    Value* derived_op;

    PerformEigenReduceK(EigenReduceOp op, Value* derived)
        : reduce_op(op), derived_op(derived) {}
    ~PerformEigenReduceK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// EigenProductK - Direct Eigen matrix product (optimizer E1 pattern)
// For +.× inner product: vec·vec, mat×mat, vec×mat, mat×vec.
// Created only by StaticOptimizer; never emitted by the parser.
// ============================================================================

class EigenProductK : public Continuation {
public:
    Continuation* left_cont;
    Continuation* right_cont;
    Value* derived_op;  // for fallback

    EigenProductK(Continuation* left, Continuation* right, Value* derived)
        : left_cont(left), right_cont(right), derived_op(derived) {}
    ~EigenProductK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

class EvalEigenProductLeftK : public Continuation {
public:
    Continuation* left_cont;
    Value* right_val;
    Value* derived_op;

    EvalEigenProductLeftK(Continuation* left, Value* right, Value* derived)
        : left_cont(left), right_val(right), derived_op(derived) {}
    ~EvalEigenProductLeftK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

class PerformEigenProductK : public Continuation {
public:
    Value* left_val;
    Value* right_val;
    Value* derived_op;

    PerformEigenProductK(Value* left, Value* right, Value* derived)
        : left_val(left), right_val(right), derived_op(derived) {}
    ~PerformEigenProductK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// EigenOuterK - Direct Eigen outer product (optimizer E2 pattern)
// For ∘.f outer product on numeric vectors: ×, +, ⌊, ⌈.
// Created only by StaticOptimizer; never emitted by the parser.
// ============================================================================

enum class EigenOuterOp { TIMES, PLUS, MIN, MAX, EQ, NE, LT, GT, LE, GE };

class EigenOuterK : public Continuation {
public:
    EigenOuterOp outer_op;
    Continuation* left_cont;
    Continuation* right_cont;
    Value* derived_op;  // for fallback

    EigenOuterK(EigenOuterOp op, Continuation* left, Continuation* right, Value* derived)
        : outer_op(op), left_cont(left), right_cont(right), derived_op(derived) {}
    ~EigenOuterK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

class EvalEigenOuterLeftK : public Continuation {
public:
    EigenOuterOp outer_op;
    Continuation* left_cont;
    Value* right_val;
    Value* derived_op;

    EvalEigenOuterLeftK(EigenOuterOp op, Continuation* left, Value* right, Value* derived)
        : outer_op(op), left_cont(left), right_val(right), derived_op(derived) {}
    ~EvalEigenOuterLeftK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

class PerformEigenOuterK : public Continuation {
public:
    EigenOuterOp outer_op;
    Value* left_val;
    Value* right_val;
    Value* derived_op;

    PerformEigenOuterK(EigenOuterOp op, Value* left, Value* right, Value* derived)
        : outer_op(op), left_val(left), right_val(right), derived_op(derived) {}
    ~PerformEigenOuterK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// EigenScanK - Direct Eigen scan (optimizer E4 pattern)
// For prefix scan (+\, ×\, ⌈\, ⌊\) on type-proven vectors.
// Created only by StaticOptimizer; never emitted by the parser.
// ============================================================================

class EigenScanK : public Continuation {
public:
    EigenReduceOp scan_op;  // reuse EigenReduceOp enum for scan ops
    Continuation* arg_cont;
    Value* derived_op;

    EigenScanK(EigenReduceOp op, Continuation* arg, Value* derived)
        : scan_op(op), arg_cont(arg), derived_op(derived) {}
    ~EigenScanK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

class PerformEigenScanK : public Continuation {
public:
    EigenReduceOp scan_op;
    Value* derived_op;

    PerformEigenScanK(EigenReduceOp op, Value* derived)
        : scan_op(op), derived_op(derived) {}
    ~PerformEigenScanK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// EigenReduceFirstK - Direct Eigen column-wise reduction (optimizer E5 pattern)
// For reduce-first (+⌿, ×⌿, ⌈⌿, ⌊⌿) on type-proven matrices.
// Created only by StaticOptimizer; never emitted by the parser.
// ============================================================================

class EigenReduceFirstK : public Continuation {
public:
    EigenReduceOp reduce_op;
    Continuation* arg_cont;
    Value* derived_op;

    EigenReduceFirstK(EigenReduceOp op, Continuation* arg, Value* derived)
        : reduce_op(op), arg_cont(arg), derived_op(derived) {}
    ~EigenReduceFirstK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

class PerformEigenReduceFirstK : public Continuation {
public:
    EigenReduceOp reduce_op;
    Value* derived_op;

    PerformEigenReduceFirstK(EigenReduceOp op, Value* derived)
        : reduce_op(op), derived_op(derived) {}
    ~PerformEigenReduceFirstK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ============================================================================
// EigenSortK - Direct sort (optimizer I1 pattern)
// For X[⍋X] (ascending) and X[⍒X] (descending) on type-proven vectors.
// Created only by StaticOptimizer; never emitted by the parser.
// ============================================================================

enum class EigenSortDir { ASCENDING, DESCENDING };

class EigenSortK : public Continuation {
public:
    EigenSortDir direction;
    Continuation* arg_cont;

    EigenSortK(EigenSortDir dir, Continuation* arg)
        : direction(dir), arg_cont(arg) {}
    ~EigenSortK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

class PerformEigenSortK : public Continuation {
public:
    EigenSortDir direction;

    explicit PerformEigenSortK(EigenSortDir dir)
        : direction(dir) {}
    ~PerformEigenSortK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

} // namespace apl
