// Continuation - Abstract base class for CEK machine continuations

#pragma once

#include <vector>
#include <string>
#include "value.h"

namespace apl {

// Forward declarations
class Machine;
class APLHeap;

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

} // namespace apl
