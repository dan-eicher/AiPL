// Continuation - Abstract base class for CEK machine continuations

#pragma once

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

    // Execute this continuation
    // Returns the result value (or nullptr if execution should continue)
    virtual Value* invoke(Machine* machine) = 0;

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
};

// HaltK - Terminal continuation that stops execution
class HaltK : public Continuation {
public:
    Value* invoke(Machine* machine) override;
    void mark(APLHeap* heap) override;
};

// LiteralK - Parse-time continuation for literal values
// Stores a double directly (not a Value*) for GC safety during parsing
// At runtime, this gets converted to a Value* by the Machine
class LiteralK : public Continuation {
public:
    double literal_value;       // The literal number
    Continuation* next;         // Next continuation

    LiteralK(double val, Continuation* k)
        : literal_value(val), next(k) {}

    ~LiteralK() override {
        // Don't delete next - it's GC-managed
    }

    Value* invoke(Machine* machine) override;
    void mark(APLHeap* heap) override;
};

// BinOpK - Parse-time continuation for binary operations
// Stores the operator name (not a PrimitiveFn*) for GC safety during parsing
// At runtime, looks up the operator and applies it
class BinOpK : public Continuation {
public:
    const char* op_name;        // Operator symbol (e.g., "+", "×")
    Continuation* left;         // Left operand continuation
    Continuation* right;        // Right operand continuation

    BinOpK(const char* op, Continuation* l, Continuation* r)
        : op_name(op), left(l), right(r) {}

    ~BinOpK() override {
        // Don't delete left/right - they're GC-managed
    }

    Value* invoke(Machine* machine) override;
    void mark(APLHeap* heap) override;
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

    Value* invoke(Machine* machine) override;
    void mark(APLHeap* heap) override;
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

    Value* invoke(Machine* machine) override;
    void mark(APLHeap* heap) override;

    // FrameK marks function boundaries
    bool is_function_boundary() const override { return true; }
};

} // namespace apl
