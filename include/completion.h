// Completion Records - Structured control flow for APL

#pragma once

#include "value.h"
#include <cstring>

namespace apl {

// Forward declarations
class Heap;

// Completion types for control flow
enum class CompletionType {
    NORMAL,      // Normal evaluation - continue execution
    BREAK,       // :Leave or break - exit from loop/block
    CONTINUE,    // Continue in loop - jump to next iteration
    RETURN,      // → return or :Return - exit from function
    THROW        // Error/exception - propagate error
};

// Completion record structure
// Represents the result of evaluating an expression or statement
// Used for structured control flow (returns, breaks, exceptions)
// Now GC-managed for proper memory management
struct Completion : public GCObject {
private:
    // Only Heap can allocate/deallocate Completion objects
    friend class Heap;

    // Private new/delete operators enforce heap-only allocation
    void* operator new(size_t size) { return ::operator new(size); }
    void operator delete(void* ptr) { ::operator delete(ptr); }

public:
    CompletionType type;    // Type of completion
    Value* value;           // Result value (nullptr for non-value completions)
    const char* target;     // Target label for break/continue (nullptr if not used)

    // Constructors (public so heap's template allocate works, but new/delete are private)
    Completion()
        : GCObject(), type(CompletionType::NORMAL), value(nullptr), target(nullptr) {}

    Completion(CompletionType t, Value* v = nullptr, const char* tgt = nullptr)
        : GCObject(), type(t), value(v), target(tgt) {}

    // GC support
    void mark(Heap* heap) override;

    // Query methods
    bool is_normal() const { return type == CompletionType::NORMAL; }
    bool is_abrupt() const { return type != CompletionType::NORMAL; }
    bool is_return() const { return type == CompletionType::RETURN; }
    bool is_break() const { return type == CompletionType::BREAK; }
    bool is_continue() const { return type == CompletionType::CONTINUE; }
    bool is_throw() const { return type == CompletionType::THROW; }

    // Check if this completion matches a target label
    bool matches_target(const char* label) const {
        if (!target) return false;
        if (!label) return false;
        return std::strcmp(target, label) == 0;
    }
};

} // namespace apl
