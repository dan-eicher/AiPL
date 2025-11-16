// Completion Records - Structured control flow for APL

#pragma once

#include "value.h"
#include <cstring>

namespace apl {

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
struct APLCompletion {
    CompletionType type;    // Type of completion
    Value* value;           // Result value (nullptr for non-value completions)
    const char* target;     // Target label for break/continue (nullptr if not used)

    // Constructors
    APLCompletion()
        : type(CompletionType::NORMAL), value(nullptr), target(nullptr) {}

    APLCompletion(CompletionType t, Value* v = nullptr, const char* tgt = nullptr)
        : type(t), value(v), target(tgt) {}

    // Factory methods for common completion types
    static APLCompletion* normal(Value* v) {
        return new APLCompletion(CompletionType::NORMAL, v, nullptr);
    }

    static APLCompletion* return_value(Value* v) {
        return new APLCompletion(CompletionType::RETURN, v, nullptr);
    }

    static APLCompletion* break_completion(const char* label = nullptr) {
        return new APLCompletion(CompletionType::BREAK, nullptr, label);
    }

    static APLCompletion* continue_completion(const char* label = nullptr) {
        return new APLCompletion(CompletionType::CONTINUE, nullptr, label);
    }

    static APLCompletion* throw_error(Value* error_value) {
        return new APLCompletion(CompletionType::THROW, error_value, nullptr);
    }

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
