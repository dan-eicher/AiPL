// Machine - Complete CEK machine implementation

#pragma once

#include "control.h"
#include "continuation.h"
#include "heap.h"
#include "environment.h"
#include "string_pool.h"
#include <vector>
#include <unordered_map>
#include <string>

namespace apl {

// Forward declaration
class Parser;

// Machine - The CEK machine execution engine
class Machine {
public:
    // CEK Registers
    Control ctrl;                           // Control register (C)
    Environment* env;                       // Environment register (E)
    std::vector<Continuation*> kont_stack;  // Continuation stack (K)

    // Memory management
    APLHeap* heap;                          // Garbage-collected heap
    StringPool string_pool;                 // Interned strings

    // Parser (owned by machine)
    Parser* parser;

    // Caches for optimization
    std::unordered_map<std::string, Continuation*> function_cache;
    // guard_cache will be added in later phases when we implement optimization

    // Constructor
    Machine();

    // Destructor
    ~Machine();

    // Execution methods

    // Execute the machine until halt
    Value* execute();

    // Stack manipulation

    // Push continuation onto stack
    void push_kont(Continuation* k) {
        kont_stack.push_back(k);
    }

    // Pop continuation from stack
    Continuation* pop_kont() {
        if (kont_stack.empty()) {
            return nullptr;
        }
        Continuation* k = kont_stack.back();
        kont_stack.pop_back();
        return k;
    }

    // Utility methods

    // Check if execution should continue (stack not empty)
    bool should_continue() const {
        return !kont_stack.empty();
    }

    // Trigger GC if needed
    void maybe_gc() {
        if (heap->should_gc()) {
            heap->collect(this);
        }
    }
};

} // namespace apl
