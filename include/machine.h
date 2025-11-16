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

    // Caches for optimization
    std::unordered_map<std::string, Continuation*> function_cache;
    // guard_cache will be added in later phases when we implement optimization

    // Constructor
    Machine() {
        heap = new APLHeap();
        env = new Environment();  // Global environment
    }

    // Destructor
    ~Machine() {
        // Clean up continuation stack
        for (auto k : kont_stack) {
            delete k;
        }
        kont_stack.clear();

        // Clean up cached continuations
        for (auto& pair : function_cache) {
            delete pair.second;
        }
        function_cache.clear();

        // Clean up environment chain
        delete env;

        // Clean up heap (which cleans up all values)
        delete heap;
    }

    // Execution methods

    // Execute the machine until halt
    Value* execute();

    // Handle completion records
    void handle_completion();

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

    // Pop continuation and invoke it
    Value* pop_kont_and_invoke() {
        Continuation* k = pop_kont();
        if (!k) {
            ctrl.halt();
            return ctrl.value;
        }
        return k->invoke(this);
    }

    // Unwind stack to a boundary (for RETURN/BREAK/CONTINUE)
    bool unwind_to_boundary(bool (*predicate)(Continuation*), const char* label = nullptr);

    // Utility methods

    // Halt the machine
    void halt() {
        ctrl.halt();
    }

    // Check if execution should continue
    bool should_continue() const {
        return ctrl.mode != ExecMode::HALTED;
    }

    // Trigger GC if needed
    void maybe_gc() {
        if (heap->should_gc()) {
            heap->collect(this);
        }
    }
};

} // namespace apl
