// Machine - CEK machine core (forward declaration for now)

#pragma once

#include "control.h"
#include "continuation.h"
#include <vector>

namespace apl {

// Forward declaration
class APLHeap;

// Machine - The CEK machine execution engine
// This is a minimal implementation for Phase 1.5
// Will be expanded in Phase 1.7
class Machine {
public:
    Control ctrl;                           // Control register
    std::vector<Continuation*> kont_stack;  // Continuation stack
    APLHeap* heap;                          // Memory heap (nullptr for now)

    Machine() : heap(nullptr) {}

    ~Machine() {
        // Clean up continuation stack
        for (auto k : kont_stack) {
            delete k;
        }
        kont_stack.clear();
    }

    // Push continuation onto stack
    void push_kont(Continuation* k) {
        kont_stack.push_back(k);
    }

    // Pop continuation and invoke it
    Value* pop_kont_and_invoke() {
        if (kont_stack.empty()) {
            ctrl.halt();
            return ctrl.value;
        }

        Continuation* k = kont_stack.back();
        kont_stack.pop_back();
        return k->invoke(this);
    }

    // Halt the machine
    void halt() {
        ctrl.halt();
    }
};

} // namespace apl
