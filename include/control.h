// Control - Control register for CEK machine

#pragma once

#include "completion.h"
#include "value.h"

namespace apl {

// Execution mode for the machine
enum class ExecMode {
    EVALUATING,  // Currently evaluating expressions
    HALTED       // Execution complete or error
};

// Control register - manages evaluation state
// Note: Parsing is done by the Parser class, not here
class Control {
public:
    ExecMode mode;                  // Current execution mode
    Value* value;                   // Current evaluation result
    APLCompletion* completion;      // Current control flow state

    // Constructor
    Control()
        : mode(ExecMode::HALTED),
          value(nullptr),
          completion(nullptr) {}

    // Destructor
    ~Control() {
        cleanup();
    }

    // Initialize for evaluation
    void init_evaluating() {
        cleanup();
        mode = ExecMode::EVALUATING;
        completion = APLCompletion::normal(nullptr);
    }

    // Set completion record
    void set_completion(APLCompletion* comp) {
        if (completion) {
            delete completion;
        }
        completion = comp;
    }

    // Set value (creates NORMAL completion)
    void set_value(Value* v) {
        value = v;
        set_completion(APLCompletion::normal(v));
    }

    // Halt execution
    void halt() {
        mode = ExecMode::HALTED;
    }

    // Check if execution should continue
    bool should_continue() const {
        return mode != ExecMode::HALTED &&
               completion &&
               completion->is_normal();
    }

    // Check for abrupt completion
    bool has_abrupt_completion() const {
        return completion && completion->is_abrupt();
    }

private:
    void cleanup() {
        // Don't delete value - it's managed by the heap
        if (completion) {
            delete completion;
            completion = nullptr;
        }
    }
};

} // namespace apl
