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

    // No destructor needed - completion is GC-managed

    // Initialize for evaluation
    void init_evaluating() {
        mode = ExecMode::EVALUATING;
        value = nullptr;
        completion = nullptr;  // nullptr = NORMAL completion
    }

    // Set completion record
    void set_completion(APLCompletion* comp) {
        // No delete - GC will handle it
        completion = comp;
    }

    // Set value (NORMAL completion - use nullptr)
    void set_value(Value* v) {
        value = v;
        completion = nullptr;  // nullptr = NORMAL completion
    }

    // Halt execution
    void halt() {
        mode = ExecMode::HALTED;
    }

    // Check if execution should continue (nullptr = NORMAL)
    bool should_continue() const {
        return mode != ExecMode::HALTED && completion == nullptr;
    }

    // Check for abrupt completion (non-null = abrupt)
    bool has_abrupt_completion() const {
        return completion != nullptr;
    }
};

} // namespace apl
