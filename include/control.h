// Control - Control register for CEK machine

#pragma once

#include "completion.h"
#include "token.h"
#include "value.h"

namespace apl {

// Forward declaration
struct LexerState;

// Execution mode for the machine
enum class ExecMode {
    PARSING,     // Currently parsing input
    EVALUATING,  // Currently evaluating expressions
    HALTED       // Execution complete or error
};

// Control register - manages execution state
class Control {
public:
    ExecMode mode;                  // Current execution mode
    LexerState* lexer_state;        // Lexer state (for parsing)
    Token current_token;            // Current token being processed
    Value* value;                   // Current evaluation result
    APLCompletion* completion;      // Current control flow state

    // Constructor
    Control()
        : mode(ExecMode::HALTED),
          lexer_state(nullptr),
          value(nullptr),
          completion(nullptr) {
        current_token = Token();  // Default EOF token
    }

    // Destructor
    ~Control() {
        cleanup();
    }

    // Initialize for parsing
    void init_parsing(LexerState* state) {
        cleanup();
        mode = ExecMode::PARSING;
        lexer_state = state;
        completion = APLCompletion::normal(nullptr);
        advance_token();
    }

    // Initialize for evaluation
    void init_evaluating() {
        mode = ExecMode::EVALUATING;
        completion = APLCompletion::normal(nullptr);
    }

    // Advance to next token
    void advance_token();

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
        // Don't delete lexer_state - it's managed externally
        // Don't delete value - it's managed by the heap
        if (completion) {
            delete completion;
            completion = nullptr;
        }
    }
};

} // namespace apl
