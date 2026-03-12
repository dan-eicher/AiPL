// Machine - Complete CEK machine implementation

#pragma once

#include "continuation.h"
#include "heap.h"
#include "environment.h"
#include "string_pool.h"
#include "sysvar.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <exception>
#include <random>
#include <climits>

namespace apl {

// Implementation limits (ISO 13751 §A.3 - LIMIT ERROR when exceeded)
constexpr size_t MAX_ARRAY_SIZE = INT_MAX;    // Limited by int-based indexing throughout codebase
constexpr size_t MAX_IDENTIFIER_LENGTH = 256; // Maximum identifier name length

// Implementation parameters (ISO 13751 §5.2.5, §5.5)
constexpr double INTEGER_TOLERANCE = 1e-10;   // Tolerance for near-integer checks

// APLError - Exception for APL runtime errors (DOMAIN ERROR, etc.)
// These are user-visible errors that should be caught and displayed.
// Distinct from std::runtime_error which indicates VM bugs.
class APLError : public std::exception {
    std::string message;
public:
    explicit APLError(const char* msg) : message(msg) {}
    explicit APLError(const std::string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

// Forward declarations
class Parser;
class SpecializationBackend;

// Machine - The CEK machine execution engine
class Machine {
public:
    // Machine state (CEK machine)
    Continuation* control;                  // C: Current continuation being executed
    Environment* env;                       // E: Environment (variable bindings)
    std::vector<Continuation*> kont_stack;  // K: Continuation stack
    Value* result;                          // Result of last continuation
    std::vector<Continuation*> error_stack; // Stack snapshot at last error (for traces)
    int io = 1;                             // Index origin (⎕IO): 0 or 1
    int pp = 10;                            // Print precision (⎕PP): 1-17, default 10
    double ct = 1E-13;                      // Comparison tolerance (⎕CT): ISO 13751 initial value; 0 enables Eigen fast path
    uint64_t rl;                            // Random link (⎕RL): seeded from system at startup
    std::mt19937_64 rng;                    // Random number generator (seeded by rl)
    String* lx = nullptr;                   // Latent expression (⎕LX): interned in string_pool
    uint32_t sysvar_mask = SYSVAR_ALL;      // Enabled system variables (for sandboxing)
    bool optimizer_enabled = true;          // set false to disable static optimizer (for benchmarking)
    SpecializationBackend* dir_backend = nullptr;  // DIR specialization (CloningBackend)

    // Error state (ISO 13751 §11.4.4-11.4.5)
    int event_type[2] = {0, 0};             // {class, subclass} - 0 0 = no error
    String* event_message = nullptr;        // Error message (⎕EM) - interned in string_pool

    // Memory management
    Heap* heap;                          // Garbage-collected heap
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

    // High-level eval: parse and execute an expression
    Value* eval(const std::string& input);

    // Execute the machine until halt
    Value* execute();

public:
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

    // Format the error stack trace for display
    // Returns empty string if no error stack captured
    std::string format_stack_trace() const;

    // Throw an error: captures stack trace, creates ThrowErrorK, and pushes it
    // source: the continuation where the error originated (for location info)
    // msg: the error message (will be interned)
    // error_class: ISO 13751 error class (0=unclassified, 1=SYNTAX, 2=VALUE, 3=INDEX, 4=RANK, 5=LENGTH, 11=DOMAIN, etc.)
    // error_subclass: ISO 13751 error subclass (usually 0)
    void throw_error(const char* msg, Continuation* source,
                     int error_class, int error_subclass);

    // Trigger GC if needed
    void maybe_gc() {
        if (heap->should_gc()) {
            heap->collect(this);
        }
    }

private:
    // Initialize global environment with APL primitives and operators
    void init_globals();
};

} // namespace apl
