// Environment - Variable bindings for APL

#pragma once

#include "value.h"
#include <unordered_map>
#include <string>

namespace apl {

// Environment - Maps names to values
// Simple implementation for Phase 1; will be expanded later
class Environment {
public:
    // Parent environment for lexical scoping
    Environment* parent;

    // Variable bindings
    std::unordered_map<std::string, Value*> bindings;

    // Constructor
    Environment(Environment* p = nullptr) : parent(p) {}

    // Destructor
    ~Environment() {
        // Don't delete values - they're managed by the heap
        // Don't delete parent - it's managed externally
    }

    // Lookup a variable
    Value* lookup(const char* name) {
        auto it = bindings.find(name);
        if (it != bindings.end()) {
            return it->second;
        }

        // Check parent environment
        if (parent) {
            return parent->lookup(name);
        }

        return nullptr;  // Not found
    }

    // Define or update a variable
    void define(const char* name, Value* value) {
        bindings[name] = value;
    }

    // Update existing variable (searches parent chain)
    bool update(const char* name, Value* value) {
        auto it = bindings.find(name);
        if (it != bindings.end()) {
            it->second = value;
            return true;
        }

        if (parent) {
            return parent->update(name, value);
        }

        return false;  // Not found
    }

    // Mark all values for GC
    void mark(class APLHeap* heap);
};

// Initialize the global environment with built-in primitives
void init_global_environment(Environment* env);

} // namespace apl
