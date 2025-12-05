// Environment - Variable bindings for APL

#pragma once

#include "value.h"
#include "heap.h"
#include <unordered_map>
#include <string>

namespace apl {

// Environment - Maps names to values
// Now GC-managed for proper memory safety
class Environment : public GCObject {
private:
    // Only Heap can allocate/deallocate Environment objects
    friend class Heap;

    // Private new/delete operators enforce heap-only allocation
    void* operator new(size_t size) { return ::operator new(size); }
    void operator delete(void* ptr) { ::operator delete(ptr); }

public:
    // Parent environment for lexical scoping
    Environment* parent;

    // Variable bindings
    std::unordered_map<std::string, Value*> bindings;

    // Constructor
    Environment(Environment* p = nullptr) : GCObject(), parent(p) {}

    // Destructor
    ~Environment() override {
        // Don't delete values - they're managed by the heap
        // Don't delete parent - it's managed by the heap
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

    // Mark all values for GC (override from GCObject)
    void mark(Heap* heap) override;
};

} // namespace apl
