// Control - Control register for CEK machine

#pragma once

#include "value.h"

namespace apl {

// Control register - holds the current evaluation result
// In a pure CEK machine, this is the only state the Control register needs
class Control {
public:
    Value* value;  // Current evaluation result

    // Constructor
    Control() : value(nullptr) {}

    // No destructor needed - value is GC-managed

    // Set value
    void set_value(Value* v) {
        value = v;
    }
};

} // namespace apl
