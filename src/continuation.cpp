// Continuation implementations

#include "continuation.h"
#include "machine.h"

namespace apl {

// Forward declaration of APLHeap for now
// Will be implemented in Phase 1.6
class APLHeap;

// HaltK implementation
Value* HaltK::invoke(Machine* machine) {
    // Terminal continuation - halt the machine
    machine->halt();
    return machine->ctrl.value;
}

void HaltK::mark(APLHeap* heap) {
    // HaltK has no references to mark
    (void)heap;  // Unused
}

// ArgK implementation
Value* ArgK::invoke(Machine* machine) {
    // Set the argument value and continue with next continuation
    machine->ctrl.set_value(arg_value);

    if (next) {
        return next->invoke(machine);
    }

    // No next continuation - halt
    machine->halt();
    return arg_value;
}

void ArgK::mark(APLHeap* heap) {
    // Mark the argument value if heap exists
    if (heap && arg_value) {
        // Heap marking will be implemented in Phase 1.6
        // For now, this is a placeholder
    }

    // Mark values in next continuation
    if (next) {
        next->mark(heap);
    }
}

// FrameK implementation
Value* FrameK::invoke(Machine* machine) {
    // Function frame - set the current value and return
    // The return continuation will handle what happens next

    if (return_k) {
        return return_k->invoke(machine);
    }

    // No return continuation - halt
    machine->halt();
    return machine->ctrl.value;
}

void FrameK::mark(APLHeap* heap) {
    // Mark values in return continuation
    if (return_k) {
        return_k->mark(heap);
    }
}

} // namespace apl
