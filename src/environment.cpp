// Environment implementation

#include "environment.h"
#include "heap.h"

namespace apl {

void Environment::mark(APLHeap* heap) {
    if (!heap) return;

    // Mark all values in this environment
    for (auto& pair : bindings) {
        if (pair.second) {
            heap->mark_value(pair.second);
        }
    }

    // Mark parent environment's values
    if (parent) {
        parent->mark(heap);
    }
}

} // namespace apl
