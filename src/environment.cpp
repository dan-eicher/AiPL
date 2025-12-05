// Environment implementation

#include "environment.h"
#include "heap.h"

namespace apl {

void Environment::mark(Heap* heap) {
    if (!heap) return;

    // Mark all values in this environment
    for (auto& pair : bindings) {
        if (pair.second) {
            heap->mark_value(pair.second);
        }
    }

    // Mark parent environment (now GC-managed)
    if (parent && !parent->marked) {
        parent->marked = true;
        parent->mark(heap);
    }
}

} // namespace apl
