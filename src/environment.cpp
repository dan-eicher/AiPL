// Environment implementation

#include "environment.h"
#include "heap.h"

namespace apl {

void Environment::mark(Heap* heap) {
    if (!heap) return;

    // Mark all values in this environment
    for (auto& pair : bindings) {
        heap->mark(pair.second);
    }

    // Mark parent environment
    heap->mark(parent);
}

} // namespace apl
