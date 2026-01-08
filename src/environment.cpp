// Environment implementation

#include "environment.h"
#include "heap.h"

namespace apl {

void Environment::mark(Heap* heap) {
    if (!heap) return;

    // Mark all keys and values in this environment
    for (auto& pair : bindings) {
        heap->mark(pair.first);   // Mark the String* key
        heap->mark(pair.second);  // Mark the Value*
    }

    // Mark parent environment
    heap->mark(parent);
}

} // namespace apl
