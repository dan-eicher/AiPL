// Completion implementation

#include "completion.h"
#include "heap.h"

namespace apl {

void Completion::mark(Heap* heap) {
    // Mark the value if present
    heap->mark(value);
    // Note: target is a const char* to interned strings, not a GC object
}

} // namespace apl
