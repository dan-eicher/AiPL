// APLCompletion implementation

#include "completion.h"
#include "heap.h"

namespace apl {

void APLCompletion::mark(APLHeap* heap) {
    // Mark the value if present
    if (value) {
        heap->mark_value(value);
    }
    // Note: target is a const char* to interned strings, not a GC object
}

} // namespace apl
