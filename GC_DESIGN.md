# GC Design: Everything Under GC Control

## The Fundamental Insight

**Problem:** User-defined APL functions are first-class values. A function body is a continuation graph. If Values can contain continuation graphs, then continuations must be GC-managed.

**Example:**
```apl
square ← {⍵ × ⍵}     ⍝ Define function
fn ← square          ⍝ Assign to another variable (function is a value!)
result ← fn 5        ⍝ Call via the copied reference
```

The continuation graph for `{⍵ × ⍵}` must survive as long as both `square` and `fn` reference it.

## Current Value System Gap

From `include/value.h:40`:
```cpp
union Data {
    double scalar;
    Eigen::MatrixXd* matrix;
    PrimitiveFn* function;      // ← C function pointer (built-ins)
    PrimitiveOp* op;
};
```

**Missing:** A way to store user-defined function bodies (continuation graphs)!

**Needed:**
```cpp
// Forward declaration
class Continuation;

union Data {
    double scalar;
    Eigen::MatrixXd* matrix;
    PrimitiveFn* primitive_fn;   // Built-in function (C pointer)
    Continuation* closure;        // User-defined function (continuation graph)
    PrimitiveOp* op;
};

// Updated ValueType
enum class ValueType {
    SCALAR,
    VECTOR,
    MATRIX,
    PRIMITIVE,   // Built-in function
    CLOSURE,     // User-defined function
    OPERATOR
};
```

## GC Ownership Model

If Values can reference Continuations, and Values are GC-managed, then **Continuations must also be GC-managed**.

### What Gets GC'd?

1. **Values** - already GC-managed in young/old generations
2. **Continuations** - NEW: must also be GC-managed
3. **Eigen::MatrixXd*** - already managed (deleted when Value is swept)

### GC Roots

From `heap.cpp:150-175`, current roots are:

```cpp
void APLHeap::mark_from_roots(Machine* machine) {
    // 1. Value in control register
    if (machine->ctrl.value) mark_value(machine->ctrl.value);

    // 2. Values referenced by continuations on kont_stack
    for (Continuation* k : machine->kont_stack) {
        if (k) k->mark(this);
    }

    // 3. Values in environment
    if (machine->env) machine->env->mark(this);
}
```

**Problem:** Continuations on `kont_stack` are NOT marked as GC objects themselves, only the Values they reference!

**Fix Needed:**
```cpp
void APLHeap::mark_from_roots(Machine* machine) {
    // 1. Value in control register
    if (machine->ctrl.value) mark_value(machine->ctrl.value);

    // 2. Continuations on kont_stack (they're objects too!)
    for (Continuation* k : machine->kont_stack) {
        if (k) mark_continuation(k);  // ← NEW: Mark the continuation itself
    }

    // 3. Values in environment
    if (machine->env) machine->env->mark(this);

    // 4. Continuations in function cache
    for (auto& pair : machine->function_cache) {
        if (pair.second) mark_continuation(pair.second);
    }
}
```

## Two-Heap Design

Since Continuations and Values are different types, we need separate heaps:

### Option 1: Unified GC Heap (Recommended)

Use a single heap with a type tag:

```cpp
enum class GCObjectType {
    VALUE,
    CONTINUATION
};

class GCObject {
public:
    GCObjectType gc_type;
    bool marked;
    bool in_old_generation;

    virtual ~GCObject() {}
    virtual void mark_references(APLHeap* heap) = 0;
};

// Value becomes a GCObject
class Value : public GCObject {
    // ... existing Value members ...

    void mark_references(APLHeap* heap) override {
        // Mark Values this Value references
        // e.g., if this is a CLOSURE, mark the continuation graph
        if (tag == ValueType::CLOSURE && data.closure) {
            heap->mark_continuation(data.closure);
        }
    }
};

// Continuation becomes a GCObject
class Continuation : public GCObject {
    // ... existing Continuation members ...

    void mark_references(APLHeap* heap) override {
        // Mark Values and Continuations this Continuation references
        // e.g., ApplyDyadicK might hold Value* left
        // e.g., all continuations have Continuation* next
    }
};

// Heap manages both types
class APLHeap {
    std::vector<GCObject*> young_objects;
    std::vector<GCObject*> old_objects;

    GCObject* allocate(GCObject* obj);
    void mark_object(GCObject* obj);
    void sweep();
};
```

**Pros:**
- Single unified GC
- Natural object graph (Values reference Continuations reference Values)
- Simpler mental model

**Cons:**
- Requires refactoring Value and Continuation to inherit from GCObject
- Virtual dispatch overhead (probably negligible)

### Option 2: Separate Heaps

Keep separate heaps but cross-reference:

```cpp
class APLHeap {
    // Value heap
    std::vector<Value*> young_values;
    std::vector<Value*> old_values;

    // Continuation heap
    std::vector<Continuation*> young_continuations;
    std::vector<Continuation*> old_continuations;

    Value* allocate_value(Value* v);
    Continuation* allocate_continuation(Continuation* k);

    void mark_value(Value* v);
    void mark_continuation(Continuation* k);
};
```

**Pros:**
- Simpler refactoring (less intrusive)
- Can tune GC separately for Values vs Continuations

**Cons:**
- Two parallel heaps to manage
- More complex marking logic
- Cache locality worse

## Reference Graph Examples

### Example 1: Simple Function

```apl
square ← {⍵ × ⍵}
```

**Object graph:**
```
Environment["square"] → Value(CLOSURE) → LiteralK(⍵) → ApplyDyadicK(×) → ...
                             ↑                                    ↑
                             └────── GC must keep alive ──────────┘
```

**GC marking:**
1. Mark environment (root)
2. Mark Value "square" (referenced by environment)
3. Mark Continuation graph (referenced by Value)
4. All reachable

### Example 2: Function Passed as Argument

```apl
apply ← {⍺ ⍵}        ⍝ Apply left function to right argument
square ← {⍵ × ⍵}
result ← square apply 5
```

**During `apply` execution:**
```
kont_stack: [
    FunctionCallK  (for apply)
]

ctrl.value = Value(CLOSURE) for square  ← Must not be swept!
                    ↓
            Continuation graph for {⍵ × ⍵}  ← Must not be swept!
```

**GC marking:**
1. Mark ctrl.value (root)
2. Mark the CLOSURE Value
3. Mark the Continuation graph referenced by CLOSURE
4. All reachable

### Example 3: Continuation Built During Parsing

```apl
Parsing: {⍵ × 2}
```

**Problem with current approach:**
```cpp
// Spirit semantic action builds:
Continuation* graph =
    new LookupK("⍵",                    // ← Allocated with 'new'
        new ApplyDyadicK(...,           // ← Not on kont_stack yet!
            new LiteralK(2.0, nullptr)  // ← Not tracked anywhere!
        )
    );

// If GC runs HERE, graph gets swept!
Value* fn = Value::from_closure(graph);  // ← Too late!
```

**Solution:** Allocate continuations through the GC heap:

```cpp
// Spirit semantic action:
Continuation* graph =
    machine->heap->allocate_continuation(
        new LookupK("⍵",
            machine->heap->allocate_continuation(
                new ApplyDyadicK(...,
                    machine->heap->allocate_continuation(
                        new LiteralK(2.0, nullptr)
                    )
                )
            )
        )
    );

// All continuations are now in young_objects, safe from GC
Value* fn = machine->heap->allocate_value(
    Value::from_closure(graph)
);
```

## Implementation Plan

### Step 1: Add Continuation Heap

```cpp
// In heap.h
class APLHeap {
public:
    // Value heaps (existing)
    std::vector<Value*> young_values;
    std::vector<Value*> old_values;

    // Continuation heaps (NEW)
    std::vector<Continuation*> young_continuations;
    std::vector<Continuation*> old_continuations;

    // Allocate continuation (NEW)
    Continuation* allocate_continuation(Continuation* k);

    // Mark continuation and its references (NEW)
    void mark_continuation(Continuation* k);
};
```

### Step 2: Update Value to Hold Closures

```cpp
// In value.h
enum class ValueType {
    SCALAR,
    VECTOR,
    MATRIX,
    PRIMITIVE,   // Built-in function (PrimitiveFn*)
    CLOSURE,     // User-defined function (Continuation*)
    OPERATOR
};

union Data {
    double scalar;
    Eigen::MatrixXd* matrix;
    PrimitiveFn* primitive_fn;
    Continuation* closure;  // NEW
    PrimitiveOp* op;
};

// In value.cpp
void Value::mark_references(APLHeap* heap) {
    if (tag == ValueType::CLOSURE && data.closure) {
        heap->mark_continuation(data.closure);
    }
}
```

### Step 3: Update Continuation::mark()

```cpp
// Continuation base class
class Continuation {
public:
    Continuation* next;

    // Mark this continuation as reachable
    virtual void mark_references(APLHeap* heap) {
        // Mark the next continuation in the chain
        if (next) {
            heap->mark_continuation(next);
        }

        // Subclasses override to mark their specific references
    }
};

// LiteralK doesn't reference any GC objects
class LiteralK : public Continuation {
    double value;  // Not a GC object

    void mark_references(APLHeap* heap) override {
        Continuation::mark_references(heap);  // Just mark next
    }
};

// ApplyDyadicK references a Value
class ApplyDyadicK : public Continuation {
    Value* left;  // GC object!

    void mark_references(APLHeap* heap) override {
        if (left) heap->mark_value(left);
        Continuation::mark_references(heap);  // Mark next
    }
};
```

### Step 4: Update GC Marking

```cpp
// In heap.cpp
void APLHeap::mark_continuation(Continuation* k) {
    if (!k) return;
    if (k->marked) return;  // Already marked

    k->marked = true;

    // Mark objects this continuation references
    k->mark_references(this);
}

void APLHeap::mark_from_roots(Machine* machine) {
    // Mark Values
    if (machine->ctrl.value) mark_value(machine->ctrl.value);

    // Mark Continuations on kont_stack
    for (Continuation* k : machine->kont_stack) {
        if (k) mark_continuation(k);
    }

    // Mark Continuations in function_cache
    for (auto& pair : machine->function_cache) {
        if (pair.second) mark_continuation(pair.second);
    }

    // Mark Values in environment
    if (machine->env) machine->env->mark(this);
}
```

### Step 5: Update Sweep

```cpp
void APLHeap::sweep() {
    // Sweep Values
    sweep_vector(young_values);
    sweep_vector(old_values);

    // Sweep Continuations
    sweep_vector(young_continuations);
    sweep_vector(old_continuations);
}

template<typename T>
void APLHeap::sweep_vector(std::vector<T*>& objects) {
    auto it = objects.begin();
    while (it != objects.end()) {
        T* obj = *it;
        if (!obj->marked) {
            delete obj;
            it = objects.erase(it);
        } else {
            obj->marked = false;  // Clear mark for next GC
            ++it;
        }
    }
}
```

### Step 6: Update Parsing to Use GC Allocation

```cpp
// In Spirit semantic actions:
auto const term_def =
    double_[([](auto& ctx) {
        Machine* machine = get_machine(ctx);  // Get machine from context

        // Allocate through GC heap!
        Continuation* lit = machine->heap->allocate_continuation(
            new LiteralK(_attr(ctx), nullptr)
        );

        _val(ctx) = lit;
    })];
```

### Step 7: Remove Manual Delete

```cpp
// In machine.cpp
Value* Machine::execute() {
    while (!kont_stack.empty()) {
        Continuation* k = pop_kont();
        k->invoke(this);
        // DON'T delete k!  ← GC handles it now
    }
}
```

## Migration Strategy

1. **Add continuation heap to APLHeap** (separate young/old vectors)
2. **Add allocate_continuation() method**
3. **Update mark_from_roots() to mark continuations**
4. **Update sweep() to sweep continuations**
5. **Update Value to hold CLOSURE type**
6. **Update parsing to allocate through heap**
7. **Remove manual delete calls**
8. **Test thoroughly with GC stress tests**

## Testing

```cpp
TEST(GC, ContinuationSurvival) {
    Machine m;

    // Create a function value
    Continuation* body = m.heap->allocate_continuation(
        new LiteralK(42.0, nullptr)
    );

    Value* fn = m.heap->allocate_value(
        Value::from_closure(body)
    );

    // Store in environment (GC root)
    m.env->define("f", fn);

    // Force GC
    m.heap->gc_threshold = 0;
    m.heap->collect(&m);

    // Function should still be alive
    Value* retrieved = m.env->lookup("f");
    ASSERT_NE(retrieved, nullptr);
    ASSERT_EQ(retrieved->tag, ValueType::CLOSURE);
    ASSERT_EQ(retrieved->data.closure, body);
}

TEST(GC, ContinuationChainSurvival) {
    Machine m;

    // Build a continuation chain
    Continuation* chain = m.heap->allocate_continuation(
        new LiteralK(1.0,
            m.heap->allocate_continuation(
                new LiteralK(2.0,
                    m.heap->allocate_continuation(
                        new LiteralK(3.0, nullptr)
                    )
                )
            )
        )
    );

    // Only store the head in a Value
    Value* fn = m.heap->allocate_value(Value::from_closure(chain));
    m.env->define("f", fn);

    // Force GC
    m.heap->collect(&m);

    // Entire chain should survive (reachable from head)
    Value* retrieved = m.env->lookup("f");
    ASSERT_NE(retrieved->data.closure->next, nullptr);
    ASSERT_NE(retrieved->data.closure->next->next, nullptr);
}
```

## Key Insight

**The task list was wrong.** It said continuations are manually managed, but that's incompatible with functions as first-class values. Everything must be under GC control for the object graph to be safe.

The correct design:
- **Values are GC-managed** ✓ (already true)
- **Continuations are GC-managed** ✓ (need to add)
- **Both use same generational GC** ✓ (unified design)
- **No manual new/delete** ✓ (except initial construction before heap allocation)
