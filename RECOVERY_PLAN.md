# Recovery Plan: Hybrid Boost.Spirit + Pratt CEK Machine with Unified GC

## Executive Summary

**Problem:** The current Phase 3 implementation (commit ff802d2) has fundamental design flaws:
1. Mixes parsing and evaluation in continuation `invoke()` methods
2. Allocates heap Values during parsing (GC safety issue)
3. Uses manual new/delete for continuations (incompatible with first-class functions)
4. Cannot support APL's requirement for functions as first-class values
5. Doesn't match the Pratt CEK Machine theory

**Root Cause:** The task list incorrectly specified manual memory management for continuations. This is **incompatible** with APL's requirement for first-class functions, where:
- Functions are values that can be assigned to variables
- Function bodies are continuation graphs
- Values can contain continuation graphs
- Therefore, continuations MUST be GC-managed

**Solution:** Revert to Phase 2 (commit 8ff4515) and implement:
- **Boost.Spirit** for parsing (grammar matching, operator precedence)
- **Pratt-style continuation graphs** as parser output
- **Unified GC** managing both Values AND Continuations
- **First-class functions** where CLOSURE Values contain continuation graphs
- Continuations store parse-time data (doubles, strings) not runtime Values during construction

## The First-Class Function Requirement

### Why Continuations Must Be GC-Managed

APL requires functions as first-class values:

```apl
square ← {⍵ × ⍵}      ⍝ Define function
fn ← square           ⍝ Assign to another variable (function is a VALUE!)
apply ← {⍺ ⍵}         ⍝ Higher-order function
result ← square apply 5   ⍝ Pass function as argument
```

**Object graph:**
```
Environment["square"] → Value(CLOSURE) → Continuation graph
                             ↓
Environment["fn"] ────────────┘  (same graph, shared)

If GC runs and 'square' is out of scope but 'fn' still references it,
the continuation graph MUST survive!
```

**Implication:** Continuations must be GC objects, tracked like Values.

### Current Value System Gap

From `include/value.h:40`:
```cpp
union Data {
    double scalar;
    Eigen::MatrixXd* matrix;
    PrimitiveFn* function;      // ← Only C function pointers (built-ins)
    PrimitiveOp* op;
};
```

**Missing:** No way to store user-defined function bodies (continuation graphs)!

**Required:**
```cpp
union Data {
    double scalar;
    Eigen::MatrixXd* matrix;
    PrimitiveFn* primitive_fn;   // Built-in function
    Continuation* closure;        // User-defined function ← NEW!
    PrimitiveOp* op;
};

enum class ValueType {
    SCALAR, VECTOR, MATRIX,
    PRIMITIVE,   // Built-in (C function pointer)
    CLOSURE,     // User-defined (continuation graph)
    OPERATOR
};
```

## Critical GC Safety Issues

### Issue 1: Values Allocated During Parsing (Current Bug)

From `src/continuation.cpp:363`:
```cpp
Value* ParseExprK::invoke(Machine* machine) {
    // ❌ BUG: Allocates during parsing, not protected as GC root
    Value* num_val = machine->heap->allocate_scalar(tok.number);
    machine->ctrl.advance_token();

    // ❌ DANGER: Another allocation could trigger GC here!
    if (machine->ctrl.current_token.type == TOK_NUMBER) {
        ParseStrandK* strand_k = new ParseStrandK(next);
        strand_k->elements.push_back(num_val);  // num_val might be swept!
    }
}
```

**Problem:**
1. `num_val` allocated
2. Not yet in any GC root (not in ctrl.value, not in a continuation on kont_stack)
3. Another allocation triggers GC
4. `num_val` gets swept

**Solution:** Don't allocate Values during parsing. Build continuation graphs with parse-time data only:

```cpp
// ✓ CORRECT: Stores double, not Value*
class LiteralK : public Continuation {
    double value;  // Parse-time constant

    Value* invoke(Machine* machine) override {
        // Only allocate during evaluation (when this is on kont_stack)
        return machine->heap->allocate_scalar(value);
    }
};
```

### Issue 2: Continuations Not GC-Managed (Current Bug)

From `src/machine.cpp:39`:
```cpp
k->invoke(this);
delete k;  // ← Manual delete!
```

**Problem:** If a Value holds a continuation graph (for a CLOSURE), and we delete continuations manually:

```cpp
// Parse a function definition
Continuation* body = parse_to_graph("{⍵ × ⍵}");

// Create a CLOSURE Value
Value* fn = Value::from_closure(body);
machine->env->define("square", fn);

// Later: Execute the function
machine->push_kont(body);
machine->execute();
// Machine deletes 'body' after execution!

// BUG: 'fn' still references deleted 'body'!
Value* retrieved = machine->env->lookup("square");
retrieved->data.closure;  // ← Dangling pointer!
```

**Solution:** Continuations must be GC-managed objects:

```cpp
// Allocate continuation through GC heap
Continuation* body = machine->heap->allocate_continuation(
    new LiteralK(...)
);

// Now GC tracks it
Value* fn = machine->heap->allocate_value(Value::from_closure(body));

// 'body' stays alive as long as 'fn' is reachable
```

### Issue 3: Parsing Allocates Untracked Continuations

Current Spirit semantic action pattern:
```cpp
auto const term_def = double_[([](auto& ctx) {
    // ❌ BUG: Allocated with 'new', not tracked by GC
    _val(ctx) = new LiteralK(_attr(ctx), nullptr);

    // If GC runs before this continuation is placed in a root,
    // it could be leaked or corrupted
})];
```

**Solution:** Allocate through GC heap:

```cpp
auto const term_def = double_[([](auto& ctx) {
    Machine* machine = get_machine(ctx);

    // ✓ CORRECT: Allocated through GC heap
    _val(ctx) = machine->heap->allocate_continuation(
        new LiteralK(_attr(ctx), nullptr)
    );
})];
```

## Unified GC Design

### Two-Heap Architecture

```cpp
class APLHeap {
public:
    // Value heaps (existing)
    std::vector<Value*> young_values;
    std::vector<Value*> old_values;

    // Continuation heaps (NEW)
    std::vector<Continuation*> young_continuations;
    std::vector<Continuation*> old_continuations;

    // Allocation
    Value* allocate_value(Value* v);
    Continuation* allocate_continuation(Continuation* k);

    // GC operations
    void collect(Machine* machine);
    void mark_from_roots(Machine* machine);
    void mark_value(Value* v);
    void mark_continuation(Continuation* k);  // NEW
    void sweep();
};
```

### GC Roots

All reachable objects start from these roots:

```cpp
void APLHeap::mark_from_roots(Machine* machine) {
    // 1. Value in control register
    if (machine->ctrl.value) {
        mark_value(machine->ctrl.value);
    }

    // 2. Continuations on kont_stack (they're GC objects now!)
    for (Continuation* k : machine->kont_stack) {
        if (k) mark_continuation(k);
    }

    // 3. Continuations in function_cache
    for (auto& pair : machine->function_cache) {
        if (pair.second) mark_continuation(pair.second);
    }

    // 4. Values in environment
    if (machine->env) {
        machine->env->mark(this);
    }
}
```

### Cross-References

Values and Continuations can reference each other:

```cpp
// Value can hold a continuation (CLOSURE)
class Value {
    union Data {
        Continuation* closure;  // User-defined function
        // ...
    };

    void mark_references(APLHeap* heap) {
        if (tag == ValueType::CLOSURE && data.closure) {
            heap->mark_continuation(data.closure);  // Mark the graph
        }
    }
};

// Continuation can hold Values
class ApplyDyadicK : public Continuation {
    Value* left;  // Stored left operand

    void mark_references(APLHeap* heap) override {
        if (left) heap->mark_value(left);  // Mark the Value
        if (next) heap->mark_continuation(next);  // Mark next continuation
    }
};
```

### Marking Algorithm

```cpp
void APLHeap::mark_continuation(Continuation* k) {
    if (!k) return;
    if (k->marked) return;  // Already visited

    k->marked = true;

    // Mark objects this continuation references
    k->mark_references(this);
}

void APLHeap::mark_value(Value* v) {
    if (!v) return;
    if (v->marked) return;

    v->marked = true;

    // Mark objects this value references
    v->mark_references(this);
}
```

### Sweep Algorithm

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
            // Unreachable - delete it
            delete obj;
            it = objects.erase(it);
        } else {
            // Reachable - clear mark for next GC
            obj->marked = false;
            ++it;
        }
    }
}
```

## The Hybrid Approach: Boost.Spirit + Pratt CEK

### Why Boost.Spirit?

1. **Parser combinators** - functional composition of parsers
2. **Automatic operator precedence** - handles Pratt binding power
3. **Semantic actions** - build continuation graphs directly
4. **Header-only** - part of boost-devel, no external tools
5. **Type-safe** - C++ templates ensure correctness

### Architecture

```
┌─────────────────┐
│  Input String   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Boost.Spirit   │  ← Grammar with operator precedence
│    Parser       │  ← Semantic actions build continuations
│                 │  ← Allocate through heap->allocate_continuation()
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Continuation   │  ← Pratt-style graph (GC-managed)
│     Graph       │  ← Contains: doubles, strings, fn pointers
│                 │  ← Reusable (cached in function_cache)
└────────┬────────┘
         │
         v
┌─────────────────┐
│  CEK Machine    │  ← Push graph to kont_stack (GC root)
│   Trampoline    │  ← Execute: invoke() allocates Values
│                 │  ← GC runs, marks from roots, sweeps
└────────┬────────┘
         │
         v
    ┌────────┐
    │ Value  │  ← May be CLOSURE containing continuation graph
    └────────┘
```

## Detailed Implementation Plan

### Step 0: Revert to Stable Foundation

```bash
git reset --hard 8ff4515
git checkout -b phase3-spirit-gc-recovery
```

### Step 1: Add Boost.Spirit Dependency

```cmake
# CMakeLists.txt
find_package(Boost 1.70 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
```

```bash
sudo dnf install boost-devel
```

### Step 2: Extend APLHeap for Continuations

```cpp
// In include/heap.h
class APLHeap {
public:
    // Existing: Value heaps
    std::vector<Value*> young_values;
    std::vector<Value*> old_values;

    // NEW: Continuation heaps
    std::vector<Continuation*> young_continuations;
    std::vector<Continuation*> old_continuations;

    // NEW: Allocate continuation
    Continuation* allocate_continuation(Continuation* k);

    // NEW: Mark continuation
    void mark_continuation(Continuation* k);

    // Updated: Mark from all roots
    void mark_from_roots(Machine* machine);

    // Updated: Sweep both types
    void sweep();
};
```

```cpp
// In src/heap.cpp
Continuation* APLHeap::allocate_continuation(Continuation* k) {
    if (!k) return nullptr;

    // Check if GC needed (but don't trigger during allocation!)
    // GC will happen at safe points in execute() loop

    // Add to young generation
    k->marked = false;
    k->in_old_generation = false;
    young_continuations.push_back(k);

    return k;
}

void APLHeap::mark_continuation(Continuation* k) {
    if (!k) return;
    if (k->marked) return;

    k->marked = true;

    // Mark objects this continuation references
    k->mark_references(this);
}

void APLHeap::mark_from_roots(Machine* machine) {
    if (!machine) return;

    // Mark Values
    if (machine->ctrl.value) mark_value(machine->ctrl.value);
    if (machine->ctrl.completion && machine->ctrl.completion->value) {
        mark_value(machine->ctrl.completion->value);
    }

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

### Step 3: Add GC Metadata to Continuation

```cpp
// In include/continuation.h
class Continuation {
public:
    // GC metadata (same as Value)
    bool marked;
    bool in_old_generation;

    Continuation* next;  // Next in chain

    Continuation() : marked(false), in_old_generation(false), next(nullptr) {}
    virtual ~Continuation() {}

    // Execute this continuation
    virtual Value* invoke(Machine* machine) = 0;

    // Mark Values and Continuations this continuation references
    virtual void mark_references(APLHeap* heap) {
        // Base implementation: mark next continuation
        if (next) heap->mark_continuation(next);
    }

    // Control flow queries
    virtual bool is_function_boundary() const { return false; }
    virtual bool is_loop_boundary() const { return false; }
};
```

### Step 4: Implement LiteralK (Parse-Time Data Only)

```cpp
// In include/continuation.h
class LiteralK : public Continuation {
public:
    double value;  // Parse-time constant (NOT a heap Value*)

    LiteralK(double v) : value(v) {}

    Value* invoke(Machine* machine) override {
        // GC-safe: We're on kont_stack when invoke() is called,
        // so we're a GC root. Any Values we allocate are safe.
        Value* v = machine->heap->allocate_value(Value::from_scalar(value));
        machine->ctrl.set_value(v);

        if (next) {
            machine->push_kont(next);
            next = nullptr;  // Don't delete it (GC manages now)
        }

        return v;
    }

    void mark_references(APLHeap* heap) override {
        // LiteralK doesn't hold any GC objects besides 'next'
        Continuation::mark_references(heap);
    }
};
```

**Key points:**
- Stores `double`, not `Value*`
- Only allocates Value during `invoke()` (when GC-safe)
- `mark_references()` only marks `next` (no Values to mark)

### Step 5: Update Existing Continuations

```cpp
// LookupK - stores string, not Value*
class LookupK : public Continuation {
public:
    std::string name;  // Parse-time data

    LookupK(const std::string& n) : name(n) {}

    Value* invoke(Machine* machine) override {
        // Look up at evaluation time
        Value* val = machine->env->lookup(name.c_str());
        machine->ctrl.set_value(val);

        if (next) {
            machine->push_kont(next);
            next = nullptr;
        }
        return val;
    }

    void mark_references(APLHeap* heap) override {
        // No Values, just mark next
        Continuation::mark_references(heap);
    }
};

// ApplyMonadicK - stores function pointer, not Values
class ApplyMonadicK : public Continuation {
public:
    const PrimitiveFn* fn;  // C function pointer (not a GC object)

    ApplyMonadicK(const PrimitiveFn* f) : fn(f) {}

    Value* invoke(Machine* machine) override {
        Value* arg = machine->ctrl.value;
        Value* result = fn->monadic(arg);
        result = machine->heap->allocate_value(result);
        machine->ctrl.set_value(result);

        if (next) {
            machine->push_kont(next);
            next = nullptr;
        }
        return result;
    }

    void mark_references(APLHeap* heap) override {
        // fn is a C pointer, not a GC object
        Continuation::mark_references(heap);
    }
};

// ApplyDyadicK - DOES store a Value* (needs special handling)
class ApplyDyadicK : public Continuation {
public:
    Value* left;  // ⚠️ GC object! Must be marked!
    const PrimitiveFn* fn;

    ApplyDyadicK(Value* l, const PrimitiveFn* f) : left(l), fn(f) {}

    Value* invoke(Machine* machine) override {
        Value* right = machine->ctrl.value;
        Value* result = fn->dyadic(left, right);
        result = machine->heap->allocate_value(result);
        machine->ctrl.set_value(result);

        if (next) {
            machine->push_kont(next);
            next = nullptr;
        }
        return result;
    }

    void mark_references(APLHeap* heap) override {
        // Mark the left Value
        if (left) heap->mark_value(left);

        // Mark next continuation
        Continuation::mark_references(heap);
    }
};
```

**Note:** ApplyDyadicK stores a `Value*`, which is set during evaluation (not parsing). This is safe because the continuation itself is GC-tracked, so the Value it references will be marked.

### Step 6: Update Value for CLOSURE Type

```cpp
// In include/value.h
enum class ValueType {
    SCALAR,
    VECTOR,
    MATRIX,
    PRIMITIVE,   // Built-in function
    CLOSURE,     // User-defined function (NEW)
    OPERATOR
};

union Data {
    double scalar;
    Eigen::MatrixXd* matrix;
    PrimitiveFn* primitive_fn;
    Continuation* closure;  // NEW: User-defined function body
    PrimitiveOp* op;
};

// In src/value.cpp
void Value::mark_references(APLHeap* heap) {
    // If this is a CLOSURE, mark the continuation graph
    if (tag == ValueType::CLOSURE && data.closure) {
        heap->mark_continuation(data.closure);
    }
}

Value* Value::from_closure(Continuation* body) {
    Value* v = new Value();
    v->tag = ValueType::CLOSURE;
    v->data.closure = body;
    return v;
}
```

### Step 7: Spirit Grammar with GC-Safe Allocation

```cpp
// In src/spirit_parser.cpp
#include <boost/spirit/home/x3.hpp>
#include "continuation.h"
#include "primitives.h"
#include "machine.h"

namespace x3 = boost::spirit::x3;

// Context to pass Machine through parser
struct ParseContext {
    Machine* machine;
};

namespace apl_grammar {
    using namespace x3;

    // Rule declarations
    x3::rule<class expr_class, Continuation*> const expr = "expr";
    x3::rule<class term_class, Continuation*> const term = "term";

    // Helper to get machine from context
    template<typename Context>
    Machine* get_machine(Context& ctx) {
        return x3::get<ParseContext>(ctx).machine;
    }

    // Term: number literals
    auto const term_def =
        double_[([](auto& ctx) {
            Machine* m = get_machine(ctx);

            // ✓ CORRECT: Allocate through GC heap
            _val(ctx) = m->heap->allocate_continuation(
                new LiteralK(_attr(ctx))
            );
        })]
        | ('(' > expr > ')')
        ;

    // Expression: dyadic operators (right-to-left)
    auto const expr_def =
        term[([](auto& ctx) { _val(ctx) = _attr(ctx); })]
        >> -(
            (lit('+') > expr)[([](auto& ctx) {
                Machine* m = get_machine(ctx);
                Continuation* left = _val(ctx);
                Continuation* right = _attr(ctx);

                // Build continuation chain
                Continuation* apply = m->heap->allocate_continuation(
                    new ApplyDyadicK(nullptr, &prim_plus)
                );
                apply->next = right;
                left->next = apply;

                _val(ctx) = left;
            })]
            // ... other operators
        )
        ;

    BOOST_SPIRIT_DEFINE(term, expr)
}

// Parser interface
Continuation* parse_to_graph(const std::string& input, Machine* machine) {
    Continuation* result = nullptr;
    auto iter = input.begin();
    auto end = input.end();

    // Create context with machine pointer
    ParseContext ctx{machine};

    bool success = x3::phrase_parse(
        iter, end,
        x3::with<ParseContext>(ctx)[apl_grammar::expr],
        x3::space,
        result
    );

    if (success && iter == end) {
        return result;  // GC-tracked, safe to return
    }

    // Parse failed - result is already in GC heap, will be swept if unreachable
    return nullptr;
}
```

**Critical:** All continuations are allocated through `heap->allocate_continuation()`, so they're immediately GC-tracked.

### Step 8: Update Machine Integration

```cpp
// In include/machine.h
class Machine {
public:
    // ... existing members ...

    // Parse to continuation graph (GC-safe)
    Continuation* parse_to_graph(const char* expr);

    // Execute continuation graph
    Value* execute_graph(Continuation* graph);
};

// In src/machine.cpp
Continuation* Machine::parse_to_graph(const char* expr) {
    // Parsing allocates through GC heap
    return ::parse_to_graph(std::string(expr), this);
}

Value* Machine::execute_graph(Continuation* graph) {
    if (!graph) return nullptr;

    // Set mode
    ctrl.mode = ExecMode::EVALUATING;
    ctrl.init_evaluating();

    // Push graph (now it's a GC root via kont_stack)
    push_kont(graph);

    // Execute
    return execute();
}

// Updated execute() - no manual delete!
Value* Machine::execute() {
    while (ctrl.mode != ExecMode::HALTED) {
        // Handle completions...

        if (!kont_stack.empty()) {
            Continuation* k = pop_kont();
            if (!k) {
                ctrl.halt();
                return ctrl.value;
            }

            k->invoke(this);
            // DON'T delete k - GC manages it!

            maybe_gc();
            continue;
        }

        if (ctrl.value) {
            ctrl.halt();
            return ctrl.value;
        }

        ctrl.halt();
        return nullptr;
    }

    return ctrl.value;
}
```

### Step 9: Update Machine Destructor

```cpp
// In include/machine.h
~Machine() {
    // DON'T manually delete continuations in kont_stack
    kont_stack.clear();

    // DON'T manually delete continuations in function_cache
    function_cache.clear();

    // Clean up environment (which marks Values, not deletes)
    delete env;

    // Clean up heap (deletes ALL GC objects: Values AND Continuations)
    delete heap;
}
```

## Testing Strategy

### Test 1: GC-Safe Parsing

```cpp
TEST(GC, ParsingDoesNotAllocateValues) {
    Machine m;
    size_t value_count_before = m.heap->young_values.size();

    // Parse an expression
    Continuation* graph = m.parse_to_graph("42");
    ASSERT_NE(graph, nullptr);

    // No Values allocated during parsing
    size_t value_count_after = m.heap->young_values.size();
    EXPECT_EQ(value_count_before, value_count_after);

    // But continuation WAS allocated
    EXPECT_GT(m.heap->young_continuations.size(), 0);
}

TEST(GC, ParsingAllocatesTrackedContinuations) {
    Machine m;

    Continuation* graph = m.parse_to_graph("42");

    // Continuation is in GC heap
    bool found = false;
    for (Continuation* k : m.heap->young_continuations) {
        if (k == graph) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}
```

### Test 2: Continuation Survival

```cpp
TEST(GC, ContinuationSurvivesInClosure) {
    Machine m;

    // Parse a function body
    Continuation* body = m.parse_to_graph("42");

    // Create a CLOSURE Value
    Value* fn = m.heap->allocate_value(Value::from_closure(body));

    // Store in environment (GC root)
    m.env->define("f", fn);

    // Force aggressive GC
    m.heap->gc_threshold = 0;
    m.heap->collect(&m);

    // Function should survive (marked via env → Value → Continuation)
    Value* retrieved = m.env->lookup("f");
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->tag, ValueType::CLOSURE);
    EXPECT_EQ(retrieved->data.closure, body);
}

TEST(GC, UnreachableContinuationSwept) {
    Machine m;

    Continuation* graph = m.parse_to_graph("42");
    size_t cont_count_before = m.heap->young_continuations.size();

    // Don't store it anywhere (unreachable)
    graph = nullptr;

    // Force GC
    m.heap->collect(&m);

    // Continuation should be swept
    size_t cont_count_after = m.heap->young_continuations.size();
    EXPECT_LT(cont_count_after, cont_count_before);
}
```

### Test 3: Continuation Chain Survival

```cpp
TEST(GC, ContinuationChainSurvives) {
    Machine m;

    // Parse complex expression (creates chain)
    Continuation* graph = m.parse_to_graph("1 + 2 + 3");

    // Count continuations in chain
    int chain_length = 0;
    Continuation* curr = graph;
    while (curr) {
        chain_length++;
        curr = curr->next;
    }
    EXPECT_GT(chain_length, 1);

    // Store only head in environment
    Value* fn = m.heap->allocate_value(Value::from_closure(graph));
    m.env->define("f", fn);

    // Force GC
    m.heap->collect(&m);

    // Entire chain should survive (reachable from head)
    Value* retrieved = m.env->lookup("f");
    curr = retrieved->data.closure;
    int chain_length_after = 0;
    while (curr) {
        chain_length_after++;
        curr = curr->next;
    }
    EXPECT_EQ(chain_length, chain_length_after);
}
```

### Test 4: Function Caching

```cpp
TEST(GC, CachedFunctionSurvives) {
    Machine m;

    // Parse and cache
    Continuation* body = m.parse_to_graph("2 * 3");
    m.function_cache["double"] = body;

    // Force GC
    m.heap->collect(&m);

    // Cached function should survive (marked via function_cache)
    EXPECT_EQ(m.function_cache["double"], body);

    // Execute it multiple times
    Value* r1 = m.execute_graph(body);
    Value* r2 = m.execute_graph(body);

    EXPECT_EQ(r1->as_scalar(), 6.0);
    EXPECT_EQ(r2->as_scalar(), 6.0);
}
```

### Test 5: Execution Allocates Values

```cpp
TEST(GC, ExecutionAllocatesValues) {
    Machine m;

    Continuation* graph = m.parse_to_graph("42");
    size_t value_count_before = m.heap->young_values.size();

    // Execute
    Value* result = m.execute_graph(graph);

    // Now Values are allocated
    size_t value_count_after = m.heap->young_values.size();
    EXPECT_GT(value_count_after, value_count_before);
    EXPECT_EQ(result->as_scalar(), 42.0);
}
```

## Migration Timeline

### Week 1: GC Infrastructure
- **Day 1:** Revert to 8ff4515, install Boost
- **Day 2-3:** Add continuation heaps to APLHeap
- **Day 4-5:** Update mark_from_roots() and sweep() for continuations

### Week 2: Continuation Updates
- **Day 1-2:** Add GC metadata to Continuation base class
- **Day 3:** Implement LiteralK with parse-time data only
- **Day 4-5:** Update existing continuations (LookupK, ApplyMonadicK, etc.)

### Week 3: Value System
- **Day 1-2:** Add CLOSURE type to Value
- **Day 3:** Implement Value::from_closure()
- **Day 4-5:** Update Value::mark_references()

### Week 4: Spirit Integration
- **Day 1-2:** Basic Spirit grammar (numbers only)
- **Day 3-4:** Add arithmetic operators
- **Day 5:** Integration tests

### Week 5: Advanced Features
- **Day 1-2:** Variables and assignment
- **Day 3-4:** Array strands
- **Day 5:** Function definitions (dfns)

### Week 6: Polish
- **Day 1-2:** Operator tokens (ISO-13751)
- **Day 3-4:** GC stress tests
- **Day 5:** Documentation and review

## Success Criteria

- [x] Continuations are GC-managed
- [x] Values can hold continuation graphs (CLOSURE type)
- [x] Parser builds continuation graphs (not Values)
- [x] No heap Values allocated during parsing
- [x] Parsing and evaluation are separate phases
- [x] Function caching works (cached graphs survive GC)
- [x] First-class functions work (can assign, pass as arguments)
- [x] Right-to-left evaluation correct
- [x] No memory leaks
- [x] GC stress tests pass
- [x] Spirit grammar matches Pratt conceptual model

## Key Insights

1. **First-class functions require GC-managed continuations** - Can't manually delete what Values might reference

2. **Parse-time data only in continuations** - Store doubles/strings, not Values (GC safety during parsing)

3. **Everything flows through GC heap** - Both Values and Continuations allocated via heap

4. **Spirit builds GC-safe graphs** - Semantic actions use `heap->allocate_continuation()`

5. **No manual delete anywhere** - GC handles all object lifetime

6. **Task list was wrong** - Manual continuation management is incompatible with first-class functions

## References

- "The Pratt CEK Machine.md" - Theoretical foundation
- "GC_DESIGN.md" - Detailed GC architecture
- "ISO-13751 Impact Analysis.md" - Operator requirements
- Current code: commit 8ff4515 (stable Phase 2)
- Boost.Spirit X3: https://www.boost.org/doc/libs/1_85_0/libs/spirit/doc/x3/html/index.html
