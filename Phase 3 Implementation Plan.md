# Phase 3 Implementation Plan: Lemon Parser → Continuation Graphs

## Overview
Use Lemon parser generator to parse Grammar G2 and **directly build continuation graphs** in the grammar actions (no AST intermediate). This cleanly separates parsing from evaluation while building reusable continuation graphs.

## Key Architectural Decision

**Approach:** Lemon parser → Continuation graphs (direct, no AST intermediate)

**Rationale:**
- Clean separation between parsing and evaluation
- Grammar actions build Continuation* directly
- No need for AST intermediate representation
- Graphs are reusable and cacheable
- Matches industry practice while staying true to CEK machine design

## The Fundamental Problem We're Fixing

The broken Phase 3 implementation (now reverted) mixed parsing with immediate evaluation:

**Before (broken):**
```cpp
Value* ParseExprK::invoke(Machine* machine) {
    // ❌ WRONG: Allocates heap values during parsing
    Value* num_val = machine->heap->allocate_scalar(tok.number);
    // ❌ WRONG: Executes immediately during parsing
    if (next) return next->invoke(machine);
    return num_val;
}
```

**After (correct):**
```lemon
// Grammar action builds continuation (doesn't execute)
term(R) ::= NUMBER(N). {
    R = new LiteralK(N.number, nullptr);  // ✓ Build graph node
}
```

Then separately:
```cpp
// Execution happens in EVALUATING mode only
Value* LiteralK::invoke(Machine* machine) {
    return machine->heap->allocate_scalar(this->value);
}
```

## Prerequisites
- Phase 2 complete (commit 8ff4515)
- Re-read current continuation.h and value.h to understand existing types
- Verify Lemon is available on Fedora (`sudo dnf install lemon`)

## Phase 3.1: New Continuation Types (Evaluation-Only)

Create continuations that DON'T execute during parsing but ARE executed during evaluation:

### 1. LiteralK - Unevaluated numeric literal
```cpp
class LiteralK : public Continuation {
public:
    double value;           // The literal number (NOT a heap Value*)
    Continuation* next;

    LiteralK(double v, Continuation* k) : value(v), next(k) {}

    ~LiteralK() override {
        delete next;
    }

    Value* invoke(Machine* machine) override;
    void mark(APLHeap* heap) override;
};
```

**Implementation notes:**
- During PARSING: Just stores the number, doesn't allocate
- During EVALUATING: Calls `heap->allocate_scalar(value)`
- Can be reused: Same LiteralK can be invoked multiple times

### 2. Verify Existing Continuations

Check that these already exist and work correctly:
- **LookupK** - Variable name resolution
- **ApplyMonadicK** - Monadic function application
- **ApplyDyadicK** - Dyadic function application
- **EvalStrandK** - Array strand evaluation

Ensure they:
- Don't parse tokens
- Only execute during EVALUATING mode
- Are reusable

### 3. Tests for Phase 3.1

Create `tests/test_literal_continuation.cpp`:
```cpp
TEST(LiteralKTest, BuildWithoutExecution) {
    // Building LiteralK doesn't allocate heap values
    LiteralK* lit = new LiteralK(42.0, nullptr);
    EXPECT_EQ(lit->value, 42.0);
    // No heap allocation happened yet
}

TEST(LiteralKTest, ExecuteDuringEval) {
    Machine machine;
    LiteralK* lit = new LiteralK(42.0, new HaltK());

    machine.ctrl.mode = ExecMode::EVALUATING;
    machine.push_kont(lit);
    Value* result = machine.execute();

    EXPECT_TRUE(result->is_scalar());
    EXPECT_EQ(result->as_scalar(), 42.0);
}

TEST(LiteralKTest, ReuseGraph) {
    Machine machine;
    LiteralK* lit = new LiteralK(42.0, new HaltK());

    // Execute same graph 3 times
    Value* r1 = machine.execute_graph(lit);
    Value* r2 = machine.execute_graph(lit);
    Value* r3 = machine.execute_graph(lit);

    // Three separate Value objects, all with value 42
    EXPECT_NE(r1, r2);  // Different pointers
    EXPECT_EQ(r1->as_scalar(), 42.0);
    EXPECT_EQ(r2->as_scalar(), 42.0);
    EXPECT_EQ(r3->as_scalar(), 42.0);
}
```

## Phase 3.2: Lemon Grammar File

Create `src/grammar.y` implementing Grammar G2 from Georgeff paper.

### Grammar Structure

```lemon
%name AplParser
%token_type {Token}
%extra_argument {ParseContext* ctx}
%type expr {Continuation*}
%type term {Continuation*}

%include {
    #include "continuation.h"
    #include "primitives.h"
    #include "token.h"
    #include <stdlib.h>

    struct ParseContext {
        Machine* machine;
        Continuation* result;
        LexerArena* arena;
    };
}

%syntax_error {
    ctx->result = nullptr;
    // Set error state
}

// Grammar G2: Right-to-left evaluation
// APL evaluates right-to-left: 2 + 3 × 4 = 2 + 12 = 14

expr(R) ::= term(T). {
    R = T;
}

expr(R) ::= term(L) PLUS expr(R_arg). {
    // Build: L → ApplyDyadicK(+, R_arg)
    // L is left operand graph, R_arg is right operand graph
    Continuation* apply = new ApplyDyadicK(nullptr, &prim_plus, R_arg);
    R = chain_continuations(L, apply);
}

expr(R) ::= term(L) MINUS expr(R_arg). {
    Continuation* apply = new ApplyDyadicK(nullptr, &prim_minus, R_arg);
    R = chain_continuations(L, apply);
}

expr(R) ::= term(L) TIMES expr(R_arg). {
    Continuation* apply = new ApplyDyadicK(nullptr, &prim_times, R_arg);
    R = chain_continuations(L, apply);
}

expr(R) ::= term(L) DIVIDE expr(R_arg). {
    Continuation* apply = new ApplyDyadicK(nullptr, &prim_divide, R_arg);
    R = chain_continuations(L, apply);
}

expr(R) ::= PLUS expr(E). {
    // Monadic plus (identity)
    R = new ApplyMonadicK(&prim_plus, E);
}

expr(R) ::= MINUS expr(E). {
    // Monadic minus (negation)
    R = new ApplyMonadicK(&prim_minus, E);
}

term(R) ::= NUMBER(N). {
    R = new LiteralK(N.number, nullptr);
}

term(R) ::= NAME(N). {
    const char* interned = ctx->machine->intern_string(N.name);
    R = new LookupK(interned, nullptr);
}

term(R) ::= LPAREN expr(E) RPAREN. {
    R = E;
}

// Array strands: 1 2 3
term(R) ::= NUMBER(N1) NUMBER(N2). {
    // Start building strand
    std::vector<Continuation*> elements;
    elements.push_back(new LiteralK(N1.number, nullptr));
    elements.push_back(new LiteralK(N2.number, nullptr));
    R = new EvalStrandK(elements.size(), nullptr);
    // Store elements in EvalStrandK
}

program ::= stmt_list.
stmt_list ::= stmt.
stmt_list ::= stmt_list NEWLINE stmt.
stmt_list ::= stmt_list DIAMOND stmt.

stmt ::= expr(E). {
    ctx->result = E;
}

stmt ::= NAME(N) ARROW expr(E). {
    const char* interned = ctx->machine->intern_string(N.name);
    ctx->result = new AssignK(interned, E);
}
```

### Helper Functions

```cpp
// In grammar.y %include section or separate helper file

// Chain two continuation graphs together
Continuation* chain_continuations(Continuation* first, Continuation* second) {
    // Find the end of first chain (the one with nullptr next)
    Continuation* curr = first;
    while (curr->get_next() != nullptr) {
        curr = curr->get_next();
    }
    // Attach second chain
    curr->set_next(second);
    return first;
}
```

## Phase 3.3: CMake Integration

Update `CMakeLists.txt`:

```cmake
# Find Lemon parser generator
find_program(LEMON_EXECUTABLE lemon)
if(NOT LEMON_EXECUTABLE)
    message(FATAL_ERROR "Lemon parser generator not found. Install with: sudo dnf install lemon")
endif()

# Generate parser from grammar
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/src/parser.c
    COMMAND ${LEMON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/src/grammar.y
    COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_SOURCE_DIR}/src/grammar.c ${CMAKE_SOURCE_DIR}/src/parser.c
    DEPENDS ${CMAKE_SOURCE_DIR}/src/grammar.y
    COMMENT "Generating parser from grammar.y"
)

# Add generated parser to sources
set(PARSER_SOURCES
    ${CMAKE_SOURCE_DIR}/src/parser.c
)

# Machine executable needs parser
add_executable(apl_machine
    ${CMAKE_SOURCE_DIR}/src/machine.cpp
    ${CMAKE_SOURCE_DIR}/src/continuation.cpp
    ${CMAKE_SOURCE_DIR}/src/primitives.cpp
    ${CMAKE_SOURCE_DIR}/src/value.cpp
    ${PARSER_SOURCES}
)
```

## Phase 3.4: Parse-to-Graph Function

Implement `Machine::parse_to_graph()`:

```cpp
// In machine.h
class Machine {
public:
    // Parse expression to continuation graph (doesn't execute)
    Continuation* parse_to_graph(const char* expr);

    // Execute a pre-built continuation graph
    Value* execute_graph(Continuation* graph);
};

// In machine.cpp
Continuation* Machine::parse_to_graph(const char* expr) {
    // Initialize lexer
    LexerArena arena;
    LexerState lexer;
    lexer_init(&lexer, expr, &arena);

    // Set up parse context
    ParseContext ctx;
    ctx.machine = this;
    ctx.result = nullptr;
    ctx.arena = &arena;

    // Create parser (Lemon-generated)
    void* parser = AplParserAlloc(malloc);

    // Feed tokens to parser
    Token tok;
    do {
        tok = lexer_next(&lexer);
        AplParser(parser, tok.type, tok, &ctx);
    } while (tok.type != TOK_EOF);

    // Finalize
    AplParser(parser, 0, tok, &ctx);
    AplParserFree(parser, free);

    // Return the built graph (not a value!)
    return ctx.result;
}
```

**Critical:** This function:
- Does NOT set mode to PARSING (Lemon handles parsing)
- Does NOT allocate heap Values
- Does NOT execute the graph
- Returns Continuation*, not Value*

## Phase 3.5: Execute-Graph Function

Implement `Machine::execute_graph()`:

```cpp
Value* Machine::execute_graph(Continuation* graph) {
    // Clone the graph (for reusability)
    Continuation* graph_copy = clone_continuation(graph);

    // Set mode to EVALUATING
    ctrl.mode = ExecMode::EVALUATING;
    ctrl.init_evaluating();

    // Push graph onto continuation stack
    push_kont(graph_copy);

    // Run trampoline loop
    return execute();
}
```

**Key points:**
- Clones graph to allow reuse (same graph can be executed multiple times)
- Sets mode to EVALUATING
- Runs standard trampoline loop

## Phase 3.6: Integration and Testing

Create `tests/test_phase3_integration.cpp`:

```cpp
TEST(Phase3Integration, ParseOnceExecuteMultiple) {
    Machine machine;

    // Parse "42" to continuation graph
    Continuation* graph = machine.parse_to_graph("42");
    ASSERT_NE(graph, nullptr);

    // Execute same graph 3 times
    Value* r1 = machine.execute_graph(graph);
    Value* r2 = machine.execute_graph(graph);
    Value* r3 = machine.execute_graph(graph);

    // Three separate Values, all 42
    EXPECT_NE(r1, r2);  // Different objects
    EXPECT_EQ(r1->as_scalar(), 42.0);
    EXPECT_EQ(r2->as_scalar(), 42.0);
    EXPECT_EQ(r3->as_scalar(), 42.0);

    delete graph;
}

TEST(Phase3Integration, FunctionCaching) {
    Machine machine;

    // Parse function body
    Continuation* body = machine.parse_to_graph("⍵ × 2");

    // Store in function cache
    machine.function_cache["double"] = body;

    // Later: retrieve and execute with different arguments
    Environment env1;
    env1.bind("⍵", machine.heap->allocate_scalar(5.0));
    machine.env = &env1;
    Value* r1 = machine.execute_graph(body);
    EXPECT_EQ(r1->as_scalar(), 10.0);

    Environment env2;
    env2.bind("⍵", machine.heap->allocate_scalar(7.0));
    machine.env = &env2;
    Value* r2 = machine.execute_graph(body);
    EXPECT_EQ(r2->as_scalar(), 14.0);
}

TEST(Phase3Integration, NoImmediateExecution) {
    Machine machine;
    size_t heap_size_before = machine.heap->allocation_count();

    // Parsing should NOT allocate heap values
    Continuation* graph = machine.parse_to_graph("1 + 2 + 3");

    size_t heap_size_after = machine.heap->allocation_count();
    EXPECT_EQ(heap_size_before, heap_size_after);  // No allocations

    // Only execution allocates
    Value* result = machine.execute_graph(graph);
    EXPECT_GT(machine.heap->allocation_count(), heap_size_after);

    delete graph;
}

TEST(Phase3Integration, RightToLeftEvaluation) {
    Machine machine;

    // APL: 2 + 3 × 4 = 2 + 12 = 14 (not 20!)
    Continuation* graph = machine.parse_to_graph("2 + 3 × 4");
    Value* result = machine.execute_graph(graph);

    EXPECT_EQ(result->as_scalar(), 14.0);

    delete graph;
}

TEST(Phase3Integration, ModeSeparation) {
    Machine machine;

    // Parsing doesn't change mode (Lemon is external)
    EXPECT_EQ(machine.ctrl.mode, ExecMode::HALTED);

    Continuation* graph = machine.parse_to_graph("42");
    EXPECT_EQ(machine.ctrl.mode, ExecMode::HALTED);  // Still halted

    // Execution sets mode to EVALUATING
    Value* result = machine.execute_graph(graph);
    EXPECT_EQ(machine.ctrl.mode, ExecMode::HALTED);  // Returns to halted

    delete graph;
}
```

## Phase 3.7: Statement Parser

Extend grammar for statements:

```lemon
// Assignment
stmt ::= NAME(N) ARROW expr(E). {
    const char* interned = ctx->machine->intern_string(N.name);
    ctx->result = new AssignK(interned, E);
}

// Multiple statements
program ::= stmt_list.
stmt_list ::= stmt.
stmt_list ::= stmt_list NEWLINE stmt.
stmt_list ::= stmt_list DIAMOND stmt.  // ⋄ separator
```

Create `AssignK` continuation:

```cpp
class AssignK : public Continuation {
public:
    std::string name;           // Variable name to assign to
    Continuation* expr;         // Expression to evaluate
    Continuation* next;         // Continuation after assignment

    AssignK(const std::string& n, Continuation* e, Continuation* k)
        : name(n), expr(e), next(k) {}

    Value* invoke(Machine* machine) override {
        // Evaluate expression
        Value* val = machine->execute_graph(expr);

        // Bind in environment
        machine->env->bind(name, val);

        // Continue
        if (next) return next->invoke(machine);
        machine->halt();
        return val;
    }
};
```

## Success Criteria

- [x] Research complete: understand what Phase 3 should do
- [ ] Lemon parser generates parser.c successfully
- [ ] Grammar builds continuation graphs (not AST)
- [ ] LiteralK continuation implemented and tested
- [ ] Parsing and evaluation are completely separate
- [ ] Same graph can be executed multiple times
- [ ] Function caching works
- [ ] All tests pass with no memory leaks
- [ ] Mode switches work correctly (PARSING vs EVALUATING)
- [ ] Right-to-left evaluation works correctly

## Implementation Order

1. **Day 1:** Phase 3.1 - Create LiteralK, test it thoroughly
2. **Day 2:** Phase 3.2 - Write grammar.y, integrate with CMake
3. **Day 3:** Phase 3.4 & 3.5 - Implement parse_to_graph() and execute_graph()
4. **Day 4:** Phase 3.6 - Integration tests, verify separation
5. **Day 5:** Phase 3.7 - Statement parser, assignments

## Key Differences from Broken Implementation

| Aspect | Broken Implementation | Correct Implementation |
|--------|----------------------|------------------------|
| Parser | ParseExprK continuation inside trampoline | Lemon parser generator (external) |
| Parsing output | Immediate Values | Continuation graphs |
| Heap allocation | During parsing | During evaluation only |
| Reusability | Can't reuse (values consumed) | Can reuse (graphs are immutable) |
| Mode separation | Mixed PARSING/EVALUATING | Clean separation |
| Function caching | Impossible | Natural and easy |

## References

- **Task List:** APL-Eigen CEK Machine Implementation Task List.md, Phase 3 (lines 242-292)
- **Design Doc:** APL-Eigen CEK Machine.md
- **Paper:** "Parsing and evaluation of APL with operators.pdf" (Grammar G2, pages 3-4)
- **Previous commit:** 8ff4515 (Phase 2 Post-Audit)
- **Reverted commit:** ff802d2 (broken Phase 3)
