# Task List Updates Required for Lemon Parser Approach

## Summary

The decision to use **Lemon parser → direct continuation graphs** (instead of continuation-based parsing) has knock-on effects throughout the task list. This document outlines what needs to change.

## Core Architectural Change

**Before:** Parser continuations (ParseExprK, ParseStrandK, ParseTermK) run inside the CEK machine trampoline and build/execute simultaneously.

**After:** Lemon parser generates parser.c from grammar.y, builds continuation graphs directly in grammar actions, then CEK machine executes the graphs.

## Phase 3 Changes Required

### 3.1 Evaluation Context System - **KEEP BUT CLARIFY**

Current task list says:
```
- [ ] Define EvalContext enum (eCgv, eCts, eCtn, eCto, eCtf)
- [ ] Implement ContextK continuation
```

**Analysis:** Looking at the design document, EvalContext is for **runtime type conversions** (like JavaScript's ToString, ToNumber, etc.). This is an **evaluation-time feature**, not a parsing feature.

**Decision:**
- Keep this section but clarify it's for evaluation, not parsing
- OR move to Phase 2 (Essential Operations) as it's more fundamental
- OR defer to Phase 4 when we actually need type-directed operations

**Recommendation:** **Defer to Phase 4** or later. It's not needed for basic parsing and evaluation. We can add it when we implement operators that need different type contexts.

### 3.2 Expression Parser - **COMPLETE REWRITE**

Current task list says:
```
- [ ] Implement ParseExprK continuation
- [ ] Implement ParseStrandK for space-separated arrays (1 2 3)
- [ ] Implement ParseTermK for individual terms
- [ ] Add operator precedence parsing (APL: right-to-left evaluation)
```

**Changes needed:**
```
### 3.2 Lemon Grammar for Expressions
- [ ] Create src/grammar.y with Grammar G2
- [ ] Add token declarations (%token_type, %type)
- [ ] Implement expression rules (expr, term)
- [ ] Add operator rules (PLUS, MINUS, TIMES, DIVIDE, POWER)
  - [ ] Right-to-left associativity via grammar structure
- [ ] Add array strand rules (1 2 3)
- [ ] Handle parenthesized expressions
- [ ] Grammar actions build continuation graphs directly
- [ ] Test grammar generates valid parser.c
```

### 3.3 Statement Parser - **MINOR CHANGES**

Current task list says:
```
- [ ] Parse assignment statements
- [ ] Parse control flow (if/while/for)
- [ ] Handle multi-line programs
- [ ] Add comment support
```

**Changes needed:**
- Reword to clarify these are **grammar rules**, not continuation implementations
- Control flow parsing will happen in Phase 4 (when we implement control flow)

**Updated:**
```
### 3.3 Statement Parser (Grammar Rules)
- [ ] Add assignment grammar rule (name ← expr)
- [ ] Add statement separator rules (NEWLINE, DIAMOND ⋄)
- [ ] Add comment token handling in lexer
- [ ] Grammar actions build AssignK continuations
- [ ] Test multi-statement programs parse correctly
- [ ] Note: Control flow grammar deferred to Phase 4
```

### 3.4 Evaluation Continuations - **KEEP, ADD LITERAL**

Current task list says:
```
- [ ] Implement EvalStrandK for evaluating array strands
- [ ] Implement ApplyMonadicK for unary function application
- [ ] Implement ApplyDyadicK for binary function application
- [ ] Implement LookupK for variable resolution
```

**Changes needed:**
- Add LiteralK (new, critical for the architecture)
- Clarify these are **execution-time** continuations (invoked during EVALUATING mode)
- Emphasize they do NOT parse

**Updated:**
```
### 3.4 Evaluation Continuations
- [ ] Implement LiteralK for numeric literals
  - [ ] Stores double value (NOT a heap Value*)
  - [ ] invoke(): Allocates Value during EVALUATING mode only
  - [ ] Mark with extensive tests (build without exec, reuse graph, etc.)
- [ ] Verify EvalStrandK exists and works correctly
- [ ] Verify ApplyMonadicK exists and works correctly
- [ ] Verify ApplyDyadicK exists and works correctly
- [ ] Verify LookupK exists and works correctly
- [ ] All continuations: execute only, never parse
```

### 3.5 Parse-Eval Integration - **COMPLETE REWRITE**

Current task list says:
```
- [ ] Connect parser to continuation creation
- [ ] Implement ParseK continuation for deferred parsing
- [ ] Add runtime type-directed parsing
- [ ] Create continuation graph builder
- [ ] Add function_cache integration for caching parsed functions
```

**Changes needed:**
- Remove "ParseK continuation" (doesn't exist with Lemon)
- Focus on parse_to_graph() and execute_graph() functions
- Clarify function_cache integration

**Updated:**
```
### 3.5 Machine Integration
- [ ] Implement Machine::parse_to_graph(const char* expr)
  - [ ] Initialize lexer and feed tokens to Lemon parser
  - [ ] Parser builds continuation graph in grammar actions
  - [ ] Returns Continuation* (not Value*)
  - [ ] Does NOT allocate heap Values
  - [ ] Does NOT execute the graph
- [ ] Implement Machine::execute_graph(Continuation* graph)
  - [ ] Set mode to EVALUATING
  - [ ] Clone graph for reusability
  - [ ] Push graph onto continuation stack
  - [ ] Run trampoline loop
  - [ ] Return Value*
- [ ] Add function_cache integration
  - [ ] Store graphs in std::unordered_map<string, Continuation*>
  - [ ] Test: parse once, cache, retrieve multiple times
- [ ] Add graph cloning for reusability
- [ ] Comprehensive integration tests (see Phase 3 Implementation Plan.md)
```

### 3.6 CMake Integration - **NEW SECTION**

**Add:**
```
### 3.6 Build System Integration
- [ ] Find Lemon executable in CMakeLists.txt
- [ ] Add custom command to generate parser.c from grammar.y
- [ ] Add parser.c to build sources
- [ ] Verify lemon package available on Fedora
- [ ] Test clean builds (grammar.y → parser.c → parser.o)
```

## Phase 4 Changes Required

### 4.2 Control Structure Integration - **MODIFY**

Current says:
```
- [ ] Connect parser to control flow continuations
```

**Change to:**
```
- [ ] Add grammar rules for control flow keywords
  - [ ] :If / :Then / :Else / :EndIf
  - [ ] :While / :EndWhile
  - [ ] :For / :In / :EndFor
  - [ ] :Return
  - [ ] :Leave (break)
- [ ] Grammar actions build control flow continuation graphs
```

### 4.4 Function Definition - **MODIFY**

Current says:
```
- [ ] Parse function definitions (dfns)
```

**Change to:**
```
- [ ] Add grammar rules for function definitions
  - [ ] { ... } dfn syntax
  - [ ] ⍺ (alpha) and ⍵ (omega) parameter binding
- [ ] Grammar actions build function closure continuations
```

## Phase 5+ Changes

Any task that says "parse" or "parser" should be understood as:
- **"Add grammar rules to grammar.y"**
- **"Grammar actions build continuation graphs"**
- **NOT "write continuation code that parses tokens"**

## What DOESN'T Change

These aspects are unaffected by the parser decision:

1. **Phase 1:** Lexer, Value system, GC, Machine core - all unchanged
2. **Phase 2:** Primitives, array operations, environment - all unchanged
3. **Continuation execution:** The trampoline loop, continuation invocation, GC marking - all unchanged
4. **Memory management:** Heap, GC, continuation lifetime - all unchanged
5. **Control flow execution:** IfK, WhileK, etc. still work the same way - only how they're **created** changes

## Summary of Changes by Phase

| Phase | Change Level | Description |
|-------|--------------|-------------|
| Phase 1 | None | Core foundation unchanged |
| Phase 2 | None | Operations unchanged |
| Phase 3 | Major | Complete rewrite: Lemon parser instead of parser continuations |
| Phase 4 | Minor | Add grammar rules for control flow |
| Phase 5 | Minor | Add grammar rules for advanced features |
| Phase 6 | None | Optimization and testing unchanged |

## Implementation Impact

**Effort Change:**
- Phase 3 complexity **reduces** (Lemon is simpler than continuation-based parsing)
- Overall project complexity **reduces** (cleaner separation)
- Testing becomes **easier** (parser testable independently)

**Timeline Impact:**
- Phase 3 may be **faster** with Lemon (grammar is ~100-150 LOC vs complex continuation logic)
- Less debugging needed (parser generator handles edge cases)

**Quality Impact:**
- **Better:** Clean separation of concerns
- **Better:** Easier to test and debug
- **Better:** Matches industry practice
- **Better:** Enables function caching naturally

## Recommendation

**Update the task list** with these changes before starting implementation. The architecture is significantly different from what the current task list describes.

**Priority:** High - The current task list would lead us back into the broken continuation-based parsing approach.

## Files to Update

1. **APL-Eigen CEK Machine Implementation Task List.md** - Rewrite Phase 3 section
2. Keep **Phase 3 Implementation Plan.md** as the detailed guide
3. Add note in task list pointing to the implementation plan for Phase 3 details
