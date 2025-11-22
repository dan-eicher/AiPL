# Phase 3 Final Task List - Complete Implementation Guide

## Overview

This document consolidates all research and planning for Phase 3 implementation. It reflects:
1. The decision to use Lemon parser → continuation graphs (direct)
2. ISO-13751 operator semantics requirements
3. Lessons learned from the broken Phase 3 attempt

## What Changed From Original Plan

### Original (Broken) Approach
- ParseExprK, ParseStrandK, ParseTermK continuations
- Parser continuations ran inside CEK machine trampoline
- Mixed parsing with immediate evaluation
- Could not build reusable continuation graphs
- Function caching impossible

### New (Correct) Approach
- Lemon parser generator external to CEK machine
- Grammar actions build continuation graphs directly
- Clean separation: parsing builds, evaluation executes
- Continuation graphs are reusable and cacheable
- Operator infrastructure included from the start

## Critical Additions From ISO-13751 Analysis

### 1. Operator Tokens (Must Be In Phase 3)
**Reason:** Operators bind tighter than functions (`f/B` = `(f/)B`)

Required tokens:
- `/` REDUCE
- `⌿` REDUCE_FIRST
- `\` SCAN
- `⍀` SCAN_FIRST
- `∘.` OUTER
- `.` DOT (careful: also decimal point!)
- `¨` EACH
- `⍨` COMMUTE
- `⍤` RANK

### 2. Derived Function Grammar Rules
**Reason:** Parser must understand operator precedence

```lemon
derived_fn ::= fn REDUCE.              // f/ creates derived function
derived_fn ::= fn SCAN.                // f\ creates derived function
derived_fn ::= fn OUTER DOT fn.        // f∘.g creates derived function
derived_fn ::= fn DOT fn.              // f.g (inner product)
expr ::= derived_fn term.              // Apply derived function
```

### 3. Type System Extensions
**Reason:** Can't represent operators without new types

```cpp
enum ValueTag {
    SCALAR,
    VECTOR,
    MATRIX,
    FUNCTION,
    OPERATOR,          // NEW
    DERIVED_FUNCTION   // NEW
};

enum OperatorType {
    REDUCE, SCAN, OUTER, INNER, EACH, COMMUTE, RANK
};

struct DerivedFunction {
    OperatorType op_type;
    const PrimitiveFn* operand1;
    const PrimitiveFn* operand2;
    Value* operand_value;  // For rank operator
};
```

### 4. Stub Implementation Strategy
**Reason:** Parse operators in Phase 3, implement semantics in Phase 5

```cpp
// DerivedFunctionK::invoke() in Phase 3
Value* DerivedFunctionK::invoke(Machine* machine) {
    // Stub: signal error
    machine->signal_error("OPERATOR NOT YET IMPLEMENTED");
    return nullptr;
}

// Full implementation in Phase 5
```

## Complete Phase 3 Task Breakdown

### 3.1: Lemon Grammar for Expressions
**Estimated Time:** 2-3 days

1. **Install Lemon**
   - `sudo dnf install lemon`
   - Verify: `lemon -v`

2. **Create grammar.y structure**
   ```lemon
   %name AplParser
   %token_type {Token}
   %extra_argument {ParseContext*}
   %type expr {Continuation*}
   %type term {Continuation*}
   %type derived_fn {Continuation*}
   ```

3. **Add operator tokens to lexer.re**
   - All 9 operator tokens
   - Handle `.` ambiguity (decimal point vs operator)

4. **Write expression grammar**
   - Base cases: term, number, name
   - Function application: dyadic and monadic
   - Derived function application
   - Operator precedence rules

5. **Write derived function grammar**
   - All monadic operators (/, \, ¨, ⍨)
   - All dyadic operators (∘., ., ⍤)

6. **Grammar actions**
   - Build Continuation* (never Value*)
   - No heap allocation
   - No execution

7. **Test grammar**
   - Verify parser.c generates
   - Test operator precedence
   - Test parsing (not execution)

### 3.2: Statement Parser
**Estimated Time:** 1 day

1. **Statement rules**
   - Expression statements
   - Assignment (name ← expr)
   - Multi-statement (NEWLINE, DIAMOND)

2. **Grammar actions**
   - Build AssignK for assignments
   - Chain statements

3. **Comment handling**
   - Lexer treats comments as whitespace

### 3.3: Evaluation Continuations and Type System
**Estimated Time:** 2-3 days

1. **Extend Value type system**
   - Add OPERATOR tag
   - Add DERIVED_FUNCTION tag
   - Create OperatorType enum
   - Create DerivedFunction struct
   - Update Value union
   - Update Value::mark() for GC

2. **Implement LiteralK**
   - Stores double (not Value*)
   - Creates Value during evaluation only
   - Extensive tests

3. **Implement DerivedFunctionK (stub)**
   - Stores DerivedFunction*
   - invoke() signals "NOT YET IMPLEMENTED"
   - Full implementation in Phase 5

4. **Verify existing continuations**
   - LookupK, ApplyMonadicK, ApplyDyadicK, EvalStrandK
   - All work correctly
   - All are reusable

### 3.4: Machine Integration
**Estimated Time:** 2 days

1. **Implement parse_to_graph()**
   - Initialize lexer
   - Feed tokens to Lemon parser
   - Return Continuation* (not Value*)
   - No heap allocation
   - No execution

2. **Implement execute_graph()**
   - Set mode to EVALUATING
   - Clone graph (optional)
   - Push to continuation stack
   - Run trampoline
   - Return Value*

3. **Add function_cache**
   - `std::unordered_map<string, Continuation*>`
   - Test: parse once, execute multiple times

4. **Add intern_string()**
   - Uses StringPool
   - Returns stable const char*

### 3.5: CMake Build Integration
**Estimated Time:** 0.5 days

1. **Find Lemon**
   - find_program(LEMON_EXECUTABLE lemon)
   - FATAL_ERROR if not found

2. **Generate parser**
   - Custom command: grammar.y → parser.c
   - Add parser.c to sources
   - DEPENDS on grammar.y

3. **Test builds**
   - Clean build works
   - Grammar changes trigger rebuild

### 3.6: Integration Testing
**Estimated Time:** 2 days

1. **Core functionality tests**
   - Parse once, execute multiple
   - Function caching
   - No immediate execution
   - Mode separation
   - Right-to-left evaluation

2. **Expression tests**
   - Array strands
   - Assignment
   - Multi-statement programs

3. **Operator parsing tests**
   - `+/1 2 3` parses
   - `+\1 2 3` parses
   - `∘.+` parses
   - `+.×` parses
   - Operator precedence correct
   - Execution gives "NOT YET IMPLEMENTED"

4. **Memory tests**
   - Zero leaks
   - GC works correctly
   - Graph reuse works

## Phase 2 Addition: Identity Elements

**When:** Can be done in Phase 2 post-audit or Phase 3
**Estimated Time:** 0.5 days

Add to fn_reduce() and fn_scan():
```cpp
if (omega->is_vector() && omega->vector_length() == 0) {
    // Table 5 from ISO-13751
    if (func == &prim_plus || func == &prim_minus)
        return heap->allocate_scalar(0.0);
    if (func == &prim_times || func == &prim_divide)
        return heap->allocate_scalar(1.0);
    // ... etc for max/min
}
```

Tests:
- `+/⍬` → 0
- `×/⍬` → 1
- etc.

## Total Estimated Time: Phase 3

- 3.1: 2-3 days
- 3.2: 1 day
- 3.3: 2-3 days
- 3.4: 2 days
- 3.5: 0.5 days
- 3.6: 2 days
- Identity elements: 0.5 days

**Total: 10-12 days** (within 15-20% budget estimate)

## Success Criteria Checklist

- [ ] Lemon parser generates valid parser.c
- [ ] Grammar parses all expressions correctly
- [ ] Operator tokens recognized
- [ ] Operator precedence correct (`f/B` = `(f/)B`)
- [ ] Parse and evaluation completely separated
- [ ] Parsing doesn't allocate heap Values
- [ ] Same graph executes multiple times
- [ ] Function caching works
- [ ] Operator parsing works (execution stubbed)
- [ ] Right-to-left evaluation correct
- [ ] All tests pass
- [ ] Zero memory leaks
- [ ] Identity elements work (if implemented)

## Reference Documents

1. **Phase 3 Implementation Plan.md** - Detailed code examples and implementation guide
2. **ISO-13751 Impact Analysis.md** - Why operator support is needed in Phase 3
3. **Task List Updates Required.md** - Analysis of required changes
4. **APL-Eigen CEK Machine Implementation Task List.md** - Official task list (updated)
5. **Georgeff et al. paper** - Grammar G2 specification
6. **ISO-13751-operator-semantics.md** - Operator semantics from standard

## Risk Mitigation

### Risk: Grammar gets too complex
**Mitigation:** Start with minimal grammar (numbers, functions), add operators incrementally

### Risk: Operator precedence wrong
**Mitigation:** Test precedence early with simple expressions like `+/1 2`

### Risk: Type system changes break existing code
**Mitigation:** Add new types without removing old ones, update gradually

### Risk: Phase 3 takes too long
**Mitigation:** Stub operators early, defer full implementation to Phase 5

### Risk: Lemon not available
**Mitigation:** Have fallback plan (hand-written recursive descent)

## Key Principles to Remember

1. **Parsing builds graphs, evaluation executes them** - Never mix these
2. **Grammar actions create Continuation*, never Value*** - No heap allocation during parsing
3. **Operators bind tighter than functions** - Grammar must enforce this
4. **Stub what you can't implement yet** - Don't let Phase 5 block Phase 3
5. **Test parsing separately from execution** - Parser should work even if execution stubbed
6. **Right-to-left evaluation** - Grammar enforces this via structure
7. **Graphs are reusable** - Same graph, multiple executions

## Next Steps After Phase 3

1. **Phase 4:** Control flow (IF, WHILE, FOR, RETURN, LEAVE)
   - Add grammar rules for control structures
   - Implement control flow continuations
   - Function definitions (dfns)

2. **Phase 5:** Operator implementation
   - Implement DerivedFunctionK::invoke() for real
   - All operator semantics per ISO-13751
   - Remove "NOT YET IMPLEMENTED" stubs

## Questions Resolved

**Q: Should operators be in Phase 3 or Phase 5?**
**A:** Grammar and types in Phase 3, semantics in Phase 5

**Q: Can we defer all operators to Phase 5?**
**A:** No - grammar precedence requires operator tokens in Phase 3

**Q: How do we test operators if they're not implemented?**
**A:** Test parsing succeeds, execution gives predictable error

**Q: What about type-driven parsing (EvalContext)?**
**A:** Deferred - not needed for basic parsing, add later if needed

**Q: Should we use Lemon or hand-written parser?**
**A:** Lemon - cleaner separation, easier to maintain

**Q: Fix identity elements in Phase 2 or Phase 3?**
**A:** Either works, but Phase 2 is cleaner (it's a primitive bug)
