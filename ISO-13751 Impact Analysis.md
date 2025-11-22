# ISO-13751 Operator Semantics Impact Analysis

## Document Reviewed
ISO-13751-operator-semantics.md (extracted from APL standard)

## Impact on Pre-Phase 5 Work

### CRITICAL: Phase 3 Grammar Must Support Operators

**Finding:** Lines 438-441 state:
> - Operators bind tighter than functions
> - `f/B` is `(f/)B` not `f(/B)`
> - Need operator precedence in parser

**Impact:** Our Lemon grammar cannot just parse function application. We need:

1. **Operator tokens** in the lexer (Phase 3.1):
   ```c
   REDUCE      "/"
   REDUCE_FIRST "⌿"
   SCAN        "\\"
   SCAN_FIRST  "⍀"
   OUTER       "∘."
   INNER       "."
   EACH        "¨"
   COMMUTE     "⍨"
   RANK        "⍤"
   ```

2. **Derived function rules** in grammar.y:
   ```lemon
   // Operator application creates derived functions
   derived_fn ::= fn REDUCE.              // f/ → derived function
   derived_fn ::= fn SCAN.                // f\ → derived function
   derived_fn ::= fn EACH.                // f¨ → derived function

   // Dyadic operators
   derived_fn ::= fn OUTER DOT fn.        // f∘.g
   derived_fn ::= fn DOT fn.              // f.g (inner product)

   // Derived functions apply to arguments
   expr ::= derived_fn term.              // (f/) B
   expr ::= term derived_fn term.         // A (f/) B (for dyadic)
   ```

3. **Precedence rules:**
   - Operators bind tighter than function application
   - `+/1 2 3` parses as `(+/) (1 2 3)` not `+ (/(1 2 3))`

**Decision Required:** Should we implement basic operator support in Phase 3, or defer all operators to Phase 5?

**Recommendation:** Implement **minimal operator infrastructure** in Phase 3:
- Add operator tokens to lexer
- Add derived function type to Value system
- Add grammar rules for operator binding
- But implement actual operator semantics in Phase 5

This prevents us from painting ourselves into a corner with the grammar.

---

### Phase 2 Status Check: Reduce/Scan

**Finding:** Document claims (lines 418-423):
> - ✓ Basic reduction working (right-to-left)
> - ✓ Scan working (right-to-left)

**Verification:** Checked `include/primitives.h` - functions exist:
- `fn_reduce()` - reduce along last axis
- `fn_reduce_first()` - reduce along first axis
- `fn_scan()` - scan along last axis
- `fn_scan_first()` - scan along first axis

**Status:** Phase 2 has basic reduce/scan primitives. ✓

---

### Missing: Identity Elements for Empty Vectors

**Finding:** Lines 63-70 specify identity elements:
```
When reducing an empty vector, return:
- `+/⍬` → 0
- `-/⍬` → 0
- `×/⍬` → 1
- `÷/⍬` → 1
- `⌈/⍬` → Negative infinity
- `⌊/⍬` → Positive infinity
```

**Current Status:** Need to check if `fn_reduce()` handles empty vectors.

**Action Required:** Add identity element handling to reduce/scan primitives in Phase 2 (or fix in Phase 3 if broken).

**Implementation:**
```cpp
Value* fn_reduce(Value* func, Value* omega) {
    if (omega->is_vector() && omega->vector_length() == 0) {
        // Return identity element based on function
        if (func == &prim_plus || func == &prim_minus) return allocate_scalar(0.0);
        if (func == &prim_times || func == &prim_divide) return allocate_scalar(1.0);
        // ... etc
    }
    // Normal reduction logic
}
```

---

### Type System Extension Required

**Finding:** Lines 443-447 state:
> - Add operator type to Value system
> - Derived functions vs primitive functions
> - Operator currying/partial application

**Current Value System:** (need to verify)
```cpp
enum ValueTag {
    SCALAR,
    VECTOR,
    MATRIX,
    FUNCTION,
    // OPERATOR missing?
    // DERIVED_FUNCTION missing?
};
```

**Required Extensions:**
1. Add `OPERATOR` tag for operator values
2. Add `DERIVED_FUNCTION` tag for derived functions (result of operator application)
3. Derived function stores:
   - Original operator
   - Operand function(s)
   - Invoke method that executes the operator semantics

**When to Add:** Should be added in Phase 3 (when grammar starts handling operators) or Phase 4 at latest.

**Alternative:** Could defer to Phase 5 if we parse operators but don't evaluate them until Phase 5.

---

### Right-to-Left Evaluation

**Finding:** Lines 49-52 confirm APL standard:
> Return »B1 f ºf/B2    (RIGHT-TO-LEFT evaluation!)

**Status:** Our Phase 3 grammar plan already accounts for this via right-associative grammar rules.

**No Action Required:** Already planned correctly.

---

### N-wise Reduction (Dyadic Reduce)

**Finding:** Section 3 describes N-wise reduction:
```apl
2+/1 2 3 4      → 3 5 7    (pairwise addition)
3×/1 2 3 4 5    → 6 24 60  (product of 3 consecutive)
```

**Current Status:** Not implemented (correctly deferred to Phase 5).

**No Action Required:** Phase 5 work as planned.

---

## Recommendations

### 1. Update Phase 3 Plan: Add Minimal Operator Support

**Add to Phase 3.1 (Lemon Grammar):**
- [ ] Add operator tokens to lexer (/, ⌿, \, ⍀, ∘., ., ¨, ⍨, ⍤)
- [ ] Add derived_fn grammar rules for operator binding
- [ ] Parse operators but don't evaluate them yet
- [ ] This ensures grammar is correct from the start

**Add to Phase 3.3 (Evaluation Continuations):**
- [ ] Add OPERATOR and DERIVED_FUNCTION to Value::tag enum
- [ ] Create DerivedFunction struct to store operator + operand(s)
- [ ] Stub implementations that signal "not yet implemented" error
- [ ] Full implementation deferred to Phase 5

**Rationale:** If we wait until Phase 5 to add operators to the grammar, we'll have to refactor the entire parser. Better to get the structure right in Phase 3.

### 2. Fix Phase 2: Add Identity Elements to Reduce

**Add to Phase 2 (if not already done):**
- [ ] Implement Table 5 identity elements in fn_reduce()
- [ ] Test: `+/⍬` returns 0
- [ ] Test: `×/⍬` returns 1
- [ ] Test: `⌈/⍬` returns negative infinity

**Priority:** Medium - Can be fixed in Phase 3 if easier.

### 3. No Changes to Phase 4 Required

Phase 4 (control flow) is unaffected by operator semantics.

### 4. Phase 5 Remains Mostly As-Is

Phase 5 will implement the actual operator semantics, but the grammar and type infrastructure will already exist from Phase 3.

---

## Updated Phase 3 Tasks (Additions Only)

### 3.1 Lemon Grammar for Expressions (ADDITIONS)

Add after existing expression rules:

- [ ] **Add operator tokens to lexer.re**
  - [ ] REDUCE (`/`)
  - [ ] REDUCE_FIRST (`⌿`)
  - [ ] SCAN (`\\`)
  - [ ] SCAN_FIRST (`⍀`)
  - [ ] OUTER (`∘.`)
  - [ ] DOT (`.`) - careful: also decimal point!
  - [ ] EACH (`¨`)
  - [ ] COMMUTE (`⍨`)
  - [ ] RANK (`⍤`)

- [ ] **Add derived function grammar rules**
  ```lemon
  %type derived_fn {DerivedFunction*}

  derived_fn ::= fn REDUCE.
  derived_fn ::= fn SCAN.
  derived_fn ::= fn EACH.
  derived_fn ::= fn OUTER DOT fn.
  derived_fn ::= fn DOT fn.
  derived_fn ::= fn COMMUTE.
  derived_fn ::= fn RANK term.
  ```

- [ ] **Update expression rules for derived functions**
  ```lemon
  expr ::= derived_fn term.               // Monadic derived function
  expr ::= term derived_fn term.          // Dyadic derived function
  ```

- [ ] **Test operator precedence**
  - [ ] `+/1 2 3` parses as `(+/) (1 2 3)`
  - [ ] `2×/1 2 3 4` parses as `2 (×/) (1 2 3 4)`

### 3.3 Evaluation Continuations (ADDITIONS)

- [ ] **Add operator types to Value system**
  ```cpp
  enum ValueTag {
      SCALAR,
      VECTOR,
      MATRIX,
      FUNCTION,
      OPERATOR,          // NEW
      DERIVED_FUNCTION   // NEW
  };
  ```

- [ ] **Create DerivedFunction struct**
  ```cpp
  struct DerivedFunction {
      OperatorType op;           // Which operator (REDUCE, SCAN, etc.)
      const PrimitiveFn* operand1;  // Left operand function
      const PrimitiveFn* operand2;  // Right operand (for dyadic operators)

      Value* invoke_monadic(Value* omega, Machine* machine);
      Value* invoke_dyadic(Value* alpha, Value* omega, Machine* machine);
  };
  ```

- [ ] **Stub implementations**
  - [ ] All derived function invocations signal "OPERATOR NOT YET IMPLEMENTED"
  - [ ] This allows parsing to work while deferring execution to Phase 5

### 3.6 Integration Testing (ADDITIONS)

- [ ] **Test operator parsing (not execution)**
  - [ ] `+/1 2 3` parses successfully (even if execution fails)
  - [ ] `2 3∘.+1 2` parses successfully
  - [ ] `+.×` parses as inner product
  - [ ] Parse error messages are clear when execution not implemented

---

## Decision Points for User

**Question 1:** Should we add minimal operator support (grammar + type system) to Phase 3?
- **Option A (Recommended):** Yes - add grammar rules and types now, implement semantics in Phase 5
- **Option B:** No - defer everything to Phase 5, risk grammar refactor later

**Question 2:** Should we fix identity elements for reduce in Phase 2 or Phase 3?
- **Option A:** Fix now in Phase 2 (small change)
- **Option B:** Fix in Phase 3 (along with other parser integration)
- **Option C:** Defer to Phase 5 (not recommended - it's a bug)

**Question 3:** How deeply should operator support go in Phase 3?
- **Option A (Minimal):** Just grammar + types, all invocations error
- **Option B (Partial):** Grammar + types + basic reduce/scan work (already exist)
- **Option C (Full):** Implement all operators in Phase 3 (not recommended - too much)

---

## Summary

The ISO-13751 operator semantics document reveals that:

1. **Operators must be in the grammar from Phase 3** (precedence rules)
2. **Type system needs OPERATOR and DERIVED_FUNCTION tags** (can't defer to Phase 5)
3. **Identity elements for reduce are missing** (should fix)
4. **Phase 2 reduce/scan exist but need identity element support**
5. **Right-to-left evaluation already planned correctly**

**Recommendation:** Add minimal operator infrastructure to Phase 3 (grammar + types), implement semantics in Phase 5.
