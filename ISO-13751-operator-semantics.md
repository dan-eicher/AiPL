# APL Operator Semantics Reference
## Extracted from ISO/IEC 13751:2000

This document contains the formal semantics for APL operators as defined in the ISO-13751 standard.
These will be implemented in Phase 5 (Operators).

---

## 1. Reduction Operator

### Syntax
- `f/B` - Reduce along last axis
- `f/[K]B` - Reduce along axis K
- `f⌿B` - Reduce along first axis
- `f⌿[K]B` - Reduce along axis K (alternative form)

### Informal Description
Z is the value produced by placing the dyadic function f between adjacent items along a designated axis of B and evaluating the resulting expression.

### Two Conforming Styles

**Important:** There are TWO conforming definitions for reduction in the APL standard:

1. **Enclose-Reduction-Style (APL2 style)**
   - Based on enclose
   - Preserves identity: `⍴⍴Z is 0⌈1+⍴⍴B`
   - This is what we're currently implementing

2. **Insert-Reduction-Style (Sharp/J style)**
   - Does not universally preserve the above identity
   - Function f is inserted between successive cells
   - Rank of f controls evaluation

**Note from spec:** Since reduction is used in other operators, the choice of Reduction-Style has a pervasive effect on the implementation.

### Evaluation Sequence (Enclose-Reduction-Style)

For vectors (our current implementation):
```
If the length of B is zero:
    Take action from Table 5 (identity elements)

If the length of B is one:
    Return scalar Z where ravel-list of Z is ravel-list of B

If the length of B is greater than one:
    Set B1 to the first-scalar in B
    Set B2 to the remainder-of B
    Return »B1 f ºf/B2    (RIGHT-TO-LEFT evaluation!)
```

**Key insight:** The spec explicitly says `B1 f (f/B2)` which confirms right-to-left evaluation.

For matrices:
```
Return array Z where:
    - Shape of Z is shape of B with axis K1 omitted
    - Each item Z1 of Z corresponds to vector B3 along axis K1
    - Z1 is f/B3 (recursive reduction)
```

### Identity Elements (Table 5)
When reducing an empty vector, return:
- `+/⍬` → 0
- `-/⍬` → 0
- `×/⍬` → 1
- `÷/⍬` → 1
- `⌈/⍬` → Negative infinity
- `⌊/⍬` → Positive infinity
- etc.

---

## 2. Scan Operator

### Syntax
- `f\B` - Scan along last axis
- `f\[K]B` - Scan along axis K
- `f⍀B` - Scan along first axis
- `f⍀[K]B` - Scan along axis K (alternative form)

### Informal Description
Z is an array having the same shape as B and containing the results produced by f reduction over all prefixes of a designated axis of B.

### Evaluation Sequence

For vectors:
```
If count of B is less than two:
    Return B

Otherwise:
    Return Z, a vector where:
    - Shape of Z equals shape of B
    - Item I of Z is f/B[⍳I] for all I in index-set
    - Type of Z is sufficient-type under mixed
```

**Key:** Each element is reduction of all elements up to that position.

For matrices:
```
Each vector-item along-axis K1 of Z is f\B1
where B1 is corresponding vector-item along-axis K1 of B
```

### Examples from spec:
```apl
+\1 1 1         → 1 2 3
⌊\1 1 1 0 0 0 1 1 1  → 1 1 1 0 0 0 0 0 0
-\'A'           → A
=\'AB'          → A 0
```

### Performance Note
The spec describes a quadratic algorithm. If function f is associative, scan may be implemented with a linear algorithm (optimization opportunity).

---

## 3. N-wise Reduction Operator

### Syntax
- `N f/B` - N-wise reduction along last axis
- `N f/[K]B` - N-wise reduction along axis K
- `N f⌿B` - N-wise reduction along first axis
- `N f⌿[K]B` - N-wise reduction along axis K

### Informal Description
N-wise reduction is the dyadic invocation of reduction. Z is produced by placing function f between subarrays of B. Each subarray has length N along a common dimension. If N is negative, subarrays are reversed before applying f.

### Validation
- If rank of N > 1, signal RANK ERROR
- If length of N > 1, signal LENGTH ERROR
- If N is not near-integer, signal DOMAIN ERROR
- Set N1 to integer-nearest-to N
- Set M1 to magnitude of N1

### Use Cases
```apl
2+/1 2 3 4      → 3 5 7    (pairwise addition)
3×/1 2 3 4 5    → 6 24 60  (product of 3 consecutive)
¯2-/1 2 3 4     → ¯1 ¯1 ¯1 (reversed pairs)
```

---

## 4. Outer Product Operator

### Syntax
- `A ∘.f B`

### Informal Description
Z is an array of shape `(⍴A),⍴B`. The elements of Z are the result of applying dyadic-function f to every possible combination of scalar arguments where left argument is from A and right from B.

### Shape Rule
If I is an index-list selecting single element of Z:
- First `⍴⍴A` items of I select from A (left argument to f)
- Last `⍴⍴B` items of I select from B (right argument to f)

### Evaluation Sequence
```
Shape of Z = (⍴A),⍴B

For each item I in ravel of A:
    For each item J in ravel of B:
        X = item I of ravel of A
        Y = item J of ravel of B
        N = count of B
        P = J + (N × (I-1))
        Q = X f Y
        Item P of ravel of Z is Q
```

### Example from spec:
```apl
10 20 30 ∘.+ 1 2 3
11 12 13
21 22 23
31 32 33
```

### Common Uses
- `A ∘.× B` - Multiplication table
- `A ∘.= B` - Equality matrix
- `A ∘., B` - Cartesian product

---

## 5. Inner Product Operator

### Syntax
- `A f.g B`

### Informal Description
Z is an array of shape `(⍴A)[¯1↓⍳1+⍴⍴A],(⍴B)[1+1↓⍳1+⍴⍴B]`. The elements are results from evaluating `f/X g Y` for all combinations of X (vector along last axis of A) and Y (vector along first axis of B).

### Shape Rule
Result shape = `(¯1↓⍴A),1↓⍴B`

### Length Constraint
Last dimension of A must equal first dimension of B, otherwise signal LENGTH ERROR.

### Evaluation Sequence

For vectors (both A and B are vectors):
```
Return f/A g B
```

For matrices:
```
For each vector X along last axis of A:
    For each vector Y along first axis of B:
        Compute Q = f/X g Y
        Place Q in appropriate position of Z
```

### Examples from spec:
```apl
4 2 1 +.× 1 0 1     → 5        (dot product)
N←2 2⍴0 1 1 0
N +.× 0 1           → 1 2 2 1
N +.× 1 0           → 1 1 0 0
N +.× 2 2⍴0 1 1 0   → 1 2 0 1
                       2 1 1 0
```

### Common Uses
- `A +.× B` - Matrix multiplication
- `A ∨.∧ B` - Boolean matrix multiplication
- `A ⌈.⌊ B` - Max-min product

### Special Case: Empty Vectors
If A or B is empty vector, result is `f/⍬0` (reduction of empty).

---

## 6. Duplicate Operator

### Syntax
- `f⍨B` (monadic)

### Informal Description
Z is `B f B`. The duplicate operator applies the function to the argument twice (as both left and right arguments).

### Evaluation Sequence
```
Return B f B
```

### Examples from spec:
```apl
∘.≠⍨⍳3     → Identity matrix (each with not-equal)
```

### Use Cases
- `⍳⍨5` → `⍳5` (identity, since iota is monadic)
- `+⍨3` → `3+3` = 6
- `∘.=⍨⍳5` → 5×5 identity matrix

### Note
If f is not an ambivalent function, a VALENCE ERROR will be signaled.

---

## 7. Commute Operator

### Syntax
- `A f⍨B` (dyadic)

### Informal Description
Z is `B f A`. The commute operator swaps the left and right arguments.

### Evaluation Sequence
```
Return B f A
```

### Examples from spec:
```apl
3-⍨4           → 1  (4-3, not 3-4)
+/2*⍨2 2⍴4 7 1 8  → 65 65  (sum of squares)
```

### Use Cases
- `-⍨` → subtract from (flip subtraction)
- `÷⍨` → divide into (flip division)
- `*⍨` → square (when used monadically with duplicate)

---

## 8. Each Operator (¨)

### Syntax
- `f¨B` (monadic)
- `A f¨B` (dyadic)

### Informal Description
The operand function f is applied independently to corresponding items of the arguments. Results are assembled in an array of the same shape as the argument(s).

### Evaluation Sequence

For dyadic case `A f¨B`:
```
If rank of A differs from rank of B:
    If A is scalar and B is not: set A to (⍴B)⍴A
    Else if B is scalar and A is not: set B to (⍴A)⍴B
    Else: signal RANK ERROR

If shape of A differs from shape of B:
    Signal LENGTH ERROR

For each index I in ravel of result:
    X = item I of ravel of A
    Y = item I of ravel of B
    Z[I] = X f Y
```

For monadic case `f¨B`:
```
For each index I in ravel of B:
    X = item I of ravel of B
    Z[I] = f X
```

### Examples from spec:
```apl
⍴¨⊂'AB',⊂'CDE'  → 2 3
```

### Use Cases
- Map function over nested arrays
- Element-wise operations on non-conforming shapes (with scalar extension)
- Apply function to each enclosed item

---

## 9. Rank Operator (⍤)

### Syntax
- `f⍤y B` (monadic)
- `A f⍤y B` (dyadic)

### Informal Description
The rank operator applies function f to cells of specified rank within the arguments. This is one of the most powerful and complex operators in APL.

### Key Concepts

**Cell:** The last k items of the shape-list of an array determine rank-k cells of the array.

**Frame:** For an array of rank r, its frame with respect to its cells of rank k is the r–k leading elements of its shape-list.

**Rank Vector:** Specifies the rank of cells to which function is applied:
- Single value: monadic rank (or both dyadic ranks if used dyadically)
- Two elements: [left dyadic rank, right dyadic rank]
- Three elements: [monadic rank, left dyadic rank, right dyadic rank]

### Conform Rules
If results from different cells don't agree in shape:
1. If ranks differ, bring to common maximum rank by introducing leading unit lengths
2. If shapes differ, use take on each result to bring to maximum shape

### Evaluation Sequence (Monadic)

```
Parse rank-vector y into canonical 3-element form
Extract monadic rank y3
If y3 > rank of B, use rank of B
If y3 negative, add to rank of B (negative from end)
Apply f to each rank-y6 cell of B
Conform individual results
Return result with shape = frame , common-result-shape
```

### Evaluation Sequence (Dyadic)

```
Parse rank-vector y into canonical 3-element form
Extract left rank y4 and right rank y5
Calculate effective ranks y8 (for A) and y9 (for B)
Calculate frames y10 and y11

Match frames (with scalar extension):
    If both empty: use A and B as-is
    If one empty: conform scalar to other
    If both non-empty: must match or signal error

Apply f between each rank-y8 cell of A and rank-y9 cell of B
Conform individual results
```

### Examples from spec:
```apl
,⍤2 ⍳2 3 3         → Ravel each matrix
⍉⍤2 ⍳2 3 3         → Transpose each matrix
⍳⍤0 ⍳3             → Apply iota to each scalar
0 1 2⍴⍤0 1 'ABC'   → Reshape at different ranks
⍳3 4 +⍤2 ⍳2 3 4    → Add matrices element-wise
```

### Use Cases
- Apply operations to subarrays of specific rank
- Generalize scalar extension to arbitrary ranks
- Process high-dimensional arrays in structured ways

---

## Implementation Notes

### Phase 5 Checklist

1. **Operator Type System**
   - Operators are distinct from functions
   - Take functions as operands
   - Return derived functions
   - Operator binding is stronger than function application

2. **Reduce/Scan Already Implemented**
   - ✓ Basic reduction working (right-to-left)
   - ✓ Scan working (right-to-left)
   - ✓ Axis variants implemented
   - ⚠ Identity elements not yet implemented (Table 5)
   - ⚠ N-wise reduction not implemented

3. **To Implement**
   - [ ] Outer product `∘.f`
   - [ ] Inner product `f.g`
   - [ ] N-wise reduction (dyadic reduction)
   - [ ] Duplicate operator `f⍨`
   - [ ] Commute operator `⍨` (dyadic)
   - [ ] Each operator `f¨`
   - [ ] Rank operator `f⍤y`
   - [ ] Identity element table for empty reductions
   - [ ] Proper operator type/rank checking
   - [ ] Operator composition rules

4. **Parser Considerations**
   - Operators bind tighter than functions
   - `f/B` is `(f/)B` not `f(/B)`
   - `f.g` binds to create derived function
   - Need operator precedence in parser

5. **Type System Updates**
   - Add operator type to Value system
   - Derived functions vs primitive functions
   - Operator currying/partial application

---

## Operator Summary Table

| Operator | Symbol | Monadic | Dyadic | Description | Priority |
|----------|--------|---------|--------|-------------|----------|
| Reduction | `/` `⌿` | No | Yes | Insert function between elements | High |
| Scan | `\` `⍀` | No | Yes | Cumulative reduction | High |
| N-wise Reduction | `/` `⌿` | No | Yes | Windowed reduction | Medium |
| Outer Product | `∘.` | No | Yes | Apply to all combinations | High |
| Inner Product | `.` | No | Yes | Generalized matrix product | High |
| Duplicate | `⍨` | Yes | No | Apply to self (monadic) | Low |
| Commute | `⍨` | No | Yes | Swap arguments (dyadic) | Low |
| Each | `¨` | Yes | Yes | Apply to each element | Medium |
| Rank | `⍤` | Yes | Yes | Apply to cells of given rank | Low |

**Note:** Duplicate and Commute use the same symbol `⍨` but differ in valence.

## References

- ISO/IEC 13751:2000 (E) - Programming languages — APL
- Section 9.2: Monadic Operators (pages 110-119)
  - 9.2.1: Reduction
  - 9.2.2: Scan
  - 9.2.3: N-wise Reduction
  - 9.2.4: Duplicate
  - 9.2.5: Commute
  - 9.2.6: Each
- Section 9.3: Dyadic Operators (pages 120-126)
  - 9.3.1: Outer Product
  - 9.3.2: Inner Product
  - 9.3.3-9.3.5: Rank Operator
- Table 5: Identity Elements for Reduction
- Table 6: N-wise Reduction Actions

---

*This document extracted from `/tmp/apl_spec.txt` on 2025-11-17*
*For use in Phase 5: Operators implementation*
*Updated to include all 9 standard APL operators*
