// Primitives - APL primitive functions and operations

#pragma once

#include "value.h"

namespace apl {

// Forward declaration
class Machine;

// PrimitiveFn is defined in value.h

// Monadic built-in functions
// All primitives set machine->ctrl.value on success or push ThrowErrorK on error
// axis is nullptr if no axis specified, otherwise the axis value from F[k] syntax
void fn_conjugate(Machine* m, Value* axis, Value* omega);     // + monadic (conjugate/identity)
void fn_negate(Machine* m, Value* axis, Value* omega);        // - monadic (negation)
void fn_signum(Machine* m, Value* axis, Value* omega);        // × monadic (signum/sign)
void fn_reciprocal(Machine* m, Value* axis, Value* omega);    // ÷ monadic (reciprocal)
void fn_exponential(Machine* m, Value* axis, Value* omega);   // * monadic (exponential e^x)
void fn_ceiling(Machine* m, Value* axis, Value* omega);       // ⌈ monadic (ceiling)
void fn_floor(Machine* m, Value* axis, Value* omega);         // ⌊ monadic (floor)
void fn_not(Machine* m, Value* axis, Value* omega);           // ~ monadic (not)
void fn_magnitude(Machine* m, Value* axis, Value* omega);     // | monadic (absolute value)
void fn_natural_log(Machine* m, Value* axis, Value* omega);   // ⍟ monadic (natural logarithm)
void fn_factorial(Machine* m, Value* axis, Value* omega);     // ! monadic (factorial)
void fn_reverse(Machine* m, Value* axis, Value* omega);       // ⌽ monadic (reverse last axis)
void fn_reverse_first(Machine* m, Value* axis, Value* omega); // ⊖ monadic (reverse first axis)
void fn_tally(Machine* m, Value* axis, Value* omega);         // ≢ monadic (count along first axis)
void fn_enlist(Machine* m, Value* axis, Value* omega);        // ∊ monadic (enlist/flatten)
void fn_grade_up(Machine* m, Value* axis, Value* omega);      // ⍋ monadic (grade up)
void fn_grade_down(Machine* m, Value* axis, Value* omega);    // ⍒ monadic (grade down)
void fn_grade_up_dyadic(Machine* m, Value* axis, Value* lhs, Value* rhs);   // ⍋ dyadic (character grade up)
void fn_grade_down_dyadic(Machine* m, Value* axis, Value* lhs, Value* rhs); // ⍒ dyadic (character grade down)
void fn_unique(Machine* m, Value* axis, Value* omega);        // ∪ monadic (unique)
void fn_pi_times(Machine* m, Value* axis, Value* omega);      // ○ monadic (pi times)
void fn_roll(Machine* m, Value* axis, Value* omega);          // ? monadic (random)
void fn_execute(Machine* m, Value* axis, Value* omega);       // ⍎ monadic (execute string as APL)
void fn_format_monadic(Machine* m, Value* axis, Value* omega);           // ⍕ monadic (format to chars)
void fn_format_dyadic(Machine* m, Value* axis, Value* alpha, Value* omega); // ⍕ dyadic (format with spec)
void fn_table(Machine* m, Value* axis, Value* omega);         // ⍪ monadic (table - convert to matrix)
void fn_depth(Machine* m, Value* axis, Value* omega);         // ≡ monadic (depth - nesting level)
void fn_match(Machine* m, Value* axis, Value* alpha, Value* omega); // ≡ dyadic (match - identical?)
void fn_enclose(Machine* m, Value* axis, Value* omega);        // ⊂ monadic (enclose - create nested scalar)
void fn_disclose(Machine* m, Value* axis, Value* omega);       // ⊃ monadic (disclose/first - unwrap nested)
void fn_pick(Machine* m, Value* axis, Value* alpha, Value* omega); // ⊃ dyadic (pick - index into nested)

// Dyadic built-in functions
void fn_catenate_first(Machine* m, Value* axis, Value* lhs, Value* rhs); // ⍪ dyadic (join along first axis)
void fn_add(Machine* m, Value* axis, Value* lhs, Value* rhs);       // + dyadic (addition)
void fn_subtract(Machine* m, Value* axis, Value* lhs, Value* rhs);  // - dyadic (subtraction)
void fn_multiply(Machine* m, Value* axis, Value* lhs, Value* rhs);  // × dyadic (multiplication)
void fn_divide(Machine* m, Value* axis, Value* lhs, Value* rhs);    // ÷ dyadic (division)
void fn_power(Machine* m, Value* axis, Value* lhs, Value* rhs);     // * dyadic (power)
void fn_equal(Machine* m, Value* axis, Value* lhs, Value* rhs);     // = dyadic (equality)
void fn_not_equal(Machine* m, Value* axis, Value* lhs, Value* rhs); // ≠ dyadic (not equal)
void fn_less(Machine* m, Value* axis, Value* lhs, Value* rhs);      // < dyadic (less than)
void fn_greater(Machine* m, Value* axis, Value* lhs, Value* rhs);   // > dyadic (greater than)
void fn_less_eq(Machine* m, Value* axis, Value* lhs, Value* rhs);   // ≤ dyadic (less or equal)
void fn_greater_eq(Machine* m, Value* axis, Value* lhs, Value* rhs);// ≥ dyadic (greater or equal)
void fn_maximum(Machine* m, Value* axis, Value* lhs, Value* rhs);   // ⌈ dyadic (maximum)
void fn_minimum(Machine* m, Value* axis, Value* lhs, Value* rhs);   // ⌊ dyadic (minimum)
void fn_and(Machine* m, Value* axis, Value* lhs, Value* rhs);       // ∧ dyadic (and/lcm)
void fn_or(Machine* m, Value* axis, Value* lhs, Value* rhs);        // ∨ dyadic (or/gcd)
void fn_nand(Machine* m, Value* axis, Value* lhs, Value* rhs);      // ⍲ dyadic (nand)
void fn_nor(Machine* m, Value* axis, Value* lhs, Value* rhs);       // ⍱ dyadic (nor)
void fn_residue(Machine* m, Value* axis, Value* lhs, Value* rhs);   // | dyadic (modulo/residue)
void fn_logarithm(Machine* m, Value* axis, Value* lhs, Value* rhs); // ⍟ dyadic (logarithm base)
void fn_binomial(Machine* m, Value* axis, Value* lhs, Value* rhs);  // ! dyadic (binomial/combinations)
void fn_rotate(Machine* m, Value* axis, Value* lhs, Value* rhs);    // ⌽ dyadic (rotate last axis)
void fn_rotate_first(Machine* m, Value* axis, Value* lhs, Value* rhs); // ⊖ dyadic (rotate first axis)
void fn_index_of(Machine* m, Value* axis, Value* lhs, Value* rhs);  // ⍳ dyadic (index of)
void fn_member_of(Machine* m, Value* axis, Value* lhs, Value* rhs); // ∊ dyadic (member of)
void fn_replicate(Machine* m, Value* axis, Value* lhs, Value* rhs); // / dyadic (replicate)
void fn_union(Machine* m, Value* axis, Value* lhs, Value* rhs);     // ∪ dyadic (union)
void fn_without(Machine* m, Value* axis, Value* lhs, Value* rhs);   // ~ dyadic (without)
void fn_circular(Machine* m, Value* axis, Value* lhs, Value* rhs);  // ○ dyadic (circular functions)
void fn_deal(Machine* m, Value* axis, Value* lhs, Value* rhs);      // ? dyadic (deal)
void fn_expand(Machine* m, Value* axis, Value* lhs, Value* rhs);       // \ dyadic (expand)
void fn_expand_first(Machine* m, Value* axis, Value* lhs, Value* rhs); // ⍀ dyadic (expand first)
void fn_decode(Machine* m, Value* axis, Value* lhs, Value* rhs);    // ⊥ dyadic (decode/base value)
void fn_encode(Machine* m, Value* axis, Value* lhs, Value* rhs);    // ⊤ dyadic (encode/representation)

// PrimitiveFn structs that combine monadic and dyadic forms
extern PrimitiveFn prim_plus;      // + symbol
extern PrimitiveFn prim_minus;     // - symbol
extern PrimitiveFn prim_times;     // × symbol
extern PrimitiveFn prim_divide;    // ÷ symbol
extern PrimitiveFn prim_star;      // * symbol
extern PrimitiveFn prim_equal;     // = symbol
extern PrimitiveFn prim_not_equal; // ≠ symbol
extern PrimitiveFn prim_less;      // < symbol
extern PrimitiveFn prim_greater;   // > symbol
extern PrimitiveFn prim_less_eq;   // ≤ symbol
extern PrimitiveFn prim_greater_eq;// ≥ symbol
extern PrimitiveFn prim_ceiling;   // ⌈ symbol
extern PrimitiveFn prim_floor;     // ⌊ symbol
extern PrimitiveFn prim_and;       // ∧ symbol
extern PrimitiveFn prim_or;        // ∨ symbol
extern PrimitiveFn prim_not;       // ~ symbol
extern PrimitiveFn prim_nand;      // ⍲ symbol
extern PrimitiveFn prim_nor;       // ⍱ symbol
extern PrimitiveFn prim_stile;     // | symbol (magnitude/residue)
extern PrimitiveFn prim_log;       // ⍟ symbol (logarithm)
extern PrimitiveFn prim_factorial; // ! symbol (factorial/binomial)
extern PrimitiveFn prim_reverse;   // ⌽ symbol (reverse/rotate)
extern PrimitiveFn prim_reverse_first; // ⊖ symbol (reverse first/rotate first)
extern PrimitiveFn prim_tally;     // ≢ symbol (tally)
extern PrimitiveFn prim_depth;     // ≡ symbol (depth)
extern PrimitiveFn prim_member;    // ∊ symbol (enlist/member)
extern PrimitiveFn prim_grade_up;  // ⍋ symbol (grade up)
extern PrimitiveFn prim_grade_down;// ⍒ symbol (grade down)
extern PrimitiveFn prim_union;     // ∪ symbol (unique/union)
extern PrimitiveFn prim_circle;    // ○ symbol (pi times/circular)
extern PrimitiveFn prim_question;  // ? symbol (roll/deal)
extern PrimitiveFn prim_decode;    // ⊥ symbol (decode/base value)
extern PrimitiveFn prim_encode;    // ⊤ symbol (encode/representation)
extern PrimitiveFn prim_execute;   // ⍎ symbol (execute)
extern PrimitiveFn prim_format;    // ⍕ symbol (format)
extern PrimitiveFn prim_table;     // ⍪ symbol (table/catenate first)
extern PrimitiveFn prim_left;      // ⊣ symbol (left - ISO 10.2.17)
extern PrimitiveFn prim_right;     // ⊢ symbol (right - ISO 10.2.18)
extern PrimitiveFn prim_enclose;   // ⊂ symbol (enclose - ISO 10.2.26)
extern PrimitiveFn prim_disclose;  // ⊃ symbol (disclose/pick - ISO 10.2.22/24)

// Identity functions (ISO 10.2.17-18)
void fn_right(Machine* m, Value* axis, Value* omega);                    // ⊢ monadic (identity)
void fn_left(Machine* m, Value* axis, Value* alpha, Value* omega);       // ⊣ dyadic (return left)
void fn_right_dyadic(Machine* m, Value* axis, Value* alpha, Value* omega); // ⊢ dyadic (return right)

// Array operation functions
void fn_shape(Machine* m, Value* axis, Value* omega);                    // ⍴ monadic (get shape)
void fn_reshape(Machine* m, Value* axis, Value* lhs, Value* rhs);        // ⍴ dyadic (reshape)
void fn_ravel(Machine* m, Value* axis, Value* omega);                    // , monadic (flatten to vector)
void fn_catenate(Machine* m, Value* axis, Value* lhs, Value* rhs);       // , dyadic (concatenate)
void fn_transpose(Machine* m, Value* axis, Value* omega);                // ⍉ monadic (transpose)
void fn_dyadic_transpose(Machine* m, Value* axis, Value* lhs, Value* rhs); // ⍉ dyadic (reorder axes)
void fn_matrix_inverse(Machine* m, Value* axis, Value* omega);           // ⌹ monadic (matrix inverse)
void fn_matrix_divide(Machine* m, Value* axis, Value* lhs, Value* rhs);  // ⌹ dyadic (matrix divide)
void fn_iota(Machine* m, Value* axis, Value* omega);                     // ⍳ monadic (index generator)
void fn_first(Machine* m, Value* axis, Value* omega);                    // ↑ monadic (first element)
void fn_take(Machine* m, Value* axis, Value* lhs, Value* rhs);           // ↑ dyadic (take)
void fn_drop(Machine* m, Value* axis, Value* lhs, Value* rhs);           // ↓ dyadic (drop)
void fn_squad(Machine* m, Value* axis, Value* lhs, Value* rhs);          // ⌷ dyadic (squad/index)

// Array operation PrimitiveFn structs
extern PrimitiveFn prim_rho;       // ⍴ symbol (shape/reshape)
extern PrimitiveFn prim_comma;     // , symbol (ravel/catenate)
extern PrimitiveFn prim_transpose; // ⍉ symbol (transpose)
extern PrimitiveFn prim_domino;    // ⌹ symbol (matrix inverse/divide)
extern PrimitiveFn prim_iota;      // ⍳ symbol (index generator)
extern PrimitiveFn prim_uptack;    // ↑ symbol (first/take)
extern PrimitiveFn prim_downtack;  // ↓ symbol (drop)
extern PrimitiveFn prim_squad;     // ⌷ symbol (index/squad) - used by A[I] syntax

// Error handling system functions (ISO 13751 §11.4.4-11.6.5)
void fn_quad_es(Machine* m, Value* axis, Value* omega);           // ⎕ES monadic (signal error)
void fn_quad_es_dyadic(Machine* m, Value* axis, Value* lhs, Value* rhs); // ⎕ES dyadic (signal with message)
void fn_quad_ea(Machine* m, Value* axis, Value* lhs, Value* rhs); // ⎕EA dyadic (execute alternate)

extern PrimitiveFn prim_quad_es;   // ⎕ES (error signal)
extern PrimitiveFn prim_quad_ea;   // ⎕EA (execute alternate)

// Other system functions (ISO 13751 §11.5)
void fn_quad_dl(Machine* m, Value* axis, Value* omega);           // ⎕DL monadic (delay)
void fn_quad_nc(Machine* m, Value* axis, Value* omega);           // ⎕NC monadic (name class)
void fn_quad_ex(Machine* m, Value* axis, Value* omega);           // ⎕EX monadic (expunge)
void fn_quad_nl(Machine* m, Value* axis, Value* omega);           // ⎕NL monadic (name list)
void fn_quad_nl_dyadic(Machine* m, Value* axis, Value* lhs, Value* rhs); // ⎕NL dyadic (filtered name list)

extern PrimitiveFn prim_quad_dl;   // ⎕DL (delay)
extern PrimitiveFn prim_quad_nc;   // ⎕NC (name class)
extern PrimitiveFn prim_quad_ex;   // ⎕EX (expunge)
extern PrimitiveFn prim_quad_nl;   // ⎕NL (name list)

} // namespace apl
