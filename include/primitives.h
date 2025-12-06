// Primitives - APL primitive functions and operations

#pragma once

#include "value.h"

namespace apl {

// Forward declaration
class Machine;

// PrimitiveFn is defined in value.h

// Monadic built-in functions
// All primitives set machine->ctrl.value on success or push ThrowErrorK on error
void fn_conjugate(Machine* m, Value* omega);     // + monadic (conjugate/identity)
void fn_negate(Machine* m, Value* omega);        // - monadic (negation)
void fn_signum(Machine* m, Value* omega);        // × monadic (signum/sign)
void fn_reciprocal(Machine* m, Value* omega);    // ÷ monadic (reciprocal)
void fn_exponential(Machine* m, Value* omega);   // * monadic (exponential e^x)
void fn_ceiling(Machine* m, Value* omega);       // ⌈ monadic (ceiling)
void fn_floor(Machine* m, Value* omega);         // ⌊ monadic (floor)
void fn_not(Machine* m, Value* omega);           // ~ monadic (not)
void fn_magnitude(Machine* m, Value* omega);     // | monadic (absolute value)
void fn_natural_log(Machine* m, Value* omega);   // ⍟ monadic (natural logarithm)
void fn_factorial(Machine* m, Value* omega);     // ! monadic (factorial)

// Dyadic built-in functions
void fn_add(Machine* m, Value* lhs, Value* rhs);       // + dyadic (addition)
void fn_subtract(Machine* m, Value* lhs, Value* rhs);  // - dyadic (subtraction)
void fn_multiply(Machine* m, Value* lhs, Value* rhs);  // × dyadic (multiplication)
void fn_divide(Machine* m, Value* lhs, Value* rhs);    // ÷ dyadic (division)
void fn_power(Machine* m, Value* lhs, Value* rhs);     // * dyadic (power)
void fn_equal(Machine* m, Value* lhs, Value* rhs);     // = dyadic (equality)
void fn_not_equal(Machine* m, Value* lhs, Value* rhs); // ≠ dyadic (not equal)
void fn_less(Machine* m, Value* lhs, Value* rhs);      // < dyadic (less than)
void fn_greater(Machine* m, Value* lhs, Value* rhs);   // > dyadic (greater than)
void fn_less_eq(Machine* m, Value* lhs, Value* rhs);   // ≤ dyadic (less or equal)
void fn_greater_eq(Machine* m, Value* lhs, Value* rhs);// ≥ dyadic (greater or equal)
void fn_maximum(Machine* m, Value* lhs, Value* rhs);   // ⌈ dyadic (maximum)
void fn_minimum(Machine* m, Value* lhs, Value* rhs);   // ⌊ dyadic (minimum)
void fn_and(Machine* m, Value* lhs, Value* rhs);       // ∧ dyadic (and/lcm)
void fn_or(Machine* m, Value* lhs, Value* rhs);        // ∨ dyadic (or/gcd)
void fn_nand(Machine* m, Value* lhs, Value* rhs);      // ⍲ dyadic (nand)
void fn_nor(Machine* m, Value* lhs, Value* rhs);       // ⍱ dyadic (nor)
void fn_residue(Machine* m, Value* lhs, Value* rhs);   // | dyadic (modulo/residue)
void fn_logarithm(Machine* m, Value* lhs, Value* rhs); // ⍟ dyadic (logarithm base)
void fn_binomial(Machine* m, Value* lhs, Value* rhs);  // ! dyadic (binomial/combinations)

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

// Array operation functions
void fn_shape(Machine* m, Value* omega);                    // ⍴ monadic (get shape)
void fn_reshape(Machine* m, Value* lhs, Value* rhs);        // ⍴ dyadic (reshape)
void fn_ravel(Machine* m, Value* omega);                    // , monadic (flatten to vector)
void fn_catenate(Machine* m, Value* lhs, Value* rhs);       // , dyadic (concatenate)
void fn_transpose(Machine* m, Value* omega);                // ⍉ monadic (transpose)
void fn_iota(Machine* m, Value* omega);                     // ⍳ monadic (index generator)
void fn_take(Machine* m, Value* lhs, Value* rhs);           // ↑ dyadic (take)
void fn_drop(Machine* m, Value* lhs, Value* rhs);           // ↓ dyadic (drop)

// Array operation PrimitiveFn structs
extern PrimitiveFn prim_rho;       // ⍴ symbol (shape/reshape)
extern PrimitiveFn prim_comma;     // , symbol (ravel/catenate)
extern PrimitiveFn prim_transpose; // ⍉ symbol (transpose)
extern PrimitiveFn prim_iota;      // ⍳ symbol (index generator)
extern PrimitiveFn prim_uptack;    // ↑ symbol (take)
extern PrimitiveFn prim_downtack;  // ↓ symbol (drop)

} // namespace apl
