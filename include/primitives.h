// Primitives - APL primitive functions and operations

#pragma once

#include "value.h"

namespace apl {

// Forward declaration
class Machine;

// PrimitiveFn is defined in value.h

// Monadic built-in functions
Value* fn_conjugate(Value* omega);     // + monadic (conjugate/identity)
Value* fn_negate(Value* omega);        // - monadic (negation)
Value* fn_signum(Value* omega);        // × monadic (signum/sign)
Value* fn_reciprocal(Value* omega);    // ÷ monadic (reciprocal)
Value* fn_exponential(Value* omega);   // * monadic (exponential e^x)

// Dyadic built-in functions
Value* fn_add(Value* lhs, Value* rhs);       // + dyadic (addition)
Value* fn_subtract(Value* lhs, Value* rhs);  // - dyadic (subtraction)
Value* fn_multiply(Value* lhs, Value* rhs);  // × dyadic (multiplication)
Value* fn_divide(Value* lhs, Value* rhs);    // ÷ dyadic (division)
Value* fn_power(Value* lhs, Value* rhs);     // * dyadic (power)
Value* fn_equal(Value* lhs, Value* rhs);     // = dyadic (equality)

// PrimitiveFn structs that combine monadic and dyadic forms
extern PrimitiveFn prim_plus;      // + symbol
extern PrimitiveFn prim_minus;     // - symbol
extern PrimitiveFn prim_times;     // × symbol
extern PrimitiveFn prim_divide;    // ÷ symbol
extern PrimitiveFn prim_star;      // * symbol
extern PrimitiveFn prim_equal;     // = symbol

// Array operation functions
Value* fn_shape(Value* omega);                    // ⍴ monadic (get shape)
Value* fn_reshape(Value* lhs, Value* rhs);        // ⍴ dyadic (reshape)
Value* fn_ravel(Value* omega);                    // , monadic (flatten to vector)
Value* fn_catenate(Value* lhs, Value* rhs);       // , dyadic (concatenate)
Value* fn_transpose(Value* omega);                // ⍉ monadic (transpose)
Value* fn_iota(Value* omega);                     // ⍳ monadic (index generator)
Value* fn_take(Value* lhs, Value* rhs);           // ↑ dyadic (take)
Value* fn_drop(Value* lhs, Value* rhs);           // ↓ dyadic (drop)

// Array operation PrimitiveFn structs
extern PrimitiveFn prim_rho;       // ⍴ symbol (shape/reshape)
extern PrimitiveFn prim_comma;     // , symbol (ravel/catenate)
extern PrimitiveFn prim_transpose; // ⍉ symbol (transpose)
extern PrimitiveFn prim_iota;      // ⍳ symbol (index generator)
extern PrimitiveFn prim_uptack;    // ↑ symbol (take)
extern PrimitiveFn prim_downtack;  // ↓ symbol (drop)

// Reduction/scan operation functions
// These take a dyadic function and apply it across an array
Value* fn_reduce(Value* func, Value* omega);         // / reduce along last axis
Value* fn_reduce_first(Value* func, Value* omega);   // ⌿ reduce along first axis
Value* fn_scan(Value* func, Value* omega);           // \ scan along last axis
Value* fn_scan_first(Value* func, Value* omega);     // ⍀ scan along first axis

// Note: Reduction operators are higher-order - they take functions as arguments
// For now, we'll implement them as regular functions that take a PrimitiveFn*
// extracted from a Value. Full operator support comes in Phase 5.

} // namespace apl
