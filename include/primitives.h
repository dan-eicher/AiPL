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

// PrimitiveFn structs that combine monadic and dyadic forms
extern PrimitiveFn prim_plus;      // + symbol
extern PrimitiveFn prim_minus;     // - symbol
extern PrimitiveFn prim_times;     // × symbol
extern PrimitiveFn prim_divide;    // ÷ symbol
extern PrimitiveFn prim_star;      // * symbol

} // namespace apl
