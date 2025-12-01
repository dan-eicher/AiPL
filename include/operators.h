// Operators - APL operators (higher-order functions)

#pragma once

#include "value.h"

namespace apl {

// Forward declaration
class Machine;

// PrimitiveOp is defined in value.h

// Dyadic operators
void op_outer_product(Machine* m, Value* lhs, Value* f, Value* g, Value* rhs);  // ∘. outer product
void op_inner_product(Machine* m, Value* lhs, Value* f, Value* g, Value* rhs);  // . inner product

// Monadic operators
void op_each(Machine* m, Value* f, Value* omega);       // ¨ each (apply to each element)
void op_commute(Machine* m, Value* f, Value* omega);    // ⍨ commute/duplicate

// PrimitiveOp structs that combine monadic and dyadic forms
extern PrimitiveOp op_dot;         // . operator (inner product dyadic)
extern PrimitiveOp op_outer_dot;   // ∘. operator (outer product)
extern PrimitiveOp op_diaeresis;   // ¨ operator (each)
extern PrimitiveOp op_tilde;       // ⍨ operator (commute/duplicate)

} // namespace apl
