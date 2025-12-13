// Operators - APL operators (higher-order functions)

#pragma once

#include "value.h"

namespace apl {

// Forward declaration
class Machine;

// PrimitiveOp is defined in value.h
// All operator functions now take axis as first parameter after machine
// This unifies axis handling across all operators

// Dyadic operators (axis, lhs, f, g, rhs)
void op_outer_product(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs);  // ∘. outer product
void op_inner_product(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs);  // . inner product
void op_each_dyadic(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs);    // ¨ each (dyadic)
void op_commute_dyadic(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs); // ⍨ commute (dyadic)

// Monadic operators (axis, f, omega)
void op_each(Machine* m, Value* axis, Value* f, Value* omega);       // ¨ each (apply to each element)
void op_commute(Machine* m, Value* axis, Value* f, Value* omega);    // ⍨ duplicate (monadic)

// Reduction and Scan operators - unified with axis support
void fn_reduce(Machine* m, Value* axis, Value* func, Value* omega);       // f/B or f/[k]B
void fn_reduce_first(Machine* m, Value* axis, Value* func, Value* omega); // f⌿B or f⌿[k]B
void fn_scan(Machine* m, Value* axis, Value* func, Value* omega);         // f\B or f\[k]B
void fn_scan_first(Machine* m, Value* axis, Value* func, Value* omega);   // f⍀B or f⍀[k]B

// N-wise reduction (dyadic forms) - unified with axis support
void fn_reduce_nwise(Machine* m, Value* axis, Value* lhs, Value* func, Value* g, Value* rhs);       // N f/B or N f/[k]B
void fn_reduce_first_nwise(Machine* m, Value* axis, Value* lhs, Value* func, Value* g, Value* rhs); // N f⌿B or N f⌿[k]B

// PrimitiveOp structs that combine monadic and dyadic forms
extern PrimitiveOp op_dot;           // . operator (inner product dyadic)
extern PrimitiveOp op_outer_dot;     // ∘. operator (outer product)
extern PrimitiveOp op_diaeresis;     // ¨ operator (each)
extern PrimitiveOp op_tilde;         // ⍨ operator (commute/duplicate)
extern PrimitiveOp op_reduce;        // / operator (reduce)
extern PrimitiveOp op_reduce_first;  // ⌿ operator (reduce first axis)
extern PrimitiveOp op_scan;          // \ operator (scan)
extern PrimitiveOp op_scan_first;    // ⍀ operator (scan first axis)
extern PrimitiveOp op_rank_op;       // ⍤ operator (rank)

// Rank operator functions
void op_rank_monadic(Machine* m, Value* axis, Value* f, Value* omega);
void op_rank(Machine* m, Value* axis, Value* lhs, Value* f, Value* rank_spec, Value* rhs);

// Catenate/Laminate with axis (,[k])
// When k is near-integer: catenate along axis k
// When k is fractional: laminate (create new axis at ⌊k)
void fn_catenate_axis_monadic(Machine* m, Value* curry_axis, Value* axis_operand, Value* omega);
void fn_catenate_axis_dyadic(Machine* m, Value* curry_axis, Value* lhs, Value* axis_operand, Value* unused, Value* rhs);
extern PrimitiveOp op_catenate_axis;  // ,[k] operator

} // namespace apl
