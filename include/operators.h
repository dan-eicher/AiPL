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
void op_commute(Machine* m, Value* f, Value* omega);    // ⍨ duplicate (monadic)

// Helper for commute (dyadic form - not exposed as PrimitiveOp yet)
void op_commute_dyadic(Machine* m, Value* lhs, Value* f, Value* g, Value* rhs);  // ⍨ commute (dyadic)

// Reduction and Scan operators
// These are operators (not primitives) because they take functions as operands
void fn_reduce(Machine* m, Value* func, Value* omega);         // / reduce along last axis
void fn_reduce_first(Machine* m, Value* func, Value* omega);   // ⌿ reduce along first axis
void fn_scan(Machine* m, Value* func, Value* omega);           // \ scan along last axis
void fn_scan_first(Machine* m, Value* func, Value* omega);     // ⍀ scan along first axis

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
void op_rank_monadic(Machine* m, Value* f, Value* omega);
void op_rank(Machine* m, Value* lhs, Value* f, Value* rank_spec, Value* rhs);

} // namespace apl
