// Operators implementation

#include "operators.h"
#include "machine.h"
#include "heap.h"
#include "primitives.h"
#include <Eigen/Dense>

namespace apl {

// ========================================================================
// Outer Product Operator: A ∘.f B
// ========================================================================
// Result shape: (⍴A),⍴B
// For each item I in A and J in B: Z[I,J] = I f J
//
// Example: 10 20 30 ∘.+ 1 2 3
//          11 12 13
//          21 22 23
//          31 32 33

void op_outer_product(Machine* m, Value* lhs, Value* f, Value* g, Value* rhs) {
    // Outer product only uses the left function operand f
    // g should be nullptr for outer product (∘.f syntax)

    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: outer product requires a function operand"));
        return;
    }

    // Get the primitive function
    if (!f->is_primitive()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: outer product currently only supports primitive functions"));
        return;
    }

    PrimitiveFn* fn = f->data.primitive_fn;
    if (!fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: outer product requires a dyadic function"));
        return;
    }

    // Get dimensions of both arguments
    int lhs_rows = lhs->rows();
    int lhs_cols = lhs->cols();
    int lhs_size = lhs_rows * lhs_cols;

    int rhs_rows = rhs->rows();
    int rhs_cols = rhs->cols();
    int rhs_size = rhs_rows * rhs_cols;

    // Result dimensions for outer product
    // If A is scalar: result has shape of B
    // If B is scalar: result has shape of A
    // If both arrays: result is (⍴A),⍴B

    // For now, implement the simple 2D case: both are vectors or scalars
    // Full spec would handle arbitrary rank combinations

    int result_rows = lhs_size;  // Total elements in lhs
    int result_cols = rhs_size;  // Total elements in rhs

    // Create result matrix on stack (GC will manage the final copy)
    Eigen::MatrixXd result(result_rows, result_cols);

    // Apply f to every combination
    // Outer product uses raveled values (treated as 1D)
    for (int i = 0; i < lhs_size; i++) {
        for (int j = 0; j < rhs_size; j++) {
            // Get scalar values from raveled arrays
            double lhs_val, rhs_val;

            if (lhs->is_scalar()) {
                lhs_val = lhs->as_scalar();
            } else {
                const Eigen::MatrixXd* lhs_mat = lhs->as_matrix();
                lhs_val = (*lhs_mat)(i / lhs_cols, i % lhs_cols);
            }

            if (rhs->is_scalar()) {
                rhs_val = rhs->as_scalar();
            } else {
                const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();
                rhs_val = (*rhs_mat)(j / rhs_cols, j % rhs_cols);
            }

            // Create temporary scalar values for function application
            Value* temp_lhs = m->heap->allocate_scalar(lhs_val);
            Value* temp_rhs = m->heap->allocate_scalar(rhs_val);

            // Apply the function
            fn->dyadic(m, temp_lhs, temp_rhs);

            // Check for error
            if (!m->kont_stack.empty() && dynamic_cast<ThrowErrorK*>(m->kont_stack.back())) {
                return;  // Error continuation already pushed
            }

            // Get result from ctrl.value
            Value* item_result = m->ctrl.value;
            if (!item_result || !item_result->is_scalar()) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: outer product function must return scalar"));
                return;
            }

            result(i, j) = item_result->as_scalar();
        }
    }

    // Return result
    // If result is 1×1, return as scalar
    if (result_rows == 1 && result_cols == 1) {
        m->ctrl.set_value(m->heap->allocate_scalar(result(0, 0)));
    } else if (result_cols == 1) {
        // Return as vector
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        // Return as matrix
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
}

// ========================================================================
// Inner Product Operator: A f.g B
// ========================================================================
// Placeholder - to be implemented

void op_inner_product(Machine* m, Value* lhs, Value* f, Value* g, Value* rhs) {
    m->push_kont(m->heap->allocate<ThrowErrorK>("NOT IMPLEMENTED: inner product"));
}

// ========================================================================
// Each Operator: f¨B (monadic) or A f¨B (dyadic)
// ========================================================================
// Placeholder - to be implemented

void op_each(Machine* m, Value* f, Value* omega) {
    m->push_kont(m->heap->allocate<ThrowErrorK>("NOT IMPLEMENTED: each operator"));
}

// ========================================================================
// Commute/Duplicate Operator: f⍨B (monadic) or A f⍨B (dyadic)
// ========================================================================
// Placeholder - to be implemented

void op_commute(Machine* m, Value* f, Value* omega) {
    m->push_kont(m->heap->allocate<ThrowErrorK>("NOT IMPLEMENTED: commute/duplicate operator"));
}

// ========================================================================
// PrimitiveOp struct definitions
// ========================================================================

PrimitiveOp op_dot = {
    ".",
    nullptr,              // No monadic form
    op_inner_product      // Dyadic: A f.g B
};

PrimitiveOp op_outer_dot = {
    "∘.",
    nullptr,              // No monadic form
    op_outer_product      // Dyadic: A ∘.f B
};

PrimitiveOp op_diaeresis = {
    "¨",
    op_each,              // Monadic: f¨B
    nullptr               // Dyadic form uses different evaluation (not yet implemented)
};

PrimitiveOp op_tilde = {
    "⍨",
    op_commute,           // Monadic: f⍨B (duplicate)
    nullptr               // Dyadic: A f⍨B (commute) - uses different evaluation
};

} // namespace apl
