// Primitives implementation

#include "primitives.h"
#include "value.h"
#include "machine.h"
#include "continuation.h"
#include <cmath>
#include <stdexcept>

namespace apl {

// PrimitiveFn structs combining monadic and dyadic forms
PrimitiveFn prim_plus    = { "+", fn_conjugate, fn_add };
PrimitiveFn prim_minus   = { "-", fn_negate, fn_subtract };
PrimitiveFn prim_times   = { "×", fn_signum, fn_multiply };
PrimitiveFn prim_divide  = { "÷", fn_reciprocal, fn_divide };
PrimitiveFn prim_star    = { "*", fn_exponential, fn_power };
PrimitiveFn prim_equal   = { "=", nullptr, fn_equal };  // No monadic form for equals
PrimitiveFn prim_not_equal = { "≠", nullptr, fn_not_equal };
PrimitiveFn prim_less      = { "<", nullptr, fn_less };
PrimitiveFn prim_greater   = { ">", nullptr, fn_greater };
PrimitiveFn prim_less_eq   = { "≤", nullptr, fn_less_eq };
PrimitiveFn prim_greater_eq = { "≥", nullptr, fn_greater_eq };
PrimitiveFn prim_ceiling   = { "⌈", fn_ceiling, fn_maximum };
PrimitiveFn prim_floor     = { "⌊", fn_floor, fn_minimum };
PrimitiveFn prim_and       = { "∧", nullptr, fn_and };
PrimitiveFn prim_or        = { "∨", nullptr, fn_or };
PrimitiveFn prim_not       = { "~", fn_not, nullptr };
PrimitiveFn prim_nand      = { "⍲", nullptr, fn_nand };
PrimitiveFn prim_nor       = { "⍱", nullptr, fn_nor };
PrimitiveFn prim_stile     = { "|", fn_magnitude, fn_residue };
PrimitiveFn prim_log       = { "⍟", fn_natural_log, fn_logarithm };
PrimitiveFn prim_factorial = { "!", fn_factorial, fn_binomial };

// Array operation primitives
PrimitiveFn prim_rho       = { "⍴", fn_shape, fn_reshape };
PrimitiveFn prim_comma     = { ",", fn_ravel, fn_catenate };
PrimitiveFn prim_transpose = { "⍉", fn_transpose, nullptr };
PrimitiveFn prim_iota      = { "⍳", fn_iota, fn_index_of };
PrimitiveFn prim_uptack    = { "↑", nullptr, fn_take };
PrimitiveFn prim_downtack  = { "↓", nullptr, fn_drop };
PrimitiveFn prim_reverse   = { "⌽", fn_reverse, fn_rotate };
PrimitiveFn prim_reverse_first = { "⊖", fn_reverse_first, fn_rotate_first };
PrimitiveFn prim_tally     = { "≢", fn_tally, nullptr };
PrimitiveFn prim_member    = { "∊", fn_enlist, fn_member_of };

// ============================================================================
// Dyadic Arithmetic Functions
// ============================================================================

// Addition (+)
void fn_add(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar + scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar + rhs->data.scalar);
        return;
    }

    // Scalar extension using Eigen broadcasting
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar + rhs->as_matrix()->array();
        // Preserve vector/matrix distinction
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() + rhs->data.scalar;
        // Preserve vector/matrix distinction
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array + Array: element-wise
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    // Shape checking
    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in addition"));
        return;
    }

    Eigen::MatrixXd result = lmat->array() + rmat->array();
    // Preserve vector/matrix distinction
    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Subtraction (-)
void fn_subtract(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar - scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar - rhs->data.scalar);
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar - rhs->as_matrix()->array();
        // Preserve vector/matrix distinction
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() - rhs->data.scalar;
        // Preserve vector/matrix distinction
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array - Array
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in subtraction"));
        return;
    }

    Eigen::MatrixXd result = lmat->array() - rmat->array();
    // Preserve vector/matrix distinction
    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Multiplication (×)
void fn_multiply(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar × scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar * rhs->data.scalar);
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar * rhs->as_matrix()->array();
        // Preserve vector/matrix distinction
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() * rhs->data.scalar;
        // Preserve vector/matrix distinction
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array × Array: element-wise multiplication
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in multiplication"));
        return;
    }

    Eigen::MatrixXd result = lmat->array() * rmat->array();
    // Preserve vector/matrix distinction
    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Division (÷)
void fn_divide(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar ÷ scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        if (rhs->data.scalar == 0.0) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: division by zero"));
            return;
        }
        m->result = m->heap->allocate_scalar(lhs->data.scalar / rhs->data.scalar);
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        // Check for zeros in divisor
        if ((rmat->array() == 0.0).any()) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: division by zero"));
            return;
        }
        Eigen::MatrixXd result =
            lhs->data.scalar / rmat->array();
        // Preserve vector/matrix distinction
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        if (rhs->data.scalar == 0.0) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: division by zero"));
            return;
        }
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() / rhs->data.scalar;
        // Preserve vector/matrix distinction
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array ÷ Array
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in division"));
        return;
    }

    // Check for zeros in divisor
    if ((rmat->array() == 0.0).any()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: division by zero"));
        return;
    }

    Eigen::MatrixXd result = lmat->array() / rmat->array();
    // Preserve vector/matrix distinction
    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Power (*)
void fn_power(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar * scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::pow(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        // lhs is scalar base, rhs is array of exponents: lhs^rhs[i]
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = std::pow(lhs->data.scalar, rmat->data()[i]);
        }
        // Preserve vector/matrix distinction
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        // lhs is array of bases, rhs is scalar exponent
        Eigen::MatrixXd result =
            lhs->as_matrix()->array().pow(rhs->data.scalar);
        // Preserve vector/matrix distinction
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array * Array: element-wise power
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in power"));
        return;
    }

    Eigen::MatrixXd result = lmat->array().pow(rmat->array());
    // Preserve vector/matrix distinction
    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Equality (=)
void fn_equal(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar = scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar == rhs->data.scalar ? 1.0 : 0.0);
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar == rmat->data()[i]) ? 1.0 : 0.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] == rhs->data.scalar) ? 1.0 : 0.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array = Array: element-wise equality
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in equality"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] == rmat->data()[i]) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Not Equal (≠)
void fn_not_equal(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar ≠ scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar != rhs->data.scalar ? 1.0 : 0.0);
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar != rmat->data()[i]) ? 1.0 : 0.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] != rhs->data.scalar) ? 1.0 : 0.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array ≠ Array: element-wise not equal
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in not-equal"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] != rmat->data()[i]) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Less Than (<)
void fn_less(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar < scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar < rhs->data.scalar ? 1.0 : 0.0);
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar < rmat->data()[i]) ? 1.0 : 0.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] < rhs->data.scalar) ? 1.0 : 0.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array < Array: element-wise less than
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in less-than"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] < rmat->data()[i]) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Greater Than (>)
void fn_greater(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar > scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar > rhs->data.scalar ? 1.0 : 0.0);
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar > rmat->data()[i]) ? 1.0 : 0.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] > rhs->data.scalar) ? 1.0 : 0.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array > Array: element-wise greater than
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in greater-than"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] > rmat->data()[i]) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Less Than or Equal (≤)
void fn_less_eq(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar ≤ scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar <= rhs->data.scalar ? 1.0 : 0.0);
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar <= rmat->data()[i]) ? 1.0 : 0.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] <= rhs->data.scalar) ? 1.0 : 0.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array ≤ Array: element-wise less than or equal
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in less-or-equal"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] <= rmat->data()[i]) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Greater Than or Equal (≥)
void fn_greater_eq(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar ≥ scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar >= rhs->data.scalar ? 1.0 : 0.0);
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar >= rmat->data()[i]) ? 1.0 : 0.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] >= rhs->data.scalar) ? 1.0 : 0.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Array ≥ Array: element-wise greater than or equal
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in greater-or-equal"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] >= rmat->data()[i]) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// ============================================================================
// Min/Max Functions (⌈ ⌊)
// ============================================================================

// Maximum (⌈) - dyadic
void fn_maximum(Machine* m, Value* lhs, Value* rhs) {
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::max(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result = rmat->array().max(lhs->data.scalar);
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result = lmat->array().max(rhs->data.scalar);
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in maximum"));
        return;
    }

    Eigen::MatrixXd result = lmat->array().max(rmat->array());

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Minimum (⌊) - dyadic
void fn_minimum(Machine* m, Value* lhs, Value* rhs) {
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::min(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result = rmat->array().min(lhs->data.scalar);
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result = lmat->array().min(rhs->data.scalar);
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in minimum"));
        return;
    }

    Eigen::MatrixXd result = lmat->array().min(rmat->array());

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Ceiling (⌈) - monadic
void fn_ceiling(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::ceil(omega->data.scalar));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = mat->array().ceil();

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Floor (⌊) - monadic
void fn_floor(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::floor(omega->data.scalar));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = mat->array().floor();

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// ============================================================================
// Logical Functions (∧ ∨ ~ ⍲ ⍱)
// ============================================================================

// And (∧) - dyadic
// For booleans: logical AND
// For integers: LCM (Least Common Multiple) - not implemented yet
void fn_and(Machine* m, Value* lhs, Value* rhs) {
    if (lhs->is_scalar() && rhs->is_scalar()) {
        // Boolean interpretation: both non-zero
        double result = (lhs->data.scalar != 0.0 && rhs->data.scalar != 0.0) ? 1.0 : 0.0;
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar != 0.0 && rmat->data()[i] != 0.0) ? 1.0 : 0.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] != 0.0 && rhs->data.scalar != 0.0) ? 1.0 : 0.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in and"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] != 0.0 && rmat->data()[i] != 0.0) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Or (∨) - dyadic
// For booleans: logical OR
// For integers: GCD (Greatest Common Divisor) - not implemented yet
void fn_or(Machine* m, Value* lhs, Value* rhs) {
    if (lhs->is_scalar() && rhs->is_scalar()) {
        double result = (lhs->data.scalar != 0.0 || rhs->data.scalar != 0.0) ? 1.0 : 0.0;
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar != 0.0 || rmat->data()[i] != 0.0) ? 1.0 : 0.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] != 0.0 || rhs->data.scalar != 0.0) ? 1.0 : 0.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in or"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] != 0.0 || rmat->data()[i] != 0.0) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Not (~) - monadic
void fn_not(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->data.scalar == 0.0 ? 1.0 : 0.0);
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result(mat->rows(), mat->cols());
    for (int i = 0; i < mat->size(); ++i) {
        result(i) = (mat->data()[i] == 0.0) ? 1.0 : 0.0;
    }

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Nand (⍲) - dyadic
void fn_nand(Machine* m, Value* lhs, Value* rhs) {
    if (lhs->is_scalar() && rhs->is_scalar()) {
        double result = (lhs->data.scalar != 0.0 && rhs->data.scalar != 0.0) ? 0.0 : 1.0;
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar != 0.0 && rmat->data()[i] != 0.0) ? 0.0 : 1.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] != 0.0 && rhs->data.scalar != 0.0) ? 0.0 : 1.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in nand"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] != 0.0 && rmat->data()[i] != 0.0) ? 0.0 : 1.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Nor (⍱) - dyadic
void fn_nor(Machine* m, Value* lhs, Value* rhs) {
    if (lhs->is_scalar() && rhs->is_scalar()) {
        double result = (lhs->data.scalar != 0.0 || rhs->data.scalar != 0.0) ? 0.0 : 1.0;
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = (lhs->data.scalar != 0.0 || rmat->data()[i] != 0.0) ? 0.0 : 1.0;
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = (lmat->data()[i] != 0.0 || rhs->data.scalar != 0.0) ? 0.0 : 1.0;
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in nor"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = (lmat->data()[i] != 0.0 || rmat->data()[i] != 0.0) ? 0.0 : 1.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// ============================================================================
// Arithmetic Extension Functions (| ⍟ !)
// ============================================================================

// Magnitude (|) - monadic (absolute value)
void fn_magnitude(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::abs(omega->data.scalar));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = mat->array().abs();

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Residue (|) - dyadic (modulo: lhs | rhs means rhs mod lhs)
// APL semantics: A|B gives the remainder when B is divided by A
// Result has the same sign as A (or 0)
void fn_residue(Machine* m, Value* lhs, Value* rhs) {
    auto residue = [](double a, double b) -> double {
        if (a == 0.0) return b;  // 0|B = B
        double r = std::fmod(b, a);
        // Adjust sign to match divisor (APL semantics)
        if (r != 0.0 && ((a > 0.0) != (r > 0.0))) {
            r += a;
        }
        return r;
    };

    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(residue(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = residue(lhs->data.scalar, rmat->data()[i]);
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = residue(lmat->data()[i], rhs->data.scalar);
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in residue"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = residue(lmat->data()[i], rmat->data()[i]);
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Natural Logarithm (⍟) - monadic
void fn_natural_log(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::log(omega->data.scalar));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = mat->array().log();

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Logarithm (⍟) - dyadic (lhs ⍟ rhs = log base lhs of rhs)
void fn_logarithm(Machine* m, Value* lhs, Value* rhs) {
    // log_a(b) = ln(b) / ln(a)
    auto log_base = [](double base, double val) -> double {
        return std::log(val) / std::log(base);
    };

    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(log_base(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        double ln_base = std::log(lhs->data.scalar);
        Eigen::MatrixXd result = rmat->array().log() / ln_base;
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        double ln_val = std::log(rhs->data.scalar);
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = ln_val / std::log(lmat->data()[i]);
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in logarithm"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = log_base(lmat->data()[i], rmat->data()[i]);
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Factorial (!) - monadic
// Uses gamma function: n! = gamma(n+1)
void fn_factorial(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::tgamma(omega->data.scalar + 1.0));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result(mat->rows(), mat->cols());
    for (int i = 0; i < mat->size(); ++i) {
        result(i) = std::tgamma(mat->data()[i] + 1.0);
    }

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Binomial (!) - dyadic (lhs ! rhs = "rhs choose lhs" = C(rhs, lhs))
// Uses gamma function: C(n,k) = n! / (k! * (n-k)!) = gamma(n+1) / (gamma(k+1) * gamma(n-k+1))
void fn_binomial(Machine* m, Value* lhs, Value* rhs) {
    auto binomial = [](double k, double n) -> double {
        // C(n,k) using gamma function for generalized binomial
        return std::tgamma(n + 1.0) / (std::tgamma(k + 1.0) * std::tgamma(n - k + 1.0));
    };

    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(binomial(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = binomial(lhs->data.scalar, rmat->data()[i]);
        }
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = binomial(lmat->data()[i], rhs->data.scalar);
        }
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: mismatched shapes in binomial"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = binomial(lmat->data()[i], rmat->data()[i]);
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// ============================================================================
// Monadic Arithmetic Functions
// ============================================================================

// Conjugate/Identity (+)
void fn_conjugate(Machine* m, Value* omega) {
    // For real numbers, identity just returns the value
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->data.scalar);
        return;
    }

    // For arrays, return a copy preserving vector/matrix distinction
    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(omega->as_matrix()->col(0));
    } else {
        m->result = m->heap->allocate_matrix(*omega->as_matrix());
    }
}

// Negation (-)
void fn_negate(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(-omega->data.scalar);
        return;
    }

    Eigen::MatrixXd result = -omega->as_matrix()->array();
    // Preserve vector/matrix distinction
    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Signum/Sign (×)
void fn_signum(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        double val = omega->data.scalar;
        double sign = (val > 0.0) ? 1.0 : (val < 0.0) ? -1.0 : 0.0;
        m->result = m->heap->allocate_scalar(sign);
        return;
    }

    // For arrays, apply sign element-wise
    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result(mat->rows(), mat->cols());

    for (int i = 0; i < mat->rows(); ++i) {
        for (int j = 0; j < mat->cols(); ++j) {
            double val = (*mat)(i, j);
            result(i, j) = (val > 0.0) ? 1.0 : (val < 0.0) ? -1.0 : 0.0;
        }
    }

    // Preserve vector/matrix distinction
    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Reciprocal (÷)
void fn_reciprocal(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        if (omega->data.scalar == 0.0) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reciprocal of zero"));
            return;
        }
        m->result = m->heap->allocate_scalar(1.0 / omega->data.scalar);
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Check for zeros
    if ((mat->array() == 0.0).any()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reciprocal of zero"));
        return;
    }

    Eigen::MatrixXd result = 1.0 / mat->array();
    // Preserve vector/matrix distinction
    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Exponential (*)
void fn_exponential(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::exp(omega->data.scalar));
        return;
    }

    Eigen::MatrixXd result = omega->as_matrix()->array().exp();
    // Preserve vector/matrix distinction
    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// ============================================================================
// Array Operation Functions
// ============================================================================

// Shape (⍴) - monadic: returns shape as vector
void fn_shape(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        // Scalar has empty shape
        Eigen::VectorXd shape(0);
        m->result = m->heap->allocate_vector(shape);
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        // Vector shape is just its length
        Eigen::VectorXd shape(1);
        shape(0) = mat->rows();
        m->result = m->heap->allocate_vector(shape);
        return;
    }

    // Matrix shape is (rows, cols)
    Eigen::VectorXd shape(2);
    shape(0) = mat->rows();
    shape(1) = mat->cols();
    m->result = m->heap->allocate_vector(shape);
}

// Reshape (⍴) - dyadic: reshape rhs to shape given by lhs
void fn_reshape(Machine* m, Value* lhs, Value* rhs) {
    // lhs must be a scalar or vector specifying new shape
    if (!lhs->is_scalar() && !lhs->is_vector()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("RANK ERROR: left argument to reshape must be scalar or vector"));
        return;
    }

    // Get target shape
    int target_rows, target_cols;

    if (lhs->is_scalar()) {
        // Scalar shape means 1D vector of that length
        double dim = lhs->as_scalar();
        // Validate: must be non-negative integer
        if (dim < 0.0) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reshape dimension must be non-negative"));
            return;
        }
        if (dim != std::floor(dim)) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reshape dimension must be an integer"));
            return;
        }
        target_rows = static_cast<int>(dim);
        target_cols = 1;
    } else {
        const Eigen::MatrixXd* shape_mat = lhs->as_matrix();
        if (shape_mat->rows() == 1) {
            // Single element: vector of that length
            double dim = (*shape_mat)(0, 0);
            // Validate: must be non-negative integer
            if (dim < 0.0) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reshape dimension must be non-negative"));
                return;
            }
            if (dim != std::floor(dim)) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reshape dimension must be an integer"));
                return;
            }
            target_rows = static_cast<int>(dim);
            target_cols = 1;
        } else if (shape_mat->rows() == 2) {
            // Two elements: matrix of that shape
            double dim1 = (*shape_mat)(0, 0);
            double dim2 = (*shape_mat)(1, 0);
            // Validate: must be non-negative integers
            if (dim1 < 0.0 || dim2 < 0.0) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reshape dimensions must be non-negative"));
                return;
            }
            if (dim1 != std::floor(dim1) || dim2 != std::floor(dim2)) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reshape dimensions must be integers"));
                return;
            }
            target_rows = static_cast<int>(dim1);
            target_cols = static_cast<int>(dim2);
        } else {
            m->push_kont(m->heap->allocate<ThrowErrorK>("RANK ERROR: reshape shape must have 1 or 2 elements"));
            return;
        }
    }

    int target_size = target_rows * target_cols;

    // Get source data (row-major order per APL)
    Eigen::VectorXd source;
    if (rhs->is_scalar()) {
        source.resize(1);
        source(0) = rhs->as_scalar();
    } else {
        const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();
        // Flatten to vector in row-major order
        int size = rhs_mat->size();
        int cols = rhs_mat->cols();
        source.resize(size);
        for (int i = 0; i < size; ++i) {
            source(i) = (*rhs_mat)(i / cols, i % cols);
        }
    }

    // APL reshape cycles through source data; empty source with non-empty target is an error
    if (source.size() == 0 && target_size > 0) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: cannot reshape empty array to non-empty shape"));
        return;
    }

    // Build result by cycling through source data (row-major order per APL)
    Eigen::MatrixXd result(target_rows, target_cols);
    for (int i = 0; i < target_size; ++i) {
        result(i / target_cols, i % target_cols) = source(i % source.size());
    }

    if (target_cols == 1) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Ravel (,) - monadic: flatten to vector
void fn_ravel(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        Eigen::VectorXd v(1);
        v(0) = omega->as_scalar();
        m->result = m->heap->allocate_vector(v);
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Flatten in row-major order (APL standard)
    int size = mat->size();
    Eigen::VectorXd result(size);
    int rows = mat->rows();
    int cols = mat->cols();
    for (int i = 0; i < size; ++i) {
        result(i) = (*mat)(i / cols, i % cols);
    }
    m->result = m->heap->allocate_vector(result);
}

// Catenate (,) - dyadic: concatenate arrays
void fn_catenate(Machine* m, Value* lhs, Value* rhs) {
    // Convert both to matrices for uniform handling
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    // For vectors or compatible matrices, concatenate along first dimension
    if (lmat->cols() != rmat->cols()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: incompatible shapes for catenation"));
        return;
    }

    Eigen::MatrixXd result(lmat->rows() + rmat->rows(), lmat->cols());
    result << *lmat, *rmat;

    if (result.cols() == 1) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Transpose (⍉) - monadic: reverse dimensions
void fn_transpose(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        // Scalar transpose is identity
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    if (omega->is_vector()) {
        // Vector transpose is identity (returns vector unchanged)
        m->result = m->heap->allocate_vector(omega->as_matrix()->col(0));
        return;
    }

    // Matrix transpose
    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = mat->transpose();
    m->result = m->heap->allocate_matrix(result);
}

// Iota (⍳) - monadic: generate indices from 0 to n-1
void fn_iota(Machine* m, Value* omega) {
    if (!omega->is_scalar()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("RANK ERROR: iota argument must be scalar"));
        return;
    }

    double val = omega->as_scalar();

    // Validate: must be non-negative integer
    if (val < 0.0) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: iota argument must be non-negative"));
        return;
    }
    if (val != std::floor(val)) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: iota argument must be an integer"));
        return;
    }

    int n = static_cast<int>(val);

    Eigen::VectorXd result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = i;
    }
    m->result = m->heap->allocate_vector(result);
}

// Take (↑) - dyadic: take first n elements
void fn_take(Machine* m, Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("RANK ERROR: take count must be scalar"));
        return;
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_scalar()) {
        // Taking from scalar: replicate
        Eigen::VectorXd result(std::abs(n));
        result.setConstant(rhs->as_scalar());
        m->result = m->heap->allocate_vector(result);
        return;
    }

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        int len = mat->rows();
        int abs_n = std::abs(n);

        Eigen::VectorXd result(abs_n);

        if (n >= 0) {
            // Take from beginning
            for (int i = 0; i < abs_n; ++i) {
                result(i) = (i < len) ? (*mat)(i, 0) : 0.0;
            }
        } else {
            // Take from end
            for (int i = 0; i < abs_n; ++i) {
                int src_idx = len - abs_n + i;
                result(i) = (src_idx >= 0) ? (*mat)(src_idx, 0) : 0.0;
            }
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // For matrices, take rows
    int rows = mat->rows();
    int abs_n = std::abs(n);

    Eigen::MatrixXd result(abs_n, mat->cols());

    if (n >= 0) {
        for (int i = 0; i < abs_n; ++i) {
            if (i < rows) {
                result.row(i) = mat->row(i);
            } else {
                result.row(i).setZero();
            }
        }
    } else {
        for (int i = 0; i < abs_n; ++i) {
            int src_idx = rows - abs_n + i;
            if (src_idx >= 0) {
                result.row(i) = mat->row(src_idx);
            } else {
                result.row(i).setZero();
            }
        }
    }

    m->result = m->heap->allocate_matrix(result);
}

// Drop (↓) - dyadic: drop first n elements
void fn_drop(Machine* m, Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("RANK ERROR: drop count must be scalar"));
        return;
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_scalar()) {
        // Dropping from scalar gives empty vector
        Eigen::VectorXd result(0);
        m->result = m->heap->allocate_vector(result);
        return;
    }

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        int len = mat->rows();
        int abs_n = std::abs(n);

        if (abs_n >= len) {
            // Drop everything
            Eigen::VectorXd result(0);
            m->result = m->heap->allocate_vector(result);
            return;
        }

        int result_len = len - abs_n;
        Eigen::VectorXd result(result_len);

        if (n >= 0) {
            // Drop from beginning
            for (int i = 0; i < result_len; ++i) {
                result(i) = (*mat)(abs_n + i, 0);
            }
        } else {
            // Drop from end
            for (int i = 0; i < result_len; ++i) {
                result(i) = (*mat)(i, 0);
            }
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // For matrices, drop rows
    int rows = mat->rows();
    int abs_n = std::abs(n);

    if (abs_n >= rows) {
        // Drop everything - return empty matrix
        Eigen::MatrixXd result(0, mat->cols());
        m->result = m->heap->allocate_matrix(result);
        return;
    }

    int result_rows = rows - abs_n;
    Eigen::MatrixXd result(result_rows, mat->cols());

    if (n >= 0) {
        result = mat->bottomRows(result_rows);
    } else {
        result = mat->topRows(result_rows);
    }

    m->result = m->heap->allocate_matrix(result);
}

// ============================================================================
// Reverse/Rotate Functions
// ============================================================================

// Reverse (⌽) - monadic: reverse elements along last axis
void fn_reverse(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        // Scalar reversal is identity
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        // Reverse vector elements
        Eigen::VectorXd result(mat->rows());
        for (int i = 0; i < mat->rows(); ++i) {
            result(i) = (*mat)(mat->rows() - 1 - i, 0);
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // For matrices: reverse columns within each row (last axis)
    Eigen::MatrixXd result(mat->rows(), mat->cols());
    for (int i = 0; i < mat->rows(); ++i) {
        for (int j = 0; j < mat->cols(); ++j) {
            result(i, j) = (*mat)(i, mat->cols() - 1 - j);
        }
    }
    m->result = m->heap->allocate_matrix(result);
}

// Reverse First (⊖) - monadic: reverse elements along first axis
void fn_reverse_first(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        // Scalar reversal is identity
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        // For vectors, first axis is the only axis, so same as reverse
        Eigen::VectorXd result(mat->rows());
        for (int i = 0; i < mat->rows(); ++i) {
            result(i) = (*mat)(mat->rows() - 1 - i, 0);
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // For matrices: reverse rows (first axis)
    Eigen::MatrixXd result(mat->rows(), mat->cols());
    for (int i = 0; i < mat->rows(); ++i) {
        result.row(i) = mat->row(mat->rows() - 1 - i);
    }
    m->result = m->heap->allocate_matrix(result);
}

// Tally (≢) - monadic: count along first axis
void fn_tally(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        // Scalar has no first axis, tally is 1
        m->result = m->heap->allocate_scalar(1.0);
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // First axis is number of rows
    m->result = m->heap->allocate_scalar(static_cast<double>(mat->rows()));
}

// Rotate (⌽) - dyadic: rotate elements along last axis
void fn_rotate(Machine* m, Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("RANK ERROR: rotate count must be scalar"));
        return;
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_scalar()) {
        // Rotating a scalar is identity
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        int len = mat->rows();
        if (len == 0) {
            m->result = m->heap->allocate_vector(mat->col(0));
            return;
        }
        // Normalize rotation (positive = left rotate, APL convention)
        n = ((n % len) + len) % len;
        Eigen::VectorXd result(len);
        for (int i = 0; i < len; ++i) {
            result(i) = (*mat)((i + n) % len, 0);
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // For matrices: rotate columns within each row
    int cols = mat->cols();
    if (cols == 0) {
        m->result = m->heap->allocate_matrix(*mat);
        return;
    }
    n = ((n % cols) + cols) % cols;
    Eigen::MatrixXd result(mat->rows(), cols);
    for (int i = 0; i < mat->rows(); ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = (*mat)(i, (j + n) % cols);
        }
    }
    m->result = m->heap->allocate_matrix(result);
}

// Rotate First (⊖) - dyadic: rotate elements along first axis
void fn_rotate_first(Machine* m, Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("RANK ERROR: rotate count must be scalar"));
        return;
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_scalar()) {
        // Rotating a scalar is identity
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        // For vectors, first axis is the only axis
        int len = mat->rows();
        if (len == 0) {
            m->result = m->heap->allocate_vector(mat->col(0));
            return;
        }
        n = ((n % len) + len) % len;
        Eigen::VectorXd result(len);
        for (int i = 0; i < len; ++i) {
            result(i) = (*mat)((i + n) % len, 0);
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // For matrices: rotate rows (first axis)
    int rows = mat->rows();
    if (rows == 0) {
        m->result = m->heap->allocate_matrix(*mat);
        return;
    }
    n = ((n % rows) + rows) % rows;
    Eigen::MatrixXd result(rows, mat->cols());
    for (int i = 0; i < rows; ++i) {
        result.row(i) = mat->row((i + n) % rows);
    }
    m->result = m->heap->allocate_matrix(result);
}

// ============================================================================
// Search Functions (⍳ dyadic, ∊)
// ============================================================================

// Helper: flatten a Value to a row-major VectorXd
static Eigen::VectorXd flatten_value(Value* val) {
    if (val->is_scalar()) {
        Eigen::VectorXd v(1);
        v(0) = val->as_scalar();
        return v;
    }
    const Eigen::MatrixXd* mat = val->as_matrix();
    int size = mat->size();
    int cols = mat->cols();
    Eigen::VectorXd result(size);
    for (int i = 0; i < size; ++i) {
        result(i) = (*mat)(i / cols, i % cols);
    }
    return result;
}

// Index Of (⍳) - dyadic: find indices of rhs elements in lhs
// Returns index of first occurrence of each element, or ≢lhs if not found (0-origin)
void fn_index_of(Machine* m, Value* lhs, Value* rhs) {
    // Get lhs as a flat array of values to search in
    Eigen::VectorXd haystack = flatten_value(lhs);
    double not_found = static_cast<double>(haystack.size());

    // Search for needle in haystack, return index or not_found
    auto find_index = [&haystack, not_found](double needle) -> double {
        for (int i = 0; i < haystack.size(); ++i) {
            if (haystack(i) == needle) {
                return static_cast<double>(i);
            }
        }
        return not_found;
    };

    if (rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(find_index(rhs->as_scalar()));
        return;
    }

    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (rhs->is_vector()) {
        Eigen::VectorXd result(rmat->rows());
        for (int i = 0; i < rmat->rows(); ++i) {
            result(i) = find_index((*rmat)(i, 0));
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    Eigen::MatrixXd result(rmat->rows(), rmat->cols());
    for (int i = 0; i < rmat->size(); ++i) {
        result(i / rmat->cols(), i % rmat->cols()) = find_index((*rmat)(i / rmat->cols(), i % rmat->cols()));
    }
    m->result = m->heap->allocate_matrix(result);
}

// Enlist (∊) - monadic: flatten nested structure to simple vector
// For simple arrays, this is equivalent to ravel (,)
void fn_enlist(Machine* m, Value* omega) {
    // For simple numeric arrays (our current implementation), enlist = ravel
    fn_ravel(m, omega);
}

// Member Of (∊) - dyadic: check if elements of lhs are in rhs
// Returns boolean array with 1 where element is found, 0 otherwise
void fn_member_of(Machine* m, Value* lhs, Value* rhs) {
    // Get rhs as flat array to search in
    Eigen::VectorXd set = flatten_value(rhs);

    // Check if val is in set
    auto is_member = [&set](double val) -> double {
        for (int i = 0; i < set.size(); ++i) {
            if (set(i) == val) {
                return 1.0;
            }
        }
        return 0.0;
    };

    if (lhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(is_member(lhs->as_scalar()));
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();

    if (lhs->is_vector()) {
        Eigen::VectorXd result(lmat->rows());
        for (int i = 0; i < lmat->rows(); ++i) {
            result(i) = is_member((*lmat)(i, 0));
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i / lmat->cols(), i % lmat->cols()) = is_member((*lmat)(i / lmat->cols(), i % lmat->cols()));
    }
    m->result = m->heap->allocate_matrix(result);
}

} // namespace apl
