// Primitives implementation

#include "primitives.h"
#include "value.h"
#include "machine.h"
#include "continuation.h"
#include "parser.h"
#include <cmath>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <map>

namespace apl {

// ============================================================================
// Axis Validation Helper (ISO 13751)
// ============================================================================

// Reject axis specification for functions that don't support it
#define REJECT_AXIS(m, axis) \
    if ((axis) != nullptr) { \
        (m)->throw_error("AXIS ERROR: function does not support axis"); \
        return; \
    }

// ============================================================================
// Tolerant Comparison Helpers (ISO 13751)
// ============================================================================

// Tolerant equality: |A - B| <= CT * max(|A|, |B|)
inline bool tolerant_eq(double a, double b, double ct) {
    if (ct == 0.0) return a == b;
    double diff = std::abs(a - b);
    double magnitude = std::max(std::abs(a), std::abs(b));
    return diff <= ct * magnitude;
}

// Tolerant floor: largest integer n where n ≤ X (with tolerance)
inline double tolerant_floor(double x, double ct) {
    double f = std::floor(x);
    // If x is tolerantly equal to ceiling, use ceiling
    if (tolerant_eq(x, f + 1.0, ct)) {
        return f + 1.0;
    }
    return f;
}

// Tolerant ceiling: smallest integer n where n ≥ X (with tolerance)
inline double tolerant_ceiling(double x, double ct) {
    double c = std::ceil(x);
    // If x is tolerantly equal to floor, use floor
    if (tolerant_eq(x, c - 1.0, ct)) {
        return c - 1.0;
    }
    return c;
}

// Check if value is a negative integer (used by factorial and binomial)
inline bool is_negative_int(double x) {
    return x < 0 && x == std::floor(x);
}

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
PrimitiveFn prim_not       = { "~", fn_not, fn_without };
PrimitiveFn prim_nand      = { "⍲", nullptr, fn_nand };
PrimitiveFn prim_nor       = { "⍱", nullptr, fn_nor };
PrimitiveFn prim_stile     = { "|", fn_magnitude, fn_residue };
PrimitiveFn prim_log       = { "⍟", fn_natural_log, fn_logarithm };
PrimitiveFn prim_factorial = { "!", fn_factorial, fn_binomial };

// Array operation primitives
PrimitiveFn prim_rho       = { "⍴", fn_shape, fn_reshape };
PrimitiveFn prim_comma     = { ",", fn_ravel, fn_catenate };
PrimitiveFn prim_transpose = { "⍉", fn_transpose, fn_dyadic_transpose };
PrimitiveFn prim_domino    = { "⌹", fn_matrix_inverse, fn_matrix_divide };
PrimitiveFn prim_iota      = { "⍳", fn_iota, fn_index_of };
PrimitiveFn prim_uptack    = { "↑", fn_first, fn_take };
PrimitiveFn prim_downtack  = { "↓", nullptr, fn_drop };
PrimitiveFn prim_reverse   = { "⌽", fn_reverse, fn_rotate };
PrimitiveFn prim_reverse_first = { "⊖", fn_reverse_first, fn_rotate_first };
PrimitiveFn prim_tally     = { "≢", fn_tally, nullptr };
PrimitiveFn prim_depth     = { "≡", fn_depth, fn_match };  // ISO 13751: depth/match
PrimitiveFn prim_member    = { "∊", fn_enlist, fn_member_of };
PrimitiveFn prim_grade_up  = { "⍋", fn_grade_up, fn_grade_up_dyadic };
PrimitiveFn prim_grade_down = { "⍒", fn_grade_down, fn_grade_down_dyadic };
PrimitiveFn prim_union     = { "∪", fn_unique, fn_union };
PrimitiveFn prim_circle    = { "○", fn_pi_times, fn_circular };
PrimitiveFn prim_question  = { "?", fn_roll, fn_deal };
PrimitiveFn prim_decode    = { "⊥", nullptr, fn_decode };
PrimitiveFn prim_encode    = { "⊤", nullptr, fn_encode };
PrimitiveFn prim_execute   = { "⍎", fn_execute, nullptr };
PrimitiveFn prim_format    = { "⍕", fn_format_monadic, fn_format_dyadic };
PrimitiveFn prim_table     = { "⍪", fn_table, fn_catenate_first };
PrimitiveFn prim_squad     = { "⌷", nullptr, fn_squad };
PrimitiveFn prim_left      = { "⊣", fn_right, fn_left };   // ISO 10.2.17: monadic ⊣ is identity, dyadic returns left
PrimitiveFn prim_right     = { "⊢", fn_right, fn_right_dyadic }; // ISO 10.2.18: both return right argument

// ============================================================================
// Dyadic Arithmetic Functions
// ============================================================================

// Addition (+)
void fn_add(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // Fast path: scalar + scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar + rhs->data.scalar);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

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
        m->throw_error("LENGTH ERROR: mismatched shapes in addition");
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
void fn_subtract(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // Fast path: scalar - scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar - rhs->data.scalar);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

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
        m->throw_error("LENGTH ERROR: mismatched shapes in subtraction");
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
void fn_multiply(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // Fast path: scalar × scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(lhs->data.scalar * rhs->data.scalar);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

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
        m->throw_error("LENGTH ERROR: mismatched shapes in multiplication");
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

// Helper for ISO 13751 division: 0÷0=1, else A÷0 is domain error
static bool safe_divide(double a, double b, double& result) {
    if (b == 0.0) {
        if (a == 0.0) {
            result = 1.0;  // ISO 13751 7.2.4: 0÷0 returns 1
            return true;
        }
        return false;  // domain error
    }
    result = a / b;
    return true;
}

// Division (÷)
// ISO 13751 7.2.4: If B is zero and A is zero, return one; else if B is zero, domain-error
void fn_divide(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // Fast path: scalar ÷ scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        double result;
        if (!safe_divide(lhs->data.scalar, rhs->data.scalar, result)) {
            m->throw_error("DOMAIN ERROR: division by zero");
            return;
        }
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Scalar extension: scalar ÷ array
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        double lval = lhs->data.scalar;
        for (int i = 0; i < rmat->size(); ++i) {
            if (!safe_divide(lval, rmat->data()[i], result(i))) {
                m->throw_error("DOMAIN ERROR: division by zero");
                return;
            }
        }
        // Preserve vector/matrix distinction
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // Scalar extension: array ÷ scalar
    if (rhs->is_scalar()) {
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        double rval = rhs->data.scalar;
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            if (!safe_divide(lmat->data()[i], rval, result(i))) {
                m->throw_error("DOMAIN ERROR: division by zero");
                return;
            }
        }
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
        m->throw_error("LENGTH ERROR: mismatched shapes in division");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        if (!safe_divide(lmat->data()[i], rmat->data()[i], result(i))) {
            m->throw_error("DOMAIN ERROR: division by zero");
            return;
        }
    }
    // Preserve vector/matrix distinction
    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Power (*)
// ISO 13751 7.2.7: Helper for power edge cases
static double power_scalar(Machine* m, double base, double exp) {
    // ISO 13751 7.2.7: If A is zero and B is zero, return one
    if (base == 0.0 && exp == 0.0) {
        return 1.0;
    }
    // ISO 13751 7.2.7: If A is zero and real-part of B is positive, return zero
    if (base == 0.0 && exp > 0.0) {
        return 0.0;
    }
    // ISO 13751 7.2.7: If A is zero and real-part of B is negative, signal domain-error
    if (base == 0.0 && exp < 0.0) {
        m->throw_error("DOMAIN ERROR: 0 raised to negative power");
    }
    return std::pow(base, exp);
}

void fn_power(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // Fast path: scalar * scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(power_scalar(m, lhs->data.scalar, rhs->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Scalar extension
    if (lhs->is_scalar()) {
        // lhs is scalar base, rhs is array of exponents: lhs^rhs[i]
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = power_scalar(m, lhs->data.scalar, rmat->data()[i]);
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
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = power_scalar(m, lmat->data()[i], rhs->data.scalar);
        }
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
        m->throw_error("LENGTH ERROR: mismatched shapes in power");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = power_scalar(m, lmat->data()[i], rmat->data()[i]);
    }
    // Preserve vector/matrix distinction
    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Equality (=)
void fn_equal(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    double ct = m->ct;

    // Fast path: scalar = scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(tolerant_eq(lhs->data.scalar, rhs->data.scalar, ct) ? 1.0 : 0.0);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = tolerant_eq(lhs->data.scalar, rmat->data()[i], ct) ? 1.0 : 0.0;
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
            result(i) = tolerant_eq(lmat->data()[i], rhs->data.scalar, ct) ? 1.0 : 0.0;
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
        m->throw_error("LENGTH ERROR: mismatched shapes in equality");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = tolerant_eq(lmat->data()[i], rmat->data()[i], ct) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Not Equal (≠)
void fn_not_equal(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    double ct = m->ct;

    // Fast path: scalar ≠ scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(!tolerant_eq(lhs->data.scalar, rhs->data.scalar, ct) ? 1.0 : 0.0);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = !tolerant_eq(lhs->data.scalar, rmat->data()[i], ct) ? 1.0 : 0.0;
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
            result(i) = !tolerant_eq(lmat->data()[i], rhs->data.scalar, ct) ? 1.0 : 0.0;
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
        m->throw_error("LENGTH ERROR: mismatched shapes in not-equal");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = !tolerant_eq(lmat->data()[i], rmat->data()[i], ct) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Less Than (<) - strictly less and not tolerantly equal
void fn_less(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    double ct = m->ct;

    // Fast path: scalar < scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        double a = lhs->data.scalar, b = rhs->data.scalar;
        m->result = m->heap->allocate_scalar((a < b && !tolerant_eq(a, b, ct)) ? 1.0 : 0.0);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        double a = lhs->data.scalar;
        for (int i = 0; i < rmat->size(); ++i) {
            double b = rmat->data()[i];
            result(i) = (a < b && !tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
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
        double b = rhs->data.scalar;
        for (int i = 0; i < lmat->size(); ++i) {
            double a = lmat->data()[i];
            result(i) = (a < b && !tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
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
        m->throw_error("LENGTH ERROR: mismatched shapes in less-than");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        double a = lmat->data()[i], b = rmat->data()[i];
        result(i) = (a < b && !tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Greater Than (>) - strictly greater and not tolerantly equal
void fn_greater(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    double ct = m->ct;

    // Fast path: scalar > scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        double a = lhs->data.scalar, b = rhs->data.scalar;
        m->result = m->heap->allocate_scalar((a > b && !tolerant_eq(a, b, ct)) ? 1.0 : 0.0);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        double a = lhs->data.scalar;
        for (int i = 0; i < rmat->size(); ++i) {
            double b = rmat->data()[i];
            result(i) = (a > b && !tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
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
        double b = rhs->data.scalar;
        for (int i = 0; i < lmat->size(); ++i) {
            double a = lmat->data()[i];
            result(i) = (a > b && !tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
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
        m->throw_error("LENGTH ERROR: mismatched shapes in greater-than");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        double a = lmat->data()[i], b = rmat->data()[i];
        result(i) = (a > b && !tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Less Than or Equal (≤) - less than or tolerantly equal
void fn_less_eq(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    double ct = m->ct;

    // Fast path: scalar ≤ scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        double a = lhs->data.scalar, b = rhs->data.scalar;
        m->result = m->heap->allocate_scalar((a <= b || tolerant_eq(a, b, ct)) ? 1.0 : 0.0);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        double a = lhs->data.scalar;
        for (int i = 0; i < rmat->size(); ++i) {
            double b = rmat->data()[i];
            result(i) = (a <= b || tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
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
        double b = rhs->data.scalar;
        for (int i = 0; i < lmat->size(); ++i) {
            double a = lmat->data()[i];
            result(i) = (a <= b || tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
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
        m->throw_error("LENGTH ERROR: mismatched shapes in less-or-equal");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        double a = lmat->data()[i], b = rmat->data()[i];
        result(i) = (a <= b || tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Greater Than or Equal (≥) - greater than or tolerantly equal
void fn_greater_eq(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    double ct = m->ct;

    // Fast path: scalar ≥ scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        double a = lhs->data.scalar, b = rhs->data.scalar;
        m->result = m->heap->allocate_scalar((a >= b || tolerant_eq(a, b, ct)) ? 1.0 : 0.0);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Scalar extension
    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        double a = lhs->data.scalar;
        for (int i = 0; i < rmat->size(); ++i) {
            double b = rmat->data()[i];
            result(i) = (a >= b || tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
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
        double b = rhs->data.scalar;
        for (int i = 0; i < lmat->size(); ++i) {
            double a = lmat->data()[i];
            result(i) = (a >= b || tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
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
        m->throw_error("LENGTH ERROR: mismatched shapes in greater-or-equal");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        double a = lmat->data()[i], b = rmat->data()[i];
        result(i) = (a >= b || tolerant_eq(a, b, ct)) ? 1.0 : 0.0;
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
void fn_maximum(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::max(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

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
        m->throw_error("LENGTH ERROR: mismatched shapes in maximum");
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
void fn_minimum(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::min(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

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
        m->throw_error("LENGTH ERROR: mismatched shapes in minimum");
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
void fn_ceiling(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    double ct = m->ct;

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(tolerant_ceiling(omega->data.scalar, ct));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Fast path: no tolerance, use Eigen vectorized ceil
    if (ct == 0.0) {
        Eigen::MatrixXd result = mat->array().ceil();
        if (omega->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    Eigen::MatrixXd result(mat->rows(), mat->cols());
    for (int i = 0; i < mat->size(); ++i) {
        result(i) = tolerant_ceiling(mat->data()[i], ct);
    }

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Floor (⌊) - monadic
void fn_floor(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    double ct = m->ct;

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(tolerant_floor(omega->data.scalar, ct));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Fast path: no tolerance, use Eigen vectorized floor
    if (ct == 0.0) {
        Eigen::MatrixXd result = mat->array().floor();
        if (omega->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    Eigen::MatrixXd result(mat->rows(), mat->cols());
    for (int i = 0; i < mat->size(); ++i) {
        result(i) = tolerant_floor(mat->data()[i], ct);
    }

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// ============================================================================
// Logical Functions (∧ ∨ ~ ⍲ ⍱)
// ============================================================================

// Helper: check if value is near-boolean (tolerantly close to 0 or 1)
static bool is_near_boolean(double d) {
    const double tol = 1E-10;
    return (std::abs(d) < tol) || (std::abs(d - 1.0) < tol);
}

// GCD helper for real numbers using Euclidean algorithm
// ISO 13751 uses an implementation-algorithm for GCD
static double gcd_real(double a, double b) {
    a = std::abs(a);
    b = std::abs(b);
    if (a == 0.0) return b;
    if (b == 0.0) return a;
    // Euclidean algorithm with tolerance for floating point
    const double tol = 1E-10;
    while (b > tol) {
        double t = std::fmod(a, b);
        a = b;
        b = t;
    }
    return a;
}

// LCM helper: LCM(a,b) = |a*b| / GCD(a,b)
static double lcm_real(double a, double b) {
    if (a == 0.0 || b == 0.0) return 0.0;
    double g = gcd_real(a, b);
    return std::abs(a * b) / g;
}

// And/LCM (∧) helper - returns result for scalar pair
// ISO 13751 7.2.12: For near-boolean, AND; otherwise LCM
static double and_lcm(double a, double b) {
    if (is_near_boolean(a) && is_near_boolean(b)) {
        // Boolean AND
        int a1 = (std::abs(a - 1.0) < std::abs(a)) ? 1 : 0;
        int b1 = (std::abs(b - 1.0) < std::abs(b)) ? 1 : 0;
        return (a1 && b1) ? 1.0 : 0.0;
    }
    // LCM for non-boolean
    return lcm_real(a, b);
}

// And/LCM (∧) - dyadic
// ISO 13751 7.2.12: For near-boolean it's AND, otherwise LCM
void fn_and(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(and_lcm(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        double lval = lhs->data.scalar;
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = and_lcm(lval, rmat->data()[i]);
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
        double rval = rhs->data.scalar;
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = and_lcm(lmat->data()[i], rval);
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
        m->throw_error("LENGTH ERROR: mismatched shapes in and");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = and_lcm(lmat->data()[i], rmat->data()[i]);
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Or/GCD (∨) helper - returns result for scalar pair
// ISO 13751 7.2.13: For near-boolean, OR; otherwise GCD
static double or_gcd(double a, double b) {
    if (is_near_boolean(a) && is_near_boolean(b)) {
        // Boolean OR
        int a1 = (std::abs(a - 1.0) < std::abs(a)) ? 1 : 0;
        int b1 = (std::abs(b - 1.0) < std::abs(b)) ? 1 : 0;
        return (a1 || b1) ? 1.0 : 0.0;
    }
    // GCD for non-boolean
    return gcd_real(a, b);
}

// Or/GCD (∨) - dyadic
// ISO 13751 7.2.13: For near-boolean it's OR, otherwise GCD
void fn_or(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(or_gcd(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        double lval = lhs->data.scalar;
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = or_gcd(lval, rmat->data()[i]);
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
        double rval = rhs->data.scalar;
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = or_gcd(lmat->data()[i], rval);
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
        m->throw_error("LENGTH ERROR: mismatched shapes in or");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = or_gcd(lmat->data()[i], rmat->data()[i]);
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Not (~) - monadic
// ISO 13751 7.1.12: If B is not near-Boolean, signal domain-error
void fn_not(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        double d = omega->data.scalar;
        if (!is_near_boolean(d)) {
            m->throw_error("DOMAIN ERROR: ~ requires boolean argument");
            return;
        }
        // Round to nearest integer (0 or 1) then complement
        int b = (std::abs(d - 1.0) < std::abs(d)) ? 1 : 0;
        m->result = m->heap->allocate_scalar(b == 1 ? 0.0 : 1.0);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Check all elements are near-boolean
    for (int i = 0; i < mat->size(); ++i) {
        if (!is_near_boolean(mat->data()[i])) {
            m->throw_error("DOMAIN ERROR: ~ requires boolean argument");
            return;
        }
    }

    Eigen::MatrixXd result(mat->rows(), mat->cols());
    for (int i = 0; i < mat->size(); ++i) {
        double d = mat->data()[i];
        int b = (std::abs(d - 1.0) < std::abs(d)) ? 1 : 0;
        result(i) = (b == 1) ? 0.0 : 1.0;
    }

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Nand (⍲) helper - validates boolean domain and computes ~(A∧B)
// Returns -1 on domain error, otherwise result
static double nand_bool(double a, double b, bool& error) {
    error = false;
    if (!is_near_boolean(a) || !is_near_boolean(b)) {
        error = true;
        return 0.0;
    }
    int a1 = (std::abs(a - 1.0) < std::abs(a)) ? 1 : 0;
    int b1 = (std::abs(b - 1.0) < std::abs(b)) ? 1 : 0;
    return (a1 && b1) ? 0.0 : 1.0;
}

// Nand (⍲) - dyadic
// ISO 13751 7.2.14: If either A or B is not near-Boolean, signal domain-error
void fn_nand(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    bool error = false;

    if (lhs->is_scalar() && rhs->is_scalar()) {
        double result = nand_bool(lhs->data.scalar, rhs->data.scalar, error);
        if (error) {
            m->throw_error("DOMAIN ERROR: ⍲ requires boolean arguments");
            return;
        }
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = nand_bool(lhs->data.scalar, rmat->data()[i], error);
            if (error) {
                m->throw_error("DOMAIN ERROR: ⍲ requires boolean arguments");
                return;
            }
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
            result(i) = nand_bool(lmat->data()[i], rhs->data.scalar, error);
            if (error) {
                m->throw_error("DOMAIN ERROR: ⍲ requires boolean arguments");
                return;
            }
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
        m->throw_error("LENGTH ERROR: mismatched shapes in nand");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = nand_bool(lmat->data()[i], rmat->data()[i], error);
        if (error) {
            m->throw_error("DOMAIN ERROR: ⍲ requires boolean arguments");
            return;
        }
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Nor (⍱) helper - validates boolean domain and computes ~(A∨B)
static double nor_bool(double a, double b, bool& error) {
    error = false;
    if (!is_near_boolean(a) || !is_near_boolean(b)) {
        error = true;
        return 0.0;
    }
    int a1 = (std::abs(a - 1.0) < std::abs(a)) ? 1 : 0;
    int b1 = (std::abs(b - 1.0) < std::abs(b)) ? 1 : 0;
    return (a1 || b1) ? 0.0 : 1.0;
}

// Nor (⍱) - dyadic
// ISO 13751 7.2.15: If either A or B is not near-Boolean, signal domain-error
void fn_nor(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    bool error = false;

    if (lhs->is_scalar() && rhs->is_scalar()) {
        double result = nor_bool(lhs->data.scalar, rhs->data.scalar, error);
        if (error) {
            m->throw_error("DOMAIN ERROR: ⍱ requires boolean arguments");
            return;
        }
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = nor_bool(lhs->data.scalar, rmat->data()[i], error);
            if (error) {
                m->throw_error("DOMAIN ERROR: ⍱ requires boolean arguments");
                return;
            }
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
            result(i) = nor_bool(lmat->data()[i], rhs->data.scalar, error);
            if (error) {
                m->throw_error("DOMAIN ERROR: ⍱ requires boolean arguments");
                return;
            }
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
        m->throw_error("LENGTH ERROR: mismatched shapes in nor");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = nor_bool(lmat->data()[i], rmat->data()[i], error);
        if (error) {
            m->throw_error("DOMAIN ERROR: ⍱ requires boolean arguments");
            return;
        }
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
void fn_magnitude(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::abs(omega->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

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
// ISO 13751 7.2.9: Uses comparison-tolerance
void fn_residue(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    double ct = m->ct;

    auto residue = [ct](double a, double b) -> double {
        // ISO 13751: If A is zero, return B
        if (a == 0.0) return b;

        // ISO 13751: If B/A is integral within tolerance, return 0
        if (ct > 0.0) {
            double quotient = b / a;
            double nearest_int = std::round(quotient);
            if (tolerant_eq(quotient, nearest_int, ct)) {
                return 0.0;
            }
        }

        double r = std::fmod(b, a);
        // Adjust sign to match divisor (APL semantics)
        if (r != 0.0 && ((a > 0.0) != (r > 0.0))) {
            r += a;
        }
        // ISO 13751: If Z is A, return zero
        if (r == a) return 0.0;
        return r;
    };

    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(residue(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

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
        m->throw_error("LENGTH ERROR: mismatched shapes in residue");
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
void fn_natural_log(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        double val = omega->data.scalar;
        if (val <= 0.0) {
            m->throw_error("DOMAIN ERROR: logarithm of non-positive number");
            return;
        }
        m->result = m->heap->allocate_scalar(std::log(val));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Check for non-positive values
    if ((mat->array() <= 0.0).any()) {
        m->throw_error("DOMAIN ERROR: logarithm of non-positive number");
        return;
    }
    Eigen::MatrixXd result = mat->array().log();

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Logarithm (⍟) - dyadic (lhs ⍟ rhs = log base lhs of rhs)
void fn_logarithm(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // log_a(b) = ln(b) / ln(a)
    // Domain errors: base <= 0, base == 1, value <= 0
    auto check_domain = [m](double base, double val) -> bool {
        if (base <= 0.0 || base == 1.0) {
            m->throw_error("DOMAIN ERROR: invalid logarithm base");
            return false;
        }
        if (val <= 0.0) {
            m->throw_error("DOMAIN ERROR: logarithm of non-positive number");
            return false;
        }
        return true;
    };

    auto log_base = [](double base, double val) -> double {
        return std::log(val) / std::log(base);
    };

    if (lhs->is_scalar() && rhs->is_scalar()) {
        if (!check_domain(lhs->data.scalar, rhs->data.scalar)) return;
        m->result = m->heap->allocate_scalar(log_base(lhs->data.scalar, rhs->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

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
        m->throw_error("LENGTH ERROR: mismatched shapes in logarithm");
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
// DOMAIN ERROR for negative integers (gamma has poles at non-positive integers)
void fn_factorial(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    // is_negative_int is defined as a global helper at top of file

    if (omega->is_scalar()) {
        double val = omega->data.scalar;
        if (is_negative_int(val)) {
            m->throw_error("DOMAIN ERROR: factorial of negative integer");
            return;
        }
        m->result = m->heap->allocate_scalar(std::tgamma(val + 1.0));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Check for negative integers
    for (int i = 0; i < mat->size(); ++i) {
        if (is_negative_int(mat->data()[i])) {
            m->throw_error("DOMAIN ERROR: factorial of negative integer");
            return;
        }
    }

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
// ISO 13751 7.2.10: Handles negative integer cases per the spec table
// A = lhs (k), B = rhs (n)
static double binomial_scalar(Machine* m, double A, double B) {
    // Check if values are negative integers
    bool A_neg_int = is_negative_int(A);
    bool B_neg_int = is_negative_int(B);
    double B_minus_A = B - A;
    bool BmA_neg_int = is_negative_int(B_minus_A);

    // ISO 13751 7.2.10 case table:
    // Case A B B-A
    if (!A_neg_int && !B_neg_int && !BmA_neg_int) {
        // Case 0 0 0: Return (!B)÷(!A)×!B-A (standard gamma formula)
        return std::tgamma(B + 1.0) / (std::tgamma(A + 1.0) * std::tgamma(B - A + 1.0));
    }
    if (!A_neg_int && !B_neg_int && BmA_neg_int) {
        // Case 0 0 1: A non-neg, B non-neg, B-A negative integer → return 0
        return 0.0;
    }
    if (!A_neg_int && B_neg_int && !BmA_neg_int) {
        // Case 0 1 0: A non-neg, B neg int, B-A non-neg → signal domain-error
        m->throw_error("DOMAIN ERROR: invalid binomial arguments");
        return 0.0;
    }
    if (!A_neg_int && B_neg_int && BmA_neg_int) {
        // Case 0 1 1: A non-neg, B neg int, B-A neg int
        // Return (¯1*A)×A!(A-B-1)
        // Note: The spec says A!A-B+1 but looking at the table values, it should be:
        // For A non-neg int, B neg int: use formula (¯1^A) × C(A-B-1, A)
        double sign = (static_cast<int>(A) % 2 == 0) ? 1.0 : -1.0;
        double new_n = A - B - 1.0;  // This is now positive
        return sign * std::tgamma(new_n + 1.0) / (std::tgamma(A + 1.0) * std::tgamma(new_n - A + 1.0));
    }
    if (A_neg_int && !B_neg_int && !BmA_neg_int) {
        // Case 1 0 0: A neg int, B non-neg, B-A non-neg → return 0
        return 0.0;
    }
    // Case 1 0 1 cannot arise (if A is neg int and B is non-neg, B-A > B >= 0)
    if (A_neg_int && B_neg_int && !BmA_neg_int) {
        // Case 1 1 0: A neg int, B neg int, B-A non-neg int (B >= A)
        // Return (¯1^(B-A)) × (|B+1|)!(|A+1|)
        int exp = static_cast<int>(B - A);
        double sign = (exp % 2 == 0) ? 1.0 : -1.0;
        double abs_B_plus_1 = std::abs(B + 1.0);
        double abs_A_plus_1 = std::abs(A + 1.0);
        // (|B+1|)!(|A+1|) = C(|A+1|, |B+1|) = (|A+1|)! / ((|B+1|)! × (|A+1|-|B+1|)!)
        return sign * std::tgamma(abs_A_plus_1 + 1.0) /
               (std::tgamma(abs_B_plus_1 + 1.0) * std::tgamma(abs_A_plus_1 - abs_B_plus_1 + 1.0));
    }
    if (A_neg_int && B_neg_int && BmA_neg_int) {
        // Case 1 1 1: A neg int, B neg int, B-A neg int (B < A)
        // ISO 13751: Return zero
        return 0.0;
    }

    // Default: use gamma formula (shouldn't reach here for integers)
    return std::tgamma(B + 1.0) / (std::tgamma(A + 1.0) * std::tgamma(B - A + 1.0));
}

void fn_binomial(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);

    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(binomial_scalar(m, lhs->data.scalar, rhs->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (lhs->is_scalar()) {
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = binomial_scalar(m, lhs->data.scalar, rmat->data()[i]);
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
            result(i) = binomial_scalar(m, lmat->data()[i], rhs->data.scalar);
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
        m->throw_error("LENGTH ERROR: mismatched shapes in binomial");
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = binomial_scalar(m, lmat->data()[i], rmat->data()[i]);
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// ============================================================================
// Circular Functions (○)
// ============================================================================

// Pi Times (○) - monadic: multiply by pi
void fn_pi_times(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(M_PI * omega->data.scalar);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = M_PI * mat->array();

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Helper for circular function dispatch
// Returns NaN to signal domain error (caller must check)
static double circular_function(int fn_code, double x) {
    switch (fn_code) {
        case 0:  // sqrt(1-x²) - requires |x| ≤ 1
            if (x < -1.0 || x > 1.0) return std::nan("");
            return std::sqrt(1.0 - x * x);
        case 1:  return std::sin(x);                      // sin
        case 2:  return std::cos(x);                      // cos
        case 3:  return std::tan(x);                      // tan
        case 4:  return std::sqrt(1.0 + x * x);           // sqrt(1+x²)
        case 5:  return std::sinh(x);                     // sinh
        case 6:  return std::cosh(x);                     // cosh
        case 7:  return std::tanh(x);                     // tanh
        case -1: return std::asin(x);                     // asin
        case -2: return std::acos(x);                     // acos
        case -3: return std::atan(x);                     // atan
        case -4: // ISO 13751: If B is ¯1 return zero, else (B+1)×((B-1)÷(B+1))*0.5
            if (x == -1.0) return 0.0;
            return (x + 1.0) * std::sqrt((x - 1.0) / (x + 1.0));
        case -5: return std::asinh(x);                    // asinh
        case -6: return std::acosh(x);                    // acosh
        case -7: // ISO 13751: If B is ±1, signal domain-error
            if (x == 1.0 || x == -1.0) return std::nan("");
            return std::atanh(x);
        // Cases 8-12 and -8 to -12 require complex arithmetic (not supported for general real inputs)
        case 8:  // sqrt(-1-x²) - always requires complex for real x
            return std::nan("");  // Domain error: requires complex
        case -8: // -sqrt(-1-x²) - always requires complex for real x
            return std::nan("");  // Domain error: requires complex
        case 9:  // real part (identity for real)
            return x;
        case -9: // identity
            return x;
        case 10: // magnitude (abs for real)
            return std::abs(x);
        case -10: // conjugate (identity for real)
            return x;
        case 11: // imaginary part (0 for real)
            return 0.0;
        case -11: // x×i (requires complex)
            return std::nan("");
        case 12: // arc/phase (0 or π for real)
            return (x >= 0) ? 0.0 : M_PI;
        case -12: // exp(x×i) (requires complex)
            return std::nan("");
        default:
            throw std::runtime_error("VM BUG: circular function code not validated before dispatch");
    }
}

// Circular Functions (○) - dyadic: A○B applies function A to B
void fn_circular(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // Left argument must be scalar integer in range -12 to 12
    if (!lhs->is_scalar()) {
        m->throw_error("RANK ERROR: circular function code must be scalar");
        return;
    }

    double fn_val = lhs->data.scalar;
    int fn_code = static_cast<int>(std::round(fn_val));

    if (std::abs(fn_val - fn_code) > 1e-10) {
        m->throw_error("DOMAIN ERROR: circular function code must be integer");
        return;
    }

    // ISO 13751: fn_code must be in [-12, 12]
    if (fn_code < -12 || fn_code > 12) {
        m->throw_error("DOMAIN ERROR: circular function code must be -12 to 12");
        return;
    }

    // Apply to right argument
    if (rhs->is_scalar()) {
        double result = circular_function(fn_code, rhs->data.scalar);
        if (std::isnan(result)) {
            m->throw_error("DOMAIN ERROR: invalid argument for circular function");
            return;
        }
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    // String → char vector conversion for array operations
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = rhs->as_matrix();
    Eigen::MatrixXd result(mat->rows(), mat->cols());

    for (int i = 0; i < mat->size(); ++i) {
        result(i) = circular_function(fn_code, mat->data()[i]);
        if (std::isnan(result(i))) {
            m->throw_error("DOMAIN ERROR: invalid argument for circular function");
            return;
        }
    }

    if (rhs->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// ============================================================================
// Monadic Arithmetic Functions
// ============================================================================

// Conjugate/Identity (+)
void fn_conjugate(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    // For real numbers, identity just returns the value
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->data.scalar);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    // For arrays, return a copy preserving vector/matrix distinction
    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(omega->as_matrix()->col(0), omega->is_char_data());
    } else {
        m->result = m->heap->allocate_matrix(*omega->as_matrix(), omega->is_char_data());
    }
}

// Negation (-)
void fn_negate(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(-omega->data.scalar);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    Eigen::MatrixXd result = -omega->as_matrix()->array();
    // Preserve vector/matrix distinction
    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Signum/Sign (×)
void fn_signum(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        double val = omega->data.scalar;
        double sign = (val > 0.0) ? 1.0 : (val < 0.0) ? -1.0 : 0.0;
        m->result = m->heap->allocate_scalar(sign);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

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
void fn_reciprocal(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        if (omega->data.scalar == 0.0) {
            m->throw_error("DOMAIN ERROR: reciprocal of zero");
            return;
        }
        m->result = m->heap->allocate_scalar(1.0 / omega->data.scalar);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Check for zeros
    if ((mat->array() == 0.0).any()) {
        m->throw_error("DOMAIN ERROR: reciprocal of zero");
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
void fn_exponential(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(std::exp(omega->data.scalar));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

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
void fn_shape(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // Scalar has empty shape
        Eigen::VectorXd shape(0);
        m->result = m->heap->allocate_vector(shape);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

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
void fn_reshape(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // lhs must be a scalar or vector specifying new shape
    if (!lhs->is_scalar() && !lhs->is_vector()) {
        m->throw_error("RANK ERROR: left argument to reshape must be scalar or vector");
        return;
    }

    // Get target shape
    int target_rows, target_cols;

    if (lhs->is_scalar()) {
        // Scalar shape means 1D vector of that length
        double dim = lhs->as_scalar();
        // Validate: must be non-negative integer
        if (dim < 0.0) {
            m->throw_error("DOMAIN ERROR: reshape dimension must be non-negative");
            return;
        }
        if (dim != std::floor(dim)) {
            m->throw_error("DOMAIN ERROR: reshape dimension must be an integer");
            return;
        }
        target_rows = static_cast<int>(dim);
        target_cols = 1;
    } else {
        const Eigen::MatrixXd* shape_mat = lhs->as_matrix();
        if (shape_mat->rows() == 0) {
            // ISO 13751 Section 8.3.1: Empty shape produces scalar
            // (⍳0)⍴5 → 5 (scalar)
            // Get first element of source
            double scalar_val;
            if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);
            if (rhs->is_scalar()) {
                scalar_val = rhs->as_scalar();
            } else if (rhs->size() > 0) {
                const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();
                scalar_val = (*rhs_mat)(0, 0);
            } else {
                m->throw_error("DOMAIN ERROR: cannot reshape empty array to scalar");
                return;
            }
            m->result = m->heap->allocate_scalar(scalar_val);
            return;
        } else if (shape_mat->rows() == 1) {
            // Single element: vector of that length
            double dim = (*shape_mat)(0, 0);
            // Validate: must be non-negative integer
            if (dim < 0.0) {
                m->throw_error("DOMAIN ERROR: reshape dimension must be non-negative");
                return;
            }
            if (dim != std::floor(dim)) {
                m->throw_error("DOMAIN ERROR: reshape dimension must be an integer");
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
                m->throw_error("DOMAIN ERROR: reshape dimensions must be non-negative");
                return;
            }
            if (dim1 != std::floor(dim1) || dim2 != std::floor(dim2)) {
                m->throw_error("DOMAIN ERROR: reshape dimensions must be integers");
                return;
            }
            target_rows = static_cast<int>(dim1);
            target_cols = static_cast<int>(dim2);
        } else {
            m->throw_error("RANK ERROR: reshape shape must have 1 or 2 elements");
            return;
        }
    }

    int target_size = target_rows * target_cols;

    // Check implementation limit (ISO 13751 §A.3)
    if (static_cast<size_t>(target_size) > MAX_ARRAY_SIZE) {
        m->throw_error("LIMIT ERROR: array size exceeds implementation limit");
        return;
    }

    // String → char vector conversion for array operations
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

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
        m->throw_error("DOMAIN ERROR: cannot reshape empty array to non-empty shape");
        return;
    }

    // Build result by cycling through source data (row-major order per APL)
    Eigen::MatrixXd result(target_rows, target_cols);
    for (int i = 0; i < target_size; ++i) {
        result(i / target_cols, i % target_cols) = source(i % source.size());
    }

    // Preserve character data flag from source
    bool is_char = rhs->is_char_data();

    if (target_cols == 1) {
        m->result = m->heap->allocate_vector(result.col(0), is_char);
    } else {
        m->result = m->heap->allocate_matrix(result, is_char);
    }
}

// Ravel (,) - monadic: flatten to vector
void fn_ravel(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        Eigen::VectorXd v(1);
        v(0) = omega->as_scalar();
        m->result = m->heap->allocate_vector(v);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    bool is_char = omega->is_char_data();
    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Flatten in row-major order (APL standard)
    int size = mat->size();
    Eigen::VectorXd result(size);
    int rows = mat->rows();
    int cols = mat->cols();
    for (int i = 0; i < size; ++i) {
        result(i) = (*mat)(i / cols, i % cols);
    }
    m->result = m->heap->allocate_vector(result, is_char);
}

// Catenate (,) - dyadic: concatenate arrays (ISO 13751 Section 10.2.6)
// Near-integer K: catenate along existing axis K
// Non-integer K: laminate (create new axis at ⌊K)
void fn_catenate(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Preserve char data if both operands are char data
    bool is_char = lhs->is_char_data() && rhs->is_char_data();

    // Handle scalar cases by promoting to 1-element vector
    if (lhs->is_scalar() && rhs->is_scalar()) {
        if (axis != nullptr) {
            m->throw_error("AXIS ERROR: cannot catenate scalars with axis");
            return;
        }
        Eigen::VectorXd result(2);
        result(0) = lhs->as_scalar();
        result(1) = rhs->as_scalar();
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // Convert both to matrices for uniform handling
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    bool both_vectors = (lhs->is_scalar() || lhs->is_vector()) &&
                        (rhs->is_scalar() || rhs->is_vector());

    // Determine axis and whether this is catenate or laminate
    double k_val = 0;
    bool is_laminate = false;

    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("RANK ERROR: axis must be scalar");
            return;
        }
        k_val = axis->as_scalar();
        // Check if K is a near-integer (catenate) or not (laminate)
        double rounded = std::round(k_val);
        is_laminate = std::abs(k_val - rounded) > 1e-10;
    }

    if (is_laminate) {
        // Laminate: create new axis at ⌊K
        int new_axis = static_cast<int>(std::floor(k_val));

        // Both must have same shape for laminate
        if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
            m->throw_error("LENGTH ERROR: incompatible shapes for laminate");
            return;
        }

        if (both_vectors) {
            int len = lmat->rows();
            if (new_axis == 0) {
                // New axis at position 0: 2xN matrix (vectors become rows)
                Eigen::MatrixXd result(2, len);
                for (int i = 0; i < len; ++i) {
                    result(0, i) = (*lmat)(i, 0);
                    result(1, i) = (*rmat)(i, 0);
                }
                m->result = m->heap->allocate_matrix(result, is_char);
            } else {
                // New axis at position 1: Nx2 matrix (vectors become columns)
                Eigen::MatrixXd result(len, 2);
                for (int i = 0; i < len; ++i) {
                    result(i, 0) = (*lmat)(i, 0);
                    result(i, 1) = (*rmat)(i, 0);
                }
                m->result = m->heap->allocate_matrix(result, is_char);
            }
        } else {
            // Laminate matrices - not yet supported
            m->throw_error("RANK ERROR: laminate of matrices not supported");
        }
        return;
    }

    // Catenate along existing axis
    if (both_vectors) {
        // Vectors only have axis 1
        if (axis != nullptr) {
            int cat_axis = static_cast<int>(std::round(k_val));
            if (cat_axis != 1) {
                m->throw_error("AXIS ERROR: vectors only have axis 1");
                return;
            }
        }
        // Just concatenate
        Eigen::VectorXd result(lmat->rows() + rmat->rows());
        for (int i = 0; i < lmat->rows(); ++i) {
            result(i) = (*lmat)(i, 0);
        }
        for (int i = 0; i < rmat->rows(); ++i) {
            result(lmat->rows() + i) = (*rmat)(i, 0);
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // Matrix catenation
    int cat_axis = 2;  // Default: last axis (columns for matrices)
    if (axis != nullptr) {
        cat_axis = static_cast<int>(std::round(k_val));
        if (cat_axis < 1 || cat_axis > 2) {
            m->throw_error("AXIS ERROR: axis must be 1 or 2 for matrix");
            return;
        }
    }

    if (cat_axis == 1) {
        // Catenate along first axis (rows)
        if (lmat->cols() != rmat->cols()) {
            m->throw_error("LENGTH ERROR: incompatible shapes for catenation");
            return;
        }

        Eigen::MatrixXd result(lmat->rows() + rmat->rows(), lmat->cols());
        result << *lmat, *rmat;
        m->result = m->heap->allocate_matrix(result, is_char);
    } else {
        // Catenate along second axis (columns)
        if (lmat->rows() != rmat->rows()) {
            m->throw_error("LENGTH ERROR: incompatible shapes for catenation");
            return;
        }

        Eigen::MatrixXd result(lmat->rows(), lmat->cols() + rmat->cols());
        result << *lmat, *rmat;
        m->result = m->heap->allocate_matrix(result, is_char);
    }
}

// Transpose (⍉) - monadic: reverse dimensions
void fn_transpose(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // Scalar transpose is identity
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    if (omega->is_vector()) {
        // Vector transpose is identity (returns vector unchanged)
        m->result = m->heap->allocate_vector(omega->as_matrix()->col(0), omega->is_char_data());
        return;
    }

    // Matrix transpose
    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = mat->transpose();
    m->result = m->heap->allocate_matrix(result, omega->is_char_data());
}

// Dyadic Transpose (⍉) - reorder axes
// For 2D: 1 0⍉M is transpose, 0 1⍉M is identity
void fn_dyadic_transpose(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // String → char vector conversion for array operations
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Get permutation as vector
    Eigen::VectorXd perm;
    if (lhs->is_scalar()) {
        perm.resize(1);
        perm(0) = lhs->as_scalar();
    } else if (lhs->is_vector()) {
        perm = lhs->as_matrix()->col(0);
    } else {
        // Flatten matrix row-major
        const Eigen::MatrixXd* mat = lhs->as_matrix();
        perm.resize(mat->size());
        int idx = 0;
        for (int r = 0; r < mat->rows(); ++r) {
            for (int c = 0; c < mat->cols(); ++c) {
                perm(idx++) = (*mat)(r, c);
            }
        }
    }

    if (rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    if (rhs->is_vector()) {
        if (perm.size() != 1 || perm(0) != 0.0) {
            m->throw_error("AXIS ERROR: invalid axis permutation");
            return;
        }
        m->result = m->heap->allocate_vector(rhs->as_matrix()->col(0));
        return;
    }

    // Matrix: permutation must be 0 1 (identity) or 1 0 (transpose)
    if (perm.size() != 2) {
        m->throw_error("LENGTH ERROR: permutation must match array rank");
        return;
    }

    int p0 = static_cast<int>(perm(0));
    int p1 = static_cast<int>(perm(1));

    if (p0 == 0 && p1 == 1) {
        m->result = m->heap->allocate_matrix(*rhs->as_matrix());
    } else if (p0 == 1 && p1 == 0) {
        m->result = m->heap->allocate_matrix(rhs->as_matrix()->transpose());
    } else {
        m->throw_error("AXIS ERROR: invalid axis permutation");
    }
}

// Matrix Inverse (⌹) - monadic
void fn_matrix_inverse(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        double val = omega->as_scalar();
        if (val == 0.0) {
            m->throw_error("DOMAIN ERROR: cannot invert zero");
            return;
        }
        m->result = m->heap->allocate_scalar(1.0 / val);
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = mat->completeOrthogonalDecomposition().pseudoInverse();

    if (omega->is_vector()) {
        // Pseudo-inverse of column vector is row vector (1×n matrix)
        m->result = m->heap->allocate_matrix(result);
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Matrix Divide (⌹) - dyadic: solve B×X = A for X
void fn_matrix_divide(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (rhs->is_scalar()) {
        double divisor = rhs->as_scalar();
        if (divisor == 0.0) {
            m->throw_error("DOMAIN ERROR: division by zero");
            return;
        }
        if (lhs->is_scalar()) {
            m->result = m->heap->allocate_scalar(lhs->as_scalar() / divisor);
        } else {
            Eigen::MatrixXd result = lhs->as_matrix()->array() / divisor;
            if (lhs->is_vector()) {
                m->result = m->heap->allocate_vector(result.col(0));
            } else {
                m->result = m->heap->allocate_matrix(result);
            }
        }
        return;
    }

    // Solve B×X = A using least squares
    Eigen::MatrixXd A = lhs->is_scalar()
        ? Eigen::MatrixXd::Constant(1, 1, lhs->as_scalar())
        : *lhs->as_matrix();
    const Eigen::MatrixXd& B = *rhs->as_matrix();

    Eigen::MatrixXd result = B.colPivHouseholderQr().solve(A);

    if (result.size() == 1) {
        m->result = m->heap->allocate_scalar(result(0, 0));
    } else if (result.cols() == 1) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Iota (⍳) - monadic: generate indices from 0 to n-1
void fn_iota(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (!omega->is_scalar()) {
        m->throw_error("RANK ERROR: iota argument must be scalar");
        return;
    }

    double val = omega->as_scalar();

    // Validate: must be non-negative integer
    if (val < 0.0) {
        m->throw_error("DOMAIN ERROR: iota argument must be non-negative");
        return;
    }
    if (val != std::floor(val)) {
        m->throw_error("DOMAIN ERROR: iota argument must be an integer");
        return;
    }

    int n = static_cast<int>(val);

    // Check implementation limit (ISO 13751 §A.3)
    if (static_cast<size_t>(n) > MAX_ARRAY_SIZE) {
        m->throw_error("LIMIT ERROR: array size exceeds implementation limit");
        return;
    }

    Eigen::VectorXd result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = i + m->io;  // Index origin (⎕IO)
    }
    m->result = m->heap->allocate_vector(result);
}

// First (↑) - monadic: return first element
void fn_first(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // First of scalar is the scalar itself
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);
    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (mat->size() == 0) {
        // First of empty array - return typical element (ISO 13751 §5.3.2)
        // Character arrays: blank ' ', Numeric arrays: zero
        double typical = omega->is_char_data() ? 32.0 : 0.0;
        m->result = m->heap->allocate_scalar(typical);
        return;
    }

    if (omega->is_vector()) {
        // First of vector is the first element as scalar
        m->result = m->heap->allocate_scalar((*mat)(0, 0));
        return;
    }

    // First of matrix is the first row as vector
    Eigen::VectorXd first_row = mat->row(0);
    m->result = m->heap->allocate_vector(first_row, omega->is_char_data());
}

// Take (↑) - dyadic: take first n elements along specified axis
// N↑B takes along first axis; N↑[K]B takes along axis K
void fn_take(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        m->throw_error("RANK ERROR: take count must be scalar");
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

    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);
    bool is_char = rhs->is_char_data();
    const Eigen::MatrixXd* mat = rhs->as_matrix();

    // Determine which axis to take along
    int rank = rhs->is_vector() ? 1 : 2;
    int take_axis = 1;  // Default to first axis

    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be a scalar");
            return;
        }
        take_axis = static_cast<int>(axis->as_scalar());
        if (take_axis < 1 || take_axis > rank) {
            m->throw_error("AXIS ERROR: invalid axis for array rank");
            return;
        }
    }

    // Typical element: blank for char, zero for numeric (ISO 13751 §5.3.2)
    double fill = is_char ? 32.0 : 0.0;

    if (rhs->is_vector()) {
        int len = mat->rows();
        int abs_n = std::abs(n);

        Eigen::VectorXd result(abs_n);

        if (n >= 0) {
            // Take from beginning
            for (int i = 0; i < abs_n; ++i) {
                result(i) = (i < len) ? (*mat)(i, 0) : fill;
            }
        } else {
            // Take from end
            for (int i = 0; i < abs_n; ++i) {
                int src_idx = len - abs_n + i;
                result(i) = (src_idx >= 0) ? (*mat)(src_idx, 0) : fill;
            }
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // Matrix case
    int rows = mat->rows();
    int cols = mat->cols();
    int abs_n = std::abs(n);

    if (take_axis == 1) {
        // Take along first axis (rows)
        Eigen::MatrixXd result(abs_n, cols);

        if (n >= 0) {
            for (int i = 0; i < abs_n; ++i) {
                if (i < rows) {
                    result.row(i) = mat->row(i);
                } else {
                    result.row(i).setConstant(fill);
                }
            }
        } else {
            for (int i = 0; i < abs_n; ++i) {
                int src_idx = rows - abs_n + i;
                if (src_idx >= 0) {
                    result.row(i) = mat->row(src_idx);
                } else {
                    result.row(i).setConstant(fill);
                }
            }
        }
        m->result = m->heap->allocate_matrix(result, is_char);
    } else {
        // Take along second axis (columns)
        Eigen::MatrixXd result(rows, abs_n);

        if (n >= 0) {
            for (int j = 0; j < abs_n; ++j) {
                if (j < cols) {
                    result.col(j) = mat->col(j);
                } else {
                    result.col(j).setConstant(fill);
                }
            }
        } else {
            for (int j = 0; j < abs_n; ++j) {
                int src_idx = cols - abs_n + j;
                if (src_idx >= 0) {
                    result.col(j) = mat->col(src_idx);
                } else {
                    result.col(j).setConstant(fill);
                }
            }
        }
        m->result = m->heap->allocate_matrix(result, is_char);
    }
}

// Drop (↓) - dyadic: drop first n elements
void fn_drop(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        m->throw_error("RANK ERROR: drop count must be scalar");
        return;
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_scalar()) {
        // Dropping from scalar gives empty vector
        Eigen::VectorXd result(0);
        m->result = m->heap->allocate_vector(result);
        return;
    }

    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);
    bool is_char = rhs->is_char_data();
    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        int len = mat->rows();
        int abs_n = std::abs(n);

        if (abs_n >= len) {
            // Drop everything
            Eigen::VectorXd result(0);
            m->result = m->heap->allocate_vector(result, is_char);
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
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // For matrices, determine which axis to drop along
    // Default is axis 1 (first axis = rows)
    // ↓[2] drops along second axis (columns)
    int drop_axis = 1;  // Default: first axis (rows)
    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("RANK ERROR: axis must be scalar");
            return;
        }
        drop_axis = static_cast<int>(axis->as_scalar());
        if (drop_axis < 1 || drop_axis > 2) {
            m->throw_error("AXIS ERROR: axis must be 1 or 2 for matrix");
            return;
        }
    }

    int rows = mat->rows();
    int cols = mat->cols();
    int abs_n = std::abs(n);

    if (drop_axis == 1) {
        // Drop along first axis (rows)
        if (abs_n >= rows) {
            Eigen::MatrixXd result(0, cols);
            m->result = m->heap->allocate_matrix(result, is_char);
            return;
        }

        int result_rows = rows - abs_n;
        Eigen::MatrixXd result(result_rows, cols);

        if (n >= 0) {
            result = mat->bottomRows(result_rows);
        } else {
            result = mat->topRows(result_rows);
        }

        m->result = m->heap->allocate_matrix(result, is_char);
    } else {
        // Drop along second axis (columns)
        if (abs_n >= cols) {
            Eigen::MatrixXd result(rows, 0);
            m->result = m->heap->allocate_matrix(result, is_char);
            return;
        }

        int result_cols = cols - abs_n;
        Eigen::MatrixXd result(rows, result_cols);

        if (n >= 0) {
            result = mat->rightCols(result_cols);
        } else {
            result = mat->leftCols(result_cols);
        }

        m->result = m->heap->allocate_matrix(result, is_char);
    }
}

// ============================================================================
// Reverse/Rotate Functions
// ============================================================================

// Reverse (⌽) - monadic: reverse elements along last axis (or specified axis)
// ⌽B reverses along last axis; ⌽[K]B reverses along axis K
void fn_reverse(Machine* m, Value* axis, Value* omega) {
    if (omega->is_scalar()) {
        // Scalar reversal is identity
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    bool is_char = omega->is_char_data();
    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Determine which axis to reverse along
    // Default: last axis (2 for matrix, 1 for vector)
    int rank = omega->is_vector() ? 1 : 2;
    int reverse_axis = rank;  // Default to last axis (1-indexed)

    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be a scalar");
            return;
        }
        reverse_axis = static_cast<int>(axis->as_scalar());
        // Validate axis (1-indexed with ⎕IO=1)
        if (reverse_axis < 1 || reverse_axis > rank) {
            m->throw_error("AXIS ERROR: invalid axis for array rank");
            return;
        }
    }

    if (omega->is_vector()) {
        // Reverse vector elements (only axis 1 is valid)
        Eigen::VectorXd result(mat->rows());
        for (int i = 0; i < mat->rows(); ++i) {
            result(i) = (*mat)(mat->rows() - 1 - i, 0);
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // Matrix case
    Eigen::MatrixXd result(mat->rows(), mat->cols());

    if (reverse_axis == 1) {
        // Reverse along first axis (rows): swap row order
        for (int i = 0; i < mat->rows(); ++i) {
            for (int j = 0; j < mat->cols(); ++j) {
                result(i, j) = (*mat)(mat->rows() - 1 - i, j);
            }
        }
    } else {
        // Reverse along second/last axis (columns): reverse within each row
        for (int i = 0; i < mat->rows(); ++i) {
            for (int j = 0; j < mat->cols(); ++j) {
                result(i, j) = (*mat)(i, mat->cols() - 1 - j);
            }
        }
    }
    m->result = m->heap->allocate_matrix(result, is_char);
}

// Reverse First (⊖) - monadic: reverse elements along first axis (or specified axis)
// ⊖B reverses along first axis; ⊖[K]B reverses along axis K
void fn_reverse_first(Machine* m, Value* axis, Value* omega) {
    if (omega->is_scalar()) {
        // Scalar reversal is identity
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);
    bool is_char = omega->is_char_data();
    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Determine which axis to reverse along
    // Default: first axis (1)
    int rank = omega->is_vector() ? 1 : 2;
    int reverse_axis = 1;  // Default to first axis (1-indexed)

    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be a scalar");
            return;
        }
        reverse_axis = static_cast<int>(axis->as_scalar());
        // Validate axis (1-indexed with ⎕IO=1)
        if (reverse_axis < 1 || reverse_axis > rank) {
            m->throw_error("AXIS ERROR: invalid axis for array rank");
            return;
        }
    }

    if (omega->is_vector()) {
        // For vectors, first axis is the only axis, so same as reverse
        Eigen::VectorXd result(mat->rows());
        for (int i = 0; i < mat->rows(); ++i) {
            result(i) = (*mat)(mat->rows() - 1 - i, 0);
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // Matrix case
    Eigen::MatrixXd result(mat->rows(), mat->cols());

    if (reverse_axis == 1) {
        // Reverse along first axis (rows): swap row order
        for (int i = 0; i < mat->rows(); ++i) {
            result.row(i) = mat->row(mat->rows() - 1 - i);
        }
    } else {
        // Reverse along second/last axis (columns): reverse within each row
        for (int i = 0; i < mat->rows(); ++i) {
            for (int j = 0; j < mat->cols(); ++j) {
                result(i, j) = (*mat)(i, mat->cols() - 1 - j);
            }
        }
    }
    m->result = m->heap->allocate_matrix(result, is_char);
}

// Tally (≢) - monadic: count along first axis
void fn_tally(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // Scalar has no first axis, tally is 1
        m->result = m->heap->allocate_scalar(1.0);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // First axis is number of rows
    m->result = m->heap->allocate_scalar(static_cast<double>(mat->rows()));
}

// Rotate (⌽) - dyadic: rotate elements along last axis
void fn_rotate(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        m->throw_error("RANK ERROR: rotate count must be scalar");
        return;
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_scalar()) {
        // Rotating a scalar is identity
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);
    bool is_char = rhs->is_char_data();
    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        int len = mat->rows();
        if (len == 0) {
            m->result = m->heap->allocate_vector(mat->col(0), is_char);
            return;
        }
        // Normalize rotation (positive = left rotate, APL convention)
        n = ((n % len) + len) % len;
        Eigen::VectorXd result(len);
        for (int i = 0; i < len; ++i) {
            result(i) = (*mat)((i + n) % len, 0);
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // For matrices: rotate columns within each row
    int cols = mat->cols();
    if (cols == 0) {
        m->result = m->heap->allocate_matrix(*mat, is_char);
        return;
    }
    n = ((n % cols) + cols) % cols;
    Eigen::MatrixXd result(mat->rows(), cols);
    for (int i = 0; i < mat->rows(); ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = (*mat)(i, (j + n) % cols);
        }
    }
    m->result = m->heap->allocate_matrix(result, is_char);
}

// Rotate First (⊖) - dyadic: rotate elements along first axis
void fn_rotate_first(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        m->throw_error("RANK ERROR: rotate count must be scalar");
        return;
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_scalar()) {
        // Rotating a scalar is identity
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);
    bool is_char = rhs->is_char_data();
    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        // For vectors, first axis is the only axis
        int len = mat->rows();
        if (len == 0) {
            m->result = m->heap->allocate_vector(mat->col(0), is_char);
            return;
        }
        n = ((n % len) + len) % len;
        Eigen::VectorXd result(len);
        for (int i = 0; i < len; ++i) {
            result(i) = (*mat)((i + n) % len, 0);
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // For matrices: rotate rows (first axis)
    int rows = mat->rows();
    if (rows == 0) {
        m->result = m->heap->allocate_matrix(*mat, is_char);
        return;
    }
    n = ((n % rows) + rows) % rows;
    Eigen::MatrixXd result(rows, mat->cols());
    for (int i = 0; i < rows; ++i) {
        result.row(i) = mat->row((i + n) % rows);
    }
    m->result = m->heap->allocate_matrix(result, is_char);
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
// Returns index of first occurrence of each element, or ⎕IO+≢lhs if not found
void fn_index_of(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Get lhs as a flat array of values to search in
    Eigen::VectorXd haystack = flatten_value(lhs);
    int io = m->io;
    double not_found = static_cast<double>(haystack.size() + io);  // ⎕IO + length

    // Search for needle in haystack, return index or not_found
    auto find_index = [&haystack, not_found, io](double needle) -> double {
        for (int i = 0; i < haystack.size(); ++i) {
            if (haystack(i) == needle) {
                return static_cast<double>(i + io);  // ⎕IO
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
// ISO 13751 §10.2.25: Returns a simple vector containing all simple scalars in B
//
// TODO: NESTED ARRAYS NOT YET IMPLEMENTED
// Currently this just calls ravel because we don't have nested arrays.
// When nested arrays are implemented:
//   - ∊ should recursively descend into nested structures
//   - ∊(1 (2 3) 4) should return 1 2 3 4 (flatten the nested vector)
//   - ∊ on a simple array should be equivalent to ravel
//
// Related: PerformJuxtaposeK in continuation.cpp is where stranding like
// "1 (2 3) 4" or "{⍵ ⍵}(1 2 3)" would create nested arrays. Currently it
// flattens strands into simple vectors, which is incorrect for APL2-style
// semantics but works until nested arrays are implemented.
//
void fn_enlist(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    // Without nested arrays, enlist = ravel
    fn_ravel(m, axis, omega);
}

// Member Of (∊) - dyadic: check if elements of lhs are in rhs
// Returns boolean array with 1 where element is found, 0 otherwise
void fn_member_of(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

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

// ============================================================================
// Grade Functions (⍋ ⍒)
// ============================================================================

// Grade Up (⍋) - monadic: return indices that would sort array in ascending order
// ⍋ 3 1 4 1 5 → 2 4 1 3 5 (with ⎕IO=1)
void fn_grade_up(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // RANK ERROR: grade requires array, not scalar
        m->throw_error("RANK ERROR: grade requires array");
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);
    Eigen::VectorXd data = flatten_value(omega);
    int n = data.size();

    // Create index array
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Sort indices by corresponding data values (ascending)
    std::sort(indices.begin(), indices.end(), [&data](int a, int b) {
        return data(a) < data(b);
    });

    // Convert to result vector (⎕IO)
    Eigen::VectorXd result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = static_cast<double>(indices[i] + m->io);
    }

    m->result = m->heap->allocate_vector(result);
}

// Grade Down (⍒) - monadic: return indices that would sort array in descending order
// ⍒ 3 1 4 1 5 → 5 3 1 2 4 (with ⎕IO=1)
void fn_grade_down(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // RANK ERROR: grade requires array, not scalar
        m->throw_error("RANK ERROR: grade requires array");
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);
    Eigen::VectorXd data = flatten_value(omega);
    int n = data.size();

    // Create index array
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Sort indices by corresponding data values (descending)
    std::sort(indices.begin(), indices.end(), [&data](int a, int b) {
        return data(a) > data(b);
    });

    // Convert to result vector (⎕IO)
    Eigen::VectorXd result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = static_cast<double>(indices[i] + m->io);
    }

    m->result = m->heap->allocate_vector(result);
}

// Character Grade Up (A⍋B) - dyadic: sort B according to collating sequence A
// ISO 13751 Section 10.2.21
// A must be a character array with rank > 0
// B must be a character array
// Result is a permutation of ⍳1↑⍴B that sorts B's subarrays along first axis
// Uses stable sort; characters not in A sort after all characters in A
void fn_grade_up_dyadic(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // Validate A (collating sequence) is character array with rank > 0
    if (lhs->is_scalar()) {
        m->throw_error("RANK ERROR: collating sequence must have rank > 0");
        return;
    }
    if (!lhs->is_char_data() && !lhs->is_string()) {
        m->throw_error("DOMAIN ERROR: collating sequence must be character");
        return;
    }

    // Validate B is character array
    if (!rhs->is_char_data() && !rhs->is_string()) {
        m->throw_error("DOMAIN ERROR: right argument must be character");
        return;
    }

    Value* A = lhs;
    Value* B = rhs;
    if (A->is_string()) A = A->to_char_vector(m->heap);
    if (B->is_string()) B = B->to_char_vector(m->heap);

    const Eigen::MatrixXd* A_mat = A->as_matrix();
    const Eigen::MatrixXd* B_mat = B->as_matrix();

    // First axis length of B
    int first_axis_len = B->rows();

    // Handle empty B - return empty vector
    if (first_axis_len == 0) {
        m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
        return;
    }

    // Handle single element - return ⍳1
    if (first_axis_len == 1) {
        Eigen::VectorXd result(1);
        result(0) = m->io;
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // Handle empty A - return identity permutation (all chars equal)
    if (A->size() == 0) {
        Eigen::VectorXd result(first_axis_len);
        for (int i = 0; i < first_axis_len; ++i) {
            result(i) = i + m->io;
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // Build collating lookup: char -> position (first occurrence)
    // Characters not in A get position = A->size() (sort last)
    std::map<int, int> char_to_pos;
    int A_size = A->size();

    if (A->is_vector()) {
        for (int i = 0; i < A_size; ++i) {
            int ch = static_cast<int>((*A_mat)(i, 0));
            if (char_to_pos.find(ch) == char_to_pos.end()) {
                char_to_pos[ch] = i;
            }
        }
    } else {
        // For matrix A, use ravel order (last axis varies fastest)
        int rows = A->rows();
        int cols = A->cols();
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int ch = static_cast<int>((*A_mat)(r, c));
                if (char_to_pos.find(ch) == char_to_pos.end()) {
                    char_to_pos[ch] = r * cols + c;
                }
            }
        }
    }

    int not_found_pos = A_size;  // Characters not in A sort last

    // Create indices to sort
    std::vector<int> indices(first_axis_len);
    for (int i = 0; i < first_axis_len; ++i) {
        indices[i] = i;
    }

    // Helper to get collating position of a character
    auto get_pos = [&](int ch) -> int {
        auto it = char_to_pos.find(ch);
        return (it != char_to_pos.end()) ? it->second : not_found_pos;
    };

    // Comparison function based on B's structure
    if (B->is_vector()) {
        // Grade individual characters (1D case)
        auto compare = [&](int i, int j) -> bool {
            int ch_i = static_cast<int>((*B_mat)(i, 0));
            int ch_j = static_cast<int>((*B_mat)(j, 0));
            return get_pos(ch_i) < get_pos(ch_j);
        };
        std::stable_sort(indices.begin(), indices.end(), compare);
    } else {
        // Grade rows lexicographically (matrix case)
        int cols = B->cols();
        auto compare = [&](int i, int j) -> bool {
            for (int k = 0; k < cols; ++k) {
                int ch_i = static_cast<int>((*B_mat)(i, k));
                int ch_j = static_cast<int>((*B_mat)(j, k));
                int pos_i = get_pos(ch_i);
                int pos_j = get_pos(ch_j);
                if (pos_i < pos_j) return true;
                if (pos_i > pos_j) return false;
            }
            return false;  // Rows are equal
        };
        std::stable_sort(indices.begin(), indices.end(), compare);
    }

    // Convert to result vector with ⎕IO
    Eigen::VectorXd result(first_axis_len);
    for (int i = 0; i < first_axis_len; ++i) {
        result(i) = static_cast<double>(indices[i] + m->io);
    }

    m->result = m->heap->allocate_vector(result);
}

// Character Grade Down (A⍒B) - dyadic: sort B descending according to collating sequence A
// ISO 13751 Section 10.2.20
// Same as grade up but with reversed comparison
void fn_grade_down_dyadic(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // Validate A (collating sequence) is character array with rank > 0
    if (lhs->is_scalar()) {
        m->throw_error("RANK ERROR: collating sequence must have rank > 0");
        return;
    }
    if (!lhs->is_char_data() && !lhs->is_string()) {
        m->throw_error("DOMAIN ERROR: collating sequence must be character");
        return;
    }

    // Validate B is character array
    if (!rhs->is_char_data() && !rhs->is_string()) {
        m->throw_error("DOMAIN ERROR: right argument must be character");
        return;
    }

    Value* A = lhs;
    Value* B = rhs;
    if (A->is_string()) A = A->to_char_vector(m->heap);
    if (B->is_string()) B = B->to_char_vector(m->heap);

    const Eigen::MatrixXd* A_mat = A->as_matrix();
    const Eigen::MatrixXd* B_mat = B->as_matrix();

    // First axis length of B
    int first_axis_len = B->rows();

    // Handle empty B - return empty vector
    if (first_axis_len == 0) {
        m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
        return;
    }

    // Handle single element - return ⍳1
    if (first_axis_len == 1) {
        Eigen::VectorXd result(1);
        result(0) = m->io;
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // Handle empty A - return identity permutation (all chars equal)
    if (A->size() == 0) {
        Eigen::VectorXd result(first_axis_len);
        for (int i = 0; i < first_axis_len; ++i) {
            result(i) = i + m->io;
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // Build collating lookup: char -> position (first occurrence)
    std::map<int, int> char_to_pos;
    int A_size = A->size();

    if (A->is_vector()) {
        for (int i = 0; i < A_size; ++i) {
            int ch = static_cast<int>((*A_mat)(i, 0));
            if (char_to_pos.find(ch) == char_to_pos.end()) {
                char_to_pos[ch] = i;
            }
        }
    } else {
        int rows = A->rows();
        int cols = A->cols();
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int ch = static_cast<int>((*A_mat)(r, c));
                if (char_to_pos.find(ch) == char_to_pos.end()) {
                    char_to_pos[ch] = r * cols + c;
                }
            }
        }
    }

    int not_found_pos = A_size;

    std::vector<int> indices(first_axis_len);
    for (int i = 0; i < first_axis_len; ++i) {
        indices[i] = i;
    }

    auto get_pos = [&](int ch) -> int {
        auto it = char_to_pos.find(ch);
        return (it != char_to_pos.end()) ? it->second : not_found_pos;
    };

    // For grade down, unknowns still sort AFTER all known chars (ISO 13751)
    // Compare known chars descending, but unknowns always sort last
    auto compare_desc = [&](int pos_i, int pos_j) -> int {
        bool i_unknown = (pos_i == not_found_pos);
        bool j_unknown = (pos_j == not_found_pos);
        if (i_unknown && j_unknown) return 0;  // Both unknown: equal
        if (i_unknown) return 1;               // i unknown: i sorts after j
        if (j_unknown) return -1;              // j unknown: j sorts after i
        // Both known: descending order
        if (pos_i > pos_j) return -1;
        if (pos_i < pos_j) return 1;
        return 0;
    };

    if (B->is_vector()) {
        // Grade individual characters descending
        auto compare = [&](int i, int j) -> bool {
            int ch_i = static_cast<int>((*B_mat)(i, 0));
            int ch_j = static_cast<int>((*B_mat)(j, 0));
            return compare_desc(get_pos(ch_i), get_pos(ch_j)) < 0;
        };
        std::stable_sort(indices.begin(), indices.end(), compare);
    } else {
        int cols = B->cols();
        auto compare = [&](int i, int j) -> bool {
            for (int k = 0; k < cols; ++k) {
                int ch_i = static_cast<int>((*B_mat)(i, k));
                int ch_j = static_cast<int>((*B_mat)(j, k));
                int cmp = compare_desc(get_pos(ch_i), get_pos(ch_j));
                if (cmp < 0) return true;
                if (cmp > 0) return false;
            }
            return false;
        };
        std::stable_sort(indices.begin(), indices.end(), compare);
    }

    Eigen::VectorXd result(first_axis_len);
    for (int i = 0; i < first_axis_len; ++i) {
        result(i) = static_cast<double>(indices[i] + m->io);
    }

    m->result = m->heap->allocate_vector(result);
}

// ============================================================================
// Replicate Function (/)
// ============================================================================

// Replicate (/) - dyadic: replicate elements of rhs by counts in lhs
// 2 0 3 / 1 2 3 → 1 1 3 3 3
// 1 1 1 / 4 5 6 → 4 5 6 (compress)
// 0 1 0 / 4 5 6 → 5 (filter)
void fn_replicate(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    bool is_char = rhs->is_char_data();

    // Get counts from lhs
    Eigen::VectorXd counts = flatten_value(lhs);

    // For now, support vectors only (last axis replication)
    if (!rhs->is_scalar() && !rhs->is_vector()) {
        // Matrix case: replicate along last axis (columns)
        const Eigen::MatrixXd* mat = rhs->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();

        if (counts.size() != cols) {
            m->throw_error("LENGTH ERROR: replicate count must match array length");
            return;
        }

        // Calculate total output columns
        int total_cols = 0;
        for (int i = 0; i < counts.size(); ++i) {
            int c = static_cast<int>(counts(i));
            if (c < 0) {
                m->throw_error("DOMAIN ERROR: replicate count must be non-negative");
                return;
            }
            total_cols += c;
        }

        if (total_cols == 0) {
            // Empty result - return empty vector (shape 0)
            Eigen::VectorXd empty(0);
            m->result = m->heap->allocate_vector(empty, is_char);
            return;
        }

        Eigen::MatrixXd result(rows, total_cols);
        int out_col = 0;
        for (int j = 0; j < cols; ++j) {
            int rep = static_cast<int>(counts(j));
            for (int r = 0; r < rep; ++r) {
                result.col(out_col++) = mat->col(j);
            }
        }

        m->result = m->heap->allocate_matrix(result, is_char);
        return;
    }

    // Scalar or vector case
    Eigen::VectorXd data = flatten_value(rhs);

    if (counts.size() != data.size()) {
        m->throw_error("LENGTH ERROR: replicate count must match array length");
        return;
    }

    // Calculate total output size
    int total = 0;
    for (int i = 0; i < counts.size(); ++i) {
        int c = static_cast<int>(counts(i));
        if (c < 0) {
            m->throw_error("DOMAIN ERROR: replicate count must be non-negative");
            return;
        }
        total += c;
    }

    if (total == 0) {
        // Empty result
        Eigen::VectorXd empty(0);
        m->result = m->heap->allocate_vector(empty, is_char);
        return;
    }

    // Build result
    Eigen::VectorXd result(total);
    int out_idx = 0;
    for (int i = 0; i < data.size(); ++i) {
        int rep = static_cast<int>(counts(i));
        for (int r = 0; r < rep; ++r) {
            result(out_idx++) = data(i);
        }
    }

    m->result = m->heap->allocate_vector(result, is_char);
}

// ============================================================================
// Set Functions (∪ ~)
// ============================================================================

// Helper: check if value exists in first n elements of array
static bool value_in_array(double val, const Eigen::VectorXd& arr, int n) {
    for (int i = 0; i < n; ++i) {
        if (arr(i) == val) return true;
    }
    return false;
}

// Unique (∪ monadic) - return unique elements in order of first appearance
// ∪ 1 2 2 3 1 4 → 1 2 3 4
void fn_unique(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        m->result = omega;
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);
    Eigen::VectorXd data = flatten_value(omega);
    int n = data.size();

    // First pass: count unique elements
    int unique_count = 0;
    for (int i = 0; i < n; ++i) {
        bool found = false;
        for (int j = 0; j < i; ++j) {
            if (data(j) == data(i)) {
                found = true;
                break;
            }
        }
        if (!found) unique_count++;
    }

    if (unique_count == 0) {
        m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
        return;
    }

    // Second pass: collect unique elements
    Eigen::VectorXd result(unique_count);
    int out_idx = 0;
    for (int i = 0; i < n; ++i) {
        if (!value_in_array(data(i), result, out_idx)) {
            result(out_idx++) = data(i);
        }
    }

    m->result = m->heap->allocate_vector(result);
}

// Union (∪ dyadic) - unique elements from both arrays (left first, then unique from right)
// 1 2 3 ∪ 3 4 5 → 1 2 3 4 5
void fn_union(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    Eigen::VectorXd left = flatten_value(lhs);
    Eigen::VectorXd right = flatten_value(rhs);

    // First pass: count unique from left
    int left_unique = 0;
    for (int i = 0; i < left.size(); ++i) {
        bool found = false;
        for (int j = 0; j < i; ++j) {
            if (left(j) == left(i)) {
                found = true;
                break;
            }
        }
        if (!found) left_unique++;
    }

    // Build unique left first (we need it to check right against)
    Eigen::VectorXd left_uniq(left_unique);
    int idx = 0;
    for (int i = 0; i < left.size(); ++i) {
        if (!value_in_array(left(i), left_uniq, idx)) {
            left_uniq(idx++) = left(i);
        }
    }

    // Count how many from right are new
    int right_new = 0;
    for (int i = 0; i < right.size(); ++i) {
        bool in_left = value_in_array(right(i), left_uniq, left_unique);
        bool dup_in_right = false;
        for (int j = 0; j < i; ++j) {
            if (right(j) == right(i)) {
                dup_in_right = true;
                break;
            }
        }
        if (!in_left && !dup_in_right) right_new++;
    }

    // Build result
    Eigen::VectorXd result(left_unique + right_new);
    result.head(left_unique) = left_uniq;

    idx = left_unique;
    for (int i = 0; i < right.size(); ++i) {
        if (!value_in_array(right(i), result, idx)) {
            result(idx++) = right(i);
        }
    }

    m->result = m->heap->allocate_vector(result);
}

// Without (~ dyadic) - elements of left that are not in right
// 1 2 3 4 5 ~ 2 4 → 1 3 5
void fn_without(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    Eigen::VectorXd left = flatten_value(lhs);
    Eigen::VectorXd right = flatten_value(rhs);

    // First pass: count elements not in right
    int count = 0;
    for (int i = 0; i < left.size(); ++i) {
        bool in_right = false;
        for (int j = 0; j < right.size(); ++j) {
            if (right(j) == left(i)) {
                in_right = true;
                break;
            }
        }
        if (!in_right) count++;
    }

    if (count == 0) {
        m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
        return;
    }

    // Second pass: collect elements
    Eigen::VectorXd result(count);
    int out_idx = 0;
    for (int i = 0; i < left.size(); ++i) {
        bool in_right = false;
        for (int j = 0; j < right.size(); ++j) {
            if (right(j) == left(i)) {
                in_right = true;
                break;
            }
        }
        if (!in_right) {
            result(out_idx++) = left(i);
        }
    }

    m->result = m->heap->allocate_vector(result);
}

// ============================================================================
// Random Functions (?)
// ============================================================================

// Roll (? monadic) - random integer from ⎕IO to N-1+⎕IO
// ?N returns random integer in [⎕IO, N-1+⎕IO]
// Uses machine's RNG seeded by ⎕RL for reproducibility
void fn_roll(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    int io = m->io;
    if (omega->is_scalar()) {
        int n = static_cast<int>(omega->data.scalar);
        if (n <= 0) {
            m->throw_error("DOMAIN ERROR: roll argument must be positive");
            return;
        }
        std::uniform_int_distribution<int> dist(io, n - 1 + io);  // ⎕IO
        m->result = m->heap->allocate_scalar(static_cast<double>(dist(m->rng)));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result(mat->rows(), mat->cols());

    for (int i = 0; i < mat->size(); ++i) {
        int n = static_cast<int>(mat->data()[i]);
        if (n <= 0) {
            m->throw_error("DOMAIN ERROR: roll argument must be positive");
            return;
        }
        std::uniform_int_distribution<int> dist(io, n - 1 + io);  // ⎕IO
        result(i) = static_cast<double>(dist(m->rng));
    }

    if (omega->is_vector()) {
        m->result = m->heap->allocate_vector(result.col(0));
    } else {
        m->result = m->heap->allocate_matrix(result);
    }
}

// Deal (? dyadic) - A unique random values from 1 to B
// A?B returns A unique random integers from [1, B] (1-based per ISO 13751)
void fn_deal(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (!lhs->is_scalar() || !rhs->is_scalar()) {
        m->throw_error("DOMAIN ERROR: deal arguments must be scalars");
        return;
    }

    int a = static_cast<int>(lhs->data.scalar);
    int b = static_cast<int>(rhs->data.scalar);

    if (a < 0 || b <= 0) {
        m->throw_error("DOMAIN ERROR: deal arguments must be positive");
        return;
    }

    if (a > b) {
        m->throw_error("DOMAIN ERROR: cannot deal more values than range");
        return;
    }

    if (a == 0) {
        m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
        return;
    }

    // Fisher-Yates shuffle approach for unique selection
    // Create array 1..b (1-based per ISO 13751), shuffle first 'a' elements
    Eigen::VectorXd pool(b);
    for (int i = 0; i < b; ++i) {
        pool(i) = static_cast<double>(i + m->io);  // ⎕IO
    }

    // Partial Fisher-Yates: only shuffle first 'a' positions
    for (int i = 0; i < a; ++i) {
        std::uniform_int_distribution<int> dist(i, b - 1);
        int j = dist(m->rng);
        std::swap(pool(i), pool(j));
    }

    // Return first 'a' elements
    Eigen::VectorXd result = pool.head(a);
    m->result = m->heap->allocate_vector(result);
}

// ============================================================================
// Expand Function (\)
// ============================================================================

// Expand (\ dyadic) - opposite of replicate
// Where A is 1, take from B. Where A is 0, insert fill element (0)
// 1 0 1 0 0 1 \ 'ABC' → 'A B  C' (with spaces being fill)
// 1 0 1 1 \ 1 2 3 → 1 0 2 3
void fn_expand(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    bool is_char = rhs->is_char_data();

    // Get boolean mask from lhs
    Eigen::VectorXd mask = flatten_value(lhs);

    // Count number of 1s in mask - must equal length of rhs
    int ones_count = 0;
    for (int i = 0; i < mask.size(); ++i) {
        int val = static_cast<int>(mask(i));
        if (val != 0 && val != 1) {
            m->throw_error("DOMAIN ERROR: expand mask must be boolean");
            return;
        }
        if (val == 1) ones_count++;
    }

    // Typical element: blank for char, zero for numeric (ISO 13751 §5.3.2)
    double fill = is_char ? 32.0 : 0.0;

    // Handle scalar/vector rhs
    // ISO 10.2.6: "If B is a scalar, set B1 to (+/A1)µB" - extend scalar to ones_count copies
    if (rhs->is_scalar()) {
        double val = rhs->data.scalar;
        Eigen::VectorXd result(mask.size());
        for (int i = 0; i < mask.size(); ++i) {
            if (static_cast<int>(mask(i)) == 1) {
                result(i) = val;  // Use scalar value for each 1
            } else {
                result(i) = fill;  // Fill element for each 0
            }
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    if (!rhs->is_vector()) {
        // Matrix case: expand along last axis (columns)
        const Eigen::MatrixXd* mat = rhs->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();

        if (ones_count != cols) {
            m->throw_error("LENGTH ERROR: expand mask ones must match array length");
            return;
        }

        Eigen::MatrixXd result(rows, mask.size());
        int src_col = 0;
        for (int j = 0; j < mask.size(); ++j) {
            if (static_cast<int>(mask(j)) == 1) {
                result.col(j) = mat->col(src_col++);
            } else {
                result.col(j).setConstant(fill);  // Fill column
            }
        }

        m->result = m->heap->allocate_matrix(result, is_char);
        return;
    }

    // Vector case
    Eigen::VectorXd data = flatten_value(rhs);

    if (ones_count != data.size()) {
        m->throw_error("LENGTH ERROR: expand mask ones must match array length");
        return;
    }

    Eigen::VectorXd result(mask.size());
    int src_idx = 0;
    for (int i = 0; i < mask.size(); ++i) {
        if (static_cast<int>(mask(i)) == 1) {
            result(i) = data(src_idx++);
        } else {
            result(i) = fill;  // Fill element
        }
    }

    m->result = m->heap->allocate_vector(result, is_char);
}

// Expand-first (⍀ dyadic) - insert fill elements along first axis
// A⍀B - A is boolean mask, B is data array
// Expands along first axis (rows) for matrices
void fn_expand_first(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    bool is_char = rhs->is_char_data();

    // Get boolean mask from lhs
    Eigen::VectorXd mask = flatten_value(lhs);

    // Count number of 1s in mask - must equal length of rhs along first axis
    int ones_count = 0;
    for (int i = 0; i < mask.size(); ++i) {
        int val = static_cast<int>(mask(i));
        if (val != 0 && val != 1) {
            m->throw_error("DOMAIN ERROR: expand mask must be boolean");
            return;
        }
        if (val == 1) ones_count++;
    }

    // Typical element: blank for char, zero for numeric (ISO 13751 §5.3.2)
    double fill = is_char ? 32.0 : 0.0;

    // Handle scalar rhs
    if (rhs->is_scalar()) {
        double val = rhs->data.scalar;
        Eigen::VectorXd result(mask.size());
        for (int i = 0; i < mask.size(); ++i) {
            if (static_cast<int>(mask(i)) == 1) {
                result(i) = val;
            } else {
                result(i) = fill;
            }
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // For vectors, expand-first is same as expand (only one axis)
    if (rhs->is_vector()) {
        fn_expand(m, axis, lhs, rhs);
        return;
    }

    // Matrix case: expand along first axis (rows)
    const Eigen::MatrixXd* mat = rhs->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (ones_count != rows) {
        m->throw_error("LENGTH ERROR: expand mask ones must match array length");
        return;
    }

    Eigen::MatrixXd result(mask.size(), cols);
    int src_row = 0;
    for (int i = 0; i < mask.size(); ++i) {
        if (static_cast<int>(mask(i)) == 1) {
            result.row(i) = mat->row(src_row++);
        } else {
            result.row(i).setConstant(fill);  // Fill row
        }
    }

    m->result = m->heap->allocate_matrix(result, is_char);
}

// ============================================================================
// Encode/Decode Functions (⊥ ⊤)
// ============================================================================

// Decode (⊥ dyadic) - base value / polynomial evaluation
// A⊥B evaluates B as digits in radix A using Horner's method
// 2⊥1 0 1 1 → 11 (binary to decimal)
// 10⊥1 2 3 → 123 (decimal digits)
// 24 60 60⊥1 30 45 → 5445 (hours:mins:secs to seconds)
void fn_decode(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // Fast path: scalar radix with scalar digit → just return the digit
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Get the radix and digits as vectors
    Eigen::VectorXd radix = flatten_value(lhs);
    Eigen::VectorXd digits = flatten_value(rhs);

    int n = digits.size();

    // If radix is scalar, extend it to match digits length
    if (radix.size() == 1) {
        double r = radix(0);
        radix.resize(n);
        radix.setConstant(r);
    }

    // Radix and digits must have same length after extension
    if (radix.size() != n) {
        m->throw_error("LENGTH ERROR: decode radix and digits must have same length");
        return;
    }

    if (n == 0) {
        m->result = m->heap->allocate_scalar(0.0);
        return;
    }

    // Horner's method: Z = digits[0], then Z = radix[i]*Z + digits[i] for i=1..n-1
    double z = digits(0);
    for (int i = 1; i < n; ++i) {
        z = radix(i) * z + digits(i);
    }

    m->result = m->heap->allocate_scalar(z);
}

// Encode (⊤ dyadic) - representation / convert to digits in radix
// A⊤B converts B to representation in radix A
// 2 2 2 2⊤11 → 1 0 1 1 (decimal to binary)
// 10 10 10⊤345 → 3 4 5 (decimal digits)
// 24 60 60⊤5445 → 1 30 45 (seconds to hours:mins:secs)
void fn_encode(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Get the radix vector
    Eigen::VectorXd radix = flatten_value(lhs);
    int n = radix.size();

    if (n == 0) {
        m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
        return;
    }

    // Right argument should be scalar for basic encode
    if (!rhs->is_scalar()) {
        // For vector rhs, apply encode to each element (result is matrix)
        Eigen::VectorXd values = flatten_value(rhs);
        int num_values = values.size();

        Eigen::MatrixXd result(n, num_values);
        for (int v = 0; v < num_values; ++v) {
            double val = values(v);
            // Work from right to left (last radix first)
            for (int i = n - 1; i >= 0; --i) {
                double r = radix(i);
                if (r == 0) {
                    // Special case: radix 0 means "remainder" (no modulo)
                    result(i, v) = val;
                    val = 0;
                } else {
                    double digit = std::fmod(val, r);
                    if (digit < 0) digit += r;  // Ensure non-negative
                    result(i, v) = digit;
                    val = std::floor(val / r);
                }
            }
        }
        m->result = m->heap->allocate_matrix(result);
        return;
    }

    double val = rhs->as_scalar();
    Eigen::VectorXd result(n);

    // Work from right to left (last radix first)
    for (int i = n - 1; i >= 0; --i) {
        double r = radix(i);
        if (r == 0) {
            // Special case: radix 0 means "remainder" (no modulo)
            result(i) = val;
            val = 0;
        } else {
            double digit = std::fmod(val, r);
            if (digit < 0) digit += r;  // Ensure non-negative
            result(i) = digit;
            val = std::floor(val / r);
        }
    }

    m->result = m->heap->allocate_vector(result);
}

// ============================================================================
// Squad (Index): I⌷A per ISO 13751
// ============================================================================
// APL uses 1-based indexing (⎕IO=1)
// I⌷A: I=indices (left), A=array (right)

void fn_squad(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    // ISO 13751: I⌷A where lhs=indices, rhs=array
    // Curry finalization is handled by DispatchFunctionK in the continuation graph
    Value* indices = lhs;
    Value* array = rhs;

    // Function-with-axis: F[k] parses as k⌷F where lhs=axis, rhs=function
    // Creates a curry with axis set - the function waits for its operand(s)
    if (rhs->is_function()) {
        Value* axis = lhs;
        Value* fn = rhs;
        // G_PRIME curry with no first_arg, but with axis set
        Value* curried = m->heap->allocate_curried_fn(fn, nullptr, Value::CurryType::G_PRIME, axis);
        m->result = curried;
        return;
    }

    // Convert STRING to char vector so all arrays use the same code path
    if (array->is_string()) array = array->to_char_vector(m->heap);

    bool is_char = array->is_char_data();

    // Handle array indexing - scalars cannot be indexed (ISO 13751)
    if (array->is_scalar()) {
        m->throw_error("RANK ERROR: cannot index scalar");
        return;
    }
    if (!array->is_array()) {
        m->throw_error("DOMAIN ERROR: cannot index non-array value");
        return;
    }

    // Helper lambda to validate index is near-integer
    auto validate_index = [m](double val) -> bool {
        double rounded = std::round(val);
        if (std::abs(val - rounded) > 1e-10) {
            m->throw_error("DOMAIN ERROR: index must be integer");
            return false;
        }
        return true;
    };

    const Eigen::MatrixXd* arr = array->as_matrix();
    int rows = arr->rows();
    int cols = arr->cols();
    bool is_vec = (cols == 1);

    if (indices->is_scalar()) {
        double idx_val = indices->as_scalar();
        if (!validate_index(idx_val)) return;
        int idx = static_cast<int>(std::round(idx_val)) - m->io;  // ⎕IO
        if (is_vec) {
            if (idx < 0 || idx >= rows) {
                m->throw_error("INDEX ERROR: index out of bounds");
                return;
            }
            m->result = m->heap->allocate_scalar((*arr)(idx, 0));
        } else {
            // Linear indexing into matrix (row-major order)
            int size = rows * cols;
            if (idx < 0 || idx >= size) {
                m->throw_error("INDEX ERROR: index out of bounds");
                return;
            }
            int row = idx / cols;
            int col = idx % cols;
            m->result = m->heap->allocate_scalar((*arr)(row, col));
        }
    } else if (indices->is_array()) {
        const Eigen::MatrixXd* idx_mat = indices->as_matrix();
        int n = idx_mat->rows();

        if (is_vec) {
            Eigen::VectorXd result(n);
            for (int i = 0; i < n; i++) {
                double idx_val = (*idx_mat)(i, 0);
                if (!validate_index(idx_val)) return;
                int idx = static_cast<int>(std::round(idx_val)) - m->io;  // ⎕IO
                if (idx < 0 || idx >= rows) {
                    m->throw_error("INDEX ERROR: index out of bounds");
                    return;
                }
                result(i) = (*arr)(idx, 0);
            }
            m->result = m->heap->allocate_vector(result, is_char);
        } else {
            Eigen::MatrixXd result(n, cols);
            for (int i = 0; i < n; i++) {
                double idx_val = (*idx_mat)(i, 0);
                if (!validate_index(idx_val)) return;
                int idx = static_cast<int>(std::round(idx_val)) - m->io;  // ⎕IO
                if (idx < 0 || idx >= rows) {
                    m->throw_error("INDEX ERROR: index out of bounds");
                    return;
                }
                result.row(i) = arr->row(idx);
            }
            m->result = m->heap->allocate_matrix(result, is_char);
        }
    } else {
        m->throw_error("DOMAIN ERROR: index must be numeric");
    }
}

// ============================================================================
// Table Function (⍸)
// ============================================================================

// Table (⍸) - monadic: convert array to matrix
// ISO 13751: Z is a matrix containing the elements of B
// - If B is a scalar, Z has shape 1 1
// - If B is a vector (length n), Z has shape n 1
// - If B has shape s1 s2 ... sk, Z has shape s1 (s2×s3×...×sk)
void fn_table(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // Scalar → 1×1 matrix
        Eigen::MatrixXd result(1, 1);
        result(0, 0) = omega->as_scalar();
        m->result = m->heap->allocate_matrix(result);
        return;
    }

    // String → char vector conversion
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (omega->is_vector()) {
        // Vector → n×1 matrix
        Eigen::MatrixXd result(rows, 1);
        for (int i = 0; i < rows; ++i) {
            result(i, 0) = (*mat)(i, 0);
        }
        m->result = m->heap->allocate_matrix(result);
        return;
    }

    // Matrix → same matrix (already 2D, so shape s1 × s2 is unchanged)
    // For higher-dimensional arrays (not yet supported), would be s1 × (product of rest)
    m->result = m->heap->allocate_matrix(*mat);
}

// Depth (≡ monadic) - nesting level of array
// ISO 13751 Section 8.2.5: simple-scalar → 0, array → 1 + max depth of elements
// Since we don't support nested arrays, depth is always 0 for scalars, 1 for arrays
void fn_depth(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(0.0);
    } else {
        // All our arrays are simple (non-nested), so depth is 1
        m->result = m->heap->allocate_scalar(1.0);
    }
}

// Match (≡ dyadic) - returns 1 if arguments are identical, 0 otherwise
// ISO 13751 Section 10.2.53: A≡B returns 1 if A and B are identical
void fn_match(Machine* m, Value* axis, Value* alpha, Value* omega) {
    REJECT_AXIS(m, axis);
    // Different types never match
    if (alpha->tag != omega->tag) {
        m->result = m->heap->allocate_scalar(0.0);
        return;
    }

    // Handle scalars
    if (alpha->is_scalar() && omega->is_scalar()) {
        double a = alpha->as_scalar();
        double o = omega->as_scalar();
        // Handle NaN: NaN ≡ NaN should be 1 (identical)
        if (std::isnan(a) && std::isnan(o)) {
            m->result = m->heap->allocate_scalar(1.0);
        } else {
            m->result = m->heap->allocate_scalar(a == o ? 1.0 : 0.0);
        }
        return;
    }

    // Handle strings - pointer comparison is valid since all strings are interned
    if (alpha->is_string() && omega->is_string()) {
        m->result = m->heap->allocate_scalar(
            alpha->as_string() == omega->as_string() ? 1.0 : 0.0);
        return;
    }

    // Handle arrays
    if (alpha->is_array() && omega->is_array()) {
        const Eigen::MatrixXd* a_mat = alpha->as_matrix();
        const Eigen::MatrixXd* o_mat = omega->as_matrix();

        // Shape must match
        if (a_mat->rows() != o_mat->rows() || a_mat->cols() != o_mat->cols()) {
            m->result = m->heap->allocate_scalar(0.0);
            return;
        }

        // Character data flag must match
        if (alpha->is_char_data() != omega->is_char_data()) {
            m->result = m->heap->allocate_scalar(0.0);
            return;
        }

        // All elements must match
        for (int i = 0; i < a_mat->rows(); i++) {
            for (int j = 0; j < a_mat->cols(); j++) {
                double av = (*a_mat)(i, j);
                double ov = (*o_mat)(i, j);
                // Handle NaN comparison
                if (std::isnan(av) && std::isnan(ov)) continue;
                if (av != ov) {
                    m->result = m->heap->allocate_scalar(0.0);
                    return;
                }
            }
        }
        m->result = m->heap->allocate_scalar(1.0);
        return;
    }

    // Different types (scalar vs array, etc.)
    m->result = m->heap->allocate_scalar(0.0);
}

// Right (⊢ monadic and ⊣ monadic) - identity function
// ISO 13751 Section 10.2.18: ⊢B returns B
// Also used for monadic ⊣ per spec (both monadic forms return the argument)
void fn_right(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    m->result = omega;
}

// Left (⊣ dyadic) - returns left argument
// ISO 13751 Section 10.2.17: A⊣B returns A
void fn_left(Machine* m, Value* axis, Value* alpha, Value* /* omega */) {
    REJECT_AXIS(m, axis);
    (void)axis;  // Unused
    m->result = alpha;
}

// Right (⊢ dyadic) - returns right argument
// ISO 13751 Section 10.2.18: A⊢B returns B
void fn_right_dyadic(Machine* m, Value* axis, Value* /* alpha */, Value* omega) {
    REJECT_AXIS(m, axis);
    (void)axis;  // Unused
    m->result = omega;
}

// Catenate First (⍪ dyadic) - join along first axis
// ISO 13751 Section 8.3.2: A⍪B is A,[1]B
void fn_catenate_first(Machine* m, Value* axis, Value* alpha, Value* omega) {
    // String → char vector conversion
    if (alpha->is_string()) alpha = alpha->to_char_vector(m->heap);
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    // Preserve char data if both operands are char data
    bool is_char = alpha->is_char_data() && omega->is_char_data();

    // Handle scalars: treat as 1×1 matrices
    if (alpha->is_scalar() && omega->is_scalar()) {
        Eigen::MatrixXd result(2, 1);
        result(0, 0) = alpha->as_scalar();
        result(1, 0) = omega->as_scalar();
        m->result = m->heap->allocate_matrix(result);
        return;
    }

    // Determine target column count for scalar extension
    int target_cols = 1;
    if (!alpha->is_scalar()) {
        target_cols = alpha->is_vector() ? alpha->size() : alpha->cols();
    }
    if (!omega->is_scalar()) {
        int omega_cols = omega->is_vector() ? omega->size() : omega->cols();
        if (target_cols == 1) {
            target_cols = omega_cols;
        }
    }

    // Convert scalars/vectors to matrix form for consistent handling
    Eigen::MatrixXd mat_a, mat_b;

    if (alpha->is_scalar()) {
        // Scalar extension: replicate to match target columns
        mat_a = Eigen::MatrixXd::Constant(1, target_cols, alpha->as_scalar());
    } else if (alpha->is_vector()) {
        // Vector becomes 1×n row (for first-axis catenation)
        const Eigen::MatrixXd* v = alpha->as_matrix();
        mat_a = Eigen::MatrixXd(1, v->rows());
        for (int i = 0; i < v->rows(); ++i) {
            mat_a(0, i) = (*v)(i, 0);
        }
    } else {
        mat_a = *alpha->as_matrix();
    }

    if (omega->is_scalar()) {
        // Scalar extension: replicate to match target columns
        mat_b = Eigen::MatrixXd::Constant(1, target_cols, omega->as_scalar());
    } else if (omega->is_vector()) {
        // Vector becomes 1×n row (for first-axis catenation)
        const Eigen::MatrixXd* v = omega->as_matrix();
        mat_b = Eigen::MatrixXd(1, v->rows());
        for (int i = 0; i < v->rows(); ++i) {
            mat_b(0, i) = (*v)(i, 0);
        }
    } else {
        mat_b = *omega->as_matrix();
    }

    // Check column compatibility (must match for first-axis catenation)
    if (mat_a.cols() != mat_b.cols()) {
        m->throw_error("LENGTH ERROR: incompatible shapes for ⍪");
        return;
    }

    // Concatenate along first axis (vertically stack rows)
    Eigen::MatrixXd result(mat_a.rows() + mat_b.rows(), mat_a.cols());
    result.topRows(mat_a.rows()) = mat_a;
    result.bottomRows(mat_b.rows()) = mat_b;

    m->result = m->heap->allocate_matrix(result, is_char);
}

// ============================================================================
// Execute Function (⍎)
// ============================================================================

// Execute (⍎) - parse and evaluate a string as APL code
void fn_execute(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    // Convert to STRING if needed (handles both STRING and char vectors)
    Value* str_val;
    if (omega->is_string()) {
        str_val = omega;
    } else if (omega->is_array() && omega->is_char_data()) {
        str_val = omega->to_string_value(m->heap);
    } else {
        m->throw_error("DOMAIN ERROR: execute requires a string");
        return;
    }

    const char* code = str_val->as_string();

    // Empty string returns zilde (empty numeric vector)
    if (code[0] == '\0') {
        m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
        return;
    }

    Continuation* k = m->parser->parse(code);

    if (!k) {
        // Parse error
        m->throw_error(m->parser->get_error().c_str());
        return;
    }

    // Push the parsed continuation to execute
    m->push_kont(k);
}

// ============================================================================
// Format (⍕) - ISO 13751 Section 15.4
// ============================================================================

// Helper: format a single number according to print precision
// Returns string with appropriate format (decimal or exponential)
static std::string format_number_pp(double val, int pp) {
    // Handle special values
    if (std::isinf(val)) {
        return val > 0 ? "∞" : "¯∞";
    }
    if (std::isnan(val)) {
        return "NaN";
    }

    // Handle zero specially
    if (val == 0.0) {
        return "0";
    }

    bool negative = val < 0;
    double abs_val = std::abs(val);

    // Determine if exponential form is needed:
    // - Value >= 10^pp (too many digits left of decimal)
    // - Value < 10^-5 (more than 5 leading zeros)
    double upper_limit = std::pow(10.0, pp);
    double lower_limit = 1e-5;

    bool use_exponential = (abs_val >= upper_limit) || (abs_val < lower_limit && abs_val > 0);

    std::ostringstream oss;
    oss << std::setprecision(pp);

    if (use_exponential) {
        // Exponential form: d.dddE±nn
        oss << std::scientific << abs_val;
        std::string s = oss.str();

        // Convert 'e' to 'E' and handle exponent sign
        size_t e_pos = s.find('e');
        if (e_pos != std::string::npos) {
            s[e_pos] = 'E';
            // Check for negative exponent
            if (e_pos + 1 < s.length() && s[e_pos + 1] == '-') {
                s.replace(e_pos + 1, 1, "¯");
            } else if (e_pos + 1 < s.length() && s[e_pos + 1] == '+') {
                // Remove the + sign
                s.erase(e_pos + 1, 1);
            }
            // Remove leading zeros from exponent
            size_t exp_start = e_pos + 1;
            if (s[exp_start] == '\xC2') exp_start += 2; // Skip ¯ if present
            while (exp_start < s.length() - 1 && s[exp_start] == '0') {
                s.erase(exp_start, 1);
            }
        }

        return (negative ? "¯" : "") + s;
    } else {
        // Decimal form
        // Check if it's an integer
        if (abs_val == std::floor(abs_val) && abs_val < 1e15) {
            oss.str("");
            oss << std::fixed << std::setprecision(0) << abs_val;
        } else {
            oss << std::defaultfloat << abs_val;
            std::string s = oss.str();
            // Remove trailing zeros after decimal point
            if (s.find('.') != std::string::npos) {
                size_t last = s.find_last_not_of('0');
                if (last != std::string::npos && s[last] == '.') {
                    last--;  // Also remove the decimal point if nothing after
                }
                s = s.substr(0, last + 1);
            }
            return (negative ? "¯" : "") + s;
        }

        return (negative ? "¯" : "") + oss.str();
    }
}

// Helper: format a number with fixed decimal places
static std::string format_number_fixed(double val, int width, int decimals) {
    bool negative = val < 0;
    double abs_val = std::abs(val);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << abs_val;
    std::string num_str = oss.str();

    // Add high minus for negative
    std::string result = (negative ? "¯" : "") + num_str;

    // Right-justify in field
    if ((int)result.length() < width) {
        result = std::string(width - result.length(), ' ') + result;
    }

    return result;
}

// Helper: format a number in exponential form with specified mantissa digits
static std::string format_number_exponential(double val, int width, int mantissa_digits) {
    bool negative = val < 0;
    double abs_val = std::abs(val);

    std::ostringstream oss;
    // mantissa_digits is the total significant digits (including the one before decimal)
    oss << std::scientific << std::setprecision(mantissa_digits - 1) << abs_val;
    std::string s = oss.str();

    // Convert 'e' to 'E' and fix exponent
    size_t e_pos = s.find('e');
    if (e_pos != std::string::npos) {
        s[e_pos] = 'E';
        if (e_pos + 1 < s.length() && s[e_pos + 1] == '-') {
            s.replace(e_pos + 1, 1, "¯");
        } else if (e_pos + 1 < s.length() && s[e_pos + 1] == '+') {
            s.erase(e_pos + 1, 1);
        }
        // Remove leading zeros from exponent
        size_t exp_start = e_pos + 1;
        if (exp_start < s.length() && s[exp_start] == '\xC2') exp_start += 2; // Skip ¯
        while (exp_start < s.length() - 1 && s[exp_start] == '0') {
            s.erase(exp_start, 1);
        }
    }

    std::string result = (negative ? "¯" : "") + s;

    // Right-justify in field
    if ((int)result.length() < width) {
        result = std::string(width - result.length(), ' ') + result;
    }

    return result;
}

// Monadic format: ⍕ B
void fn_format_monadic(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    // Character input: return unchanged
    if (omega->is_string()) {
        m->result = omega;
        return;
    }
    if (omega->is_array() && omega->is_char_data()) {
        // Convert char vector back to string for consistency
        m->result = omega->to_string_value(m->heap);
        return;
    }

    // Empty array: return empty string
    if (omega->is_array() && omega->size() == 0) {
        m->result = m->heap->allocate_string("");
        return;
    }

    // Scalar
    if (omega->is_scalar()) {
        std::string formatted = format_number_pp(omega->as_scalar(), m->pp);
        m->result = m->heap->allocate_string(formatted.c_str());
        return;
    }

    // Vector or Matrix: format as space-separated values
    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (omega->is_vector()) {
        // Vector: single row, space-separated
        std::ostringstream oss;
        for (int i = 0; i < rows; i++) {
            if (i > 0) oss << " ";
            oss << format_number_pp((*mat)(i, 0), m->pp);
        }
        m->result = m->heap->allocate_string(oss.str().c_str());
        return;
    }

    // Matrix: newline-separated rows
    std::ostringstream oss;
    for (int i = 0; i < rows; i++) {
        if (i > 0) oss << "\n";
        for (int j = 0; j < cols; j++) {
            if (j > 0) oss << " ";
            oss << format_number_pp((*mat)(i, j), m->pp);
        }
    }
    m->result = m->heap->allocate_string(oss.str().c_str());
}

// Dyadic format: A ⍕ B
void fn_format_dyadic(Machine* m, Value* axis, Value* alpha, Value* omega) {
    REJECT_AXIS(m, axis);
    // A must be numeric
    if (alpha->is_string() || (alpha->is_array() && alpha->is_char_data())) {
        m->throw_error("DOMAIN ERROR: format left argument must be numeric");
        return;
    }

    // B must be numeric
    if (omega->is_string() || (omega->is_array() && omega->is_char_data())) {
        m->throw_error("DOMAIN ERROR: format right argument must be numeric");
        return;
    }

    // A must be a vector (rank <= 1)
    if (alpha->is_matrix()) {
        m->throw_error("RANK ERROR: format left argument must be a vector");
        return;
    }

    // Get format specifications from A
    std::vector<std::pair<int, int>> specs;  // (width, precision) pairs

    if (alpha->is_scalar()) {
        // Single scalar - interpret as width with 0 decimals
        int w = (int)std::round(alpha->as_scalar());
        specs.push_back({w, 0});
    } else {
        const Eigen::MatrixXd* a_mat = alpha->as_matrix();
        int a_size = a_mat->rows();

        if (a_size % 2 != 0) {
            m->throw_error("LENGTH ERROR: format left argument must have even length");
            return;
        }

        for (int i = 0; i < a_size; i += 2) {
            int w = (int)std::round((*a_mat)(i, 0));
            int p = (int)std::round((*a_mat)(i + 1, 0));
            specs.push_back({w, p});
        }
    }

    // Validate width is positive
    for (const auto& spec : specs) {
        if (spec.first <= 0) {
            m->throw_error("DOMAIN ERROR: format width must be positive");
            return;
        }
    }

    // Handle empty B
    if (omega->is_array() && omega->size() == 0) {
        int total_width = 0;
        for (const auto& spec : specs) {
            total_width += spec.first;
        }
        m->result = m->heap->allocate_string("");
        return;
    }

    // Format scalar B
    if (omega->is_scalar()) {
        double val = omega->as_scalar();
        int width = specs[0].first;
        int precision = specs[0].second;

        std::string formatted;
        if (precision >= 0) {
            formatted = format_number_fixed(val, width, precision);
        } else {
            formatted = format_number_exponential(val, width, -precision);
        }

        // Check if it fits
        // Note: We count UTF-8 characters, not bytes
        int char_count = 0;
        for (size_t i = 0; i < formatted.length(); ) {
            unsigned char c = formatted[i];
            if ((c & 0x80) == 0) { i += 1; }
            else if ((c & 0xE0) == 0xC0) { i += 2; }
            else if ((c & 0xF0) == 0xE0) { i += 3; }
            else { i += 4; }
            char_count++;
        }

        if (char_count > width) {
            m->throw_error("DOMAIN ERROR: format width too narrow");
            return;
        }

        m->result = m->heap->allocate_string(formatted.c_str());
        return;
    }

    // Format vector or matrix
    const Eigen::MatrixXd* mat = omega->as_matrix();

    // For vectors, treat as single row with N elements
    // For matrices, use actual rows/cols
    int rows, cols;
    if (omega->is_vector()) {
        rows = 1;
        cols = mat->rows();  // Vector elements become columns
    } else {
        rows = mat->rows();
        cols = mat->cols();
    }

    // If single spec, apply to all columns
    if (specs.size() == 1) {
        std::vector<std::pair<int, int>> expanded(cols, specs[0]);
        specs = expanded;
    }

    // Check we have right number of specs
    if ((int)specs.size() != cols) {
        m->throw_error("LENGTH ERROR: format specs must match number of columns");
        return;
    }

    std::ostringstream oss;

    for (int i = 0; i < rows; i++) {
        if (i > 0) oss << "\n";

        for (int j = 0; j < cols; j++) {
            double val = omega->is_vector() ? (*mat)(j, 0) : (*mat)(i, j);
            int width = specs[j].first;
            int precision = specs[j].second;

            std::string formatted;
            if (precision >= 0) {
                formatted = format_number_fixed(val, width, precision);
            } else {
                formatted = format_number_exponential(val, width, -precision);
            }

            oss << formatted;
        }
    }

    m->result = m->heap->allocate_string(oss.str().c_str());
}

} // namespace apl
