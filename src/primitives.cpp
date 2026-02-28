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
#include <chrono>
#include <thread>
#include <algorithm>

namespace apl {

// ============================================================================
// Axis Validation Helper (ISO 13751)
// ============================================================================

// Reject axis specification for functions that don't support it
#define REJECT_AXIS(m, axis) \
    if ((axis) != nullptr) { \
        (m)->throw_error("AXIS ERROR: function does not support axis", (m)->control, 4, 0); \
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

// Check if value is a near-integer (within INTEGER_TOLERANCE)
static inline bool is_near_integer(double x, double tol) {
    return std::abs(x - std::round(x)) < tol;
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
// Scalar (pervasive) functions - auto-penetrate nested arrays
PrimitiveFn prim_plus    = { "+", fn_conjugate, fn_add, true };
PrimitiveFn prim_minus   = { "-", fn_negate, fn_subtract, true };
PrimitiveFn prim_times   = { "×", fn_signum, fn_multiply, true };
PrimitiveFn prim_divide  = { "÷", fn_reciprocal, fn_divide, true };
PrimitiveFn prim_star    = { "*", fn_exponential, fn_power, true };
PrimitiveFn prim_equal   = { "=", nullptr, fn_equal, true };
PrimitiveFn prim_not_equal = { "≠", nullptr, fn_not_equal, true };
PrimitiveFn prim_less      = { "<", nullptr, fn_less, true };
PrimitiveFn prim_greater   = { ">", nullptr, fn_greater, true };
PrimitiveFn prim_less_eq   = { "≤", nullptr, fn_less_eq, true };
PrimitiveFn prim_greater_eq = { "≥", nullptr, fn_greater_eq, true };
PrimitiveFn prim_ceiling   = { "⌈", fn_ceiling, fn_maximum, true };
PrimitiveFn prim_floor     = { "⌊", fn_floor, fn_minimum, true };
PrimitiveFn prim_and       = { "∧", nullptr, fn_and, true };
PrimitiveFn prim_or        = { "∨", nullptr, fn_or, true };
PrimitiveFn prim_not       = { "~", fn_not, fn_without, true };
PrimitiveFn prim_nand      = { "⍲", nullptr, fn_nand, true };
PrimitiveFn prim_nor       = { "⍱", nullptr, fn_nor, true };
PrimitiveFn prim_stile     = { "|", fn_magnitude, fn_residue, true };
PrimitiveFn prim_log       = { "⍟", fn_natural_log, fn_logarithm, true };
PrimitiveFn prim_factorial = { "!", fn_factorial, fn_binomial, true };

// Array operation primitives - structural, not pervasive
PrimitiveFn prim_rho       = { "⍴", fn_shape, fn_reshape, false };
PrimitiveFn prim_comma     = { ",", fn_ravel, fn_catenate, false };
PrimitiveFn prim_transpose = { "⍉", fn_transpose, fn_dyadic_transpose, false };
PrimitiveFn prim_domino    = { "⌹", fn_matrix_inverse, fn_matrix_divide, false };
PrimitiveFn prim_iota      = { "⍳", fn_iota, fn_index_of, false };
PrimitiveFn prim_uptack    = { "↑", fn_first, fn_take, false };
PrimitiveFn prim_downtack  = { "↓", nullptr, fn_drop, false };
PrimitiveFn prim_reverse   = { "⌽", fn_reverse, fn_rotate, false };
PrimitiveFn prim_reverse_first = { "⊖", fn_reverse_first, fn_rotate_first, false };
PrimitiveFn prim_tally     = { "≢", fn_tally, nullptr, false };
PrimitiveFn prim_depth     = { "≡", fn_depth, fn_match, false };
PrimitiveFn prim_member    = { "∊", fn_enlist, fn_member_of, false };
PrimitiveFn prim_grade_up  = { "⍋", fn_grade_up, fn_grade_up_dyadic, false };
PrimitiveFn prim_grade_down = { "⍒", fn_grade_down, fn_grade_down_dyadic, false };
PrimitiveFn prim_union     = { "∪", fn_unique, fn_union, false };
PrimitiveFn prim_circle    = { "○", fn_pi_times, fn_circular, true };  // Circular is pervasive
PrimitiveFn prim_question  = { "?", fn_roll, fn_deal, true };  // Roll/deal is pervasive
PrimitiveFn prim_decode    = { "⊥", nullptr, fn_decode, false };
PrimitiveFn prim_encode    = { "⊤", nullptr, fn_encode, false };
PrimitiveFn prim_execute   = { "⍎", fn_execute, nullptr, false };
PrimitiveFn prim_format    = { "⍕", fn_format_monadic, fn_format_dyadic, false };
PrimitiveFn prim_table     = { "⍪", fn_table, fn_catenate_first, false };
PrimitiveFn prim_squad     = { "⌷", nullptr, fn_squad, false };
PrimitiveFn prim_left      = { "⊣", fn_right, fn_left, false };
PrimitiveFn prim_right     = { "⊢", fn_right, fn_right_dyadic, false };
PrimitiveFn prim_enclose   = { "⊂", fn_enclose, nullptr, false };
PrimitiveFn prim_disclose  = { "⊃", fn_disclose, fn_pick, false };

// Error handling system functions (ISO 13751 §11.5.7-11.6.5)
// Note: ⎕ET and ⎕EM are system variables, accessed via SysVarReadK
PrimitiveFn prim_quad_es   = { "⎕ES", fn_quad_es, fn_quad_es_dyadic, false };
PrimitiveFn prim_quad_ea   = { "⎕EA", nullptr, fn_quad_ea, false };

// Other system functions (ISO 13751 §11.5)
PrimitiveFn prim_quad_dl   = { "⎕DL", fn_quad_dl, nullptr, false };
PrimitiveFn prim_quad_nc   = { "⎕NC", fn_quad_nc, nullptr, false };
PrimitiveFn prim_quad_ex   = { "⎕EX", fn_quad_ex, nullptr, false };
PrimitiveFn prim_quad_nl   = { "⎕NL", fn_quad_nl, nullptr, false };

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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: + requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: + requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: + requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    // Shape checking
    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in addition", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: - requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: - requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: - requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in subtraction", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: × requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: × requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: × requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in multiplication", nullptr, 5, 0);
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
            m->throw_error("DOMAIN ERROR: division by zero", nullptr, 11, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ÷ requires numeric argument", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        double lval = lhs->data.scalar;
        for (int i = 0; i < rmat->size(); ++i) {
            if (!safe_divide(lval, rmat->data()[i], result(i))) {
                m->throw_error("DOMAIN ERROR: division by zero", nullptr, 11, 0);
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ÷ requires numeric argument", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        double rval = rhs->data.scalar;
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            if (!safe_divide(lmat->data()[i], rval, result(i))) {
                m->throw_error("DOMAIN ERROR: division by zero", nullptr, 11, 0);
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ÷ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in division", nullptr, 5, 0);
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        if (!safe_divide(lmat->data()[i], rmat->data()[i], result(i))) {
            m->throw_error("DOMAIN ERROR: division by zero", nullptr, 11, 0);
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
        m->throw_error("DOMAIN ERROR: 0 raised to negative power", nullptr, 11, 0);
        return 0.0;  // unreachable; throw_error longjmps
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: * requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: * requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: * requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in power", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: = requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: = requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: = requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in equality", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ≠ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ≠ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ≠ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in not-equal", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: < requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: < requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: < requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in less-than", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: > requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: > requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: > requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in greater-than", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ≤ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ≤ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ≤ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in less-or-equal", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ≥ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ≥ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ≥ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in greater-or-equal", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⌈ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⌈ requires numeric argument", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result = lmat->array().max(rhs->data.scalar);
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌈ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in maximum", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⌊ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⌊ requires numeric argument", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result = lmat->array().min(rhs->data.scalar);
        if (lhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌊ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in minimum", nullptr, 5, 0);
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

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌈ requires numeric argument", nullptr, 11, 0);
        return;
    }
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

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌊ requires numeric argument", nullptr, 11, 0);
        return;
    }
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ∧ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ∧ requires numeric argument", nullptr, 11, 0);
            return;
        }
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

    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ∧ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in and", nullptr, 5, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ∨ requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ∨ requires numeric argument", nullptr, 11, 0);
            return;
        }
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

    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ∨ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in or", nullptr, 5, 0);
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
            m->throw_error("DOMAIN ERROR: ~ requires boolean argument", nullptr, 11, 0);
            return;
        }
        // Round to nearest integer (0 or 1) then complement
        int b = (std::abs(d - 1.0) < std::abs(d)) ? 1 : 0;
        m->result = m->heap->allocate_scalar(b == 1 ? 0.0 : 1.0);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ~ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Check all elements are near-boolean
    for (int i = 0; i < mat->size(); ++i) {
        if (!is_near_boolean(mat->data()[i])) {
            m->throw_error("DOMAIN ERROR: ~ requires boolean argument", nullptr, 11, 0);
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
            m->throw_error("DOMAIN ERROR: ⍲ requires boolean arguments", nullptr, 11, 0);
            return;
        }
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (lhs->is_scalar()) {
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⍲ requires numeric argument", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = nand_bool(lhs->data.scalar, rmat->data()[i], error);
            if (error) {
                m->throw_error("DOMAIN ERROR: ⍲ requires boolean arguments", nullptr, 11, 0);
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⍲ requires numeric argument", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = nand_bool(lmat->data()[i], rhs->data.scalar, error);
            if (error) {
                m->throw_error("DOMAIN ERROR: ⍲ requires boolean arguments", nullptr, 11, 0);
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

    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍲ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in nand", nullptr, 5, 0);
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = nand_bool(lmat->data()[i], rmat->data()[i], error);
        if (error) {
            m->throw_error("DOMAIN ERROR: ⍲ requires boolean arguments", nullptr, 11, 0);
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
            m->throw_error("DOMAIN ERROR: ⍱ requires boolean arguments", nullptr, 11, 0);
            return;
        }
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (lhs->is_scalar()) {
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⍱ requires numeric argument", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = nor_bool(lhs->data.scalar, rmat->data()[i], error);
            if (error) {
                m->throw_error("DOMAIN ERROR: ⍱ requires boolean arguments", nullptr, 11, 0);
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⍱ requires numeric argument", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        Eigen::MatrixXd result(lmat->rows(), lmat->cols());
        for (int i = 0; i < lmat->size(); ++i) {
            result(i) = nor_bool(lmat->data()[i], rhs->data.scalar, error);
            if (error) {
                m->throw_error("DOMAIN ERROR: ⍱ requires boolean arguments", nullptr, 11, 0);
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

    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍱ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in nor", nullptr, 5, 0);
        return;
    }

    Eigen::MatrixXd result(lmat->rows(), lmat->cols());
    for (int i = 0; i < lmat->size(); ++i) {
        result(i) = nor_bool(lmat->data()[i], rmat->data()[i], error);
        if (error) {
            m->throw_error("DOMAIN ERROR: ⍱ requires boolean arguments", nullptr, 11, 0);
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

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: | requires numeric argument", nullptr, 11, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: | requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: | requires numeric argument", nullptr, 11, 0);
            return;
        }
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

    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: | requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in residue", nullptr, 5, 0);
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
            m->throw_error("DOMAIN ERROR: ⍟ of non-positive number", nullptr, 11, 0);
            return;
        }
        m->result = m->heap->allocate_scalar(std::log(val));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍟ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Check for non-positive values
    if ((mat->array() <= 0.0).any()) {
        m->throw_error("DOMAIN ERROR: logarithm of non-positive number", nullptr, 11, 0);
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
            m->throw_error("DOMAIN ERROR: invalid logarithm base", nullptr, 11, 0);
            return false;
        }
        if (val <= 0.0) {
            m->throw_error("DOMAIN ERROR: logarithm of non-positive number", nullptr, 11, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⍟ requires numeric argument", nullptr, 11, 0);
            return;
        }
        double base = lhs->data.scalar;
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        // Check base domain once (scalar extension)
        if (base <= 0.0 || base == 1.0) {
            m->throw_error("DOMAIN ERROR: invalid logarithm base", nullptr, 11, 0);
            return;
        }
        // Check all values are positive
        for (int i = 0; i < rmat->size(); ++i) {
            if (rmat->data()[i] <= 0.0) {
                m->throw_error("DOMAIN ERROR: logarithm of non-positive number", nullptr, 11, 0);
                return;
            }
        }
        double ln_base = std::log(base);
        Eigen::MatrixXd result = rmat->array().log() / ln_base;
        if (rhs->is_vector()) {
            m->result = m->heap->allocate_vector(result.col(0));
        } else {
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    if (rhs->is_scalar()) {
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⍟ requires numeric argument", nullptr, 11, 0);
            return;
        }
        double val = rhs->data.scalar;
        if (val <= 0.0) {
            m->throw_error("DOMAIN ERROR: logarithm of non-positive number", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* lmat = lhs->as_matrix();
        // Check all bases are valid
        for (int i = 0; i < lmat->size(); ++i) {
            if (lmat->data()[i] <= 0.0 || lmat->data()[i] == 1.0) {
                m->throw_error("DOMAIN ERROR: invalid logarithm base", nullptr, 11, 0);
                return;
            }
        }
        double ln_val = std::log(val);
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

    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍟ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in logarithm", nullptr, 5, 0);
        return;
    }

    // Validate all element pairs before computing
    for (int i = 0; i < lmat->size(); ++i) {
        if (!check_domain(lmat->data()[i], rmat->data()[i])) return;
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
            m->throw_error("DOMAIN ERROR: ! of negative integer", nullptr, 11, 0);
            return;
        }
        m->result = m->heap->allocate_scalar(std::tgamma(val + 1.0));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ! requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Check for negative integers
    for (int i = 0; i < mat->size(); ++i) {
        if (is_negative_int(mat->data()[i])) {
            m->throw_error("DOMAIN ERROR: factorial of negative integer", nullptr, 11, 0);
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
        m->throw_error("DOMAIN ERROR: invalid binomial arguments", nullptr, 11, 0);
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
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ! requires numeric argument", nullptr, 11, 0);
            return;
        }
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ! requires numeric argument", nullptr, 11, 0);
            return;
        }
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

    if (!lhs->is_array() || !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ! requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        m->throw_error("LENGTH ERROR: mismatched shapes in binomial", nullptr, 5, 0);
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

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ○ requires numeric argument", nullptr, 11, 0);
        return;
    }
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
        m->throw_error("RANK ERROR: circular function code must be scalar", nullptr, 4, 0);
        return;
    }

    double fn_val = lhs->data.scalar;
    int fn_code = static_cast<int>(std::round(fn_val));

    if (std::abs(fn_val - fn_code) > 1e-10) {
        m->throw_error("DOMAIN ERROR: circular function code must be integer", nullptr, 11, 0);
        return;
    }

    // ISO 13751: fn_code must be in [-12, 12]
    if (fn_code < -12 || fn_code > 12) {
        m->throw_error("DOMAIN ERROR: circular function code must be -12 to 12", nullptr, 11, 0);
        return;
    }

    // Apply to right argument
    if (rhs->is_scalar()) {
        double result = circular_function(fn_code, rhs->data.scalar);
        if (std::isnan(result)) {
            m->throw_error("DOMAIN ERROR: invalid argument for circular function", nullptr, 11, 0);
            return;
        }
        m->result = m->heap->allocate_scalar(result);
        return;
    }

    // String → char vector conversion for array operations
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ○ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* mat = rhs->as_matrix();
    Eigen::MatrixXd result(mat->rows(), mat->cols());

    for (int i = 0; i < mat->size(); ++i) {
        result(i) = circular_function(fn_code, mat->data()[i]);
        if (std::isnan(result(i))) {
            m->throw_error("DOMAIN ERROR: invalid argument for circular function", nullptr, 11, 0);
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
    } else if (omega->is_matrix()) {
        m->result = m->heap->allocate_matrix(*omega->as_matrix(), omega->is_char_data());
    } else {
        m->throw_error("DOMAIN ERROR: + requires numeric argument", nullptr, 11, 0);
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

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: - requires numeric argument", nullptr, 11, 0);
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

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: × requires numeric argument", nullptr, 11, 0);
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
void fn_reciprocal(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        if (omega->data.scalar == 0.0) {
            m->throw_error("DOMAIN ERROR: reciprocal of zero", nullptr, 11, 0);
            return;
        }
        m->result = m->heap->allocate_scalar(1.0 / omega->data.scalar);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ÷ requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Check for zeros
    if ((mat->array() == 0.0).any()) {
        m->throw_error("DOMAIN ERROR: reciprocal of zero", nullptr, 11, 0);
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

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: * requires numeric argument", nullptr, 11, 0);
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
void fn_shape(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // Scalar has empty shape
        Eigen::VectorXd shape(0);
        m->result = m->heap->allocate_vector(shape);
        return;
    }

    // Strand shape is its element count (rank 1)
    if (omega->is_strand()) {
        Eigen::VectorXd shape(1);
        shape(0) = static_cast<double>(omega->as_strand()->size());
        m->result = m->heap->allocate_vector(shape);
        return;
    }

    // NDARRAY shape is its shape vector
    if (omega->is_ndarray()) {
        const auto& nd_shape = omega->ndarray_shape();
        Eigen::VectorXd shape(nd_shape.size());
        for (size_t i = 0; i < nd_shape.size(); ++i) {
            shape(i) = static_cast<double>(nd_shape[i]);
        }
        m->result = m->heap->allocate_vector(shape);
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍴ requires array argument", nullptr, 11, 0);
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
void fn_reshape(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    // lhs must be a scalar or vector specifying new shape
    if (!lhs->is_scalar() && !lhs->is_vector()) {
        m->throw_error("RANK ERROR: left argument to reshape must be scalar or vector", nullptr, 4, 0);
        return;
    }

    // Get target shape as a vector of ints
    std::vector<int> target_shape;

    if (lhs->is_scalar()) {
        // Scalar shape means 1D vector of that length
        double dim = lhs->as_scalar();
        if (dim < 0.0) {
            m->throw_error("DOMAIN ERROR: reshape dimension must be non-negative", nullptr, 11, 0);
            return;
        }
        if (dim != std::floor(dim)) {
            m->throw_error("DOMAIN ERROR: reshape dimension must be an integer", nullptr, 11, 0);
            return;
        }
        target_shape.push_back(static_cast<int>(dim));
    } else {
        const Eigen::MatrixXd* shape_mat = lhs->as_matrix();
        int shape_len = shape_mat->rows();

        if (shape_len == 0) {
            // ISO 13751 Section 8.3.1: Empty shape produces scalar
            // (⍳0)⍴5 → 5 (scalar)
            double scalar_val;
            if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);
            if (rhs->is_scalar()) {
                scalar_val = rhs->as_scalar();
            } else if (rhs->size() > 0) {
                if (rhs->is_ndarray()) {
                    scalar_val = (*rhs->ndarray_data())(0);
                } else {
                    const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();
                    scalar_val = (*rhs_mat)(0, 0);
                }
            } else {
                m->throw_error("DOMAIN ERROR: cannot reshape empty array to scalar", nullptr, 11, 0);
                return;
            }
            m->result = m->heap->allocate_scalar(scalar_val);
            return;
        }

        // Validate and collect all dimensions
        for (int i = 0; i < shape_len; ++i) {
            double dim = (*shape_mat)(i, 0);
            if (dim < 0.0) {
                m->throw_error("DOMAIN ERROR: reshape dimension must be non-negative", nullptr, 11, 0);
                return;
            }
            if (dim != std::floor(dim)) {
                m->throw_error("DOMAIN ERROR: reshape dimension must be an integer", nullptr, 11, 0);
                return;
            }
            target_shape.push_back(static_cast<int>(dim));
        }
    }

    // Compute total target size
    int target_size = 1;
    for (int dim : target_shape) {
        target_size *= dim;
    }

    // Check implementation limit (ISO 13751 §A.3)
    if (static_cast<size_t>(target_size) > MAX_ARRAY_SIZE) {
        m->throw_error("LIMIT ERROR: array size exceeds implementation limit", m->control, 10, 0);
        return;
    }

    // String → char vector conversion for array operations
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Get source data (row-major order per APL)
    Eigen::VectorXd source;
    if (rhs->is_scalar()) {
        source.resize(1);
        source(0) = rhs->as_scalar();
    } else if (rhs->is_ndarray()) {
        // NDARRAY is already flat in row-major order
        source = *rhs->ndarray_data();
    } else {
        if (!rhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⍴ requires array argument", nullptr, 11, 0);
            return;
        }
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
        m->throw_error("DOMAIN ERROR: cannot reshape empty array to non-empty shape", nullptr, 11, 0);
        return;
    }

    // Preserve character data flag from source
    bool is_char = rhs->is_char_data();

    // Build result by cycling through source data (row-major order per APL)
    int rank = static_cast<int>(target_shape.size());

    if (rank == 1) {
        // Vector result
        Eigen::VectorXd result(target_shape[0]);
        for (int i = 0; i < target_size; ++i) {
            result(i) = source(i % source.size());
        }
        m->result = m->heap->allocate_vector(result, is_char);
    } else if (rank == 2) {
        // Matrix result
        int rows = target_shape[0];
        int cols = target_shape[1];
        Eigen::MatrixXd result(rows, cols);
        for (int i = 0; i < target_size; ++i) {
            result(i / cols, i % cols) = source(i % source.size());
        }
        m->result = m->heap->allocate_matrix(result, is_char);
    } else {
        // NDARRAY result (rank 3+)
        Eigen::VectorXd result(target_size);
        for (int i = 0; i < target_size; ++i) {
            result(i) = source(i % source.size());
        }
        m->result = m->heap->allocate_ndarray(std::move(result), std::move(target_shape));
        m->result->set_char_data(is_char);
    }
}

// Ravel (,) - monadic: flatten to vector
// ISO 13751 §8.2.1: Z is a vector containing the elements of B in row-major order
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

    // NDARRAY: data is already stored flat in row-major order
    if (omega->is_ndarray()) {
        const Eigen::VectorXd* data = omega->ndarray_data();
        m->result = m->heap->allocate_vector(*data, omega->is_char_data());
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: , requires array argument", nullptr, 11, 0);
        return;
    }
    bool is_char = omega->is_char_data();
    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Flatten in row-major order (APL standard)
    int size = mat->size();
    Eigen::VectorXd result(size);
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
    // Handle strands (nested arrays) - catenate element-wise
    if (lhs->is_strand() || rhs->is_strand()) {
        if (axis != nullptr) {
            m->throw_error("AXIS ERROR: cannot catenate strands with axis", m->control, 4, 0);
            return;
        }
        std::vector<Value*> result;

        // Add elements from lhs
        if (lhs->is_strand()) {
            std::vector<Value*>* left_strand = lhs->as_strand();
            result.insert(result.end(), left_strand->begin(), left_strand->end());
        } else {
            result.push_back(lhs);
        }

        // Add elements from rhs
        if (rhs->is_strand()) {
            std::vector<Value*>* right_strand = rhs->as_strand();
            result.insert(result.end(), right_strand->begin(), right_strand->end());
        } else {
            result.push_back(rhs);
        }

        m->result = m->heap->allocate_strand(result);
        return;
    }

    // String → char vector conversion for array operations
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Preserve char data if both operands are char data
    bool is_char = lhs->is_char_data() && rhs->is_char_data();

    // Handle scalar cases by promoting to 1-element vector
    if (lhs->is_scalar() && rhs->is_scalar()) {
        if (axis != nullptr) {
            m->throw_error("AXIS ERROR: cannot catenate scalars with axis", m->control, 4, 0);
            return;
        }
        Eigen::VectorXd result(2);
        result(0) = lhs->as_scalar();
        result(1) = rhs->as_scalar();
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    // Validate both arguments are arrays (or scalars which were handled above)
    if (!lhs->is_scalar() && !lhs->is_array()) {
        m->throw_error("DOMAIN ERROR: , requires array argument", nullptr, 11, 0);
        return;
    }
    if (!rhs->is_scalar() && !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: , requires array argument", nullptr, 11, 0);
        return;
    }

    // Handle NDARRAY catenation
    if (lhs->is_ndarray() || rhs->is_ndarray()) {
        // Determine axis (default: last axis)
        int cat_axis = -1;  // Will be set to last axis
        bool is_laminate = false;

        if (axis != nullptr) {
            if (!axis->is_scalar()) {
                m->throw_error("RANK ERROR: axis must be scalar", nullptr, 4, 0);
                return;
            }
            double k_val = axis->as_scalar();
            double rounded = std::round(k_val);
            is_laminate = std::abs(k_val - rounded) > 1e-10;
            cat_axis = static_cast<int>(std::floor(k_val)) - m->io;
        }

        if (is_laminate) {
            m->throw_error("RANK ERROR: laminate of NDARRAY not yet supported", nullptr, 4, 0);
            return;
        }

        // Get shapes of both operands
        std::vector<int> lhs_shape, rhs_shape;
        const Eigen::VectorXd* lhs_data = nullptr;
        const Eigen::VectorXd* rhs_data = nullptr;

        if (lhs->is_ndarray()) {
            lhs_shape = lhs->ndarray_shape();
            lhs_data = lhs->ndarray_data();
        } else if (lhs->is_scalar()) {
            lhs_shape = {};
        } else {
            // VECTOR or MATRIX
            lhs_shape.push_back(lhs->as_matrix()->rows());
            if (!lhs->is_vector()) {
                lhs_shape.push_back(lhs->as_matrix()->cols());
            }
        }

        if (rhs->is_ndarray()) {
            rhs_shape = rhs->ndarray_shape();
            rhs_data = rhs->ndarray_data();
        } else if (rhs->is_scalar()) {
            rhs_shape = {};
        } else {
            // VECTOR or MATRIX
            rhs_shape.push_back(rhs->as_matrix()->rows());
            if (!rhs->is_vector()) {
                rhs_shape.push_back(rhs->as_matrix()->cols());
            }
        }

        // Handle scalar extension
        int target_rank = std::max(static_cast<int>(lhs_shape.size()),
                                   static_cast<int>(rhs_shape.size()));
        if (target_rank == 0) {
            m->throw_error("RANK ERROR: cannot catenate scalars along axis", nullptr, 4, 0);
            return;
        }

        // Default axis is last axis
        if (cat_axis < 0) cat_axis = target_rank - 1;
        if (cat_axis >= target_rank) {
            m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
            return;
        }

        // Promote shapes to same rank by prepending 1s
        while (static_cast<int>(lhs_shape.size()) < target_rank) {
            lhs_shape.insert(lhs_shape.begin(), 1);
        }
        while (static_cast<int>(rhs_shape.size()) < target_rank) {
            rhs_shape.insert(rhs_shape.begin(), 1);
        }

        // Verify all axes except cat_axis match
        for (int i = 0; i < target_rank; ++i) {
            if (i != cat_axis && lhs_shape[i] != rhs_shape[i]) {
                m->throw_error("LENGTH ERROR: incompatible shapes for catenation", nullptr, 5, 0);
                return;
            }
        }

        // Compute result shape
        std::vector<int> result_shape = lhs_shape;
        result_shape[cat_axis] = lhs_shape[cat_axis] + rhs_shape[cat_axis];

        // Compute strides for lhs, rhs, and result
        auto compute_strides = [](const std::vector<int>& shape) {
            std::vector<int> strides(shape.size());
            int stride = 1;
            for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        };

        std::vector<int> lhs_strides = compute_strides(lhs_shape);
        std::vector<int> rhs_strides = compute_strides(rhs_shape);
        std::vector<int> result_strides = compute_strides(result_shape);

        int result_size = 1;
        for (int d : result_shape) result_size *= d;

        Eigen::VectorXd result(result_size);

        // Helper to get value from source (lhs or rhs)
        auto get_lhs_value = [&](const std::vector<int>& indices) -> double {
            if (lhs->is_scalar()) return lhs->as_scalar();
            if (lhs->is_ndarray()) {
                int lin = 0;
                for (int i = 0; i < target_rank; ++i) {
                    lin += indices[i] * lhs_strides[i];
                }
                return (*lhs_data)(lin);
            }
            // VECTOR or MATRIX
            const Eigen::MatrixXd* mat = lhs->as_matrix();
            if (lhs->is_vector()) {
                return (*mat)(indices[target_rank - 1], 0);
            }
            return (*mat)(indices[target_rank - 2], indices[target_rank - 1]);
        };

        auto get_rhs_value = [&](const std::vector<int>& indices) -> double {
            if (rhs->is_scalar()) return rhs->as_scalar();
            if (rhs->is_ndarray()) {
                int lin = 0;
                for (int i = 0; i < target_rank; ++i) {
                    lin += indices[i] * rhs_strides[i];
                }
                return (*rhs_data)(lin);
            }
            // VECTOR or MATRIX
            const Eigen::MatrixXd* mat = rhs->as_matrix();
            if (rhs->is_vector()) {
                return (*mat)(indices[target_rank - 1], 0);
            }
            return (*mat)(indices[target_rank - 2], indices[target_rank - 1]);
        };

        // Fill result
        std::vector<int> result_indices(target_rank);
        for (int linear = 0; linear < result_size; ++linear) {
            // Decompose linear index
            int remaining = linear;
            for (int d = 0; d < target_rank; ++d) {
                result_indices[d] = remaining / result_strides[d];
                remaining %= result_strides[d];
            }

            // Determine if this comes from lhs or rhs
            if (result_indices[cat_axis] < lhs_shape[cat_axis]) {
                result(linear) = get_lhs_value(result_indices);
            } else {
                // Adjust index for rhs
                std::vector<int> rhs_indices = result_indices;
                rhs_indices[cat_axis] -= lhs_shape[cat_axis];
                result(linear) = get_rhs_value(rhs_indices);
            }
        }

        // Allocate result based on rank
        if (target_rank <= 2) {
            if (target_rank == 1) {
                m->result = m->heap->allocate_vector(result, is_char);
            } else {
                Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
                for (int i = 0; i < result_shape[0]; ++i) {
                    for (int j = 0; j < result_shape[1]; ++j) {
                        mat(i, j) = result(i * result_shape[1] + j);
                    }
                }
                m->result = m->heap->allocate_matrix(mat, is_char);
            }
        } else {
            m->result = m->heap->allocate_ndarray(std::move(result), std::move(result_shape));
            m->result->set_char_data(is_char);
        }
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
            m->throw_error("RANK ERROR: axis must be scalar", nullptr, 4, 0);
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
            m->throw_error("LENGTH ERROR: incompatible shapes for laminate", nullptr, 5, 0);
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
            m->throw_error("RANK ERROR: laminate of matrices not supported", nullptr, 4, 0);
        }
        return;
    }

    // Catenate along existing axis
    if (both_vectors) {
        // Vectors only have axis 1
        if (axis != nullptr) {
            int cat_axis = static_cast<int>(std::round(k_val));
            if (cat_axis != 1) {
                m->throw_error("AXIS ERROR: vectors only have axis 1", m->control, 4, 0);
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
            m->throw_error("AXIS ERROR: axis must be 1 or 2 for matrix", m->control, 4, 0);
            return;
        }
    }

    if (cat_axis == 1) {
        // Catenate along first axis (rows)
        if (lmat->cols() != rmat->cols()) {
            m->throw_error("LENGTH ERROR: incompatible shapes for catenation", nullptr, 5, 0);
            return;
        }

        Eigen::MatrixXd result(lmat->rows() + rmat->rows(), lmat->cols());
        result << *lmat, *rmat;
        m->result = m->heap->allocate_matrix(result, is_char);
    } else {
        // Catenate along second axis (columns)
        if (lmat->rows() != rmat->rows()) {
            m->throw_error("LENGTH ERROR: incompatible shapes for catenation", nullptr, 5, 0);
            return;
        }

        Eigen::MatrixXd result(lmat->rows(), lmat->cols() + rmat->cols());
        result << *lmat, *rmat;
        m->result = m->heap->allocate_matrix(result, is_char);
    }
}

// Transpose (⍉) - monadic: reverse dimensions
// ISO 13751 §10.1.5: Z is B with the order of the axes reversed
void fn_transpose(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // Scalar transpose is identity
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    // NDARRAY: reverse axes order
    // Shape {2,3,4} → {4,3,2}
    if (omega->is_ndarray()) {
        const Value::NDArrayData* nd = omega->as_ndarray();
        const auto& old_shape = nd->shape;
        const auto& old_strides = nd->strides;
        int rank = static_cast<int>(old_shape.size());

        // New shape is reversed old shape
        std::vector<int> new_shape(rank);
        for (int i = 0; i < rank; ++i) {
            new_shape[i] = old_shape[rank - 1 - i];
        }

        // Compute new strides
        std::vector<int> new_strides(rank);
        int stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
            new_strides[i] = stride;
            stride *= new_shape[i];
        }

        int total_size = nd->data->size();
        Eigen::VectorXd result(total_size);

        // For each position in result, find corresponding source position
        std::vector<int> new_indices(rank);
        for (int linear = 0; linear < total_size; ++linear) {
            // Decompose linear index into new multi-index
            int remaining = linear;
            for (int d = 0; d < rank; ++d) {
                new_indices[d] = remaining / new_strides[d];
                remaining %= new_strides[d];
            }

            // Old multi-index is reversed new multi-index
            // old[i,j,k] corresponds to new[k,j,i]
            int old_linear = 0;
            for (int d = 0; d < rank; ++d) {
                old_linear += new_indices[d] * old_strides[rank - 1 - d];
            }

            result(linear) = (*nd->data)(old_linear);
        }

        m->result = m->heap->allocate_ndarray(std::move(result), std::move(new_shape));
        m->result->set_char_data(omega->is_char_data());
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍉ requires array argument", nullptr, 11, 0);
        return;
    }

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
// ISO 13751 §10.2.10: Each element of A corresponds to an axis of B by position
// and to an axis of Z by value. Repeated elements select diagonals.
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
        if (!lhs->is_array()) {
            m->throw_error("DOMAIN ERROR: ⍉ requires array argument", nullptr, 11, 0);
            return;
        }
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

    int io = m->io;
    bool is_char = rhs->is_char_data();

    // NDARRAY: general axis permutation
    if (rhs->is_ndarray()) {
        const Value::NDArrayData* nd = rhs->as_ndarray();
        const auto& old_shape = nd->shape;
        const auto& old_strides = nd->strides;
        int rank = static_cast<int>(old_shape.size());

        if (static_cast<int>(perm.size()) != rank) {
            m->throw_error("LENGTH ERROR: permutation must match array rank", nullptr, 5, 0);
            return;
        }

        // Convert permutation to 0-indexed integers and validate
        std::vector<int> perm_int(rank);
        int max_perm = -1;
        for (int i = 0; i < rank; ++i) {
            if (!is_near_integer(perm(i), INTEGER_TOLERANCE)) {
                m->throw_error("DOMAIN ERROR: permutation values must be integers", nullptr, 11, 0);
                return;
            }
            perm_int[i] = static_cast<int>(std::round(perm(i))) - io;
            if (perm_int[i] < 0) {
                m->throw_error("DOMAIN ERROR: permutation values out of range", nullptr, 11, 0);
                return;
            }
            if (perm_int[i] > max_perm) max_perm = perm_int[i];
        }

        // ISO 13751: All axes 0..max_perm must be present in perm
        int result_rank = max_perm + 1;
        std::vector<bool> axis_present(result_rank, false);
        for (int i = 0; i < rank; ++i) {
            axis_present[perm_int[i]] = true;
        }
        for (int i = 0; i < result_rank; ++i) {
            if (!axis_present[i]) {
                m->throw_error("DOMAIN ERROR: axis missing from permutation", nullptr, 11, 0);
                return;
            }
        }

        // Compute result shape
        // For repeated perm values, take min of corresponding source axis lengths
        std::vector<int> new_shape(result_rank, INT_MAX);
        for (int i = 0; i < rank; ++i) {
            int target = perm_int[i];
            new_shape[target] = std::min(new_shape[target], old_shape[i]);
        }

        // Compute new strides
        std::vector<int> new_strides(result_rank);
        int stride = 1;
        for (int i = result_rank - 1; i >= 0; --i) {
            new_strides[i] = stride;
            stride *= new_shape[i];
        }
        int total_size = stride;

        Eigen::VectorXd result(total_size);

        // For each position in result, find corresponding source position
        std::vector<int> new_indices(result_rank);
        for (int linear = 0; linear < total_size; ++linear) {
            // Decompose linear index into new multi-index
            int remaining = linear;
            for (int d = 0; d < result_rank; ++d) {
                new_indices[d] = remaining / new_strides[d];
                remaining %= new_strides[d];
            }

            // Source index: src_idx[i] = new_indices[perm_int[i]]
            int old_linear = 0;
            for (int i = 0; i < rank; ++i) {
                old_linear += new_indices[perm_int[i]] * old_strides[i];
            }

            result(linear) = (*nd->data)(old_linear);
        }

        if (result_rank <= 2) {
            // Result is matrix or lower rank
            if (result_rank == 0 || total_size == 0) {
                m->result = m->heap->allocate_scalar(0.0);
            } else if (result_rank == 1) {
                m->result = m->heap->allocate_vector(result, is_char);
            } else {
                Eigen::MatrixXd mat(new_shape[0], new_shape[1]);
                for (int i = 0; i < new_shape[0]; ++i) {
                    for (int j = 0; j < new_shape[1]; ++j) {
                        mat(i, j) = result(i * new_shape[1] + j);
                    }
                }
                m->result = m->heap->allocate_matrix(mat, is_char);
            }
        } else {
            m->result = m->heap->allocate_ndarray(std::move(result), std::move(new_shape));
            m->result->set_char_data(is_char);
        }
        return;
    }

    if (rhs->is_vector()) {
        // Vector (rank 1) - only valid permutation is identity (single axis)
        // Check for io-adjusted permutation: with ⎕IO=1, valid value is 1
        if (perm.size() != 1) {
            m->throw_error("LENGTH ERROR: permutation must match array rank", nullptr, 5, 0);
            return;
        }
        int p0 = static_cast<int>(std::round(perm(0))) - io;
        if (p0 != 0) {
            m->throw_error("DOMAIN ERROR: permutation values out of range", nullptr, 11, 0);
            return;
        }
        m->result = m->heap->allocate_vector(rhs->as_matrix()->col(0), is_char);
        return;
    }

    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍉ requires array argument", nullptr, 11, 0);
        return;
    }

    // Matrix: permutation must be length 2
    if (perm.size() != 2) {
        m->throw_error("LENGTH ERROR: permutation must match array rank", nullptr, 5, 0);
        return;
    }

    // ISO 13751 10.2.10: Permutation values must be near-integers
    if (!is_near_integer(perm(0), INTEGER_TOLERANCE) ||
        !is_near_integer(perm(1), INTEGER_TOLERANCE)) {
        m->throw_error("DOMAIN ERROR: permutation values must be integers", nullptr, 11, 0);
        return;
    }

    // Convert to 0-indexed
    int p0 = static_cast<int>(std::round(perm(0))) - io;
    int p1 = static_cast<int>(std::round(perm(1))) - io;

    // Validate permutation values
    if (p0 < 0 || p0 > 1 || p1 < 0 || p1 > 1) {
        m->throw_error("DOMAIN ERROR: permutation values out of range", nullptr, 11, 0);
        return;
    }

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (p0 == 0 && p1 == 1) {
        // Identity: 1 2⍉M (with ⎕IO=1)
        m->result = m->heap->allocate_matrix(*mat, is_char);
    } else if (p0 == 1 && p1 == 0) {
        // Transpose: 2 1⍉M (with ⎕IO=1)
        m->result = m->heap->allocate_matrix(mat->transpose(), is_char);
    } else if (p0 == p1) {
        // ISO 13751 10.2.10: Diagonal selection (1 1⍉M or 2 2⍉M with ⎕IO=1)
        // Result length is min of the two axis lengths
        int diag_len = std::min(static_cast<int>(mat->rows()),
                                static_cast<int>(mat->cols()));
        Eigen::VectorXd diag(diag_len);
        for (int i = 0; i < diag_len; ++i) {
            diag(i) = (*mat)(i, i);
        }
        m->result = m->heap->allocate_vector(diag, is_char);
    } else {
        m->throw_error("AXIS ERROR: invalid axis permutation", m->control, 4, 0);
    }
}

// Matrix Inverse (⌹) - monadic
void fn_matrix_inverse(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        double val = omega->as_scalar();
        if (val == 0.0) {
            m->throw_error("DOMAIN ERROR: cannot invert zero", nullptr, 11, 0);
            return;
        }
        m->result = m->heap->allocate_scalar(1.0 / val);
        return;
    }

    // NDARRAY (rank 3+) not supported for matrix inverse
    if (omega->is_ndarray()) {
        m->throw_error("RANK ERROR: matrix inverse requires rank ≤ 2", nullptr, 4, 0);
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌹ requires array argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Handle empty matrix: inverse of 0×0 is 0×0
    if (mat->rows() == 0 || mat->cols() == 0) {
        m->result = m->heap->allocate_matrix(Eigen::MatrixXd(mat->cols(), mat->rows()));
        return;
    }

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

    // NDARRAY (rank 3+) not supported for matrix divide
    if (lhs->is_ndarray()) {
        m->throw_error("RANK ERROR: matrix divide requires rank ≤ 2", nullptr, 4, 0);
        return;
    }
    if (rhs->is_ndarray()) {
        m->throw_error("RANK ERROR: matrix divide requires rank ≤ 2", nullptr, 4, 0);
        return;
    }

    if (rhs->is_scalar()) {
        double divisor = rhs->as_scalar();
        if (divisor == 0.0) {
            m->throw_error("DOMAIN ERROR: division by zero", nullptr, 11, 0);
            return;
        }
        if (lhs->is_scalar()) {
            m->result = m->heap->allocate_scalar(lhs->as_scalar() / divisor);
        } else {
            if (!lhs->is_array()) {
                m->throw_error("DOMAIN ERROR: ⌹ requires array argument", nullptr, 11, 0);
                return;
            }
            Eigen::MatrixXd result = lhs->as_matrix()->array() / divisor;
            if (lhs->is_vector()) {
                m->result = m->heap->allocate_vector(result.col(0));
            } else {
                m->result = m->heap->allocate_matrix(result);
            }
        }
        return;
    }

    // Validate arguments
    if (!lhs->is_scalar() && !lhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌹ requires array argument", nullptr, 11, 0);
        return;
    }
    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌹ requires array argument", nullptr, 11, 0);
        return;
    }

    // Solve B×X = A using least squares
    Eigen::MatrixXd A = lhs->is_scalar()
        ? Eigen::MatrixXd::Constant(1, 1, lhs->as_scalar())
        : *lhs->as_matrix();
    const Eigen::MatrixXd& B = *rhs->as_matrix();

    // Check dimensions: B's rows must match A's rows
    if (B.rows() != A.rows()) {
        m->throw_error("LENGTH ERROR: incompatible shapes for matrix divide", nullptr, 5, 0);
        return;
    }

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

    // Scalar argument: simple index generator
    if (omega->is_scalar()) {
        double val = omega->as_scalar();
        if (val < 0.0) {
            m->throw_error("DOMAIN ERROR: iota argument must be non-negative", nullptr, 11, 0);
            return;
        }
        if (val != std::floor(val)) {
            m->throw_error("DOMAIN ERROR: iota argument must be an integer", nullptr, 11, 0);
            return;
        }
        int n = static_cast<int>(val);
        if (static_cast<size_t>(n) > MAX_ARRAY_SIZE) {
            m->throw_error("LIMIT ERROR: array size exceeds implementation limit", m->control, 10, 0);
            return;
        }
        Eigen::VectorXd result(n);
        for (int i = 0; i < n; ++i) {
            result(i) = i + m->io;
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // Vector argument: multi-dimensional index generator (ISO 13751 §10.1.2)
    // ⍳2 3 → 2×3 array of index pairs (strands)
    if (omega->is_vector()) {
        const Eigen::MatrixXd* shape_vec = omega->as_matrix();
        int rank = shape_vec->rows();

        // Validate all elements are non-negative integers
        std::vector<int> shape(rank);
        int total = 1;
        for (int i = 0; i < rank; ++i) {
            double val = (*shape_vec)(i, 0);
            if (val < 0.0) {
                m->throw_error("DOMAIN ERROR: iota argument must be non-negative", nullptr, 11, 0);
                return;
            }
            if (val != std::floor(val)) {
                m->throw_error("DOMAIN ERROR: iota argument must be an integer", nullptr, 11, 0);
                return;
            }
            shape[i] = static_cast<int>(val);
            total *= shape[i];
        }

        if (static_cast<size_t>(total) > MAX_ARRAY_SIZE) {
            m->throw_error("LIMIT ERROR: array size exceeds implementation limit", m->control, 10, 0);
            return;
        }

        // Compute strides for index calculation
        std::vector<int> strides(rank);
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // Build result: strand of strands (each element is index tuple)
        std::vector<Value*> result;
        result.reserve(total);

        std::vector<int> idx(rank);
        for (int i = 0; i < total; ++i) {
            // Convert linear index to multi-dimensional
            int tmp = i;
            for (int d = 0; d < rank; ++d) {
                idx[d] = tmp / strides[d];
                tmp %= strides[d];
            }

            // Create strand of indices (with ⎕IO offset)
            std::vector<Value*> index_tuple;
            index_tuple.reserve(rank);
            for (int d = 0; d < rank; ++d) {
                index_tuple.push_back(m->heap->allocate_scalar(idx[d] + m->io));
            }
            result.push_back(m->heap->allocate_strand(std::move(index_tuple)));
        }

        // Reshape to proper dimensions
        if (rank == 2) {
            // 2D: return as matrix of strands (but we use strand for now)
            // Actually need proper nested array structure
            m->result = m->heap->allocate_strand(std::move(result));
        } else {
            m->result = m->heap->allocate_strand(std::move(result));
        }
        return;
    }

    // Strand argument: also valid per ISO
    if (omega->is_strand()) {
        const std::vector<Value*>* strand = omega->as_strand();
        int rank = static_cast<int>(strand->size());

        std::vector<int> shape(rank);
        int total = 1;
        for (int i = 0; i < rank; ++i) {
            Value* elem = (*strand)[i];
            if (!elem->is_scalar()) {
                m->throw_error("DOMAIN ERROR: iota shape elements must be scalar", nullptr, 11, 0);
                return;
            }
            double val = elem->as_scalar();
            if (val < 0.0) {
                m->throw_error("DOMAIN ERROR: iota argument must be non-negative", nullptr, 11, 0);
                return;
            }
            if (val != std::floor(val)) {
                m->throw_error("DOMAIN ERROR: iota argument must be an integer", nullptr, 11, 0);
                return;
            }
            shape[i] = static_cast<int>(val);
            total *= shape[i];
        }

        if (static_cast<size_t>(total) > MAX_ARRAY_SIZE) {
            m->throw_error("LIMIT ERROR: array size exceeds implementation limit", m->control, 10, 0);
            return;
        }

        std::vector<int> strides(rank);
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        std::vector<Value*> result;
        result.reserve(total);

        std::vector<int> idx(rank);
        for (int i = 0; i < total; ++i) {
            int tmp = i;
            for (int d = 0; d < rank; ++d) {
                idx[d] = tmp / strides[d];
                tmp %= strides[d];
            }

            std::vector<Value*> index_tuple;
            index_tuple.reserve(rank);
            for (int d = 0; d < rank; ++d) {
                index_tuple.push_back(m->heap->allocate_scalar(idx[d] + m->io));
            }
            result.push_back(m->heap->allocate_strand(std::move(index_tuple)));
        }

        m->result = m->heap->allocate_strand(std::move(result));
        return;
    }

    m->throw_error("RANK ERROR: iota argument must be scalar or vector", nullptr, 4, 0);
}

// First (↑) - monadic: return first element
void fn_first(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    if (omega->is_scalar()) {
        // First of scalar is the scalar itself
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // First of strand is the first element (ISO 10.1.9: first item in row-major order)
    if (omega->is_strand()) {
        const std::vector<Value*>* strand = omega->as_strand();
        if (strand->empty()) {
            // Empty strand: return typical element (0 for numeric)
            m->result = m->heap->allocate_scalar(0.0);
            return;
        }
        m->result = (*strand)[0];
        return;
    }

    // NDARRAY: first major cell (subarray along first axis)
    if (omega->is_ndarray()) {
        const Value::NDArrayData* nd = omega->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        if (nd->data->size() == 0) {
            m->result = m->heap->allocate_scalar(0.0);
            return;
        }

        if (rank == 3) {
            // First of 3D array is a matrix (first plane)
            int rows = shape[1];
            int cols = shape[2];
            Eigen::MatrixXd result(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result(i, j) = (*nd->data)(i * cols + j);
                }
            }
            m->result = m->heap->allocate_matrix(result);
        } else {
            // For higher ranks, return NDARRAY with one less dimension
            std::vector<int> new_shape(shape.begin() + 1, shape.end());
            int cell_size = 1;
            for (size_t i = 1; i < shape.size(); ++i) {
                cell_size *= shape[i];
            }
            Eigen::VectorXd result_data(cell_size);
            for (int i = 0; i < cell_size; ++i) {
                result_data(i) = (*nd->data)(i);
            }
            m->result = m->heap->allocate_ndarray(result_data, new_shape);
        }
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);
    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ↑ requires array argument", nullptr, 11, 0);
        return;
    }
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
    if (rhs->is_scalar()) {
        if (!lhs->is_scalar()) {
            m->throw_error("RANK ERROR: take from scalar requires scalar count", nullptr, 4, 0);
            return;
        }
        int n = static_cast<int>(lhs->as_scalar());
        Eigen::VectorXd result(std::abs(n));
        result.setConstant(rhs->as_scalar());
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // Strand take: take first/last n elements
    if (rhs->is_strand()) {
        if (!lhs->is_scalar()) {
            m->throw_error("RANK ERROR: strand take requires scalar count", nullptr, 4, 0);
            return;
        }
        int n = static_cast<int>(lhs->as_scalar());
        const std::vector<Value*>* strand = rhs->as_strand();
        int len = static_cast<int>(strand->size());
        int abs_n = std::abs(n);

        std::vector<Value*> result;
        result.reserve(abs_n);

        // Fill element for over-take is scalar 0
        Value* fill = m->heap->allocate_scalar(0.0);

        for (int i = 0; i < abs_n; ++i) {
            int src_i = (n >= 0) ? i : (len - abs_n + i);
            if (src_i >= 0 && src_i < len) {
                result.push_back((*strand)[src_i]);
            } else {
                result.push_back(fill);
            }
        }
        m->result = m->heap->allocate_strand(std::move(result));
        return;
    }

    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // NDARRAY: multi-axis take
    if (rhs->is_ndarray()) {
        const Value::NDArrayData* nd = rhs->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Left argument must be vector with length = rank
        if (!lhs->is_vector()) {
            m->throw_error("LENGTH ERROR: left argument length must equal right argument rank", nullptr, 5, 0);
            return;
        }
        const Eigen::MatrixXd* counts = lhs->as_matrix();
        if (counts->rows() != rank) {
            m->throw_error("LENGTH ERROR: left argument length must equal right argument rank", nullptr, 5, 0);
            return;
        }

        // Compute new shape and source offsets
        std::vector<int> new_shape(rank);
        std::vector<int> src_offset(rank);  // Where to start taking from in source
        for (int d = 0; d < rank; ++d) {
            int n = static_cast<int>((*counts)(d, 0));
            new_shape[d] = std::abs(n);
            // Negative n means take from end
            src_offset[d] = (n >= 0) ? 0 : (shape[d] - std::abs(n));
        }

        // Compute total size and allocate result
        int total = 1;
        for (int d = 0; d < rank; ++d) total *= new_shape[d];

        Eigen::VectorXd result_data(total);
        result_data.setZero();  // Fill with zeros for padding

        // Compute strides for source and result
        std::vector<int> src_strides(rank);
        std::vector<int> dst_strides(rank);
        src_strides[rank - 1] = 1;
        dst_strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            src_strides[d] = src_strides[d + 1] * shape[d + 1];
            dst_strides[d] = dst_strides[d + 1] * new_shape[d + 1];
        }

        // Copy data with proper indexing
        std::vector<int> idx(rank, 0);
        for (int i = 0; i < total; ++i) {
            // Convert linear index i to multi-dimensional index
            int tmp = i;
            for (int d = 0; d < rank; ++d) {
                idx[d] = tmp / dst_strides[d];
                tmp %= dst_strides[d];
            }

            // Check if source index is valid
            bool valid = true;
            int src_linear = 0;
            for (int d = 0; d < rank; ++d) {
                int src_idx = idx[d] + src_offset[d];
                if (src_idx < 0 || src_idx >= shape[d]) {
                    valid = false;
                    break;
                }
                src_linear += src_idx * src_strides[d];
            }

            if (valid) {
                result_data(i) = (*nd->data)(src_linear);
            }
        }

        m->result = m->heap->allocate_ndarray(result_data, new_shape);
        return;
    }

    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ↑ requires array argument", nullptr, 11, 0);
        return;
    }
    bool is_char = rhs->is_char_data();
    const Eigen::MatrixXd* mat = rhs->as_matrix();
    double fill = is_char ? 32.0 : 0.0;

    // Vector left argument: multi-axis take
    if (lhs->is_vector()) {
        const Eigen::MatrixXd* counts = lhs->as_matrix();
        int lhs_len = counts->rows();
        int rank = rhs->is_vector() ? 1 : 2;

        if (lhs_len != rank) {
            m->throw_error("LENGTH ERROR: left argument length must equal right argument rank", nullptr, 5, 0);
            return;
        }

        if (rhs->is_vector()) {
            int n = static_cast<int>((*counts)(0, 0));
            int len = mat->rows();
            int abs_n = std::abs(n);
            Eigen::VectorXd result(abs_n);
            result.setConstant(fill);
            for (int i = 0; i < abs_n; ++i) {
                int src_i = (n >= 0) ? i : (len - abs_n + i);
                if (src_i >= 0 && src_i < len) result(i) = (*mat)(src_i, 0);
            }
            m->result = m->heap->allocate_vector(result, is_char);
            return;
        }

        int n_rows = static_cast<int>((*counts)(0, 0));
        int n_cols = static_cast<int>((*counts)(1, 0));
        int rows = mat->rows();
        int cols = mat->cols();
        int abs_rows = std::abs(n_rows);
        int abs_cols = std::abs(n_cols);

        Eigen::MatrixXd result(abs_rows, abs_cols);
        result.setConstant(fill);
        for (int i = 0; i < abs_rows; ++i) {
            int src_i = (n_rows >= 0) ? i : (rows - abs_rows + i);
            if (src_i < 0 || src_i >= rows) continue;
            for (int j = 0; j < abs_cols; ++j) {
                int src_j = (n_cols >= 0) ? j : (cols - abs_cols + j);
                if (src_j < 0 || src_j >= cols) continue;
                result(i, j) = (*mat)(src_i, src_j);
            }
        }
        m->result = m->heap->allocate_matrix(result, is_char);
        return;
    }

    if (!lhs->is_scalar()) {
        m->throw_error("RANK ERROR: take count must be scalar or vector", nullptr, 4, 0);
        return;
    }

    int n = static_cast<int>(lhs->as_scalar());
    int rank = rhs->is_vector() ? 1 : 2;
    int take_axis = 1;

    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be a scalar", m->control, 4, 0);
            return;
        }
        take_axis = static_cast<int>(axis->as_scalar());
        if (take_axis < 1 || take_axis > rank) {
            m->throw_error("AXIS ERROR: invalid axis for array rank", m->control, 4, 0);
            return;
        }
    }

    if (rhs->is_vector()) {
        int len = mat->rows();
        int abs_n = std::abs(n);
        Eigen::VectorXd result(abs_n);
        result.setConstant(fill);
        for (int i = 0; i < abs_n; ++i) {
            int src_idx = (n >= 0) ? i : (len - abs_n + i);
            if (src_idx >= 0 && src_idx < len) result(i) = (*mat)(src_idx, 0);
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    int rows = mat->rows();
    int cols = mat->cols();
    int abs_n = std::abs(n);

    if (take_axis == 1) {
        Eigen::MatrixXd result(abs_n, cols);
        result.setConstant(fill);
        for (int i = 0; i < abs_n; ++i) {
            int src_idx = (n >= 0) ? i : (rows - abs_n + i);
            if (src_idx >= 0 && src_idx < rows) result.row(i) = mat->row(src_idx);
        }
        m->result = m->heap->allocate_matrix(result, is_char);
    } else {
        Eigen::MatrixXd result(rows, abs_n);
        result.setConstant(fill);
        for (int j = 0; j < abs_n; ++j) {
            int src_idx = (n >= 0) ? j : (cols - abs_n + j);
            if (src_idx >= 0 && src_idx < cols) result.col(j) = mat->col(src_idx);
        }
        m->result = m->heap->allocate_matrix(result, is_char);
    }
}

// Drop (↓) - dyadic: drop first n elements
void fn_drop(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    if (rhs->is_scalar()) {
        Eigen::VectorXd result(0);
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // Strand drop: drop first/last n elements
    if (rhs->is_strand()) {
        if (!lhs->is_scalar()) {
            m->throw_error("RANK ERROR: strand drop requires scalar count", nullptr, 4, 0);
            return;
        }
        int n = static_cast<int>(lhs->as_scalar());
        const std::vector<Value*>* strand = rhs->as_strand();
        int len = static_cast<int>(strand->size());
        int abs_n = std::abs(n);
        int result_len = std::max(0, len - abs_n);

        std::vector<Value*> result;
        result.reserve(result_len);

        // Positive n: drop from start; Negative n: drop from end
        int start = (n >= 0) ? abs_n : 0;
        int end = (n >= 0) ? len : len - abs_n;

        for (int i = start; i < end; ++i) {
            result.push_back((*strand)[i]);
        }
        m->result = m->heap->allocate_strand(std::move(result));
        return;
    }

    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // NDARRAY: multi-axis drop
    if (rhs->is_ndarray()) {
        const Value::NDArrayData* nd = rhs->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Left argument must be vector with length = rank
        if (!lhs->is_vector()) {
            m->throw_error("LENGTH ERROR: left argument length must equal right argument rank", nullptr, 5, 0);
            return;
        }
        const Eigen::MatrixXd* counts = lhs->as_matrix();
        if (counts->rows() != rank) {
            m->throw_error("LENGTH ERROR: left argument length must equal right argument rank", nullptr, 5, 0);
            return;
        }

        // Compute new shape and source offsets
        std::vector<int> new_shape(rank);
        std::vector<int> src_offset(rank);
        for (int d = 0; d < rank; ++d) {
            int n = static_cast<int>((*counts)(d, 0));
            int abs_n = std::abs(n);
            new_shape[d] = std::max(0, shape[d] - abs_n);
            // Positive n: drop from start; Negative n: drop from end
            src_offset[d] = (n >= 0) ? abs_n : 0;
        }

        // Compute total size and allocate result
        int total = 1;
        for (int d = 0; d < rank; ++d) total *= new_shape[d];

        Eigen::VectorXd result_data(total);

        // Compute strides for source and result
        std::vector<int> src_strides(rank);
        std::vector<int> dst_strides(rank);
        src_strides[rank - 1] = 1;
        dst_strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            src_strides[d] = src_strides[d + 1] * shape[d + 1];
            dst_strides[d] = dst_strides[d + 1] * new_shape[d + 1];
        }

        // Copy data with proper indexing
        std::vector<int> idx(rank, 0);
        for (int i = 0; i < total; ++i) {
            // Convert linear index i to multi-dimensional index
            int tmp = i;
            for (int d = 0; d < rank; ++d) {
                idx[d] = tmp / dst_strides[d];
                tmp %= dst_strides[d];
            }

            // Compute source linear index
            int src_linear = 0;
            for (int d = 0; d < rank; ++d) {
                src_linear += (idx[d] + src_offset[d]) * src_strides[d];
            }

            result_data(i) = (*nd->data)(src_linear);
        }

        m->result = m->heap->allocate_ndarray(result_data, new_shape);
        return;
    }

    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ↓ requires array argument", nullptr, 11, 0);
        return;
    }
    bool is_char = rhs->is_char_data();
    const Eigen::MatrixXd* mat = rhs->as_matrix();

    // Vector left argument: multi-axis drop
    if (lhs->is_vector()) {
        const Eigen::MatrixXd* counts = lhs->as_matrix();
        int lhs_len = counts->rows();
        int rank = rhs->is_vector() ? 1 : 2;

        if (lhs_len != rank) {
            m->throw_error("LENGTH ERROR: left argument length must equal right argument rank", nullptr, 5, 0);
            return;
        }

        if (rhs->is_vector()) {
            int n = static_cast<int>((*counts)(0, 0));
            int len = mat->rows();
            int abs_n = std::abs(n);
            int result_len = std::max(0, len - abs_n);
            Eigen::VectorXd result(result_len);
            int start = (n >= 0) ? abs_n : 0;
            for (int i = 0; i < result_len; ++i) {
                result(i) = (*mat)(start + i, 0);
            }
            m->result = m->heap->allocate_vector(result, is_char);
            return;
        }

        int n_rows = static_cast<int>((*counts)(0, 0));
        int n_cols = static_cast<int>((*counts)(1, 0));
        int rows = mat->rows();
        int cols = mat->cols();
        int abs_rows = std::abs(n_rows);
        int abs_cols = std::abs(n_cols);
        int result_rows = std::max(0, rows - abs_rows);
        int result_cols = std::max(0, cols - abs_cols);

        Eigen::MatrixXd result(result_rows, result_cols);
        int row_start = (n_rows >= 0) ? abs_rows : 0;
        int col_start = (n_cols >= 0) ? abs_cols : 0;
        for (int i = 0; i < result_rows; ++i) {
            for (int j = 0; j < result_cols; ++j) {
                result(i, j) = (*mat)(row_start + i, col_start + j);
            }
        }
        m->result = m->heap->allocate_matrix(result, is_char);
        return;
    }

    if (!lhs->is_scalar()) {
        m->throw_error("RANK ERROR: drop count must be scalar or vector", nullptr, 4, 0);
        return;
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_vector()) {
        int len = mat->rows();
        int abs_n = std::abs(n);
        int result_len = std::max(0, len - abs_n);
        Eigen::VectorXd result(result_len);
        int start = (n >= 0) ? abs_n : 0;
        for (int i = 0; i < result_len; ++i) {
            result(i) = (*mat)(start + i, 0);
        }
        m->result = m->heap->allocate_vector(result, is_char);
        return;
    }

    int drop_axis = 1;
    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("RANK ERROR: axis must be scalar", nullptr, 4, 0);
            return;
        }
        drop_axis = static_cast<int>(axis->as_scalar());
        if (drop_axis < 1 || drop_axis > 2) {
            m->throw_error("AXIS ERROR: axis must be 1 or 2 for matrix", m->control, 4, 0);
            return;
        }
    }

    int rows = mat->rows();
    int cols = mat->cols();
    int abs_n = std::abs(n);

    if (drop_axis == 1) {
        int result_rows = std::max(0, rows - abs_n);
        Eigen::MatrixXd result(result_rows, cols);
        if (n >= 0) {
            result = mat->bottomRows(result_rows);
        } else {
            result = mat->topRows(result_rows);
        }
        m->result = m->heap->allocate_matrix(result, is_char);
    } else {
        int result_cols = std::max(0, cols - abs_n);
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

    // Strand reversal: reverse the order of elements (ISO 10.1.4)
    if (omega->is_strand()) {
        const std::vector<Value*>* strand = omega->as_strand();
        std::vector<Value*> reversed(strand->rbegin(), strand->rend());
        m->result = m->heap->allocate_strand(std::move(reversed));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    // NDARRAY: reverse along specified axis (default: last)
    if (omega->is_ndarray()) {
        const Value::NDArrayData* nd = omega->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Default: last axis (1-indexed)
        int reverse_axis = rank;
        if (axis != nullptr) {
            if (!axis->is_scalar()) {
                m->throw_error("AXIS ERROR: axis must be a scalar", m->control, 4, 0);
                return;
            }
            reverse_axis = static_cast<int>(axis->as_scalar());
            if (reverse_axis < 1 || reverse_axis > rank) {
                m->throw_error("AXIS ERROR: invalid axis for array rank", m->control, 4, 0);
                return;
            }
        }

        // Convert to 0-indexed
        int ax = reverse_axis - 1;
        int total = static_cast<int>(nd->data->size());

        Eigen::VectorXd result_data(total);

        // Compute strides
        std::vector<int> strides(rank);
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // For each element, compute reversed index
        std::vector<int> idx(rank);
        for (int i = 0; i < total; ++i) {
            // Convert linear index to multi-dimensional
            int tmp = i;
            for (int d = 0; d < rank; ++d) {
                idx[d] = tmp / strides[d];
                tmp %= strides[d];
            }

            // Reverse along specified axis
            int src_linear = 0;
            for (int d = 0; d < rank; ++d) {
                int src_idx = (d == ax) ? (shape[d] - 1 - idx[d]) : idx[d];
                src_linear += src_idx * strides[d];
            }

            result_data(i) = (*nd->data)(src_linear);
        }

        m->result = m->heap->allocate_ndarray(result_data, shape);
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌽ requires array argument", nullptr, 11, 0);
        return;
    }
    bool is_char = omega->is_char_data();
    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Determine which axis to reverse along
    // Default: last axis (2 for matrix, 1 for vector)
    int rank = omega->is_vector() ? 1 : 2;
    int reverse_axis = rank;  // Default to last axis (1-indexed)

    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be a scalar", m->control, 4, 0);
            return;
        }
        reverse_axis = static_cast<int>(axis->as_scalar());
        // Validate axis (1-indexed with ⎕IO=1)
        if (reverse_axis < 1 || reverse_axis > rank) {
            m->throw_error("AXIS ERROR: invalid axis for array rank", m->control, 4, 0);
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

    // NDARRAY: reverse along specified axis (default: first)
    if (omega->is_ndarray()) {
        const Value::NDArrayData* nd = omega->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Default: first axis (1-indexed)
        int reverse_axis = 1;
        if (axis != nullptr) {
            if (!axis->is_scalar()) {
                m->throw_error("AXIS ERROR: axis must be a scalar", m->control, 4, 0);
                return;
            }
            reverse_axis = static_cast<int>(axis->as_scalar());
            if (reverse_axis < 1 || reverse_axis > rank) {
                m->throw_error("AXIS ERROR: invalid axis for array rank", m->control, 4, 0);
                return;
            }
        }

        // Convert to 0-indexed
        int ax = reverse_axis - 1;
        int total = static_cast<int>(nd->data->size());

        Eigen::VectorXd result_data(total);

        // Compute strides
        std::vector<int> strides(rank);
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // For each element, compute reversed index
        std::vector<int> idx(rank);
        for (int i = 0; i < total; ++i) {
            // Convert linear index to multi-dimensional
            int tmp = i;
            for (int d = 0; d < rank; ++d) {
                idx[d] = tmp / strides[d];
                tmp %= strides[d];
            }

            // Reverse along specified axis
            int src_linear = 0;
            for (int d = 0; d < rank; ++d) {
                int src_idx = (d == ax) ? (shape[d] - 1 - idx[d]) : idx[d];
                src_linear += src_idx * strides[d];
            }

            result_data(i) = (*nd->data)(src_linear);
        }

        m->result = m->heap->allocate_ndarray(result_data, shape);
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⊖ requires array argument", nullptr, 11, 0);
        return;
    }
    bool is_char = omega->is_char_data();
    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Determine which axis to reverse along
    // Default: first axis (1)
    int rank = omega->is_vector() ? 1 : 2;
    int reverse_axis = 1;  // Default to first axis (1-indexed)

    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be a scalar", m->control, 4, 0);
            return;
        }
        reverse_axis = static_cast<int>(axis->as_scalar());
        // Validate axis (1-indexed with ⎕IO=1)
        if (reverse_axis < 1 || reverse_axis > rank) {
            m->throw_error("AXIS ERROR: invalid axis for array rank", m->control, 4, 0);
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
        m->result = m->heap->allocate_scalar(1.0);
        return;
    }

    if (omega->is_strand()) {
        m->result = m->heap->allocate_scalar(static_cast<double>(omega->as_strand()->size()));
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ≢ requires array argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* mat = omega->as_matrix();
    m->result = m->heap->allocate_scalar(static_cast<double>(mat->rows()));
}

// Rotate (⌽) - dyadic: rotate elements along last axis
void fn_rotate(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    // ISO 13751 10.2.7: Validate axis if provided
    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be scalar", m->control, 4, 0);
            return;
        }
        double ax = axis->as_scalar();
        if (!is_near_integer(ax, INTEGER_TOLERANCE)) {
            m->throw_error("AXIS ERROR: axis must be integer", m->control, 4, 0);
            return;
        }
        int ax_int = static_cast<int>(std::round(ax));
        int io = m->io;
        int rank = rhs->is_scalar() ? 0 : (rhs->is_vector() ? 1 : 2);
        if (ax_int < io || ax_int > io + rank - 1) {
            m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
            return;
        }
    }

    if (rhs->is_scalar()) {
        // ISO 13751 10.2.7: Rotating a scalar - just validate left arg
        if (lhs->is_scalar()) {
            double val = lhs->as_scalar();
            if (!is_near_integer(val, INTEGER_TOLERANCE)) {
                m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
                return;
            }
        }
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    // Strand rotation: rotate elements (ISO 10.2.7)
    if (rhs->is_strand()) {
        if (!lhs->is_scalar()) {
            m->throw_error("RANK ERROR: strand rotation requires scalar count", nullptr, 4, 0);
            return;
        }
        double val = lhs->as_scalar();
        if (!is_near_integer(val, INTEGER_TOLERANCE)) {
            m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
            return;
        }
        int n = static_cast<int>(std::round(val));
        const std::vector<Value*>* strand = rhs->as_strand();
        int len = static_cast<int>(strand->size());
        if (len == 0) {
            m->result = m->heap->allocate_strand(std::vector<Value*>());
            return;
        }
        n = ((n % len) + len) % len;  // Normalize to [0, len)
        std::vector<Value*> rotated;
        rotated.reserve(len);
        for (int i = 0; i < len; i++) {
            rotated.push_back((*strand)[(i + n) % len]);
        }
        m->result = m->heap->allocate_strand(std::move(rotated));
        return;
    }

    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // NDARRAY: rotate along specified axis (default: last)
    if (rhs->is_ndarray()) {
        const Value::NDArrayData* nd = rhs->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Default: last axis (1-indexed)
        int rotate_axis = rank;
        if (axis != nullptr) {
            rotate_axis = static_cast<int>(axis->as_scalar());
            if (rotate_axis < 1 || rotate_axis > rank) {
                m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
                return;
            }
        }

        // Convert to 0-indexed
        int ax = rotate_axis - 1;
        int ax_len = shape[ax];

        if (ax_len == 0) {
            m->result = m->heap->allocate_ndarray(*nd->data, shape);
            return;
        }

        // Compute strides
        std::vector<int> strides(rank);
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        int total = static_cast<int>(nd->data->size());
        Eigen::VectorXd result_data(total);

        if (lhs->is_scalar()) {
            // Scalar rotation count - rotate all by same amount
            double val = lhs->as_scalar();
            if (!is_near_integer(val, INTEGER_TOLERANCE)) {
                m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
                return;
            }
            int n = static_cast<int>(std::round(val));
            n = ((n % ax_len) + ax_len) % ax_len;

            std::vector<int> idx(rank);
            for (int i = 0; i < total; ++i) {
                int tmp = i;
                for (int d = 0; d < rank; ++d) {
                    idx[d] = tmp / strides[d];
                    tmp %= strides[d];
                }

                int src_linear = 0;
                for (int d = 0; d < rank; ++d) {
                    int src_idx = (d == ax) ? ((idx[d] + n) % ax_len) : idx[d];
                    src_linear += src_idx * strides[d];
                }

                result_data(i) = (*nd->data)(src_linear);
            }
        } else if (lhs->is_vector()) {
            // Vector rotation: each "row" (along axis before rotate axis) gets different rotation
            const Eigen::MatrixXd* rotations = lhs->as_matrix();
            int num_rotations = rotations->rows();

            // The rotation vector should match dimension before the rotate axis
            // For 2 3 4⍴⍳24 with default axis 3 (last), rotations match dim 2 (3 rows)
            int match_dim = (ax > 0) ? ax - 1 : 0;
            if (num_rotations != shape[match_dim]) {
                m->throw_error("LENGTH ERROR: rotation count length must match array dimension", nullptr, 5, 0);
                return;
            }

            // Validate all rotation counts are integers
            for (int i = 0; i < num_rotations; ++i) {
                if (!is_near_integer((*rotations)(i, 0), INTEGER_TOLERANCE)) {
                    m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
                    return;
                }
            }

            std::vector<int> idx(rank);
            for (int i = 0; i < total; ++i) {
                int tmp = i;
                for (int d = 0; d < rank; ++d) {
                    idx[d] = tmp / strides[d];
                    tmp %= strides[d];
                }

                // Get rotation count based on the matching dimension index
                int rot_idx = idx[match_dim];
                int n = static_cast<int>(std::round((*rotations)(rot_idx, 0)));
                n = ((n % ax_len) + ax_len) % ax_len;

                int src_linear = 0;
                for (int d = 0; d < rank; ++d) {
                    int src_idx = (d == ax) ? ((idx[d] + n) % ax_len) : idx[d];
                    src_linear += src_idx * strides[d];
                }

                result_data(i) = (*nd->data)(src_linear);
            }
        } else {
            m->throw_error("RANK ERROR: rotate count must be scalar or vector", nullptr, 4, 0);
            return;
        }

        m->result = m->heap->allocate_ndarray(result_data, shape);
        return;
    }

    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌽ requires array argument", nullptr, 11, 0);
        return;
    }
    bool is_char = rhs->is_char_data();
    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (lhs->is_scalar()) {
        // Scalar rotation count
        double val = lhs->as_scalar();
        if (!is_near_integer(val, INTEGER_TOLERANCE)) {
            m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
            return;
        }
        int n = static_cast<int>(std::round(val));

        if (rhs->is_vector()) {
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

        // For matrices: rotate columns within each row (last axis)
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
        return;
    }

    // ISO 13751 10.2.7: Vector rotation count - each row rotated by different amount
    if (!lhs->is_vector()) {
        m->throw_error("RANK ERROR: rotate count must be scalar or vector", nullptr, 4, 0);
        return;
    }

    if (!rhs->is_matrix()) {
        m->throw_error("RANK ERROR: vector rotate requires matrix right argument", nullptr, 4, 0);
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (lmat->rows() != rows) {
        m->throw_error("LENGTH ERROR: rotate count length must match number of rows", nullptr, 5, 0);
        return;
    }

    // Validate all rotation counts are near-integers
    for (int i = 0; i < rows; ++i) {
        if (!is_near_integer((*lmat)(i, 0), INTEGER_TOLERANCE)) {
            m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
            return;
        }
    }

    if (cols == 0) {
        m->result = m->heap->allocate_matrix(*mat, is_char);
        return;
    }

    Eigen::MatrixXd result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        int n = static_cast<int>(std::round((*lmat)(i, 0)));
        n = ((n % cols) + cols) % cols;
        for (int j = 0; j < cols; ++j) {
            result(i, j) = (*mat)(i, (j + n) % cols);
        }
    }
    m->result = m->heap->allocate_matrix(result, is_char);
}

// Rotate First (⊖) - dyadic: rotate elements along first axis
void fn_rotate_first(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    // ISO 13751 10.2.7: Validate axis if provided
    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be scalar", m->control, 4, 0);
            return;
        }
        double ax = axis->as_scalar();
        if (!is_near_integer(ax, INTEGER_TOLERANCE)) {
            m->throw_error("AXIS ERROR: axis must be integer", m->control, 4, 0);
            return;
        }
        int ax_int = static_cast<int>(std::round(ax));
        int io = m->io;
        int rank = rhs->is_scalar() ? 0 : (rhs->is_vector() ? 1 : 2);
        if (ax_int < io || ax_int > io + rank - 1) {
            m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
            return;
        }
    }

    if (rhs->is_scalar()) {
        // ISO 13751 10.2.7: Rotating a scalar - just validate left arg
        if (lhs->is_scalar()) {
            double val = lhs->as_scalar();
            if (!is_near_integer(val, INTEGER_TOLERANCE)) {
                m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
                return;
            }
        }
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // NDARRAY: rotate along specified axis (default: first)
    if (rhs->is_ndarray()) {
        if (!lhs->is_scalar()) {
            m->throw_error("RANK ERROR: NDARRAY rotate requires scalar count", nullptr, 4, 0);
            return;
        }
        double val = lhs->as_scalar();
        if (!is_near_integer(val, INTEGER_TOLERANCE)) {
            m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
            return;
        }
        int n = static_cast<int>(std::round(val));

        const Value::NDArrayData* nd = rhs->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Default: first axis (1-indexed)
        int rotate_axis = 1;
        if (axis != nullptr) {
            rotate_axis = static_cast<int>(axis->as_scalar());
            if (rotate_axis < 1 || rotate_axis > rank) {
                m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
                return;
            }
        }

        // Convert to 0-indexed
        int ax = rotate_axis - 1;
        int ax_len = shape[ax];

        if (ax_len == 0) {
            m->result = m->heap->allocate_ndarray(*nd->data, shape);
            return;
        }

        // Normalize rotation
        n = ((n % ax_len) + ax_len) % ax_len;

        int total = static_cast<int>(nd->data->size());
        Eigen::VectorXd result_data(total);

        // Compute strides
        std::vector<int> strides(rank);
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // For each element, compute rotated source index
        std::vector<int> idx(rank);
        for (int i = 0; i < total; ++i) {
            // Convert linear index to multi-dimensional
            int tmp = i;
            for (int d = 0; d < rank; ++d) {
                idx[d] = tmp / strides[d];
                tmp %= strides[d];
            }

            // Rotate along specified axis
            int src_linear = 0;
            for (int d = 0; d < rank; ++d) {
                int src_idx = (d == ax) ? ((idx[d] + n) % ax_len) : idx[d];
                src_linear += src_idx * strides[d];
            }

            result_data(i) = (*nd->data)(src_linear);
        }

        m->result = m->heap->allocate_ndarray(result_data, shape);
        return;
    }

    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⊖ requires array argument", nullptr, 11, 0);
        return;
    }
    bool is_char = rhs->is_char_data();
    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (lhs->is_scalar()) {
        double val = lhs->as_scalar();
        if (!is_near_integer(val, INTEGER_TOLERANCE)) {
            m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
            return;
        }
        int n = static_cast<int>(std::round(val));

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
        return;
    }

    // ISO 13751 10.2.7: Vector rotation count - each column rotated by different amount
    if (!lhs->is_vector()) {
        m->throw_error("RANK ERROR: rotate count must be scalar or vector", nullptr, 4, 0);
        return;
    }

    if (!rhs->is_matrix()) {
        m->throw_error("RANK ERROR: vector rotate requires matrix right argument", nullptr, 4, 0);
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (lmat->rows() != cols) {
        m->throw_error("LENGTH ERROR: rotate count length must match number of columns", nullptr, 5, 0);
        return;
    }

    // Validate all rotation counts are near-integers
    for (int i = 0; i < cols; ++i) {
        if (!is_near_integer((*lmat)(i, 0), INTEGER_TOLERANCE)) {
            m->throw_error("DOMAIN ERROR: rotate count must be integer", nullptr, 11, 0);
            return;
        }
    }

    if (rows == 0) {
        m->result = m->heap->allocate_matrix(*mat, is_char);
        return;
    }

    Eigen::MatrixXd result(rows, cols);
    for (int j = 0; j < cols; ++j) {
        int n = static_cast<int>(std::round((*lmat)(j, 0)));
        n = ((n % rows) + rows) % rows;
        for (int i = 0; i < rows; ++i) {
            result(i, j) = (*mat)((i + n) % rows, j);
        }
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

    // ISO 13751: left argument must be a vector
    if (!lhs->is_scalar() && !lhs->is_vector()) {
        m->throw_error("RANK ERROR: left argument of index-of must be a vector", nullptr, 4, 0);
        return;
    }

    // Get lhs as a flat array of values to search in
    Eigen::VectorXd haystack = flatten_value(lhs);
    int io = m->io;
    double ct = m->ct;
    double not_found = static_cast<double>(haystack.size() + io);  // ⎕IO + length

    // Tolerant equality per ISO 13751: |a-b| ≤ ⎕CT × max(|a|, |b|)
    auto tolerant_equal = [ct](double a, double b) -> bool {
        if (a == b) return true;
        double magnitude = std::max(std::abs(a), std::abs(b));
        return std::abs(a - b) <= ct * magnitude;
    };

    // Search for needle in haystack, return index or not_found
    auto find_index = [&haystack, not_found, io, &tolerant_equal](double needle) -> double {
        for (int i = 0; i < haystack.size(); ++i) {
            if (tolerant_equal(haystack(i), needle)) {
                return static_cast<double>(i + io);  // ⎕IO
            }
        }
        return not_found;
    };

    // Scalar or single-element vector returns scalar result
    if (rhs->is_scalar() || (rhs->is_vector() && rhs->size() == 1)) {
        double needle = rhs->is_scalar() ? rhs->as_scalar() : rhs->as_matrix()->coeff(0, 0);
        m->result = m->heap->allocate_scalar(find_index(needle));
        return;
    }

    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍳ requires array argument", nullptr, 11, 0);
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
// Recursively descends into strands (nested arrays) to collect all scalars.
//
// Helper to recursively enlist a value into a vector of scalars
static void enlist_into(Machine* m, Value* v, std::vector<double>& out) {
    if (v->is_scalar()) {
        out.push_back(v->as_scalar());
    } else if (v->is_strand()) {
        // Recursively enlist each strand element (ISO 8.2.6)
        const std::vector<Value*>* strand = v->as_strand();
        for (Value* elem : *strand) {
            enlist_into(m, elem, out);
        }
    } else if (v->is_string()) {
        v = v->to_char_vector(m->heap);
        const Eigen::MatrixXd* mat = v->as_matrix();
        for (int i = 0; i < mat->size(); i++) {
            out.push_back((*mat)(i / mat->cols(), i % mat->cols()));
        }
    } else if (v->is_array()) {
        const Eigen::MatrixXd* mat = v->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();
        for (int i = 0; i < mat->size(); i++) {
            out.push_back((*mat)(i / cols, i % cols));
        }
    }
}

void fn_enlist(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);

    // For strands: recursively flatten all elements (ISO 8.2.6)
    if (omega->is_strand()) {
        std::vector<double> result;
        enlist_into(m, omega, result);
        if (result.empty()) {
            m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
            return;
        }
        Eigen::VectorXd vec(result.size());
        for (size_t i = 0; i < result.size(); i++) {
            vec(i) = result[i];
        }
        m->result = m->heap->allocate_vector(vec);
        return;
    }

    // For simple arrays, enlist = ravel
    fn_ravel(m, axis, omega);
}

// Member Of (∊) - dyadic: check if elements of lhs are in rhs
// Returns boolean array with 1 where element is found, 0 otherwise
void fn_member_of(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (!lhs->is_scalar() && !lhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ∊ requires array argument", nullptr, 11, 0);
        return;
    }
    if (!rhs->is_scalar() && !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ∊ requires array argument", nullptr, 11, 0);
        return;
    }

    // Get rhs as flat array to search in
    Eigen::VectorXd set = flatten_value(rhs);
    double ct = m->ct;

    // Tolerant equality per ISO 13751: |a-b| ≤ ⎕CT × max(|a|, |b|)
    auto tolerant_equal = [ct](double a, double b) -> bool {
        if (a == b) return true;
        double magnitude = std::max(std::abs(a), std::abs(b));
        return std::abs(a - b) <= ct * magnitude;
    };

    // Check if val is in set
    auto is_member = [&set, &tolerant_equal](double val) -> double {
        for (int i = 0; i < set.size(); ++i) {
            if (tolerant_equal(set(i), val)) {
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
        m->throw_error("RANK ERROR: grade requires array", nullptr, 4, 0);
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);
    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍋ requires array argument", nullptr, 11, 0);
        return;
    }

    // Matrix case: grade rows lexicographically
    if (omega->is_matrix()) {
        const Eigen::MatrixXd* mat = omega->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();

        // Create index array for rows
        std::vector<int> indices(rows);
        for (int i = 0; i < rows; ++i) {
            indices[i] = i;
        }

        // Sort row indices lexicographically (stable: equal rows preserve index order)
        std::stable_sort(indices.begin(), indices.end(), [mat, cols](int a, int b) {
            for (int j = 0; j < cols; ++j) {
                if ((*mat)(a, j) < (*mat)(b, j)) return true;
                if ((*mat)(a, j) > (*mat)(b, j)) return false;
            }
            return false;
        });

        // Convert to result vector (⎕IO)
        Eigen::VectorXd result(rows);
        for (int i = 0; i < rows; ++i) {
            result(i) = static_cast<double>(indices[i] + m->io);
        }

        m->result = m->heap->allocate_vector(result);
        return;
    }

    // Vector case
    Eigen::VectorXd data = flatten_value(omega);
    int n = data.size();

    // Create index array
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Sort indices by corresponding data values (ascending, stable)
    std::stable_sort(indices.begin(), indices.end(), [&data](int a, int b) {
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
        m->throw_error("RANK ERROR: grade requires array", nullptr, 4, 0);
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);
    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍒ requires array argument", nullptr, 11, 0);
        return;
    }

    // Matrix case: grade rows lexicographically (descending)
    if (omega->is_matrix()) {
        const Eigen::MatrixXd* mat = omega->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();

        // Create index array for rows
        std::vector<int> indices(rows);
        for (int i = 0; i < rows; ++i) {
            indices[i] = i;
        }

        // Sort row indices lexicographically (descending, stable)
        std::stable_sort(indices.begin(), indices.end(), [mat, cols](int a, int b) {
            for (int j = 0; j < cols; ++j) {
                if ((*mat)(a, j) > (*mat)(b, j)) return true;
                if ((*mat)(a, j) < (*mat)(b, j)) return false;
            }
            return false;
        });

        // Convert to result vector (⎕IO)
        Eigen::VectorXd result(rows);
        for (int i = 0; i < rows; ++i) {
            result(i) = static_cast<double>(indices[i] + m->io);
        }

        m->result = m->heap->allocate_vector(result);
        return;
    }

    // Vector case
    Eigen::VectorXd data = flatten_value(omega);
    int n = data.size();

    // Create index array
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Sort indices by corresponding data values (descending, stable)
    std::stable_sort(indices.begin(), indices.end(), [&data](int a, int b) {
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
        m->throw_error("RANK ERROR: collating sequence must have rank > 0", nullptr, 4, 0);
        return;
    }
    if (!lhs->is_char_data() && !lhs->is_string()) {
        m->throw_error("DOMAIN ERROR: collating sequence must be character", nullptr, 11, 0);
        return;
    }

    // Validate B is character array
    if (!rhs->is_char_data() && !rhs->is_string()) {
        m->throw_error("DOMAIN ERROR: right argument must be character", nullptr, 11, 0);
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
        m->throw_error("RANK ERROR: collating sequence must have rank > 0", nullptr, 4, 0);
        return;
    }
    if (!lhs->is_char_data() && !lhs->is_string()) {
        m->throw_error("DOMAIN ERROR: collating sequence must be character", nullptr, 11, 0);
        return;
    }

    // Validate B is character array
    if (!rhs->is_char_data() && !rhs->is_string()) {
        m->throw_error("DOMAIN ERROR: right argument must be character", nullptr, 11, 0);
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
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    bool is_char = rhs->is_char_data();

    // Get counts from lhs
    Eigen::VectorXd counts = flatten_value(lhs);

    // Strand case: replicate top-level elements
    if (rhs->is_strand()) {
        REJECT_AXIS(m, axis);
        std::vector<Value*>* strand = rhs->as_strand();
        int n = static_cast<int>(strand->size());

        // Scalar extension for counts
        if (counts.size() == 1 && n > 1) {
            double scalar_count = counts(0);
            counts.resize(n);
            counts.setConstant(scalar_count);
        }

        if (static_cast<int>(counts.size()) != n) {
            m->throw_error("LENGTH ERROR: replicate count must match strand length", nullptr, 5, 0);
            return;
        }

        // Calculate total size: negative counts insert |c| fill elements
        int total = 0;
        for (int i = 0; i < n; ++i) {
            int c = static_cast<int>(counts(i));
            total += std::abs(c);
        }

        if (total == 0) {
            m->result = m->heap->allocate_strand({});
            return;
        }

        Value* fill = m->heap->allocate_scalar(0.0);
        std::vector<Value*> result;
        result.reserve(total);
        for (int i = 0; i < n; ++i) {
            int c = static_cast<int>(counts(i));
            if (c >= 0) {
                for (int r = 0; r < c; ++r) {
                    result.push_back((*strand)[i]);
                }
            } else {
                // Negative: insert |c| fill elements, drop the original
                for (int r = 0; r < -c; ++r) {
                    result.push_back(fill);
                }
            }
        }

        m->result = m->heap->allocate_strand(result);
        return;
    }

    // NDARRAY case: replicate along specified axis (default: last)
    if (rhs->is_ndarray()) {
        const Value::NDArrayData* nd = rhs->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Determine axis (default: last)
        int ax = rank - 1;
        if (axis) {
            if (!axis->is_scalar()) {
                m->throw_error("AXIS ERROR: axis must be scalar", m->control, 4, 0);
                return;
            }
            ax = static_cast<int>(axis->as_scalar()) - m->io;
            if (ax < 0 || ax >= rank) {
                m->throw_error("AXIS ERROR: axis out of bounds", m->control, 4, 0);
                return;
            }
        }

        int axis_len = shape[ax];

        // Scalar extension for counts
        if (counts.size() == 1) {
            double scalar_count = counts(0);
            counts.resize(axis_len);
            counts.setConstant(scalar_count);
        }

        if (static_cast<int>(counts.size()) != axis_len) {
            m->throw_error("LENGTH ERROR: replicate count must match axis length", nullptr, 5, 0);
            return;
        }

        // Calculate new axis length: negative counts contribute |c| fill positions
        int new_axis_len = 0;
        for (int i = 0; i < counts.size(); ++i) {
            int c = static_cast<int>(counts(i));
            new_axis_len += std::abs(c);
        }

        // Build result shape
        std::vector<int> result_shape = shape;
        result_shape[ax] = new_axis_len;

        int result_size = 1;
        for (int d : result_shape) result_size *= d;

        if (result_size == 0) {
            Eigen::VectorXd empty(0);
            m->result = m->heap->allocate_vector(empty, is_char);
            return;
        }

        // Build mapping: for each position along result axis,
        // store source index or -1 for fill
        std::vector<int> axis_map(new_axis_len);
        int pos = 0;
        for (int i = 0; i < axis_len; ++i) {
            int c = static_cast<int>(counts(i));
            if (c >= 0) {
                for (int r = 0; r < c; ++r) axis_map[pos++] = i;
            } else {
                for (int r = 0; r < -c; ++r) axis_map[pos++] = -1;  // fill
            }
        }

        double fill = is_char ? 32.0 : 0.0;

        // Compute strides for source and result
        std::vector<int> src_strides(rank), res_strides(rank);
        src_strides[rank - 1] = 1;
        res_strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            src_strides[d] = src_strides[d + 1] * shape[d + 1];
            res_strides[d] = res_strides[d + 1] * result_shape[d + 1];
        }

        Eigen::VectorXd result(result_size);

        // Iterate through result positions
        std::vector<int> res_idx(rank, 0);
        for (int lin = 0; lin < result_size; ++lin) {
            // Decompose linear index to result indices
            int tmp = lin;
            for (int d = 0; d < rank; ++d) {
                res_idx[d] = tmp / res_strides[d];
                tmp %= res_strides[d];
            }

            int src_ax_idx = axis_map[res_idx[ax]];
            if (src_ax_idx < 0) {
                result(lin) = fill;
            } else {
                // Compute source linear index
                int src_lin = 0;
                for (int d = 0; d < rank; ++d) {
                    int idx = (d == ax) ? src_ax_idx : res_idx[d];
                    src_lin += idx * src_strides[d];
                }
                result(lin) = (*nd->data)(src_lin);
            }
        }

        // Allocate result based on rank
        if (result_shape.size() == 1) {
            m->result = m->heap->allocate_vector(result, is_char);
        } else if (result_shape.size() == 2) {
            Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
            for (int i = 0; i < result_shape[0]; ++i) {
                for (int j = 0; j < result_shape[1]; ++j) {
                    mat(i, j) = result(i * result_shape[1] + j);
                }
            }
            m->result = m->heap->allocate_matrix(mat, is_char);
        } else {
            m->result = m->heap->allocate_ndarray(std::move(result), std::move(result_shape));
        }
        return;
    }

    // Determine axis for matrix operations (default is last axis)
    int k = 0;  // 0 means use default (last axis)
    if (axis) {
        int max_rank = rhs->is_vector() ? 1 : (rhs->is_matrix() ? 2 : 0);
        if (max_rank == 0) {
            m->throw_error("AXIS ERROR: replicate with axis requires array argument", m->control, 4, 0);
            return;
        }
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be a scalar", m->control, 4, 0);
            return;
        }
        k = static_cast<int>(axis->as_scalar()) - static_cast<int>(m->io);
        if (k < 0 || k >= max_rank) {
            m->throw_error("AXIS ERROR: axis out of bounds", m->control, 4, 0);
            return;
        }
        k += 1;  // Convert to 1-based internal representation
    }

    // Matrix case: replicate along specified axis
    if (!rhs->is_scalar() && !rhs->is_vector()) {
        const Eigen::MatrixXd* mat = rhs->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();

        // Default axis: last (k=2 means columns)
        if (k == 0) k = 2;

        if (k == 1) {
            // Replicate along first axis (rows)
            if (counts.size() != rows) {
                m->throw_error("LENGTH ERROR: replicate count must match array length", nullptr, 5, 0);
                return;
            }

            // Calculate total output rows (negative counts contribute |c| fill rows)
            int total_rows = 0;
            for (int i = 0; i < counts.size(); ++i) {
                total_rows += std::abs(static_cast<int>(counts(i)));
            }

            if (total_rows == 0) {
                Eigen::VectorXd empty(0);
                m->result = m->heap->allocate_vector(empty, is_char);
                return;
            }

            double fill = is_char ? 32.0 : 0.0;
            Eigen::MatrixXd result(total_rows, cols);
            int out_row = 0;
            for (int i = 0; i < rows; ++i) {
                int c = static_cast<int>(counts(i));
                if (c >= 0) {
                    for (int r = 0; r < c; ++r) {
                        result.row(out_row++) = mat->row(i);
                    }
                } else {
                    for (int r = 0; r < -c; ++r) {
                        result.row(out_row++).setConstant(fill);
                    }
                }
            }

            m->result = m->heap->allocate_matrix(result, is_char);
            return;
        }

        // k == 2: Replicate along last axis (columns)
        if (counts.size() != cols) {
            m->throw_error("LENGTH ERROR: replicate count must match array length", nullptr, 5, 0);
            return;
        }

        // Calculate total output columns (negative counts contribute |c| fill columns)
        int total_cols = 0;
        for (int i = 0; i < counts.size(); ++i) {
            total_cols += std::abs(static_cast<int>(counts(i)));
        }

        if (total_cols == 0) {
            Eigen::VectorXd empty(0);
            m->result = m->heap->allocate_vector(empty, is_char);
            return;
        }

        double fill = is_char ? 32.0 : 0.0;
        Eigen::MatrixXd result(rows, total_cols);
        int out_col = 0;
        for (int j = 0; j < cols; ++j) {
            int c = static_cast<int>(counts(j));
            if (c >= 0) {
                for (int r = 0; r < c; ++r) {
                    result.col(out_col++) = mat->col(j);
                }
            } else {
                for (int r = 0; r < -c; ++r) {
                    result.col(out_col++).setConstant(fill);
                }
            }
        }

        m->result = m->heap->allocate_matrix(result, is_char);
        return;
    }

    // Scalar or vector case
    Eigen::VectorXd data = flatten_value(rhs);

    // Scalar extension: if lhs is scalar, extend to match rhs length
    if (counts.size() == 1 && data.size() > 1) {
        double scalar_count = counts(0);
        counts.resize(data.size());
        counts.setConstant(scalar_count);
    }

    if (counts.size() != data.size()) {
        m->throw_error("LENGTH ERROR: replicate count must match array length", nullptr, 5, 0);
        return;
    }

    // Calculate total output size (negative counts contribute |c| fill positions)
    int total = 0;
    for (int i = 0; i < counts.size(); ++i) {
        total += std::abs(static_cast<int>(counts(i)));
    }

    if (total == 0) {
        // Empty result
        Eigen::VectorXd empty(0);
        m->result = m->heap->allocate_vector(empty, is_char);
        return;
    }

    // Build result: positive counts replicate, negative counts insert fill
    double fill = is_char ? 32.0 : 0.0;
    Eigen::VectorXd result(total);
    int out_idx = 0;
    for (int i = 0; i < data.size(); ++i) {
        int c = static_cast<int>(counts(i));
        if (c >= 0) {
            for (int r = 0; r < c; ++r) {
                result(out_idx++) = data(i);
            }
        } else {
            for (int r = 0; r < -c; ++r) {
                result(out_idx++) = fill;
            }
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

    // ISO 13751 10.1.8: Unique requires vector or scalar
    if (omega->is_matrix()) {
        m->throw_error("RANK ERROR: unique requires vector or scalar argument", nullptr, 4, 0);
        return;
    }

    if (omega->is_scalar()) {
        m->result = omega;
        return;
    }

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);
    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ∪ requires array argument", nullptr, 11, 0);
        return;
    }
    bool is_char = omega->is_char_data();
    const Eigen::MatrixXd* mat = omega->as_matrix();
    int n = mat->rows();

    // Get comparison tolerance
    double ct = m->ct;

    // First pass: count unique elements using tolerant equality
    int unique_count = 0;
    for (int i = 0; i < n; ++i) {
        bool found = false;
        for (int j = 0; j < i; ++j) {
            if (tolerant_eq((*mat)(j, 0), (*mat)(i, 0), ct)) {
                found = true;
                break;
            }
        }
        if (!found) unique_count++;
    }

    if (unique_count == 0) {
        m->result = m->heap->allocate_vector(Eigen::VectorXd(0), is_char);
        return;
    }

    // Second pass: collect unique elements
    Eigen::VectorXd result(unique_count);
    int out_idx = 0;
    for (int i = 0; i < n; ++i) {
        bool found = false;
        for (int j = 0; j < out_idx; ++j) {
            if (tolerant_eq(result(j), (*mat)(i, 0), ct)) {
                found = true;
                break;
            }
        }
        if (!found) {
            result(out_idx++) = (*mat)(i, 0);
        }
    }

    m->result = m->heap->allocate_vector(result, is_char);
}

// Union (∪ dyadic) - unique elements from both arrays (left first, then unique from right)
// 1 2 3 ∪ 3 4 5 → 1 2 3 4 5
void fn_union(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (!lhs->is_scalar() && !lhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ∪ requires array argument", nullptr, 11, 0);
        return;
    }
    if (!rhs->is_scalar() && !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ∪ requires array argument", nullptr, 11, 0);
        return;
    }

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

    if (!lhs->is_scalar() && !lhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ~ requires array argument", nullptr, 11, 0);
        return;
    }
    if (!rhs->is_scalar() && !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ~ requires array argument", nullptr, 11, 0);
        return;
    }

    Eigen::VectorXd left = flatten_value(lhs);
    Eigen::VectorXd right = flatten_value(rhs);
    double ct = m->ct;

    // First pass: count elements not in right (tolerant comparison per ISO 10.2.16)
    int count = 0;
    for (int i = 0; i < left.size(); ++i) {
        bool in_right = false;
        for (int j = 0; j < right.size(); ++j) {
            if (tolerant_eq(left(i), right(j), ct)) {
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
            if (tolerant_eq(left(i), right(j), ct)) {
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
        double val = omega->data.scalar;
        int n = static_cast<int>(val);
        if (n != val) {
            m->throw_error("DOMAIN ERROR: ? argument must be integer", nullptr, 11, 0);
            return;
        }
        if (n <= 0) {
            m->throw_error("DOMAIN ERROR: ? argument must be positive", nullptr, 11, 0);
            return;
        }
        std::uniform_int_distribution<int> dist(io, n - 1 + io);  // ⎕IO
        m->result = m->heap->allocate_scalar(static_cast<double>(dist(m->rng)));
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ? requires numeric argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result(mat->rows(), mat->cols());

    for (int i = 0; i < mat->size(); ++i) {
        double val = mat->data()[i];
        int n = static_cast<int>(val);
        if (n != val) {
            m->throw_error("DOMAIN ERROR: roll argument must be integer", nullptr, 11, 0);
            return;
        }
        if (n <= 0) {
            m->throw_error("DOMAIN ERROR: roll argument must be positive", nullptr, 11, 0);
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
        m->throw_error("DOMAIN ERROR: deal arguments must be scalars", nullptr, 11, 0);
        return;
    }

    double lval = lhs->data.scalar;
    double rval = rhs->data.scalar;

    // Check for integer values using ⎕CT
    if (std::abs(lval - std::round(lval)) > m->ct) {
        m->throw_error("DOMAIN ERROR: deal count must be integer", nullptr, 11, 0);
        return;
    }
    if (std::abs(rval - std::round(rval)) > m->ct) {
        m->throw_error("DOMAIN ERROR: deal range must be integer", nullptr, 11, 0);
        return;
    }

    int a = static_cast<int>(std::round(lval));
    int b = static_cast<int>(std::round(rval));

    if (a < 0 || b <= 0) {
        m->throw_error("DOMAIN ERROR: deal arguments must be positive", nullptr, 11, 0);
        return;
    }

    if (a > b) {
        m->throw_error("DOMAIN ERROR: cannot deal more values than range", nullptr, 11, 0);
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
            m->throw_error("DOMAIN ERROR: expand mask must be boolean", nullptr, 11, 0);
            return;
        }
        if (val == 1) ones_count++;
    }

    // Typical element: blank for char, zero for numeric (ISO 13751 §5.3.2)
    double fill = is_char ? 32.0 : 0.0;

    // Strand case: expand top-level elements with fill (scalar 0)
    if (rhs->is_strand()) {
        REJECT_AXIS(m, axis);
        std::vector<Value*>* strand = rhs->as_strand();
        int n = static_cast<int>(strand->size());

        if (ones_count != n) {
            m->throw_error("LENGTH ERROR: expand mask ones must match strand length", nullptr, 5, 0);
            return;
        }

        std::vector<Value*> result;
        result.reserve(mask.size());
        int src_idx = 0;
        Value* strand_fill = m->heap->allocate_scalar(0.0);  // Numeric fill for strands
        for (int i = 0; i < mask.size(); ++i) {
            if (static_cast<int>(mask(i)) == 1) {
                result.push_back((*strand)[src_idx++]);
            } else {
                result.push_back(strand_fill);
            }
        }

        m->result = m->heap->allocate_strand(result);
        return;
    }

    // NDARRAY case: expand along specified axis (default: last)
    if (rhs->is_ndarray()) {
        const Value::NDArrayData* nd = rhs->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Determine axis (default: last)
        int ax = rank - 1;
        if (axis) {
            if (!axis->is_scalar()) {
                m->throw_error("AXIS ERROR: axis must be scalar", m->control, 4, 0);
                return;
            }
            ax = static_cast<int>(axis->as_scalar()) - m->io;
            if (ax < 0 || ax >= rank) {
                m->throw_error("AXIS ERROR: axis out of bounds", m->control, 4, 0);
                return;
            }
        }

        int axis_len = shape[ax];

        // Number of 1s in mask must equal axis length
        if (ones_count != axis_len) {
            m->throw_error("LENGTH ERROR: expand mask ones must match axis length", nullptr, 5, 0);
            return;
        }

        // Build result shape - axis dimension becomes mask length
        std::vector<int> result_shape = shape;
        result_shape[ax] = static_cast<int>(mask.size());

        int result_size = 1;
        for (int d : result_shape) result_size *= d;

        // Compute strides for source and result
        std::vector<int> src_strides(rank), res_strides(rank);
        src_strides[rank - 1] = 1;
        res_strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            src_strides[d] = src_strides[d + 1] * shape[d + 1];
            res_strides[d] = res_strides[d + 1] * result_shape[d + 1];
        }

        Eigen::VectorXd result(result_size);

        // Iterate through result positions
        std::vector<int> res_idx(rank, 0);
        for (int lin = 0; lin < result_size; ++lin) {
            // Decompose linear index to result indices
            int tmp = lin;
            for (int d = 0; d < rank; ++d) {
                res_idx[d] = tmp / res_strides[d];
                tmp %= res_strides[d];
            }

            int res_ax_idx = res_idx[ax];
            if (static_cast<int>(mask(res_ax_idx)) == 0) {
                // Fill element
                result(lin) = fill;
            } else {
                // Map result axis index to source axis index
                int src_ax_idx = 0;
                int ones_seen = 0;
                for (int i = 0; i <= res_ax_idx; ++i) {
                    if (static_cast<int>(mask(i)) == 1) {
                        if (i == res_ax_idx) {
                            src_ax_idx = ones_seen;
                            break;
                        }
                        ones_seen++;
                    }
                }

                // Compute source linear index
                int src_lin = 0;
                for (int d = 0; d < rank; ++d) {
                    int idx = (d == ax) ? src_ax_idx : res_idx[d];
                    src_lin += idx * src_strides[d];
                }

                result(lin) = (*nd->data)(src_lin);
            }
        }

        // Allocate result based on rank
        if (result_shape.size() == 1) {
            m->result = m->heap->allocate_vector(result, is_char);
        } else if (result_shape.size() == 2) {
            Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
            for (int i = 0; i < result_shape[0]; ++i) {
                for (int j = 0; j < result_shape[1]; ++j) {
                    mat(i, j) = result(i * result_shape[1] + j);
                }
            }
            m->result = m->heap->allocate_matrix(mat, is_char);
        } else {
            m->result = m->heap->allocate_ndarray(std::move(result), std::move(result_shape));
        }
        return;
    }

    // Determine axis for matrix operations (default is last axis)
    int k = 0;  // 0 means use default (last axis)
    if (axis) {
        int max_rank = rhs->is_vector() ? 1 : (rhs->is_matrix() ? 2 : 0);
        if (max_rank == 0 && !rhs->is_scalar()) {
            m->throw_error("AXIS ERROR: expand with axis requires array argument", m->control, 4, 0);
            return;
        }
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be a scalar", m->control, 4, 0);
            return;
        }
        k = static_cast<int>(axis->as_scalar()) - static_cast<int>(m->io);
        if (k < 0 || k >= max_rank) {
            m->throw_error("AXIS ERROR: axis out of bounds", m->control, 4, 0);
            return;
        }
        k += 1;  // Convert to 1-based internal representation
    }

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
        // Matrix case: expand along specified axis
        const Eigen::MatrixXd* mat = rhs->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();

        // Default axis: last (k=2 means columns)
        if (k == 0) k = 2;

        if (k == 1) {
            // Expand along first axis (rows)
            if (ones_count != rows) {
                m->throw_error("LENGTH ERROR: expand mask ones must match array length", nullptr, 5, 0);
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
            return;
        }

        // k == 2: Expand along last axis (columns)
        if (ones_count != cols) {
            m->throw_error("LENGTH ERROR: expand mask ones must match array length", nullptr, 5, 0);
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
        m->throw_error("LENGTH ERROR: expand mask ones must match array length", nullptr, 5, 0);
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
            m->throw_error("DOMAIN ERROR: expand mask must be boolean", nullptr, 11, 0);
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
        m->throw_error("LENGTH ERROR: expand mask ones must match array length", nullptr, 5, 0);
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

    // Reject character data
    if (lhs->is_string() || lhs->is_char_data()) {
        m->throw_error("DOMAIN ERROR: decode requires numeric arguments", nullptr, 11, 0);
        return;
    }
    if (rhs->is_string() || rhs->is_char_data()) {
        m->throw_error("DOMAIN ERROR: decode requires numeric arguments", nullptr, 11, 0);
        return;
    }

    // Helper: Horner's method for decode
    auto horner = [](const Eigen::VectorXd& radix, const Eigen::VectorXd& digits) -> double {
        if (digits.size() == 0) return 0.0;
        double z = digits(0);
        for (int i = 1; i < digits.size(); ++i) {
            z = radix(i) * z + digits(i);
        }
        return z;
    };

    // Fast path: scalar radix with scalar digit → just return the digit
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    // Matrix case: A⊥B where A is m×n and B is n×p → result is m×p
    if (lhs->is_matrix() && rhs->is_matrix()) {
        const Eigen::MatrixXd* A = lhs->as_matrix();
        const Eigen::MatrixXd* B = rhs->as_matrix();
        int m_rows = A->rows();
        int n = A->cols();
        int p = B->cols();

        if (B->rows() != n) {
            m->throw_error("LENGTH ERROR: decode inner dimensions must match", nullptr, 5, 0);
            return;
        }

        Eigen::MatrixXd result(m_rows, p);
        for (int i = 0; i < m_rows; ++i) {
            Eigen::VectorXd radix = A->row(i).transpose();
            for (int j = 0; j < p; ++j) {
                Eigen::VectorXd digits = B->col(j);
                result(i, j) = horner(radix, digits);
            }
        }
        m->result = m->heap->allocate_matrix(result);
        return;
    }

    // Validate arguments before flattening
    if (!lhs->is_scalar() && !lhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⊥ requires array argument", nullptr, 11, 0);
        return;
    }
    if (!rhs->is_scalar() && !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⊥ requires array argument", nullptr, 11, 0);
        return;
    }

    // Get the radix and digits as vectors
    Eigen::VectorXd radix = flatten_value(lhs);
    Eigen::VectorXd digits = flatten_value(rhs);

    // Empty radix → 0 (per ISO 13751 spec)
    if (radix.size() == 0) {
        m->result = m->heap->allocate_scalar(0.0);
        return;
    }

    int n = digits.size();

    // If radix is scalar, extend it to match digits length
    if (radix.size() == 1) {
        double r = radix(0);
        radix.resize(n);
        radix.setConstant(r);
    }

    // Radix and digits must have same length after extension
    if (radix.size() != n) {
        m->throw_error("LENGTH ERROR: decode radix and digits must have same length", nullptr, 5, 0);
        return;
    }

    if (n == 0) {
        m->result = m->heap->allocate_scalar(0.0);
        return;
    }

    m->result = m->heap->allocate_scalar(horner(radix, digits));
}

// Encode (⊤ dyadic) - representation / convert to digits in radix
// A⊤B converts B to representation in radix A
// 2 2 2 2⊤11 → 1 0 1 1 (decimal to binary)
// 10 10 10⊤345 → 3 4 5 (decimal digits)
// 24 60 60⊤5445 → 1 30 45 (seconds to hours:mins:secs)
void fn_encode(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);

    // Reject character data
    if (lhs->is_string() || lhs->is_char_data()) {
        m->throw_error("DOMAIN ERROR: encode requires numeric arguments", nullptr, 11, 0);
        return;
    }
    if (rhs->is_string() || rhs->is_char_data()) {
        m->throw_error("DOMAIN ERROR: encode requires numeric arguments", nullptr, 11, 0);
        return;
    }

    // Validate arguments before flattening
    if (!lhs->is_scalar() && !lhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⊤ requires array argument", nullptr, 11, 0);
        return;
    }
    if (!rhs->is_scalar() && !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⊤ requires array argument", nullptr, 11, 0);
        return;
    }

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
        m->throw_error("RANK ERROR: cannot index scalar", nullptr, 4, 0);
        return;
    }
    if (!array->is_array() && !array->is_strand()) {
        m->throw_error("DOMAIN ERROR: cannot index non-array value", nullptr, 11, 0);
        return;
    }

    // Helper lambda to validate index is near-integer
    auto validate_index = [m](double val) -> bool {
        double rounded = std::round(val);
        if (std::abs(val - rounded) > 1e-10) {
            m->throw_error("DOMAIN ERROR: index must be integer", nullptr, 11, 0);
            return false;
        }
        return true;
    };

    // Helper to check if index is elided (empty vector = select all)
    auto is_elided = [](Value* idx) -> bool {
        if (idx->is_vector() && idx->size() == 0) return true;
        return false;
    };

    // Helper to get indices as vector (handles scalar, vector, elided)
    // Returns empty vector and sets had_error=true if invalid
    bool had_error = false;
    auto get_index_vector = [&](Value* idx, int max_dim) -> std::vector<int> {
        std::vector<int> result;
        if (is_elided(idx)) {
            // Elided: all indices 0..max_dim-1
            result.reserve(max_dim);
            for (int i = 0; i < max_dim; i++) {
                result.push_back(i);
            }
        } else if (idx->is_scalar()) {
            double val = idx->as_scalar();
            if (!validate_index(val)) { had_error = true; return {}; }
            int i = static_cast<int>(std::round(val)) - m->io;
            if (i < 0 || i >= max_dim) {
                m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
                had_error = true;
                return {};
            }
            result.push_back(i);
        } else if (idx->is_array()) {
            const Eigen::MatrixXd* mat = idx->as_matrix();
            result.reserve(mat->rows());
            for (int j = 0; j < mat->rows(); j++) {
                double val = (*mat)(j, 0);
                if (!validate_index(val)) { had_error = true; return {}; }
                int i = static_cast<int>(std::round(val)) - m->io;
                if (i < 0 || i >= max_dim) {
                    m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
                    had_error = true;
                    return {};
                }
                result.push_back(i);
            }
        } else {
            m->throw_error("DOMAIN ERROR: index must be numeric", nullptr, 11, 0);
            had_error = true;
            return {};
        }
        return result;
    };

    // Multi-axis indexing: strand of indices for A[I;J;...]
    if (indices->is_strand()) {
        auto* idx_strand = indices->as_strand();
        size_t num_axes = idx_strand->size();

        if (!array->is_array()) {
            m->throw_error("RANK ERROR: multi-axis indexing requires array", nullptr, 4, 0);
            return;
        }

        // NDARRAY indexing: A[I;J;K;...] for rank 3+
        if (array->is_ndarray()) {
            const auto& shape = array->ndarray_shape();
            const auto& strides = array->ndarray_strides();
            const Eigen::VectorXd* data = array->ndarray_data();

            if (num_axes != shape.size()) {
                m->throw_error("RANK ERROR: index count doesn't match array rank", nullptr, 4, 0);
                return;
            }

            // Get indices for each axis
            std::vector<std::vector<int>> axis_indices(num_axes);
            for (size_t ax = 0; ax < num_axes; ++ax) {
                axis_indices[ax] = get_index_vector((*idx_strand)[ax], shape[ax]);
                if (had_error) return;
            }

            // Compute result shape (axes with 1 index become scalar, dropped from shape)
            std::vector<int> result_shape;
            for (size_t ax = 0; ax < num_axes; ++ax) {
                if (axis_indices[ax].size() > 1) {
                    result_shape.push_back(static_cast<int>(axis_indices[ax].size()));
                }
            }

            // Compute total result size
            int result_size = 1;
            for (int dim : result_shape) {
                result_size *= dim;
            }

            // Single element result → scalar
            if (result_size == 1) {
                int linear = 0;
                for (size_t ax = 0; ax < num_axes; ++ax) {
                    linear += axis_indices[ax][0] * strides[ax];
                }
                m->result = m->heap->allocate_scalar((*data)(linear));
                return;
            }

            // Collect result elements
            Eigen::VectorXd result_data(result_size);
            std::vector<int> current(num_axes, 0);  // Current position in each axis_indices

            for (int i = 0; i < result_size; ++i) {
                // Compute linear index in source array
                int linear = 0;
                for (size_t ax = 0; ax < num_axes; ++ax) {
                    linear += axis_indices[ax][current[ax]] * strides[ax];
                }
                result_data(i) = (*data)(linear);

                // Advance position (row-major order, last axis varies fastest)
                for (int ax = static_cast<int>(num_axes) - 1; ax >= 0; --ax) {
                    current[ax]++;
                    if (current[ax] < static_cast<int>(axis_indices[ax].size())) {
                        break;
                    }
                    current[ax] = 0;
                }
            }

            // Create result based on shape
            int result_rank = static_cast<int>(result_shape.size());
            if (result_rank == 0) {
                m->result = m->heap->allocate_scalar(result_data(0));
            } else if (result_rank == 1) {
                m->result = m->heap->allocate_vector(result_data, is_char);
            } else if (result_rank == 2) {
                Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
                for (int i = 0; i < result_size; ++i) {
                    mat(i / result_shape[1], i % result_shape[1]) = result_data(i);
                }
                m->result = m->heap->allocate_matrix(mat, is_char);
            } else {
                m->result = m->heap->allocate_ndarray(std::move(result_data), std::move(result_shape));
                m->result->set_char_data(is_char);
            }
            return;
        }

        // Matrix indexing (rank 2)
        const Eigen::MatrixXd* arr = array->as_matrix();
        int rows = arr->rows();
        int cols = arr->cols();
        bool is_vec = (cols == 1);

        if (num_axes != 2) {
            m->throw_error("RANK ERROR: matrix requires exactly 2 indices", nullptr, 4, 0);
            return;
        }

        Value* row_idx = (*idx_strand)[0];
        Value* col_idx = (*idx_strand)[1];

        // Get row and column indices
        std::vector<int> row_indices = get_index_vector(row_idx, rows);
        if (had_error) return;
        std::vector<int> col_indices = get_index_vector(col_idx, cols);
        if (had_error) return;

        int result_rows = static_cast<int>(row_indices.size());
        int result_cols = static_cast<int>(col_indices.size());

        if (result_rows == 1 && result_cols == 1) {
            // Scalar result: M[i;j]
            m->result = m->heap->allocate_scalar((*arr)(row_indices[0], col_indices[0]));
        } else if (result_rows == 1) {
            // Row selection, single row: M[i;] or M[i;j k l] → vector
            Eigen::VectorXd result(result_cols);
            for (int c = 0; c < result_cols; c++) {
                result(c) = (*arr)(row_indices[0], col_indices[c]);
            }
            m->result = m->heap->allocate_vector(result, is_char);
        } else if (result_cols == 1) {
            // Column selection, single column: M[;j] or M[i k l;j] → vector
            Eigen::VectorXd result(result_rows);
            for (int r = 0; r < result_rows; r++) {
                result(r) = (*arr)(row_indices[r], col_indices[0]);
            }
            m->result = m->heap->allocate_vector(result, is_char);
        } else {
            // Matrix result: M[i j;k l]
            Eigen::MatrixXd result(result_rows, result_cols);
            for (int r = 0; r < result_rows; r++) {
                for (int c = 0; c < result_cols; c++) {
                    result(r, c) = (*arr)(row_indices[r], col_indices[c]);
                }
            }
            m->result = m->heap->allocate_matrix(result, is_char);
        }
        return;
    }

    // Strand indexing: S[I] selects elements from strand
    if (array->is_strand()) {
        std::vector<Value*>* strand = array->as_strand();
        int len = static_cast<int>(strand->size());

        if (indices->is_scalar()) {
            double idx_val = indices->as_scalar();
            if (!validate_index(idx_val)) return;
            int idx = static_cast<int>(std::round(idx_val)) - m->io;
            if (idx < 0 || idx >= len) {
                m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
                return;
            }
            m->result = (*strand)[idx];
        } else if (indices->is_array()) {
            const Eigen::MatrixXd* idx_mat = indices->as_matrix();
            int n = idx_mat->rows();
            std::vector<Value*> result;
            result.reserve(n);
            for (int i = 0; i < n; i++) {
                double idx_val = (*idx_mat)(i, 0);
                if (!validate_index(idx_val)) return;
                int idx = static_cast<int>(std::round(idx_val)) - m->io;
                if (idx < 0 || idx >= len) {
                    m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
                    return;
                }
                result.push_back((*strand)[idx]);
            }
            m->result = m->heap->allocate_strand(std::move(result));
        } else {
            m->throw_error("DOMAIN ERROR: index must be numeric", nullptr, 11, 0);
        }
        return;
    }

    // NDARRAY single-axis indexing (linear indexing into ravel)
    if (array->is_ndarray()) {
        const Eigen::VectorXd* data = array->ndarray_data();
        int arr_size = array->size();

        if (indices->is_scalar()) {
            double idx_val = indices->as_scalar();
            if (!validate_index(idx_val)) return;
            int idx = static_cast<int>(std::round(idx_val)) - m->io;
            if (idx < 0 || idx >= arr_size) {
                m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
                return;
            }
            m->result = m->heap->allocate_scalar((*data)(idx));
        } else if (indices->is_array()) {
            const Eigen::MatrixXd* idx_mat = indices->as_matrix();
            int n = idx_mat->rows();
            Eigen::VectorXd result(n);
            for (int i = 0; i < n; i++) {
                double idx_val = (*idx_mat)(i, 0);
                if (!validate_index(idx_val)) return;
                int idx = static_cast<int>(std::round(idx_val)) - m->io;
                if (idx < 0 || idx >= arr_size) {
                    m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
                    return;
                }
                result(i) = (*data)(idx);
            }
            m->result = m->heap->allocate_vector(result, is_char);
        } else {
            m->throw_error("DOMAIN ERROR: index must be numeric", nullptr, 11, 0);
        }
        return;
    }

    // Single-axis indexing for vector/matrix
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
                m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
                return;
            }
            m->result = m->heap->allocate_scalar((*arr)(idx, 0));
        } else {
            // Linear indexing into matrix (row-major order)
            int size = rows * cols;
            if (idx < 0 || idx >= size) {
                m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
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
                    m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
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
                    m->throw_error("INDEX ERROR: index out of bounds", nullptr, 3, 0);
                    return;
                }
                result.row(i) = arr->row(idx);
            }
            m->result = m->heap->allocate_matrix(result, is_char);
        }
    } else {
        m->throw_error("DOMAIN ERROR: index must be numeric", nullptr, 11, 0);
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

    // Strand: rank 1, so n×1 result - stays as strand (nested data)
    if (omega->is_strand()) {
        // Already 1D, Table conceptually makes it n×1 but we keep as strand
        m->result = omega;
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍪ requires array argument", nullptr, 11, 0);
        return;
    }

    // NDARRAY: shape s1×s2×...×sn → s1 × (s2×s3×...×sn) matrix
    // ISO 13751 §8.2.4: first-item × product-of-rest
    if (omega->is_ndarray()) {
        const Value::NDArrayData* nd = omega->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int first_dim = shape[0];
        int rest_product = 1;
        for (size_t i = 1; i < shape.size(); ++i) {
            rest_product *= shape[i];
        }
        Eigen::MatrixXd result(first_dim, rest_product);
        // Data is already in row-major order, just reshape
        for (int i = 0; i < first_dim; ++i) {
            for (int j = 0; j < rest_product; ++j) {
                result(i, j) = (*nd->data)(i * rest_product + j);
            }
        }
        m->result = m->heap->allocate_matrix(result);
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();

    if (omega->is_vector()) {
        // Vector → n×1 matrix
        Eigen::MatrixXd result(rows, 1);
        for (int i = 0; i < rows; ++i) {
            result(i, 0) = (*mat)(i, 0);
        }
        m->result = m->heap->allocate_matrix(result);
        return;
    }

    // Matrix → same matrix (already 2D)
    m->result = m->heap->allocate_matrix(*mat);
}

// Helper: compute depth recursively
static int compute_depth(Value* v) {
    if (v->is_scalar()) {
        return 0;
    }
    if (v->is_strand()) {
        // Depth is 1 + max depth of elements
        std::vector<Value*>* strand = v->as_strand();
        int max_depth = 0;
        for (Value* elem : *strand) {
            int d = compute_depth(elem);
            if (d > max_depth) max_depth = d;
        }
        return 1 + max_depth;
    }
    // Simple arrays (vectors, matrices, strings) have depth 1
    return 1;
}

// Depth (≡ monadic) - nesting level of array
// ISO 13751 Section 8.2.5: simple-scalar → 0, array → 1 + max depth of elements
void fn_depth(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);
    m->result = m->heap->allocate_scalar(static_cast<double>(compute_depth(omega)));
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
    // NDARRAY: delegate to fn_catenate with axis=first (⎕IO)
    if (alpha->is_ndarray() || omega->is_ndarray()) {
        Value* first_axis = m->heap->allocate_scalar(static_cast<double>(m->io));
        fn_catenate(m, first_axis, alpha, omega);
        return;
    }

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

    // Validate both arguments are arrays (or scalars which are handled below)
    if (!alpha->is_scalar() && !alpha->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍪ requires array argument", nullptr, 11, 0);
        return;
    }
    if (!omega->is_scalar() && !omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍪ requires array argument", nullptr, 11, 0);
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
        m->throw_error("LENGTH ERROR: incompatible shapes for ⍪", nullptr, 5, 0);
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
        m->throw_error("DOMAIN ERROR: execute requires a string", nullptr, 11, 0);
        return;
    }

    const char* code = str_val->as_string()->c_str();

    // Empty string returns zilde (empty numeric vector)
    if (code[0] == '\0') {
        m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
        return;
    }

    Continuation* k = m->parser->parse(code);

    if (!k) {
        // Parse error
        m->throw_error(m->parser->get_error().c_str(), m->control, 1, 0);
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
    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍕ requires array argument", nullptr, 11, 0);
        return;
    }
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

    // Matrix: newline-separated rows with aligned columns
    // ISO 13751 §15.4.1: columns should be aligned

    // First pass: format all elements and find max width per column
    std::vector<std::vector<std::string>> formatted(rows, std::vector<std::string>(cols));
    std::vector<size_t> col_widths(cols, 0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            formatted[i][j] = format_number_pp((*mat)(i, j), m->pp);
            // Count UTF-8 characters, not bytes
            size_t char_count = 0;
            for (size_t k = 0; k < formatted[i][j].length(); ) {
                unsigned char c = formatted[i][j][k];
                if ((c & 0x80) == 0) { k += 1; }
                else if ((c & 0xE0) == 0xC0) { k += 2; }
                else if ((c & 0xF0) == 0xE0) { k += 3; }
                else { k += 4; }
                char_count++;
            }
            if (char_count > col_widths[j]) {
                col_widths[j] = char_count;
            }
        }
    }

    // Second pass: output with padding for alignment
    std::ostringstream oss;
    for (int i = 0; i < rows; i++) {
        if (i > 0) oss << "\n";
        for (int j = 0; j < cols; j++) {
            if (j > 0) oss << " ";
            // Right-justify: pad with spaces on left
            size_t char_count = 0;
            for (size_t k = 0; k < formatted[i][j].length(); ) {
                unsigned char c = formatted[i][j][k];
                if ((c & 0x80) == 0) { k += 1; }
                else if ((c & 0xE0) == 0xC0) { k += 2; }
                else if ((c & 0xF0) == 0xE0) { k += 3; }
                else { k += 4; }
                char_count++;
            }
            for (size_t p = char_count; p < col_widths[j]; p++) {
                oss << " ";
            }
            oss << formatted[i][j];
        }
    }
    m->result = m->heap->allocate_string(oss.str().c_str());
}

// Dyadic format: A ⍕ B
void fn_format_dyadic(Machine* m, Value* axis, Value* alpha, Value* omega) {
    REJECT_AXIS(m, axis);
    // A must be numeric
    if (alpha->is_string() || (alpha->is_array() && alpha->is_char_data())) {
        m->throw_error("DOMAIN ERROR: format left argument must be numeric", nullptr, 11, 0);
        return;
    }

    // B must be numeric
    if (omega->is_string() || (omega->is_array() && omega->is_char_data())) {
        m->throw_error("DOMAIN ERROR: format right argument must be numeric", nullptr, 11, 0);
        return;
    }

    // A must be a vector (rank <= 1)
    if (alpha->is_matrix()) {
        m->throw_error("RANK ERROR: format left argument must be a vector", nullptr, 4, 0);
        return;
    }

    // Get format specifications from A
    // ISO 13751 §15.4.2: If any item of A is not a near-integer, signal domain-error
    std::vector<std::pair<int, int>> specs;  // (width, precision) pairs

    // Helper lambda to check if value is a near-integer
    // ISO 13751: use comparison tolerance (⎕CT) for near-integer test
    auto is_near_integer = [m](double v) -> bool {
        double rounded = std::round(v);
        return std::abs(v - rounded) <= m->ct;
    };

    if (alpha->is_scalar()) {
        // Single scalar - interpret as width with 0 decimals
        double v = alpha->as_scalar();
        if (!is_near_integer(v)) {
            m->throw_error("DOMAIN ERROR: format specification must be integer", nullptr, 11, 0);
            return;
        }
        int w = (int)std::round(v);
        specs.push_back({w, 0});
    } else {
        if (!alpha->is_array()) {
            m->throw_error("DOMAIN ERROR: ⍕ requires array argument", nullptr, 11, 0);
            return;
        }
        const Eigen::MatrixXd* a_mat = alpha->as_matrix();
        int a_size = a_mat->rows();

        if (a_size % 2 != 0) {
            m->throw_error("LENGTH ERROR: format left argument must have even length", nullptr, 5, 0);
            return;
        }

        for (int i = 0; i < a_size; i += 2) {
            double w_val = (*a_mat)(i, 0);
            double p_val = (*a_mat)(i + 1, 0);
            if (!is_near_integer(w_val) || !is_near_integer(p_val)) {
                m->throw_error("DOMAIN ERROR: format specification must be integer", nullptr, 11, 0);
                return;
            }
            int w = (int)std::round(w_val);
            int p = (int)std::round(p_val);
            specs.push_back({w, p});
        }
    }

    // Validate width is positive
    for (const auto& spec : specs) {
        if (spec.first <= 0) {
            m->throw_error("DOMAIN ERROR: format width must be positive", nullptr, 11, 0);
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
            m->throw_error("DOMAIN ERROR: format width too narrow", nullptr, 11, 0);
            return;
        }

        m->result = m->heap->allocate_string(formatted.c_str());
        return;
    }

    // Format vector or matrix
    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍕ requires array argument", nullptr, 11, 0);
        return;
    }
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
        m->throw_error("LENGTH ERROR: format specs must match number of columns", nullptr, 5, 0);
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

// ============================================================================
// Enclose and Disclose (ISO 13751 Sections 10.2.22, 10.2.24, 10.2.26)
// ============================================================================

// Enclose (⊂ monadic) - wrap value in a single-element strand (nested scalar)
// ISO 13751 Section 10.2.26: ⊂B creates a scalar containing B
// ISO 13751 Section 10.2.27: ⊂[K]B partitions B along axis K
// Note: If B is a simple-scalar, Z is B (scalars don't get enclosed)
void fn_enclose(Machine* m, Value* axis, Value* omega) {
    // ISO 13751: If B is a simple-scalar, return B unchanged
    if (omega->is_scalar()) {
        m->result = omega;
        return;
    }

    // Handle axis specification - ISO 13751 Section 10.2.27
    if (axis != nullptr) {
        if (!axis->is_scalar()) {
            m->throw_error("AXIS ERROR: axis must be scalar", m->control, 4, 0);
            return;
        }
        int enc_axis = static_cast<int>(axis->as_scalar());

        // Convert string to char vector only when we need array operations
        if (omega->is_string()) omega = omega->to_char_vector(m->heap);

        int rank = omega->rank();
        if (enc_axis < 1 || enc_axis > rank) {
            m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
            return;
        }

        // For vectors: ⊂[1]vec returns strand of scalars
        if (omega->is_vector()) {
            const Eigen::MatrixXd* mat = omega->as_matrix();
            std::vector<Value*> elements;
            for (int i = 0; i < mat->rows(); ++i) {
                elements.push_back(m->heap->allocate_scalar((*mat)(i, 0)));
            }
            m->result = m->heap->allocate_strand(std::move(elements));
            return;
        }

        // For matrices: partition along the specified axis
        if (omega->is_matrix()) {
            const Eigen::MatrixXd* mat = omega->as_matrix();
            std::vector<Value*> elements;
            bool is_char = omega->is_char_data();

            if (enc_axis == 1) {
                // ⊂[1]matrix - each column becomes an element
                for (int j = 0; j < mat->cols(); ++j) {
                    Eigen::VectorXd col = mat->col(j);
                    elements.push_back(m->heap->allocate_vector(col, is_char));
                }
            } else {  // enc_axis == 2
                // ⊂[2]matrix - each row becomes an element
                for (int i = 0; i < mat->rows(); ++i) {
                    Eigen::VectorXd row = mat->row(i).transpose();
                    elements.push_back(m->heap->allocate_vector(row, is_char));
                }
            }
            m->result = m->heap->allocate_strand(std::move(elements));
            return;
        }

        // Strands with axis - partition elements
        if (omega->is_strand()) {
            // For strands, axis 1 means each element becomes enclosed
            std::vector<Value*>* src = omega->as_strand();
            std::vector<Value*> elements;
            for (Value* v : *src) {
                std::vector<Value*> single = {v};
                elements.push_back(m->heap->allocate_strand(std::move(single)));
            }
            m->result = m->heap->allocate_strand(std::move(elements));
            return;
        }

        m->throw_error("DOMAIN ERROR: cannot enclose with axis on this type", nullptr, 11, 0);
        return;
    }

    // No axis - simple enclose (string stays as string, wrapped in strand)
    std::vector<Value*> elements = {omega};
    m->result = m->heap->allocate_strand(std::move(elements));
}

// Disclose (⊃ monadic) - extract first element, unwrap nested array
// Helper: transpose array according to axis permutation
// perm is 0-indexed permutation of axes
static Value* transpose_by_perm(Machine* m, Value* arr, const std::vector<int>& perm) {
    if (!arr->is_matrix() || perm.size() != 2) {
        // For vectors or non-standard cases, just return as-is
        return arr;
    }

    const Eigen::MatrixXd* mat = arr->as_matrix();

    // perm[0]=0, perm[1]=1 means no change
    // perm[0]=1, perm[1]=0 means transpose
    if (perm[0] == 1 && perm[1] == 0) {
        Eigen::MatrixXd transposed = mat->transpose();
        return m->heap->allocate_matrix(transposed, arr->is_char_data());
    }

    return arr;
}

// ISO 13751 Section 10.2.24/25: ⊃B returns the first element of B
// ⊃[K]B specifies which axes of the result are "new" (from nested structure)
void fn_disclose(Machine* m, Value* axis, Value* omega) {
    // First, compute ⊃B (the disclosed value)
    Value* disclosed = nullptr;

    if (omega->is_strand()) {
        std::vector<Value*>* strand = omega->as_strand();
        if (strand->empty()) {
            disclosed = m->heap->allocate_scalar(0.0);
        } else {
            disclosed = (*strand)[0];
        }
    } else if (omega->is_scalar()) {
        disclosed = omega;
    } else if (omega->is_string()) {
        const char* s = omega->as_string()->c_str();
        if (*s) {
            // Convert to char vector (proper UTF-8 decoding) and get first codepoint
            Value* char_vec = omega->to_char_vector(m->heap);
            const Eigen::MatrixXd* mat = char_vec->as_matrix();
            disclosed = m->heap->allocate_scalar((*mat)(0, 0));
        } else {
            disclosed = m->heap->allocate_scalar(32.0);  // Space for empty string
        }
    } else if (omega->is_array()) {
        const Eigen::MatrixXd* mat = omega->as_matrix();
        if (mat->size() == 0) {
            double typical = omega->is_char_data() ? 32.0 : 0.0;
            disclosed = m->heap->allocate_scalar(typical);
        } else if (omega->is_vector()) {
            disclosed = m->heap->allocate_scalar((*mat)(0, 0));
        } else {
            Eigen::VectorXd first_row = mat->row(0);
            disclosed = m->heap->allocate_vector(first_row, omega->is_char_data());
        }
    } else {
        // Functions, operators, etc. - return as-is
        disclosed = omega;
    }

    // Handle axis specification (ISO 13751 Section 10.2.25)
    if (axis != nullptr) {
        int rank = disclosed->rank();

        // Get axis values
        std::vector<int> axis_vals;
        if (axis->is_scalar()) {
            axis_vals.push_back(static_cast<int>(axis->as_scalar()));
        } else if (axis->is_vector()) {
            const Eigen::MatrixXd* mat = axis->as_matrix();
            for (int i = 0; i < mat->rows(); ++i) {
                axis_vals.push_back(static_cast<int>((*mat)(i, 0)));
            }
        } else {
            m->throw_error("AXIS ERROR: axis must be scalar or vector", m->control, 4, 0);
            return;
        }

        // Validate: count of axis values must equal rank of disclosed value
        if (static_cast<int>(axis_vals.size()) != rank) {
            m->throw_error("AXIS ERROR: axis count must equal rank of result", m->control, 4, 0);
            return;
        }

        // Validate: all axis values must be valid and distinct
        std::vector<bool> seen(rank, false);
        for (int a : axis_vals) {
            if (a < 1 || a > rank) {
                m->throw_error("AXIS ERROR: axis value out of range", m->control, 4, 0);
                return;
            }
            if (seen[a - 1]) {
                m->throw_error("AXIS ERROR: duplicate axis value", m->control, 4, 0);
                return;
            }
            seen[a - 1] = true;
        }

        // Build permutation: (((⍳⍴⍴Z)~A),A) means put non-A axes first, then A axes
        // But since axis_vals IS the target positions, we need the inverse permutation
        // axis_vals[i] tells us where axis i should go in the result
        // For transpose, we need: result axis j comes from source axis perm[j]
        std::vector<int> perm(rank);
        for (int i = 0; i < rank; ++i) {
            perm[axis_vals[i] - 1] = i;
        }

        disclosed = transpose_by_perm(m, disclosed, perm);
    }

    m->result = disclosed;
}

// Pick (⊃ dyadic) - index into nested array
// ISO 13751 Section 10.2.22: A⊃B selects element at index A from B
void fn_pick(Machine* m, Value* axis, Value* alpha, Value* omega) {
    REJECT_AXIS(m, axis);

    // Alpha must be a scalar integer or vector of indices
    if (alpha->is_scalar()) {
        int idx = static_cast<int>(alpha->as_scalar());

        // Handle strand
        if (omega->is_strand()) {
            std::vector<Value*>* strand = omega->as_strand();
            int adjusted = idx - m->io;
            if (adjusted < 0 || adjusted >= static_cast<int>(strand->size())) {
                m->throw_error("INDEX ERROR: index out of range", nullptr, 3, 0);
                return;
            }
            m->result = (*strand)[adjusted];
            return;
        }

        // Handle regular arrays
        if (omega->is_vector()) {
            const Eigen::MatrixXd* mat = omega->as_matrix();
            int adjusted = idx - m->io;
            if (adjusted < 0 || adjusted >= mat->rows()) {
                m->throw_error("INDEX ERROR: index out of range", nullptr, 3, 0);
                return;
            }
            m->result = m->heap->allocate_scalar((*mat)(adjusted, 0));
            return;
        }

        if (omega->is_matrix()) {
            m->throw_error("RANK ERROR: scalar pick on matrix requires vector index", nullptr, 4, 0);
            return;
        }
    }

    // Vector of indices for deep pick
    if (alpha->is_vector()) {
        const Eigen::MatrixXd* indices = alpha->as_matrix();
        Value* current = omega;

        for (int i = 0; i < indices->rows(); i++) {
            int idx = static_cast<int>((*indices)(i, 0)) - m->io;

            if (current->is_strand()) {
                std::vector<Value*>* strand = current->as_strand();
                if (idx < 0 || idx >= static_cast<int>(strand->size())) {
                    m->throw_error("INDEX ERROR: index out of range", nullptr, 3, 0);
                    return;
                }
                current = (*strand)[idx];
            } else if (current->is_vector()) {
                const Eigen::MatrixXd* mat = current->as_matrix();
                if (idx < 0 || idx >= mat->rows()) {
                    m->throw_error("INDEX ERROR: index out of range", nullptr, 3, 0);
                    return;
                }
                current = m->heap->allocate_scalar((*mat)(idx, 0));
            } else if (current->is_matrix()) {
                m->throw_error("RANK ERROR: cannot pick from matrix with scalar index", nullptr, 4, 0);
                return;
            } else {
                m->throw_error("RANK ERROR: cannot pick from scalar", nullptr, 4, 0);
                return;
            }
        }

        m->result = current;
        return;
    }

    m->throw_error("DOMAIN ERROR: left argument of ⊃ must be integer index", nullptr, 11, 0);
}

// ============================================================================
// Error Handling System Functions (ISO 13751 §11.5.7-11.6.5)
// Note: ⎕ET and ⎕EM are system variables, implemented in SysVarReadK
// ============================================================================

// Helper: check if a double is a near-integer per ISO 13751
static bool is_near_integer(double x) {
    double rounded = std::round(x);
    return std::fabs(x - rounded) <= INTEGER_TOLERANCE;
}

// ⎕ES - Error Signal (monadic) - ISO 13751 §11.5.7
// Signal an error based on the argument:
//   - 2-element numeric vector {class, subclass}: signal that error type
//   - Character vector: signal unclassified error with that message
//   - 0 0: clear error state without signaling
//   - Empty: conditional no-op
void fn_quad_es(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);

    // ISO 13751: B must be scalar or vector (rank ≤ 1)
    if (omega->is_array() && omega->rank() > 1) {
        m->throw_error("RANK ERROR: ⎕ES argument must be scalar or vector", nullptr, 4, 0);
        return;
    }

    // Empty argument = conditional no-op
    if (omega->is_string()) {
        if (omega->as_string()->empty()) {
            m->result = omega;
            return;
        }
    } else if (omega->is_vector() && omega->size() == 0) {
        m->result = omega;
        return;
    }

    if ((omega->is_vector() || omega->is_scalar()) && omega->size() == 2 && !omega->is_char_data()) {
        // 2-element numeric vector: {class, subclass}
        const Eigen::MatrixXd* mat = omega->as_matrix();
        double class_val = (*mat)(0, 0);
        double subclass_val = (*mat)(1, 0);

        // ISO 13751: elements must be near-integers
        if (!is_near_integer(class_val) || !is_near_integer(subclass_val)) {
            m->throw_error("DOMAIN ERROR: ⎕ES error codes must be integers", nullptr, 11, 0);
            return;
        }

        int error_class = static_cast<int>(std::round(class_val));
        int error_subclass = static_cast<int>(std::round(subclass_val));

        // 0 0 = clear error state, no signal
        if (error_class == 0 && error_subclass == 0) {
            m->event_type[0] = 0;
            m->event_type[1] = 0;
            m->event_message = nullptr;
            m->result = omega;
            return;
        }

        // Signal with specified type, preserving existing message if any
        const char* msg = m->event_message ? m->event_message->c_str() : "";
        m->throw_error(msg, nullptr, error_class, error_subclass);

    } else if (omega->is_string() || (omega->is_vector() && omega->is_char_data())) {
        // Character vector/string: signal unclassified error with this message
        const char* msg = omega->as_string()->c_str();
        m->throw_error(msg, nullptr, 0, 1);  // 0 1 = unclassified

    } else {
        m->throw_error("DOMAIN ERROR: ⎕ES requires 2-element vector or character vector", nullptr, 11, 0);
    }
}

// ⎕ES - Dyadic Event Simulation - ISO 13751 §11.6.5
// A ⎕ES B: A is character (message), B is 2-element integer (class, subclass) or empty
void fn_quad_es_dyadic(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);

    // ISO 13751: A must be scalar or vector (rank ≤ 1)
    if (lhs->is_array() && lhs->rank() > 1) {
        m->throw_error("RANK ERROR: ⎕ES left argument must be scalar or vector", nullptr, 4, 0);
        return;
    }

    // A must be character scalar or vector
    bool lhs_is_char = lhs->is_string() || (lhs->is_array() && lhs->is_char_data());
    if (!lhs_is_char) {
        m->throw_error("DOMAIN ERROR: ⎕ES left argument must be character", nullptr, 11, 0);
        return;
    }

    // ISO 13751: B must be scalar or vector (rank ≤ 1)
    if (rhs->is_array() && rhs->rank() > 1) {
        m->throw_error("RANK ERROR: ⎕ES right argument must be scalar or vector", nullptr, 4, 0);
        return;
    }

    // Empty B = conditional no-op
    if (rhs->is_vector() && rhs->size() == 0) {
        m->result = lhs;
        return;
    }

    // B must be numeric
    if (rhs->is_char_data()) {
        m->throw_error("DOMAIN ERROR: ⎕ES right argument must be numeric", nullptr, 11, 0);
        return;
    }
    if (rhs->size() != 2) {
        m->throw_error("LENGTH ERROR: ⎕ES right argument must be 2-element vector", nullptr, 5, 0);
        return;
    }

    const Eigen::MatrixXd* mat = rhs->as_matrix();
    double class_val = (*mat)(0, 0);
    double subclass_val = (*mat)(1, 0);

    // ISO 13751: elements must be near-integers
    if (!is_near_integer(class_val) || !is_near_integer(subclass_val)) {
        m->throw_error("DOMAIN ERROR: ⎕ES error codes must be integers", nullptr, 11, 0);
        return;
    }

    int error_class = static_cast<int>(std::round(class_val));
    int error_subclass = static_cast<int>(std::round(subclass_val));

    // Get message from left argument
    const char* msg = lhs->is_string() ? lhs->as_string()->c_str()
                      : lhs->to_string_value(m->heap)->as_string()->c_str();

    // 0 0 = clear error state, no signal
    if (error_class == 0 && error_subclass == 0) {
        m->event_type[0] = 0;
        m->event_type[1] = 0;
        m->event_message = nullptr;
        m->result = lhs;
        return;
    }

    // Signal with specified type and message
    m->throw_error(msg, nullptr, error_class, error_subclass);
}

// ⎕EA - Execute Alternate (dyadic) - ISO 13751 §11.6.4
// A ⎕EA B: A and B are character vectors representing executable expressions.
// Execute B first. If error occurs, execute A instead.
void fn_quad_ea(Machine* m, Value* axis, Value* lhs, Value* rhs) {
    REJECT_AXIS(m, axis);

    // ISO 13751: A must be scalar or vector (rank ≤ 1)
    if (lhs->is_array() && lhs->rank() > 1) {
        m->throw_error("RANK ERROR: ⎕EA left argument must be scalar or vector", nullptr, 4, 0);
        return;
    }

    // ISO 13751: B must be scalar or vector (rank ≤ 1)
    if (rhs->is_array() && rhs->rank() > 1) {
        m->throw_error("RANK ERROR: ⎕EA right argument must be scalar or vector", nullptr, 4, 0);
        return;
    }

    // Both arguments must be character vectors (ISO 13751: "A and B are both character vectors")
    bool lhs_is_char = lhs->is_string() || (lhs->is_array() && lhs->is_char_data());
    bool rhs_is_char = rhs->is_string() || (rhs->is_array() && rhs->is_char_data());

    if (!lhs_is_char) {
        m->throw_error("DOMAIN ERROR: ⎕EA left argument must be character vector", nullptr, 11, 0);
        return;
    }
    if (!rhs_is_char) {
        m->throw_error("DOMAIN ERROR: ⎕EA right argument must be character vector", nullptr, 11, 0);
        return;
    }

    // Get the strings to execute
    const char* try_code = rhs->is_string() ? rhs->as_string()->c_str()
                           : rhs->to_string_value(m->heap)->as_string()->c_str();
    const char* alt_code = lhs->is_string() ? lhs->as_string()->c_str()
                           : lhs->to_string_value(m->heap)->as_string()->c_str();

    // Parse the alternate expression (handler) - parse now so errors are caught early
    Continuation* alt_k = m->parser->parse(alt_code);
    if (!alt_k) {
        m->throw_error(m->parser->get_error().c_str(), m->control, 1, 0);
        return;
    }

    // Parse the try expression
    Continuation* try_k = m->parser->parse(try_code);
    if (!try_k) {
        m->throw_error(m->parser->get_error().c_str(), m->control, 1, 0);
        return;
    }

    // Push the error catcher with handler (alt expression)
    CatchErrorK* catcher = m->heap->allocate_ephemeral<CatchErrorK>(alt_k);
    m->push_kont(catcher);

    // Wrap the try expression in FinalizeK to ensure G_PRIME curries (e.g., from dfn calls)
    // are finalized BEFORE CatchErrorK is reached. Otherwise errors during finalization
    // would not be caught.
    FinalizeK* finalize_k = m->heap->allocate_ephemeral<FinalizeK>(try_k, true);
    m->push_kont(finalize_k);
}

// ⎕DL - Delay (monadic) - ISO 13751 §11.5.1
// Pause execution for B seconds, return actual delay time
void fn_quad_dl(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);

    if (!omega->is_scalar()) {
        m->throw_error("RANK ERROR: ⎕DL requires scalar argument", nullptr, 4, 0);
        return;
    }

    double seconds = omega->as_scalar();
    if (seconds < 0) {
        m->throw_error("DOMAIN ERROR: ⎕DL requires non-negative argument", nullptr, 11, 0);
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    auto end = std::chrono::high_resolution_clock::now();

    double actual = std::chrono::duration<double>(end - start).count();
    m->result = m->heap->allocate_scalar(actual);
}

// Helper: classify a Value for ⎕NC
// Returns: 0=undefined, 2=variable, 3=function, 4=operator
static int classify_value(Value* v) {
    if (!v) return 0;  // undefined

    switch (v->tag) {
        case ValueType::SCALAR:
        case ValueType::VECTOR:
        case ValueType::MATRIX:
        case ValueType::NDARRAY:
        case ValueType::STRING:
        case ValueType::STRAND:
            return 2;  // variable (data)

        case ValueType::PRIMITIVE:
        case ValueType::CLOSURE:
        case ValueType::CURRIED_FN:
            return 3;  // function

        case ValueType::DERIVED_OPERATOR:
            // Derived operators can act as functions when fully applied
            return 3;

        case ValueType::OPERATOR:
        case ValueType::DEFINED_OPERATOR:
            return 4;  // operator

        default:
            return 0;  // unknown
    }
}

// Helper: encode a single Unicode codepoint to UTF-8
static void append_codepoint_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Helper: extract a name string from a character matrix row (codepoints to UTF-8)
static std::string extract_name_from_row(const Eigen::MatrixXd* mat, int row, int cols) {
    std::string name;
    for (int c = 0; c < cols; ++c) {
        uint32_t cp = static_cast<uint32_t>((*mat)(row, c));
        if (cp == ' ') continue;  // Skip spaces
        append_codepoint_utf8(name, cp);
    }
    // Trim trailing spaces
    while (!name.empty() && name.back() == ' ') {
        name.pop_back();
    }
    return name;
}

// Helper: extract a name string from a character vector (codepoints to UTF-8)
static std::string extract_name_from_vector(const Eigen::MatrixXd* mat) {
    std::string name;
    for (int i = 0; i < mat->rows(); ++i) {
        uint32_t cp = static_cast<uint32_t>((*mat)(i, 0));
        append_codepoint_utf8(name, cp);
    }
    return name;
}

// ⎕NC - Name Class (ISO 13751 §11.5.2)
// Returns classification of names: 0=undefined, 1=label, 2=variable, 3=function, 4=operator
void fn_quad_nc(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);

    // Must be character data
    bool is_char = omega->is_string() || (omega->is_array() && omega->is_char_data());
    if (!is_char) {
        m->throw_error("DOMAIN ERROR: ⎕NC requires character argument", nullptr, 11, 0);
        return;
    }

    // Handle string (simple name)
    if (omega->is_string()) {
        String* name = omega->as_string();
        Value* v = m->env->lookup(name);
        m->result = m->heap->allocate_scalar(static_cast<double>(classify_value(v)));
        return;
    }

    // Handle character vector (single name)
    if (omega->is_vector()) {
        std::string name_str = extract_name_from_vector(omega->as_matrix());
        Value* v = m->env->lookup(m->string_pool.intern(name_str));
        m->result = m->heap->allocate_scalar(static_cast<double>(classify_value(v)));
        return;
    }

    // Handle character matrix (each row is a name)
    if (omega->is_matrix()) {
        const Eigen::MatrixXd* mat = omega->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();

        Eigen::VectorXd result(rows);
        for (int r = 0; r < rows; ++r) {
            std::string name_str = extract_name_from_row(mat, r, cols);
            Value* v = m->env->lookup(m->string_pool.intern(name_str));
            result(r) = static_cast<double>(classify_value(v));
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    m->throw_error("RANK ERROR: ⎕NC requires vector or matrix argument", nullptr, 4, 0);
}

// Helper: check if a name is protected (system variable/function)
// Protected names start with ⎕ (U+2395)
static bool is_protected_name(const std::string& name) {
    // ⎕ in UTF-8 is 0xE2 0x8E 0x95
    return name.size() >= 3 &&
           static_cast<unsigned char>(name[0]) == 0xE2 &&
           static_cast<unsigned char>(name[1]) == 0x8E &&
           static_cast<unsigned char>(name[2]) == 0x95;
}

// Helper: check if first codepoint is ⎕ (U+2395)
static bool starts_with_quad(const Eigen::MatrixXd* mat, int row, int cols) {
    if (cols == 0) return false;
    uint32_t cp = static_cast<uint32_t>((*mat)(row, 0));
    return cp == 0x2395;
}

static bool vector_starts_with_quad(const Eigen::MatrixXd* mat) {
    if (mat->rows() == 0) return false;
    uint32_t cp = static_cast<uint32_t>((*mat)(0, 0));
    return cp == 0x2395;
}

// ⎕EX - Expunge (ISO 13751 §11.5.3)
// Removes names from the workspace
// Returns: 1=expunged, 0=not expunged (undefined or protected)
void fn_quad_ex(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);

    // Must be character data
    bool is_char = omega->is_string() || (omega->is_array() && omega->is_char_data());
    if (!is_char) {
        m->throw_error("DOMAIN ERROR: ⎕EX requires character argument", nullptr, 11, 0);
        return;
    }

    // Handle string (simple name)
    if (omega->is_string()) {
        String* name = omega->as_string();
        if (is_protected_name(name->str())) {
            m->result = m->heap->allocate_scalar(0.0);  // Protected
            return;
        }
        bool removed = m->env->erase(name);
        m->result = m->heap->allocate_scalar(removed ? 1.0 : 0.0);
        return;
    }

    // Handle character vector (single name)
    if (omega->is_vector()) {
        const Eigen::MatrixXd* mat = omega->as_matrix();
        if (vector_starts_with_quad(mat)) {
            m->result = m->heap->allocate_scalar(0.0);  // Protected
            return;
        }
        std::string name_str = extract_name_from_vector(mat);
        bool removed = m->env->erase(m->string_pool.intern(name_str));
        m->result = m->heap->allocate_scalar(removed ? 1.0 : 0.0);
        return;
    }

    // Handle character matrix (each row is a name)
    if (omega->is_matrix()) {
        const Eigen::MatrixXd* mat = omega->as_matrix();
        int rows = mat->rows();
        int cols = mat->cols();

        Eigen::VectorXd result(rows);
        for (int r = 0; r < rows; ++r) {
            if (starts_with_quad(mat, r, cols)) {
                result(r) = 0.0;  // Protected
            } else {
                std::string name_str = extract_name_from_row(mat, r, cols);
                bool removed = m->env->erase(m->string_pool.intern(name_str));
                result(r) = removed ? 1.0 : 0.0;
            }
        }
        m->result = m->heap->allocate_vector(result);
        return;
    }

    m->throw_error("RANK ERROR: ⎕EX requires vector or matrix argument", nullptr, 4, 0);
}

// Helper: decode UTF-8 string to vector of codepoints
static std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> result;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s.c_str());
    while (*p) {
        uint32_t cp;
        if ((*p & 0x80) == 0) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p++ & 0x1F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p++ & 0x0F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p++ & 0x07) << 18;
            cp |= (*p++ & 0x3F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else {
            cp = *p++;  // Invalid, just use byte
        }
        result.push_back(cp);
    }
    return result;
}

// ⎕NL - Name List (ISO 13751 §11.5.4)
// Returns character matrix of names matching specified classes
// Classes: 1=label, 2=variable, 3=function, 4=operator
void fn_quad_nl(Machine* m, Value* axis, Value* omega) {
    REJECT_AXIS(m, axis);

    // Collect requested classes
    std::vector<int> classes;
    if (omega->is_scalar()) {
        classes.push_back(static_cast<int>(omega->as_scalar()));
    } else if (omega->is_vector()) {
        const Eigen::MatrixXd* mat = omega->as_matrix();
        for (int i = 0; i < mat->rows(); ++i) {
            classes.push_back(static_cast<int>((*mat)(i, 0)));
        }
    } else {
        m->throw_error("RANK ERROR: ⎕NL requires scalar or vector argument", nullptr, 4, 0);
        return;
    }

    // Validate class values
    for (int c : classes) {
        if (c < 1 || c > 4) {
            m->throw_error("DOMAIN ERROR: ⎕NL class must be 1-4", nullptr, 11, 0);
            return;
        }
    }

    // Collect matching names from environment
    std::vector<std::string> names;
    for (const auto& kv : m->env->bindings) {
        Value* v = kv.second;
        int nc = classify_value(v);
        for (int c : classes) {
            if (nc == c) {
                names.push_back(kv.first->str());
                break;
            }
        }
    }

    // Sort alphabetically
    std::sort(names.begin(), names.end());

    // If no names match, return empty matrix
    if (names.empty()) {
        Eigen::MatrixXd empty(0, 0);
        m->result = m->heap->allocate_matrix(empty, true);  // is_char_data = true
        return;
    }

    // Convert names to codepoints and find max length
    std::vector<std::vector<uint32_t>> codepoint_names;
    size_t max_len = 0;
    for (const auto& name : names) {
        auto cps = utf8_to_codepoints(name);
        max_len = std::max(max_len, cps.size());
        codepoint_names.push_back(std::move(cps));
    }

    // Build character matrix (rows = names, cols = max_len, padded with spaces)
    Eigen::MatrixXd mat(names.size(), max_len);
    for (size_t r = 0; r < names.size(); ++r) {
        const auto& cps = codepoint_names[r];
        for (size_t c = 0; c < max_len; ++c) {
            mat(r, c) = (c < cps.size()) ? static_cast<double>(cps[c]) : 32.0;  // pad with space
        }
    }

    m->result = m->heap->allocate_matrix(mat, true);  // is_char_data = true
}

} // namespace apl
