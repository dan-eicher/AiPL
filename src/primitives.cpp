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

// Array operation primitives
PrimitiveFn prim_rho       = { "⍴", fn_shape, fn_reshape };
PrimitiveFn prim_comma     = { ",", fn_ravel, fn_catenate };
PrimitiveFn prim_transpose = { "⍉", fn_transpose, nullptr };
PrimitiveFn prim_iota      = { "⍳", fn_iota, nullptr };
PrimitiveFn prim_uptack    = { "↑", nullptr, fn_take };
PrimitiveFn prim_downtack  = { "↓", nullptr, fn_drop };

// ============================================================================
// Dyadic Arithmetic Functions
// ============================================================================

// Addition (+)
void fn_add(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar + scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(lhs->data.scalar + rhs->data.scalar));
        return;
    }

    // Scalar extension using Eigen broadcasting
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar + rhs->as_matrix()->array();
        // Preserve vector/matrix distinction
        if (rhs->is_vector()) {
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
        }
        return;
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() + rhs->data.scalar;
        // Preserve vector/matrix distinction
        if (lhs->is_vector()) {
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
}

// Subtraction (-)
void fn_subtract(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar - scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(lhs->data.scalar - rhs->data.scalar));
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar - rhs->as_matrix()->array();
        // Preserve vector/matrix distinction
        if (rhs->is_vector()) {
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
        }
        return;
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() - rhs->data.scalar;
        // Preserve vector/matrix distinction
        if (lhs->is_vector()) {
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
}

// Multiplication (×)
void fn_multiply(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar × scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(lhs->data.scalar * rhs->data.scalar));
        return;
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar * rhs->as_matrix()->array();
        // Preserve vector/matrix distinction
        if (rhs->is_vector()) {
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
        }
        return;
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() * rhs->data.scalar;
        // Preserve vector/matrix distinction
        if (lhs->is_vector()) {
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
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
        m->ctrl.set_value(m->heap->allocate_scalar(lhs->data.scalar / rhs->data.scalar));
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
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
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
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
}

// Power (*)
void fn_power(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar * scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(std::pow(lhs->data.scalar, rhs->data.scalar)));
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
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
        }
        return;
    }

    if (rhs->is_scalar()) {
        // lhs is array of bases, rhs is scalar exponent
        Eigen::MatrixXd result =
            lhs->as_matrix()->array().pow(rhs->data.scalar);
        // Preserve vector/matrix distinction
        if (lhs->is_vector()) {
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
}

// Equality (=)
void fn_equal(Machine* m, Value* lhs, Value* rhs) {
    // Fast path: scalar = scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(lhs->data.scalar == rhs->data.scalar ? 1.0 : 0.0));
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
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
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
            m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
        } else {
            m->ctrl.set_value(m->heap->allocate_matrix(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
}

// ============================================================================
// Monadic Arithmetic Functions
// ============================================================================

// Conjugate/Identity (+)
void fn_conjugate(Machine* m, Value* omega) {
    // For real numbers, identity just returns the value
    if (omega->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(omega->data.scalar));
        return;
    }

    // For arrays, return a copy
    m->ctrl.set_value(m->heap->allocate_matrix(*omega->as_matrix()));
}

// Negation (-)
void fn_negate(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(-omega->data.scalar));
        return;
    }

    Eigen::MatrixXd result = -omega->as_matrix()->array();
    m->ctrl.set_value(m->heap->allocate_matrix(result));
}

// Signum/Sign (×)
void fn_signum(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        double val = omega->data.scalar;
        double sign = (val > 0.0) ? 1.0 : (val < 0.0) ? -1.0 : 0.0;
        m->ctrl.set_value(m->heap->allocate_scalar(sign));
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

    m->ctrl.set_value(m->heap->allocate_matrix(result));
}

// Reciprocal (÷)
void fn_reciprocal(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        if (omega->data.scalar == 0.0) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reciprocal of zero"));
            return;
        }
        m->ctrl.set_value(m->heap->allocate_scalar(1.0 / omega->data.scalar));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Check for zeros
    if ((mat->array() == 0.0).any()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reciprocal of zero"));
        return;
    }

    Eigen::MatrixXd result = 1.0 / mat->array();
    m->ctrl.set_value(m->heap->allocate_matrix(result));
}

// Exponential (*)
void fn_exponential(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(std::exp(omega->data.scalar)));
        return;
    }

    Eigen::MatrixXd result = omega->as_matrix()->array().exp();
    m->ctrl.set_value(m->heap->allocate_matrix(result));
}

// ============================================================================
// Array Operation Functions
// ============================================================================

// Shape (⍴) - monadic: returns shape as vector
void fn_shape(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        // Scalar has empty shape
        Eigen::VectorXd shape(0);
        m->ctrl.set_value(m->heap->allocate_vector(shape));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        // Vector shape is just its length
        Eigen::VectorXd shape(1);
        shape(0) = mat->rows();
        m->ctrl.set_value(m->heap->allocate_vector(shape));
        return;
    }

    // Matrix shape is (rows, cols)
    Eigen::VectorXd shape(2);
    shape(0) = mat->rows();
    shape(1) = mat->cols();
    m->ctrl.set_value(m->heap->allocate_vector(shape));
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

    // Get source data
    Eigen::VectorXd source;
    if (rhs->is_scalar()) {
        source.resize(1);
        source(0) = rhs->as_scalar();
    } else {
        const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();
        // Flatten to vector (column-major order)
        source = Eigen::Map<const Eigen::VectorXd>(rhs_mat->data(), rhs_mat->size());
    }

    // Validate: target size must match source size (no cycling/truncating for now)
    if (target_size != static_cast<int>(source.size())) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: reshape size must match array size"));
        return;
    }

    // Build result by cycling through source data
    Eigen::MatrixXd result(target_rows, target_cols);
    for (int i = 0; i < target_size; ++i) {
        result(i % target_rows, i / target_rows) = source(i % source.size());
    }

    if (target_cols == 1) {
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
}

// Ravel (,) - monadic: flatten to vector
void fn_ravel(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        Eigen::VectorXd v(1);
        v(0) = omega->as_scalar();
        m->ctrl.set_value(m->heap->allocate_vector(v));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Flatten in column-major order
    Eigen::VectorXd result = Eigen::Map<const Eigen::VectorXd>(mat->data(), mat->size());
    m->ctrl.set_value(m->heap->allocate_vector(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
}

// Transpose (⍉) - monadic: reverse dimensions
void fn_transpose(Machine* m, Value* omega) {
    if (omega->is_scalar()) {
        // Scalar transpose is identity
        m->ctrl.set_value(m->heap->allocate_scalar(omega->as_scalar()));
        return;
    }

    if (omega->is_vector()) {
        // Vector transpose gives a 1×n matrix
        const Eigen::MatrixXd* vec = omega->as_matrix();
        Eigen::MatrixXd result = vec->transpose();
        m->ctrl.set_value(m->heap->allocate_matrix(result));
        return;
    }

    // Matrix transpose
    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = mat->transpose();
    m->ctrl.set_value(m->heap->allocate_matrix(result));
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
    m->ctrl.set_value(m->heap->allocate_vector(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result));
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

    m->ctrl.set_value(m->heap->allocate_matrix(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result));
        return;
    }

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        int len = mat->rows();
        int abs_n = std::abs(n);

        if (abs_n >= len) {
            // Drop everything
            Eigen::VectorXd result(0);
            m->ctrl.set_value(m->heap->allocate_vector(result));
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
        m->ctrl.set_value(m->heap->allocate_vector(result));
        return;
    }

    // For matrices, drop rows
    int rows = mat->rows();
    int abs_n = std::abs(n);

    if (abs_n >= rows) {
        // Drop everything - return empty matrix
        Eigen::MatrixXd result(0, mat->cols());
        m->ctrl.set_value(m->heap->allocate_matrix(result));
        return;
    }

    int result_rows = rows - abs_n;
    Eigen::MatrixXd result(result_rows, mat->cols());

    if (n >= 0) {
        result = mat->bottomRows(result_rows);
    } else {
        result = mat->topRows(result_rows);
    }

    m->ctrl.set_value(m->heap->allocate_matrix(result));
}

} // namespace apl
