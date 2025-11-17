// Primitives implementation

#include "primitives.h"
#include "value.h"
#include <cmath>
#include <stdexcept>

namespace apl {

// PrimitiveFn structs combining monadic and dyadic forms
PrimitiveFn prim_plus    = { "+", fn_conjugate, fn_add };
PrimitiveFn prim_minus   = { "-", fn_negate, fn_subtract };
PrimitiveFn prim_times   = { "×", fn_signum, fn_multiply };
PrimitiveFn prim_divide  = { "÷", fn_reciprocal, fn_divide };
PrimitiveFn prim_star    = { "*", fn_exponential, fn_power };

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
Value* fn_add(Value* lhs, Value* rhs) {
    // Fast path: scalar + scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        return Value::from_scalar(lhs->data.scalar + rhs->data.scalar);
    }

    // Scalar extension using Eigen broadcasting
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar + rhs->as_matrix()->array();
        return Value::from_matrix(result);
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() + rhs->data.scalar;
        return Value::from_matrix(result);
    }

    // Array + Array: element-wise
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    // Shape checking
    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        throw std::runtime_error("LENGTH ERROR: mismatched shapes in addition");
    }

    Eigen::MatrixXd result = lmat->array() + rmat->array();
    return Value::from_matrix(result);
}

// Subtraction (-)
Value* fn_subtract(Value* lhs, Value* rhs) {
    // Fast path: scalar - scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        return Value::from_scalar(lhs->data.scalar - rhs->data.scalar);
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar - rhs->as_matrix()->array();
        return Value::from_matrix(result);
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() - rhs->data.scalar;
        return Value::from_matrix(result);
    }

    // Array - Array
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        throw std::runtime_error("LENGTH ERROR: mismatched shapes in subtraction");
    }

    Eigen::MatrixXd result = lmat->array() - rmat->array();
    return Value::from_matrix(result);
}

// Multiplication (×)
Value* fn_multiply(Value* lhs, Value* rhs) {
    // Fast path: scalar × scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        return Value::from_scalar(lhs->data.scalar * rhs->data.scalar);
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar * rhs->as_matrix()->array();
        return Value::from_matrix(result);
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() * rhs->data.scalar;
        return Value::from_matrix(result);
    }

    // Array × Array: element-wise multiplication
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        throw std::runtime_error("LENGTH ERROR: mismatched shapes in multiplication");
    }

    Eigen::MatrixXd result = lmat->array() * rmat->array();
    return Value::from_matrix(result);
}

// Division (÷)
Value* fn_divide(Value* lhs, Value* rhs) {
    // Fast path: scalar ÷ scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        if (rhs->data.scalar == 0.0) {
            throw std::runtime_error("DOMAIN ERROR: division by zero");
        }
        return Value::from_scalar(lhs->data.scalar / rhs->data.scalar);
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->data.scalar / rhs->as_matrix()->array();
        return Value::from_matrix(result);
    }

    if (rhs->is_scalar()) {
        if (rhs->data.scalar == 0.0) {
            throw std::runtime_error("DOMAIN ERROR: division by zero");
        }
        Eigen::MatrixXd result =
            lhs->as_matrix()->array() / rhs->data.scalar;
        return Value::from_matrix(result);
    }

    // Array ÷ Array
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        throw std::runtime_error("LENGTH ERROR: mismatched shapes in division");
    }

    Eigen::MatrixXd result = lmat->array() / rmat->array();
    return Value::from_matrix(result);
}

// Power (*)
Value* fn_power(Value* lhs, Value* rhs) {
    // Fast path: scalar * scalar
    if (lhs->is_scalar() && rhs->is_scalar()) {
        return Value::from_scalar(std::pow(lhs->data.scalar, rhs->data.scalar));
    }

    // Scalar extension
    if (lhs->is_scalar()) {
        Eigen::MatrixXd result =
            rhs->as_matrix()->array().pow(lhs->data.scalar);
        return Value::from_matrix(result);
    }

    if (rhs->is_scalar()) {
        Eigen::MatrixXd result =
            lhs->as_matrix()->array().pow(rhs->data.scalar);
        return Value::from_matrix(result);
    }

    // Array * Array: element-wise power
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    if (lmat->rows() != rmat->rows() || lmat->cols() != rmat->cols()) {
        throw std::runtime_error("LENGTH ERROR: mismatched shapes in power");
    }

    Eigen::MatrixXd result = lmat->array().pow(rmat->array());
    return Value::from_matrix(result);
}

// ============================================================================
// Monadic Arithmetic Functions
// ============================================================================

// Conjugate/Identity (+)
Value* fn_conjugate(Value* omega) {
    // For real numbers, identity just returns the value
    if (omega->is_scalar()) {
        return Value::from_scalar(omega->data.scalar);
    }

    // For arrays, return a copy
    return Value::from_matrix(*omega->as_matrix());
}

// Negation (-)
Value* fn_negate(Value* omega) {
    if (omega->is_scalar()) {
        return Value::from_scalar(-omega->data.scalar);
    }

    Eigen::MatrixXd result = -omega->as_matrix()->array();
    return Value::from_matrix(result);
}

// Signum/Sign (×)
Value* fn_signum(Value* omega) {
    if (omega->is_scalar()) {
        double val = omega->data.scalar;
        double sign = (val > 0.0) ? 1.0 : (val < 0.0) ? -1.0 : 0.0;
        return Value::from_scalar(sign);
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

    return Value::from_matrix(result);
}

// Reciprocal (÷)
Value* fn_reciprocal(Value* omega) {
    if (omega->is_scalar()) {
        if (omega->data.scalar == 0.0) {
            throw std::runtime_error("DOMAIN ERROR: reciprocal of zero");
        }
        return Value::from_scalar(1.0 / omega->data.scalar);
    }

    Eigen::MatrixXd result = 1.0 / omega->as_matrix()->array();
    return Value::from_matrix(result);
}

// Exponential (*)
Value* fn_exponential(Value* omega) {
    if (omega->is_scalar()) {
        return Value::from_scalar(std::exp(omega->data.scalar));
    }

    Eigen::MatrixXd result = omega->as_matrix()->array().exp();
    return Value::from_matrix(result);
}

// ============================================================================
// Array Operation Functions
// ============================================================================

// Shape (⍴) - monadic: returns shape as vector
Value* fn_shape(Value* omega) {
    if (omega->is_scalar()) {
        // Scalar has empty shape
        Eigen::VectorXd shape(0);
        return Value::from_vector(shape);
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        // Vector shape is just its length
        Eigen::VectorXd shape(1);
        shape(0) = mat->rows();
        return Value::from_vector(shape);
    }

    // Matrix shape is (rows, cols)
    Eigen::VectorXd shape(2);
    shape(0) = mat->rows();
    shape(1) = mat->cols();
    return Value::from_vector(shape);
}

// Reshape (⍴) - dyadic: reshape rhs to shape given by lhs
Value* fn_reshape(Value* lhs, Value* rhs) {
    // lhs must be a scalar or vector specifying new shape
    if (!lhs->is_scalar() && !lhs->is_vector()) {
        throw std::runtime_error("RANK ERROR: left argument to reshape must be scalar or vector");
    }

    // Get target shape
    int target_rows, target_cols;

    if (lhs->is_scalar()) {
        // Scalar shape means 1D vector of that length
        target_rows = static_cast<int>(lhs->as_scalar());
        target_cols = 1;
    } else {
        const Eigen::MatrixXd* shape_mat = lhs->as_matrix();
        if (shape_mat->rows() == 1) {
            // Single element: vector of that length
            target_rows = static_cast<int>((*shape_mat)(0, 0));
            target_cols = 1;
        } else if (shape_mat->rows() == 2) {
            // Two elements: matrix of that shape
            target_rows = static_cast<int>((*shape_mat)(0, 0));
            target_cols = static_cast<int>((*shape_mat)(1, 0));
        } else {
            throw std::runtime_error("RANK ERROR: reshape shape must have 1 or 2 elements");
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

    // Build result by cycling through source data
    Eigen::MatrixXd result(target_rows, target_cols);
    for (int i = 0; i < target_size; ++i) {
        result(i % target_rows, i / target_rows) = source(i % source.size());
    }

    if (target_cols == 1) {
        return Value::from_vector(result.col(0));
    }
    return Value::from_matrix(result);
}

// Ravel (,) - monadic: flatten to vector
Value* fn_ravel(Value* omega) {
    if (omega->is_scalar()) {
        Eigen::VectorXd v(1);
        v(0) = omega->as_scalar();
        return Value::from_vector(v);
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Flatten in column-major order
    Eigen::VectorXd result = Eigen::Map<const Eigen::VectorXd>(mat->data(), mat->size());
    return Value::from_vector(result);
}

// Catenate (,) - dyadic: concatenate arrays
Value* fn_catenate(Value* lhs, Value* rhs) {
    // Convert both to matrices for uniform handling
    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();

    // For vectors or compatible matrices, concatenate along first dimension
    if (lmat->cols() != rmat->cols()) {
        throw std::runtime_error("LENGTH ERROR: incompatible shapes for catenation");
    }

    Eigen::MatrixXd result(lmat->rows() + rmat->rows(), lmat->cols());
    result << *lmat, *rmat;

    if (result.cols() == 1) {
        return Value::from_vector(result.col(0));
    }
    return Value::from_matrix(result);
}

// Transpose (⍉) - monadic: reverse dimensions
Value* fn_transpose(Value* omega) {
    if (omega->is_scalar()) {
        // Scalar transpose is identity
        return Value::from_scalar(omega->as_scalar());
    }

    if (omega->is_vector()) {
        // Vector transpose gives a 1×n matrix
        const Eigen::MatrixXd* vec = omega->as_matrix();
        Eigen::MatrixXd result = vec->transpose();
        return Value::from_matrix(result);
    }

    // Matrix transpose
    const Eigen::MatrixXd* mat = omega->as_matrix();
    Eigen::MatrixXd result = mat->transpose();
    return Value::from_matrix(result);
}

// Iota (⍳) - monadic: generate indices from 0 to n-1
Value* fn_iota(Value* omega) {
    if (!omega->is_scalar()) {
        throw std::runtime_error("RANK ERROR: iota argument must be scalar");
    }

    int n = static_cast<int>(omega->as_scalar());
    if (n < 0) {
        throw std::runtime_error("DOMAIN ERROR: iota argument must be non-negative");
    }

    Eigen::VectorXd result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = i;
    }
    return Value::from_vector(result);
}

// Take (↑) - dyadic: take first n elements
Value* fn_take(Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        throw std::runtime_error("RANK ERROR: take count must be scalar");
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_scalar()) {
        // Taking from scalar: replicate
        Eigen::VectorXd result(std::abs(n));
        result.setConstant(rhs->as_scalar());
        return Value::from_vector(result);
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
        return Value::from_vector(result);
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

    return Value::from_matrix(result);
}

// Drop (↓) - dyadic: drop first n elements
Value* fn_drop(Value* lhs, Value* rhs) {
    if (!lhs->is_scalar()) {
        throw std::runtime_error("RANK ERROR: drop count must be scalar");
    }

    int n = static_cast<int>(lhs->as_scalar());

    if (rhs->is_scalar()) {
        // Dropping from scalar gives empty vector
        Eigen::VectorXd result(0);
        return Value::from_vector(result);
    }

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        int len = mat->rows();
        int abs_n = std::abs(n);

        if (abs_n >= len) {
            // Drop everything
            Eigen::VectorXd result(0);
            return Value::from_vector(result);
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
        return Value::from_vector(result);
    }

    // For matrices, drop rows
    int rows = mat->rows();
    int abs_n = std::abs(n);

    if (abs_n >= rows) {
        // Drop everything - return empty matrix
        Eigen::MatrixXd result(0, mat->cols());
        return Value::from_matrix(result);
    }

    int result_rows = rows - abs_n;
    Eigen::MatrixXd result(result_rows, mat->cols());

    if (n >= 0) {
        result = mat->bottomRows(result_rows);
    } else {
        result = mat->topRows(result_rows);
    }

    return Value::from_matrix(result);
}

} // namespace apl
