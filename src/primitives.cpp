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
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        // Check for zeros in divisor
        if ((rmat->array() == 0.0).any()) {
            throw std::runtime_error("DOMAIN ERROR: division by zero");
        }
        Eigen::MatrixXd result =
            lhs->data.scalar / rmat->array();
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

    // Check for zeros in divisor
    if ((rmat->array() == 0.0).any()) {
        throw std::runtime_error("DOMAIN ERROR: division by zero");
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
        // lhs is scalar base, rhs is array of exponents: lhs^rhs[i]
        const Eigen::MatrixXd* rmat = rhs->as_matrix();
        Eigen::MatrixXd result(rmat->rows(), rmat->cols());
        for (int i = 0; i < rmat->size(); ++i) {
            result(i) = std::pow(lhs->data.scalar, rmat->data()[i]);
        }
        return Value::from_matrix(result);
    }

    if (rhs->is_scalar()) {
        // lhs is array of bases, rhs is scalar exponent
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

    const Eigen::MatrixXd* mat = omega->as_matrix();
    // Check for zeros
    if ((mat->array() == 0.0).any()) {
        throw std::runtime_error("DOMAIN ERROR: reciprocal of zero");
    }

    Eigen::MatrixXd result = 1.0 / mat->array();
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
        double dim = lhs->as_scalar();
        // Validate: must be non-negative integer
        if (dim < 0.0) {
            throw std::runtime_error("DOMAIN ERROR: reshape dimension must be non-negative");
        }
        if (dim != std::floor(dim)) {
            throw std::runtime_error("DOMAIN ERROR: reshape dimension must be an integer");
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
                throw std::runtime_error("DOMAIN ERROR: reshape dimension must be non-negative");
            }
            if (dim != std::floor(dim)) {
                throw std::runtime_error("DOMAIN ERROR: reshape dimension must be an integer");
            }
            target_rows = static_cast<int>(dim);
            target_cols = 1;
        } else if (shape_mat->rows() == 2) {
            // Two elements: matrix of that shape
            double dim1 = (*shape_mat)(0, 0);
            double dim2 = (*shape_mat)(1, 0);
            // Validate: must be non-negative integers
            if (dim1 < 0.0 || dim2 < 0.0) {
                throw std::runtime_error("DOMAIN ERROR: reshape dimensions must be non-negative");
            }
            if (dim1 != std::floor(dim1) || dim2 != std::floor(dim2)) {
                throw std::runtime_error("DOMAIN ERROR: reshape dimensions must be integers");
            }
            target_rows = static_cast<int>(dim1);
            target_cols = static_cast<int>(dim2);
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

    // Validate: target size must match source size (no cycling/truncating for now)
    if (target_size != static_cast<int>(source.size())) {
        throw std::runtime_error("LENGTH ERROR: reshape size must match array size");
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

    double val = omega->as_scalar();

    // Validate: must be non-negative integer
    if (val < 0.0) {
        throw std::runtime_error("DOMAIN ERROR: iota argument must be non-negative");
    }
    if (val != std::floor(val)) {
        throw std::runtime_error("DOMAIN ERROR: iota argument must be an integer");
    }

    int n = static_cast<int>(val);

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

// ============================================================================
// Reduction and Scan Operations
// ============================================================================

// Reduce (/) - apply dyadic function between elements, right to left
Value* fn_reduce(Value* func, Value* omega) {
    if (!func->is_function()) {
        throw std::runtime_error("DOMAIN ERROR: reduce requires a function");
    }

    PrimitiveFn* fn = func->data.primitive_fn;
    if (!fn->dyadic) {
        throw std::runtime_error("DOMAIN ERROR: reduce requires a dyadic function");
    }

    if (omega->is_scalar()) {
        // Reducing a scalar is identity
        return Value::from_scalar(omega->as_scalar());
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        // Reduce vector to scalar
        int len = mat->rows();
        if (len == 0) {
            throw std::runtime_error("LENGTH ERROR: cannot reduce empty vector");
        }
        if (len == 1) {
            return Value::from_scalar((*mat)(0, 0));
        }

        // Right-to-left reduction (APL standard: first f (f/rest))
        Value* acc = Value::from_scalar((*mat)(len - 1, 0));
        for (int i = len - 2; i >= 0; --i) {
            Value* elem = Value::from_scalar((*mat)(i, 0));
            Value* result = fn->dyadic(elem, acc);
            delete elem;
            delete acc;
            acc = result;
        }
        return acc;
    }

    // For matrix, reduce along last axis (columns)
    // Result is a vector with one element per row
    int rows = mat->rows();
    int cols = mat->cols();

    if (cols == 0) {
        throw std::runtime_error("LENGTH ERROR: cannot reduce empty dimension");
    }

    Eigen::VectorXd result(rows);

    for (int r = 0; r < rows; ++r) {
        // Reduce this row right-to-left
        Value* acc = Value::from_scalar((*mat)(r, cols - 1));
        for (int c = cols - 2; c >= 0; --c) {
            Value* elem = Value::from_scalar((*mat)(r, c));
            Value* new_acc = fn->dyadic(elem, acc);
            delete elem;
            delete acc;
            acc = new_acc;
        }
        result(r) = acc->as_scalar();
        delete acc;
    }

    return Value::from_vector(result);
}

// Reduce-first (⌿) - reduce along first axis (rows)
Value* fn_reduce_first(Value* func, Value* omega) {
    if (!func->is_function()) {
        throw std::runtime_error("DOMAIN ERROR: reduce-first requires a function");
    }

    PrimitiveFn* fn = func->data.primitive_fn;
    if (!fn->dyadic) {
        throw std::runtime_error("DOMAIN ERROR: reduce-first requires a dyadic function");
    }

    if (omega->is_scalar()) {
        return Value::from_scalar(omega->as_scalar());
    }

    if (omega->is_vector()) {
        // For vector, same as regular reduce
        return fn_reduce(func, omega);
    }

    // For matrix, reduce along first axis (rows)
    // Result is a row vector
    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (rows == 0) {
        throw std::runtime_error("LENGTH ERROR: cannot reduce empty dimension");
    }

    Eigen::VectorXd result(cols);

    for (int c = 0; c < cols; ++c) {
        // Reduce this column bottom-to-top (right-to-left in first axis)
        Value* acc = Value::from_scalar((*mat)(rows - 1, c));
        for (int r = rows - 2; r >= 0; --r) {
            Value* elem = Value::from_scalar((*mat)(r, c));
            Value* new_acc = fn->dyadic(elem, acc);
            delete elem;
            delete acc;
            acc = new_acc;
        }
        result(c) = acc->as_scalar();
        delete acc;
    }

    return Value::from_vector(result);
}

// Scan (\) - apply dyadic function cumulatively, right to left
Value* fn_scan(Value* func, Value* omega) {
    if (!func->is_function()) {
        throw std::runtime_error("DOMAIN ERROR: scan requires a function");
    }

    PrimitiveFn* fn = func->data.primitive_fn;
    if (!fn->dyadic) {
        throw std::runtime_error("DOMAIN ERROR: scan requires a dyadic function");
    }

    if (omega->is_scalar()) {
        return Value::from_scalar(omega->as_scalar());
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        int len = mat->rows();
        if (len == 0) {
            return Value::from_vector(Eigen::VectorXd(0));
        }

        Eigen::VectorXd result(len);

        // Right-to-left scan (APL standard)
        result(len - 1) = (*mat)(len - 1, 0);
        Value* acc = Value::from_scalar((*mat)(len - 1, 0));

        for (int i = len - 2; i >= 0; --i) {
            Value* elem = Value::from_scalar((*mat)(i, 0));
            Value* new_acc = fn->dyadic(elem, acc);
            result(i) = new_acc->as_scalar();
            delete elem;
            delete acc;
            acc = new_acc;
        }
        delete acc;

        return Value::from_vector(result);
    }

    // For matrix, scan along last axis (columns)
    int rows = mat->rows();
    int cols = mat->cols();

    Eigen::MatrixXd result(rows, cols);

    for (int r = 0; r < rows; ++r) {
        // Scan this row right-to-left
        result(r, cols - 1) = (*mat)(r, cols - 1);
        Value* acc = Value::from_scalar((*mat)(r, cols - 1));

        for (int c = cols - 2; c >= 0; --c) {
            Value* elem = Value::from_scalar((*mat)(r, c));
            Value* new_acc = fn->dyadic(elem, acc);
            result(r, c) = new_acc->as_scalar();
            delete elem;
            delete acc;
            acc = new_acc;
        }
        delete acc;
    }

    return Value::from_matrix(result);
}

// Scan-first (⍀) - scan along first axis (rows)
Value* fn_scan_first(Value* func, Value* omega) {
    if (!func->is_function()) {
        throw std::runtime_error("DOMAIN ERROR: scan-first requires a function");
    }

    PrimitiveFn* fn = func->data.primitive_fn;
    if (!fn->dyadic) {
        throw std::runtime_error("DOMAIN ERROR: scan-first requires a dyadic function");
    }

    if (omega->is_scalar()) {
        return Value::from_scalar(omega->as_scalar());
    }

    if (omega->is_vector()) {
        // For vector, same as regular scan
        return fn_scan(func, omega);
    }

    // For matrix, scan along first axis (rows)
    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    Eigen::MatrixXd result(rows, cols);

    for (int c = 0; c < cols; ++c) {
        // Scan this column bottom-to-top (right-to-left in first axis)
        result(rows - 1, c) = (*mat)(rows - 1, c);
        Value* acc = Value::from_scalar((*mat)(rows - 1, c));

        for (int r = rows - 2; r >= 0; --r) {
            Value* elem = Value::from_scalar((*mat)(r, c));
            Value* new_acc = fn->dyadic(elem, acc);
            result(r, c) = new_acc->as_scalar();
            delete elem;
            delete acc;
            acc = new_acc;
        }
        delete acc;
    }

    return Value::from_matrix(result);
}

} // namespace apl
