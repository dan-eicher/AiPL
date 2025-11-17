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

} // namespace apl
