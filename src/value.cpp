// Value implementation

#include "value.h"
#include <cassert>
#include <stdexcept>

namespace apl {

// Destructor
Value::~Value() {
    cleanup();
}

// Internal cleanup helper
void Value::cleanup() {
    // Clean up based on type
    if (tag == ValueType::VECTOR || tag == ValueType::MATRIX) {
        delete data.matrix;
        data.matrix = nullptr;
    }
    // Note: Functions and operators are not owned by Value
    // They will be managed separately (likely by a function table)

    // Clean up promoted matrix cache if it exists
    if (promoted_matrix_) {
        delete promoted_matrix_;
        promoted_matrix_ = nullptr;
    }
}

// Type queries
int Value::rank() const {
    switch (tag) {
        case ValueType::SCALAR:
            return 0;
        case ValueType::VECTOR:
            return 1;
        case ValueType::MATRIX:
            return 2;
        default:
            throw std::runtime_error("rank() called on non-array value");
    }
}

int Value::rows() const {
    if (tag == ValueType::SCALAR) return 1;
    if (tag == ValueType::VECTOR || tag == ValueType::MATRIX) {
        return data.matrix->rows();
    }
    throw std::runtime_error("rows() called on non-array value");
}

int Value::cols() const {
    if (tag == ValueType::SCALAR) return 1;
    if (tag == ValueType::VECTOR) return 1;  // Vectors are n×1
    if (tag == ValueType::MATRIX) {
        return data.matrix->cols();
    }
    throw std::runtime_error("cols() called on non-array value");
}

int Value::size() const {
    if (tag == ValueType::SCALAR) return 1;
    if (tag == ValueType::VECTOR || tag == ValueType::MATRIX) {
        return data.matrix->rows() * data.matrix->cols();
    }
    throw std::runtime_error("size() called on non-array value");
}

// Access methods
double Value::as_scalar() const {
    if (tag != ValueType::SCALAR) {
        throw std::runtime_error("as_scalar() called on non-scalar value");
    }
    return data.scalar;
}

Eigen::MatrixXd* Value::as_matrix() {
    if (tag == ValueType::SCALAR) {
        // Lazy promotion: create 1×1 matrix only when needed
        if (!promoted_matrix_) {
            promoted_matrix_ = new Eigen::MatrixXd(1, 1);
            (*promoted_matrix_)(0, 0) = data.scalar;
        }
        return promoted_matrix_;
    }

    if (tag == ValueType::VECTOR || tag == ValueType::MATRIX) {
        return data.matrix;
    }

    throw std::runtime_error("as_matrix() called on non-array value");
}

const Eigen::MatrixXd* Value::as_matrix() const {
    // For const version, we need to cast away constness for lazy promotion
    // This is safe because promoted_matrix_ is mutable
    return const_cast<Value*>(this)->as_matrix();
}

// Factory methods
Value* Value::from_scalar(double d) {
    Value* val = new Value();
    val->tag = ValueType::SCALAR;
    val->data.scalar = d;
    val->promoted_matrix_ = nullptr;
    return val;
}

Value* Value::from_vector(const Eigen::VectorXd& v) {
    Value* val = new Value();
    val->tag = ValueType::VECTOR;
    val->promoted_matrix_ = nullptr;

    // Store vector as n×1 matrix (zero-copy using Eigen::Map)
    // We need to allocate a new matrix and copy the data
    // since we don't own the input vector
    val->data.matrix = new Eigen::MatrixXd(v.size(), 1);
    val->data.matrix->col(0) = v;

    return val;
}

Value* Value::from_matrix(const Eigen::MatrixXd& m) {
    Value* val = new Value();
    val->tag = ValueType::MATRIX;
    val->promoted_matrix_ = nullptr;

    // Allocate and copy matrix data
    val->data.matrix = new Eigen::MatrixXd(m);

    return val;
}

Value* Value::from_function(PrimitiveFn* fn) {
    Value* val = new Value();
    val->tag = ValueType::FUNCTION;
    val->data.function = fn;
    val->promoted_matrix_ = nullptr;
    return val;
}

Value* Value::from_operator(PrimitiveOp* op) {
    Value* val = new Value();
    val->tag = ValueType::OPERATOR;
    val->data.op = op;
    val->promoted_matrix_ = nullptr;
    return val;
}

} // namespace apl
