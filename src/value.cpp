// Value implementation

#include "value.h"
#include "heap.h"
#include "continuation.h"
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

void Value::mark(APLHeap* heap) {
    // If this is a CLOSURE, mark the continuation graph
    if (tag == ValueType::CLOSURE && data.closure) {
        heap->mark_continuation(data.closure);
    }
    // PRIMITIVEs and OPERATORs are C pointers, not GC objects
    // Matrices will be handled when we add nested Value support
}

} // namespace apl
