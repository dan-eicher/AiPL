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

    // Clean up G2 grammar structures (heap-allocated)
    if (tag == ValueType::DERIVED_OPERATOR) {
        delete data.derived_op;
        data.derived_op = nullptr;
    }
    if (tag == ValueType::CURRIED_FN) {
        delete data.curried_fn;
        data.curried_fn = nullptr;
    }

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

void Value::mark(Heap* heap) {
    // If this is a CLOSURE, mark the continuation graph
    if (tag == ValueType::CLOSURE && data.closure) {
        heap->mark_continuation(data.closure);
    }

    // Mark referenced Values in G2 grammar structures
    if (tag == ValueType::DERIVED_OPERATOR && data.derived_op) {
        if (data.derived_op->first_operand) {
            heap->mark_value(data.derived_op->first_operand);
        }
    }

    if (tag == ValueType::CURRIED_FN && data.curried_fn) {
        if (data.curried_fn->fn) {
            heap->mark_value(data.curried_fn->fn);
        }
        if (data.curried_fn->first_arg) {
            heap->mark_value(data.curried_fn->first_arg);
        }
    }

    // PRIMITIVEs and OPERATORs are C pointers, not GC objects
    // Matrices will be handled when we add nested Value support
}

// Convert STRING to character vector (UTF-8 decode)
Value* Value::to_char_vector(Heap* heap) {
    // Already an array? Return as-is
    if (is_array()) {
        return this;
    }

    // Must be STRING
    if (!is_string()) {
        throw std::runtime_error("to_char_vector requires STRING or array");
    }

    const char* s = data.string;
    std::vector<double> codepoints;

    // Decode UTF-8 to codepoints
    while (*s) {
        unsigned char c = static_cast<unsigned char>(*s);
        uint32_t cp;

        if ((c & 0x80) == 0) {
            // 1-byte (ASCII)
            cp = c;
            s += 1;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte
            cp = (c & 0x1F) << 6;
            cp |= (static_cast<unsigned char>(s[1]) & 0x3F);
            s += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte
            cp = (c & 0x0F) << 12;
            cp |= (static_cast<unsigned char>(s[1]) & 0x3F) << 6;
            cp |= (static_cast<unsigned char>(s[2]) & 0x3F);
            s += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte
            cp = (c & 0x07) << 18;
            cp |= (static_cast<unsigned char>(s[1]) & 0x3F) << 12;
            cp |= (static_cast<unsigned char>(s[2]) & 0x3F) << 6;
            cp |= (static_cast<unsigned char>(s[3]) & 0x3F);
            s += 4;
        } else {
            // Invalid UTF-8, treat as single byte
            cp = c;
            s += 1;
        }
        codepoints.push_back(static_cast<double>(cp));
    }

    Eigen::VectorXd vec(codepoints.size());
    for (size_t i = 0; i < codepoints.size(); ++i) {
        vec(i) = codepoints[i];
    }

    return heap->allocate_vector(vec, true);  // is_char_data = true
}

// Convert character vector to STRING (UTF-8 encode)
Value* Value::to_string_value(Heap* heap) {
    // Already STRING? Return as-is
    if (is_string()) {
        return this;
    }

    // Must be an array with character data
    if (!is_array()) {
        throw std::runtime_error("to_string_value requires STRING or array");
    }

    const Eigen::MatrixXd* mat = as_matrix();
    std::string result;

    // Encode codepoints to UTF-8
    for (int i = 0; i < mat->size(); ++i) {
        uint32_t cp = static_cast<uint32_t>((*mat)(i % mat->rows(), i / mat->rows()));

        if (cp < 0x80) {
            result += static_cast<char>(cp);
        } else if (cp < 0x800) {
            result += static_cast<char>(0xC0 | (cp >> 6));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            result += static_cast<char>(0xE0 | (cp >> 12));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            result += static_cast<char>(0xF0 | (cp >> 18));
            result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    return heap->allocate_string(result.c_str());
}

} // namespace apl
