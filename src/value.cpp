// Value implementation

#include "value.h"
#include "heap.h"
#include "continuation.h"
#include "environment.h"
#include "dir.h"
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace apl {

// ============================================================================
// String implementation - UTF-8 utilities consolidated here
// ============================================================================

// Internal constants for UTF-8 encoding/decoding
namespace {
    // Leading byte patterns (for encoding)
    constexpr uint8_t PREFIX_2BYTE = 0xC0;    // 110xxxxx
    constexpr uint8_t PREFIX_3BYTE = 0xE0;    // 1110xxxx
    constexpr uint8_t PREFIX_4BYTE = 0xF0;    // 11110xxx
    constexpr uint8_t PREFIX_CONT  = 0x80;    // 10xxxxxx

    // Masks for decoding
    constexpr uint8_t MASK_1BYTE = 0x80;      // Check if ASCII
    constexpr uint8_t MASK_2BYTE = 0xE0;      // Check for 2-byte lead
    constexpr uint8_t MASK_3BYTE = 0xF0;      // Check for 3-byte lead
    constexpr uint8_t MASK_4BYTE = 0xF8;      // Check for 4-byte lead
    constexpr uint8_t MASK_CONT  = 0x3F;      // Continuation byte data (6 bits)

    // Data masks for leading bytes
    constexpr uint8_t DATA_2BYTE = 0x1F;      // 5 bits from 2-byte lead
    constexpr uint8_t DATA_3BYTE = 0x0F;      // 4 bits from 3-byte lead
    constexpr uint8_t DATA_4BYTE = 0x07;      // 3 bits from 4-byte lead
}

void String::mark(Heap* /*heap*/) {
    // String is a leaf node - nothing to mark
}

// Get byte length of UTF-8 sequence starting with given byte
int String::utf8_sequence_length(unsigned char lead_byte) {
    if ((lead_byte & MASK_1BYTE) == 0) return 1;
    if ((lead_byte & MASK_2BYTE) == PREFIX_2BYTE) return 2;
    if ((lead_byte & MASK_3BYTE) == PREFIX_3BYTE) return 3;
    if ((lead_byte & MASK_4BYTE) == PREFIX_4BYTE) return 4;
    return 1;  // Invalid, treat as single byte
}

// Decode one codepoint from UTF-8, advancing the pointer
uint32_t String::decode_one(const unsigned char*& p) {
    unsigned char c = *p;
    uint32_t cp;

    if ((c & MASK_1BYTE) == 0) {
        cp = c;
        p += 1;
    } else if ((c & MASK_2BYTE) == PREFIX_2BYTE) {
        cp = (c & DATA_2BYTE) << 6;
        cp |= (p[1] & MASK_CONT);
        p += 2;
    } else if ((c & MASK_3BYTE) == PREFIX_3BYTE) {
        cp = (c & DATA_3BYTE) << 12;
        cp |= (p[1] & MASK_CONT) << 6;
        cp |= (p[2] & MASK_CONT);
        p += 3;
    } else if ((c & MASK_4BYTE) == PREFIX_4BYTE) {
        cp = (c & DATA_4BYTE) << 18;
        cp |= (p[1] & MASK_CONT) << 12;
        cp |= (p[2] & MASK_CONT) << 6;
        cp |= (p[3] & MASK_CONT);
        p += 4;
    } else {
        // Invalid UTF-8, treat as single byte
        cp = c;
        p += 1;
    }
    return cp;
}

// Encode a single codepoint to UTF-8
std::string String::encode_codepoint(uint32_t cp) {
    std::string result;
    if (cp <= MAX_1BYTE) {
        result += static_cast<char>(cp);
    } else if (cp <= MAX_2BYTE) {
        result += static_cast<char>(PREFIX_2BYTE | (cp >> 6));
        result += static_cast<char>(PREFIX_CONT | (cp & MASK_CONT));
    } else if (cp <= MAX_3BYTE) {
        result += static_cast<char>(PREFIX_3BYTE | (cp >> 12));
        result += static_cast<char>(PREFIX_CONT | ((cp >> 6) & MASK_CONT));
        result += static_cast<char>(PREFIX_CONT | (cp & MASK_CONT));
    } else if (cp <= MAX_4BYTE) {
        result += static_cast<char>(PREFIX_4BYTE | (cp >> 18));
        result += static_cast<char>(PREFIX_CONT | ((cp >> 12) & MASK_CONT));
        result += static_cast<char>(PREFIX_CONT | ((cp >> 6) & MASK_CONT));
        result += static_cast<char>(PREFIX_CONT | (cp & MASK_CONT));
    }
    return result;
}

// Encode a vector of codepoints to UTF-8 string
std::string String::encode_codepoints(const std::vector<uint32_t>& cps) {
    std::string result;
    for (uint32_t cp : cps) {
        result += encode_codepoint(cp);
    }
    return result;
}

// Decode UTF-8 string to codepoints
std::vector<uint32_t> String::decode_utf8(const char* s) {
    std::vector<uint32_t> result;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s);
    while (*p) {
        result.push_back(decode_one(p));
    }
    return result;
}

std::vector<uint32_t> String::decode_utf8(const std::string& s) {
    return decode_utf8(s.c_str());
}

// Instance methods using the static utilities

size_t String::length() const {
    size_t count = 0;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(data_.c_str());
    while (*p) {
        decode_one(p);
        ++count;
    }
    return count;
}

uint32_t String::at(size_t index) const {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(data_.c_str());
    size_t i = 0;
    while (*p) {
        uint32_t cp = decode_one(p);
        if (i == index) return cp;
        ++i;
    }
    // Index out of bounds - return replacement character
    return 0xFFFD;
}

std::vector<uint32_t> String::to_codepoints() const {
    return decode_utf8(data_);
}

// ============================================================================
// Value implementation
// ============================================================================

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
    // Clean up NDARRAY (owns both the NDArrayData struct and the VectorXd inside it)
    if (tag == ValueType::NDARRAY) {
        if (data.ndarray) {
            delete data.ndarray->data;  // Delete the Eigen::VectorXd
            delete data.ndarray;        // Delete the NDArrayData struct
            data.ndarray = nullptr;
        }
    }
    // Clean up strand vector (elements are GC-managed, but the vector itself is owned)
    if (tag == ValueType::STRAND) {
        delete data.strand;
        data.strand = nullptr;
    }
    // Note: PRIMITIVE and OPERATOR values point to static globals (prim_plus, op_reduce, etc.)
    // CLOSURE body points to GC-managed Continuation (marked via mark(), not deleted here)
    // TypeDirectedK (type_dispatch) is GC-tracked via heap; just null the pointer.
    if (tag == ValueType::CLOSURE) {
        if (data.closure) {
            data.closure->type_dispatch = nullptr;
        }
        delete data.closure;  // ClosureData struct, not the Continuation inside
        data.closure = nullptr;
    }

    // Clean up G2 grammar structures (heap-allocated)
    if (tag == ValueType::DEFINED_OPERATOR) {
        delete data.defined_op_data;
        data.defined_op_data = nullptr;
    }
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
        case ValueType::STRAND:  // Strands are rank 1 (vectors of values)
            return 1;
        case ValueType::MATRIX:
            return 2;
        case ValueType::NDARRAY:
            return static_cast<int>(data.ndarray->shape.size());
        default:
            throw std::runtime_error("rank() called on non-array value");
    }
}

int Value::rows() const {
    if (tag == ValueType::SCALAR) return 1;
    if (tag == ValueType::VECTOR || tag == ValueType::MATRIX) {
        return data.matrix->rows();
    }
    if (tag == ValueType::STRAND) {
        return static_cast<int>(data.strand->size());
    }
    if (tag == ValueType::NDARRAY) {
        // For NDARRAY, return first axis size
        return data.ndarray->shape.empty() ? 0 : data.ndarray->shape[0];
    }
    throw std::runtime_error("rows() called on non-array value");
}

int Value::cols() const {
    if (tag == ValueType::SCALAR) return 1;
    if (tag == ValueType::VECTOR) return 1;  // Vectors are n×1
    if (tag == ValueType::STRAND) return 1;  // Strands are rank-1 (like vectors)
    if (tag == ValueType::MATRIX) {
        return data.matrix->cols();
    }
    if (tag == ValueType::NDARRAY) {
        // For NDARRAY, return second axis size (or 1 if rank < 2)
        return data.ndarray->shape.size() >= 2 ? data.ndarray->shape[1] : 1;
    }
    throw std::runtime_error("cols() called on non-array value");
}

int Value::size() const {
    if (tag == ValueType::SCALAR) return 1;
    if (tag == ValueType::VECTOR || tag == ValueType::MATRIX) {
        return data.matrix->rows() * data.matrix->cols();
    }
    if (tag == ValueType::STRAND) {
        return static_cast<int>(data.strand->size());
    }
    if (tag == ValueType::NDARRAY) {
        // Product of all dimensions
        int product = 1;
        for (int dim : data.ndarray->shape) {
            product *= dim;
        }
        return product;
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

// NDARRAY indexing: compute linear index from multi-dimensional indices (0-based)
// Note: ⎕IO adjustment happens at the APL evaluation level, not here
int Value::ndarray_linear_index(const std::vector<int>& indices) const {
    if (tag != ValueType::NDARRAY) {
        throw std::runtime_error("ndarray_linear_index() called on non-NDARRAY value");
    }
    if (indices.size() != data.ndarray->shape.size()) {
        throw std::runtime_error("ndarray_linear_index(): index count doesn't match rank");
    }

    int linear = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        linear += indices[i] * data.ndarray->strides[i];
    }
    return linear;
}

// NDARRAY element access by multi-dimensional index (0-based)
double& Value::ndarray_at(const std::vector<int>& indices) {
    if (tag != ValueType::NDARRAY) {
        throw std::runtime_error("ndarray_at() called on non-NDARRAY value");
    }
    return (*data.ndarray->data)(ndarray_linear_index(indices));
}

double Value::ndarray_at(const std::vector<int>& indices) const {
    if (tag != ValueType::NDARRAY) {
        throw std::runtime_error("ndarray_at() called on non-NDARRAY value");
    }
    return (*data.ndarray->data)(ndarray_linear_index(indices));
}

void Value::mark(Heap* heap) {
    // If this is a STRING, mark the interned String object
    if (tag == ValueType::STRING && data.string) {
        heap->mark(data.string);
    }

    // If this is a STRAND, mark all elements
    if (tag == ValueType::STRAND && data.strand) {
        for (Value* elem : *data.strand) {
            heap->mark(elem);
        }
    }

    // If this is a CLOSURE, mark the continuation graph body and TypeDirectedK
    if (tag == ValueType::CLOSURE && data.closure) {
        heap->mark(data.closure->body);
        if (data.closure->type_dispatch) {
            heap->mark(data.closure->type_dispatch);
        }
    }

    // DEFINED_OPERATOR: mark body, lexical environment, and interned names
    if (tag == ValueType::DEFINED_OPERATOR && data.defined_op_data) {
        heap->mark(data.defined_op_data->body);
        heap->mark(data.defined_op_data->lexical_env);
        heap->mark(data.defined_op_data->name);
        heap->mark(data.defined_op_data->left_operand_name);
        heap->mark(data.defined_op_data->right_operand_name);
        heap->mark(data.defined_op_data->left_arg_name);
        heap->mark(data.defined_op_data->right_arg_name);
        heap->mark(data.defined_op_data->result_name);
    }

    // Mark referenced Values in G2 grammar structures
    if (tag == ValueType::DERIVED_OPERATOR && data.derived_op) {
        heap->mark(data.derived_op->first_operand);
        heap->mark(data.derived_op->second_operand);
        heap->mark(data.derived_op->operator_value);
        // Note: defined_op points to DefinedOperatorData inside a DEFINED_OPERATOR Value
        // which is marked separately; primitive_op is a static global
    }

    if (tag == ValueType::CURRIED_FN && data.curried_fn) {
        heap->mark(data.curried_fn->fn);
        heap->mark(data.curried_fn->first_arg);
        heap->mark(data.curried_fn->axis);
    }

    // PRIMITIVEs and OPERATORs point to static globals, not GC objects
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

    // Use consolidated String UTF-8 decoding
    std::vector<uint32_t> codepoints = String::decode_utf8(data.string->c_str());

    Eigen::VectorXd vec(codepoints.size());
    for (size_t i = 0; i < codepoints.size(); ++i) {
        vec(i) = static_cast<double>(codepoints[i]);
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

    // Use consolidated String UTF-8 encoding
    for (int i = 0; i < mat->size(); ++i) {
        uint32_t cp = static_cast<uint32_t>((*mat)(i % mat->rows(), i / mat->rows()));
        result += String::encode_codepoint(cp);
    }

    return heap->allocate_string(result.c_str());
}

// Helper: format a single number
static std::string format_number(double d) {
    // Check for special values
    if (std::isinf(d)) {
        return d > 0 ? "∞" : "¯∞";
    }
    if (std::isnan(d)) {
        return "NaN";
    }

    // Check if it's an integer
    if (d == std::floor(d) && std::abs(d) < 1e15) {
        std::ostringstream oss;
        if (d < 0) {
            oss << "¯" << static_cast<long long>(-d);
        } else {
            oss << static_cast<long long>(d);
        }
        return oss.str();
    }

    // Format as floating point
    std::ostringstream oss;
    oss << std::setprecision(10);
    if (d < 0) {
        oss << "¯" << -d;
    } else {
        oss << d;
    }
    std::string s = oss.str();

    // Remove trailing zeros after decimal point
    if (s.find('.') != std::string::npos) {
        size_t last = s.find_last_not_of('0');
        if (last != std::string::npos && s[last] == '.') {
            last--;  // Also remove the decimal point if nothing after
        }
        s = s.substr(0, last + 1);
    }

    return s;
}

// Format a Value for display
std::string format_value(const Value* v) {
    if (!v) return "null";

    switch (v->tag) {
        case ValueType::SCALAR:
            return format_number(v->data.scalar);

        case ValueType::STRING:
            // Return quoted string
            return std::string("'") + v->data.string->c_str() + "'";

        case ValueType::VECTOR: {
            const Eigen::MatrixXd* mat = v->as_matrix();
            int n = mat->rows();

            if (n == 0) return "⍬";  // Empty vector

            std::ostringstream oss;
            if (v->is_char_data()) {
                // Character vector - display as string
                oss << "'";
                for (int i = 0; i < n; i++) {
                    uint32_t cp = static_cast<uint32_t>((*mat)(i, 0));
                    oss << String::encode_codepoint(cp);
                }
                oss << "'";
            } else {
                // Numeric vector - space-separated
                for (int i = 0; i < n; i++) {
                    if (i > 0) oss << " ";
                    oss << format_number((*mat)(i, 0));
                }
            }
            return oss.str();
        }

        case ValueType::MATRIX: {
            const Eigen::MatrixXd* mat = v->as_matrix();
            int rows = mat->rows();
            int cols = mat->cols();

            if (rows == 0 || cols == 0) return "⍬";  // Empty matrix

            std::ostringstream oss;
            for (int i = 0; i < rows; i++) {
                if (i > 0) oss << "\n";
                for (int j = 0; j < cols; j++) {
                    if (j > 0) oss << " ";
                    if (v->is_char_data()) {
                        uint32_t cp = static_cast<uint32_t>((*mat)(i, j));
                        oss << String::encode_codepoint(cp);
                    } else {
                        oss << format_number((*mat)(i, j));
                    }
                }
            }
            return oss.str();
        }

        case ValueType::NDARRAY: {
            const auto* nd = v->as_ndarray();
            const auto& shape = nd->shape;
            const Eigen::VectorXd* data = nd->data;

            if (shape.empty() || data->size() == 0) return "⍬";

            std::ostringstream oss;

            // For rank 3+, display as planes (last 2 dims) separated by blank lines
            // Higher dimensions add more blank line separators
            int rank = static_cast<int>(shape.size());
            int rows = shape[rank - 2];
            int cols = shape[rank - 1];
            int plane_size = rows * cols;

            // Number of planes (product of all but last 2 dimensions)
            int num_planes = 1;
            for (int i = 0; i < rank - 2; ++i) {
                num_planes *= shape[i];
            }

            for (int plane = 0; plane < num_planes; ++plane) {
                // Add blank lines between planes
                // More blank lines for higher dimension boundaries
                if (plane > 0) {
                    // Determine how many dimensions rolled over
                    int blanks = 1;
                    int p = plane;
                    for (int d = rank - 3; d >= 0; --d) {
                        if (p % shape[d] == 0) {
                            blanks++;
                            p /= shape[d];
                        } else {
                            break;
                        }
                    }
                    for (int b = 0; b < blanks; ++b) {
                        oss << "\n";
                    }
                }

                // Print this plane (rows x cols)
                int base = plane * plane_size;
                for (int i = 0; i < rows; ++i) {
                    if (i > 0) oss << "\n";
                    for (int j = 0; j < cols; ++j) {
                        if (j > 0) oss << " ";
                        int idx = base + i * cols + j;
                        if (v->is_char_data()) {
                            uint32_t cp = static_cast<uint32_t>((*data)(idx));
                            oss << String::encode_codepoint(cp);
                        } else {
                            oss << format_number((*data)(idx));
                        }
                    }
                }
            }
            return oss.str();
        }

        case ValueType::STRAND: {
            std::ostringstream oss;
            oss << "(";
            bool first = true;
            for (Value* elem : *v->data.strand) {
                if (!first) oss << " ";
                first = false;
                oss << format_value(elem);
            }
            oss << ")";
            return oss.str();
        }

        case ValueType::PRIMITIVE:
            return std::string("<primitive:") + (v->data.primitive_fn->name ? v->data.primitive_fn->name : "?") + ">";

        case ValueType::CLOSURE:
            return "<function>";

        case ValueType::OPERATOR:
            return std::string("<operator:") + (v->data.op->name ? v->data.op->name : "?") + ">";

        case ValueType::DEFINED_OPERATOR:
            return std::string("<defined-operator:") +
                   (v->data.defined_op_data->name ? v->data.defined_op_data->name->c_str() : "?") + ">";

        case ValueType::DERIVED_OPERATOR:
            return "<derived-operator>";

        case ValueType::CURRIED_FN:
            return "<curried-function>";

        default:
            return "<unknown>";
    }
}

// Stream output for Value pointer
std::ostream& operator<<(std::ostream& os, const Value* v) {
    os << format_value(v);
    return os;
}

// Stream output for Value reference
std::ostream& operator<<(std::ostream& os, const Value& v) {
    os << format_value(&v);
    return os;
}

// Return a human-readable type name for error messages and stack traces
std::string Value::type_name() const {
    switch (tag) {
        case ValueType::SCALAR: return "scalar";
        case ValueType::VECTOR: {
            std::string result = "vector";
            if (data.matrix) {
                result += "[" + std::to_string(data.matrix->rows()) + "]";
            }
            return result;
        }
        case ValueType::MATRIX: {
            std::string result = "matrix";
            if (data.matrix) {
                result += "[" + std::to_string(data.matrix->rows()) +
                          "×" + std::to_string(data.matrix->cols()) + "]";
            }
            return result;
        }
        case ValueType::NDARRAY: {
            std::string result = "ndarray[";
            if (data.ndarray) {
                for (size_t i = 0; i < data.ndarray->shape.size(); ++i) {
                    if (i > 0) result += "×";
                    result += std::to_string(data.ndarray->shape[i]);
                }
            }
            result += "]";
            return result;
        }
        case ValueType::STRING: return "string";
        case ValueType::STRAND: {
            std::string result = "strand";
            if (data.strand) {
                result += "[" + std::to_string(data.strand->size()) + "]";
            }
            return result;
        }
        case ValueType::PRIMITIVE:
            return std::string("primitive<") +
                   (data.primitive_fn && data.primitive_fn->name ? data.primitive_fn->name : "?") + ">";
        case ValueType::CLOSURE: return "closure";
        case ValueType::OPERATOR:
            return std::string("operator<") +
                   (data.op && data.op->name ? data.op->name : "?") + ">";
        case ValueType::DEFINED_OPERATOR:
            return std::string("defined-operator<") +
                   (data.defined_op_data && data.defined_op_data->name ? data.defined_op_data->name->c_str() : "?") + ">";
        case ValueType::DERIVED_OPERATOR: return "derived-operator";
        case ValueType::CURRIED_FN: {
            if (!data.curried_fn) return "curried-fn";
            switch (data.curried_fn->curry_type) {
                case CurryType::G_PRIME: return "curried-fn<G_PRIME>";
                case CurryType::DYADIC_CURRY: return "curried-fn<DYADIC_CURRY>";
                case CurryType::OPERATOR_CURRY: return "curried-fn<OPERATOR_CURRY>";
            }
            return "curried-fn";
        }
        default: return "unknown";
    }
}

} // namespace apl
