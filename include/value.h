// Value - Tagged union representing all APL values using Eigen types

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <ostream>
#include <vector>

namespace apl {

// Forward declarations
class Value;
class Continuation;
class Environment;
class Heap;
class Machine;

// GCObject base class - defined here to avoid circular dependencies
// This is the base for all garbage-collected objects
class GCObject {
public:
    bool marked;                // Mark bit for GC
    bool in_old_generation;     // True if in old generation

    GCObject() : marked(false), in_old_generation(false) {}
    virtual ~GCObject() = default;

    // Mark all objects referenced by this object for GC
    virtual void mark(Heap* heap) = 0;
};

// String - GC-managed interned string with UTF-8 support
// Used for variable names, operator names, and string values
class String : public GCObject {
    std::string data_;

public:
    explicit String(const char* s) : GCObject(), data_(s) {}
    explicit String(std::string s) : GCObject(), data_(std::move(s)) {}

    // Raw access
    const char* c_str() const { return data_.c_str(); }
    const std::string& str() const { return data_; }
    size_t byte_length() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    // UTF-8 aware instance operations
    size_t length() const;                    // Codepoint count
    uint32_t at(size_t index) const;          // Get codepoint at index (0-based)
    std::vector<uint32_t> to_codepoints() const;  // All codepoints as vector

    // Comparison (for identity, compare String* pointers; for content, use these)
    bool operator==(const String& other) const { return data_ == other.data_; }
    bool operator==(const char* s) const { return data_ == s; }
    bool operator==(const std::string& s) const { return data_ == s; }

    // GC support - String is a leaf node, nothing to mark
    void mark(Heap* heap) override;

    // =========================================================================
    // Static UTF-8 utilities (consolidate all UTF-8 handling here)
    // =========================================================================

    // Get byte length of UTF-8 sequence starting with given byte
    static int utf8_sequence_length(unsigned char lead_byte);

    // Decode one codepoint from UTF-8, advancing the pointer
    static uint32_t decode_one(const unsigned char*& p);

    // Encode a single codepoint to UTF-8
    static std::string encode_codepoint(uint32_t cp);

    // Encode a vector of codepoints to UTF-8 string
    static std::string encode_codepoints(const std::vector<uint32_t>& cps);

    // Decode UTF-8 string to codepoints
    static std::vector<uint32_t> decode_utf8(const char* s);
    static std::vector<uint32_t> decode_utf8(const std::string& s);

    // UTF-8 constants
    static constexpr uint32_t MAX_1BYTE = 0x7F;
    static constexpr uint32_t MAX_2BYTE = 0x7FF;
    static constexpr uint32_t MAX_3BYTE = 0xFFFF;
    static constexpr uint32_t MAX_4BYTE = 0x10FFFF;
};

// Primitive function - can have both monadic and dyadic forms
// Functions set machine->ctrl.value on success or push ThrowErrorK on error
// Axis is passed to support F[k] syntax; nullptr means no axis specified
struct PrimitiveFn {
    const char* name;  // For debugging
    void (*monadic)(Machine* m, Value* axis, Value* omega);           // Monadic form (can be nullptr)
    void (*dyadic)(Machine* m, Value* axis, Value* lhs, Value* rhs);  // Dyadic form (can be nullptr)
    bool is_pervasive;  // True for scalar functions that auto-penetrate nested arrays
};

// Operator structures - operators take functions and return derived functions
// Monadic operator: takes one function operand
// Dyadic operator: takes two function operands
// Axis is passed to support f/[k] syntax; nullptr means no axis specified
struct PrimitiveOp {
    const char* name;  // For debugging
    // Monadic operator: f OP B  (e.g., f/ for reduce, f¨ for each)
    void (*monadic)(Machine* m, Value* axis, Value* f, Value* omega);
    // Dyadic operator: A f OP g B  (e.g., f.g for inner product)
    void (*dyadic)(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs);
};

// Value type enumeration
enum class ValueType {
    SCALAR,     // Single numeric value (double)
    VECTOR,     // 1D array stored as n×1 matrix
    MATRIX,     // 2D array
    NDARRAY,    // N-dimensional array (rank 3+), flat storage with shape vector
    STRING,     // Character string (interned pointer)
    STRAND,     // Nested array (std::vector<Value*>) - can hold any values including other strands
    PRIMITIVE,  // Primitive function (C function pointer)
    CLOSURE,    // User-defined function (continuation graph)
    OPERATOR,   // Higher-order operator (takes functions, returns derived functions)
    DEFINED_OPERATOR,  // User-defined operator (closure-like)
    DERIVED_OPERATOR,  // Result of applying operator to first operand (G2 grammar)
    CURRIED_FN  // Result of applying function to first argument (G2 grammar currying)
};

// Value class - tagged union for all APL values
class Value : public GCObject {
private:
    // Only Heap can allocate/deallocate Value objects
    friend class Heap;

    // Private new/delete operators enforce heap-only allocation
    void* operator new(size_t size) { return ::operator new(size); }
    void operator delete(void* ptr) { ::operator delete(ptr); }

public:
    ValueType tag;

    // User-defined operator data
    struct DefinedOperatorData {
        Continuation* body;              // The operator's body (continuation graph)
        String* name;                    // Operator name (interned)
        bool is_dyadic_operator;         // Takes 1 or 2 operands?
        bool is_ambivalent;              // Can be called monadically or dyadically?

        // Parameter names (for binding in environment, interned)
        String* left_operand_name;       // e.g., "FF" - always present
        String* right_operand_name;      // e.g., "GG" - only for dyadic operators
        String* left_arg_name;           // e.g., "A" - for ambivalent operators (nullptr if monadic-only)
        String* right_arg_name;          // e.g., "B" - always present
        String* result_name;             // e.g., "Z"

        Environment* lexical_env;        // Captured environment (like closures)
    };

    // Data structures for G2 grammar types
    // Holds result of applying operator to first operand - works for both primitive and defined ops
    struct DerivedOperatorData {
        PrimitiveOp* primitive_op;       // For primitive operators (or nullptr)
        DefinedOperatorData* defined_op; // For defined operators (or nullptr)
        Value* first_operand;            // The first operand (always present)
        Value* operator_value;           // For defined ops: the DEFINED_OPERATOR Value (for ∇ binding)
    };

    // Curry types for CURRIED_FN values
    enum class CurryType {
        G_PRIME,         // g' transformation for overloaded functions (can compose)
        DYADIC_CURRY,    // Simple dyadic function curry (fn + right arg, waiting for left)
        OPERATOR_CURRY   // Dyadic operator curry (stores second operand without applying)
    };

    struct CurriedFnData {
        Value* fn;                 // The function being curried (can be PRIMITIVE, CLOSURE, or DERIVED_OPERATOR)
        Value* first_arg;          // The first argument (or second operand for OPERATOR_CURRY)
        CurryType curry_type;      // Type of curry determines unwrapping behavior
        Value* axis;               // Optional axis specification (from F[k] syntax), nullptr if none
    };

    // Closure data - user-defined function with niladic tracking
    struct ClosureData {
        Continuation* body;        // The function body (continuation graph)
        bool is_niladic;           // True if function doesn't reference ⍵ or ⍺
    };

    // NDARRAY metadata - shape and precomputed strides for fast indexing
    struct NDArrayData {
        Eigen::VectorXd* data;       // Flat contiguous storage
        std::vector<int> shape;      // e.g., {2, 3, 4} for 2×3×4 array
        std::vector<int> strides;    // Precomputed: {12, 4, 1} for row-major
    };

    // Union for value storage
    union Data {
        double scalar;              // For SCALAR
        Eigen::MatrixXd* matrix;    // For VECTOR and MATRIX (vectors stored as n×1)
        NDArrayData* ndarray;       // For NDARRAY (N-dimensional, rank 3+)
        String* string;             // For STRING (GC-managed, interned)
        std::vector<Value*>* strand;  // For STRAND (nested array, can contain any values)
        PrimitiveFn* primitive_fn;  // For PRIMITIVE (built-in function)
        ClosureData* closure;       // For CLOSURE (user-defined function with niladic flag)
        PrimitiveOp* op;            // For OPERATOR (primitive)
        DefinedOperatorData* defined_op_data;  // For DEFINED_OPERATOR (user-defined)
        DerivedOperatorData* derived_op;  // For DERIVED_OPERATOR
        CurriedFnData* curried_fn;  // For CURRIED_FN

        // Union constructors
        Data() : scalar(0.0) {}
        ~Data() {}  // Manual cleanup required
    } data;

    // Character data flag - true if this vector/matrix contains character codepoints
    bool is_character_data_;

    // Constructors (public so factory methods work, but new/delete are private)
    Value() : GCObject(), tag(ValueType::SCALAR), is_character_data_(false) {
        data.scalar = 0.0;
        promoted_matrix_ = nullptr;
    }
    ~Value();

    // Type checking methods
    bool is_scalar() const { return tag == ValueType::SCALAR; }
    bool is_vector() const { return tag == ValueType::VECTOR; }
    bool is_matrix() const { return tag == ValueType::MATRIX; }
    bool is_ndarray() const { return tag == ValueType::NDARRAY; }
    bool is_array() const { return tag == ValueType::VECTOR || tag == ValueType::MATRIX || tag == ValueType::NDARRAY; }
    bool is_strand() const { return tag == ValueType::STRAND; }
    bool is_string() const { return tag == ValueType::STRING; }
    bool is_primitive() const { return tag == ValueType::PRIMITIVE; }
    bool is_closure() const { return tag == ValueType::CLOSURE; }
    bool is_function() const { return tag == ValueType::PRIMITIVE || tag == ValueType::CLOSURE || tag == ValueType::CURRIED_FN || tag == ValueType::DERIVED_OPERATOR; }
    bool is_operator() const { return tag == ValueType::OPERATOR || tag == ValueType::DEFINED_OPERATOR || tag == ValueType::DERIVED_OPERATOR; }
    bool is_defined_operator() const { return tag == ValueType::DEFINED_OPERATOR; }
    bool is_derived_operator() const { return tag == ValueType::DERIVED_OPERATOR; }
    bool is_curried_fn() const { return tag == ValueType::CURRIED_FN; }
    // G2 grammar: "bas" type = basic values (scalars, vectors, matrices, strings, strands)
    bool is_basic_value() const { return is_scalar() || is_array() || is_string() || is_strand(); }
    // Rectangular arrays (numeric, not nested) - excludes STRAND
    bool is_rectangular() const { return is_scalar() || tag == ValueType::VECTOR || tag == ValueType::MATRIX || tag == ValueType::NDARRAY; }

    // Strand access
    std::vector<Value*>* as_strand() { return data.strand; }
    const std::vector<Value*>* as_strand() const { return data.strand; }

    // NDARRAY access
    NDArrayData* as_ndarray() { return data.ndarray; }
    const NDArrayData* as_ndarray() const { return data.ndarray; }
    const std::vector<int>& ndarray_shape() const { return data.ndarray->shape; }
    const std::vector<int>& ndarray_strides() const { return data.ndarray->strides; }
    Eigen::VectorXd* ndarray_data() { return data.ndarray->data; }
    const Eigen::VectorXd* ndarray_data() const { return data.ndarray->data; }
    // Element access by multi-dimensional index (0-based)
    double& ndarray_at(const std::vector<int>& indices);
    double ndarray_at(const std::vector<int>& indices) const;
    // Compute linear index from multi-dimensional indices using strides
    int ndarray_linear_index(const std::vector<int>& indices) const;

    // String access
    String* as_string() const { return data.string; }

    // Character data queries and conversion
    bool is_char_data() const { return is_character_data_; }
    void set_char_data(bool v) { is_character_data_ = v; }
    Value* to_char_vector(Heap* heap);      // STRING → char VECTOR (returns this if already array)
    Value* to_string_value(Heap* heap);     // char VECTOR → STRING (returns this if already STRING)

    // Shape queries (for arrays)
    int rank() const;           // 0 for scalar, 1 for vector, 2 for matrix
    int rows() const;           // Number of rows (for arrays)
    int cols() const;           // Number of columns (for arrays)
    int size() const;           // Total number of elements

    // Access methods
    double as_scalar() const;                   // Get scalar value (error if not scalar)
    Eigen::MatrixXd* as_matrix();              // Get matrix (with lazy scalar promotion)
    const Eigen::MatrixXd* as_matrix() const;  // Const version

    // GC support - mark all objects this Value references (override from GCObject)
    void mark(Heap* heap) override;

    // Description for error messages and stack traces
    std::string type_name() const;

private:
    // Helper for lazy scalar promotion
    mutable Eigen::MatrixXd* promoted_matrix_;  // Cached promoted matrix for scalars

    void cleanup();  // Internal cleanup helper
};

// Value formatting for display
// Returns APL-style representation: "1 2 3" for vectors, multi-line for matrices
std::string format_value(const Value* v);

// Stream output operators
std::ostream& operator<<(std::ostream& os, const Value* v);
std::ostream& operator<<(std::ostream& os, const Value& v);

} // namespace apl
