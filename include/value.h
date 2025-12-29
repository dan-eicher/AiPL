// Value - Tagged union representing all APL values using Eigen types

#pragma once

#include <Eigen/Dense>
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

// Primitive function - can have both monadic and dyadic forms
// Functions set machine->ctrl.value on success or push ThrowErrorK on error
// Axis is passed to support F[k] syntax; nullptr means no axis specified
struct PrimitiveFn {
    const char* name;  // For debugging
    void (*monadic)(Machine* m, Value* axis, Value* omega);           // Monadic form (can be nullptr)
    void (*dyadic)(Machine* m, Value* axis, Value* lhs, Value* rhs);  // Dyadic form (can be nullptr)
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
        const char* name;                // Operator name
        bool is_dyadic_operator;         // Takes 1 or 2 operands?
        bool is_ambivalent;              // Can be called monadically or dyadically?

        // Parameter names (for binding in environment)
        const char* left_operand_name;   // e.g., "FF" - always present
        const char* right_operand_name;  // e.g., "GG" - only for dyadic operators
        const char* left_arg_name;       // e.g., "A" - for ambivalent operators (nullptr if monadic-only)
        const char* right_arg_name;      // e.g., "B" - always present
        const char* result_name;         // e.g., "Z"

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

    // Union for value storage
    union Data {
        double scalar;              // For SCALAR
        Eigen::MatrixXd* matrix;    // For VECTOR and MATRIX (vectors stored as n×1)
        const char* string;         // For STRING (interned pointer, not owned)
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
    bool is_array() const { return tag == ValueType::VECTOR || tag == ValueType::MATRIX; }
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

    // Strand access
    std::vector<Value*>* as_strand() { return data.strand; }
    const std::vector<Value*>* as_strand() const { return data.strand; }

    // String access
    const char* as_string() const { return data.string; }

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
