// Value - Tagged union representing all APL values using Eigen types

#pragma once

#include <Eigen/Dense>

namespace apl {

// Forward declarations
class Value;
class Continuation;
class APLHeap;

// Primitive function - can have both monadic and dyadic forms
struct PrimitiveFn {
    const char* name;  // For debugging
    Value* (*monadic)(Value* omega);           // Monadic form (can be nullptr)
    Value* (*dyadic)(Value* lhs, Value* rhs);  // Dyadic form (can be nullptr)
};

// Forward declaration for operators
struct PrimitiveOp;

// Value type enumeration
enum class ValueType {
    SCALAR,     // Single numeric value (double)
    VECTOR,     // 1D array stored as n×1 matrix
    MATRIX,     // 2D array
    PRIMITIVE,  // Primitive function (C function pointer)
    CLOSURE,    // User-defined function (continuation graph)
    OPERATOR    // Higher-order operator
};

// Value class - tagged union for all APL values
class Value {
public:
    ValueType tag;

    // Union for value storage
    union Data {
        double scalar;              // For SCALAR
        Eigen::MatrixXd* matrix;    // For VECTOR and MATRIX (vectors stored as n×1)
        PrimitiveFn* primitive_fn;  // For PRIMITIVE (built-in function)
        Continuation* closure;      // For CLOSURE (user-defined function body)
        PrimitiveOp* op;            // For OPERATOR

        // Union constructors
        Data() : scalar(0.0) {}
        ~Data() {}  // Manual cleanup required
    } data;

    // GC metadata
    bool marked;                // Mark bit for GC
    bool in_old_generation;     // True if in old generation

    // Constructors
    Value() : tag(ValueType::SCALAR), marked(false), in_old_generation(false) {
        data.scalar = 0.0;
        promoted_matrix_ = nullptr;
    }
    ~Value();

    // Type checking methods
    bool is_scalar() const { return tag == ValueType::SCALAR; }
    bool is_vector() const { return tag == ValueType::VECTOR; }
    bool is_matrix() const { return tag == ValueType::MATRIX; }
    bool is_array() const { return tag == ValueType::VECTOR || tag == ValueType::MATRIX; }
    bool is_primitive() const { return tag == ValueType::PRIMITIVE; }
    bool is_closure() const { return tag == ValueType::CLOSURE; }
    bool is_function() const { return tag == ValueType::PRIMITIVE || tag == ValueType::CLOSURE; }
    bool is_operator() const { return tag == ValueType::OPERATOR; }

    // Shape queries (for arrays)
    int rank() const;           // 0 for scalar, 1 for vector, 2 for matrix
    int rows() const;           // Number of rows (for arrays)
    int cols() const;           // Number of columns (for arrays)
    int size() const;           // Total number of elements

    // Access methods
    double as_scalar() const;                   // Get scalar value (error if not scalar)
    Eigen::MatrixXd* as_matrix();              // Get matrix (with lazy scalar promotion)
    const Eigen::MatrixXd* as_matrix() const;  // Const version

    // Factory methods for creating values
    static Value* from_scalar(double d);
    static Value* from_vector(const Eigen::VectorXd& v);     // Zero-copy wrapping
    static Value* from_matrix(const Eigen::MatrixXd& m);
    static Value* from_primitive(PrimitiveFn* fn);
    static Value* from_closure(Continuation* body);
    static Value* from_operator(PrimitiveOp* op);

    // GC support - mark all objects this Value references
    void mark_references(APLHeap* heap);

private:
    // Helper for lazy scalar promotion
    mutable Eigen::MatrixXd* promoted_matrix_;  // Cached promoted matrix for scalars

    void cleanup();  // Internal cleanup helper
};

} // namespace apl
