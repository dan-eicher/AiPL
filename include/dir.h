// DIR - Definition-site Instantiation with Re-optimization
//
// At first dfn call, deep-clone the body, re-run StaticOptimizer with
// concrete argument types (e.g., ⍵ → TM_VECTOR), and cache the result.
// Second call with same argument type hits the cache — zero overhead.
//
// The SpecializationBackend is pluggable: CloningBackend (clone + re-optimize)
// is the first implementation.

#pragma once

#include "value.h"
#include "optimizer.h"
#include <unordered_map>
#include <functional>

namespace apl {

// Forward declarations
class Heap;
class Continuation;

// ---------------------------------------------------------------------------
// TypeSig — cache key for specialization
// ---------------------------------------------------------------------------

struct TypeSig {
    ValueType omega_type;
    ValueType alpha_type;
    bool has_alpha;

    bool operator==(const TypeSig& o) const {
        return omega_type == o.omega_type &&
               alpha_type == o.alpha_type &&
               has_alpha  == o.has_alpha;
    }
};

struct TypeSigHash {
    size_t operator()(const TypeSig& s) const {
        size_t h = std::hash<int>{}(static_cast<int>(s.omega_type));
        h ^= std::hash<int>{}(static_cast<int>(s.alpha_type)) << 8;
        h ^= std::hash<bool>{}(s.has_alpha) << 16;
        return h;
    }
};

// Specialization cache: TypeSig → specialized body (nullptr = already tried, no benefit)
using SpecCache = std::unordered_map<TypeSig, Continuation*, TypeSigHash>;

// Return type tracking: TypeSig → observed return ValueType
using ReturnTypeMap = std::unordered_map<TypeSig, ValueType, TypeSigHash>;

// ---------------------------------------------------------------------------
// Helper: ValueType → TypeMask
// ---------------------------------------------------------------------------

inline TypeMask tm_from_value_type(ValueType vt) {
    switch (vt) {
        case ValueType::SCALAR:           return TM_SCALAR;
        case ValueType::VECTOR:           return TM_VECTOR;
        case ValueType::MATRIX:           return TM_MATRIX;
        case ValueType::NDARRAY:          return TM_NDARRAY;
        case ValueType::STRING:           return TM_STRING;
        case ValueType::STRAND:           return TM_STRAND;
        case ValueType::PRIMITIVE:        return TM_PRIMITIVE;
        case ValueType::CLOSURE:          return TM_CLOSURE;
        case ValueType::OPERATOR:         return TM_OPERATOR;
        case ValueType::DEFINED_OPERATOR: return TM_DEF_OP;
        case ValueType::DERIVED_OPERATOR: return TM_DERIVED;
        case ValueType::CURRIED_FN:       return TM_CURRIED;
        default:                          return TM_TOP;
    }
}

// ---------------------------------------------------------------------------
// SpecializationBackend — abstract interface
// ---------------------------------------------------------------------------

class SpecializationBackend {
public:
    virtual ~SpecializationBackend() = default;

    // Specialize a body continuation tree with concrete argument types.
    // Returns the specialized body (GC-tracked), or nullptr if no improvement.
    virtual Continuation* specialize(
        Continuation* body, Heap* heap,
        ValueType omega_type, bool has_alpha, ValueType alpha_type) = 0;
};

// ---------------------------------------------------------------------------
// CloningBackend — deep-clone + re-optimize with StaticOptimizer
// ---------------------------------------------------------------------------

class CloningBackend : public SpecializationBackend {
public:
    Continuation* specialize(
        Continuation* body, Heap* heap,
        ValueType omega_type, bool has_alpha, ValueType alpha_type) override;
};

// ---------------------------------------------------------------------------
// Deep-clone a continuation tree (GC-tracked allocations)
// ---------------------------------------------------------------------------

Continuation* clone_tree(Continuation* root, Heap* heap);

// ---------------------------------------------------------------------------
// TypeDirectedK — dispatch continuation for DIR specialization
// ---------------------------------------------------------------------------
// Lives on ClosureData (lazy-initialized). FunctionCallK::invoke calls
// dispatch() directly with known arg types. The invoke() method exists for
// future JIT integration where TypeDirectedK would be embedded in the tree.

class TypeDirectedK : public Continuation {
public:
    SpecCache cache;            // TypeSig → specialized body (nullptr = no benefit)
    ReturnTypeMap returns;      // TypeSig → observed return ValueType

    Continuation* original_body;
    SpecializationBackend* backend;
    Heap* heap_ref;

    TypeDirectedK(Continuation* body, SpecializationBackend* be, Heap* h)
        : original_body(body), backend(be), heap_ref(h) {}
    ~TypeDirectedK() override {}

    // Fast dispatch: called from FunctionCallK::invoke with concrete arg types.
    // Returns the specialized body for this TypeSig, or original_body.
    Continuation* dispatch(ValueType omega, bool has_alpha, ValueType alpha);

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

// ---------------------------------------------------------------------------
// ReturnTypeRecordK — records return type after dfn body executes
// ---------------------------------------------------------------------------
// Pushed in the call frame when TypeDirectedK is active. On invoke, records
// machine->result->tag into the TypeDirectedK's return map. Result unchanged.

class ReturnTypeRecordK : public Continuation {
public:
    TypeDirectedK* tdk;
    TypeSig sig;

    ReturnTypeRecordK(TypeDirectedK* t, const TypeSig& s)
        : tdk(t), sig(s) {}
    ~ReturnTypeRecordK() override {}

    void mark(Heap* heap) override;
    void accept(ContinuationVisitor& v) override { v.visit(this); }

protected:
    void invoke(Machine* machine) override;
};

} // namespace apl
