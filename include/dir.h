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
// DIR lookup-or-specialize — the main entry point
// ---------------------------------------------------------------------------

// Look up a specialized body for the given closure + argument types.
// On cache miss, calls backend->specialize() and caches the result.
// Returns the specialized body, or the original body if no improvement.
Continuation* dir_lookup_or_specialize(
    Value* closure_val, const TypeSig& sig,
    Heap* heap, SpecializationBackend* backend);

} // namespace apl
