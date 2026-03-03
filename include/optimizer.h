// Static optimizer for AiPL APL interpreter
//
// Implements a single-pass wBurg (Proebsting & Whaley) optimizer over the
// static continuation graph produced by the parser.  The traversal is
// bottom-up: each node is rewritten *after* its children are processed, so
// the final state of a child is available when the parent decides what to do.
// This is equivalent to 0-CFA abstract interpretation combined with BURS
// rewriting in a single recursive pass.
//
// Pattern categories implemented:
//   C  – constant folding for scalar arithmetic
//   O  – operator/operand pre-building (e.g. +/ → ValueK(DERIVED_OPERATOR))
//   F  – FinalizeK elimination when inner type is provably non-function
//
// The optimiser needs a Heap* to allocate new continuation and Value nodes.

#pragma once

#include "continuation.h"
#include "environment.h"
#include "heap.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace apl {

// ---------------------------------------------------------------------------
// Abstract type bitmask (0-CFA states)
// ---------------------------------------------------------------------------

using TypeMask = uint32_t;

constexpr TypeMask TM_BOT        = 0;           // ⊥  unreachable
constexpr TypeMask TM_SCALAR     = 1u <<  0;
constexpr TypeMask TM_VECTOR     = 1u <<  1;
constexpr TypeMask TM_MATRIX     = 1u <<  2;
constexpr TypeMask TM_NDARRAY    = 1u <<  3;
constexpr TypeMask TM_STRING     = 1u <<  4;
constexpr TypeMask TM_STRAND     = 1u <<  5;
constexpr TypeMask TM_PRIMITIVE  = 1u <<  6;
constexpr TypeMask TM_CLOSURE    = 1u <<  7;
constexpr TypeMask TM_OPERATOR   = 1u <<  8;
constexpr TypeMask TM_DEF_OP     = 1u <<  9;
constexpr TypeMask TM_DERIVED    = 1u << 10;
constexpr TypeMask TM_CURRIED    = 1u << 11;

// Composite masks
constexpr TypeMask TM_NUMERIC    = TM_SCALAR | TM_VECTOR | TM_MATRIX | TM_NDARRAY;
constexpr TypeMask TM_FN         = TM_PRIMITIVE | TM_CLOSURE | TM_DERIVED | TM_CURRIED;
constexpr TypeMask TM_OP         = TM_OPERATOR | TM_DEF_OP;
constexpr TypeMask TM_TOP        = 0xFFFFFFFFu;  // ⊤  unknown

// ---------------------------------------------------------------------------
// OptState – abstract value for a node
// ---------------------------------------------------------------------------

struct OptState {
    TypeMask mask      = TM_BOT;
    Value*   singleton = nullptr;  // non-null iff there is exactly one known value
};

// Abstract environment: variable name (std::string) → OptState
using AbsEnv = std::unordered_map<std::string, OptState>;

// Build an AbsEnv from a concrete environment (walks the full parent chain).
AbsEnv build_abs_env(Environment* env);

// Map a concrete Value* to its OptState (singleton set).
OptState opt_state_from_value(Value* v);

// ---------------------------------------------------------------------------
// StaticOptimizer
// ---------------------------------------------------------------------------

class StaticOptimizer {
public:
    // Run a single optimisation pass over the continuation tree rooted at
    // `root`.  Returns the (possibly new) root node.
    // `heap` is used to allocate ValueK and folded scalar Value nodes.
    // `abs_env` provides abstract types / singleton values for variable names.
    Continuation* run(Continuation* root, Heap* heap, const AbsEnv& abs_env);

private:
    Heap*  heap_ = nullptr;
    AbsEnv env_;

    // Internal result type: rewritten continuation + its abstract state
    struct Rewrite {
        Continuation* kont;
        OptState      state;
    };

    // Main dispatch – handles all continuation types
    Rewrite rewrite(Continuation* k);

    // Per-type rewrite handlers
    Rewrite rewrite_literal(LiteralK* k);
    Rewrite rewrite_literal_strand(LiteralStrandK* k);
    Rewrite rewrite_lookup(LookupK* k);
    Rewrite rewrite_juxtapose(JuxtaposeK* k);
    Rewrite rewrite_monadic(MonadicK* k);
    Rewrite rewrite_dyadic(DyadicK* k);
    Rewrite rewrite_finalize(FinalizeK* k);
    Rewrite rewrite_closure_literal(ClosureLiteralK* k);
    Rewrite rewrite_derived_op(DerivedOperatorK* k);
    Rewrite rewrite_assign(AssignK* k);
    Rewrite rewrite_seq(SeqK* k);

    // ---------------------------------------------------------------------------
    // Category C – constant folding helpers
    // ---------------------------------------------------------------------------

    // Returns a folded scalar Value* or nullptr if this op+values cannot be folded.
    Value* fold_dyadic(const std::string& op, double l, double r);
    Value* fold_monadic(const std::string& op, double v);
};

} // namespace apl
