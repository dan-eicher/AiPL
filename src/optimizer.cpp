// Static optimizer – wBurg single-pass continuation graph rewriter
//
// Traversal is purely bottom-up: we recurse into children first, obtain their
// abstract states, then decide whether to rewrite the parent node.  This is
// equivalent to the Compose' procedure from Proebsting & Whaley "One-Pass,
// Optimal Tree Parsing – With Or Without Trees" (wBurg).
//
// Pattern categories:
//   C1  – DyadicK(op, LiteralK(a), LiteralK(b))  →  ValueK(a op b)
//   C2  – MonadicK(op, LiteralK(v))               →  ValueK(op v)
//   O1  – DerivedOperatorK(op, primitive_fn, nil) →  ValueK(DERIVED_OPERATOR)
//   F1  – FinalizeK(inner) where inner ∉ TM_FN    →  inner  (eliminated)
//   D1  – FinalizeK(JuxtaposeK(fn, arg))          →  MonadicCallK(fn, arg)
//   D2  – JuxtaposeK(l:BASIC, JuxtaposeK(fn, r))  →  DyadicCallK(fn, l, r)
//   D3  – recursive D1 through ensure_finalized    (chains -⌊x, ⍴⍴A, etc.)

#include "optimizer.h"
#include "continuation.h"
#include "environment.h"
#include "heap.h"
#include "value.h"
#include <cmath>
#include <cassert>

namespace apl {

// ---------------------------------------------------------------------------
// Abstract apply tables
// ---------------------------------------------------------------------------

static TypeMask abstract_apply_monadic(const std::string& name, TypeMask arg) {
    // Pervasive: preserve numeric type
    if (name == "+" || name == "-" || name == "×" || name == "÷" ||
        name == "*" || name == "|" || name == "⌈" || name == "⌊" ||
        name == "⍟" || name == "!" || name == "○" || name == "?" ||
        name == "~") {
        if ((arg & TM_NUMERIC) && !(arg & ~TM_NUMERIC)) return arg;
        return TM_TOP;
    }
    // Shape-preserving structural
    if (name == "⌽" || name == "⊖" || name == "⍉") {
        if ((arg & TM_NUMERIC) && !(arg & ~TM_NUMERIC)) return arg;
        return TM_TOP;
    }
    // Always VECTOR
    if (name == "⍴" || name == "," || name == "∊" || name == "⍋" || name == "⍒")
        return TM_VECTOR;
    // Always SCALAR
    if (name == "≢" || name == "≡")
        return TM_SCALAR;
    // Always STRING
    if (name == "⍕")
        return TM_STRING;
    // Always MATRIX
    if (name == "⍪")
        return TM_MATRIX;
    // Iota
    if (name == "⍳") {
        if (arg == TM_SCALAR) return TM_VECTOR;
        return TM_TOP;
    }
    // Identity
    if (name == "⊣" || name == "⊢")
        return arg;
    // Enclose
    if (name == "⊂")
        return TM_STRAND;
    // Default
    return TM_TOP;
}

static TypeMask abstract_apply_dyadic(const std::string& name, TypeMask left, TypeMask right) {
    // Broadcast rule for pervasive dyadic
    auto broadcast = [](TypeMask l, TypeMask r) -> TypeMask {
        if (!(l & TM_NUMERIC) || (l & ~TM_NUMERIC) ||
            !(r & TM_NUMERIC) || (r & ~TM_NUMERIC))
            return TM_TOP;
        if (l == r) return l;            // vec + vec = vec
        if (l == TM_SCALAR) return r;    // scalar extends
        if (r == TM_SCALAR) return l;    // scalar extends
        return l | r;                    // conservative union
    };

    // Pervasive
    if (name == "+" || name == "-" || name == "×" || name == "÷" ||
        name == "*" || name == "|" || name == "⌈" || name == "⌊" ||
        name == "⍟" || name == "!" || name == "○" || name == "?" ||
        name == "=" || name == "≠" || name == "<" || name == ">" ||
        name == "≤" || name == "≥" || name == "∧" || name == "∨" ||
        name == "⍲" || name == "⍱")
        return broadcast(left, right);
    // Rotate
    if (name == "⌽" || name == "⊖") {
        if ((right & TM_NUMERIC) && !(right & ~TM_NUMERIC)) return right;
        return TM_TOP;
    }
    // Match
    if (name == "≡")
        return TM_SCALAR;
    // Format
    if (name == "⍕")
        return TM_STRING;
    // Identity
    if (name == "⊣") return left;
    if (name == "⊢") return right;
    // Without
    if (name == "~")
        return TM_VECTOR;
    // Union / Intersect
    if (name == "∪" || name == "∩")
        return TM_VECTOR;
    // Default
    return TM_TOP;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

OptState opt_state_from_value(Value* v) {
    if (!v) return {TM_BOT, nullptr};
    TypeMask mask = TM_BOT;
    switch (v->tag) {
        case ValueType::SCALAR:            mask = TM_SCALAR;    break;
        case ValueType::VECTOR:            mask = TM_VECTOR;    break;
        case ValueType::MATRIX:            mask = TM_MATRIX;    break;
        case ValueType::NDARRAY:           mask = TM_NDARRAY;   break;
        case ValueType::STRING:            mask = TM_STRING;    break;
        case ValueType::STRAND:            mask = TM_STRAND;    break;
        case ValueType::PRIMITIVE:         mask = TM_PRIMITIVE; break;
        case ValueType::CLOSURE:           mask = TM_CLOSURE;   break;
        case ValueType::OPERATOR:          mask = TM_OPERATOR;  break;
        case ValueType::DEFINED_OPERATOR:  mask = TM_DEF_OP;    break;
        case ValueType::DERIVED_OPERATOR:  mask = TM_DERIVED;   break;
        case ValueType::CURRIED_FN:        mask = TM_CURRIED;   break;
        default:                           mask = TM_TOP;       break;
    }
    return {mask, v};
}

AbsEnv build_abs_env(Environment* env) {
    AbsEnv result;
    if (!env) return result;
    // Walk parent chain first so that local bindings take precedence.
    if (env->parent) {
        result = build_abs_env(env->parent);
    }
    for (auto& [name, val] : env->bindings) {
        if (name && val) {
            result[name->str()] = opt_state_from_value(val);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// StaticOptimizer::run
// ---------------------------------------------------------------------------

Continuation* StaticOptimizer::run(Continuation* root, Heap* heap,
                                    const AbsEnv& abs_env) {
    heap_ = heap;
    env_  = abs_env;
    state_cache_.clear();
    if (!root) return root;
    return rewrite(root).kont;
}

// ---------------------------------------------------------------------------
// Main dispatch
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite(Continuation* k) {
    if (!k) return {nullptr, {TM_BOT, nullptr}};

    // Check state cache — if already processed, return immediately
    auto cached = state_cache_.find(k);
    if (cached != state_cache_.end()) {
        return {k, cached->second};
    }

    Rewrite result;

    // Already optimised (e.g. a ValueK created during this same pass)
    if (auto* vk = dynamic_cast<ValueK*>(k)) {
        result = {vk, opt_state_from_value(vk->value)};
    }
    else if (auto* lit = dynamic_cast<LiteralK*>(k)) {
        result = rewrite_literal(lit);
    }
    else if (auto* ls = dynamic_cast<LiteralStrandK*>(k)) {
        result = rewrite_literal_strand(ls);
    }
    else if (auto* lk = dynamic_cast<LookupK*>(k)) {
        result = rewrite_lookup(lk);
    }
    else if (auto* jk = dynamic_cast<JuxtaposeK*>(k)) {
        result = rewrite_juxtapose(jk);
    }
    else if (auto* mk = dynamic_cast<MonadicK*>(k)) {
        result = rewrite_monadic(mk);
    }
    else if (auto* dk = dynamic_cast<DyadicK*>(k)) {
        result = rewrite_dyadic(dk);
    }
    else if (auto* fk = dynamic_cast<FinalizeK*>(k)) {
        result = rewrite_finalize(fk);
    }
    else if (auto* ck = dynamic_cast<ClosureLiteralK*>(k)) {
        result = rewrite_closure_literal(ck);
    }
    else if (auto* dok = dynamic_cast<DerivedOperatorK*>(k)) {
        result = rewrite_derived_op(dok);
    }
    else if (auto* ak = dynamic_cast<AssignK*>(k)) {
        result = rewrite_assign(ak);
    }
    else if (auto* sk = dynamic_cast<SeqK*>(k)) {
        result = rewrite_seq(sk);
    }
    // Optimizer-produced nodes — recurse into children and infer types
    else if (auto* mck = dynamic_cast<MonadicCallK*>(k)) {
        result = rewrite_monadic_call(mck);
    }
    else if (auto* dck = dynamic_cast<DyadicCallK*>(k)) {
        result = rewrite_dyadic_call(dck);
    }
    else {
        // Unknown node type – return unchanged with TM_TOP
        result = {k, {TM_TOP, nullptr}};
    }

    // Cache state for both original and replacement nodes
    state_cache_[k] = result.state;
    if (result.kont != k) {
        state_cache_[result.kont] = result.state;
    }
    return result;
}

// ---------------------------------------------------------------------------
// State lookup and D3 finalization helper
// ---------------------------------------------------------------------------

OptState StaticOptimizer::lookup_state(Continuation* k) {
    if (!k) return {TM_BOT, nullptr};
    auto it = state_cache_.find(k);
    if (it != state_cache_.end()) return it->second;
    // Not cached — should only happen for newly created nodes
    return {TM_TOP, nullptr};
}

Continuation* StaticOptimizer::finalize_if_needed(Continuation* c, TypeMask m) {
    // Known basic — no curry possible, no wrapping needed
    bool known_basic = (m != TM_BOT && m != TM_TOP &&
                        (m & TM_BASIC) && !(m & ~TM_BASIC));
    if (known_basic) return c;

    // D3: if c is a JuxtaposeK with a CALLABLE function child,
    // directly emit MonadicCallK using cached states — no re-walking.
    if (auto* jux = dynamic_cast<JuxtaposeK*>(c)) {
        TypeMask lm = lookup_state(jux->left).mask;
        TypeMask rm = lookup_state(jux->right).mask;

        auto is_callable = [](TypeMask mask) -> bool {
            return mask != TM_BOT && mask != TM_TOP &&
                   (mask & TM_CALLABLE) && !(mask & ~TM_CALLABLE);
        };

        // Case (a): left is function, right is argument → fn(arg)
        if (is_callable(lm)) {
            Continuation* fin_arg = finalize_if_needed(jux->right, rm);
            auto* mcall = heap_->allocate<MonadicCallK>(jux->left, fin_arg);
            if (c->has_location()) mcall->set_location(c->line(), c->column());
            // Cache state for the newly created MonadicCallK
            OptState fn_st = lookup_state(jux->left);
            TypeMask arg_mask = lookup_state(fin_arg).mask;
            TypeMask res = TM_TOP;
            if (fn_st.singleton && fn_st.mask == TM_PRIMITIVE)
                res = abstract_apply_monadic(fn_st.singleton->data.primitive_fn->name, arg_mask);
            state_cache_[mcall] = {res, nullptr};
            return mcall;
        }
        // Case (b): left is basic, right is function → fn(left)
        if (is_callable(rm)) {
            bool left_basic = (lm != TM_BOT && lm != TM_TOP &&
                               (lm & TM_BASIC) && !(lm & ~TM_BASIC));
            if (left_basic) {
                auto* mcall = heap_->allocate<MonadicCallK>(jux->right, jux->left);
                if (c->has_location()) mcall->set_location(c->line(), c->column());
                OptState fn_st = lookup_state(jux->right);
                TypeMask res = TM_TOP;
                if (fn_st.singleton && fn_st.mask == TM_PRIMITIVE)
                    res = abstract_apply_monadic(fn_st.singleton->data.primitive_fn->name, lm);
                state_cache_[mcall] = {res, nullptr};
                return mcall;
            }
        }
    }

    // Fallback: wrap in FinalizeK
    auto* fin = heap_->allocate<FinalizeK>(c, true);
    if (c->has_location()) fin->set_location(c->line(), c->column());
    return fin;
}

// ---------------------------------------------------------------------------
// Leaf nodes
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_literal(LiteralK* k) {
    // A LiteralK is already essentially a constant; we expose it as a
    // singleton so that parent nodes (DyadicK, MonadicK) can fold it.
    // We allocate the scalar Value here so the singleton is valid.
    Value* scalar = heap_->allocate_scalar(k->literal_value);
    return {k, {TM_SCALAR, scalar}};
}

StaticOptimizer::Rewrite StaticOptimizer::rewrite_literal_strand(LiteralStrandK* k) {
    return {k, {TM_VECTOR, k->vector_value}};
}

StaticOptimizer::Rewrite StaticOptimizer::rewrite_lookup(LookupK* k) {
    if (!k->var_name) return {k, {TM_TOP, nullptr}};
    auto it = env_.find(k->var_name->str());
    if (it != env_.end()) {
        return {k, it->second};
    }
    return {k, {TM_TOP, nullptr}};
}

// ---------------------------------------------------------------------------
// Category C – constant folding helpers
// ---------------------------------------------------------------------------

Value* StaticOptimizer::fold_dyadic(const std::string& op, double l, double r) {
    double result;
    if      (op == "+")  result = l + r;
    else if (op == "-")  result = l - r;
    else if (op == "×")  result = l * r;
    else if (op == "÷") {
        if (r == 0.0) return nullptr;   // DivByZero – leave to runtime
        result = l / r;
    }
    else if (op == "*")  result = std::pow(l, r);
    else if (op == "⌈")  result = std::max(l, r);
    else if (op == "⌊")  result = std::min(l, r);
    else if (op == "<")  result = (l <  r) ? 1.0 : 0.0;
    else if (op == "≤")  result = (l <= r) ? 1.0 : 0.0;
    else if (op == "=")  result = (l == r) ? 1.0 : 0.0;
    else if (op == "≠")  result = (l != r) ? 1.0 : 0.0;
    else if (op == ">")  result = (l >  r) ? 1.0 : 0.0;
    else if (op == "≥")  result = (l >= r) ? 1.0 : 0.0;
    else if (op == "∧")  result = ((l != 0.0) && (r != 0.0)) ? 1.0 : 0.0;
    else if (op == "∨")  result = ((l != 0.0) || (r != 0.0)) ? 1.0 : 0.0;
    else return nullptr;   // Unknown or unsafe op – don't fold
    return heap_->allocate_scalar(result);
}

Value* StaticOptimizer::fold_monadic(const std::string& op, double v) {
    double result;
    if      (op == "-")  result = -v;
    else if (op == "+")  result =  v;              // monadic + is identity
    else if (op == "×")  result = (v > 0.0) ? 1.0 : (v < 0.0) ? -1.0 : 0.0;  // signum
    else if (op == "⌈")  result = std::ceil(v);
    else if (op == "⌊")  result = std::floor(v);
    else if (op == "|")  result = std::abs(v);
    else if (op == "*")  result = std::exp(v);     // e^v
    else if (op == "~") {
        // Boolean NOT: only safe for 0/1
        if (v == 0.0) result = 1.0;
        else if (v == 1.0) result = 0.0;
        else return nullptr;  // Non-boolean – runtime will signal error
    }
    else if (op == "⍟") {
        if (v <= 0.0) return nullptr;  // Domain error – leave to runtime
        result = std::log(v);
    }
    else return nullptr;
    return heap_->allocate_scalar(result);
}

// ---------------------------------------------------------------------------
// Monadic expression  (C2)
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_monadic(MonadicK* k) {
    auto [new_operand, op_state] = rewrite(k->operand);

    // C2 – monadic scalar fold
    if (op_state.mask == TM_SCALAR && op_state.singleton) {
        double v = op_state.singleton->data.scalar;
        std::string op = k->op_name->str();
        Value* folded = fold_monadic(op, v);
        if (folded) {
            return {heap_->allocate<ValueK>(folded), opt_state_from_value(folded)};
        }
    }

    if (new_operand != k->operand) k->operand = new_operand;

    // Abstract apply: infer result type from primitive name + argument type
    TypeMask result_mask = abstract_apply_monadic(k->op_name->str(), op_state.mask);
    return {k, {result_mask, nullptr}};
}

// ---------------------------------------------------------------------------
// Dyadic expression  (C1)
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_dyadic(DyadicK* k) {
    auto [new_left,  left_state]  = rewrite(k->left);
    auto [new_right, right_state] = rewrite(k->right);

    // C1 – dyadic scalar fold
    if (left_state.mask == TM_SCALAR && left_state.singleton &&
        right_state.mask == TM_SCALAR && right_state.singleton) {
        double l = left_state.singleton->data.scalar;
        double r = right_state.singleton->data.scalar;
        std::string op = k->op_name->str();
        Value* folded = fold_dyadic(op, l, r);
        if (folded) {
            return {heap_->allocate<ValueK>(folded), opt_state_from_value(folded)};
        }
    }

    if (new_left  != k->left)  k->left  = new_left;
    if (new_right != k->right) k->right = new_right;

    // Abstract apply: infer result type from primitive name + argument types
    TypeMask result_mask = abstract_apply_dyadic(k->op_name->str(), left_state.mask, right_state.mask);
    return {k, {result_mask, nullptr}};
}

// ---------------------------------------------------------------------------
// FinalizeK  (D1, F1)
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_finalize(FinalizeK* k) {
    auto [new_inner, inner_state] = rewrite(k->inner);

    // D1 – Monadic call at finalization boundary
    // FinalizeK(JuxtaposeK(fn, arg)) → MonadicCallK(fn, arg)
    // FinalizeK proves no left argument will arrive, so the G_PRIME curry
    // can be replaced with a direct monadic call.
    //
    // Guard: only fire when gprime finalization would actually apply:
    //   - finalize_gprime=true (top-level finalization), OR
    //   - fn is TM_CLOSURE (closures always finalize regardless of gprime flag)
    // With gprime=false (parenthesized context like (2×)), the G_PRIME curry
    // is intentionally preserved for potential later dyadic application.
    if (auto* jux = dynamic_cast<JuxtaposeK*>(new_inner)) {
        // Read child states from cache — no re-walking needed
        TypeMask lm = lookup_state(jux->left).mask;
        TypeMask rm = lookup_state(jux->right).mask;

        auto is_callable = [](TypeMask m) -> bool {
            return m != TM_BOT && m != TM_TOP && (m & TM_CALLABLE) && !(m & ~TM_CALLABLE);
        };

        auto can_finalize = [&](TypeMask fn_mask) -> bool {
            if (k->finalize_gprime) return true;
            // With gprime=false, only closures finalize (they don't use G_PRIME)
            return fn_mask == TM_CLOSURE;
        };

        // Case (a): left is function, right is argument → fn(arg)
        if (is_callable(lm) && can_finalize(lm)) {
            // D3: finalize_if_needed uses cached states to chain MonadicCallK
            // through nested function applications without re-walking
            Continuation* fin_arg = finalize_if_needed(jux->right, rm);
            auto* mcall = heap_->allocate<MonadicCallK>(jux->left, fin_arg);
            if (k->has_location()) mcall->set_location(k->line(), k->column());
            // Infer result type via apply table if fn is a known primitive
            OptState fn_st = lookup_state(jux->left);
            TypeMask arg_mask = lookup_state(fin_arg).mask;
            TypeMask res = TM_TOP;
            if (fn_st.singleton && fn_st.mask == TM_PRIMITIVE)
                res = abstract_apply_monadic(fn_st.singleton->data.primitive_fn->name, arg_mask);
            return {mcall, {res, nullptr}};
        }
        // Case (b): left is basic, right is function → fn(left)
        if (is_callable(rm) && can_finalize(rm)) {
            bool left_is_basic = (lm != TM_BOT && lm != TM_TOP &&
                                  (lm & TM_BASIC) && !(lm & ~TM_BASIC));
            if (left_is_basic) {
                // left is proven basic — no finalization needed
                auto* mcall = heap_->allocate<MonadicCallK>(jux->right, jux->left);
                if (k->has_location()) mcall->set_location(k->line(), k->column());
                OptState fn_st = lookup_state(jux->right);
                TypeMask res = TM_TOP;
                if (fn_st.singleton && fn_st.mask == TM_PRIMITIVE)
                    res = abstract_apply_monadic(fn_st.singleton->data.primitive_fn->name, lm);
                return {mcall, {res, nullptr}};
            }
        }
    }

    // F1 – eliminate FinalizeK when the inner expression is provably non-function
    // (can never produce a CURRIED_FN that needs finalization).
    bool has_fn_bits = (inner_state.mask & (TM_FN | TM_OP | TM_CURRIED)) != 0;
    bool is_known    = (inner_state.mask != TM_BOT) && (inner_state.mask != TM_TOP);

    if (is_known && !has_fn_bits) {
        // Safe to unwrap the FinalizeK entirely
        return {new_inner, inner_state};
    }

    if (new_inner != k->inner) k->inner = new_inner;

    // FinalizeK preserves the value type of its inner expression
    return {k, inner_state};
}

// ---------------------------------------------------------------------------
// JuxtaposeK – G2 grammar function application  (D2)
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_juxtapose(JuxtaposeK* k) {
    auto [new_left,  left_state]  = rewrite(k->left);
    auto [new_right, right_state] = rewrite(k->right);

    if (new_left  != k->left)  k->left  = new_left;
    if (new_right != k->right) k->right = new_right;

    TypeMask lm = left_state.mask;
    TypeMask rm = right_state.mask;

    auto is_basic = [](TypeMask m) -> bool {
        return m != TM_BOT && m != TM_TOP && (m & TM_BASIC) && !(m & ~TM_BASIC);
    };

    // D2 – Dyadic call from known-basic left
    // JuxtaposeK(left:BASIC, JuxtaposeK(fn:CALLABLE, right)) → DyadicCallK(fn, left, right)
    //
    // At runtime: inner JuxtaposeK creates G_PRIME(fn, right), outer sees
    // left is basic → dispatches G_PRIME dyadically as fn(left, right).
    // DyadicCallK skips the curry entirely.
    if (is_basic(lm)) {
        if (auto* inner_jux = dynamic_cast<JuxtaposeK*>(new_right)) {
            // Read inner juxtapose child states from cache — no re-walking
            TypeMask fn_mask  = lookup_state(inner_jux->left).mask;
            TypeMask arg_mask = lookup_state(inner_jux->right).mask;

            bool fn_callable = (fn_mask != TM_BOT && fn_mask != TM_TOP &&
                                (fn_mask & TM_CALLABLE) && !(fn_mask & ~TM_CALLABLE));

            if (fn_callable) {
                // D3: finalize_if_needed chains MonadicCallK for the right arg
                // using cached states, without re-walking
                Continuation* right_arg = finalize_if_needed(inner_jux->right, arg_mask);

                auto* dcall = heap_->allocate<DyadicCallK>(
                    inner_jux->left, new_left, right_arg);
                if (k->has_location()) dcall->set_location(k->line(), k->column());
                // Infer result type via apply table if fn is a known primitive
                OptState fn_st = lookup_state(inner_jux->left);
                TypeMask right_mask = lookup_state(right_arg).mask;
                TypeMask res = TM_TOP;
                if (fn_st.singleton && fn_st.mask == TM_PRIMITIVE)
                    res = abstract_apply_dyadic(fn_st.singleton->data.primitive_fn->name, lm, right_mask);
                return {dcall, {res, nullptr}};
            }
        }
    }

    return {k, {TM_TOP, nullptr}};
}

// ---------------------------------------------------------------------------
// DerivedOperatorK  (O1 – operator resolution)
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_derived_op(DerivedOperatorK* k) {
    auto [new_operand, op_state] = rewrite(k->operand_cont);

    // O1 – if operand is a known primitive and there's no axis, pre-build the
    // DERIVED_OPERATOR value so that the runtime dispatch is avoided.
    if (k->axis_cont == nullptr &&
        (op_state.mask & TM_PRIMITIVE) && op_state.singleton) {

        // Look up the operator by name in the abstract environment.
        auto it = env_.find(k->op_name->str());
        if (it != env_.end() &&
            (it->second.mask & TM_OPERATOR) && it->second.singleton) {

            PrimitiveOp* the_op = it->second.singleton->data.op;
            Value* derived = heap_->allocate_derived_operator(the_op, op_state.singleton);
            return {heap_->allocate<ValueK>(derived), {TM_DERIVED, derived}};
        }
    }

    if (new_operand != k->operand_cont) k->operand_cont = new_operand;

    // With axis_cont, DerivedOperatorK produces CURRIED_FN(OPERATOR_CURRY), not DERIVED_OPERATOR.
    if (k->axis_cont)
        return {k, {TM_CURRIED, nullptr}};
    return {k, {TM_DERIVED, nullptr}};
}

// ---------------------------------------------------------------------------
// ClosureLiteralK – dfn body (recurse with ⍵/⍺ as TM_TOP)
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_closure_literal(ClosureLiteralK* k) {
    // Save current env, extend with ⍵ and ⍺ as unknown
    AbsEnv saved = env_;
    env_["⍵"] = {TM_TOP, nullptr};
    env_["⍺"] = {TM_TOP, nullptr};

    auto [new_body, body_state] = rewrite(k->body);

    env_ = std::move(saved);

    if (new_body != k->body) k->body = new_body;
    return {k, {TM_CLOSURE, nullptr}};
}

// ---------------------------------------------------------------------------
// AssignK – recurse into expression, no rewrite of the assignment itself
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_assign(AssignK* k) {
    auto [new_expr, expr_state] = rewrite(k->expr);
    if (new_expr != k->expr) k->expr = new_expr;

    // After an assignment the result is the value that was assigned.
    // We don't track the binding here (would need to update env_ dynamically).
    return {k, {TM_TOP, nullptr}};
}

// ---------------------------------------------------------------------------
// SeqK – recurse into each statement
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_seq(SeqK* k) {
    OptState last_state = {TM_TOP, nullptr};
    for (size_t i = 0; i < k->statements.size(); ++i) {
        auto [new_stmt, stmt_state] = rewrite(k->statements[i]);
        if (new_stmt != k->statements[i]) k->statements[i] = new_stmt;
        last_state = stmt_state;
    }
    // Result of a sequence is the result of its last statement.
    return {k, last_state};
}

// ---------------------------------------------------------------------------
// MonadicCallK / DyadicCallK – recurse into children, apply table
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_monadic_call(MonadicCallK* k) {
    auto [new_fn,  fn_state]  = rewrite(k->fn_cont);
    auto [new_arg, arg_state] = rewrite(k->arg_cont);
    if (new_fn  != k->fn_cont)  k->fn_cont  = new_fn;
    if (new_arg != k->arg_cont) k->arg_cont = new_arg;

    if (fn_state.singleton && fn_state.mask == TM_PRIMITIVE) {
        TypeMask r = abstract_apply_monadic(
            fn_state.singleton->data.primitive_fn->name, arg_state.mask);
        return {k, {r, nullptr}};
    }
    return {k, {TM_TOP, nullptr}};
}

StaticOptimizer::Rewrite StaticOptimizer::rewrite_dyadic_call(DyadicCallK* k) {
    auto [new_fn,    fn_state]    = rewrite(k->fn_cont);
    auto [new_left,  left_state]  = rewrite(k->left_cont);
    auto [new_right, right_state] = rewrite(k->right_cont);
    if (new_fn    != k->fn_cont)    k->fn_cont    = new_fn;
    if (new_left  != k->left_cont)  k->left_cont  = new_left;
    if (new_right != k->right_cont) k->right_cont = new_right;

    if (fn_state.singleton && fn_state.mask == TM_PRIMITIVE) {
        TypeMask r = abstract_apply_dyadic(
            fn_state.singleton->data.primitive_fn->name,
            left_state.mask, right_state.mask);
        return {k, {r, nullptr}};
    }
    return {k, {TM_TOP, nullptr}};
}

} // namespace apl
