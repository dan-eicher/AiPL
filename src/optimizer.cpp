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
//   O2  – DerivedOperatorK(op, primitive_fn, nil) →  ValueK(DERIVED_OPERATOR)
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
    if (!root) return root;
    return rewrite(root).kont;
}

// ---------------------------------------------------------------------------
// Main dispatch
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite(Continuation* k) {
    if (!k) return {nullptr, {TM_BOT, nullptr}};

    // Already optimised (e.g. a ValueK created during this same pass)
    if (auto* vk = dynamic_cast<ValueK*>(k)) {
        return {vk, opt_state_from_value(vk->value)};
    }
    if (auto* lit = dynamic_cast<LiteralK*>(k)) {
        return rewrite_literal(lit);
    }
    if (auto* ls = dynamic_cast<LiteralStrandK*>(k)) {
        return rewrite_literal_strand(ls);
    }
    if (auto* lk = dynamic_cast<LookupK*>(k)) {
        return rewrite_lookup(lk);
    }
    if (auto* jk = dynamic_cast<JuxtaposeK*>(k)) {
        return rewrite_juxtapose(jk);
    }
    if (auto* mk = dynamic_cast<MonadicK*>(k)) {
        return rewrite_monadic(mk);
    }
    if (auto* dk = dynamic_cast<DyadicK*>(k)) {
        return rewrite_dyadic(dk);
    }
    if (auto* fk = dynamic_cast<FinalizeK*>(k)) {
        return rewrite_finalize(fk);
    }
    if (auto* ck = dynamic_cast<ClosureLiteralK*>(k)) {
        return rewrite_closure_literal(ck);
    }
    if (auto* dok = dynamic_cast<DerivedOperatorK*>(k)) {
        return rewrite_derived_op(dok);
    }
    if (auto* ak = dynamic_cast<AssignK*>(k)) {
        return rewrite_assign(ak);
    }
    if (auto* sk = dynamic_cast<SeqK*>(k)) {
        return rewrite_seq(sk);
    }

    // Optimizer-produced nodes — pass through with TM_TOP
    if (dynamic_cast<MonadicCallK*>(k) || dynamic_cast<DyadicCallK*>(k)) {
        return {k, {TM_TOP, nullptr}};
    }

    // Unknown node type – return unchanged with TM_TOP
    return {k, {TM_TOP, nullptr}};
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

    // Infer result state conservatively
    OptState state = {TM_TOP, nullptr};
    return {k, state};
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

    return {k, {TM_TOP, nullptr}};
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
        // Recurse into the juxtapose children to get their states.
        // (new_inner was already rewritten, but it may still be a JuxtaposeK
        // whose children have been rewritten — their states weren't returned.)
        auto left_r  = rewrite(jux->left);
        auto right_r = rewrite(jux->right);
        jux->left  = left_r.kont;
        jux->right = right_r.kont;

        TypeMask lm = left_r.state.mask;
        TypeMask rm = right_r.state.mask;

        auto is_callable = [](TypeMask m) -> bool {
            return m != TM_BOT && m != TM_TOP && (m & TM_CALLABLE) && !(m & ~TM_CALLABLE);
        };

        auto can_finalize = [&](TypeMask fn_mask) -> bool {
            if (k->finalize_gprime) return true;
            // With gprime=false, only closures finalize (they don't use G_PRIME)
            return fn_mask == TM_CLOSURE;
        };

        // Helper: wrap a continuation in FinalizeK if it might produce a curry.
        // JuxtaposeK produces G_PRIME curries; apply_function_immediate cannot
        // handle those, so we must finalize first.
        // D3: recursively apply rewrite() on the new FinalizeK so that D1 chains
        // through nested function calls (e.g. -⌊x → MonadicCallK(-, MonadicCallK(⌊, x))).
        auto ensure_finalized = [&](Continuation* c, TypeMask m) -> Continuation* {
            bool known_basic = (m != TM_BOT && m != TM_TOP &&
                                (m & TM_BASIC) && !(m & ~TM_BASIC));
            if (known_basic) return c;  // basic values are never curries
            auto* fin = heap_->allocate<FinalizeK>(c, true);
            if (c->has_location()) fin->set_location(c->line(), c->column());
            return rewrite(fin).kont;  // D3: recursive D1
        };

        // Case (a): left is function, right is argument → fn(arg)
        if (is_callable(lm) && can_finalize(lm)) {
            auto* mcall = heap_->allocate<MonadicCallK>(
                jux->left, ensure_finalized(jux->right, rm));
            if (k->has_location()) mcall->set_location(k->line(), k->column());
            return {mcall, {TM_TOP, nullptr}};
        }
        // Case (b): left is basic, right is function → fn(left)
        if (is_callable(rm) && can_finalize(rm)) {
            bool left_is_basic = (lm != TM_BOT && lm != TM_TOP &&
                                  (lm & TM_BASIC) && !(lm & ~TM_BASIC));
            if (left_is_basic) {
                // left is proven basic — no finalization needed
                auto* mcall = heap_->allocate<MonadicCallK>(jux->right, jux->left);
                if (k->has_location()) mcall->set_location(k->line(), k->column());
                return {mcall, {TM_TOP, nullptr}};
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
            // Get the inner juxtapose's children states
            auto fn_r  = rewrite(inner_jux->left);
            auto arg_r = rewrite(inner_jux->right);
            inner_jux->left  = fn_r.kont;
            inner_jux->right = arg_r.kont;

            TypeMask fn_mask = fn_r.state.mask;
            bool fn_callable = (fn_mask != TM_BOT && fn_mask != TM_TOP &&
                                (fn_mask & TM_CALLABLE) && !(fn_mask & ~TM_CALLABLE));

            if (fn_callable) {
                // Right arg may be a JuxtaposeK that produces a G_PRIME curry;
                // wrap in FinalizeK to resolve it before apply_function_immediate.
                // Left arg is proven BASIC — no finalization needed.
                TypeMask arg_mask = arg_r.state.mask;
                Continuation* right_arg = inner_jux->right;
                bool arg_known_basic = (arg_mask != TM_BOT && arg_mask != TM_TOP &&
                                        (arg_mask & TM_BASIC) && !(arg_mask & ~TM_BASIC));
                if (!arg_known_basic) {
                    auto* fin = heap_->allocate<FinalizeK>(right_arg, true);
                    if (right_arg->has_location()) fin->set_location(right_arg->line(), right_arg->column());
                    right_arg = rewrite(fin).kont;  // D3: recursive D1
                }

                auto* dcall = heap_->allocate<DyadicCallK>(
                    inner_jux->left, new_left, right_arg);
                if (k->has_location()) dcall->set_location(k->line(), k->column());
                return {dcall, {TM_TOP, nullptr}};
            }
        }
    }

    return {k, {TM_TOP, nullptr}};
}

// ---------------------------------------------------------------------------
// DerivedOperatorK  (O2 – operator resolution)
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_derived_op(DerivedOperatorK* k) {
    auto [new_operand, op_state] = rewrite(k->operand_cont);

    // O2 – if operand is a known primitive and there's no axis, pre-build the
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

} // namespace apl
