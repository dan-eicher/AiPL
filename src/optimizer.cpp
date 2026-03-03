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
// FinalizeK  (F1)
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_finalize(FinalizeK* k) {
    auto [new_inner, inner_state] = rewrite(k->inner);

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
// JuxtaposeK – G2 grammar function application
// ---------------------------------------------------------------------------

StaticOptimizer::Rewrite StaticOptimizer::rewrite_juxtapose(JuxtaposeK* k) {
    auto [new_left,  left_state]  = rewrite(k->left);
    auto [new_right, right_state] = rewrite(k->right);

    if (new_left  != k->left)  k->left  = new_left;
    if (new_right != k->right) k->right = new_right;

    // We don't attempt to inline the call here (D-patterns); just propagate.
    // Result type: if left is a known operator/function, result is TM_TOP for now.
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
