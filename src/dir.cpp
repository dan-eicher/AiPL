// DIR - Definition-site Instantiation with Re-optimization
//
// CloningBackend: deep-clone continuation tree + re-run StaticOptimizer
// with concrete argument types from the call site.

#include "dir.h"
#include "continuation.h"
#include "heap.h"
#include "optimizer.h"
#include "machine.h"
#include <unordered_map>

namespace apl {

// ---------------------------------------------------------------------------
// Deep-clone a continuation tree
// ---------------------------------------------------------------------------
// Uses a memo map to preserve DAG sharing (same input pointer → same output).
// Parser-produced and optimizer-produced continuation types are cloned;
// runtime-generated types (ephemeral) are returned unchanged.

namespace {

Continuation* clone_impl(Continuation* k, Heap* heap,
                         std::unordered_map<Continuation*, Continuation*>& memo) {
    if (!k) return nullptr;

    auto it = memo.find(k);
    if (it != memo.end()) return it->second;

    Continuation* result = nullptr;

    // --- Leaves (no Continuation* children) ---

    if (auto* lit = dynamic_cast<LiteralK*>(k)) {
        auto* c = heap->allocate<LiteralK>(lit->literal_value);
        c->set_location(lit->line(), lit->column());
        result = c;
    }
    else if (auto* lk = dynamic_cast<LookupK*>(k)) {
        auto* c = heap->allocate<LookupK>(lk->var_name);
        c->set_location(lk->line(), lk->column());
        result = c;
    }
    else if (auto* vk = dynamic_cast<ValueK*>(k)) {
        auto* c = heap->allocate<ValueK>(vk->value);
        c->set_location(vk->line(), vk->column());
        result = c;
    }
    else if (auto* ls = dynamic_cast<LiteralStrandK*>(k)) {
        auto* c = heap->allocate<LiteralStrandK>(ls->vector_value);
        c->set_location(ls->line(), ls->column());
        result = c;
    }

    // --- Binary (two Continuation* children) ---

    else if (auto* jk = dynamic_cast<JuxtaposeK*>(k)) {
        auto* c = heap->allocate<JuxtaposeK>(
            clone_impl(jk->left, heap, memo),
            clone_impl(jk->right, heap, memo));
        c->set_location(jk->line(), jk->column());
        result = c;
    }
    else if (auto* dk = dynamic_cast<DyadicK*>(k)) {
        auto* c = heap->allocate<DyadicK>(
            dk->op_name,
            clone_impl(dk->left, heap, memo),
            clone_impl(dk->right, heap, memo));
        c->set_location(dk->line(), dk->column());
        result = c;
    }
    else if (auto* dck = dynamic_cast<DyadicCallK*>(k)) {
        auto* c = heap->allocate<DyadicCallK>(
            clone_impl(dck->fn_cont, heap, memo),
            clone_impl(dck->left_cont, heap, memo),
            clone_impl(dck->right_cont, heap, memo));
        c->set_location(dck->line(), dck->column());
        result = c;
    }

    // --- Unary (one Continuation* child) ---

    else if (auto* fk = dynamic_cast<FinalizeK*>(k)) {
        auto* c = heap->allocate<FinalizeK>(
            clone_impl(fk->inner, heap, memo),
            fk->finalize_gprime);
        c->set_location(fk->line(), fk->column());
        result = c;
    }
    else if (auto* mk = dynamic_cast<MonadicK*>(k)) {
        auto* c = heap->allocate<MonadicK>(
            mk->op_name,
            clone_impl(mk->operand, heap, memo));
        c->set_location(mk->line(), mk->column());
        result = c;
    }
    else if (auto* mck = dynamic_cast<MonadicCallK*>(k)) {
        auto* c = heap->allocate<MonadicCallK>(
            clone_impl(mck->fn_cont, heap, memo),
            clone_impl(mck->arg_cont, heap, memo));
        c->set_location(mck->line(), mck->column());
        result = c;
    }
    else if (auto* dok = dynamic_cast<DerivedOperatorK*>(k)) {
        auto* c = heap->allocate<DerivedOperatorK>(
            clone_impl(dok->operand_cont, heap, memo),
            dok->op_name,
            clone_impl(dok->axis_cont, heap, memo));
        c->set_location(dok->line(), dok->column());
        result = c;
    }

    // --- Compound ---

    else if (auto* sk = dynamic_cast<SeqK*>(k)) {
        std::vector<Continuation*> cloned_stmts;
        cloned_stmts.reserve(sk->statements.size());
        for (auto* stmt : sk->statements) {
            cloned_stmts.push_back(clone_impl(stmt, heap, memo));
        }
        auto* c = heap->allocate<SeqK>(cloned_stmts);
        c->set_location(sk->line(), sk->column());
        result = c;
    }
    else if (auto* ak = dynamic_cast<AssignK*>(k)) {
        auto* c = heap->allocate<AssignK>(
            ak->var_name,
            clone_impl(ak->expr, heap, memo));
        c->set_location(ak->line(), ak->column());
        result = c;
    }
    else if (auto* cl = dynamic_cast<ClosureLiteralK*>(k)) {
        auto* c = heap->allocate<ClosureLiteralK>(
            clone_impl(cl->body, heap, memo),
            cl->is_niladic);
        c->set_location(cl->line(), cl->column());
        result = c;
    }

    // --- Control flow ---

    else if (auto* ik = dynamic_cast<IfK*>(k)) {
        auto* c = heap->allocate<IfK>(
            clone_impl(ik->condition, heap, memo),
            clone_impl(ik->then_branch, heap, memo),
            clone_impl(ik->else_branch, heap, memo));
        c->set_location(ik->line(), ik->column());
        result = c;
    }
    else if (auto* wk = dynamic_cast<WhileK*>(k)) {
        auto* c = heap->allocate<WhileK>(
            clone_impl(wk->condition, heap, memo),
            clone_impl(wk->body, heap, memo));
        c->set_location(wk->line(), wk->column());
        result = c;
    }
    else if (auto* fk = dynamic_cast<ForK*>(k)) {
        auto* c = heap->allocate<ForK>(
            fk->var_name,
            clone_impl(fk->array_expr, heap, memo),
            clone_impl(fk->body, heap, memo));
        c->set_location(fk->line(), fk->column());
        result = c;
    }

    // --- Eigen fast-path nodes (optimizer-produced) ---

    else if (auto* erk = dynamic_cast<EigenReduceK*>(k)) {
        auto* c = heap->allocate<EigenReduceK>(
            erk->reduce_op,
            clone_impl(erk->arg_cont, heap, memo),
            erk->derived_op);
        c->set_location(erk->line(), erk->column());
        result = c;
    }
    else if (auto* epk = dynamic_cast<EigenProductK*>(k)) {
        auto* c = heap->allocate<EigenProductK>(
            clone_impl(epk->left_cont, heap, memo),
            clone_impl(epk->right_cont, heap, memo),
            epk->derived_op);
        c->set_location(epk->line(), epk->column());
        result = c;
    }
    else if (auto* eok = dynamic_cast<EigenOuterK*>(k)) {
        auto* c = heap->allocate<EigenOuterK>(
            eok->outer_op,
            clone_impl(eok->left_cont, heap, memo),
            clone_impl(eok->right_cont, heap, memo),
            eok->derived_op);
        c->set_location(eok->line(), eok->column());
        result = c;
    }

    else if (auto* esk = dynamic_cast<EigenScanK*>(k)) {
        auto* c = heap->allocate<EigenScanK>(
            esk->scan_op,
            clone_impl(esk->arg_cont, heap, memo),
            esk->derived_op);
        c->set_location(esk->line(), esk->column());
        result = c;
    }
    else if (auto* erfk = dynamic_cast<EigenReduceFirstK*>(k)) {
        auto* c = heap->allocate<EigenReduceFirstK>(
            erfk->reduce_op,
            clone_impl(erfk->arg_cont, heap, memo),
            erfk->derived_op);
        c->set_location(erfk->line(), erfk->column());
        result = c;
    }
    else if (auto* esk = dynamic_cast<EigenSortK*>(k)) {
        auto* c = heap->allocate<EigenSortK>(
            esk->direction,
            clone_impl(esk->arg_cont, heap, memo));
        c->set_location(esk->line(), esk->column());
        result = c;
    }

    // --- Default: return original pointer unchanged (runtime-generated konts) ---

    else {
        result = k;
    }

    memo[k] = result;
    return result;
}

} // anonymous namespace

Continuation* clone_tree(Continuation* root, Heap* heap) {
    if (!root) return nullptr;
    std::unordered_map<Continuation*, Continuation*> memo;
    return clone_impl(root, heap, memo);
}

// ---------------------------------------------------------------------------
// CloningBackend::specialize
// ---------------------------------------------------------------------------

Continuation* CloningBackend::specialize(
    Continuation* body, Heap* heap,
    ValueType omega_type, bool has_alpha, ValueType alpha_type)
{
    Continuation* clone = clone_tree(body, heap);

    AbsEnv env;
    env["\xe2\x8d\xb5"] = {tm_from_value_type(omega_type), nullptr};  // "⍵"
    if (has_alpha)
        env["\xe2\x8d\xba"] = {tm_from_value_type(alpha_type), nullptr};  // "⍺"
    else
        env["\xe2\x8d\xba"] = {TM_TOP, nullptr};  // "⍺"

    StaticOptimizer opt;
    return opt.run(clone, heap, env);
}

// ---------------------------------------------------------------------------
// TypeDirectedK implementation
// ---------------------------------------------------------------------------

Continuation* TypeDirectedK::dispatch(ValueType omega, bool has_alpha, ValueType alpha) {
    TypeSig sig{omega, alpha, has_alpha};

    // Cache lookup
    auto it = cache.find(sig);
    if (it != cache.end()) {
        // nullptr means "already tried, no benefit" — return original
        return it->second ? it->second : original_body;
    }

    // Cache miss — specialize
    Continuation* specialized = backend->specialize(
        original_body, heap_ref, sig.omega_type, sig.has_alpha, sig.alpha_type);

    // If specialize returned the exact same pointer, store nullptr to avoid retrying
    if (specialized == original_body) {
        cache[sig] = nullptr;
        return original_body;
    }

    cache[sig] = specialized;
    return specialized;
}

void TypeDirectedK::invoke(Machine* /*machine*/) {
    // Future JIT integration: read ⍵/⍺ from environment, call dispatch().
    // For CloningBackend, FunctionCallK::invoke calls dispatch() directly.
}

void TypeDirectedK::mark(Heap* heap) {
    heap->mark(original_body);
    for (auto& [sig, kont] : cache) {
        if (kont) heap->mark(kont);
    }
}

// ---------------------------------------------------------------------------
// ReturnTypeRecordK implementation
// ---------------------------------------------------------------------------

void ReturnTypeRecordK::invoke(Machine* machine) {
    if (tdk && machine->result) {
        tdk->returns[sig] = machine->result->tag;
    }
    // Result passes through unchanged.
}

void ReturnTypeRecordK::mark(Heap* heap) {
    heap->mark(tdk);
}

} // namespace apl
