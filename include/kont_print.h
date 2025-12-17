// kont_print.h - Continuation pretty printing via visitor pattern

#pragma once

#include "continuation.h"
#include <sstream>
#include <string>

namespace apl {

// ContinuationPrinter - Pretty prints continuations and continuation graphs
// Supports both flat (stack trace) and recursive (tree) modes
class ContinuationPrinter : public ContinuationVisitor {
public:
    std::ostringstream out;
    int indent_level = 0;
    bool recursive = true;  // If true, recurse into child continuations

    // Print a single continuation (entry point)
    std::string print(Continuation* k) {
        out.str("");
        out.clear();
        if (k) {
            k->accept(*this);
        } else {
            out << "(null)";
        }
        return out.str();
    }

    // Print a flat stack trace (no recursion into children)
    std::string print_stack(const std::vector<Continuation*>& stack) {
        out.str("");
        out.clear();
        bool old_recursive = recursive;
        recursive = false;
        for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
            if (*it) {
                out << "  ";
                (*it)->accept(*this);
                out << "\n";
            }
        }
        recursive = old_recursive;
        return out.str();
    }

private:
    void print_indent() {
        for (int i = 0; i < indent_level; ++i) out << "  ";
    }

    void print_location(Continuation* k) {
        if (k->has_location()) {
            out << " [" << k->line() << ":" << k->column() << "]";
        }
    }

    void print_child(const char* label, Continuation* k) {
        if (!recursive || !k) return;
        out << "\n";
        print_indent();
        out << label << ": ";
        indent_level++;
        k->accept(*this);
        indent_level--;
    }

    void print_children(const char* label, const std::vector<Continuation*>& kids) {
        if (!recursive || kids.empty()) return;
        out << "\n";
        print_indent();
        out << label << ": [";
        indent_level++;
        for (size_t i = 0; i < kids.size(); ++i) {
            out << "\n";
            print_indent();
            out << "[" << i << "] ";
            if (kids[i]) {
                kids[i]->accept(*this);
            } else {
                out << "(null)";
            }
        }
        indent_level--;
        out << "]";
    }

    void print_value(const char* label, Value* v) {
        if (!v) return;
        out << " " << label << "=";
        switch (v->tag) {
            case ValueType::SCALAR: out << v->as_scalar(); break;
            case ValueType::VECTOR:
            case ValueType::MATRIX: {
                auto* m = v->as_matrix();
                if (v->is_char_data()) {
                    // Show character content (truncated if long)
                    out << "'";
                    int len = m->rows();
                    int show = std::min(len, 20);
                    for (int i = 0; i < show; ++i) {
                        int cp = static_cast<int>((*m)(i, 0));
                        if (cp >= 32 && cp < 127) {
                            out << static_cast<char>(cp);
                        } else {
                            out << "?";
                        }
                    }
                    if (len > 20) out << "...";
                    out << "'";
                } else if (m->cols() == 1 && m->rows() <= 6) {
                    // Small vector - show values inline
                    out << "(";
                    for (int i = 0; i < m->rows(); ++i) {
                        if (i > 0) out << " ";
                        out << (*m)(i, 0);
                    }
                    out << ")";
                } else {
                    out << "<" << m->rows() << "x" << m->cols() << " array>";
                }
                break;
            }
            case ValueType::STRING:
                out << "\"" << (v->data.string ? v->data.string : "") << "\"";
                break;
            case ValueType::PRIMITIVE:
                out << "<prim:" << (v->data.primitive_fn ? v->data.primitive_fn->name : "?") << ">";
                break;
            case ValueType::CLOSURE: out << "<closure>"; break;
            case ValueType::OPERATOR: out << "<operator>"; break;
            case ValueType::DEFINED_OPERATOR: out << "<def-op>"; break;
            case ValueType::DERIVED_OPERATOR: out << "<derived>"; break;
            case ValueType::CURRIED_FN: out << "<curried>"; break;
        }
    }

public:
    // ========== Visitor implementations ==========

    void visit(HaltK* k) override {
        out << "HaltK";
        print_location(k);
    }

    void visit(PropagateCompletionK* k) override {
        out << "PropagateCompletionK";
        print_location(k);
    }

    void visit(CatchReturnK* k) override {
        out << "CatchReturnK";
        if (k->function_name) out << "(" << k->function_name << ")";
        print_location(k);
    }

    void visit(CatchBreakK* k) override {
        out << "CatchBreakK";
        print_location(k);
    }

    void visit(CatchContinueK* k) override {
        out << "CatchContinueK";
        print_location(k);
        print_child("loop", k->loop_cont);
    }

    void visit(CatchErrorK* k) override {
        out << "CatchErrorK";
        print_location(k);
    }

    void visit(ThrowErrorK* k) override {
        out << "ThrowErrorK";
        if (k->error_message) out << "(\"" << k->error_message << "\")";
        print_location(k);
    }

    void visit(LiteralK* k) override {
        out << "LiteralK(" << k->literal_value << ")";
        print_location(k);
    }

    void visit(ClosureLiteralK* k) override {
        out << "ClosureLiteralK";
        print_location(k);
        print_child("body", k->body);
    }

    void visit(DefinedOperatorLiteralK* k) override {
        out << "DefinedOperatorLiteralK(" << k->operator_name;
        out << ", " << k->left_operand_name;
        if (k->right_operand_name) out << ", " << k->right_operand_name;
        out << ")";
        print_location(k);
        print_child("body", k->body);
    }

    void visit(LookupK* k) override {
        out << "LookupK(" << k->var_name << ")";
        print_location(k);
    }

    void visit(AssignK* k) override {
        out << "AssignK(" << k->var_name << ")";
        print_location(k);
        print_child("expr", k->expr);
    }

    void visit(PerformAssignK* k) override {
        out << "PerformAssignK(" << k->var_name << ")";
        print_location(k);
    }

    void visit(SysVarReadK* k) override {
        out << "SysVarReadK(" << static_cast<int>(k->var_id) << ")";
        print_location(k);
    }

    void visit(SysVarAssignK* k) override {
        out << "SysVarAssignK(" << static_cast<int>(k->var_id) << ")";
        print_location(k);
        print_child("expr", k->expr);
    }

    void visit(PerformSysVarAssignK* k) override {
        out << "PerformSysVarAssignK(" << static_cast<int>(k->var_id) << ")";
        print_location(k);
    }

    void visit(LiteralStrandK* k) override {
        out << "LiteralStrandK";
        print_value("vec", k->vector_value);
        print_location(k);
    }

    void visit(StrandK* k) override {
        out << "StrandK";
        print_value("left", k->left_val);
        print_value("right", k->right_val);
        print_location(k);
    }

    void visit(JuxtaposeK* k) override {
        out << "JuxtaposeK";
        print_location(k);
        print_child("left", k->left);
        print_child("right", k->right);
    }

    void visit(EvalJuxtaposeLeftK* k) override {
        out << "EvalJuxtaposeLeftK";
        print_value("right", k->right_val);
        print_location(k);
        print_child("left", k->left);
    }

    void visit(PerformJuxtaposeK* k) override {
        out << "PerformJuxtaposeK";
        print_value("right", k->right_val);
        print_location(k);
    }

    void visit(FinalizeK* k) override {
        out << "FinalizeK";
        if (!k->finalize_gprime) out << "(dyadic-only)";
        print_location(k);
        print_child("inner", k->inner);
    }

    void visit(PerformFinalizeK* k) override {
        out << "PerformFinalizeK";
        if (!k->finalize_gprime) out << "(dyadic-only)";
        print_location(k);
    }

    void visit(MonadicK* k) override {
        out << "MonadicK(" << k->op_name << ")";
        print_location(k);
        print_child("operand", k->operand);
    }

    void visit(DyadicK* k) override {
        out << "DyadicK(" << k->op_name << ")";
        print_location(k);
        print_child("left", k->left);
        print_child("right", k->right);
    }

    void visit(EvalDyadicLeftK* k) override {
        out << "EvalDyadicLeftK(" << k->op_name << ")";
        print_value("right", k->right_val);
        print_location(k);
        print_child("left", k->left);
    }

    void visit(ApplyMonadicK* k) override {
        out << "ApplyMonadicK(" << k->op_name << ")";
        print_location(k);
    }

    void visit(ApplyDyadicK* k) override {
        out << "ApplyDyadicK(" << k->op_name << ")";
        print_value("right", k->right_val);
        print_location(k);
    }

    void visit(ArgK* k) override {
        out << "ArgK";
        print_value("arg", k->arg_value);
        print_location(k);
        print_child("next", k->next);
    }

    void visit(EvalStrandElementK* k) override {
        out << "EvalStrandElementK(remaining=" << k->remaining_elements.size()
            << ", done=" << k->evaluated_values.size() << ")";
        print_location(k);
        print_children("remaining", k->remaining_elements);
    }

    void visit(BuildStrandK* k) override {
        out << "BuildStrandK(" << k->values.size() << " values)";
        print_location(k);
    }

    void visit(FrameK* k) override {
        out << "FrameK";
        if (k->function_name) out << "(" << k->function_name << ")";
        print_location(k);
        print_child("return", k->return_k);
    }

    void visit(ApplyFunctionK* k) override {
        out << "ApplyFunctionK";
        print_location(k);
        print_child("fn", k->fn_cont);
        print_child("left", k->left_arg);
        print_child("right", k->right_arg);
    }

    void visit(EvalApplyFunctionLeftK* k) override {
        out << "EvalApplyFunctionLeftK";
        print_value("right", k->right_val);
        print_location(k);
        print_child("fn", k->fn_cont);
        print_child("left", k->left_arg);
    }

    void visit(EvalApplyFunctionMonadicK* k) override {
        out << "EvalApplyFunctionMonadicK";
        print_value("arg", k->arg_val);
        print_location(k);
        print_child("fn", k->fn_cont);
    }

    void visit(EvalApplyFunctionDyadicK* k) override {
        out << "EvalApplyFunctionDyadicK";
        print_value("left", k->left_val);
        print_value("right", k->right_val);
        print_location(k);
        print_child("fn", k->fn_cont);
    }

    void visit(DispatchFunctionK* k) override {
        out << "DispatchFunctionK";
        if (k->fn_val) {
            if (k->fn_val->tag == ValueType::PRIMITIVE && k->fn_val->data.primitive_fn) {
                out << "(" << k->fn_val->data.primitive_fn->name << ")";
            } else if (k->fn_val->tag == ValueType::DERIVED_OPERATOR) {
                out << "(derived)";
            } else if (k->fn_val->tag == ValueType::CLOSURE) {
                out << "(dfn)";
            }
        }
        print_value("left", k->left_val);
        print_value("right", k->right_val);
        print_location(k);
    }

    void visit(DeferredDispatchK* k) override {
        out << "DeferredDispatchK";
        print_value("fn", k->fn_val);
        print_value("left", k->left_val);
        print_location(k);
    }

    void visit(SeqK* k) override {
        out << "SeqK(" << k->statements.size() << " stmts)";
        print_location(k);
        print_children("statements", k->statements);
    }

    void visit(ExecNextStatementK* k) override {
        out << "ExecNextStatementK(" << k->next_index << "/" << k->statements.size() << ")";
        print_location(k);
        print_children("statements", k->statements);
    }

    void visit(IfK* k) override {
        out << "IfK";
        print_location(k);
        print_child("condition", k->condition);
        print_child("then", k->then_branch);
        print_child("else", k->else_branch);
    }

    void visit(SelectBranchK* k) override {
        out << "SelectBranchK";
        print_location(k);
        print_child("then", k->then_branch);
        print_child("else", k->else_branch);
    }

    void visit(WhileK* k) override {
        out << "WhileK";
        print_location(k);
        print_child("condition", k->condition);
        print_child("body", k->body);
    }

    void visit(CheckWhileCondK* k) override {
        out << "CheckWhileCondK";
        print_location(k);
        print_child("condition", k->condition);
        print_child("body", k->body);
    }

    void visit(ForK* k) override {
        out << "ForK(" << k->var_name << ")";
        print_location(k);
        print_child("array", k->array_expr);
        print_child("body", k->body);
    }

    void visit(ForIterateK* k) override {
        out << "ForIterateK(" << k->var_name << ", idx=" << k->index << ")";
        print_value("array", k->array);
        print_location(k);
        print_child("body", k->body);
    }

    void visit(LeaveK* k) override {
        out << "LeaveK";
        print_location(k);
    }

    void visit(ContinueK* k) override {
        out << "ContinueK";
        print_location(k);
    }

    void visit(ReturnK* k) override {
        out << "ReturnK";
        print_location(k);
        print_child("value", k->value_expr);
    }

    void visit(CreateReturnK* k) override {
        out << "CreateReturnK";
        print_location(k);
    }

    void visit(BranchK* k) override {
        out << "BranchK";
        print_location(k);
        print_child("target", k->target_expr);
    }

    void visit(CheckBranchK* k) override {
        out << "CheckBranchK";
        print_value("saved", k->saved_result);
        print_location(k);
    }

    void visit(FunctionCallK* k) override {
        out << "FunctionCallK";
        print_value("fn", k->fn_value);
        print_value("left", k->left_arg);
        print_value("right", k->right_arg);
        print_location(k);
    }

    void visit(RestoreEnvK* k) override {
        out << "RestoreEnvK";
        print_location(k);
    }

    void visit(DerivedOperatorK* k) override {
        out << "DerivedOperatorK(" << k->op_name << ")";
        print_location(k);
        print_child("operand", k->operand_cont);
        print_child("axis", k->axis_cont);
    }

    void visit(ApplyDerivedOperatorK* k) override {
        out << "ApplyDerivedOperatorK(" << k->op_name << ")";
        print_location(k);
        print_child("axis", k->axis_cont);
    }

    void visit(ApplyAxisK* k) override {
        out << "ApplyAxisK";
        print_value("derived", k->derived_op);
        print_location(k);
    }

    void visit(CellIterK* k) override {
        out << "CellIterK(cell=" << k->current_cell << "/" << k->total_cells << ")";
        print_value("fn", k->fn);
        print_location(k);
    }

    void visit(CellCollectK* k) override {
        out << "CellCollectK";
        print_location(k);
    }

    void visit(RowReduceK* k) override {
        out << "RowReduceK(row=" << k->current_row << "/" << k->total_rows << ")";
        print_value("fn", k->fn);
        print_location(k);
    }

    void visit(RowReduceCollectK* k) override {
        out << "RowReduceCollectK";
        print_location(k);
    }

    void visit(NwiseReduceK* k) override {
        out << "NwiseReduceK(win=" << k->current_window << "/" << k->total_windows
            << ", size=" << k->window_size << ")";
        print_value("fn", k->fn);
        print_location(k);
    }

    void visit(NwiseCollectK* k) override {
        out << "NwiseCollectK";
        print_location(k);
    }

    void visit(NwiseMatrixReduceK* k) override {
        out << "NwiseMatrixReduceK(slice=" << k->current_slice << "/" << k->total_slices << ")";
        print_value("fn", k->fn);
        print_location(k);
    }

    void visit(NwiseMatrixCollectK* k) override {
        out << "NwiseMatrixCollectK";
        print_location(k);
    }

    void visit(PrefixScanK* k) override {
        out << "PrefixScanK(prefix=" << k->current_prefix << "/" << k->total_len << ")";
        print_value("fn", k->fn);
        print_location(k);
    }

    void visit(PrefixScanCollectK* k) override {
        out << "PrefixScanCollectK";
        print_location(k);
    }

    void visit(RowScanK* k) override {
        out << "RowScanK(row=" << k->current_row << "/" << k->total_rows << ")";
        print_value("fn", k->fn);
        print_location(k);
    }

    void visit(RowScanCollectK* k) override {
        out << "RowScanCollectK";
        print_location(k);
    }

    void visit(ReduceResultK* k) override {
        out << "ReduceResultK";
        print_value("fn", k->fn);
        print_location(k);
    }

    void visit(InnerProductIterK* k) override {
        out << "InnerProductIterK(" << k->current_i << "," << k->current_j
            << " / " << k->lhs_rows << "x" << k->rhs_cols << ")";
        print_value("f", k->f_fn);
        print_value("g", k->g_fn);
        print_location(k);
    }

    void visit(InnerProductCollectK* k) override {
        out << "InnerProductCollectK";
        print_location(k);
    }

    void visit(IndexedAssignK* k) override {
        out << "IndexedAssignK(" << k->var_name << ")";
        print_location(k);
        print_child("index", k->index_cont);
        print_child("value", k->value_cont);
    }

    void visit(IndexedAssignIndexK* k) override {
        out << "IndexedAssignIndexK(" << k->var_name << ")";
        print_value("value", k->value_val);
        print_location(k);
        print_child("index", k->index_cont);
    }

    void visit(PerformIndexedAssignK* k) override {
        out << "PerformIndexedAssignK(" << k->var_name << ")";
        print_value("value", k->value_val);
        print_value("index", k->index_val);
        print_location(k);
    }

    void visit(InvokeDefinedOperatorK* k) override {
        out << "InvokeDefinedOperatorK";
        print_value("op", k->operator_value);
        print_value("left-op", k->left_operand);
        print_value("right-op", k->right_operand);
        print_value("left-arg", k->left_arg);
        print_value("right-arg", k->right_arg);
        print_location(k);
    }
};

} // namespace apl
