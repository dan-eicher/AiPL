// Operators implementation

#include "operators.h"
#include "machine.h"
#include "heap.h"
#include "primitives.h"
#include "continuation.h"
#include <Eigen/Dense>

namespace apl {

// ========================================================================
// Outer Product Operator: A ∘.f B
// ========================================================================
// Result shape: (⍴A),⍴B
// For each item I in A and J in B: Z[I,J] = I f J
//
// Example: 10 20 30 ∘.+ 1 2 3
//          11 12 13
//          21 22 23
//          31 32 33

void op_outer_product(Machine* m, Value* lhs, Value* f, Value* g, Value* rhs) {
    // Outer product only uses the left function operand f
    // g should be nullptr for outer product (∘.f syntax)
    (void)g;

    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: outer product requires a function operand"));
        return;
    }

    // Get dimensions of both arguments
    int lhs_rows = lhs->rows();
    int lhs_cols = lhs->cols();
    int lhs_size = lhs_rows * lhs_cols;

    int rhs_rows = rhs->rows();
    int rhs_cols = rhs->cols();
    int rhs_size = rhs_rows * rhs_cols;

    // Use CellIterK with OUTER mode for Cartesian product iteration
    m->push_kont(m->heap->allocate<CellIterK>(
        f, lhs, rhs, lhs_size, rhs_size, lhs_cols, rhs_cols));
}

// ========================================================================
// Inner Product Operator: A f.g B
// ========================================================================
// Result shape: (¯1↓⍴A),1↓⍴B
// For vectors: f/A g B (element-wise g, then reduce with f)
// Example: A +.× B is matrix multiplication

void op_inner_product(Machine* m, Value* lhs, Value* f, Value* g, Value* rhs) {
    // Validate that both f and g are functions
    if (!f || !f->is_function() || !g || !g->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: inner product requires two function operands"));
        return;
    }

    // Get dimensions
    int lhs_rows = lhs->rows();
    int lhs_cols = lhs->cols();
    int rhs_rows = rhs->rows();
    int rhs_cols = rhs->cols();

    // Special case: both vectors (1D inner product)
    if (lhs->is_vector() && rhs->is_vector()) {
        // For vectors, check that lengths match
        if (lhs_rows != rhs_rows) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: inner product dimension mismatch"));
            return;
        }
        int n = lhs_rows;  // Common dimension

        // Vector inner product: f/ (lhs g rhs)
        // Push ReduceResultK(f), then dyadic CellIterK(g) for element-wise
        m->push_kont(m->heap->allocate<ReduceResultK>(f));
        m->push_kont(m->heap->allocate<CellIterK>(
            g, lhs, rhs, 0, 0, n,
            CellIterMode::COLLECT, n, 1, true));
        return;
    }

    // General case: matrix inner product
    // LENGTH constraint: last dimension of A must equal first dimension of B
    if (lhs_cols != rhs_rows) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: inner product dimension mismatch"));
        return;
    }

    // Use InnerProductIterK for matrix case
    m->push_kont(m->heap->allocate<InnerProductIterK>(
        f, g, lhs, rhs, lhs_rows, lhs_cols, rhs_cols));
}

// ========================================================================
// Each Operator: f¨B (monadic) or A f¨B (dyadic)
// ========================================================================
// Applies function to each element independently
// Result has same shape as argument(s)
// Uses CellIterK COLLECT mode for continuation-based execution

void op_each(Machine* m, Value* f, Value* omega) {
    if (!f || !f->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: each operator requires a function operand"));
        return;
    }

    if (omega->is_scalar()) {
        m->push_kont(m->heap->allocate<DispatchFunctionK>(f, nullptr, omega, true));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    int rows = omega->rows();
    int cols = omega->cols();
    int num_cells = rows * cols;

    m->push_kont(m->heap->allocate<CellIterK>(
        f, nullptr, omega, 0, 0, num_cells,
        CellIterMode::COLLECT, rows, cols, omega->is_vector()));
}

// ========================================================================
// Duplicate Operator: f⍨B (monadic)
// ========================================================================
// Duplicate: applies f to omega twice (as both arguments)
// f⍨B → B f B

void op_commute(Machine* m, Value* f, Value* omega) {
    // This is the monadic form: duplicate
    // Semantics: f⍨B → B f B
    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: duplicate operator requires a function operand"));
        return;
    }

    // Handle curried functions (from G2 grammar)
    if (f->tag == ValueType::CURRIED_FN) {
        // Curried function: B f B
        // Create DyadicK to apply the function with both arguments as omega
        m->push_kont(m->heap->allocate<DispatchFunctionK>(f, omega, omega));
        return;
    }

    // Handle derived operators (like -⍨⍨ - commute applied to commuted function)
    if (f->tag == ValueType::DERIVED_OPERATOR) {
        // Derived operator: apply with duplicated arguments
        m->push_kont(m->heap->allocate<DispatchFunctionK>(f, omega, omega));
        return;
    }

    // Handle primitive functions
    if (!f->is_primitive()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: duplicate operator requires primitive, curried, or derived function"));
        return;
    }

    PrimitiveFn* fn = f->data.primitive_fn;

    // f must have dyadic form for duplicate
    if (!fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: duplicate operator requires function with dyadic form"));
        return;
    }

    // Apply: omega f omega
    fn->dyadic(m, omega, omega);
}

// ========================================================================
// Commute Operator: A f⍨B (dyadic)
// ========================================================================
// Commute: swaps left and right arguments
// A f⍨B → B f A

void op_commute_dyadic(Machine* m, Value* lhs, Value* f, Value* g, Value* rhs) {
    (void)g;  // Commute only uses one function operand
    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: commute operator requires a function operand"));
        return;
    }

    // Handle curried functions (from G2 grammar)
    if (f->tag == ValueType::CURRIED_FN) {
        // Apply curried function with swapped arguments: rhs f lhs
        m->push_kont(m->heap->allocate<DispatchFunctionK>(f, rhs, lhs));
        return;
    }

    // Handle derived operators (like -⍨⍨ - commute applied to commuted function)
    if (f->tag == ValueType::DERIVED_OPERATOR) {
        // Derived operator: apply with swapped arguments
        m->push_kont(m->heap->allocate<DispatchFunctionK>(f, rhs, lhs));
        return;
    }

    // Handle primitive functions
    if (!f->is_primitive()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: commute operator requires primitive, curried, or derived function"));
        return;
    }

    PrimitiveFn* fn = f->data.primitive_fn;

    // f must have dyadic form
    if (!fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: commute operator requires function with dyadic form"));
        return;
    }

    // Apply: rhs f lhs (swapped)
    fn->dyadic(m, rhs, lhs);
}

// ========================================================================
// Reduction and Scan Operators
// ========================================================================

// Identity Elements for Reduction (ISO-13751 Table 5)
// When reducing an empty vector, return the identity element for the function
double get_identity_element(PrimitiveFn* fn) {
    // Match by function pointer
    if (fn == &prim_plus) return 0.0;      // +/⍬ → 0
    if (fn == &prim_minus) return 0.0;     // -/⍬ → 0
    if (fn == &prim_times) return 1.0;     // ×/⍬ → 1
    if (fn == &prim_divide) return 1.0;    // ÷/⍬ → 1
    if (fn == &prim_star) return 1.0;      // */⍬ → 1

    // For functions without identity, return NaN
    return std::numeric_limits<double>::quiet_NaN();
}

// Helper: get identity element for a function (if it exists)
// Returns NaN if no identity element is defined
static double get_identity_for_function(Value* func) {
    if (func->tag == ValueType::PRIMITIVE) {
        return get_identity_element(func->data.primitive_fn);
    }
    // Non-primitive functions don't have identity elements
    return std::numeric_limits<double>::quiet_NaN();
}

// Reduce (/) - apply dyadic function between elements, right to left
// Uses CellIterK FOLD_RIGHT for continuation-based execution
void fn_reduce(Machine* m, Value* func, Value* omega) {
    // Handle replicate: if "func" is actually an array, this is A / B (replicate)
    // Note: use is_array()/is_scalar() not is_basic_value() to exclude strings
    if (func->is_array() || func->is_scalar()) {
        fn_replicate(m, func, omega);
        return;
    }

    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reduce requires a function"));
        return;
    }

    if (omega->is_scalar()) {
        // Reducing a scalar is identity
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        // Reduce vector to scalar using CellIterK FOLD_RIGHT
        int len = mat->rows();
        if (len == 0) {
            // Empty vector: return identity element (ISO-13751 Table 5)
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: function has no identity element for empty reduction"));
                return;
            }
            m->result = m->heap->allocate_scalar(identity);
            return;
        }
        if (len == 1) {
            m->result = m->heap->allocate_scalar((*mat)(0, 0));
            return;
        }

        // Use CellIterK FOLD_RIGHT for right-to-left reduction
        m->push_kont(m->heap->allocate<CellIterK>(
            func, nullptr, omega, 0, 0, len,
            CellIterMode::FOLD_RIGHT, len, 1, true));
        return;
    }

    // For matrix, reduce along last axis (columns) using RowReduceK
    // Result is a vector with one element per row
    int rows = mat->rows();
    int cols = mat->cols();

    if (cols == 0) {
        // Empty dimension: return vector of identity elements
        double identity = get_identity_for_function(func);
        if (std::isnan(identity)) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: function has no identity element for empty reduction"));
            return;
        }
        Eigen::VectorXd result = Eigen::VectorXd::Constant(rows, identity);
        m->result = m->heap->allocate_vector(result);
        return;
    }

    if (cols == 1) {
        // Single column: just return the column as a vector
        m->result = m->heap->allocate_vector(mat->col(0));
        return;
    }

    // Use RowReduceK to reduce each row independently
    m->push_kont(m->heap->allocate<RowReduceK>(func, omega, rows, cols, false));
}

// Reduce-first (⌿) - reduce along first axis (rows)
// Uses CellIterK FOLD_RIGHT for continuation-based execution
void fn_reduce_first(Machine* m, Value* func, Value* omega) {
    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reduce-first requires a function"));
        return;
    }

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    if (omega->is_vector()) {
        // For vector, same as regular reduce
        fn_reduce(m, func, omega);
        return;
    }

    // For matrix, reduce along first axis (rows)
    // Result is a row vector
    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (rows == 0) {
        // Empty dimension: return row vector of identity elements
        double identity = get_identity_for_function(func);
        if (std::isnan(identity)) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: function has no identity element for empty reduction"));
            return;
        }
        Eigen::VectorXd result = Eigen::VectorXd::Constant(cols, identity);
        m->result = m->heap->allocate_vector(result);
        return;
    }

    if (rows == 1) {
        // Single row: just return the row as a vector
        m->result = m->heap->allocate_vector(mat->row(0).transpose());
        return;
    }

    // Use RowReduceK to reduce each column independently
    // Note: cols is passed as "total_rows" since we iterate over columns
    m->push_kont(m->heap->allocate<RowReduceK>(func, omega, cols, rows, true));
}

// Scan (\) - apply dyadic function cumulatively
// ISO-13751: Item I of Z is f/B[⍳I] (reduction of first I elements)
// Uses PrefixScanK for continuation-based execution
void fn_scan(Machine* m, Value* func, Value* omega) {
    // Handle expand: if "func" is actually an array, this is A \ B (expand)
    // Note: use is_array()/is_scalar() not is_basic_value() to exclude strings
    if (func->is_array() || func->is_scalar()) {
        fn_expand(m, func, omega);
        return;
    }

    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: scan requires a function"));
        return;
    }

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        int len = mat->rows();
        if (len == 0) {
            m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
            return;
        }
        if (len == 1) {
            m->result = m->heap->allocate_scalar((*mat)(0, 0));
            return;
        }

        // Use PrefixScanK for prefix reductions
        m->push_kont(m->heap->allocate<PrefixScanK>(func, omega, len));
        return;
    }

    // For matrix, scan along last axis (columns) using RowScanK
    int rows = mat->rows();
    int cols = mat->cols();

    if (cols == 0) {
        m->result = m->heap->allocate_matrix(Eigen::MatrixXd(rows, 0));
        return;
    }

    if (cols == 1) {
        // Single column: just return the matrix as-is
        m->result = omega;
        return;
    }

    // Use RowScanK to scan each row independently
    m->push_kont(m->heap->allocate<RowScanK>(func, omega, rows, cols, false));
}

// Scan-first (⍀) - scan along first axis (rows)
// Uses PrefixScanK for continuation-based execution
void fn_scan_first(Machine* m, Value* func, Value* omega) {
    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: scan-first requires a function"));
        return;
    }

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    if (omega->is_vector()) {
        // For vector, same as regular scan
        fn_scan(m, func, omega);
        return;
    }

    // For matrix, scan along first axis (rows) using RowScanK
    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (rows == 0) {
        m->result = m->heap->allocate_matrix(Eigen::MatrixXd(0, cols));
        return;
    }

    if (rows == 1) {
        // Single row: just return the matrix as-is
        m->result = omega;
        return;
    }

    // Use RowScanK to scan each column independently
    // Note: cols is passed as "total_rows" since we iterate over columns
    m->push_kont(m->heap->allocate<RowScanK>(func, omega, cols, rows, true));
}

// ========================================================================
// Axis-specified Reduction and Scan (f/[k] and f\[k])
// ========================================================================
// These are the dyadic forms where the "second operand" is the axis specification.
// Called as: op->dyadic(m, lhs, func, axis, rhs)
// where lhs is nullptr for monadic use.

// Helper: validate axis specification and return 1-based axis number
// Returns -1 on error (after pushing ThrowErrorK)
static int validate_axis(Machine* m, Value* axis, int max_rank) {
    if (!axis || !axis->is_scalar()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: axis must be a scalar"));
        return -1;
    }

    double axis_val = axis->as_scalar();
    int k = static_cast<int>(axis_val);

    if (axis_val != static_cast<double>(k) || k < 1 || k > max_rank) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: axis out of range"));
        return -1;
    }

    return k;
}

// Reduce with axis specification: f/[k]B
// Dyadic operator form where second operand is axis
void fn_reduce_axis(Machine* m, Value* lhs, Value* func, Value* axis, Value* rhs) {
    (void)lhs;  // Always nullptr for monadic use

    // Handle replicate: if "func" is actually an array, this is invalid
    if (func->is_array() || func->is_scalar()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: replicate does not support axis specification"));
        return;
    }

    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reduce requires a function"));
        return;
    }

    if (rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    // String → char vector conversion
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    int max_rank = rhs->is_vector() ? 1 : 2;
    int k = validate_axis(m, axis, max_rank);
    if (k < 0) return;

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        // Vector only has axis 1 - same as regular reduce
        int len = mat->rows();
        if (len == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: function has no identity element for empty reduction"));
                return;
            }
            m->result = m->heap->allocate_scalar(identity);
            return;
        }
        if (len == 1) {
            m->result = m->heap->allocate_scalar((*mat)(0, 0));
            return;
        }
        m->push_kont(m->heap->allocate<CellIterK>(
            func, nullptr, rhs, 0, 0, len,
            CellIterMode::FOLD_RIGHT, len, 1, true));
        return;
    }

    // Matrix: k=1 is first axis (rows), k=2 is last axis (columns)
    int rows = mat->rows();
    int cols = mat->cols();

    if (k == 1) {
        // Reduce along first axis (like ⌿)
        if (rows == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: function has no identity element for empty reduction"));
                return;
            }
            Eigen::VectorXd result = Eigen::VectorXd::Constant(cols, identity);
            m->result = m->heap->allocate_vector(result);
            return;
        }
        if (rows == 1) {
            m->result = m->heap->allocate_vector(mat->row(0).transpose());
            return;
        }
        m->push_kont(m->heap->allocate<RowReduceK>(func, rhs, cols, rows, true));
    } else {
        // k == 2: Reduce along last axis (like /)
        if (cols == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: function has no identity element for empty reduction"));
                return;
            }
            Eigen::VectorXd result = Eigen::VectorXd::Constant(rows, identity);
            m->result = m->heap->allocate_vector(result);
            return;
        }
        if (cols == 1) {
            m->result = m->heap->allocate_vector(mat->col(0));
            return;
        }
        m->push_kont(m->heap->allocate<RowReduceK>(func, rhs, rows, cols, false));
    }
}

// Scan with axis specification: f\[k]B
// Dyadic operator form where second operand is axis
void fn_scan_axis(Machine* m, Value* lhs, Value* func, Value* axis, Value* rhs) {
    (void)lhs;  // Always nullptr for monadic use

    // Handle expand: if "func" is actually an array, this is invalid
    if (func->is_array() || func->is_scalar()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: expand does not support axis specification"));
        return;
    }

    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: scan requires a function"));
        return;
    }

    if (rhs->is_scalar()) {
        m->result = m->heap->allocate_scalar(rhs->as_scalar());
        return;
    }

    // String → char vector conversion
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    int max_rank = rhs->is_vector() ? 1 : 2;
    int k = validate_axis(m, axis, max_rank);
    if (k < 0) return;

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        // Vector only has axis 1 - same as regular scan
        int len = mat->rows();
        if (len == 0) {
            m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
            return;
        }
        if (len == 1) {
            m->result = m->heap->allocate_scalar((*mat)(0, 0));
            return;
        }
        m->push_kont(m->heap->allocate<PrefixScanK>(func, rhs, len));
        return;
    }

    // Matrix: k=1 is first axis (rows), k=2 is last axis (columns)
    int rows = mat->rows();
    int cols = mat->cols();

    if (k == 1) {
        // Scan along first axis (like ⍀)
        if (rows == 0) {
            m->result = m->heap->allocate_matrix(Eigen::MatrixXd(0, cols));
            return;
        }
        if (rows == 1) {
            m->result = rhs;
            return;
        }
        m->push_kont(m->heap->allocate<RowScanK>(func, rhs, cols, rows, true));
    } else {
        // k == 2: Scan along last axis (like \)
        if (cols == 0) {
            m->result = m->heap->allocate_matrix(Eigen::MatrixXd(rows, 0));
            return;
        }
        if (cols == 1) {
            m->result = rhs;
            return;
        }
        m->push_kont(m->heap->allocate<RowScanK>(func, rhs, rows, cols, false));
    }
}

// ========================================================================
// PrimitiveOp struct definitions
// ========================================================================

PrimitiveOp op_dot = {
    ".",
    nullptr,              // No monadic form
    op_inner_product      // Dyadic: A f.g B
};

PrimitiveOp op_outer_dot = {
    "∘.",
    nullptr,              // No monadic form
    op_outer_product      // Dyadic: A ∘.f B
};

PrimitiveOp op_diaeresis = {
    "¨",
    op_each,              // Monadic: f¨B
    nullptr               // Dyadic form uses different evaluation (not yet implemented)
};

PrimitiveOp op_tilde = {
    "⍨",
    op_commute,           // Monadic: f⍨B (duplicate)
    op_commute_dyadic     // Dyadic: A f⍨B (commute)
};

// Reduction operators (monadic - take function operand, return derived function)
// Note: These are operators, not primitives, so they need special handling
// Dyadic forms handle axis specification: f/[k]B
PrimitiveOp op_reduce = {
    "/",
    fn_reduce,            // Monadic: f/B (reduce along last axis)
    fn_reduce_axis        // Dyadic: f/[k]B (reduce along axis k)
};

PrimitiveOp op_reduce_first = {
    "⌿",
    fn_reduce_first,      // Monadic: f⌿B (reduce along first axis)
    fn_reduce_axis        // Dyadic: f⌿[k]B (reduce along axis k)
};

PrimitiveOp op_scan = {
    "\\",
    fn_scan,              // Monadic: f\B (scan along last axis)
    fn_scan_axis          // Dyadic: f\[k]B (scan along axis k)
};

PrimitiveOp op_scan_first = {
    "⍀",
    fn_scan_first,        // Monadic: f⍀B (scan along first axis)
    fn_scan_axis          // Dyadic: f⍀[k]B (scan along axis k)
};

// ========================================================================
// Rank Operator: f⍤k (ISO 13751 §9)
// ========================================================================
// Applies function f to k-cells of the argument(s)
// k-cells for 2D arrays:
//   0-cells = scalars
//   1-cells = rows (vectors)
//   2-cells = whole matrix
//
// Rank spec can be:
//   scalar k: applies to both monadic and dyadic
//   2-vector [l r]: left rank l, right rank r (dyadic only)
//   3-vector [m l r]: monadic rank m, left rank l, right rank r

// Helper: get the rank of a value (0=scalar, 1=vector, 2=matrix)
static int get_array_rank(Value* v) {
    if (v->is_scalar()) return 0;
    if (v->is_vector()) return 1;
    return 2;  // Matrix
}

// Helper: extract rank values from rank specification
static bool parse_rank_spec(Value* rank_spec, int array_rank,
                            int* monadic_rank, int* left_rank, int* right_rank) {
    if (rank_spec->is_scalar()) {
        int k = static_cast<int>(rank_spec->as_scalar());
        // Negative rank means "array rank minus k"
        if (k < 0) k = std::max(0, array_rank + k);
        k = std::min(k, array_rank);  // Clamp to actual rank
        *monadic_rank = *left_rank = *right_rank = k;
        return true;
    }

    if (rank_spec->is_vector()) {
        const Eigen::MatrixXd* mat = rank_spec->as_matrix();
        int len = mat->rows();

        if (len == 2) {
            // [left_rank right_rank]
            *left_rank = static_cast<int>((*mat)(0, 0));
            *right_rank = static_cast<int>((*mat)(1, 0));
            *monadic_rank = *right_rank;  // Use right rank for monadic
        } else if (len == 3) {
            // [monadic_rank left_rank right_rank]
            *monadic_rank = static_cast<int>((*mat)(0, 0));
            *left_rank = static_cast<int>((*mat)(1, 0));
            *right_rank = static_cast<int>((*mat)(2, 0));
        } else {
            return false;  // Invalid rank spec
        }
        return true;
    }

    return false;  // Invalid rank spec type
}

// Helper: extract a k-cell from an array
// For 2D: 0-cell = scalar at (row, col), 1-cell = row vector, 2-cell = whole matrix
static Value* extract_cell(Machine* m, Value* arr, int k, int cell_index) {
    if (k >= get_array_rank(arr)) {
        // Full rank: return whole array
        return arr;
    }

    if (arr->is_scalar()) {
        return arr;
    }

    const Eigen::MatrixXd* mat = arr->as_matrix();

    if (arr->is_vector()) {
        if (k == 0) {
            // 0-cell of vector: individual scalar
            return m->heap->allocate_scalar((*mat)(cell_index, 0));
        }
        // k >= 1: whole vector
        return arr;
    }

    // Matrix
    if (k == 0) {
        // 0-cell: scalar at linear index
        int rows = mat->rows();
        int cols = mat->cols();
        int r = cell_index / cols;
        int c = cell_index % cols;
        if (r >= rows) return nullptr;  // Out of bounds
        return m->heap->allocate_scalar((*mat)(r, c));
    } else if (k == 1) {
        // 1-cell: row vector
        if (cell_index >= mat->rows()) return nullptr;
        Eigen::VectorXd row = mat->row(cell_index).transpose();
        return m->heap->allocate_vector(row);
    }

    // k >= 2: whole matrix
    return arr;
}

// Helper: count number of k-cells in an array
static int count_cells(Value* arr, int k) {
    if (k >= get_array_rank(arr)) return 1;

    if (arr->is_scalar()) return 1;

    const Eigen::MatrixXd* mat = arr->as_matrix();

    if (arr->is_vector()) {
        return (k == 0) ? mat->rows() : 1;
    }

    // Matrix
    if (k == 0) {
        return mat->rows() * mat->cols();
    } else if (k == 1) {
        return mat->rows();
    }
    return 1;
}

// Monadic rank operator - shouldn't be called directly
void op_rank_monadic(Machine* m, Value* f, Value* omega) {
    (void)f; (void)omega;
    m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: rank operator requires rank specification"));
}

// Dyadic rank: A f⍤k B (or monadic f⍤k B where lhs is nullptr)
void op_rank(Machine* m, Value* lhs, Value* f, Value* rank_spec, Value* rhs) {
    // Validate function operand
    if (!f || !f->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: rank operator requires function operand"));
        return;
    }

    // Validate rank specification
    if (!rank_spec) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: rank operator requires rank specification"));
        return;
    }

    // Parse rank specification
    int rhs_rank = get_array_rank(rhs);
    int lhs_rank = lhs ? get_array_rank(lhs) : 0;
    int monadic_r, left_r, right_r;

    if (!parse_rank_spec(rank_spec, std::max(lhs_rank, rhs_rank), &monadic_r, &left_r, &right_r)) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: invalid rank specification"));
        return;
    }

    bool is_dyadic = (lhs != nullptr);

    if (!is_dyadic) {
        // Monadic: f⍤k B
        int k = std::min(monadic_r, rhs_rank);
        if (k < 0) k = std::max(0, rhs_rank + k);

        int num_cells = count_cells(rhs, k);

        if (num_cells == 1) {
            // Single cell: just apply f to whole array
            // Use force_monadic=true to ensure immediate application (not G_PRIME curry)
            m->push_kont(m->heap->allocate<DispatchFunctionK>(f, nullptr, rhs, true));
            return;
        }

        // Multiple cells: use CellIterK continuation to iterate
        int rows = rhs->is_scalar() ? 1 : rhs->rows();
        int cols = rhs->is_scalar() ? 1 : rhs->cols();
        m->push_kont(m->heap->allocate<CellIterK>(
            f, nullptr, rhs, k, k, num_cells,
            CellIterMode::COLLECT, rows, cols, rhs->is_vector()));
    } else {
        // Dyadic: A f⍤k B
        int lk = std::min(left_r, lhs_rank);
        int rk = std::min(right_r, rhs_rank);
        if (lk < 0) lk = std::max(0, lhs_rank + lk);
        if (rk < 0) rk = std::max(0, rhs_rank + rk);

        int left_cells = count_cells(lhs, lk);
        int right_cells = count_cells(rhs, rk);

        // Cell counts must match (or one must be 1 for scalar extension)
        if (left_cells != right_cells && left_cells != 1 && right_cells != 1) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: rank operator cell count mismatch"));
            return;
        }

        int num_cells = std::max(left_cells, right_cells);

        if (num_cells == 1) {
            // Single cell each: just apply f dyadically
            m->push_kont(m->heap->allocate<DispatchFunctionK>(f, lhs, rhs));
            return;
        }

        // Multiple cells: use CellIterK continuation
        int rows = rhs->is_scalar() ? 1 : rhs->rows();
        int cols = rhs->is_scalar() ? 1 : rhs->cols();
        m->push_kont(m->heap->allocate<CellIterK>(
            f, lhs, rhs, lk, rk, num_cells,
            CellIterMode::COLLECT, rows, cols, rhs->is_vector()));
    }
}

PrimitiveOp op_rank_op = {
    "⍤",
    nullptr,              // No monadic form - rank requires rank specification
    op_rank               // Dyadic: A f⍤k B
};

} // namespace apl
