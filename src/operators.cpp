// Operators implementation

#include "operators.h"
#include "machine.h"
#include "heap.h"
#include "primitives.h"
#include "continuation.h"
#include <Eigen/Dense>
#include <climits>

namespace apl {

// Forward declarations
static double get_identity_for_function(Value* func);

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

void op_outer_product(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs) {
    // Outer product only uses the left function operand f
    // g should be nullptr for outer product (∘.f syntax)
    (void)g;
    if (axis) {
        m->throw_error("AXIS ERROR: outer product does not support axis");
        return;
    }

    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: outer product requires a function operand");
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

void op_inner_product(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs) {
    if (axis) {
        m->throw_error("AXIS ERROR: inner product does not support axis");
        return;
    }
    // Validate that both f and g are functions
    if (!f || !f->is_function() || !g || !g->is_function()) {
        m->throw_error("SYNTAX ERROR: inner product requires two function operands");
        return;
    }

    // Validate that lhs and rhs are data values (not functions or other non-data values)
    if (!lhs->is_basic_value() || !rhs->is_basic_value()) {
        m->throw_error("DOMAIN ERROR: inner product requires array arguments");
        return;
    }

    // ISO 9.3.2: Scalar/one-element-vector extension
    // "If A is a scalar or one-element-vector and B is not, set A1 to (1ρρB)ρA"
    // "If B is a scalar or one-element-vector and A is not, set B1 to (ρA)[ρρA]ρB"
    bool lhs_is_scalar_like = lhs->is_scalar() || (lhs->is_vector() && lhs->size() == 1);
    bool rhs_is_scalar_like = rhs->is_scalar() || (rhs->is_vector() && rhs->size() == 1);

    // Special case: both scalars or one-element vectors
    if (lhs_is_scalar_like && rhs_is_scalar_like) {
        // Scalar inner product: just apply g then return (no reduction needed)
        m->push_kont(m->heap->allocate<DispatchFunctionK>(g, lhs, rhs));
        return;
    }

    // Extend lhs if it's scalar-like and rhs is not
    if (lhs_is_scalar_like && !rhs_is_scalar_like) {
        double val = lhs->is_scalar() ? lhs->as_scalar() : (*lhs->as_matrix())(0, 0);
        int extend_len = rhs->is_vector() ? rhs->size() : rhs->rows();
        Eigen::VectorXd extended(extend_len);
        extended.setConstant(val);
        lhs = m->heap->allocate_vector(extended);
    }

    // Extend rhs if it's scalar-like and lhs is not
    if (rhs_is_scalar_like && !lhs_is_scalar_like) {
        double val = rhs->is_scalar() ? rhs->as_scalar() : (*rhs->as_matrix())(0, 0);
        int extend_len = lhs->is_vector() ? lhs->size() : lhs->cols();
        Eigen::VectorXd extended(extend_len);
        extended.setConstant(val);
        rhs = m->heap->allocate_vector(extended);
    }

    // Get dimensions (after potential extension)
    int lhs_rows = lhs->rows();
    int lhs_cols = lhs->cols();
    int rhs_rows = rhs->rows();
    int rhs_cols = rhs->cols();

    // Special case: both vectors (1D inner product)
    if (lhs->is_vector() && rhs->is_vector()) {
        // For vectors, check that lengths match
        if (lhs_rows != rhs_rows) {
            m->throw_error("LENGTH ERROR: inner product dimension mismatch");
            return;
        }
        int n = lhs_rows;  // Common dimension

        // Empty vectors: return identity element for f
        if (n == 0) {
            double identity = get_identity_for_function(f);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: no identity element for empty inner product");
                return;
            }
            m->result = m->heap->allocate_scalar(identity);
            return;
        }

        // Vector inner product: f/ (lhs g rhs)
        // Push ReduceResultK(f), then dyadic CellIterK(g) for element-wise
        m->push_kont(m->heap->allocate<ReduceResultK>(f));
        m->push_kont(m->heap->allocate<CellIterK>(
            g, lhs, rhs, 0, 0, n,
            CellIterMode::COLLECT, n, 1, true));
        return;
    }

    // Vector × Matrix case: vector treated as 1×N row
    if (lhs->is_vector() && rhs->is_matrix()) {
        // Vector length must match matrix rows
        if (lhs_rows != rhs_rows) {
            m->throw_error("LENGTH ERROR: inner product dimension mismatch");
            return;
        }
        // Result is a vector of length rhs_cols
        m->push_kont(m->heap->allocate<InnerProductIterK>(
            f, g, lhs, rhs, 1, lhs_rows, rhs_cols));
        return;
    }

    // Matrix × Vector case: vector treated as N×1 column
    if (lhs->is_matrix() && rhs->is_vector()) {
        // Matrix cols must match vector length
        if (lhs_cols != rhs_rows) {
            m->throw_error("LENGTH ERROR: inner product dimension mismatch");
            return;
        }
        // Result is a vector of length lhs_rows
        m->push_kont(m->heap->allocate<InnerProductIterK>(
            f, g, lhs, rhs, lhs_rows, lhs_cols, 1));
        return;
    }

    // General case: matrix × matrix inner product
    // LENGTH constraint: last dimension of A must equal first dimension of B
    if (lhs_cols != rhs_rows) {
        m->throw_error("LENGTH ERROR: inner product dimension mismatch");
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

void op_each(Machine* m, Value* axis, Value* f, Value* omega) {
    if (axis) {
        m->throw_error("AXIS ERROR: each operator does not support axis");
        return;
    }
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: each operator requires a function operand");
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
    bool is_char = omega->is_char_data();

    m->push_kont(m->heap->allocate<CellIterK>(
        f, nullptr, omega, 0, 0, num_cells,
        CellIterMode::COLLECT, rows, cols, omega->is_vector(), is_char));
}

// Dyadic Each: A f¨B
// Applies function element-wise to corresponding elements
// Requires shapes to match, or one arg to be scalar (scalar extension)
void op_each_dyadic(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs) {
    (void)g;  // Each only uses one function operand
    if (axis) {
        m->throw_error("AXIS ERROR: each operator does not support axis");
        return;
    }
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: each operator requires a function operand");
        return;
    }

    // Both scalars: just apply function
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->push_kont(m->heap->allocate<DispatchFunctionK>(f, lhs, rhs));
        return;
    }

    // Scalar extension: scalar with array
    if (lhs->is_scalar()) {
        // Scalar left, array right
        if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);
        int rows = rhs->rows();
        int cols = rhs->cols();
        int num_cells = rows * cols;
        bool is_char = rhs->is_char_data();
        m->push_kont(m->heap->allocate<CellIterK>(
            f, lhs, rhs, 0, 0, num_cells,
            CellIterMode::COLLECT, rows, cols, rhs->is_vector(), is_char));
        return;
    }

    if (rhs->is_scalar()) {
        // Array left, scalar right
        if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
        int rows = lhs->rows();
        int cols = lhs->cols();
        int num_cells = rows * cols;
        bool is_char = lhs->is_char_data();
        m->push_kont(m->heap->allocate<CellIterK>(
            f, lhs, rhs, 0, 0, num_cells,
            CellIterMode::COLLECT, rows, cols, lhs->is_vector(), is_char));
        return;
    }

    // Both arrays: convert strings if needed, shapes must match
    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    int lhs_rows = lhs->rows();
    int lhs_cols = lhs->cols();
    int rhs_rows = rhs->rows();
    int rhs_cols = rhs->cols();

    if (lhs_rows != rhs_rows || lhs_cols != rhs_cols) {
        m->throw_error("LENGTH ERROR: each requires matching shapes or scalar");
        return;
    }

    int num_cells = lhs_rows * lhs_cols;
    bool is_char = lhs->is_char_data() && rhs->is_char_data();
    m->push_kont(m->heap->allocate<CellIterK>(
        f, lhs, rhs, 0, 0, num_cells,
        CellIterMode::COLLECT, lhs_rows, lhs_cols, lhs->is_vector(), is_char));
}

// ========================================================================
// Duplicate Operator: f⍨B (monadic)
// ========================================================================
// Duplicate: applies f to omega twice (as both arguments)
// f⍨B → B f B

void op_commute(Machine* m, Value* axis, Value* f, Value* omega) {
    // This is the monadic form: duplicate
    // Semantics: f⍨B → B f B
    if (axis) {
        m->throw_error("AXIS ERROR: commute operator does not support axis");
        return;
    }
    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: duplicate operator requires a function operand");
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
        m->throw_error("DOMAIN ERROR: duplicate operator requires primitive, curried, or derived function");
        return;
    }

    PrimitiveFn* fn = f->data.primitive_fn;

    // f must have dyadic form for duplicate
    if (!fn->dyadic) {
        m->throw_error("DOMAIN ERROR: duplicate operator requires function with dyadic form");
        return;
    }

    // Apply: omega f omega (no axis from operator context)
    fn->dyadic(m, nullptr, omega, omega);
}

// ========================================================================
// Commute Operator: A f⍨B (dyadic)
// ========================================================================
// Commute: swaps left and right arguments
// A f⍨B → B f A

void op_commute_dyadic(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs) {
    (void)g;  // Commute only uses one function operand
    if (axis) {
        m->throw_error("AXIS ERROR: commute operator does not support axis");
        return;
    }
    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: commute operator requires a function operand");
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
        m->throw_error("DOMAIN ERROR: commute operator requires primitive, curried, or derived function");
        return;
    }

    PrimitiveFn* fn = f->data.primitive_fn;

    // f must have dyadic form
    if (!fn->dyadic) {
        m->throw_error("DOMAIN ERROR: commute operator requires function with dyadic form");
        return;
    }

    // Apply: rhs f lhs (swapped, no axis from operator context)
    fn->dyadic(m, nullptr, rhs, lhs);
}

// ========================================================================
// Reduction and Scan Operators
// ========================================================================

// Identity Elements for Reduction (ISO-13751 Table 5)
// When reducing an empty vector, return the identity element for the function
double get_identity_element(PrimitiveFn* fn) {
    // Match by function pointer
    // Arithmetic
    if (fn == &prim_plus) return 0.0;       // +/⍬ → 0
    if (fn == &prim_minus) return 0.0;      // -/⍬ → 0
    if (fn == &prim_times) return 1.0;      // ×/⍬ → 1
    if (fn == &prim_divide) return 1.0;     // ÷/⍬ → 1
    if (fn == &prim_star) return 1.0;       // */⍬ → 1

    // Min/Max - ISO 13751 §5.4: positive/negative-number-limit
    if (fn == &prim_floor) return std::numeric_limits<double>::infinity();   // ⌊/⍬ → +∞
    if (fn == &prim_ceiling) return -std::numeric_limits<double>::infinity(); // ⌈/⍬ → -∞

    // Logical
    if (fn == &prim_and) return 1.0;        // ∧/⍬ → 1
    if (fn == &prim_or) return 0.0;         // ∨/⍬ → 0

    // Comparison
    if (fn == &prim_less) return 0.0;       // </⍬ → 0
    if (fn == &prim_less_eq) return 1.0;    // ≤/⍬ → 1
    if (fn == &prim_equal) return 1.0;      // =/⍬ → 1
    if (fn == &prim_greater_eq) return 1.0; // ≥/⍬ → 1
    if (fn == &prim_greater) return 0.0;    // >/⍬ → 0
    if (fn == &prim_not_equal) return 0.0;  // ≠/⍬ → 0

    // ISO Table 5: Additional identity elements
    if (fn == &prim_stile) return 0.0;      // |/⍬ → 0 (residue)
    if (fn == &prim_factorial) return 1.0;  // !/⍬ → 1 (binomial)

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

// Helper: validate axis specification and return ⎕IO-based axis number
// Returns 0 if axis is nullptr (caller should use default)
// Returns -1 on error (after pushing ThrowErrorK)
static int validate_axis(Machine* m, Value* axis, int max_rank) {
    if (!axis) {
        return 0;  // Use default axis
    }

    // ISO 13751 §5.3.2: axis must be "scalar or one-element-vector"
    double axis_val;
    if (axis->is_scalar()) {
        axis_val = axis->as_scalar();
    } else if (axis->is_vector() && axis->size() == 1) {
        axis_val = axis->as_matrix()->coeff(0, 0);
    } else {
        m->throw_error("AXIS ERROR: axis must be scalar or one-element vector");
        return -1;
    }

    // ISO 13751 §5.2.5: axis must be a "near-integer"
    if (std::abs(axis_val - std::round(axis_val)) >= INTEGER_TOLERANCE) {
        m->throw_error("AXIS ERROR: axis must be an integer");
        return -1;
    }

    int k = static_cast<int>(std::round(axis_val));
    int io = m->io;

    if (k < io || k > max_rank - 1 + io) {
        m->throw_error("AXIS ERROR: axis out of range");
        return -1;
    }

    return k;
}

// Reduce (/) - apply dyadic function between elements, right to left
// Uses CellIterK FOLD_RIGHT for continuation-based execution
// Unified implementation handles both f/B and f/[k]B
void fn_reduce(Machine* m, Value* axis, Value* func, Value* omega) {
    // Handle replicate: if "func" is actually an array, this is A / B (replicate)
    // Note: use is_array()/is_scalar() not is_basic_value() to exclude strings
    if (func->is_array() || func->is_scalar()) {
        fn_replicate(m, axis, func, omega);
        return;
    }

    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: reduce requires a function");
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

    // Determine which axis to reduce along
    // Default for / is last axis (2 for matrix, 1 for vector)
    int max_rank = omega->is_vector() ? 1 : 2;
    int k = 0;  // 0 means use default

    if (axis) {
        k = validate_axis(m, axis, max_rank);
        if (k < 0) return;  // Error already thrown
    }

    // Default axis: last axis (k=2 for matrix means columns, k=1 for vector)
    if (k == 0) k = max_rank;

    if (omega->is_vector()) {
        // Vector only has one axis
        int len = mat->rows();
        if (len == 0) {
            // Empty vector: return identity element (ISO-13751 Table 5)
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction");
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

    // Matrix: k=1 is first axis (rows), k=2 is last axis (columns)
    int rows = mat->rows();
    int cols = mat->cols();

    if (k == 1) {
        // Reduce along first axis (like ⌿)
        if (rows == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction");
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
        m->push_kont(m->heap->allocate<RowReduceK>(func, omega, cols, rows, true));
    } else {
        // k == 2: Reduce along last axis (columns)
        if (cols == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction");
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
        m->push_kont(m->heap->allocate<RowReduceK>(func, omega, rows, cols, false));
    }
}

// Reduce-first (⌿) - reduce along first axis (rows)
// Uses CellIterK FOLD_RIGHT for continuation-based execution
// Unified implementation handles both f⌿B and f⌿[k]B
void fn_reduce_first(Machine* m, Value* axis, Value* func, Value* omega) {
    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: reduce-first requires a function");
        return;
    }

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // String → char vector conversion
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Determine which axis to reduce along
    // Default for ⌿ is first axis (1)
    int max_rank = omega->is_vector() ? 1 : 2;
    int k = 0;  // 0 means use default

    if (axis) {
        k = validate_axis(m, axis, max_rank);
        if (k < 0) return;  // Error already thrown
    }

    // Default axis: first axis (k=1)
    if (k == 0) k = 1;

    if (omega->is_vector()) {
        // Vector only has one axis - same as regular reduce
        fn_reduce(m, nullptr, func, omega);
        return;
    }

    // Matrix: k=1 is first axis (rows), k=2 is last axis (columns)
    int rows = mat->rows();
    int cols = mat->cols();

    if (k == 1) {
        // Reduce along first axis (rows)
        if (rows == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction");
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
        m->push_kont(m->heap->allocate<RowReduceK>(func, omega, cols, rows, true));
    } else {
        // k == 2: Reduce along last axis (columns)
        if (cols == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction");
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
        m->push_kont(m->heap->allocate<RowReduceK>(func, omega, rows, cols, false));
    }
}

// Scan (\) - apply dyadic function cumulatively
// ISO-13751: Item I of Z is f/B[⍳I] (reduction of first I elements)
// Uses PrefixScanK for continuation-based execution
// Unified implementation handles both f\B and f\[k]B
void fn_scan(Machine* m, Value* axis, Value* func, Value* omega) {
    // Handle expand: if "func" is actually an array, this is A \ B (expand)
    // Note: use is_array()/is_scalar() not is_basic_value() to exclude strings
    if (func->is_array() || func->is_scalar()) {
        fn_expand(m, axis, func, omega);
        return;
    }

    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: scan requires a function");
        return;
    }

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // String → char vector conversion
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Determine which axis to scan along
    // Default for \ is last axis (2 for matrix, 1 for vector)
    int max_rank = omega->is_vector() ? 1 : 2;
    int k = 0;  // 0 means use default

    if (axis) {
        k = validate_axis(m, axis, max_rank);
        if (k < 0) return;  // Error already thrown
    }

    // Default axis: last axis
    if (k == 0) k = max_rank;

    if (omega->is_vector()) {
        // Vector only has one axis
        int len = mat->rows();
        if (len == 0) {
            m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
            return;
        }
        if (len == 1) {
            // Single-element vector: preserve vector shape (ISO 13751 scan semantics)
            Eigen::VectorXd result(1);
            result(0) = (*mat)(0, 0);
            m->result = m->heap->allocate_vector(result);
            return;
        }

        // Use PrefixScanK for prefix reductions
        m->push_kont(m->heap->allocate<PrefixScanK>(func, omega, len));
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
            m->result = omega;
            return;
        }
        m->push_kont(m->heap->allocate<RowScanK>(func, omega, cols, rows, true));
    } else {
        // k == 2: Scan along last axis (columns)
        if (cols == 0) {
            m->result = m->heap->allocate_matrix(Eigen::MatrixXd(rows, 0));
            return;
        }
        if (cols == 1) {
            m->result = omega;
            return;
        }
        m->push_kont(m->heap->allocate<RowScanK>(func, omega, rows, cols, false));
    }
}

// Scan-first (⍀) - scan along first axis (rows)
// Uses PrefixScanK for continuation-based execution
// Unified implementation handles both f⍀B and f⍀[k]B
void fn_scan_first(Machine* m, Value* axis, Value* func, Value* omega) {
    // Handle expand-first: if "func" is actually an array, this is A ⍀ B (expand-first)
    if (func->is_array() || func->is_scalar()) {
        if (axis) {
            m->throw_error("SYNTAX ERROR: expand does not support axis");
            return;
        }
        fn_expand_first(m, nullptr, func, omega);
        return;
    }

    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: scan-first requires a function");
        return;
    }

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // String → char vector conversion
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    const Eigen::MatrixXd* mat = omega->as_matrix();

    // Determine which axis to scan along
    // Default for ⍀ is first axis (1)
    int max_rank = omega->is_vector() ? 1 : 2;
    int k = 0;  // 0 means use default

    if (axis) {
        k = validate_axis(m, axis, max_rank);
        if (k < 0) return;  // Error already thrown
    }

    // Default axis: first axis (k=1)
    if (k == 0) k = 1;

    if (omega->is_vector()) {
        // Vector only has one axis - same as regular scan
        fn_scan(m, nullptr, func, omega);
        return;
    }

    // Matrix: k=1 is first axis (rows), k=2 is last axis (columns)
    int rows = mat->rows();
    int cols = mat->cols();

    if (k == 1) {
        // Scan along first axis (rows)
        if (rows == 0) {
            m->result = m->heap->allocate_matrix(Eigen::MatrixXd(0, cols));
            return;
        }
        if (rows == 1) {
            m->result = omega;
            return;
        }
        m->push_kont(m->heap->allocate<RowScanK>(func, omega, cols, rows, true));
    } else {
        // k == 2: Scan along last axis (columns)
        if (cols == 0) {
            m->result = m->heap->allocate_matrix(Eigen::MatrixXd(rows, 0));
            return;
        }
        if (cols == 1) {
            m->result = omega;
            return;
        }
        m->push_kont(m->heap->allocate<RowScanK>(func, omega, rows, cols, false));
    }
}

// ========================================================================
// N-wise Reduction (N f/ B) - dyadic forms
// ========================================================================

// Helper: validate N for N-wise reduction
// Returns validated N, or INT_MIN on error (since 0 is a valid N value)
static int validate_nwise(Machine* m, Value* n_val, int axis_len) {
    if (!n_val->is_scalar()) {
        m->throw_error("RANK ERROR: N must be a scalar");
        return INT_MIN;
    }

    double n_double = n_val->as_scalar();
    int n = static_cast<int>(n_double);

    if (n_double != static_cast<double>(n)) {
        m->throw_error("DOMAIN ERROR: N must be an integer");
        return INT_MIN;
    }

    // Handle negative N (reverse before reduce) - take absolute value
    int abs_n = n < 0 ? -n : n;

    if (abs_n > axis_len + 1) {
        m->throw_error("DOMAIN ERROR: N too large for array");
        return INT_MIN;
    }

    return n;  // Return original (possibly negative) for reverse handling
}

// N-wise reduction: N f/ B or N f/[k] B
// Dyadic form where lhs is N (window size)
void fn_reduce_nwise(Machine* m, Value* axis, Value* lhs, Value* func, Value* g, Value* rhs) {
    (void)g;  // Reduce only uses one function operand
    Value* n_val = lhs;

    // Handle replicate: if "func" is actually an array, N-wise doesn't apply
    if (func->is_array() || func->is_scalar()) {
        m->throw_error("SYNTAX ERROR: replicate does not support N-wise");
        return;
    }

    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: reduce requires a function");
        return;
    }

    if (rhs->is_scalar()) {
        // Scalar handling for N-wise
        int n = validate_nwise(m, n_val, 1);
        if (n == INT_MIN) return;
        int abs_n = n < 0 ? -n : n;
        if (abs_n > 2) {
            m->throw_error("DOMAIN ERROR: N too large for scalar");
            return;
        }
        if (abs_n == 0) {
            // 0 f/ scalar → 2-element result with identity
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element");
                return;
            }
            Eigen::VectorXd result(2);
            result << identity, identity;
            m->result = m->heap->allocate_vector(result);
            return;
        }
        // abs_n == 1 or 2: reshape scalar
        Eigen::VectorXd result(2 - abs_n);
        if (result.rows() > 0) result(0) = rhs->as_scalar();
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // String to char vector conversion
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Determine axis (default is last axis = 2 for matrix, 1 for vector)
    int max_rank = rhs->is_vector() ? 1 : 2;
    int k = 0;
    if (axis) {
        k = validate_axis(m, axis, max_rank);
        if (k < 0) return;
    }
    if (k == 0) k = max_rank;  // Default: last axis

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        int len = mat->rows();
        int n = validate_nwise(m, n_val, len);
        if (n == INT_MIN) return;
        int abs_n = n < 0 ? -n : n;
        bool reverse = n < 0;

        if (abs_n == 0) {
            // 0 f/ vec → (1+len) copies of identity
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element");
                return;
            }
            Eigen::VectorXd result = Eigen::VectorXd::Constant(len + 1, identity);
            m->result = m->heap->allocate_vector(result);
            return;
        }

        // Result length = len - abs_n + 1
        int result_len = len - abs_n + 1;
        if (result_len < 0) {
            m->throw_error("DOMAIN ERROR: N too large for vector");
            return;
        }
        if (result_len == 0) {
            Eigen::VectorXd empty(0);
            m->result = m->heap->allocate_vector(empty);
            return;
        }

        // Use NwiseReduceK continuation for N-wise reduction
        m->push_kont(m->heap->allocate<NwiseReduceK>(func, rhs, abs_n, reverse));
        return;
    }

    // Matrix: k=1 is first axis (rows), k=2 is last axis (columns)
    int rows = mat->rows();
    int cols = mat->cols();
    int axis_len = (k == 1) ? rows : cols;

    int n = validate_nwise(m, n_val, axis_len);
    if (n == INT_MIN) return;
    int abs_n = n < 0 ? -n : n;
    bool reverse = n < 0;

    if (abs_n == 0) {
        // 0 f/[k] matrix → expand axis by 1, fill with identity
        double identity = get_identity_for_function(func);
        if (std::isnan(identity)) {
            m->throw_error("DOMAIN ERROR: function has no identity element");
            return;
        }
        if (k == 1) {
            Eigen::MatrixXd result = Eigen::MatrixXd::Constant(rows + 1, cols, identity);
            m->result = m->heap->allocate_matrix(result);
        } else {
            Eigen::MatrixXd result = Eigen::MatrixXd::Constant(rows, cols + 1, identity);
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    // N-wise on matrix along axis k
    m->push_kont(m->heap->allocate<NwiseMatrixReduceK>(func, rhs, abs_n, k == 1, reverse));
}

// N-wise reduction-first: N f⌿ B or N f⌿[k] B
// Default axis is first (1) instead of last (2)
void fn_reduce_first_nwise(Machine* m, Value* axis, Value* lhs, Value* func, Value* g, Value* rhs) {
    (void)g;  // Reduce only uses one function operand
    Value* n_val = lhs;

    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: reduce requires a function");
        return;
    }

    if (rhs->is_scalar()) {
        // Scalar handling for N-wise (same as fn_reduce_nwise)
        int n = validate_nwise(m, n_val, 1);
        if (n == INT_MIN) return;
        int abs_n = n < 0 ? -n : n;
        if (abs_n > 2) {
            m->throw_error("DOMAIN ERROR: N too large for scalar");
            return;
        }
        if (abs_n == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element");
                return;
            }
            Eigen::VectorXd result(2);
            result << identity, identity;
            m->result = m->heap->allocate_vector(result);
            return;
        }
        Eigen::VectorXd result(2 - abs_n);
        if (result.rows() > 0) result(0) = rhs->as_scalar();
        m->result = m->heap->allocate_vector(result);
        return;
    }

    // String to char vector conversion
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    // Determine axis (default for ⌿ is first axis = 1)
    int max_rank = rhs->is_vector() ? 1 : 2;
    int k = 0;
    if (axis) {
        k = validate_axis(m, axis, max_rank);
        if (k < 0) return;
    }
    if (k == 0) k = 1;  // Default: first axis

    const Eigen::MatrixXd* mat = rhs->as_matrix();

    if (rhs->is_vector()) {
        // For vector, same as regular N-wise reduce
        fn_reduce_nwise(m, nullptr, lhs, func, g, rhs);
        return;
    }

    // Matrix: k=1 is first axis (rows), k=2 is last axis (columns)
    int rows = mat->rows();
    int cols = mat->cols();
    int axis_len = (k == 1) ? rows : cols;

    int n = validate_nwise(m, n_val, axis_len);
    if (n == INT_MIN) return;
    int abs_n = n < 0 ? -n : n;
    bool reverse = n < 0;

    if (abs_n == 0) {
        double identity = get_identity_for_function(func);
        if (std::isnan(identity)) {
            m->throw_error("DOMAIN ERROR: function has no identity element");
            return;
        }
        if (k == 1) {
            Eigen::MatrixXd result = Eigen::MatrixXd::Constant(rows + 1, cols, identity);
            m->result = m->heap->allocate_matrix(result);
        } else {
            Eigen::MatrixXd result = Eigen::MatrixXd::Constant(rows, cols + 1, identity);
            m->result = m->heap->allocate_matrix(result);
        }
        return;
    }

    m->push_kont(m->heap->allocate<NwiseMatrixReduceK>(func, rhs, abs_n, k == 1, reverse));
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
    op_each_dyadic        // Dyadic: A f¨B
};

PrimitiveOp op_tilde = {
    "⍨",
    op_commute,           // Monadic: f⍨B (duplicate)
    op_commute_dyadic     // Dyadic: A f⍨B (commute)
};

// Reduction operators
// Monadic handles both f/B and f/[k]B (axis passed via curry)
// Dyadic handles N f/B and N f/[k]B (N-wise reduction)
PrimitiveOp op_reduce = {
    "/",
    fn_reduce,            // Monadic: f/B or f/[k]B
    fn_reduce_nwise       // Dyadic: N f/B or N f/[k]B
};

PrimitiveOp op_reduce_first = {
    "⌿",
    fn_reduce_first,      // Monadic: f⌿B or f⌿[k]B
    fn_reduce_first_nwise // Dyadic: N f⌿B or N f⌿[k]B
};

// Scan operators
// Monadic handles both f\B and f\[k]B (axis passed via curry)
// No dyadic form (scan doesn't have N-wise)
PrimitiveOp op_scan = {
    "\\",
    fn_scan,              // Monadic: f\B or f\[k]B
    nullptr               // No dyadic form
};

PrimitiveOp op_scan_first = {
    "⍀",
    fn_scan_first,        // Monadic: f⍀B or f⍀[k]B
    nullptr               // No dyadic form
};

PrimitiveOp op_catenate_axis = {
    ",⌷",
    fn_catenate_axis_monadic,  // Monadic: ,[k]B (ravel along axis)
    fn_catenate_axis_dyadic    // Dyadic: A ,[k] B (catenate/laminate)
};

PrimitiveOp op_rank_op = {
    "⍤",
    nullptr,              // No monadic form - rank requires rank specification
    op_rank               // Dyadic: A f⍤k B
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

// Helper: check if a double is an integer value
static bool is_integer_value(double x) {
    return std::floor(x) == x && !std::isinf(x) && !std::isnan(x);
}

// Helper: extract rank values from rank specification
static bool parse_rank_spec(Value* rank_spec, int array_rank,
                            int* monadic_rank, int* left_rank, int* right_rank) {
    if (rank_spec->is_scalar()) {
        double val = rank_spec->as_scalar();
        // Rank must be an integer
        if (!is_integer_value(val)) {
            return false;  // DOMAIN ERROR: rank must be integer
        }
        int k = static_cast<int>(val);
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
void op_rank_monadic(Machine* m, Value* axis, Value* f, Value* omega) {
    (void)axis; (void)f; (void)omega;
    m->throw_error("SYNTAX ERROR: rank operator requires rank specification");
}

// Dyadic rank: A f⍤k B (or monadic f⍤k B where lhs is nullptr)
void op_rank(Machine* m, Value* axis, Value* lhs, Value* f, Value* rank_spec, Value* rhs) {
    if (axis) {
        m->throw_error("AXIS ERROR: rank operator does not support axis");
        return;
    }
    // Validate function operand
    if (!f || !f->is_function()) {
        m->throw_error("DOMAIN ERROR: rank operator requires function operand");
        return;
    }

    // Validate rank specification
    if (!rank_spec) {
        m->throw_error("DOMAIN ERROR: rank operator requires rank specification");
        return;
    }

    // Parse rank specification
    int rhs_rank = get_array_rank(rhs);
    int lhs_rank = lhs ? get_array_rank(lhs) : 0;
    int monadic_r, left_r, right_r;

    if (!parse_rank_spec(rank_spec, std::max(lhs_rank, rhs_rank), &monadic_r, &left_r, &right_r)) {
        m->throw_error("DOMAIN ERROR: invalid rank specification");
        return;
    }

    bool is_dyadic = (lhs != nullptr);

    if (!is_dyadic) {
        // Monadic: f⍤k B
        int k = std::min(monadic_r, rhs_rank);
        if (k < 0) k = std::max(0, rhs_rank + k);

        int num_cells = count_cells(rhs, k);

        // Handle empty array: return empty array with same shape
        if (num_cells == 0) {
            if (rhs->is_vector()) {
                m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
            } else {
                m->result = m->heap->allocate_matrix(Eigen::MatrixXd(rhs->rows(), rhs->cols()));
            }
            return;
        }

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
            m->throw_error("LENGTH ERROR: rank operator cell count mismatch");
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

// ========================================================================
// Catenate/Laminate with Axis: ,[k] (ISO 13751 §10.2.1)
// ========================================================================

static bool is_near_integer(double x) {
    return std::abs(x - std::round(x)) < INTEGER_TOLERANCE;
}

void fn_catenate_axis_monadic(Machine* m, Value* curry_axis, Value* axis_operand, Value* omega) {
    // curry_axis would be from f[k] syntax - not supported on ,[k]
    if (curry_axis) {
        m->throw_error("AXIS ERROR: catenate-axis does not support additional axis modifier");
        return;
    }
    // axis_operand is the k from ,[k] syntax
    // ISO 13751 §5.3.2: axis must be "scalar or one-element-vector"
    double k;
    if (axis_operand->is_scalar()) {
        k = axis_operand->as_scalar();
    } else if (axis_operand->is_vector() && axis_operand->size() == 1) {
        k = axis_operand->as_matrix()->coeff(0, 0);
    } else {
        m->throw_error("AXIS ERROR: axis must be scalar or one-element vector");
        return;
    }
    int axis_idx = static_cast<int>(std::round(k)) - m->io;  // ⎕IO

    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_vector(Eigen::VectorXd::Constant(1, omega->as_scalar()));
        return;
    }

    if (omega->is_vector()) {
        if (axis_idx != 0) {
            m->throw_error("AXIS ERROR: axis out of range");
            return;
        }
        m->result = m->heap->allocate_vector(omega->as_matrix()->col(0), omega->is_char_data());
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (axis_idx < 0 || axis_idx > 1) {
        m->throw_error("AXIS ERROR: axis out of range");
        return;
    }

    if (axis_idx == 0) {
        Eigen::VectorXd result(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i * cols + j) = (*mat)(i, j);
            }
        }
        m->result = m->heap->allocate_vector(result, omega->is_char_data());
    } else {
        Eigen::VectorXd result(rows * cols);
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                result(j * rows + i) = (*mat)(i, j);
            }
        }
        m->result = m->heap->allocate_vector(result, omega->is_char_data());
    }
}

void fn_catenate_axis_dyadic(Machine* m, Value* curry_axis, Value* lhs, Value* axis_operand, Value* unused, Value* rhs) {
    (void)unused;
    // curry_axis would be from f[k] syntax - not supported on ,[k]
    if (curry_axis) {
        m->throw_error("AXIS ERROR: catenate-axis does not support additional axis modifier");
        return;
    }
    // axis_operand is the k from ,[k] syntax
    // ISO 13751 §5.3.2: axis must be "scalar or one-element-vector"
    double k;
    if (axis_operand->is_scalar()) {
        k = axis_operand->as_scalar();
    } else if (axis_operand->is_vector() && axis_operand->size() == 1) {
        k = axis_operand->as_matrix()->coeff(0, 0);
    } else {
        m->throw_error("AXIS ERROR: axis must be scalar or one-element vector");
        return;
    }
    bool laminate = !is_near_integer(k);
    int axis_idx = static_cast<int>(std::floor(k)) - m->io;  // ⎕IO

    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();
    int lrows = lmat->rows();
    int lcols = lmat->cols();
    int rrows = rmat->rows();
    int rcols = rmat->cols();

    if (laminate) {
        if (lhs->is_vector() && rhs->is_vector()) {
            if (lrows != rrows) {
                m->throw_error("LENGTH ERROR: vectors must have same length for laminate");
                return;
            }
            if (axis_idx < 0) {
                Eigen::MatrixXd result(2, lrows);
                result.row(0) = lmat->col(0).transpose();
                result.row(1) = rmat->col(0).transpose();
                m->result = m->heap->allocate_matrix(result);
            } else {
                Eigen::MatrixXd result(lrows, 2);
                result.col(0) = lmat->col(0);
                result.col(1) = rmat->col(0);
                m->result = m->heap->allocate_matrix(result);
            }
            return;
        }

        // Scalar extension for laminate
        if (lhs->is_vector() && rhs->is_scalar()) {
            double scalar_val = rhs->as_scalar();
            if (axis_idx < 0) {
                Eigen::MatrixXd result(2, lrows);
                result.row(0) = lmat->col(0).transpose();
                result.row(1).setConstant(scalar_val);
                m->result = m->heap->allocate_matrix(result);
            } else {
                Eigen::MatrixXd result(lrows, 2);
                result.col(0) = lmat->col(0);
                result.col(1).setConstant(scalar_val);
                m->result = m->heap->allocate_matrix(result);
            }
            return;
        }

        if (lhs->is_scalar() && rhs->is_vector()) {
            double scalar_val = lhs->as_scalar();
            if (axis_idx < 0) {
                Eigen::MatrixXd result(2, rrows);
                result.row(0).setConstant(scalar_val);
                result.row(1) = rmat->col(0).transpose();
                m->result = m->heap->allocate_matrix(result);
            } else {
                Eigen::MatrixXd result(rrows, 2);
                result.col(0).setConstant(scalar_val);
                result.col(1) = rmat->col(0);
                m->result = m->heap->allocate_matrix(result);
            }
            return;
        }

        if (lhs->is_scalar() && rhs->is_scalar()) {
            Eigen::VectorXd result(2);
            result(0) = lhs->as_scalar();
            result(1) = rhs->as_scalar();
            m->result = m->heap->allocate_vector(result);
            return;
        }

        m->throw_error("RANK ERROR: laminate for matrices not yet implemented");
        return;
    }

    int cat_axis = static_cast<int>(std::round(k)) - m->io;  // ⎕IO

    // Scalars have no axes - cannot catenate with axis specification
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->throw_error("AXIS ERROR: cannot catenate scalars with axis");
        return;
    }

    if (lhs->is_scalar() && rhs->is_vector()) {
        Eigen::VectorXd result(1 + rrows);
        result(0) = lhs->as_scalar();
        result.tail(rrows) = rmat->col(0);
        m->result = m->heap->allocate_vector(result, rhs->is_char_data());
        return;
    }

    if (lhs->is_vector() && rhs->is_scalar()) {
        Eigen::VectorXd result(lrows + 1);
        result.head(lrows) = lmat->col(0);
        result(lrows) = rhs->as_scalar();
        m->result = m->heap->allocate_vector(result, lhs->is_char_data());
        return;
    }

    if (lhs->is_vector() && rhs->is_vector()) {
        if (cat_axis != 0) {
            m->throw_error("AXIS ERROR: axis out of range");
            return;
        }
        Eigen::VectorXd result(lrows + rrows);
        result.head(lrows) = lmat->col(0);
        result.tail(rrows) = rmat->col(0);
        m->result = m->heap->allocate_vector(result, lhs->is_char_data() || rhs->is_char_data());
        return;
    }

    if (cat_axis == 0) {
        if (lcols != rcols) {
            m->throw_error("LENGTH ERROR: column counts must match for vertical catenation");
            return;
        }
        Eigen::MatrixXd result(lrows + rrows, lcols);
        result.topRows(lrows) = *lmat;
        result.bottomRows(rrows) = *rmat;
        m->result = m->heap->allocate_matrix(result);
    } else if (cat_axis == 1) {
        if (lrows != rrows) {
            m->throw_error("LENGTH ERROR: row counts must match for horizontal catenation");
            return;
        }
        Eigen::MatrixXd result(lrows, lcols + rcols);
        result.leftCols(lcols) = *lmat;
        result.rightCols(rcols) = *rmat;
        m->result = m->heap->allocate_matrix(result);
    } else {
        m->throw_error("AXIS ERROR: axis out of range");
    }
}

} // namespace apl
