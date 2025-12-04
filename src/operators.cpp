// Operators implementation

#include "operators.h"
#include "machine.h"
#include "heap.h"
#include "primitives.h"
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

    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: outer product requires a function operand"));
        return;
    }

    // Get the primitive function
    if (!f->is_primitive()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: outer product currently only supports primitive functions"));
        return;
    }

    PrimitiveFn* fn = f->data.primitive_fn;
    if (!fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: outer product requires a dyadic function"));
        return;
    }

    // Get dimensions of both arguments
    int lhs_rows = lhs->rows();
    int lhs_cols = lhs->cols();
    int lhs_size = lhs_rows * lhs_cols;

    int rhs_rows = rhs->rows();
    int rhs_cols = rhs->cols();
    int rhs_size = rhs_rows * rhs_cols;

    // Result dimensions for outer product
    // If A is scalar: result has shape of B
    // If B is scalar: result has shape of A
    // If both arrays: result is (⍴A),⍴B

    // For now, implement the simple 2D case: both are vectors or scalars
    // Full spec would handle arbitrary rank combinations

    int result_rows = lhs_size;  // Total elements in lhs
    int result_cols = rhs_size;  // Total elements in rhs

    // Create result matrix on stack (GC will manage the final copy)
    Eigen::MatrixXd result(result_rows, result_cols);

    // Apply f to every combination
    // Outer product uses raveled values (treated as 1D)
    for (int i = 0; i < lhs_size; i++) {
        for (int j = 0; j < rhs_size; j++) {
            // Get scalar values from raveled arrays
            double lhs_val, rhs_val;

            if (lhs->is_scalar()) {
                lhs_val = lhs->as_scalar();
            } else {
                const Eigen::MatrixXd* lhs_mat = lhs->as_matrix();
                lhs_val = (*lhs_mat)(i / lhs_cols, i % lhs_cols);
            }

            if (rhs->is_scalar()) {
                rhs_val = rhs->as_scalar();
            } else {
                const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();
                rhs_val = (*rhs_mat)(j / rhs_cols, j % rhs_cols);
            }

            // Create temporary scalar values for function application
            Value* temp_lhs = m->heap->allocate_scalar(lhs_val);
            Value* temp_rhs = m->heap->allocate_scalar(rhs_val);

            // Apply the function
            fn->dyadic(m, temp_lhs, temp_rhs);

            // Check for error
            if (!m->kont_stack.empty() && dynamic_cast<ThrowErrorK*>(m->kont_stack.back())) {
                return;  // Error continuation already pushed
            }

            // Get result from ctrl.value
            Value* item_result = m->ctrl.value;
            if (!item_result || !item_result->is_scalar()) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: outer product function must return scalar"));
                return;
            }

            result(i, j) = item_result->as_scalar();
        }
    }

    // Return result
    // If result is 1×1, return as scalar
    if (result_rows == 1 && result_cols == 1) {
        m->ctrl.set_value(m->heap->allocate_scalar(result(0, 0)));
    } else if (result_cols == 1) {
        // Return as vector
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        // Return as matrix
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
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

    // For now, only support primitive functions
    if (!f->is_primitive() || !g->is_primitive()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: inner product currently only supports primitive functions"));
        return;
    }

    PrimitiveFn* f_fn = f->data.primitive_fn;
    PrimitiveFn* g_fn = g->data.primitive_fn;

    // f must have dyadic form (for reduction)
    if (!f_fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: left operand of inner product must have dyadic form"));
        return;
    }

    // g must have dyadic form
    if (!g_fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: right operand of inner product must have dyadic form"));
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
        // Result is scalar: f/A g B
        const Eigen::MatrixXd* lhs_mat = lhs->as_matrix();
        const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();

        // Apply g element-wise
        double accumulator = 0.0;
        bool first = true;

        for (int i = 0; i < n; i++) {
            double lhs_val = (*lhs_mat)(i, 0);
            double rhs_val = (*rhs_mat)(i, 0);

            // Apply g
            Value* temp_lhs = m->heap->allocate_scalar(lhs_val);
            Value* temp_rhs = m->heap->allocate_scalar(rhs_val);
            g_fn->dyadic(m, temp_lhs, temp_rhs);

            if (!m->kont_stack.empty() && dynamic_cast<ThrowErrorK*>(m->kont_stack.back())) {
                return;  // Error in g
            }

            Value* g_result = m->ctrl.value;
            if (!g_result || !g_result->is_scalar()) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: inner product g must return scalar"));
                return;
            }

            // Reduce with f
            if (first) {
                accumulator = g_result->as_scalar();
                first = false;
            } else {
                Value* acc_val = m->heap->allocate_scalar(accumulator);
                Value* new_val = m->heap->allocate_scalar(g_result->as_scalar());
                f_fn->dyadic(m, acc_val, new_val);

                if (!m->kont_stack.empty() && dynamic_cast<ThrowErrorK*>(m->kont_stack.back())) {
                    return;  // Error in f
                }

                Value* f_result = m->ctrl.value;
                if (!f_result || !f_result->is_scalar()) {
                    m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: inner product f must return scalar"));
                    return;
                }

                accumulator = f_result->as_scalar();
            }
        }

        m->ctrl.set_value(m->heap->allocate_scalar(accumulator));
        return;
    }

    // General case: matrix inner product
    // LENGTH constraint: last dimension of A must equal first dimension of B
    if (lhs_cols != rhs_rows) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("LENGTH ERROR: inner product dimension mismatch"));
        return;
    }

    int n = lhs_cols;  // Common dimension

    // Result dimensions: lhs_rows × rhs_cols
    int result_rows = lhs_rows;
    int result_cols = rhs_cols;

    Eigen::MatrixXd result(result_rows, result_cols);

    const Eigen::MatrixXd* lhs_mat = lhs->as_matrix();
    const Eigen::MatrixXd* rhs_mat = rhs->as_matrix();

    // For each position in result
    for (int i = 0; i < result_rows; i++) {
        for (int j = 0; j < result_cols; j++) {
            // Compute inner product of row i of lhs with column j of rhs
            double accumulator = 0.0;
            bool first = true;

            for (int k = 0; k < n; k++) {
                double lhs_val = (*lhs_mat)(i, k);
                double rhs_val = (*rhs_mat)(k, j);

                // Apply g
                Value* temp_lhs = m->heap->allocate_scalar(lhs_val);
                Value* temp_rhs = m->heap->allocate_scalar(rhs_val);
                g_fn->dyadic(m, temp_lhs, temp_rhs);

                if (!m->kont_stack.empty() && dynamic_cast<ThrowErrorK*>(m->kont_stack.back())) {
                    return;  // Error in g
                }

                Value* g_result = m->ctrl.value;
                if (!g_result || !g_result->is_scalar()) {
                    m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: inner product g must return scalar"));
                    return;
                }

                // Reduce with f
                if (first) {
                    accumulator = g_result->as_scalar();
                    first = false;
                } else {
                    Value* acc_val = m->heap->allocate_scalar(accumulator);
                    Value* new_val = m->heap->allocate_scalar(g_result->as_scalar());
                    f_fn->dyadic(m, acc_val, new_val);

                    if (!m->kont_stack.empty() && dynamic_cast<ThrowErrorK*>(m->kont_stack.back())) {
                        return;  // Error in f
                    }

                    Value* f_result = m->ctrl.value;
                    if (!f_result || !f_result->is_scalar()) {
                        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: inner product f must return scalar"));
                        return;
                    }

                    accumulator = f_result->as_scalar();
                }
            }

            result(i, j) = accumulator;
        }
    }

    // Return result based on shape
    if (result_rows == 1 && result_cols == 1) {
        m->ctrl.set_value(m->heap->allocate_scalar(result(0, 0)));
    } else if (result_cols == 1) {
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
}

// ========================================================================
// Each Operator: f¨B (monadic) or A f¨B (dyadic)
// ========================================================================
// Applies function to each element independently
// Result has same shape as argument(s)

void op_each(Machine* m, Value* f, Value* omega) {
    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("SYNTAX ERROR: each operator requires a function operand"));
        return;
    }

    // For now, only support primitive functions
    if (!f->is_primitive()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: each operator currently only supports primitive functions"));
        return;
    }

    PrimitiveFn* fn = f->data.primitive_fn;

    // f must have monadic form
    if (!fn->monadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: each operator requires function with monadic form"));
        return;
    }

    // Apply function to each element
    if (omega->is_scalar()) {
        // Scalar case: just apply function
        fn->monadic(m, omega);
        return;
    }

    // Array case: apply to each element
    const Eigen::MatrixXd* omega_mat = omega->as_matrix();
    int rows = omega_mat->rows();
    int cols = omega_mat->cols();

    Eigen::MatrixXd result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = (*omega_mat)(i, j);
            Value* temp = m->heap->allocate_scalar(val);

            fn->monadic(m, temp);

            if (!m->kont_stack.empty() && dynamic_cast<ThrowErrorK*>(m->kont_stack.back())) {
                return;  // Error in function
            }

            Value* item_result = m->ctrl.value;
            if (!item_result || !item_result->is_scalar()) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: each operator function must return scalar"));
                return;
            }

            result(i, j) = item_result->as_scalar();
        }
    }

    // Return result with same shape as input
    if (omega->is_vector()) {
        m->ctrl.set_value(m->heap->allocate_vector(result.col(0)));
    } else {
        m->ctrl.set_value(m->heap->allocate_matrix(result));
    }
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

    // Handle primitive functions
    if (!f->is_primitive()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: duplicate operator requires primitive or curried function"));
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

    // Handle primitive functions
    if (!f->is_primitive()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: commute operator requires primitive or curried function"));
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

// Reduce (/) - apply dyadic function between elements, right to left
void fn_reduce(Machine* m, Value* func, Value* omega) {
    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reduce requires a function"));
        return;
    }

    PrimitiveFn* fn = func->data.primitive_fn;
    if (!fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reduce requires a dyadic function"));
        return;
    }

    if (omega->is_scalar()) {
        // Reducing a scalar is identity
        m->ctrl.set_value(m->heap->allocate_scalar(omega->as_scalar()));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        // Reduce vector to scalar
        int len = mat->rows();
        if (len == 0) {
            // Empty vector: return identity element (ISO-13751 Table 5)
            double identity = get_identity_element(fn);
            if (std::isnan(identity)) {
                m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: function has no identity element for empty reduction"));
                return;
            }
            m->ctrl.set_value(m->heap->allocate_scalar(identity));
            return;
        }
        if (len == 1) {
            m->ctrl.set_value(m->heap->allocate_scalar((*mat)(0, 0)));
            return;
        }

        // Right-to-left reduction (APL standard: first f (f/rest))
        Value* acc = m->heap->allocate_scalar((*mat)(len - 1, 0));
        for (int i = len - 2; i >= 0; --i) {
            Value* elem = m->heap->allocate_scalar((*mat)(i, 0));
            fn->dyadic(m, elem, acc);
            // GC will clean up elem and old acc
            acc = m->ctrl.value;
        }
        m->ctrl.set_value(acc);
        return;
    }

    // For matrix, reduce along last axis (columns)
    // Result is a vector with one element per row
    int rows = mat->rows();
    int cols = mat->cols();

    if (cols == 0) {
        // Empty dimension: return vector of identity elements
        double identity = get_identity_element(fn);
        if (std::isnan(identity)) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: function has no identity element for empty reduction"));
            return;
        }
        Eigen::VectorXd result = Eigen::VectorXd::Constant(rows, identity);
        m->ctrl.set_value(m->heap->allocate_vector(result));
        return;
    }

    Eigen::VectorXd result(rows);

    for (int r = 0; r < rows; ++r) {
        // Reduce this row right-to-left
        Value* acc = m->heap->allocate_scalar((*mat)(r, cols - 1));
        for (int c = cols - 2; c >= 0; --c) {
            Value* elem = m->heap->allocate_scalar((*mat)(r, c));
            fn->dyadic(m, elem, acc);
            // GC will clean up elem and old acc
            acc = m->ctrl.value;
        }
        result(r) = acc->as_scalar();
        // GC will clean up acc
    }

    m->ctrl.set_value(m->heap->allocate_vector(result));
}

// Reduce-first (⌿) - reduce along first axis (rows)
void fn_reduce_first(Machine* m, Value* func, Value* omega) {
    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reduce-first requires a function"));
        return;
    }

    PrimitiveFn* fn = func->data.primitive_fn;
    if (!fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: reduce-first requires a dyadic function"));
        return;
    }

    if (omega->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(omega->as_scalar()));
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
        double identity = get_identity_element(fn);
        if (std::isnan(identity)) {
            m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: function has no identity element for empty reduction"));
            return;
        }
        Eigen::VectorXd result = Eigen::VectorXd::Constant(cols, identity);
        m->ctrl.set_value(m->heap->allocate_vector(result));
        return;
    }

    Eigen::VectorXd result(cols);

    for (int c = 0; c < cols; ++c) {
        // Reduce this column bottom-to-top (right-to-left in first axis)
        Value* acc = m->heap->allocate_scalar((*mat)(rows - 1, c));
        for (int r = rows - 2; r >= 0; --r) {
            Value* elem = m->heap->allocate_scalar((*mat)(r, c));
            fn->dyadic(m, elem, acc);
            // GC will clean up elem and old acc
            acc = m->ctrl.value;
        }
        result(c) = acc->as_scalar();
        // GC will clean up acc
    }

    m->ctrl.set_value(m->heap->allocate_vector(result));
}

// Scan (\) - apply dyadic function cumulatively
// ISO-13751: Item I of Z is f/B[⍳I] (reduction of first I elements)
// NOTE: For associative functions like + and ×, left-to-right accumulation
// gives the same result as calling reduce on each prefix. But for non-associative
// functions (-, ÷), we must actually perform the full reduction to match the spec.
void fn_scan(Machine* m, Value* func, Value* omega) {
    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: scan requires a function"));
        return;
    }

    PrimitiveFn* fn = func->data.primitive_fn;
    if (!fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: scan requires a dyadic function"));
        return;
    }

    if (omega->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(omega->as_scalar()));
        return;
    }

    const Eigen::MatrixXd* mat = omega->as_matrix();

    if (omega->is_vector()) {
        int len = mat->rows();
        if (len == 0) {
            m->ctrl.set_value(m->heap->allocate_vector(Eigen::VectorXd(0)));
            return;
        }

        Eigen::VectorXd result(len);

        // ISO-13751: Each element I is f/B[⍳I] (full reduction of prefix)
        for (int i = 0; i < len; ++i) {
            // Perform reduction on B[0..i] (right-to-left as per reduce spec)
            if (i == 0) {
                result(i) = (*mat)(0, 0);
            } else {
                // Right-to-left reduction: first f (f/(rest))
                Value* acc = m->heap->allocate_scalar((*mat)(i, 0));
                for (int j = i - 1; j >= 0; --j) {
                    Value* elem = m->heap->allocate_scalar((*mat)(j, 0));
                    fn->dyadic(m, elem, acc);
                    acc = m->ctrl.value;
                    // GC will clean up old values
                }
                result(i) = acc->as_scalar();
            }
        }

        m->ctrl.set_value(m->heap->allocate_vector(result));
        return;
    }

    // For matrix, scan along last axis (columns)
    int rows = mat->rows();
    int cols = mat->cols();

    Eigen::MatrixXd result(rows, cols);

    for (int r = 0; r < rows; ++r) {
        // Scan this row: each column is reduction of columns [0..c]
        for (int c = 0; c < cols; ++c) {
            if (c == 0) {
                result(r, c) = (*mat)(r, 0);
            } else {
                // Right-to-left reduction of row r, columns [0..c]
                Value* acc = m->heap->allocate_scalar((*mat)(r, c));
                for (int j = c - 1; j >= 0; --j) {
                    Value* elem = m->heap->allocate_scalar((*mat)(r, j));
                    fn->dyadic(m, elem, acc);
                    acc = m->ctrl.value;
                }
                result(r, c) = acc->as_scalar();
            }
        }
    }

    m->ctrl.set_value(m->heap->allocate_matrix(result));
}

// Scan-first (⍀) - scan along first axis (rows)
void fn_scan_first(Machine* m, Value* func, Value* omega) {
    if (!func->is_function()) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: scan-first requires a function"));
        return;
    }

    PrimitiveFn* fn = func->data.primitive_fn;
    if (!fn->dyadic) {
        m->push_kont(m->heap->allocate<ThrowErrorK>("DOMAIN ERROR: scan-first requires a dyadic function"));
        return;
    }

    if (omega->is_scalar()) {
        m->ctrl.set_value(m->heap->allocate_scalar(omega->as_scalar()));
        return;
    }

    if (omega->is_vector()) {
        // For vector, same as regular scan
        fn_scan(m, func, omega);
        return;
    }

    // For matrix, scan along first axis (rows)
    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    Eigen::MatrixXd result(rows, cols);

    for (int c = 0; c < cols; ++c) {
        // Scan this column bottom-to-top (right-to-left in first axis)
        result(rows - 1, c) = (*mat)(rows - 1, c);
        Value* acc = m->heap->allocate_scalar((*mat)(rows - 1, c));

        for (int r = rows - 2; r >= 0; --r) {
            Value* elem = m->heap->allocate_scalar((*mat)(r, c));
            fn->dyadic(m, elem, acc);
            Value* new_acc = m->ctrl.value;
            result(r, c) = new_acc->as_scalar();
            // GC will clean up elem and old acc
            acc = new_acc;
        }
        // GC will clean up acc
    }

    m->ctrl.set_value(m->heap->allocate_matrix(result));
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
PrimitiveOp op_reduce = {
    "/",
    fn_reduce,            // Monadic: f/B
    nullptr               // No dyadic form (N-wise reduction not yet implemented)
};

PrimitiveOp op_reduce_first = {
    "⌿",
    fn_reduce_first,      // Monadic: f⌿B
    nullptr               // No dyadic form
};

PrimitiveOp op_scan = {
    "\\",
    fn_scan,              // Monadic: f\B
    nullptr               // No dyadic form
};

PrimitiveOp op_scan_first = {
    "⍀",
    fn_scan_first,        // Monadic: f⍀B
    nullptr               // No dyadic form
};

} // namespace apl
