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
        m->throw_error("AXIS ERROR: outer product does not support axis", m->control, 4, 0);
        return;
    }

    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: outer product requires a function operand", nullptr, 1, 0);
        return;
    }

    // Get dimensions and shapes of both arguments (ISO 9.3.1)
    // Result shape is (⍴A),⍴B
    int lhs_size, lhs_cols;
    int rhs_size, rhs_cols;
    std::vector<int> lhs_shape, rhs_shape;

    if (lhs->is_scalar()) {
        lhs_size = 1;
        lhs_cols = 1;
        // Scalar has empty shape
    } else if (lhs->is_strand()) {
        lhs_size = static_cast<int>(lhs->as_strand()->size());
        lhs_cols = lhs_size;
        lhs_shape.push_back(lhs_size);
    } else if (lhs->is_ndarray()) {
        const Value::NDArrayData* nd = lhs->as_ndarray();
        lhs_size = static_cast<int>(nd->data->size());
        lhs_cols = lhs_size;  // Flat iteration
        lhs_shape = nd->shape;
    } else if (lhs->is_vector()) {
        lhs_size = lhs->rows();
        lhs_cols = 1;
        lhs_shape.push_back(lhs_size);
    } else {
        lhs_size = lhs->rows() * lhs->cols();
        lhs_cols = lhs->cols();
        lhs_shape.push_back(lhs->rows());
        lhs_shape.push_back(lhs->cols());
    }

    if (rhs->is_scalar()) {
        rhs_size = 1;
        rhs_cols = 1;
        // Scalar has empty shape
    } else if (rhs->is_strand()) {
        rhs_size = static_cast<int>(rhs->as_strand()->size());
        rhs_cols = rhs_size;
        rhs_shape.push_back(rhs_size);
    } else if (rhs->is_ndarray()) {
        const Value::NDArrayData* nd = rhs->as_ndarray();
        rhs_size = static_cast<int>(nd->data->size());
        rhs_cols = rhs_size;  // Flat iteration
        rhs_shape = nd->shape;
    } else if (rhs->is_vector()) {
        rhs_size = rhs->rows();
        rhs_cols = 1;
        rhs_shape.push_back(rhs_size);
    } else {
        rhs_size = rhs->rows() * rhs->cols();
        rhs_cols = rhs->cols();
        rhs_shape.push_back(rhs->rows());
        rhs_shape.push_back(rhs->cols());
    }

    // Result shape is (⍴lhs),⍴rhs
    std::vector<int> result_shape;
    result_shape.insert(result_shape.end(), lhs_shape.begin(), lhs_shape.end());
    result_shape.insert(result_shape.end(), rhs_shape.begin(), rhs_shape.end());

    // Use CellIterK with OUTER mode for Cartesian product iteration
    CellIterK* iter = m->heap->allocate<CellIterK>(
        f, lhs, rhs, lhs_size, rhs_size, lhs_cols, rhs_cols);
    if (result_shape.size() > 2) {
        iter->orig_ndarray_shape = result_shape;
    }
    m->push_kont(iter);
}

// ========================================================================
// Inner Product Operator: A f.g B
// ========================================================================
// Result shape: (¯1↓⍴A),1↓⍴B
// For vectors: f/A g B (element-wise g, then reduce with f)
// Example: A +.× B is matrix multiplication

void op_inner_product(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs) {
    if (axis) {
        m->throw_error("AXIS ERROR: inner product does not support axis", m->control, 4, 0);
        return;
    }
    // Validate that both f and g are functions
    if (!f || !f->is_function() || !g || !g->is_function()) {
        m->throw_error("SYNTAX ERROR: inner product requires two function operands", nullptr, 1, 0);
        return;
    }

    // Validate that lhs and rhs are data values (not functions or other non-data values)
    if (!lhs->is_basic_value() || !rhs->is_basic_value()) {
        m->throw_error("DOMAIN ERROR: inner product requires array arguments", nullptr, 11, 0);
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
        m->push_kont(m->heap->allocate_ephemeral<DispatchFunctionK>(g, lhs, rhs));
        return;
    }

    // Handle strands: treat as 1D collections for inner product
    // Strand inner product: f/ (strand1 g strand2) where g is applied element-wise
    bool lhs_is_strand = lhs->is_strand();
    bool rhs_is_strand = rhs->is_strand();

    if (lhs_is_strand || rhs_is_strand) {
        // Get sizes
        int lhs_size = lhs_is_strand ? static_cast<int>(lhs->as_strand()->size())
                                     : (lhs->is_scalar() ? 1 : lhs->size());
        int rhs_size = rhs_is_strand ? static_cast<int>(rhs->as_strand()->size())
                                     : (rhs->is_scalar() ? 1 : rhs->size());

        // Extend scalar-like to match other side
        if (lhs_is_scalar_like && !rhs_is_scalar_like) {
            Value* scalar_val = lhs;
            std::vector<Value*> extended(rhs_size, scalar_val);
            lhs = m->heap->allocate_strand(std::move(extended));
            lhs_is_strand = true;
            lhs_size = rhs_size;
        }
        if (rhs_is_scalar_like && !lhs_is_scalar_like) {
            Value* scalar_val = rhs;
            std::vector<Value*> extended(lhs_size, scalar_val);
            rhs = m->heap->allocate_strand(std::move(extended));
            rhs_is_strand = true;
            rhs_size = lhs_size;
        }

        // Check lengths match
        if (lhs_size != rhs_size) {
            m->throw_error("LENGTH ERROR: inner product dimension mismatch", nullptr, 5, 0);
            return;
        }

        int n = lhs_size;

        // Empty case: return identity element for f
        if (n == 0) {
            double identity = get_identity_for_function(f);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: no identity element for empty inner product", nullptr, 11, 0);
                return;
            }
            m->result = m->heap->allocate_scalar(identity);
            return;
        }

        // Strand inner product: f/ (lhs g¨ rhs) - apply g element-wise, then reduce with f
        m->push_kont(m->heap->allocate_ephemeral<ReduceResultK>(f));
        m->push_kont(m->heap->allocate<CellIterK>(
            g, lhs, rhs, 0, 0, n,
            CellIterMode::COLLECT, n, 1, true, false, true));  // to_strand=true for strand result
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

    // NDARRAY inner product (ISO 9.3.2)
    // Result shape: (¯1↓⍴A),1↓⍴B
    // Last dim of A must equal first dim of B
    if (lhs->is_ndarray() || rhs->is_ndarray()) {
        std::vector<int> lhs_shape, rhs_shape;

        if (lhs->is_ndarray()) {
            lhs_shape = lhs->ndarray_shape();
        } else if (lhs->is_matrix()) {
            lhs_shape = {lhs->rows(), lhs->cols()};
        } else if (lhs->is_vector()) {
            lhs_shape = {lhs->rows()};
        } else {
            lhs_shape = {};  // scalar
        }

        if (rhs->is_ndarray()) {
            rhs_shape = rhs->ndarray_shape();
        } else if (rhs->is_matrix()) {
            rhs_shape = {rhs->rows(), rhs->cols()};
        } else if (rhs->is_vector()) {
            rhs_shape = {rhs->rows()};
        } else {
            rhs_shape = {};  // scalar
        }

        // Check dimension compatibility: last of lhs must equal first of rhs
        int lhs_last = lhs_shape.empty() ? 1 : lhs_shape.back();
        int rhs_first = rhs_shape.empty() ? 1 : rhs_shape.front();
        if (lhs_last != rhs_first) {
            m->throw_error("LENGTH ERROR: inner product dimension mismatch", nullptr, 5, 0);
            return;
        }

        // Result shape: (¯1↓⍴A),1↓⍴B
        std::vector<int> result_shape;
        for (size_t i = 0; i + 1 < lhs_shape.size(); i++) {
            result_shape.push_back(lhs_shape[i]);
        }
        for (size_t i = 1; i < rhs_shape.size(); i++) {
            result_shape.push_back(rhs_shape[i]);
        }

        // Compute frame sizes
        int lhs_frame = 1;
        for (size_t i = 0; i + 1 < lhs_shape.size(); i++) {
            lhs_frame *= lhs_shape[i];
        }
        int rhs_frame = 1;
        for (size_t i = 1; i < rhs_shape.size(); i++) {
            rhs_frame *= rhs_shape[i];
        }
        int common_dim = lhs_last;

        // Handle empty case
        if (lhs_frame == 0 || rhs_frame == 0 || common_dim == 0) {
            double identity = get_identity_for_function(f);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: no identity element for empty inner product", nullptr, 11, 0);
                return;
            }
            if (result_shape.empty()) {
                m->result = m->heap->allocate_scalar(identity);
            } else if (result_shape.size() == 1) {
                Eigen::VectorXd vec(result_shape[0]);
                vec.setConstant(identity);
                m->result = m->heap->allocate_vector(vec);
            } else {
                int total = 1;
                for (int d : result_shape) total *= d;
                Eigen::VectorXd data(total);
                data.setConstant(identity);
                m->result = m->heap->allocate_ndarray(data, result_shape);
            }
            return;
        }

        // Use CellIterK INNER mode
        CellIterK* iter = m->heap->allocate<CellIterK>(
            f, g, lhs, rhs, lhs_frame, rhs_frame, common_dim, 1, 1);
        if (result_shape.size() > 2) {
            iter->orig_ndarray_shape = result_shape;
        } else if (result_shape.size() == 2) {
            // 2D result: set orig_rows/orig_cols from result shape
            iter->orig_rows = result_shape[0];
            iter->orig_cols = result_shape[1];
        } else if (result_shape.size() == 1) {
            // 1D result: vector
            iter->orig_rows = result_shape[0];
            iter->orig_cols = 1;
            iter->orig_is_vector = true;
        }
        m->push_kont(iter);
        return;
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
            m->throw_error("LENGTH ERROR: inner product dimension mismatch", nullptr, 5, 0);
            return;
        }
        int n = lhs_rows;  // Common dimension

        // Empty vectors: return identity element for f
        if (n == 0) {
            double identity = get_identity_for_function(f);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: no identity element for empty inner product", nullptr, 11, 0);
                return;
            }
            m->result = m->heap->allocate_scalar(identity);
            return;
        }

        // Vector inner product: f/ (lhs g rhs)
        // Push ReduceResultK(f), then dyadic CellIterK(g) for element-wise
        m->push_kont(m->heap->allocate_ephemeral<ReduceResultK>(f));
        m->push_kont(m->heap->allocate<CellIterK>(
            g, lhs, rhs, 0, 0, n,
            CellIterMode::COLLECT, n, 1, true));
        return;
    }

    // Vector × Matrix case: vector treated as 1×N row
    if (lhs->is_vector() && rhs->is_matrix()) {
        // Vector length must match matrix rows
        if (lhs_rows != rhs_rows) {
            m->throw_error("LENGTH ERROR: inner product dimension mismatch", nullptr, 5, 0);
            return;
        }
        // Result is a vector of length rhs_cols
        // lhs_frame=1, rhs_frame=rhs_cols, common=lhs_rows
        CellIterK* iter = m->heap->allocate<CellIterK>(
            f, g, lhs, rhs, 1, rhs_cols, lhs_rows, 1, rhs_cols);
        iter->orig_is_vector = true;
        iter->orig_rows = rhs_cols;  // Result is rhs_cols length vector
        iter->orig_cols = 1;
        m->push_kont(iter);
        return;
    }

    // Matrix × Vector case: vector treated as N×1 column
    if (lhs->is_matrix() && rhs->is_vector()) {
        // Matrix cols must match vector length
        if (lhs_cols != rhs_rows) {
            m->throw_error("LENGTH ERROR: inner product dimension mismatch", nullptr, 5, 0);
            return;
        }
        // Result is a vector of length lhs_rows
        // lhs_frame=lhs_rows, rhs_frame=1, common=lhs_cols
        CellIterK* iter = m->heap->allocate<CellIterK>(
            f, g, lhs, rhs, lhs_rows, 1, lhs_cols, lhs_cols, 1);
        iter->orig_is_vector = true;
        iter->orig_rows = lhs_rows;  // Result is lhs_rows length vector
        iter->orig_cols = 1;
        m->push_kont(iter);
        return;
    }

    // General case: matrix × matrix inner product
    // LENGTH constraint: last dimension of A must equal first dimension of B
    if (lhs_cols != rhs_rows) {
        m->throw_error("LENGTH ERROR: inner product dimension mismatch", nullptr, 5, 0);
        return;
    }

    // Use CellIterK INNER mode for matrix case
    // lhs_frame=lhs_rows, rhs_frame=rhs_cols, common=lhs_cols
    m->push_kont(m->heap->allocate<CellIterK>(
        f, g, lhs, rhs, lhs_rows, rhs_cols, lhs_cols, lhs_cols, rhs_cols));
}

// ========================================================================
// Each Operator: f¨B (monadic) or A f¨B (dyadic)
// ========================================================================
// Applies function to each element independently
// Result has same shape as argument(s)
// Uses CellIterK COLLECT mode for continuation-based execution

void op_each(Machine* m, Value* axis, Value* f, Value* omega) {
    if (axis) {
        m->throw_error("AXIS ERROR: each operator does not support axis", m->control, 4, 0);
        return;
    }
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: each operator requires a function operand", nullptr, 1, 0);
        return;
    }

    if (omega->is_scalar()) {
        apply_function_immediate(m, f, nullptr, omega);
        return;
    }

    // Strand: apply function to each strand element
    if (omega->is_strand()) {
        int num_cells = static_cast<int>(omega->as_strand()->size());
        m->push_kont(m->heap->allocate<CellIterK>(
            f, nullptr, omega, 0, 0, num_cells,
            CellIterMode::COLLECT, num_cells, 1, true, false, true));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    // NDARRAY: apply function to each element, preserve shape
    if (omega->is_ndarray()) {
        const Value::NDArrayData* nd = omega->as_ndarray();
        int num_cells = static_cast<int>(nd->data->size());
        CellIterK* iter = m->heap->allocate<CellIterK>(
            f, nullptr, omega, 0, 0, num_cells,
            CellIterMode::COLLECT, 1, num_cells, false, false, false);
        iter->orig_ndarray_shape = nd->shape;
        m->push_kont(iter);
        return;
    }

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
        m->throw_error("AXIS ERROR: each operator does not support axis", m->control, 4, 0);
        return;
    }
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: each operator requires a function operand", nullptr, 1, 0);
        return;
    }

    // Both scalars: just apply function
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->push_kont(m->heap->allocate_ephemeral<DispatchFunctionK>(f, lhs, rhs));
        return;
    }

    // Scalar extension: scalar with array/strand/ndarray
    if (lhs->is_scalar()) {
        if (rhs->is_strand()) {
            int num_cells = static_cast<int>(rhs->as_strand()->size());
            m->push_kont(m->heap->allocate<CellIterK>(
                f, lhs, rhs, 0, 0, num_cells,
                CellIterMode::COLLECT, num_cells, 1, true, false, true));
            return;
        }
        if (rhs->is_ndarray()) {
            const Value::NDArrayData* nd = rhs->as_ndarray();
            int num_cells = static_cast<int>(nd->data->size());
            CellIterK* iter = m->heap->allocate<CellIterK>(
                f, lhs, rhs, 0, 0, num_cells,
                CellIterMode::COLLECT, 1, num_cells, false, false, false);
            iter->orig_ndarray_shape = nd->shape;
            m->push_kont(iter);
            return;
        }
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
        if (lhs->is_strand()) {
            int num_cells = static_cast<int>(lhs->as_strand()->size());
            m->push_kont(m->heap->allocate<CellIterK>(
                f, lhs, rhs, 0, 0, num_cells,
                CellIterMode::COLLECT, num_cells, 1, true, false, true));
            return;
        }
        if (lhs->is_ndarray()) {
            const Value::NDArrayData* nd = lhs->as_ndarray();
            int num_cells = static_cast<int>(nd->data->size());
            CellIterK* iter = m->heap->allocate<CellIterK>(
                f, lhs, rhs, 0, 0, num_cells,
                CellIterMode::COLLECT, 1, num_cells, false, false, false);
            iter->orig_ndarray_shape = nd->shape;
            m->push_kont(iter);
            return;
        }
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

    // Both NDARRAY: shapes must match
    if (lhs->is_ndarray() && rhs->is_ndarray()) {
        const Value::NDArrayData* lnd = lhs->as_ndarray();
        const Value::NDArrayData* rnd = rhs->as_ndarray();
        if (lnd->shape != rnd->shape) {
            m->throw_error("LENGTH ERROR: each requires matching shapes", nullptr, 5, 0);
            return;
        }
        int num_cells = static_cast<int>(lnd->data->size());
        CellIterK* iter = m->heap->allocate<CellIterK>(
            f, lhs, rhs, 0, 0, num_cells,
            CellIterMode::COLLECT, 1, num_cells, false, false, false);
        iter->orig_ndarray_shape = lnd->shape;
        m->push_kont(iter);
        return;
    }

    // NDARRAY with non-NDARRAY (except scalar handled above): RANK ERROR
    if (lhs->is_ndarray() || rhs->is_ndarray()) {
        m->throw_error("RANK ERROR: each requires matching ranks", nullptr, 4, 0);
        return;
    }

    // Both strands: sizes must match
    if (lhs->is_strand() && rhs->is_strand()) {
        int lhs_size = static_cast<int>(lhs->as_strand()->size());
        int rhs_size = static_cast<int>(rhs->as_strand()->size());
        if (lhs_size != rhs_size) {
            m->throw_error("LENGTH ERROR: each requires matching shapes or scalar", nullptr, 5, 0);
            return;
        }
        m->push_kont(m->heap->allocate<CellIterK>(
            f, lhs, rhs, 0, 0, lhs_size,
            CellIterMode::COLLECT, lhs_size, 1, true, false, true));
        return;
    }

    // Mixed strand/vector: treat as parallel iteration if sizes match (ISO 9.2.6)
    // Vector elements pair with strand elements
    if (lhs->is_strand() && rhs->is_vector()) {
        int strand_size = static_cast<int>(lhs->as_strand()->size());
        int vec_size = rhs->rows();
        if (strand_size != vec_size) {
            m->throw_error("LENGTH ERROR: each requires matching shapes", nullptr, 5, 0);
            return;
        }
        // Result is strand: strand[i] f vec[i]
        m->push_kont(m->heap->allocate<CellIterK>(
            f, lhs, rhs, 0, 0, strand_size,
            CellIterMode::COLLECT, strand_size, 1, true, false, true));
        return;
    }

    if (lhs->is_vector() && rhs->is_strand()) {
        int vec_size = lhs->rows();
        int strand_size = static_cast<int>(rhs->as_strand()->size());
        if (vec_size != strand_size) {
            m->throw_error("LENGTH ERROR: each requires matching shapes", nullptr, 5, 0);
            return;
        }
        // Result is strand: vec[i] f strand[i]
        m->push_kont(m->heap->allocate<CellIterK>(
            f, lhs, rhs, 0, 0, strand_size,
            CellIterMode::COLLECT, strand_size, 1, true, false, true));
        return;
    }

    // Other mixed strand/array - error
    if (lhs->is_strand() || rhs->is_strand()) {
        m->throw_error("RANK ERROR: each requires matching types", nullptr, 4, 0);
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
        m->throw_error("LENGTH ERROR: each requires matching shapes or scalar", nullptr, 5, 0);
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
        m->throw_error("AXIS ERROR: commute operator does not support axis", m->control, 4, 0);
        return;
    }
    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: duplicate operator requires a function operand", nullptr, 1, 0);
        return;
    }

    // Use DispatchFunctionK for all function types to get proper pervasive dispatch
    // This handles primitives, curried functions, derived operators, and closures
    m->push_kont(m->heap->allocate_ephemeral<DispatchFunctionK>(f, omega, omega));
}

// ========================================================================
// Commute Operator: A f⍨B (dyadic)
// ========================================================================
// Commute: swaps left and right arguments
// A f⍨B → B f A

void op_commute_dyadic(Machine* m, Value* axis, Value* lhs, Value* f, Value* g, Value* rhs) {
    (void)g;  // Commute only uses one function operand
    if (axis) {
        m->throw_error("AXIS ERROR: commute operator does not support axis", m->control, 4, 0);
        return;
    }
    // Validate that f is a function
    if (!f || !f->is_function()) {
        m->throw_error("SYNTAX ERROR: commute operator requires a function operand", nullptr, 1, 0);
        return;
    }

    // Use DispatchFunctionK for all function types to get proper pervasive dispatch
    // This handles primitives, curried functions, derived operators, closures,
    // and pervasive operations on strands and NDARRAYs
    // Commute swaps: A f⍨ B → B f A
    m->push_kont(m->heap->allocate_ephemeral<DispatchFunctionK>(f, rhs, lhs));
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
        m->throw_error("AXIS ERROR: axis must be scalar or one-element vector", m->control, 4, 0);
        return -1;
    }

    // ISO 13751 §5.2.5: axis must be a "near-integer"
    if (std::abs(axis_val - std::round(axis_val)) >= INTEGER_TOLERANCE) {
        m->throw_error("AXIS ERROR: axis must be an integer", m->control, 4, 0);
        return -1;
    }

    int k = static_cast<int>(std::round(axis_val));
    int io = m->io;

    if (k < io || k > max_rank - 1 + io) {
        m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
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
        m->throw_error("DOMAIN ERROR: reduce requires a function", nullptr, 11, 0);
        return;
    }

    if (omega->is_scalar()) {
        // Reducing a scalar is identity
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // Strand reduction: reduce over strand elements
    if (omega->is_strand()) {
        const std::vector<Value*>* strand = omega->as_strand();
        int len = static_cast<int>(strand->size());
        if (len == 0) {
            // Empty strand: return identity element
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction", nullptr, 11, 0);
                return;
            }
            m->result = m->heap->allocate_scalar(identity);
            return;
        }
        if (len == 1) {
            m->result = (*strand)[0];
            return;
        }
        // Use CellIterK FOLD_RIGHT to reduce strand elements
        m->push_kont(m->heap->allocate<CellIterK>(
            func, nullptr, omega, 0, 0, len,
            CellIterMode::FOLD_RIGHT, len, 1, true, false, true));
        return;
    }

    // String → char vector conversion for array operations
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    // NDARRAY reduction: reduce along specified axis (default: last)
    if (omega->is_ndarray()) {
        const Value::NDArrayData* nd = omega->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Determine axis (default: last axis in ⎕IO terms)
        int k = rank - 1 + m->io;  // Last axis
        if (axis) {
            k = validate_axis(m, axis, rank);
            if (k < 0) return;
        }
        int ax = k - m->io;  // Convert to 0-indexed

        int ax_len = shape[ax];
        if (ax_len == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction", nullptr, 11, 0);
                return;
            }
            // Result shape: remove the axis dimension
            std::vector<int> result_shape;
            for (int d = 0; d < rank; ++d) {
                if (d != ax) result_shape.push_back(shape[d]);
            }
            int result_size = 1;
            for (int s : result_shape) result_size *= s;
            Eigen::VectorXd result_data = Eigen::VectorXd::Constant(result_size, identity);
            if (result_shape.size() <= 2) {
                if (result_shape.size() == 1) {
                    m->result = m->heap->allocate_vector(result_data);
                } else {
                    Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
                    for (int i = 0; i < result_size; ++i) mat.data()[i] = result_data(i);
                    m->result = m->heap->allocate_matrix(mat);
                }
            } else {
                m->result = m->heap->allocate_ndarray(result_data, result_shape);
            }
            return;
        }

        if (ax_len == 1) {
            // Single element along axis: just remove that dimension
            std::vector<int> result_shape;
            for (int d = 0; d < rank; ++d) {
                if (d != ax) result_shape.push_back(shape[d]);
            }
            if (result_shape.size() <= 2) {
                if (result_shape.size() == 1) {
                    m->result = m->heap->allocate_vector(*nd->data);
                } else {
                    Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
                    for (int i = 0; i < nd->data->size(); ++i) mat.data()[i] = (*nd->data)(i);
                    m->result = m->heap->allocate_matrix(mat);
                }
            } else {
                m->result = m->heap->allocate_ndarray(*nd->data, result_shape);
            }
            return;
        }

        // General case: reduce along axis ax
        // Result shape: remove dimension ax
        std::vector<int> result_shape;
        for (int d = 0; d < rank; ++d) {
            if (d != ax) result_shape.push_back(shape[d]);
        }

        int result_size = 1;
        for (int s : result_shape) result_size *= s;

        // Compute strides for source
        std::vector<int> src_strides(rank);
        src_strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d) {
            src_strides[d] = src_strides[d + 1] * shape[d + 1];
        }

        // For each position in result, reduce fiber along axis
        m->push_kont(m->heap->allocate<FiberReduceK>(func, omega, ax, 0, false));
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: / requires array argument", nullptr, 11, 0);
        return;
    }
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
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction", nullptr, 11, 0);
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
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction", nullptr, 11, 0);
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
        m->push_kont(m->heap->allocate<FiberReduceK>(func, omega, 0, 0, false));
    } else {
        // k == 2: Reduce along last axis (columns)
        if (cols == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction", nullptr, 11, 0);
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
        m->push_kont(m->heap->allocate<FiberReduceK>(func, omega, 1, 0, false));
    }
}

// Reduce-first (⌿) - reduce along first axis (rows)
// Uses CellIterK FOLD_RIGHT for continuation-based execution
// Unified implementation handles both f⌿B and f⌿[k]B
void fn_reduce_first(Machine* m, Value* axis, Value* func, Value* omega) {
    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: reduce-first requires a function", nullptr, 11, 0);
        return;
    }

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // Strand reduction: same as / (single axis)
    if (omega->is_strand()) {
        fn_reduce(m, axis, func, omega);
        return;
    }

    // String → char vector conversion
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    // NDARRAY reduction: reduce along first axis (default)
    if (omega->is_ndarray()) {
        // Delegate to fn_reduce - if no axis given, use first axis (⎕IO)
        Value* axis_val = axis ? axis : m->heap->allocate_scalar(m->io);
        fn_reduce(m, axis_val, func, omega);
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌿ requires array argument", nullptr, 11, 0);
        return;
    }
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
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction", nullptr, 11, 0);
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
        m->push_kont(m->heap->allocate<FiberReduceK>(func, omega, 0, 0, false));
    } else {
        // k == 2: Reduce along last axis (columns)
        if (cols == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element for empty reduction", nullptr, 11, 0);
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
        m->push_kont(m->heap->allocate<FiberReduceK>(func, omega, 1, 0, false));
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
        m->throw_error("DOMAIN ERROR: scan requires a function", nullptr, 11, 0);
        return;
    }

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // Strand scan: scan over strand elements
    if (omega->is_strand()) {
        const std::vector<Value*>* strand = omega->as_strand();
        int len = static_cast<int>(strand->size());
        if (len == 0) {
            // Empty strand: return empty strand
            m->result = m->heap->allocate_strand({});
            return;
        }
        if (len == 1) {
            // Single element: result is strand containing just that element
            m->result = omega;
            return;
        }
        // Use CellIterK SCAN_LEFT for left-to-right scan over strand elements
        m->push_kont(m->heap->allocate<CellIterK>(
            func, nullptr, omega, 0, 0, len,
            CellIterMode::SCAN_LEFT, len, 1, true, false, true));
        return;
    }

    // String → char vector conversion
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    // NDARRAY scan: scan along specified axis (default: last)
    if (omega->is_ndarray()) {
        const Value::NDArrayData* nd = omega->as_ndarray();
        const std::vector<int>& shape = nd->shape;
        int rank = static_cast<int>(shape.size());

        // Determine axis (default: last axis in ⎕IO terms)
        int k = rank - 1 + m->io;  // Last axis
        if (axis) {
            k = validate_axis(m, axis, rank);
            if (k < 0) return;
        }
        int ax = k - m->io;  // Convert to 0-indexed

        int ax_len = shape[ax];
        if (ax_len <= 1) {
            // Single element or empty along axis: return unchanged
            m->result = omega;
            return;
        }

        // Result has same shape as input
        // total_positions = product of all dimensions except scan axis
        int total_positions = 1;
        for (int d = 0; d < rank; d++) {
            if (d != ax) total_positions *= shape[d];
        }

        m->push_kont(m->heap->allocate<RowScanK>(
            func, omega, ax, shape, total_positions));
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: \\ requires array argument", nullptr, 11, 0);
        return;
    }
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
            m->throw_error("SYNTAX ERROR: expand does not support axis", nullptr, 1, 0);
            return;
        }
        fn_expand_first(m, nullptr, func, omega);
        return;
    }

    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: scan-first requires a function", nullptr, 11, 0);
        return;
    }

    if (omega->is_scalar()) {
        m->result = m->heap->allocate_scalar(omega->as_scalar());
        return;
    }

    // Strand scan: same as \ (single axis)
    if (omega->is_strand()) {
        fn_scan(m, axis, func, omega);
        return;
    }

    // String → char vector conversion
    if (omega->is_string()) omega = omega->to_char_vector(m->heap);

    // NDARRAY scan: scan along first axis (default)
    if (omega->is_ndarray()) {
        // Delegate to fn_scan - if no axis given, use first axis (⎕IO)
        Value* axis_val = axis ? axis : m->heap->allocate_scalar(m->io);
        fn_scan(m, axis_val, func, omega);
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ⍀ requires array argument", nullptr, 11, 0);
        return;
    }
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
        m->throw_error("RANK ERROR: N must be a scalar", nullptr, 4, 0);
        return INT_MIN;
    }

    double n_double = n_val->as_scalar();
    int n = static_cast<int>(n_double);

    if (n_double != static_cast<double>(n)) {
        m->throw_error("DOMAIN ERROR: N must be an integer", nullptr, 11, 0);
        return INT_MIN;
    }

    // Handle negative N (reverse before reduce) - take absolute value
    int abs_n = n < 0 ? -n : n;

    if (abs_n > axis_len + 1) {
        m->throw_error("DOMAIN ERROR: N too large for array", nullptr, 11, 0);
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
        m->throw_error("SYNTAX ERROR: replicate does not support N-wise", nullptr, 1, 0);
        return;
    }

    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: reduce requires a function", nullptr, 11, 0);
        return;
    }

    if (rhs->is_scalar()) {
        // Scalar handling for N-wise
        int n = validate_nwise(m, n_val, 1);
        if (n == INT_MIN) return;
        int abs_n = n < 0 ? -n : n;
        if (abs_n > 2) {
            m->throw_error("DOMAIN ERROR: N too large for scalar", nullptr, 11, 0);
            return;
        }
        if (abs_n == 0) {
            // 0 f/ scalar → 2-element result with identity
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element", nullptr, 11, 0);
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

    // Handle strands: treat as 1D collection
    if (rhs->is_strand()) {
        auto* strand = rhs->as_strand();
        int len = static_cast<int>(strand->size());
        int n = validate_nwise(m, n_val, len);
        if (n == INT_MIN) return;
        int abs_n = n < 0 ? -n : n;
        bool reverse = n < 0;

        if (abs_n == 0) {
            // 0 f/ strand → (1+len) copies of identity
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element", nullptr, 11, 0);
                return;
            }
            Eigen::VectorXd result = Eigen::VectorXd::Constant(len + 1, identity);
            m->result = m->heap->allocate_vector(result);
            return;
        }

        // Result length = len - abs_n + 1
        int result_len = len - abs_n + 1;
        if (result_len < 0) {
            m->throw_error("DOMAIN ERROR: N too large for strand", nullptr, 11, 0);
            return;
        }
        if (result_len == 0) {
            m->result = m->heap->allocate_strand(std::vector<Value*>());
            return;
        }

        // Use FiberReduceK for N-wise reduction on strands
        m->push_kont(m->heap->allocate<FiberReduceK>(func, rhs, 0, abs_n, reverse));
        return;
    }

    // Handle NDARRAY: N-wise reduction along specified axis (default: last)
    if (rhs->is_ndarray()) {
        const Value::NDArrayData* nd = rhs->as_ndarray();
        int rank = static_cast<int>(nd->shape.size());

        // Determine axis (default: last axis in ⎕IO terms)
        int k = rank - 1 + m->io;  // Last axis
        if (axis) {
            k = validate_axis(m, axis, rank);
            if (k < 0) return;
        }
        int ax = k - m->io;  // Convert to 0-indexed

        int axis_len = nd->shape[ax];
        int n = validate_nwise(m, n_val, axis_len);
        if (n == INT_MIN) return;
        int abs_n = n < 0 ? -n : n;
        bool is_reverse = n < 0;

        if (abs_n == 0) {
            // 0 f/[k] NDARRAY → expand axis by 1, fill with identity
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element", nullptr, 11, 0);
                return;
            }
            std::vector<int> result_shape = nd->shape;
            result_shape[ax] = axis_len + 1;
            int total_size = 1;
            for (int d : result_shape) total_size *= d;
            Eigen::VectorXd result_data = Eigen::VectorXd::Constant(total_size, identity);
            m->result = m->heap->allocate_ndarray(result_data, result_shape);
            return;
        }

        // Result shape: same as input but axis dimension changes
        int result_axis_len = axis_len - abs_n + 1;
        if (result_axis_len <= 0) {
            m->throw_error("DOMAIN ERROR: N too large for array", nullptr, 11, 0);
            return;
        }

        // Use FiberReduceK for N-wise reduction on NDARRAY
        m->push_kont(m->heap->allocate<FiberReduceK>(func, rhs, ax, abs_n, is_reverse));
        return;
    }

    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: / requires array argument", nullptr, 11, 0);
        return;
    }

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
                m->throw_error("DOMAIN ERROR: function has no identity element", nullptr, 11, 0);
                return;
            }
            Eigen::VectorXd result = Eigen::VectorXd::Constant(len + 1, identity);
            m->result = m->heap->allocate_vector(result);
            return;
        }

        // Result length = len - abs_n + 1
        int result_len = len - abs_n + 1;
        if (result_len < 0) {
            m->throw_error("DOMAIN ERROR: N too large for vector", nullptr, 11, 0);
            return;
        }
        if (result_len == 0) {
            Eigen::VectorXd empty(0);
            m->result = m->heap->allocate_vector(empty);
            return;
        }

        // Use FiberReduceK for N-wise reduction on vectors
        m->push_kont(m->heap->allocate<FiberReduceK>(func, rhs, 0, abs_n, reverse));
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
            m->throw_error("DOMAIN ERROR: function has no identity element", nullptr, 11, 0);
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

    // N-wise on matrix along axis k (k=1 → axis 0, k=2 → axis 1)
    m->push_kont(m->heap->allocate<FiberReduceK>(func, rhs, k == 1 ? 0 : 1, abs_n, reverse));
}

// N-wise reduction-first: N f⌿ B or N f⌿[k] B
// Default axis is first (1) instead of last (2)
void fn_reduce_first_nwise(Machine* m, Value* axis, Value* lhs, Value* func, Value* g, Value* rhs) {
    (void)g;  // Reduce only uses one function operand
    Value* n_val = lhs;

    if (!func->is_function()) {
        m->throw_error("DOMAIN ERROR: reduce requires a function", nullptr, 11, 0);
        return;
    }

    if (rhs->is_scalar()) {
        // Scalar handling for N-wise (same as fn_reduce_nwise)
        int n = validate_nwise(m, n_val, 1);
        if (n == INT_MIN) return;
        int abs_n = n < 0 ? -n : n;
        if (abs_n > 2) {
            m->throw_error("DOMAIN ERROR: N too large for scalar", nullptr, 11, 0);
            return;
        }
        if (abs_n == 0) {
            double identity = get_identity_for_function(func);
            if (std::isnan(identity)) {
                m->throw_error("DOMAIN ERROR: function has no identity element", nullptr, 11, 0);
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

    if (!rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ⌿ requires array argument", nullptr, 11, 0);
        return;
    }

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
            m->throw_error("DOMAIN ERROR: function has no identity element", nullptr, 11, 0);
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

    // N-wise on matrix along axis k (k=1 → axis 0, k=2 → axis 1)
    m->push_kont(m->heap->allocate<FiberReduceK>(func, rhs, k == 1 ? 0 : 1, abs_n, reverse));
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

// Helper: get the rank of a value (0=scalar, 1=vector/strand, 2=matrix)
static int get_array_rank(Value* v) {
    if (v->is_scalar()) return 0;
    if (v->is_vector()) return 1;
    if (v->is_strand()) return 1;  // Strands are 1D collections
    if (v->is_ndarray()) return static_cast<int>(v->ndarray_shape().size());
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
// For strands: 0-cell = individual element, 1-cell = whole strand
// For NDARRAY: k-cell is a subarray with last k dimensions
static Value* extract_cell(Machine* m, Value* arr, int k, int cell_index) {
    if (k >= get_array_rank(arr)) {
        // Full rank: return whole array
        return arr;
    }

    if (arr->is_scalar()) {
        return arr;
    }

    // Handle strands: 0-cells are individual elements
    if (arr->is_strand()) {
        auto* strand = arr->as_strand();
        if (k == 0) {
            if (cell_index >= static_cast<int>(strand->size())) return nullptr;
            return (*strand)[cell_index];
        }
        // k >= 1: whole strand
        return arr;
    }

    // Handle NDARRAY
    if (arr->is_ndarray()) {
        const std::vector<int>& shape = arr->ndarray_shape();
        const Eigen::VectorXd* data = arr->ndarray_data();
        int rank = static_cast<int>(shape.size());

        if (k == 0) {
            // 0-cell: individual scalar
            return m->heap->allocate_scalar((*data)(cell_index));
        }

        // k-cell: subarray with last k dimensions
        // Frame dimensions are first (rank - k) dimensions
        int frame_rank = rank - k;

        // Compute cell shape (last k dimensions)
        std::vector<int> cell_shape(shape.begin() + frame_rank, shape.end());
        int cell_size = 1;
        for (int d : cell_shape) cell_size *= d;

        // Extract cell data starting at cell_index * cell_size
        int start = cell_index * cell_size;
        Eigen::VectorXd cell_data = data->segment(start, cell_size);

        // Return appropriate type based on cell rank
        if (k == 1) {
            return m->heap->allocate_vector(cell_data);
        } else if (k == 2) {
            Eigen::MatrixXd mat(cell_shape[0], cell_shape[1]);
            for (int i = 0; i < cell_shape[0]; i++) {
                for (int j = 0; j < cell_shape[1]; j++) {
                    mat(i, j) = cell_data(i * cell_shape[1] + j);
                }
            }
            return m->heap->allocate_matrix(mat);
        } else {
            // Higher rank cell - return as NDARRAY
            return m->heap->allocate_ndarray(cell_data, cell_shape);
        }
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
// For NDARRAY: number of k-cells = product of first (rank - k) dimensions (the frame)
static int count_cells(Value* arr, int k) {
    if (k >= get_array_rank(arr)) return 1;

    if (arr->is_scalar()) return 1;

    // Handle strands: 0-cells are individual elements, 1-cell is whole strand
    if (arr->is_strand()) {
        return (k == 0) ? static_cast<int>(arr->as_strand()->size()) : 1;
    }

    // Handle NDARRAY
    if (arr->is_ndarray()) {
        const std::vector<int>& shape = arr->ndarray_shape();
        int rank = static_cast<int>(shape.size());
        int frame_rank = rank - k;  // Number of frame dimensions

        int count = 1;
        for (int i = 0; i < frame_rank; i++) {
            count *= shape[i];
        }
        return count;
    }

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
    m->throw_error("SYNTAX ERROR: rank operator requires rank specification", nullptr, 1, 0);
}

// Dyadic rank: A f⍤k B (or monadic f⍤k B where lhs is nullptr)
void op_rank(Machine* m, Value* axis, Value* lhs, Value* f, Value* rank_spec, Value* rhs) {
    if (axis) {
        m->throw_error("AXIS ERROR: rank operator does not support axis", m->control, 4, 0);
        return;
    }
    // Validate function operand
    if (!f || !f->is_function()) {
        m->throw_error("DOMAIN ERROR: rank operator requires function operand", nullptr, 11, 0);
        return;
    }

    // Validate rank specification
    if (!rank_spec) {
        m->throw_error("DOMAIN ERROR: rank operator requires rank specification", nullptr, 11, 0);
        return;
    }

    // Parse rank specification
    int rhs_rank = get_array_rank(rhs);
    int lhs_rank = lhs ? get_array_rank(lhs) : 0;
    int monadic_r, left_r, right_r;

    if (!parse_rank_spec(rank_spec, std::max(lhs_rank, rhs_rank), &monadic_r, &left_r, &right_r)) {
        m->throw_error("DOMAIN ERROR: invalid rank specification", nullptr, 11, 0);
        return;
    }

    // Validate arguments are arrays (not functions)
    if (!rhs->is_scalar() && !rhs->is_array() && !rhs->is_strand()) {
        m->throw_error("DOMAIN ERROR: ⍤ requires array argument", nullptr, 11, 0);
        return;
    }
    if (lhs && !lhs->is_scalar() && !lhs->is_array() && !lhs->is_strand()) {
        m->throw_error("DOMAIN ERROR: ⍤ requires array argument", nullptr, 11, 0);
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
            if (rhs->is_ndarray()) {
                // NDARRAY: return empty with frame shape
                const std::vector<int>& shape = rhs->ndarray_shape();
                std::vector<int> frame_shape(shape.begin(), shape.begin() + (rhs_rank - k));
                Eigen::VectorXd empty_data(0);
                m->result = m->heap->allocate_ndarray(empty_data, frame_shape);
            } else if (rhs->is_vector()) {
                m->result = m->heap->allocate_vector(Eigen::VectorXd(0));
            } else {
                m->result = m->heap->allocate_matrix(Eigen::MatrixXd(rhs->rows(), rhs->cols()));
            }
            return;
        }

        if (num_cells == 1) {
            // Single cell: just apply f to whole array
            apply_function_immediate(m, f, nullptr, rhs);
            return;
        }

        // Multiple cells: use CellIterK continuation to iterate
        bool is_strand = rhs->is_strand();
        bool is_ndarray = rhs->is_ndarray();

        if (is_ndarray) {
            // NDARRAY: frame is first (rank - k) dimensions
            const std::vector<int>& shape = rhs->ndarray_shape();
            int frame_rank = rhs_rank - k;
            std::vector<int> frame_shape(shape.begin(), shape.begin() + frame_rank);

            CellIterK* iter = m->heap->allocate<CellIterK>(
                f, nullptr, rhs, k, k, num_cells,
                CellIterMode::COLLECT, num_cells, 1, false, false, false);

            // Only use NDARRAY result for frame rank > 2
            if (frame_rank > 2) {
                iter->orig_ndarray_shape = frame_shape;
            } else if (frame_rank == 2) {
                // 2D frame: use matrix assembly
                iter->orig_rows = frame_shape[0];
                iter->orig_cols = frame_shape[1];
            } else if (frame_rank == 1) {
                // 1D frame: use vector assembly
                iter->orig_is_vector = true;
                iter->orig_rows = frame_shape[0];
                iter->orig_cols = 1;
            }
            // frame_rank == 0 means single cell, already handled above

            m->push_kont(iter);
        } else {
            // Non-NDARRAY case: use frame shape (not full array shape)
            // Frame is the first (rank - k) dimensions
            int frame_rank = rhs_rank - k;

            CellIterK* iter = m->heap->allocate<CellIterK>(
                f, nullptr, rhs, k, k, num_cells,
                CellIterMode::COLLECT, num_cells, 1, false, false, is_strand);

            if (frame_rank >= 2 && rhs->is_matrix()) {
                // 2D frame: use matrix assembly
                iter->orig_rows = rhs->rows();
                iter->orig_cols = rhs->cols();
            } else if (frame_rank == 1) {
                // 1D frame: use vector assembly
                iter->orig_is_vector = true;
                iter->orig_rows = is_strand ? static_cast<int>(rhs->as_strand()->size()) : rhs->rows();
                iter->orig_cols = 1;
            }
            // frame_rank == 0 means single cell, already handled above

            m->push_kont(iter);
        }
    } else {
        // Dyadic: A f⍤k B (ISO 9.3.5)
        int lk = std::min(left_r, lhs_rank);
        int rk = std::min(right_r, rhs_rank);
        if (lk < 0) lk = std::max(0, lhs_rank + lk);
        if (rk < 0) rk = std::max(0, rhs_rank + rk);

        // Compute frame shapes (ISO 9.3.5: y10 and y11)
        // y10 = shape of A with last lk items removed
        // y11 = shape of B with last rk items removed
        std::vector<int> left_frame, right_frame;
        int left_frame_rank = lhs_rank - lk;
        int right_frame_rank = rhs_rank - rk;

        if (lhs->is_ndarray()) {
            const std::vector<int>& shape = lhs->ndarray_shape();
            left_frame = std::vector<int>(shape.begin(), shape.begin() + left_frame_rank);
        } else if (lhs->is_matrix()) {
            if (left_frame_rank >= 2) left_frame = {lhs->rows(), lhs->cols()};
            else if (left_frame_rank == 1) left_frame = {lhs->rows()};
        } else if (lhs->is_vector()) {
            if (left_frame_rank >= 1) left_frame = {lhs->rows()};
        }

        if (rhs->is_ndarray()) {
            const std::vector<int>& shape = rhs->ndarray_shape();
            right_frame = std::vector<int>(shape.begin(), shape.begin() + right_frame_rank);
        } else if (rhs->is_matrix()) {
            if (right_frame_rank >= 2) right_frame = {rhs->rows(), rhs->cols()};
            else if (right_frame_rank == 1) right_frame = {rhs->rows()};
        } else if (rhs->is_vector()) {
            if (right_frame_rank >= 1) right_frame = {rhs->rows()};
        }

        // ISO 9.3.5: Check frame compatibility
        bool left_empty = left_frame.empty();
        bool right_empty = right_frame.empty();

        if (!left_empty && !right_empty) {
            // Both frames nonempty: must match
            if (left_frame.size() != right_frame.size()) {
                m->throw_error("RANK ERROR: frame ranks differ in rank operator", nullptr, 4, 0);
                return;
            }
            if (left_frame != right_frame) {
                m->throw_error("LENGTH ERROR: frame shapes differ in rank operator", nullptr, 5, 0);
                return;
            }
        }

        int left_cells = count_cells(lhs, lk);
        int right_cells = count_cells(rhs, rk);
        int num_cells = std::max(left_cells, right_cells);

        if (num_cells == 1) {
            // Single cell each: just apply f dyadically
            m->push_kont(m->heap->allocate_ephemeral<DispatchFunctionK>(f, lhs, rhs));
            return;
        }

        // Multiple cells: use CellIterK continuation
        bool is_strand = rhs->is_strand();
        bool rhs_ndarray = rhs->is_ndarray();
        bool lhs_ndarray = lhs->is_ndarray();

        // For result shape, use the non-scalar operand's frame
        // If both are NDARRAY, they must have same frame (checked above via cell counts)
        if (rhs_ndarray || lhs_ndarray) {
            std::vector<int> frame_shape;
            int frame_rank = 0;

            if (rhs_ndarray && right_cells > 1) {
                const std::vector<int>& shape = rhs->ndarray_shape();
                frame_rank = rhs_rank - rk;
                frame_shape = std::vector<int>(shape.begin(), shape.begin() + frame_rank);
            } else if (lhs_ndarray && left_cells > 1) {
                const std::vector<int>& shape = lhs->ndarray_shape();
                frame_rank = lhs_rank - lk;
                frame_shape = std::vector<int>(shape.begin(), shape.begin() + frame_rank);
            }

            CellIterK* iter = m->heap->allocate<CellIterK>(
                f, lhs, rhs, lk, rk, num_cells,
                CellIterMode::COLLECT, num_cells, 1, false, false, false);

            // Only use NDARRAY result for frame rank > 2
            if (frame_rank > 2) {
                iter->orig_ndarray_shape = frame_shape;
            } else if (frame_rank == 2 && !frame_shape.empty()) {
                // 2D frame: use matrix assembly
                iter->orig_rows = frame_shape[0];
                iter->orig_cols = frame_shape[1];
            } else if (frame_rank == 1 && !frame_shape.empty()) {
                // 1D frame: use vector assembly
                iter->orig_is_vector = true;
                iter->orig_rows = frame_shape[0];
                iter->orig_cols = 1;
            }

            m->push_kont(iter);
        } else {
            // Non-NDARRAY case: use frame shape (not full array shape)
            // The frame is the shape used for result assembly
            std::vector<int>& result_frame = right_cells > 1 ? right_frame : left_frame;

            CellIterK* iter = m->heap->allocate<CellIterK>(
                f, lhs, rhs, lk, rk, num_cells,
                CellIterMode::COLLECT, num_cells, 1, false, false, is_strand);

            if (result_frame.size() >= 2) {
                // 2D frame: use matrix assembly
                iter->orig_rows = result_frame[0];
                iter->orig_cols = result_frame[1];
            } else if (result_frame.size() == 1) {
                // 1D frame: use vector assembly
                iter->orig_is_vector = true;
                iter->orig_rows = result_frame[0];
                iter->orig_cols = 1;
            }
            // Empty frame means single cell, already handled above

            m->push_kont(iter);
        }
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
        m->throw_error("AXIS ERROR: catenate-axis does not support additional axis modifier", m->control, 4, 0);
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
        m->throw_error("AXIS ERROR: axis must be scalar or one-element vector", m->control, 4, 0);
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
            m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
            return;
        }
        m->result = m->heap->allocate_vector(omega->as_matrix()->col(0), omega->is_char_data());
        return;
    }

    if (!omega->is_array()) {
        m->throw_error("DOMAIN ERROR: ,[k] requires array argument", nullptr, 11, 0);
        return;
    }
    const Eigen::MatrixXd* mat = omega->as_matrix();
    int rows = mat->rows();
    int cols = mat->cols();

    if (axis_idx < 0 || axis_idx > 1) {
        m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
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
        m->throw_error("AXIS ERROR: catenate-axis does not support additional axis modifier", m->control, 4, 0);
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
        m->throw_error("AXIS ERROR: axis must be scalar or one-element vector", m->control, 4, 0);
        return;
    }
    bool laminate = !is_near_integer(k);
    int axis_idx = static_cast<int>(std::floor(k)) - m->io;  // ⎕IO

    if (lhs->is_string()) lhs = lhs->to_char_vector(m->heap);
    if (rhs->is_string()) rhs = rhs->to_char_vector(m->heap);

    if (!lhs->is_scalar() && !lhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ,[k] requires array argument", nullptr, 11, 0);
        return;
    }
    if (!rhs->is_scalar() && !rhs->is_array()) {
        m->throw_error("DOMAIN ERROR: ,[k] requires array argument", nullptr, 11, 0);
        return;
    }

    // Handle NDARRAY catenation
    if (lhs->is_ndarray() || rhs->is_ndarray()) {
        if (laminate) {
            m->throw_error("RANK ERROR: laminate of NDARRAY not yet supported", nullptr, 4, 0);
            return;
        }

        bool is_char = lhs->is_char_data() && rhs->is_char_data();

        // Get shapes of both operands
        std::vector<int> lhs_shape, rhs_shape;
        const Eigen::VectorXd* lhs_data = nullptr;
        const Eigen::VectorXd* rhs_data = nullptr;

        if (lhs->is_ndarray()) {
            lhs_shape = lhs->ndarray_shape();
            lhs_data = lhs->ndarray_data();
        } else if (lhs->is_scalar()) {
            lhs_shape = {};
        } else {
            lhs_shape.push_back(lhs->as_matrix()->rows());
            if (!lhs->is_vector()) {
                lhs_shape.push_back(lhs->as_matrix()->cols());
            }
        }

        if (rhs->is_ndarray()) {
            rhs_shape = rhs->ndarray_shape();
            rhs_data = rhs->ndarray_data();
        } else if (rhs->is_scalar()) {
            rhs_shape = {};
        } else {
            rhs_shape.push_back(rhs->as_matrix()->rows());
            if (!rhs->is_vector()) {
                rhs_shape.push_back(rhs->as_matrix()->cols());
            }
        }

        int target_rank = std::max(static_cast<int>(lhs_shape.size()),
                                   static_cast<int>(rhs_shape.size()));
        if (target_rank == 0) {
            m->throw_error("RANK ERROR: cannot catenate scalars along axis", nullptr, 4, 0);
            return;
        }

        if (axis_idx < 0 || axis_idx >= target_rank) {
            m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
            return;
        }

        // Promote shapes to same rank by prepending 1s
        while (static_cast<int>(lhs_shape.size()) < target_rank) {
            lhs_shape.insert(lhs_shape.begin(), 1);
        }
        while (static_cast<int>(rhs_shape.size()) < target_rank) {
            rhs_shape.insert(rhs_shape.begin(), 1);
        }

        // Verify all axes except cat_axis match
        for (int i = 0; i < target_rank; ++i) {
            if (i != axis_idx && lhs_shape[i] != rhs_shape[i]) {
                m->throw_error("LENGTH ERROR: incompatible shapes for catenation", nullptr, 5, 0);
                return;
            }
        }

        // Compute result shape
        std::vector<int> result_shape = lhs_shape;
        result_shape[axis_idx] = lhs_shape[axis_idx] + rhs_shape[axis_idx];

        // Compute strides
        auto compute_strides = [](const std::vector<int>& shape) {
            std::vector<int> strides(shape.size());
            int stride = 1;
            for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        };

        std::vector<int> lhs_strides = compute_strides(lhs_shape);
        std::vector<int> rhs_strides = compute_strides(rhs_shape);
        std::vector<int> result_strides = compute_strides(result_shape);

        int result_size = 1;
        for (int d : result_shape) result_size *= d;

        Eigen::VectorXd result(result_size);

        // Helper lambdas to get values
        auto get_lhs_value = [&](const std::vector<int>& indices) -> double {
            if (lhs->is_scalar()) return lhs->as_scalar();
            if (lhs->is_ndarray()) {
                int lin = 0;
                for (int i = 0; i < target_rank; ++i) {
                    lin += indices[i] * lhs_strides[i];
                }
                return (*lhs_data)(lin);
            }
            const Eigen::MatrixXd* mat = lhs->as_matrix();
            if (lhs->is_vector()) {
                return (*mat)(indices[target_rank - 1], 0);
            }
            return (*mat)(indices[target_rank - 2], indices[target_rank - 1]);
        };

        auto get_rhs_value = [&](const std::vector<int>& indices) -> double {
            if (rhs->is_scalar()) return rhs->as_scalar();
            if (rhs->is_ndarray()) {
                int lin = 0;
                for (int i = 0; i < target_rank; ++i) {
                    lin += indices[i] * rhs_strides[i];
                }
                return (*rhs_data)(lin);
            }
            const Eigen::MatrixXd* mat = rhs->as_matrix();
            if (rhs->is_vector()) {
                return (*mat)(indices[target_rank - 1], 0);
            }
            return (*mat)(indices[target_rank - 2], indices[target_rank - 1]);
        };

        // Fill result
        std::vector<int> result_indices(target_rank);
        for (int linear = 0; linear < result_size; ++linear) {
            int remaining = linear;
            for (int d = 0; d < target_rank; ++d) {
                result_indices[d] = remaining / result_strides[d];
                remaining %= result_strides[d];
            }

            if (result_indices[axis_idx] < lhs_shape[axis_idx]) {
                result(linear) = get_lhs_value(result_indices);
            } else {
                std::vector<int> rhs_indices = result_indices;
                rhs_indices[axis_idx] -= lhs_shape[axis_idx];
                result(linear) = get_rhs_value(rhs_indices);
            }
        }

        // Allocate result based on rank
        if (target_rank <= 2) {
            if (target_rank == 1) {
                m->result = m->heap->allocate_vector(result, is_char);
            } else {
                Eigen::MatrixXd mat(result_shape[0], result_shape[1]);
                for (int i = 0; i < result_shape[0]; ++i) {
                    for (int j = 0; j < result_shape[1]; ++j) {
                        mat(i, j) = result(i * result_shape[1] + j);
                    }
                }
                m->result = m->heap->allocate_matrix(mat, is_char);
            }
        } else {
            m->result = m->heap->allocate_ndarray(std::move(result), std::move(result_shape));
            m->result->set_char_data(is_char);
        }
        return;
    }

    const Eigen::MatrixXd* lmat = lhs->as_matrix();
    const Eigen::MatrixXd* rmat = rhs->as_matrix();
    int lrows = lmat->rows();
    int lcols = lmat->cols();
    int rrows = rmat->rows();
    int rcols = rmat->cols();

    if (laminate) {
        if (lhs->is_vector() && rhs->is_vector()) {
            if (lrows != rrows) {
                m->throw_error("LENGTH ERROR: vectors must have same length for laminate", nullptr, 5, 0);
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

        // Matrix laminate: creates 3D NDARRAY
        // axis_idx determines where the new axis of length 2 is inserted
        if (lhs->is_matrix() && rhs->is_matrix()) {
            if (lrows != rrows || lcols != rcols) {
                m->throw_error("LENGTH ERROR: matrices must have same shape for laminate", nullptr, 5, 0);
                return;
            }

            // Determine result shape based on axis_idx
            // axis_idx < 0: new axis before first (2×rows×cols)
            // axis_idx = 0: new axis after first (rows×2×cols)
            // axis_idx = 1: new axis after second (rows×cols×2)
            std::vector<int> result_shape;
            int new_axis_pos;
            if (axis_idx < 0) {
                result_shape = {2, lrows, lcols};
                new_axis_pos = 0;
            } else if (axis_idx == 0) {
                result_shape = {lrows, 2, lcols};
                new_axis_pos = 1;
            } else {
                result_shape = {lrows, lcols, 2};
                new_axis_pos = 2;
            }

            int total_size = 2 * lrows * lcols;
            Eigen::VectorXd result(total_size);

            // Copy data with proper indexing
            for (int i = 0; i < lrows; ++i) {
                for (int j = 0; j < lcols; ++j) {
                    int lin_lhs, lin_rhs;
                    if (new_axis_pos == 0) {
                        // Shape: 2×rows×cols
                        lin_lhs = 0 * (lrows * lcols) + i * lcols + j;
                        lin_rhs = 1 * (lrows * lcols) + i * lcols + j;
                    } else if (new_axis_pos == 1) {
                        // Shape: rows×2×cols
                        lin_lhs = i * (2 * lcols) + 0 * lcols + j;
                        lin_rhs = i * (2 * lcols) + 1 * lcols + j;
                    } else {
                        // Shape: rows×cols×2
                        lin_lhs = i * (lcols * 2) + j * 2 + 0;
                        lin_rhs = i * (lcols * 2) + j * 2 + 1;
                    }
                    result(lin_lhs) = (*lmat)(i, j);
                    result(lin_rhs) = (*rmat)(i, j);
                }
            }

            m->result = m->heap->allocate_ndarray(std::move(result), std::move(result_shape));
            return;
        }

        // Vector-matrix or matrix-vector laminate
        if ((lhs->is_vector() && rhs->is_matrix()) || (lhs->is_matrix() && rhs->is_vector())) {
            m->throw_error("RANK ERROR: laminate requires matching ranks", nullptr, 4, 0);
            return;
        }

        m->throw_error("RANK ERROR: unsupported laminate combination", nullptr, 4, 0);
        return;
    }

    int cat_axis = static_cast<int>(std::round(k)) - m->io;  // ⎕IO

    // Scalars have no axes - cannot catenate with axis specification
    if (lhs->is_scalar() && rhs->is_scalar()) {
        m->throw_error("AXIS ERROR: cannot catenate scalars with axis", m->control, 4, 0);
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
            m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
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
            m->throw_error("LENGTH ERROR: column counts must match for vertical catenation", nullptr, 5, 0);
            return;
        }
        Eigen::MatrixXd result(lrows + rrows, lcols);
        result.topRows(lrows) = *lmat;
        result.bottomRows(rrows) = *rmat;
        m->result = m->heap->allocate_matrix(result);
    } else if (cat_axis == 1) {
        if (lrows != rrows) {
            m->throw_error("LENGTH ERROR: row counts must match for horizontal catenation", nullptr, 5, 0);
            return;
        }
        Eigen::MatrixXd result(lrows, lcols + rcols);
        result.leftCols(lcols) = *lmat;
        result.rightCols(rcols) = *rmat;
        m->result = m->heap->allocate_matrix(result);
    } else {
        m->throw_error("AXIS ERROR: axis out of range", m->control, 4, 0);
    }
}

} // namespace apl
