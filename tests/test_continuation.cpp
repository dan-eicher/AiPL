// Continuation tests

#include <gtest/gtest.h>
#include "continuation.h"
#include "heap.h"
#include "machine.h"
#include "value.h"
#include "primitives.h"
#include "operators.h"

using namespace apl;

class ContinuationTest : public ::testing::Test {
protected:
    Machine* machine;
    APLHeap* heap;  // Convenience pointer to machine->heap

    void SetUp() override {
        machine = new Machine();
        heap = machine->heap;
    }

    void TearDown() override {
        delete machine;
    }
};

// Test HaltK
TEST_F(ContinuationTest, HaltK) {
    HaltK* halt = heap->allocate<HaltK>();

    Value* v = machine->heap->allocate_scalar(42.0);
    machine->ctrl.set_value(v);

    machine->push_kont(halt);
    Value* result = machine->execute();

    EXPECT_EQ(result, v);
    // Phase 1: No more ctrl.mode - halted is implicit when stack empty

}

// Test HaltK mark (should do nothing)
TEST_F(ContinuationTest, HaltKMark) {
    HaltK* halt = heap->allocate<HaltK>();

    // Should not crash
    halt->mark(nullptr);

}

// Test HaltK is not a boundary
TEST_F(ContinuationTest, HaltKNotBoundary) {
    HaltK* halt = heap->allocate<HaltK>();

    EXPECT_FALSE(halt->is_function_boundary());
    EXPECT_FALSE(halt->is_loop_boundary());
    EXPECT_FALSE(halt->matches_label("any"));

}

// Test ArgK basic
TEST_F(ContinuationTest, ArgKBasic) {
    Value* arg = machine->heap->allocate_scalar(10.0);
    HaltK* halt = heap->allocate<HaltK>();
    ArgK* argk = heap->allocate<ArgK>(arg, halt);

    machine->push_kont(argk);
    Value* result = machine->execute();

    EXPECT_EQ(result, arg);
    EXPECT_EQ(machine->ctrl.value, arg);

}

// Test ArgK without next
TEST_F(ContinuationTest, ArgKWithoutNext) {
    Value* arg = machine->heap->allocate_scalar(99.0);
    ArgK* argk = heap->allocate<ArgK>(arg, nullptr);

    machine->push_kont(argk);
    Value* result = machine->execute();

    EXPECT_EQ(result, arg);

}

// Test ArgK chaining
TEST_F(ContinuationTest, ArgKChaining) {
    Value* arg1 = machine->heap->allocate_scalar(1.0);
    Value* arg2 = machine->heap->allocate_scalar(2.0);

    HaltK* halt = heap->allocate<HaltK>();
    ArgK* argk2 = heap->allocate<ArgK>(arg2, halt);
    ArgK* argk1 = heap->allocate<ArgK>(arg1, argk2);

    // Push first ArgK and execute via trampoline
    machine->push_kont(argk1);
    Value* result = machine->execute();

    // Should eventually return arg2 (last in chain)
    EXPECT_EQ(result, arg2);
    EXPECT_EQ(machine->ctrl.value, arg2);

}

// Test ArgK mark
TEST_F(ContinuationTest, ArgKMark) {
    Value* arg = machine->heap->allocate_scalar(5.0);
    ArgK* argk = heap->allocate<ArgK>(arg, nullptr);

    // Should not crash with nullptr heap
    argk->mark(nullptr);

}

// Test ArgK is not a boundary
TEST_F(ContinuationTest, ArgKNotBoundary) {
    Value* arg = machine->heap->allocate_scalar(1.0);
    ArgK* argk = heap->allocate<ArgK>(arg, nullptr);

    EXPECT_FALSE(argk->is_function_boundary());
    EXPECT_FALSE(argk->is_loop_boundary());
    EXPECT_FALSE(argk->matches_label("label"));

}

// Test FrameK basic
TEST_F(ContinuationTest, FrameKBasic) {
    HaltK* halt = heap->allocate<HaltK>();
    FrameK* frame = heap->allocate<FrameK>("test_func", halt);

    Value* v = machine->heap->allocate_scalar(7.0);
    machine->ctrl.set_value(v);

    // Push frame onto stack and execute via trampoline
    machine->push_kont(frame);
    Value* result = machine->execute();

    EXPECT_EQ(result, v);
    // Phase 1: No more ctrl.mode - halted is implicit when stack empty

}

// Test FrameK is function boundary
TEST_F(ContinuationTest, FrameKIsFunctionBoundary) {
    FrameK* frame = heap->allocate<FrameK>("my_func", nullptr);

    EXPECT_TRUE(frame->is_function_boundary());
    EXPECT_FALSE(frame->is_loop_boundary());
    EXPECT_FALSE(frame->matches_label("label"));

}

// Test FrameK without return continuation
TEST_F(ContinuationTest, FrameKWithoutReturn) {
    FrameK* frame = heap->allocate<FrameK>("func", nullptr);

    Value* v = machine->heap->allocate_scalar(88.0);
    machine->ctrl.set_value(v);

    // Push frame onto stack and execute via trampoline
    machine->push_kont(frame);
    Value* result = machine->execute();

    EXPECT_EQ(result, v);

}

// Test FrameK mark
TEST_F(ContinuationTest, FrameKMark) {
    HaltK* halt = heap->allocate<HaltK>();
    FrameK* frame = heap->allocate<FrameK>("test", halt);

    // Should not crash
    frame->mark(nullptr);

}

// Test FrameK function name
TEST_F(ContinuationTest, FrameKFunctionName) {
    const char* name = "my_function";
    FrameK* frame = heap->allocate<FrameK>(name, nullptr);

    EXPECT_EQ(frame->function_name, name);
    EXPECT_STREQ(frame->function_name, "my_function");

}

// Test Machine push and execute
TEST_F(ContinuationTest, MachinePushPop) {
    Value* v = machine->heap->allocate_scalar(42.0);
    machine->ctrl.set_value(v);

    HaltK* halt = heap->allocate<HaltK>();
    machine->push_kont(halt);

    EXPECT_EQ(machine->kont_stack.size(), 1);

    Value* result = machine->execute();

    EXPECT_EQ(result, v);
    EXPECT_EQ(machine->kont_stack.size(), 0);
    // Phase 1: No more ctrl.mode - halted is implicit when stack empty

}

// Test Machine execute with empty stack
TEST_F(ContinuationTest, MachinePopEmpty) {
    Value* v = machine->heap->allocate_scalar(123.0);
    machine->ctrl.set_value(v);

    EXPECT_EQ(machine->kont_stack.size(), 0);

    Value* result = machine->execute();

    EXPECT_EQ(result, v);
    // Phase 1: No more ctrl.mode - halted is implicit when stack empty

}

// Test Machine with multiple continuations
TEST_F(ContinuationTest, MachineMultipleContinuations) {
    Value* arg1 = machine->heap->allocate_scalar(10.0);
    Value* arg2 = machine->heap->allocate_scalar(20.0);

    HaltK* halt = heap->allocate<HaltK>();
    ArgK* argk2 = heap->allocate<ArgK>(arg2, halt);
    ArgK* argk1 = heap->allocate<ArgK>(arg1, argk2);

    machine->push_kont(argk1);

    EXPECT_EQ(machine->kont_stack.size(), 1);

    Value* result = machine->execute();

    EXPECT_EQ(result, arg2);  // Last arg in chain

}

// Test continuation chaining with FrameK
TEST_F(ContinuationTest, FrameKChaining) {
    Value* v = machine->heap->allocate_scalar(99.0);

    HaltK* halt = heap->allocate<HaltK>();
    FrameK* frame = heap->allocate<FrameK>("outer", halt);
    ArgK* argk = heap->allocate<ArgK>(v, frame);

    machine->ctrl.set_value(v);

    machine->push_kont(argk);
    Value* result = machine->execute();

    EXPECT_EQ(result, v);

}

// ============================================================================
// LiteralK Tests - Parse-time safe continuation
// ============================================================================

// Test LiteralK basic invoke
TEST_F(ContinuationTest, LiteralKBasic) {
    auto lit = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(42.0));

    // Push onto stack and execute via trampoline
    machine->push_kont(lit);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
    EXPECT_EQ(machine->ctrl.value, result);
}

// Test LiteralK without next - now just tests basic execution
TEST_F(ContinuationTest, LiteralKWithoutNext) {
    auto lit = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(3.14));

    machine->push_kont(lit);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.14);
}

// Test LiteralK chaining via trampoline
TEST_F(ContinuationTest, LiteralKChaining) {
    auto lit1 = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(10.0));
    auto lit2 = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(20.0));

    // Push in reverse order (stack is LIFO)
    machine->push_kont(lit2);
    machine->push_kont(lit1);

    Value* result = machine->execute();

    // Last continuation wins - should have 20.0
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

// Test LiteralK uses scalar cache
TEST_F(ContinuationTest, LiteralKUsesScalarCache) {
    auto lit1 = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(42.0));
    auto lit2 = static_cast<LiteralK*>(machine->heap->allocate<LiteralK>(42.0));

    machine->push_kont(lit1);
    Value* result1 = machine->execute();

    // Phase 1: No more init_evaluating() - machine is ready
    machine->push_kont(lit2);
    Value* result2 = machine->execute();

    EXPECT_EQ(result1, result2);
    EXPECT_DOUBLE_EQ(result1->as_scalar(), 42.0);
}

// Test LiteralK parse-time safety - stores double not Value*
TEST_F(ContinuationTest, LiteralKParseTimeSafety) {
    LiteralK* lit = heap->allocate<LiteralK>(123.0);

    EXPECT_DOUBLE_EQ(lit->literal_value, 123.0);

    // lit is already heap-allocated, no need to register again

    size_t values_before = machine->heap->total_size();
    machine->push_kont(lit);
    Value* result = machine->execute();
    size_t values_after = machine->heap->total_size();

    // Should have allocated a new Value for the literal
    EXPECT_GT(values_after, values_before);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 123.0);
}

// ============================================================================
// G2 Grammar Continuation Tests
// ============================================================================

// Test DerivedOperatorK with environment lookup
TEST_F(ContinuationTest, DerivedOperatorKCreation) {
    // Create operand continuation
    LiteralK* operand_lit = heap->allocate<LiteralK>(3.0);

    // Create DerivedOperatorK for ¨ operator
    const char* op_name = machine->string_pool.intern("¨");
    DerivedOperatorK* derived_k = heap->allocate<DerivedOperatorK>(operand_lit, op_name);

    EXPECT_NE(derived_k, nullptr);
    EXPECT_EQ(derived_k->op_name, op_name);
    EXPECT_EQ(derived_k->operand_cont, operand_lit);
}

// Test ApplyDerivedOperatorK creates derived operator value
TEST_F(ContinuationTest, ApplyDerivedOperatorKExecution) {
    // Set up: operand value in ctrl.value (a function)
    Value* operand = heap->allocate_primitive(&prim_plus);
    machine->ctrl.value = operand;

    // Create ApplyDerivedOperatorK for . operator (inner product - dyadic)
    const char* op_name = machine->string_pool.intern(".");
    ApplyDerivedOperatorK* apply_k = heap->allocate<ApplyDerivedOperatorK>(op_name);

    // Push and execute
    machine->push_kont(apply_k);
    machine->execute();

    // Result should be a DERIVED_OPERATOR value
    Value* result = machine->ctrl.value;
    EXPECT_NE(result, nullptr);
    EXPECT_TRUE(result->is_derived_operator());
    EXPECT_EQ(result->data.derived_op->first_operand, operand);
    EXPECT_EQ(result->data.derived_op->op, &op_dot);
}

// Test DerivedOperatorK marking for GC
TEST_F(ContinuationTest, DerivedOperatorKMarking) {
    LiteralK* operand_cont = heap->allocate<LiteralK>(5.0);
    const char* op_name = machine->string_pool.intern("⍨");

    DerivedOperatorK* derived = heap->allocate<DerivedOperatorK>(operand_cont, op_name);

    // Clear marks
    machine->heap->clear_marks();

    // Mark the derived operator continuation
    derived->mark(machine->heap);

    // Operand continuation should be marked
    EXPECT_TRUE(operand_cont->marked);
}

// Test g' transformation finalization at top level
TEST_F(ContinuationTest, GprimeFinalizationAtTopLevel) {
    // Test that overloaded monadic functions create CURRIED_FN that gets finalized at top level
    // MonadicK("×", 5) should:
    // 1. ApplyMonadicK creates CURRIED_FN(×, 5, false) since × is overloaded
    // 2. Machine::execute() should finalize it to scalar 1.0 (signum of 5)

    // Create MonadicK for "×5" (signum of 5)
    LiteralK* lit = heap->allocate<LiteralK>(5.0);
    const char* times_name = machine->string_pool.intern("×");
    MonadicK* monadic = heap->allocate<MonadicK>(times_name, lit);

    machine->push_kont(monadic);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);

    // Debug: check what type we got
    if (result->tag == ValueType::CURRIED_FN) {
        std::cout << "ERROR: Result is still CURRIED_FN, not finalized!" << std::endl;
        std::cout << "curry_type: " << static_cast<int>(result->data.curried_fn->curry_type) << std::endl;
    }

    // Should be finalized to a scalar (signum of 5 = 1.0)
    EXPECT_TRUE(result->is_scalar()) << "Result should be scalar after g' finalization";
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);
    }
}

// ============================================================================
// DispatchFunctionK Tests - Function dispatch continuation
// ============================================================================

TEST_F(ContinuationTest, DispatchFunctionKMonadicPrimitive) {
    // Test monadic dispatch with pure monadic function (iota)
    Value* fn = machine->heap->allocate_primitive(&prim_iota);
    Value* arg = machine->heap->allocate_scalar(5.0);

    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(fn, nullptr, arg);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->rows(), 5);
}

TEST_F(ContinuationTest, DispatchFunctionKDyadicPrimitive) {
    // Test dyadic dispatch with both arguments
    Value* fn = machine->heap->allocate_primitive(&prim_plus);
    Value* left = machine->heap->allocate_scalar(3.0);
    Value* right = machine->heap->allocate_scalar(4.0);

    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(fn, left, right);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

TEST_F(ContinuationTest, DispatchFunctionKGPrimeFinalization) {
    // Test G_PRIME finalization at top level: overloaded function with one arg
    // creates G_PRIME curry which is then finalized by execute() to monadic result
    // This is correct G2 semantics: at top level, g' resolves to g1(x)
    Value* fn = machine->heap->allocate_primitive(&prim_minus);
    Value* arg = machine->heap->allocate_scalar(5.0);

    // Without force_monadic, creates G_PRIME curry, but execute() finalizes it
    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(fn, nullptr, arg, false);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    // G_PRIME is finalized at top level to monadic result: -5
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(ContinuationTest, DispatchFunctionKGPrimeDyadicApplication) {
    // Test G_PRIME curry applied with second argument -> dyadic
    // Create curry of minus with 5, then apply 3 -> 3 - 5 = -2
    Value* fn = machine->heap->allocate_primitive(&prim_minus);
    Value* five = machine->heap->allocate_scalar(5.0);

    // Create the G_PRIME curry directly (simulating what DispatchFunctionK does)
    Value* curried = machine->heap->allocate_curried_fn(fn, five, Value::CurryType::G_PRIME);

    // Now dispatch the curried function with a new argument
    Value* three = machine->heap->allocate_scalar(3.0);
    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(curried, nullptr, three);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    // G_PRIME with new arg uses dyadic: 3 - 5 = -2
    EXPECT_DOUBLE_EQ(result->as_scalar(), -2.0);
}

TEST_F(ContinuationTest, DispatchFunctionKForceMonadic) {
    // Test force_monadic=true: bypasses G_PRIME, applies monadic immediately
    Value* fn = machine->heap->allocate_primitive(&prim_minus);
    Value* arg = machine->heap->allocate_scalar(5.0);

    // With force_monadic=true, should apply monadic form directly
    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(fn, nullptr, arg, true);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);  // Negated value
}

TEST_F(ContinuationTest, DispatchFunctionKForceMonadicVector) {
    // Test force_monadic with vector argument
    Value* fn = machine->heap->allocate_primitive(&prim_minus);
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* arg = machine->heap->allocate_vector(vec);

    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(fn, nullptr, arg, true);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), -3.0);
}

TEST_F(ContinuationTest, DispatchFunctionKDyadicCurry) {
    // Test DYADIC_CURRY: pure dyadic function with one arg creates curry
    // Equal (=) is truly dyadic-only (no monadic form)
    Value* fn = machine->heap->allocate_primitive(&prim_equal);
    Value* arg = machine->heap->allocate_scalar(2.0);

    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(fn, nullptr, arg);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CURRIED_FN);
    EXPECT_EQ(result->data.curried_fn->curry_type, Value::CurryType::DYADIC_CURRY);
}

TEST_F(ContinuationTest, DispatchFunctionKCurriedFnUnwrap) {
    // Test applying a DYADIC_CURRY: captured arg is right, new arg is left
    // Create "÷2" (divide by 2), then apply to 10
    Value* fn = machine->heap->allocate_primitive(&prim_divide);
    Value* two = machine->heap->allocate_scalar(2.0);

    // First create the curry
    Value* curried = machine->heap->allocate_curried_fn(fn, two, Value::CurryType::DYADIC_CURRY);

    // Now apply curry to 10
    Value* ten = machine->heap->allocate_scalar(10.0);
    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(curried, nullptr, ten);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    // 10 ÷ 2 = 5
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(ContinuationTest, DispatchFunctionKNullFnUsesCtrl) {
    // Test that nullptr fn_val reads from ctrl.value
    Value* fn = machine->heap->allocate_primitive(&prim_iota);
    Value* arg = machine->heap->allocate_scalar(3.0);

    // Set ctrl.value to the function
    machine->ctrl.set_value(fn);

    // Create dispatch with nullptr fn - should read from ctrl
    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(nullptr, nullptr, arg);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->rows(), 3);
}

// ============================================================================
// CellIterK Tests - General-purpose cell iterator for operators
// ============================================================================

TEST_F(ContinuationTest, CellIterKCollectModeVector) {
    // Test CellIterK in COLLECT mode: apply - to each element of vector
    // -⍤0 (1 2 3 4) should give (-1 -2 -3 -4)
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    // Create CellIterK: apply fn to 0-cells (scalars) of rhs
    // 4 cells, COLLECT mode, original shape 4x1, is_vector=true
    CellIterK* iter = heap->allocate<CellIterK>(
        fn, nullptr, rhs, 0, 0, 4,
        CellIterMode::COLLECT, 4, 1, true);

    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), -3.0);
    EXPECT_DOUBLE_EQ((*m)(3, 0), -4.0);
}

TEST_F(ContinuationTest, CellIterKCollectModeMatrix) {
    // Test CellIterK with matrix: apply - to each 0-cell
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    // 6 cells (2*3), COLLECT mode
    CellIterK* iter = heap->allocate<CellIterK>(
        fn, nullptr, rhs, 0, 0, 6,
        CellIterMode::COLLECT, 2, 3, false);

    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), -6.0);
}

TEST_F(ContinuationTest, CellIterKFoldRightMode) {
    // Test FOLD_RIGHT: +/ 1 2 3 4 = 10 (right-to-left: 1+(2+(3+4)))
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    CellIterK* iter = heap->allocate<CellIterK>(
        fn, nullptr, rhs, 0, 0, 4,
        CellIterMode::FOLD_RIGHT, 4, 1, true);

    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

TEST_F(ContinuationTest, CellIterKFoldRightSubtract) {
    // Test FOLD_RIGHT with non-commutative op: -/ 10 3 2 1
    // Right-to-left: 10-(3-(2-1)) = 10-(3-1) = 10-2 = 8
    Eigen::VectorXd vec(4);
    vec << 10, 3, 2, 1;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    CellIterK* iter = heap->allocate<CellIterK>(
        fn, nullptr, rhs, 0, 0, 4,
        CellIterMode::FOLD_RIGHT, 4, 1, true);

    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 8.0);
}

TEST_F(ContinuationTest, CellIterKScanRightMode) {
    // Test SCAN_RIGHT: right-to-left cumulative scan
    // For +\ 1 2 3 4 from right: 4, 4+3=7, 7+2=9, 9+1=10
    // Reversed for output order: 10, 9, 7, 4
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    CellIterK* iter = heap->allocate<CellIterK>(
        fn, nullptr, rhs, 0, 0, 4,
        CellIterMode::SCAN_RIGHT, 4, 1, true);

    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    // Right scan gives running totals from right, then reversed
    EXPECT_DOUBLE_EQ((*m)(0, 0), 10.0);  // 1+2+3+4
    EXPECT_DOUBLE_EQ((*m)(1, 0), 9.0);   // 2+3+4
    EXPECT_DOUBLE_EQ((*m)(2, 0), 7.0);   // 3+4
    EXPECT_DOUBLE_EQ((*m)(3, 0), 4.0);   // 4
}

TEST_F(ContinuationTest, CellIterKDyadicCollect) {
    // Test dyadic COLLECT: 1 2 3 + 10 20 30 element-wise
    Eigen::VectorXd vec1(3);
    vec1 << 1, 2, 3;
    Value* lhs = machine->heap->allocate_vector(vec1);

    Eigen::VectorXd vec2(3);
    vec2 << 10, 20, 30;
    Value* rhs = machine->heap->allocate_vector(vec2);

    Value* fn = machine->heap->allocate_primitive(&prim_plus);

    CellIterK* iter = heap->allocate<CellIterK>(
        fn, lhs, rhs, 0, 0, 3,
        CellIterMode::COLLECT, 3, 1, true);

    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 22.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 33.0);
}

TEST_F(ContinuationTest, CellIterKSingleCell) {
    // Test with single cell - should just apply function once
    Value* rhs = machine->heap->allocate_scalar(5.0);
    Value* fn = machine->heap->allocate_primitive(&prim_minus);

    CellIterK* iter = heap->allocate<CellIterK>(
        fn, nullptr, rhs, 0, 0, 1,
        CellIterMode::COLLECT, 1, 1, false);

    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    // Single scalar result
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

TEST_F(ContinuationTest, CellIterKWithDerivedOperator) {
    // Test CellIterK with a DERIVED_OPERATOR (reduce)
    // Apply +/ to each row of a 3x2 matrix
    // Should return vector [3, 7, 11] (row sums)
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2,
           3, 4,
           5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);

    // Create DERIVED_OPERATOR(/, +)
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);
    Value* reduce_fn = machine->heap->allocate_derived_operator(&op_reduce, plus_fn);

    // CellIterK with rank 1 (rows) on 3x2 matrix = 3 cells
    CellIterK* iter = heap->allocate<CellIterK>(
        reduce_fn, nullptr, rhs, 1, 1, 3,
        CellIterMode::COLLECT, 3, 2, false);

    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);   // 1+2
    EXPECT_DOUBLE_EQ((*m)(1, 0), 7.0);   // 3+4
    EXPECT_DOUBLE_EQ((*m)(2, 0), 11.0);  // 5+6
}

TEST_F(ContinuationTest, CellIterKReduceRowsSimple) {
    // Apply +/ to a single vector (should reduce to scalar)
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* rhs = machine->heap->allocate_vector(vec);

    // Create DERIVED_OPERATOR(/, +)
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);
    Value* reduce_fn = machine->heap->allocate_derived_operator(&op_reduce, plus_fn);

    // Single cell (the whole vector)
    CellIterK* iter = heap->allocate<CellIterK>(
        reduce_fn, nullptr, rhs, 1, 1, 1,
        CellIterMode::COLLECT, 3, 1, true);

    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);  // 1+2+3
}

// ============================================================================
// RowReduceK Tests
// ============================================================================

TEST_F(ContinuationTest, RowReduceKBasic) {
    // Reduce each row of 2x3 matrix with +
    // [[1,2,3],[4,5,6]] -> [6, 15]
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    RowReduceK* iter = heap->allocate<RowReduceK>(plus_fn, rhs, 2, 3, false);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ((*m)(1, 0), 15.0);  // 4+5+6
}

TEST_F(ContinuationTest, RowReduceKFirstAxis) {
    // Reduce each column of 2x3 matrix with + (first axis)
    // [[1,2,3],[4,5,6]] -> [5, 7, 9]
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    // reduce_first_axis=true means iterate over columns
    RowReduceK* iter = heap->allocate<RowReduceK>(plus_fn, rhs, 3, 2, true);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 5.0);   // 1+4
    EXPECT_DOUBLE_EQ((*m)(1, 0), 7.0);   // 2+5
    EXPECT_DOUBLE_EQ((*m)(2, 0), 9.0);   // 3+6
}

TEST_F(ContinuationTest, RowReduceKMultiply) {
    // Reduce each row with × (multiply)
    // [[2,3],[4,5]] -> [6, 20]
    Eigen::MatrixXd mat(2, 2);
    mat << 2, 3,
           4, 5;
    Value* rhs = machine->heap->allocate_matrix(mat);
    Value* times_fn = machine->heap->allocate_primitive(&prim_times);

    RowReduceK* iter = heap->allocate<RowReduceK>(times_fn, rhs, 2, 2, false);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 6.0);   // 2*3
    EXPECT_DOUBLE_EQ((*m)(1, 0), 20.0);  // 4*5
}

// ============================================================================
// PrefixScanK Tests
// ============================================================================

TEST_F(ContinuationTest, PrefixScanKBasic) {
    // +\ [1, 2, 3, 4] -> [1, 3, 6, 10]
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    PrefixScanK* iter = heap->allocate<PrefixScanK>(plus_fn, rhs, 4);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);   // 1+2
    EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ((*m)(3, 0), 10.0);  // 1+2+3+4
}

TEST_F(ContinuationTest, PrefixScanKNonAssociative) {
    // -\ [1, 2, 3, 4] with right-to-left reduction
    // Position 0: 1
    // Position 1: 1-(2) = -1
    // Position 2: 1-(2-3) = 1-(-1) = 2
    // Position 3: 1-(2-(3-4)) = 1-(2-(-1)) = 1-3 = -2
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* minus_fn = machine->heap->allocate_primitive(&prim_minus);

    PrefixScanK* iter = heap->allocate<PrefixScanK>(minus_fn, rhs, 4);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), -1.0);  // 1-2
    EXPECT_DOUBLE_EQ((*m)(2, 0), 2.0);   // 1-(2-3) = 1-(-1)
    EXPECT_DOUBLE_EQ((*m)(3, 0), -2.0);  // 1-(2-(3-4))
}

// ============================================================================
// RowScanK Tests
// ============================================================================

TEST_F(ContinuationTest, RowScanKBasic) {
    // +\ on each row of 2x3 matrix
    // [[1,2,3],[4,5,6]] -> [[1,3,6],[4,9,15]]
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    RowScanK* iter = heap->allocate<RowScanK>(plus_fn, rhs, 2, 3, false);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    // Row 0: 1, 1+2=3, 1+2+3=6
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 3.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), 6.0);
    // Row 1: 4, 4+5=9, 4+5+6=15
    EXPECT_DOUBLE_EQ((*m)(1, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 9.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 15.0);
}

TEST_F(ContinuationTest, RowScanKFirstAxis) {
    // +⍀ on each column of 2x3 matrix (first axis)
    // [[1,2,3],[4,5,6]] -> [[1,2,3],[5,7,9]]
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    // scan_first_axis=true means iterate over columns
    RowScanK* iter = heap->allocate<RowScanK>(plus_fn, rhs, 3, 2, true);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);
    // Col 0: 1, 1+4=5
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 5.0);
    // Col 1: 2, 2+5=7
    EXPECT_DOUBLE_EQ((*m)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 7.0);
    // Col 2: 3, 3+6=9
    EXPECT_DOUBLE_EQ((*m)(0, 2), 3.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 9.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
