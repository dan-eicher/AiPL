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
    Heap* heap;  // Convenience pointer to machine->heap

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
    machine->result = v;

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

}

// Test ArgK basic
TEST_F(ContinuationTest, ArgKBasic) {
    Value* arg = machine->heap->allocate_scalar(10.0);
    HaltK* halt = heap->allocate<HaltK>();
    ArgK* argk = heap->allocate<ArgK>(arg, halt);

    machine->push_kont(argk);
    Value* result = machine->execute();

    EXPECT_EQ(result, arg);
    EXPECT_EQ(machine->result, arg);

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
    EXPECT_EQ(machine->result, arg2);

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
}

// Test FrameK basic
TEST_F(ContinuationTest, FrameKBasic) {
    HaltK* halt = heap->allocate<HaltK>();
    FrameK* frame = heap->allocate<FrameK>("test_func", halt);

    Value* v = machine->heap->allocate_scalar(7.0);
    machine->result = v;

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
}

// Test FrameK without return continuation
TEST_F(ContinuationTest, FrameKWithoutReturn) {
    FrameK* frame = heap->allocate<FrameK>("func", nullptr);

    Value* v = machine->heap->allocate_scalar(88.0);
    machine->result = v;

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
    machine->result = v;

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
    machine->result = v;

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

    machine->result = v;

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
    EXPECT_EQ(machine->result, result);
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
    // Set up: operand value in result (a function)
    Value* operand = heap->allocate_primitive(&prim_plus);
    machine->result = operand;

    // Create ApplyDerivedOperatorK for . operator (inner product - dyadic)
    const char* op_name = machine->string_pool.intern(".");
    ApplyDerivedOperatorK* apply_k = heap->allocate<ApplyDerivedOperatorK>(op_name);

    // Push and execute
    machine->push_kont(apply_k);
    machine->execute();

    // Result should be a DERIVED_OPERATOR value
    Value* result = machine->result;
    EXPECT_NE(result, nullptr);
    EXPECT_TRUE(result->is_derived_operator());
    EXPECT_EQ(result->data.derived_op->first_operand, operand);
    EXPECT_EQ(result->data.derived_op->primitive_op, &op_dot);
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
    // Test that nullptr fn_val reads from result
    Value* fn = machine->heap->allocate_primitive(&prim_iota);
    Value* arg = machine->heap->allocate_scalar(3.0);

    // Set result to the function
    machine->result = fn;

    // Create dispatch with nullptr fn - should read from ctrl
    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(nullptr, nullptr, arg);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->rows(), 3);
}

// ============================================================================
// DeferredDispatchK Tests - Deferred dispatch after subcomputation
// ============================================================================

TEST_F(ContinuationTest, DeferredDispatchKBasic) {
    // Test DeferredDispatchK: it should read machine->result as right_val
    // and dispatch the function with that value
    // Simulates nested reductions: when inner reduction completes,
    // DeferredDispatchK continues with the result

    Value* fn = machine->heap->allocate_primitive(&prim_minus);  // negate

    // Set up: machine->result will be 5.0 (simulating completed subcomputation)
    machine->result = machine->heap->allocate_scalar(5.0);

    // DeferredDispatchK should use result as right_val, apply fn monadically
    DeferredDispatchK* deferred = heap->allocate<DeferredDispatchK>(fn, nullptr, false);
    machine->push_kont(deferred);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);  // negate(5) = -5
}

TEST_F(ContinuationTest, DeferredDispatchKWithNestedReduce) {
    // Test the actual use case: nested reductions
    // +/ ×/ 1 2 3 = +/ 6 = 6
    // This test simulates what happens when ×/ 1 2 3 completes with result 6
    // and +/ needs to be applied to it

    // Create +/ derived operator
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);
    Value* reduce_op = machine->heap->allocate_derived_operator(&op_reduce, plus_fn);

    // Simulate inner reduction completing with result 6
    machine->result = machine->heap->allocate_scalar(6.0);

    // DeferredDispatchK should dispatch +/ to 6, giving +/ 6 = 6
    DeferredDispatchK* deferred = heap->allocate<DeferredDispatchK>(reduce_op, nullptr, false);
    machine->push_kont(deferred);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);  // +/ 6 = 6
}

TEST_F(ContinuationTest, DeferredDispatchKMarking) {
    // Test that DeferredDispatchK properly marks its values for GC
    Value* fn = machine->heap->allocate_primitive(&prim_plus);
    Value* left = machine->heap->allocate_scalar(3.0);

    DeferredDispatchK* deferred = heap->allocate<DeferredDispatchK>(fn, left, true);

    // Mark should not crash and should mark the values
    deferred->mark(heap);
    EXPECT_TRUE(fn->marked);
    EXPECT_TRUE(left->marked);
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

// ============================================================================
// NwiseReduceK Tests - N-wise reduction on vectors
// ============================================================================

TEST_F(ContinuationTest, NwiseReduceKBasicPairwise) {
    // 2 +/ 1 2 3 4 5 -> pairwise sums: 3 5 7 9
    Eigen::VectorXd vec(5);
    vec << 1, 2, 3, 4, 5;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    NwiseReduceK* iter = heap->allocate<NwiseReduceK>(plus_fn, rhs, 2, false);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);  // 5 - 2 + 1 = 4 windows
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);   // 1+2
    EXPECT_DOUBLE_EQ((*m)(1, 0), 5.0);   // 2+3
    EXPECT_DOUBLE_EQ((*m)(2, 0), 7.0);   // 3+4
    EXPECT_DOUBLE_EQ((*m)(3, 0), 9.0);   // 4+5
}

TEST_F(ContinuationTest, NwiseReduceKTriplets) {
    // 3 +/ 1 2 3 4 5 -> sums of 3: 6 9 12
    Eigen::VectorXd vec(5);
    vec << 1, 2, 3, 4, 5;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    NwiseReduceK* iter = heap->allocate<NwiseReduceK>(plus_fn, rhs, 3, false);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 3);  // 5 - 3 + 1 = 3 windows
    EXPECT_DOUBLE_EQ((*m)(0, 0), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ((*m)(1, 0), 9.0);   // 2+3+4
    EXPECT_DOUBLE_EQ((*m)(2, 0), 12.0);  // 3+4+5
}

TEST_F(ContinuationTest, NwiseReduceKReversed) {
    // ISO 13751 §9.2.3: Negative N reverses EACH WINDOW before reduction
    // For subtraction: ¯2 -/ 1 2 4 7 11
    // Windows before reversal: [1,2] [2,4] [4,7] [7,11]
    // Windows after reversal:  [2,1] [4,2] [7,4] [11,7]
    // Right-to-left reduction: 2-1=1, 4-2=2, 7-4=3, 11-7=4
    // Result: [1, 2, 3, 4]
    Eigen::VectorXd vec(5);
    vec << 1, 2, 4, 7, 11;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* minus_fn = machine->heap->allocate_primitive(&prim_minus);

    NwiseReduceK* iter = heap->allocate<NwiseReduceK>(minus_fn, rhs, 2, true);  // reverse=true
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 4);
    // Each window reversed, then reduced
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);   // [2,1] → 2-1=1
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);   // [4,2] → 4-2=2
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);   // [7,4] → 7-4=3
    EXPECT_DOUBLE_EQ((*m)(3, 0), 4.0);   // [11,7] → 11-7=4
}

TEST_F(ContinuationTest, NwiseReduceKFullWindow) {
    // N equals vector length - single result
    // 4 +/ 1 2 3 4 -> 10
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    NwiseReduceK* iter = heap->allocate<NwiseReduceK>(plus_fn, rhs, 4, false);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 1);  // 4 - 4 + 1 = 1 window
    EXPECT_DOUBLE_EQ((*m)(0, 0), 10.0);  // 1+2+3+4
}

TEST_F(ContinuationTest, NwiseReduceKMarking) {
    // Test GC marking
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* rhs = machine->heap->allocate_vector(vec);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    NwiseReduceK* iter = heap->allocate<NwiseReduceK>(plus_fn, rhs, 2, false);

    machine->heap->clear_marks();
    iter->mark(machine->heap);

    EXPECT_TRUE(plus_fn->marked);
    EXPECT_TRUE(rhs->marked);
}

// ============================================================================
// NwiseMatrixReduceK Tests - N-wise reduction on matrices
// ============================================================================

TEST_F(ContinuationTest, NwiseMatrixReduceKAxis2) {
    // 2 +/[2] on 2x4 matrix - pairwise sums along columns (axis 2)
    // [[1,2,3,4],[5,6,7,8]] -> [[3,5,7],[11,13,15]]
    Eigen::MatrixXd mat(2, 4);
    mat << 1, 2, 3, 4,
           5, 6, 7, 8;
    Value* rhs = machine->heap->allocate_matrix(mat);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    // first_axis=false means axis 2 (columns)
    NwiseMatrixReduceK* iter = heap->allocate<NwiseMatrixReduceK>(plus_fn, rhs, 2, false, false);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);
    EXPECT_EQ(m->cols(), 3);  // 4 - 2 + 1 = 3 columns
    // Row 0: 1+2=3, 2+3=5, 3+4=7
    EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(0, 1), 5.0);
    EXPECT_DOUBLE_EQ((*m)(0, 2), 7.0);
    // Row 1: 5+6=11, 6+7=13, 7+8=15
    EXPECT_DOUBLE_EQ((*m)(1, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 13.0);
    EXPECT_DOUBLE_EQ((*m)(1, 2), 15.0);
}

TEST_F(ContinuationTest, NwiseMatrixReduceKAxis1) {
    // 2 +/[1] on 3x2 matrix - pairwise sums along rows (axis 1)
    // [[1,2],[3,4],[5,6]] -> [[4,6],[8,10]]
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2,
           3, 4,
           5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    // first_axis=true means axis 1 (rows)
    NwiseMatrixReduceK* iter = heap->allocate<NwiseMatrixReduceK>(plus_fn, rhs, 2, true, false);
    machine->push_kont(iter);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    const Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_EQ(m->rows(), 2);  // 3 - 2 + 1 = 2 rows
    EXPECT_EQ(m->cols(), 2);
    // Col 0: 1+3=4, 3+5=8
    EXPECT_DOUBLE_EQ((*m)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 8.0);
    // Col 1: 2+4=6, 4+6=10
    EXPECT_DOUBLE_EQ((*m)(0, 1), 6.0);
    EXPECT_DOUBLE_EQ((*m)(1, 1), 10.0);
}

TEST_F(ContinuationTest, NwiseMatrixReduceKMarking) {
    // Test GC marking
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;
    Value* rhs = machine->heap->allocate_matrix(mat);
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);

    NwiseMatrixReduceK* iter = heap->allocate<NwiseMatrixReduceK>(plus_fn, rhs, 2, false, false);

    machine->heap->clear_marks();
    iter->mark(machine->heap);

    EXPECT_TRUE(plus_fn->marked);
    EXPECT_TRUE(rhs->marked);
}

// ============================================================================
// ApplyAxisK tests - for axis specification (f/[k])
// ============================================================================

// Test: ApplyAxisK creates OPERATOR_CURRY with axis in axis field
TEST_F(ContinuationTest, ApplyAxisKCreatesAxisCurry) {
    // Create a derived operator (+/)
    // First need a primitive function value for +
    Value* plus_prim = heap->allocate_primitive(&prim_plus);
    Value* derived = heap->allocate_derived_operator(&op_reduce, plus_prim);

    // Set up ApplyAxisK with the derived operator
    ApplyAxisK* apply_axis = heap->allocate<ApplyAxisK>(derived);

    // Set machine->result to the axis value (e.g., 1)
    machine->result = heap->allocate_scalar(1.0);

    // Push and execute
    machine->push_kont(apply_axis);
    Value* result = machine->execute();

    // Result should be an OPERATOR_CURRY with axis in axis field
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CURRIED_FN);
    EXPECT_EQ(result->data.curried_fn->curry_type, Value::CurryType::OPERATOR_CURRY);

    // The curried function should have the derived operator and axis in axis field
    EXPECT_EQ(result->data.curried_fn->fn, derived);
    EXPECT_EQ(result->data.curried_fn->first_arg, nullptr);  // No second operand
    ASSERT_NE(result->data.curried_fn->axis, nullptr);       // Axis is in axis field
    EXPECT_TRUE(result->data.curried_fn->axis->is_scalar());
    EXPECT_DOUBLE_EQ(result->data.curried_fn->axis->as_scalar(), 1.0);
}

// Test: ApplyAxisK marks derived_op during GC
TEST_F(ContinuationTest, ApplyAxisKMarksDerivedOp) {
    Value* plus_prim = heap->allocate_primitive(&prim_plus);
    Value* derived = heap->allocate_derived_operator(&op_reduce, plus_prim);
    ApplyAxisK* apply_axis = heap->allocate<ApplyAxisK>(derived);

    // Should not crash when marking
    apply_axis->mark(heap);
}

// Test: ApplyAxisK with scan operator
TEST_F(ContinuationTest, ApplyAxisKWithScanOperator) {
    Value* plus_prim = heap->allocate_primitive(&prim_plus);
    Value* derived = heap->allocate_derived_operator(&op_scan, plus_prim);

    ApplyAxisK* apply_axis = heap->allocate<ApplyAxisK>(derived);
    machine->result = heap->allocate_scalar(2.0);  // axis 2

    machine->push_kont(apply_axis);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CURRIED_FN);
    EXPECT_EQ(result->data.curried_fn->curry_type, Value::CurryType::OPERATOR_CURRY);
    EXPECT_EQ(result->data.curried_fn->first_arg, nullptr);  // No second operand
    ASSERT_NE(result->data.curried_fn->axis, nullptr);       // Axis is in axis field
    EXPECT_DOUBLE_EQ(result->data.curried_fn->axis->as_scalar(), 2.0);
}

// ============================================================================
// IndexedAssignK Tests - Indexed assignment continuations
// ============================================================================

TEST_F(ContinuationTest, IndexedAssignKBasic) {
    // Test basic indexed assignment: A←1 2 3, A[2]←99
    // First set up the variable in environment
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* arr = machine->heap->allocate_vector(vec);
    const char* var_name = machine->string_pool.intern("TestVar");
    machine->env->define(var_name, arr);

    // Create continuations: index=2, value=99
    LiteralK* index_cont = heap->allocate<LiteralK>(2.0);
    LiteralK* value_cont = heap->allocate<LiteralK>(99.0);
    IndexedAssignK* assign_k = heap->allocate<IndexedAssignK>(var_name, index_cont, value_cont);

    machine->push_kont(assign_k);
    Value* result = machine->execute();

    // Should return the assigned value
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 99.0);

    // Verify the variable was updated
    Value* updated = machine->env->lookup(var_name);
    ASSERT_NE(updated, nullptr);
    EXPECT_TRUE(updated->is_vector());
    const Eigen::MatrixXd* m = updated->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 99.0);  // index 2 (1-based) = offset 1
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

TEST_F(ContinuationTest, IndexedAssignKFirstElement) {
    // Test assigning to first element: A[1]←100
    Eigen::VectorXd vec(4);
    vec << 10, 20, 30, 40;
    Value* arr = machine->heap->allocate_vector(vec);
    const char* var_name = machine->string_pool.intern("First");
    machine->env->define(var_name, arr);

    LiteralK* index_cont = heap->allocate<LiteralK>(1.0);
    LiteralK* value_cont = heap->allocate<LiteralK>(100.0);
    IndexedAssignK* assign_k = heap->allocate<IndexedAssignK>(var_name, index_cont, value_cont);

    machine->push_kont(assign_k);
    machine->execute();

    Value* updated = machine->env->lookup(var_name);
    const Eigen::MatrixXd* m = updated->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 100.0);  // First element updated
    EXPECT_DOUBLE_EQ((*m)(1, 0), 20.0);
}

TEST_F(ContinuationTest, IndexedAssignKLastElement) {
    // Test assigning to last element: A[4]←999
    Eigen::VectorXd vec(4);
    vec << 10, 20, 30, 40;
    Value* arr = machine->heap->allocate_vector(vec);
    const char* var_name = machine->string_pool.intern("Last");
    machine->env->define(var_name, arr);

    LiteralK* index_cont = heap->allocate<LiteralK>(4.0);
    LiteralK* value_cont = heap->allocate<LiteralK>(999.0);
    IndexedAssignK* assign_k = heap->allocate<IndexedAssignK>(var_name, index_cont, value_cont);

    machine->push_kont(assign_k);
    machine->execute();

    Value* updated = machine->env->lookup(var_name);
    const Eigen::MatrixXd* m = updated->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(3, 0), 999.0);  // Last element updated
}

TEST_F(ContinuationTest, IndexedAssignKMarking) {
    // Test that IndexedAssignK properly marks continuations for GC
    LiteralK* index_cont = heap->allocate<LiteralK>(1.0);
    LiteralK* value_cont = heap->allocate<LiteralK>(42.0);
    const char* var_name = machine->string_pool.intern("MarkTest");

    IndexedAssignK* assign_k = heap->allocate<IndexedAssignK>(var_name, index_cont, value_cont);

    machine->heap->clear_marks();
    assign_k->mark(machine->heap);

    // Both continuations should be marked
    EXPECT_TRUE(index_cont->marked);
    EXPECT_TRUE(value_cont->marked);
}

TEST_F(ContinuationTest, IndexedAssignIndexKMarking) {
    // Test IndexedAssignIndexK marking
    Value* val = machine->heap->allocate_scalar(42.0);
    LiteralK* index_cont = heap->allocate<LiteralK>(1.0);
    const char* var_name = machine->string_pool.intern("IdxMarkTest");

    IndexedAssignIndexK* idx_k = heap->allocate<IndexedAssignIndexK>(var_name, val, index_cont);

    machine->heap->clear_marks();
    idx_k->mark(machine->heap);

    EXPECT_TRUE(val->marked);
    EXPECT_TRUE(index_cont->marked);
}

TEST_F(ContinuationTest, PerformIndexedAssignKMarking) {
    // Test PerformIndexedAssignK marking
    Value* val = machine->heap->allocate_scalar(42.0);
    Value* idx = machine->heap->allocate_scalar(1.0);
    const char* var_name = machine->string_pool.intern("PerfMarkTest");

    PerformIndexedAssignK* perf_k = heap->allocate<PerformIndexedAssignK>(var_name, val, idx);

    machine->heap->clear_marks();
    perf_k->mark(machine->heap);

    EXPECT_TRUE(val->marked);
    EXPECT_TRUE(idx->marked);
}

// ============================================================================
// G_PRIME Curry Edge Case Tests
// ============================================================================

// Monadic-only function (≢) in dyadic context should error
TEST_F(ContinuationTest, GPrimeMonadicOnlyErrorsInDyadicContext) {
    Value* tally_fn = machine->heap->allocate_primitive(&prim_tally);
    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;
    Value* arr = machine->heap->allocate_vector(vec);

    Value* curried = machine->heap->allocate_curried_fn(tally_fn, arr, Value::CurryType::G_PRIME);
    Value* five = machine->heap->allocate_scalar(5.0);
    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(curried, nullptr, five);
    machine->push_kont(dispatch);

    EXPECT_THROW(machine->execute(), APLError);
}

// Pure dyadic function (=) should remain as curry, not finalize
TEST_F(ContinuationTest, GPrimeValidPartialApplicationNotFinalized) {
    Value* equal_fn = machine->heap->allocate_primitive(&prim_equal);
    Value* two = machine->heap->allocate_scalar(2.0);

    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(equal_fn, nullptr, two);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::CURRIED_FN);
    EXPECT_EQ(result->data.curried_fn->curry_type, Value::CurryType::DYADIC_CURRY);
}

// G_PRIME curry of derived operator finalizes at top level: +/ 1 2 3 4 5 = 15
TEST_F(ContinuationTest, GPrimeNestedCurryFinalization) {
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);
    Value* reduce_op = machine->heap->allocate_derived_operator(&op_reduce, plus_fn);

    Eigen::VectorXd vec(5);
    vec << 1, 2, 3, 4, 5;
    Value* iota_result = machine->heap->allocate_vector(vec);

    Value* curried = machine->heap->allocate_curried_fn(reduce_op, iota_result, Value::CurryType::G_PRIME);
    machine->result = curried;
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
}

// Overloaded function G_PRIME without second arg uses monadic: -5 = -5
TEST_F(ContinuationTest, GPrimeOverloadedUsesMonadic) {
    Value* minus_fn = machine->heap->allocate_primitive(&prim_minus);
    Value* five = machine->heap->allocate_scalar(5.0);

    Value* curried = machine->heap->allocate_curried_fn(minus_fn, five, Value::CurryType::G_PRIME);
    machine->result = curried;
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -5.0);
}

// G_PRIME curry with second arg uses dyadic: 3 - 5 = -2
TEST_F(ContinuationTest, GPrimeCurryWithSecondArgUsesDyadic) {
    Value* minus_fn = machine->heap->allocate_primitive(&prim_minus);
    Value* five = machine->heap->allocate_scalar(5.0);

    Value* curried = machine->heap->allocate_curried_fn(minus_fn, five, Value::CurryType::G_PRIME);
    Value* three = machine->heap->allocate_scalar(3.0);
    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(curried, nullptr, three);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), -2.0);
}

// All monadic functions create G_PRIME curry, finalized at top level: ⍳5 = 1 2 3 4 5
TEST_F(ContinuationTest, GPrimeAllMonadicFinalizedAtTopLevel) {
    Value* iota_fn = machine->heap->allocate_primitive(&prim_iota);
    Value* five = machine->heap->allocate_scalar(5.0);

    DispatchFunctionK* dispatch = heap->allocate<DispatchFunctionK>(iota_fn, nullptr, five, false);
    machine->push_kont(dispatch);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->rows(), 5);
}

// ============================================================================
// Curry Finalization Consistency Tests
// These test that all curry finalization paths handle all function types
// ============================================================================

// Test: PerformAssignK should finalize G_PRIME curry of CLOSURE
TEST_F(ContinuationTest, PerformAssignKFinalizesClosureCurry) {
    // Create a dfn: {⍵+1}
    Value* result = machine->eval("{⍵+1}");
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->tag, ValueType::CLOSURE);

    // Create G_PRIME curry of the closure with arg 5
    Value* five = machine->heap->allocate_scalar(5.0);
    Value* curried = machine->heap->allocate_curried_fn(result, five, Value::CurryType::G_PRIME);

    // Assign it: A←curried should finalize to 6
    machine->result = curried;
    const char* var_name = machine->string_pool.intern("TestA");
    PerformAssignK* assign = heap->allocate<PerformAssignK>(var_name);
    machine->push_kont(assign);
    result = machine->execute();

    // Should be finalized to 6, not remain as curry
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar()) << "PerformAssignK should finalize CLOSURE curry";
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
    }
}

// Test: PerformAssignK should finalize G_PRIME curry of DERIVED_OPERATOR
TEST_F(ContinuationTest, PerformAssignKFinalizesDerivedOpCurry) {
    // Create +/ derived operator
    Value* plus_fn = machine->heap->allocate_primitive(&prim_plus);
    Value* reduce_op = machine->heap->allocate_derived_operator(&op_reduce, plus_fn);

    // Create G_PRIME curry with 1 2 3 4 5
    Eigen::VectorXd vec(5);
    vec << 1, 2, 3, 4, 5;
    Value* arr = machine->heap->allocate_vector(vec);
    Value* curried = machine->heap->allocate_curried_fn(reduce_op, arr, Value::CurryType::G_PRIME);

    // Assign it: A←curried should finalize to 15
    machine->result = curried;
    const char* var_name = machine->string_pool.intern("TestB");
    PerformAssignK* assign = heap->allocate<PerformAssignK>(var_name);
    machine->push_kont(assign);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar()) << "PerformAssignK should finalize DERIVED_OPERATOR curry";
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
    }
}

// Test: EvalStrandElementK should finalize G_PRIME curry of CLOSURE in strand
TEST_F(ContinuationTest, StrandFinalizesClosureCurry) {
    // {⍵+1}5 in a strand context: 1 ({⍵+1}5) 3 should give 1 6 3
    Value* result = machine->eval("1 ({⍵+1}5) 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector()) << "Strand should finalize CLOSURE curry";
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 3);
        const Eigen::MatrixXd* m = result->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
        EXPECT_DOUBLE_EQ((*m)(1, 0), 6.0);
        EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
    }
}

// Test: Assigned reduce result should be usable in strand
TEST_F(ContinuationTest, AssignedReduceUsableInStrand) {
    // A←+/1 2 3 should store 6, then 10 A 20 should give 10 6 20
    machine->eval("A←+/1 2 3");
    Value* a_val = machine->env->lookup("A");
    ASSERT_NE(a_val, nullptr);
    EXPECT_TRUE(a_val->is_scalar()) << "Assignment should store finalized value, not curry";
    if (a_val->is_scalar()) {
        EXPECT_DOUBLE_EQ(a_val->as_scalar(), 6.0);
    }

    // Now use A in a strand - should work without N-wise reduce errors
    Value* result = machine->eval("10 A 20");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 3);
        const Eigen::MatrixXd* m = result->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(0, 0), 10.0);
        EXPECT_DOUBLE_EQ((*m)(1, 0), 6.0);
        EXPECT_DOUBLE_EQ((*m)(2, 0), 20.0);
    }
}

// Test: fn_squad (⌷) should finalize G_PRIME curry used as index
TEST_F(ContinuationTest, SquadFinalizesIndexCurry) {
    // (⍳3)⌷10 20 30 40 50 should give 10 20 30 (indices 1 2 3)
    Value* result = machine->eval("(⍳3)⌷10 20 30 40 50");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector()) << "Squad should finalize index curry";
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 3);
        const Eigen::MatrixXd* m = result->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(0, 0), 10.0);
        EXPECT_DOUBLE_EQ((*m)(1, 0), 20.0);
        EXPECT_DOUBLE_EQ((*m)(2, 0), 30.0);
    }
}

// Test: ApplyMonadicK consistency - should create G_PRIME for all monadic
TEST_F(ContinuationTest, ApplyMonadicKCreatesGPrimeForAllMonadic) {
    // ⍳5 via ApplyMonadicK path should work same as DispatchFunctionK
    // Both should create G_PRIME curry that gets finalized at top level
    Value* result = machine->eval("⍳5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->rows(), 5);
}

// Test: Assignment with execute returning curry: A←⍎'+/⍳5'
TEST_F(ContinuationTest, AssignExecuteWithCurry) {
    Value* result = machine->eval("A←⍎'+/⍳5'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar()) << "Execute returning curry should be finalized before assignment";
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);
    }

    // Verify the variable got the finalized value
    Value* a_val = machine->env->lookup("A");
    ASSERT_NE(a_val, nullptr);
    EXPECT_TRUE(a_val->is_scalar());
    if (a_val->is_scalar()) {
        EXPECT_DOUBLE_EQ(a_val->as_scalar(), 15.0);
    }
}

// Test: DYADIC_CURRY as left operand of dyadic function
// Per g': when y is a function, result is y(g1(x)) - finalize first, then apply y
TEST_F(ContinuationTest, DyadicCurryAsLeftOperand) {
    // (+/1 2 3) + 10 should give 16
    // +/1 2 3 creates DYADIC_CURRY, must be finalized to 6 before dyadic +
    Value* result = machine->eval("(+/1 2 3) + 10");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar()) << "(+/1 2 3) + 10 should produce scalar 16";
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 16.0);
    }
}

// Test: DYADIC_CURRY in the middle of a strand
TEST_F(ContinuationTest, DyadicCurryInStrand) {
    // 1 2 (+/1 2 3) 4 should give 1 2 6 4
    // The curry must be finalized before strand formation
    Value* result = machine->eval("1 2 (+/1 2 3) 4");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector()) << "Strand with curry should produce vector";
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 4);
        const Eigen::MatrixXd* m = result->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
        EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
        EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);
        EXPECT_DOUBLE_EQ((*m)(3, 0), 4.0);
    }
}

// Test: G_PRIME curry as left operand
TEST_F(ContinuationTest, GPrimeAsLeftOperand) {
    // (⍳5) + 10 should give 11 12 13 14 15
    // ⍳5 creates G_PRIME, must be finalized to 1 2 3 4 5 before dyadic +
    Value* result = machine->eval("(⍳5) + 10");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector()) << "(⍳5) + 10 should produce vector";
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 5);
        const Eigen::MatrixXd* m = result->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
        EXPECT_DOUBLE_EQ((*m)(1, 0), 12.0);
        EXPECT_DOUBLE_EQ((*m)(2, 0), 13.0);
        EXPECT_DOUBLE_EQ((*m)(3, 0), 14.0);
        EXPECT_DOUBLE_EQ((*m)(4, 0), 15.0);
    }
}

// Test: Nested reduce curries
TEST_F(ContinuationTest, NestedReduceCurries) {
    // (+/1 2 3) + (+/10 20) should give 6 + 30 = 36
    Value* result = machine->eval("(+/1 2 3) + (+/10 20)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 36.0);
    }
}

// ============================================================================
// Additional Curry Finalization Edge Case Tests
// ============================================================================

// Test: Curry as bracket index - finalized curry used as index
TEST_F(ContinuationTest, CurryAsBracketIndex) {
    // (⍳5)[+/1 2] should finalize +/1 2 to 3, then index at position 3
    Value* result = machine->eval("(⍳5)[+/1 2]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar()) << "Bracket index with curry should return scalar";
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);  // ⎕IO=1, so position 3 is value 3
    }
}

// Test: Curry in indexed assignment - curry as index position
TEST_F(ContinuationTest, CurryInIndexedAssignment) {
    // A←⍳10 ⋄ A[+/1 2 3]←99 should set A[6]←99
    machine->eval("A←⍳10");
    Value* result = machine->eval("A[+/1 2 3]←99");
    ASSERT_NE(result, nullptr);

    // Check that A[6] is now 99
    Value* a_val = machine->env->lookup("A");
    ASSERT_NE(a_val, nullptr);
    EXPECT_TRUE(a_val->is_vector());
    if (a_val->is_vector()) {
        const Eigen::MatrixXd* m = a_val->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(5, 0), 99.0);  // Index 6 (⎕IO=1) is position 5 (0-based)
    }
}

// Test: Double parentheses - nested finalization
TEST_F(ContinuationTest, DoubleParenthesesFinalization) {
    // ((+/1 2 3)) should finalize to 6
    Value* result = machine->eval("((+/1 2 3))");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar()) << "Double parentheses should finalize curry";
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
    }
}

// Test: Shape of finalized curry - monadic ⍴ on finalized reduce
TEST_F(ContinuationTest, ShapeOfFinalizedCurry) {
    // ⍴(+/1 2 3) should give ⍬ (empty shape = scalar)
    Value* result = machine->eval("⍴(+/1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector()) << "Shape should return vector";
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 0) << "Shape of scalar should be empty vector";
    }
}

// Test: Dfn applied to finalized curry
TEST_F(ContinuationTest, DfnAppliedToFinalizedCurry) {
    // {⍵+1}(+/1 2 3) should finalize curry to 6, then apply dfn to get 7
    Value* result = machine->eval("{⍵+1}(+/1 2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar()) << "Dfn should receive finalized value";
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
    }
}

// Test: Parenthesized derived function applied monadically
TEST_F(ContinuationTest, ParenthesizedDerivedFunctionMonadic) {
    // (+/)1 2 3 should apply reduce monadically (not curry)
    Value* result = machine->eval("(+/)1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar()) << "Parenthesized reduce should apply monadically";
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
    }
}

// Test: Triple nested curry finalization
TEST_F(ContinuationTest, TripleNestedCurryFinalization) {
    // (((⍳3))) should finalize to 1 2 3
    Value* result = machine->eval("(((⍳3)))");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 3);
        const Eigen::MatrixXd* m = result->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
        EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
        EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
    }
}

// Test: Curry in take operation - curry as left argument
TEST_F(ContinuationTest, CurryInTakeOperation) {
    // (+/1 2)↑10 20 30 40 50 should finalize curry to 3, take first 3 elements
    Value* result = machine->eval("(+/1 2)↑10 20 30 40 50");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 3);
        const Eigen::MatrixXd* m = result->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(0, 0), 10.0);
        EXPECT_DOUBLE_EQ((*m)(1, 0), 20.0);
        EXPECT_DOUBLE_EQ((*m)(2, 0), 30.0);
    }
}

// Test: Curry in reshape - both arguments are curries
TEST_F(ContinuationTest, CurryInReshapeBothArgs) {
    // (+/1 2)⍴(×/2 3) should be 3⍴6 = 6 6 6
    Value* result = machine->eval("(+/1 2)⍴(×/2 3)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 3);
        const Eigen::MatrixXd* m = result->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(0, 0), 6.0);
        EXPECT_DOUBLE_EQ((*m)(1, 0), 6.0);
        EXPECT_DOUBLE_EQ((*m)(2, 0), 6.0);
    }
}

// Test: Curry used as right operand in operator - reduce with curry axis
// This tests OPERATOR_CURRY path in DispatchFunctionK
TEST_F(ContinuationTest, CurryAsOperatorSecondOperand) {
    // 2 3⍴⍳6 gives matrix, then +/[1]mat reduces along axis 1
    // But first test simpler case: +/(⍳5) should reduce the finalized vector
    Value* result = machine->eval("+/(⍳5)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 15.0);  // 1+2+3+4+5
    }
}

// Test: Curry in comparison chain - finalized curries in relational ops
TEST_F(ContinuationTest, CurryInComparisonChain) {
    // (+/1 2 3) = (+/2 4) tests if 6 = 6, should give 1
    Value* result = machine->eval("(+/1 2 3) = (+/2 4)");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    if (result->is_scalar()) {
        EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // 6 = 6 → 1
    }
}

// Test: N-wise reduction should NOT finalize when legitimate curry
TEST_F(ContinuationTest, NWiseReductionNotPrematurelyFinalized) {
    // 2+/1 2 3 4 5 should do N-wise reduction (2-wise sums), not finalize +
    Value* result = machine->eval("2+/1 2 3 4 5");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector()) << "N-wise reduction should produce vector";
    if (result->is_vector()) {
        EXPECT_EQ(result->rows(), 4);  // 5-2+1 = 4 windows
        const Eigen::MatrixXd* m = result->as_matrix();
        EXPECT_DOUBLE_EQ((*m)(0, 0), 3.0);   // 1+2
        EXPECT_DOUBLE_EQ((*m)(1, 0), 5.0);   // 2+3
        EXPECT_DOUBLE_EQ((*m)(2, 0), 7.0);   // 3+4
        EXPECT_DOUBLE_EQ((*m)(3, 0), 9.0);   // 4+5
    }
}

// ============================================================================
// System Variable Tests
// ============================================================================

// Test lookup_sysvar function
TEST_F(ContinuationTest, SysVarLookupIO) {
    SysVarId id = lookup_sysvar("IO", SYSVAR_ALL);
    EXPECT_EQ(id, SysVarId::IO);
}

TEST_F(ContinuationTest, SysVarLookupPP) {
    SysVarId id = lookup_sysvar("PP", SYSVAR_ALL);
    EXPECT_EQ(id, SysVarId::PP);
}

TEST_F(ContinuationTest, SysVarLookupInvalid) {
    SysVarId id = lookup_sysvar("XY", SYSVAR_ALL);
    EXPECT_EQ(id, SysVarId::INVALID);
}

TEST_F(ContinuationTest, SysVarLookupDisabledIO) {
    // Disable IO in mask
    SysVarId id = lookup_sysvar("IO", SYSVAR_PP);  // Only PP enabled
    EXPECT_EQ(id, SysVarId::INVALID);
}

TEST_F(ContinuationTest, SysVarLookupDisabledPP) {
    // Disable PP in mask
    SysVarId id = lookup_sysvar("PP", SYSVAR_IO);  // Only IO enabled
    EXPECT_EQ(id, SysVarId::INVALID);
}

TEST_F(ContinuationTest, SysVarNameIO) {
    EXPECT_STREQ(sysvar_name(SysVarId::IO), "IO");
}

TEST_F(ContinuationTest, SysVarNamePP) {
    EXPECT_STREQ(sysvar_name(SysVarId::PP), "PP");
}

// Test SysVarReadK continuation
TEST_F(ContinuationTest, SysVarReadKIO) {
    SysVarReadK* read = heap->allocate<SysVarReadK>(SysVarId::IO);
    machine->push_kont(read);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 1.0);  // Default IO is 1
}

TEST_F(ContinuationTest, SysVarReadKPP) {
    SysVarReadK* read = heap->allocate<SysVarReadK>(SysVarId::PP);
    machine->push_kont(read);
    Value* result = machine->execute();
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);  // Default PP is 10
}

// ============================================================================
// ApplyDerivedOperatorK with DEFINED_OPERATOR Tests
// ============================================================================

TEST_F(ContinuationTest, ApplyDerivedOperatorKWithDefinedOperator) {
    // First define a monadic operator in the environment
    machine->eval("(F TWICE) ← {F F ⍵}");

    // Create the operand (a function)
    Value* operand = heap->allocate_primitive(&prim_minus);
    machine->result = operand;

    // Create ApplyDerivedOperatorK for the user-defined operator
    const char* op_name = machine->string_pool.intern("TWICE");
    ApplyDerivedOperatorK* apply_k = heap->allocate<ApplyDerivedOperatorK>(op_name);

    // Push and execute
    machine->push_kont(apply_k);
    machine->execute();

    // Result should be a DERIVED_OPERATOR value
    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_derived_operator());
    // The derived operator should have the function operand
    EXPECT_EQ(result->data.derived_op->first_operand, operand);
    // And should have the defined operator (not primitive)
    EXPECT_NE(result->data.derived_op->defined_op, nullptr);
    EXPECT_EQ(result->data.derived_op->primitive_op, nullptr);
}

TEST_F(ContinuationTest, ApplyDerivedOperatorKWithDyadicDefinedOperator) {
    // Define a dyadic operator
    machine->eval("(F COMPOSE G) ← {F G ⍵}");

    // Create the first operand (a function)
    Value* first_operand = heap->allocate_primitive(&prim_minus);
    machine->result = first_operand;

    // Create ApplyDerivedOperatorK for COMPOSE
    const char* op_name = machine->string_pool.intern("COMPOSE");
    ApplyDerivedOperatorK* apply_k = heap->allocate<ApplyDerivedOperatorK>(op_name);

    // Push and execute - should create DERIVED_OPERATOR waiting for second operand
    machine->push_kont(apply_k);
    machine->execute();

    Value* result = machine->result;
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_derived_operator());
    EXPECT_EQ(result->data.derived_op->first_operand, first_operand);
    // Should be a dyadic operator
    EXPECT_TRUE(result->data.derived_op->defined_op->is_dyadic_operator);
}

TEST_F(ContinuationTest, DerivedOperatorKWithDefinedOperatorName) {
    // Define the operator first
    machine->eval("(F TWICE) ← {F F ⍵}");

    // Create DerivedOperatorK which evaluates operand and creates derived operator
    LookupK* operand_cont = heap->allocate<LookupK>(machine->string_pool.intern("-"));
    const char* op_name = machine->string_pool.intern("TWICE");
    DerivedOperatorK* derived_k = heap->allocate<DerivedOperatorK>(operand_cont, op_name);

    machine->push_kont(derived_k);
    Value* result = machine->execute();

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_derived_operator());
    // First operand should be the minus function
    EXPECT_TRUE(result->data.derived_op->first_operand->is_primitive());
    // Should have defined operator, not primitive
    EXPECT_NE(result->data.derived_op->defined_op, nullptr);
}

// =============================================================================
// Error Stack Trace Tests
// =============================================================================

TEST_F(ContinuationTest, ErrorStackTracePopulatedOnError) {
    // Trigger an error by looking up undefined variable
    EXPECT_THROW(machine->eval("undefined_var"), apl::APLError);

    // After error, error_stack should be populated
    EXPECT_FALSE(machine->error_stack.empty());
}

TEST_F(ContinuationTest, ErrorStackTraceContainsContinuationsWithLocations) {
    // Define a dfn that will error (must use ⍵ to avoid niladic auto-invocation)
    machine->eval("f ← {⍵+undefined}");

    // Now call the function to trigger error
    EXPECT_THROW(machine->eval("f 1"), apl::APLError);

    // Check that error_stack has continuations
    EXPECT_FALSE(machine->error_stack.empty());

    // At least one continuation should have a source location
    bool found_location = false;
    for (Continuation* cont : machine->error_stack) {
        if (cont->has_location()) {
            found_location = true;
            break;
        }
    }
    EXPECT_TRUE(found_location) << "No continuation in error stack has source location";
}

TEST_F(ContinuationTest, FormatStackTraceReturnsNonEmptyOnError) {
    // Trigger an error
    EXPECT_THROW(machine->eval("bad_var"), apl::APLError);

    // format_stack_trace should return something
    std::string trace = machine->format_stack_trace();
    EXPECT_FALSE(trace.empty());
    EXPECT_NE(trace.find("Stack trace"), std::string::npos);
}

TEST_F(ContinuationTest, FormatStackTraceIncludesContinuationDescriptions) {
    // Trigger an error via undefined variable
    EXPECT_THROW(machine->eval("unknown"), apl::APLError);

    std::string trace = machine->format_stack_trace();

    // Should contain continuation type names
    // LookupK is created for variable lookup
    EXPECT_NE(trace.find("LookupK"), std::string::npos)
        << "Stack trace should show LookupK for variable lookup. Trace:\n" << trace;
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
