// Continuation tests

#include <gtest/gtest.h>
#include "continuation.h"
#include "heap.h"
#include "machine.h"
#include "value.h"
#include "primitives.h"
#include "operators.h"
#include "environment.h"

using namespace apl;

class ContinuationTest : public ::testing::Test {
protected:
    Machine* machine;
    APLHeap* heap;  // Convenience pointer to machine->heap

    void SetUp() override {
        machine = new Machine();
        heap = machine->heap;  // Use machine's heap
        init_global_environment(machine);  // Initialize built-in primitives and operators
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

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
