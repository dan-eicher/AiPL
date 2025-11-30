// Completion records tests

#include <gtest/gtest.h>
#include "completion.h"
#include "control.h"
#include "value.h"
#include "heap.h"

using namespace apl;

class CompletionTest : public ::testing::Test {
protected:
    APLHeap* heap;

    void SetUp() override {
        heap = new APLHeap();
    }

    void TearDown() override {
        delete heap;
    }
};

// Test basic NORMAL completion (nullptr optimization)
TEST_F(CompletionTest, NormalCompletion) {
    // NORMAL completions are represented by nullptr now
    APLCompletion* comp = nullptr;

    // We can also test using direct construction for edge cases
    Value* v = heap->allocate_scalar(42.0);
    APLCompletion* explicit_normal = heap->allocate<APLCompletion>(CompletionType::NORMAL, v, nullptr);

    EXPECT_TRUE(explicit_normal->is_normal());
    EXPECT_FALSE(explicit_normal->is_abrupt());
    EXPECT_FALSE(explicit_normal->is_return());
    EXPECT_FALSE(explicit_normal->is_break());
    EXPECT_FALSE(explicit_normal->is_continue());
    EXPECT_FALSE(explicit_normal->is_throw());

    EXPECT_EQ(explicit_normal->type, CompletionType::NORMAL);
    EXPECT_EQ(explicit_normal->value, v);
    EXPECT_EQ(explicit_normal->target, nullptr);
}

// Test RETURN completion
TEST_F(CompletionTest, ReturnCompletion) {
    Value* v = heap->allocate_scalar(99.0);
    APLCompletion* comp = heap->allocate<APLCompletion>(CompletionType::RETURN, v, nullptr);

    EXPECT_FALSE(comp->is_normal());
    EXPECT_TRUE(comp->is_abrupt());
    EXPECT_TRUE(comp->is_return());
    EXPECT_FALSE(comp->is_break());
    EXPECT_FALSE(comp->is_continue());
    EXPECT_FALSE(comp->is_throw());

    EXPECT_EQ(comp->type, CompletionType::RETURN);
    EXPECT_EQ(comp->value, v);
    EXPECT_EQ(comp->target, nullptr);
}

// Test BREAK completion without label
TEST_F(CompletionTest, BreakWithoutLabel) {
    APLCompletion* comp = heap->allocate<APLCompletion>(CompletionType::BREAK, nullptr, nullptr);

    EXPECT_FALSE(comp->is_normal());
    EXPECT_TRUE(comp->is_abrupt());
    EXPECT_TRUE(comp->is_break());
    EXPECT_FALSE(comp->is_return());
    EXPECT_FALSE(comp->is_continue());
    EXPECT_FALSE(comp->is_throw());

    EXPECT_EQ(comp->type, CompletionType::BREAK);
    EXPECT_EQ(comp->value, nullptr);
    EXPECT_EQ(comp->target, nullptr);
}

// Test BREAK completion with label
TEST_F(CompletionTest, BreakWithLabel) {
    const char* label = "outer_loop";
    APLCompletion* comp = heap->allocate<APLCompletion>(CompletionType::BREAK, nullptr, label);

    EXPECT_TRUE(comp->is_break());
    EXPECT_EQ(comp->target, label);
    EXPECT_TRUE(comp->matches_target("outer_loop"));
    EXPECT_FALSE(comp->matches_target("inner_loop"));
    EXPECT_FALSE(comp->matches_target(nullptr));
}

// Test CONTINUE completion
TEST_F(CompletionTest, ContinueCompletion) {
    APLCompletion* comp = heap->allocate<APLCompletion>(CompletionType::CONTINUE, nullptr, nullptr);

    EXPECT_FALSE(comp->is_normal());
    EXPECT_TRUE(comp->is_abrupt());
    EXPECT_TRUE(comp->is_continue());
    EXPECT_FALSE(comp->is_break());
    EXPECT_FALSE(comp->is_return());
    EXPECT_FALSE(comp->is_throw());

    EXPECT_EQ(comp->type, CompletionType::CONTINUE);
    EXPECT_EQ(comp->value, nullptr);
}

// Test CONTINUE completion with label
TEST_F(CompletionTest, ContinueWithLabel) {
    const char* label = "my_loop";
    APLCompletion* comp = heap->allocate<APLCompletion>(CompletionType::CONTINUE, nullptr, label);

    EXPECT_TRUE(comp->is_continue());
    EXPECT_TRUE(comp->matches_target("my_loop"));
    EXPECT_FALSE(comp->matches_target("other_loop"));
}

// Test THROW completion
TEST_F(CompletionTest, ThrowCompletion) {
    Value* error = heap->allocate_scalar(-1.0);  // Error indicator
    APLCompletion* comp = heap->allocate<APLCompletion>(CompletionType::THROW, error, nullptr);

    EXPECT_FALSE(comp->is_normal());
    EXPECT_TRUE(comp->is_abrupt());
    EXPECT_TRUE(comp->is_throw());
    EXPECT_FALSE(comp->is_break());
    EXPECT_FALSE(comp->is_continue());
    EXPECT_FALSE(comp->is_return());

    EXPECT_EQ(comp->type, CompletionType::THROW);
    EXPECT_EQ(comp->value, error);
}

// Test heap allocation with parameters
TEST_F(CompletionTest, HeapAllocationWithParams) {
    Value* v = heap->allocate_scalar(7.0);
    APLCompletion* comp = heap->allocate<APLCompletion>(CompletionType::NORMAL, v, nullptr);

    EXPECT_TRUE(comp->is_normal());
    EXPECT_EQ(comp->value, v);
}

// Test heap allocation default parameters
TEST_F(CompletionTest, HeapAllocationDefault) {
    APLCompletion* comp = heap->allocate<APLCompletion>();

    EXPECT_TRUE(comp->is_normal());
    EXPECT_EQ(comp->value, nullptr);
    EXPECT_EQ(comp->target, nullptr);
}

// Test all completion types are distinct
TEST_F(CompletionTest, AllTypesDistinct) {
    APLCompletion* normal = heap->allocate<APLCompletion>(CompletionType::NORMAL, nullptr, nullptr);
    APLCompletion* ret = heap->allocate<APLCompletion>(CompletionType::RETURN, nullptr, nullptr);
    APLCompletion* brk = heap->allocate<APLCompletion>(CompletionType::BREAK, nullptr, nullptr);
    APLCompletion* cont = heap->allocate<APLCompletion>(CompletionType::CONTINUE, nullptr, nullptr);
    APLCompletion* thr = heap->allocate<APLCompletion>(CompletionType::THROW, nullptr, nullptr);

    // Each should only match its own type
    EXPECT_TRUE(normal->is_normal());
    EXPECT_FALSE(normal->is_return() || normal->is_break() ||
                 normal->is_continue() || normal->is_throw());

    EXPECT_TRUE(ret->is_return());
    EXPECT_FALSE(ret->is_normal() || ret->is_break() ||
                 ret->is_continue() || ret->is_throw());

    EXPECT_TRUE(brk->is_break());
    EXPECT_FALSE(brk->is_normal() || brk->is_return() ||
                 brk->is_continue() || brk->is_throw());

    EXPECT_TRUE(cont->is_continue());
    EXPECT_FALSE(cont->is_normal() || cont->is_return() ||
                 cont->is_break() || cont->is_throw());

    EXPECT_TRUE(thr->is_throw());
    EXPECT_FALSE(thr->is_normal() || thr->is_return() ||
                 thr->is_break() || thr->is_continue());
}

// Control class tests
class ControlTest : public ::testing::Test {
protected:
    APLHeap* heap;

    void SetUp() override {
        heap = new APLHeap();
    }

    void TearDown() override {
        delete heap;
    }
};

// Test Control default construction
TEST_F(ControlTest, DefaultConstruction) {
    Control ctrl;

    EXPECT_EQ(ctrl.mode, ExecMode::HALTED);
    EXPECT_EQ(ctrl.value, nullptr);
    EXPECT_EQ(ctrl.completion, nullptr);
}

// Test Control set_value
TEST_F(ControlTest, SetValue) {
    Control ctrl;
    Value* v = heap->allocate_scalar(123.0);

    ctrl.set_value(v);

    EXPECT_EQ(ctrl.value, v);
    // set_value sets completion to nullptr (NORMAL)
    EXPECT_EQ(ctrl.completion, nullptr);
}

// Test Control set_completion
TEST_F(ControlTest, SetCompletion) {
    Control ctrl;
    APLCompletion* comp = heap->allocate<APLCompletion>(CompletionType::RETURN, nullptr, nullptr);

    ctrl.set_completion(comp);

    EXPECT_EQ(ctrl.completion, comp);
    EXPECT_TRUE(ctrl.completion->is_return());
}

// Test Control set_completion replaces old
TEST_F(ControlTest, SetCompletionReplaces) {
    Control ctrl;
    APLCompletion* comp1 = heap->allocate<APLCompletion>(CompletionType::NORMAL, nullptr, nullptr);
    APLCompletion* comp2 = heap->allocate<APLCompletion>(CompletionType::RETURN, nullptr, nullptr);

    ctrl.set_completion(comp1);
    EXPECT_EQ(ctrl.completion, comp1);

    // Setting new completion replaces old one (GC will clean up)
    ctrl.set_completion(comp2);
    EXPECT_EQ(ctrl.completion, comp2);
    EXPECT_TRUE(ctrl.completion->is_return());
}

// Test Control halt
TEST_F(ControlTest, Halt) {
    Control ctrl;
    ctrl.mode = ExecMode::EVALUATING;

    ctrl.halt();

    EXPECT_EQ(ctrl.mode, ExecMode::HALTED);
}

// Test Control should_continue with normal completion
TEST_F(ControlTest, ShouldContinueNormal) {
    Control ctrl;
    ctrl.mode = ExecMode::EVALUATING;
    ctrl.set_completion(nullptr);  // nullptr = NORMAL

    EXPECT_TRUE(ctrl.should_continue());
}

// Test Control should_continue when halted
TEST_F(ControlTest, ShouldContinueHalted) {
    Control ctrl;
    ctrl.mode = ExecMode::HALTED;
    ctrl.set_completion(nullptr);  // nullptr = NORMAL

    EXPECT_FALSE(ctrl.should_continue());
}

// Test Control should_continue with abrupt completion
TEST_F(ControlTest, ShouldContinueAbrupt) {
    Control ctrl;
    ctrl.mode = ExecMode::EVALUATING;
    ctrl.set_completion(heap->allocate<APLCompletion>(CompletionType::RETURN, nullptr, nullptr));

    EXPECT_FALSE(ctrl.should_continue());
}

// Test Control has_abrupt_completion
TEST_F(ControlTest, HasAbruptCompletion) {
    Control ctrl;

    ctrl.set_completion(nullptr);  // nullptr = NORMAL
    EXPECT_FALSE(ctrl.has_abrupt_completion());

    ctrl.set_completion(heap->allocate<APLCompletion>(CompletionType::RETURN, nullptr, nullptr));
    EXPECT_TRUE(ctrl.has_abrupt_completion());

    ctrl.set_completion(heap->allocate<APLCompletion>(CompletionType::BREAK, nullptr, nullptr));
    EXPECT_TRUE(ctrl.has_abrupt_completion());
}

// Test Control init_evaluating
TEST_F(ControlTest, InitEvaluating) {
    Control ctrl;
    ctrl.mode = ExecMode::HALTED;

    ctrl.init_evaluating();

    EXPECT_EQ(ctrl.mode, ExecMode::EVALUATING);
    // init_evaluating sets completion to nullptr (NORMAL)
    EXPECT_EQ(ctrl.completion, nullptr);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
