// Completion records tests

#include <gtest/gtest.h>
#include "completion.h"
#include "control.h"
#include "value.h"

using namespace apl;

class CompletionTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Clean up any allocated completions
    }
};

// Test basic NORMAL completion
TEST_F(CompletionTest, NormalCompletion) {
    Value* v = Value::from_scalar(42.0);
    APLCompletion* comp = APLCompletion::normal(v);

    EXPECT_TRUE(comp->is_normal());
    EXPECT_FALSE(comp->is_abrupt());
    EXPECT_FALSE(comp->is_return());
    EXPECT_FALSE(comp->is_break());
    EXPECT_FALSE(comp->is_continue());
    EXPECT_FALSE(comp->is_throw());

    EXPECT_EQ(comp->type, CompletionType::NORMAL);
    EXPECT_EQ(comp->value, v);
    EXPECT_EQ(comp->target, nullptr);

    delete comp;
    delete v;
}

// Test RETURN completion
TEST_F(CompletionTest, ReturnCompletion) {
    Value* v = Value::from_scalar(99.0);
    APLCompletion* comp = APLCompletion::return_value(v);

    EXPECT_FALSE(comp->is_normal());
    EXPECT_TRUE(comp->is_abrupt());
    EXPECT_TRUE(comp->is_return());
    EXPECT_FALSE(comp->is_break());
    EXPECT_FALSE(comp->is_continue());
    EXPECT_FALSE(comp->is_throw());

    EXPECT_EQ(comp->type, CompletionType::RETURN);
    EXPECT_EQ(comp->value, v);
    EXPECT_EQ(comp->target, nullptr);

    delete comp;
    delete v;
}

// Test BREAK completion without label
TEST_F(CompletionTest, BreakWithoutLabel) {
    APLCompletion* comp = APLCompletion::break_completion();

    EXPECT_FALSE(comp->is_normal());
    EXPECT_TRUE(comp->is_abrupt());
    EXPECT_TRUE(comp->is_break());
    EXPECT_FALSE(comp->is_return());
    EXPECT_FALSE(comp->is_continue());
    EXPECT_FALSE(comp->is_throw());

    EXPECT_EQ(comp->type, CompletionType::BREAK);
    EXPECT_EQ(comp->value, nullptr);
    EXPECT_EQ(comp->target, nullptr);

    delete comp;
}

// Test BREAK completion with label
TEST_F(CompletionTest, BreakWithLabel) {
    const char* label = "outer_loop";
    APLCompletion* comp = APLCompletion::break_completion(label);

    EXPECT_TRUE(comp->is_break());
    EXPECT_EQ(comp->target, label);
    EXPECT_TRUE(comp->matches_target("outer_loop"));
    EXPECT_FALSE(comp->matches_target("inner_loop"));
    EXPECT_FALSE(comp->matches_target(nullptr));

    delete comp;
}

// Test CONTINUE completion
TEST_F(CompletionTest, ContinueCompletion) {
    APLCompletion* comp = APLCompletion::continue_completion();

    EXPECT_FALSE(comp->is_normal());
    EXPECT_TRUE(comp->is_abrupt());
    EXPECT_TRUE(comp->is_continue());
    EXPECT_FALSE(comp->is_break());
    EXPECT_FALSE(comp->is_return());
    EXPECT_FALSE(comp->is_throw());

    EXPECT_EQ(comp->type, CompletionType::CONTINUE);
    EXPECT_EQ(comp->value, nullptr);

    delete comp;
}

// Test CONTINUE completion with label
TEST_F(CompletionTest, ContinueWithLabel) {
    const char* label = "my_loop";
    APLCompletion* comp = APLCompletion::continue_completion(label);

    EXPECT_TRUE(comp->is_continue());
    EXPECT_TRUE(comp->matches_target("my_loop"));
    EXPECT_FALSE(comp->matches_target("other_loop"));

    delete comp;
}

// Test THROW completion
TEST_F(CompletionTest, ThrowCompletion) {
    Value* error = Value::from_scalar(-1.0);  // Error indicator
    APLCompletion* comp = APLCompletion::throw_error(error);

    EXPECT_FALSE(comp->is_normal());
    EXPECT_TRUE(comp->is_abrupt());
    EXPECT_TRUE(comp->is_throw());
    EXPECT_FALSE(comp->is_break());
    EXPECT_FALSE(comp->is_continue());
    EXPECT_FALSE(comp->is_return());

    EXPECT_EQ(comp->type, CompletionType::THROW);
    EXPECT_EQ(comp->value, error);

    delete comp;
    delete error;
}

// Test direct constructor
TEST_F(CompletionTest, DirectConstructor) {
    Value* v = Value::from_scalar(7.0);
    APLCompletion comp(CompletionType::NORMAL, v, nullptr);

    EXPECT_TRUE(comp.is_normal());
    EXPECT_EQ(comp.value, v);

    delete v;
}

// Test default constructor
TEST_F(CompletionTest, DefaultConstructor) {
    APLCompletion comp;

    EXPECT_TRUE(comp.is_normal());
    EXPECT_EQ(comp.value, nullptr);
    EXPECT_EQ(comp.target, nullptr);
}

// Test all completion types are distinct
TEST_F(CompletionTest, AllTypesDistinct) {
    APLCompletion* normal = APLCompletion::normal(nullptr);
    APLCompletion* ret = APLCompletion::return_value(nullptr);
    APLCompletion* brk = APLCompletion::break_completion();
    APLCompletion* cont = APLCompletion::continue_completion();
    APLCompletion* thr = APLCompletion::throw_error(nullptr);

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

    delete normal;
    delete ret;
    delete brk;
    delete cont;
    delete thr;
}

// Control class tests
class ControlTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Control manages its own cleanup
    }
};

// Test Control default construction
TEST_F(ControlTest, DefaultConstruction) {
    Control ctrl;

    EXPECT_EQ(ctrl.mode, ExecMode::HALTED);
    EXPECT_EQ(ctrl.lexer_state, nullptr);
    EXPECT_EQ(ctrl.value, nullptr);
    EXPECT_EQ(ctrl.completion, nullptr);
}

// Test Control set_value
TEST_F(ControlTest, SetValue) {
    Control ctrl;
    Value* v = Value::from_scalar(123.0);

    ctrl.set_value(v);

    EXPECT_EQ(ctrl.value, v);
    ASSERT_NE(ctrl.completion, nullptr);
    EXPECT_TRUE(ctrl.completion->is_normal());
    EXPECT_EQ(ctrl.completion->value, v);

    delete v;
}

// Test Control set_completion
TEST_F(ControlTest, SetCompletion) {
    Control ctrl;
    APLCompletion* comp = APLCompletion::return_value(nullptr);

    ctrl.set_completion(comp);

    EXPECT_EQ(ctrl.completion, comp);
    EXPECT_TRUE(ctrl.completion->is_return());
}

// Test Control set_completion replaces old
TEST_F(ControlTest, SetCompletionReplaces) {
    Control ctrl;
    APLCompletion* comp1 = APLCompletion::normal(nullptr);
    APLCompletion* comp2 = APLCompletion::return_value(nullptr);

    ctrl.set_completion(comp1);
    EXPECT_EQ(ctrl.completion, comp1);

    // Setting new completion should delete old one
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
    ctrl.set_completion(APLCompletion::normal(nullptr));

    EXPECT_TRUE(ctrl.should_continue());
}

// Test Control should_continue when halted
TEST_F(ControlTest, ShouldContinueHalted) {
    Control ctrl;
    ctrl.mode = ExecMode::HALTED;
    ctrl.set_completion(APLCompletion::normal(nullptr));

    EXPECT_FALSE(ctrl.should_continue());
}

// Test Control should_continue with abrupt completion
TEST_F(ControlTest, ShouldContinueAbrupt) {
    Control ctrl;
    ctrl.mode = ExecMode::EVALUATING;
    ctrl.set_completion(APLCompletion::return_value(nullptr));

    EXPECT_FALSE(ctrl.should_continue());
}

// Test Control has_abrupt_completion
TEST_F(ControlTest, HasAbruptCompletion) {
    Control ctrl;

    ctrl.set_completion(APLCompletion::normal(nullptr));
    EXPECT_FALSE(ctrl.has_abrupt_completion());

    ctrl.set_completion(APLCompletion::return_value(nullptr));
    EXPECT_TRUE(ctrl.has_abrupt_completion());

    ctrl.set_completion(APLCompletion::break_completion());
    EXPECT_TRUE(ctrl.has_abrupt_completion());
}

// Test Control init_evaluating
TEST_F(ControlTest, InitEvaluating) {
    Control ctrl;
    ctrl.mode = ExecMode::HALTED;

    ctrl.init_evaluating();

    EXPECT_EQ(ctrl.mode, ExecMode::EVALUATING);
    ASSERT_NE(ctrl.completion, nullptr);
    EXPECT_TRUE(ctrl.completion->is_normal());
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
