// Parser tests

#include <gtest/gtest.h>
#include "parser.h"
#include "machine.h"
#include "continuation.h"
#include "environment.h"

using namespace apl;

class ParserTest : public ::testing::Test {
protected:
    Machine* machine;
    Parser* parser;

    void SetUp() override {
        machine = new Machine();
        init_global_environment(machine->env);  // Initialize built-in operators
        parser = new Parser(machine->heap);
    }

    void TearDown() override {
        delete parser;
        delete machine;
    }

    // Helper: evaluate parsed continuation using the CEK machine
    Value* eval(Continuation* k) {
        // The parsed continuation graph needs to be evaluated
        // We push HaltK first so execution terminates properly
        machine->push_kont(machine->heap->allocate_continuation(new HaltK()));

        // For literals with nullptr next, set to pop from stack
        // For BinOpK, it handles its own evaluation
        wrap_literals(k);

        // Invoke the root continuation to kick off evaluation
        // This will recursively evaluate and use the machine's continuation stack
        Value* result = k->invoke(machine);

        return result;
    }

    void wrap_literals(Continuation* k) {
        if (auto* lit = dynamic_cast<LiteralK*>(k)) {
            if (!lit->next) {
                // LiteralK with no next should pop from kont_stack
                // For now just set it to HaltK since our tests don't use the stack yet
                lit->next = machine->heap->allocate_continuation(new HaltK());
            }
        } else if (auto* binop = dynamic_cast<BinOpK*>(k)) {
            wrap_literals(binop->left);
            wrap_literals(binop->right);
        }
    }
};

// Test parsing a simple literal
TEST_F(ParserTest, ParseLiteral) {
    Continuation* k = parser->parse("42");

    ASSERT_NE(k, nullptr);
    EXPECT_EQ(parser->get_error(), "");

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::SCALAR);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test parsing a negative literal
TEST_F(ParserTest, ParseNegativeLiteral) {
    Continuation* k = parser->parse("-3.14");

    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -3.14);
}

// Test parsing zero
TEST_F(ParserTest, ParseZero) {
    Continuation* k = parser->parse("0");

    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

// Test parsing failure with invalid input
TEST_F(ParserTest, ParseInvalidInput) {
    Continuation* k = parser->parse("abc");

    EXPECT_EQ(k, nullptr);
    EXPECT_NE(parser->get_error(), "");
}

// Test parse-time safety: no Values allocated during parse
TEST_F(ParserTest, ParseTimeSafety) {
    size_t values_before = machine->heap->total_size();

    Continuation* k = parser->parse("123");

    size_t values_after = machine->heap->total_size();

    // NO Values should be allocated during parsing
    EXPECT_EQ(values_before, values_after);

    // Values are only created when we EVALUATE
    eval(k);

    size_t values_after_invoke = machine->heap->total_size();
    EXPECT_GT(values_after_invoke, values_before);
}

// Test parsing simple addition
TEST_F(ParserTest, ParseAddition) {
    Continuation* k = parser->parse("2 + 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test right-to-left evaluation: 2 + 3 × 4 = 2 + 12 = 14
TEST_F(ParserTest, ParseRightToLeft) {
    Continuation* k = parser->parse("2 + 3 * 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);  // NOT 20!
}

// Test with APL symbols: 2 + 3 × 4
TEST_F(ParserTest, ParseAPLSymbols) {
    Continuation* k = parser->parse("2 + 3 × 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test division
TEST_F(ParserTest, ParseDivision) {
    Continuation* k = parser->parse("10 ÷ 2");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test subtraction
TEST_F(ParserTest, ParseSubtraction) {
    Continuation* k = parser->parse("10 - 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 7.0);
}

// Test longer chain: 1 + 2 + 3 + 4 = 1 + 2 + 7 = 1 + 9 = 10 (right-to-left)
TEST_F(ParserTest, ParseLongChain) {
    Continuation* k = parser->parse("1 + 2 + 3 + 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test mixed operators: 10 - 2 * 3 = 10 - 6 = 4
TEST_F(ParserTest, ParseMixedOperators) {
    Continuation* k = parser->parse("10 - 2 * 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 4.0);
}

// ============================================================================
// Parenthesized Expression Tests (Phase 3.2.1)
// ============================================================================

// Test simple parenthesized literal
TEST_F(ParserTest, ParseParenthesizedLiteral) {
    Continuation* k = parser->parse("(42)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

// Test parenthesized expression
TEST_F(ParserTest, ParseParenthesizedExpression) {
    Continuation* k = parser->parse("(2 + 3)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

// Test parentheses change evaluation order: 2 × (3 + 4) = 2 × 7 = 14
TEST_F(ParserTest, ParseParenthesesPrecedence) {
    Continuation* k = parser->parse("2 * (3 + 4)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);
}

// Test without parentheses for comparison: 2 × 3 + 4 = 2 × (3 + 4) = 2 × 7 = 14 (right-to-left)
TEST_F(ParserTest, ParseWithoutParentheses) {
    Continuation* k = parser->parse("2 * 3 + 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 14.0);  // 2 * (3 + 4) right-to-left
}

// Test nested parentheses: ((2 + 3) × 4) = 5 × 4 = 20
TEST_F(ParserTest, ParseNestedParentheses) {
    Continuation* k = parser->parse("((2 + 3) * 4)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 20.0);
}

// Test complex nested expression: (10 - (2 + 3)) × 2 = (10 - 5) × 2 = 5 × 2 = 10
TEST_F(ParserTest, ParseComplexNested) {
    Continuation* k = parser->parse("(10 - (2 + 3)) * 2");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test multiple parenthesized subexpressions: (1 + 2) + (3 + 4) = 3 + 7 = 10
TEST_F(ParserTest, ParseMultipleParentheses) {
    Continuation* k = parser->parse("(1 + 2) + (3 + 4)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 10.0);
}

// Test parentheses with division: (12 ÷ 2) ÷ 3 = 6 ÷ 3 = 2
TEST_F(ParserTest, ParseParenthesesDivision) {
    Continuation* k = parser->parse("(12 ÷ 2) ÷ 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 2.0);
}

// Test unclosed parenthesis error
TEST_F(ParserTest, ParseUnclosedParenthesis) {
    Continuation* k = parser->parse("(2 + 3");

    EXPECT_EQ(k, nullptr);
    EXPECT_NE(parser->get_error(), "");
}

// Test unmatched closing parenthesis
TEST_F(ParserTest, ParseUnmatchedClosingParen) {
    Continuation* k = parser->parse("2 + 3)");

    EXPECT_EQ(k, nullptr);
    EXPECT_NE(parser->get_error(), "");
}

// Test empty parentheses error
TEST_F(ParserTest, ParseEmptyParentheses) {
    Continuation* k = parser->parse("()");

    EXPECT_EQ(k, nullptr);
    EXPECT_NE(parser->get_error(), "");
}

// ============================================================================
// Array Strand Tests (Phase 3.2.2)
// ============================================================================

// Test simple 2-element strand
TEST_F(ParserTest, ParseSimpleStrand) {
    Continuation* k = parser->parse("1 2");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 2);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
}

// Test 3-element strand
TEST_F(ParserTest, ParseThreeElementStrand) {
    Continuation* k = parser->parse("1 2 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

// Test longer strand
TEST_F(ParserTest, ParseLongerStrand) {
    Continuation* k = parser->parse("10 20 30 40 50");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 5);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 10.0);
    EXPECT_DOUBLE_EQ((*m)(4, 0), 50.0);
}

// Test strand with negative numbers (using parentheses for negatives)
TEST_F(ParserTest, ParseStrandWithNegatives) {
    Continuation* k = parser->parse("(-1) 2 (-3)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), -1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), -3.0);
}

// Test strand with parenthesized expressions
TEST_F(ParserTest, ParseStrandWithParens) {
    Continuation* k = parser->parse("(1+1) 3 (2*2)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 2.0);   // 1+1
    EXPECT_DOUBLE_EQ((*m)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 4.0);   // 2*2
}

// Test strand with decimals
TEST_F(ParserTest, ParseStrandWithDecimals) {
    Continuation* k = parser->parse("1.5 2.25 3.75");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.5);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.25);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.75);
}

// Test strand as operand to binary operation: (1 2 3) + 10 = 11 12 13
TEST_F(ParserTest, ParseStrandAsLeftOperand) {
    Continuation* k = parser->parse("1 2 3 + 10");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 13.0);
}

// Test strand as right operand: 10 + (1 2 3) = 11 12 13
TEST_F(ParserTest, ParseStrandAsRightOperand) {
    Continuation* k = parser->parse("10 + 1 2 3");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 11.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 12.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 13.0);
}

// Test two strands: (1 2) + (3 4) = 4 6
TEST_F(ParserTest, ParseTwoStrands) {
    Continuation* k = parser->parse("1 2 + 3 4");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 2);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 4.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 6.0);
}

// Test parenthesized strand: (1 2 3) should still be a strand
TEST_F(ParserTest, ParseParenthesizedStrand) {
    Continuation* k = parser->parse("(1 2 3)");
    ASSERT_NE(k, nullptr);

    Value* result = eval(k);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->tag, ValueType::VECTOR);
    EXPECT_EQ(result->rows(), 3);
    Eigen::MatrixXd* m = result->as_matrix();
    EXPECT_DOUBLE_EQ((*m)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*m)(1, 0), 2.0);
    EXPECT_DOUBLE_EQ((*m)(2, 0), 3.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
