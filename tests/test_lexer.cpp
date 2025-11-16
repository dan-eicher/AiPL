// Lexer tests

#include <gtest/gtest.h>
#include "lexer.h"
#include "lexer_arena.h"
#include "token.h"

using namespace apl;

class LexerTest : public ::testing::Test {
protected:
    LexerArena arena;

    // Helper to tokenize entire string
    std::vector<Token> tokenize(const char* input) {
        LexerState* state = lexer_init(input, &arena);
        std::vector<Token> tokens;

        Token tok;
        do {
            tok = lex_next_token(state);
            tokens.push_back(tok);
        } while (tok.type != TOK_EOF && tok.type != TOK_ERROR);

        lexer_free(state);
        return tokens;
    }

    void TearDown() override {
        arena.reset();
    }
};

// Test basic number recognition
TEST_F(LexerTest, Numbers) {
    auto tokens = tokenize("42 3.14 1.5e10 2.5e-3");

    ASSERT_EQ(tokens.size(), 5);  // 4 numbers + EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[0].number, 42.0);

    EXPECT_EQ(tokens[1].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[1].number, 3.14);

    EXPECT_EQ(tokens[2].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[2].number, 1.5e10);

    EXPECT_EQ(tokens[3].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[3].number, 2.5e-3);

    EXPECT_EQ(tokens[4].type, TOK_EOF);
}

// Test name recognition
TEST_F(LexerTest, Names) {
    auto tokens = tokenize("foo bar_baz x1 _test");

    ASSERT_EQ(tokens.size(), 5);  // 4 names + EOF
    EXPECT_EQ(tokens[0].type, TOK_NAME);
    EXPECT_STREQ(tokens[0].name, "foo");

    EXPECT_EQ(tokens[1].type, TOK_NAME);
    EXPECT_STREQ(tokens[1].name, "bar_baz");

    EXPECT_EQ(tokens[2].type, TOK_NAME);
    EXPECT_STREQ(tokens[2].name, "x1");

    EXPECT_EQ(tokens[3].type, TOK_NAME);
    EXPECT_STREQ(tokens[3].name, "_test");
}

// Test basic operators
TEST_F(LexerTest, BasicOperators) {
    auto tokens = tokenize("+ - * / \\");

    ASSERT_EQ(tokens.size(), 6);
    EXPECT_EQ(tokens[0].type, TOK_PLUS);
    EXPECT_EQ(tokens[1].type, TOK_MINUS);
    EXPECT_EQ(tokens[2].type, TOK_POWER);
    EXPECT_EQ(tokens[3].type, TOK_REDUCE);
    EXPECT_EQ(tokens[4].type, TOK_SCAN);
    EXPECT_EQ(tokens[5].type, TOK_EOF);
}

// Test APL Unicode symbols
TEST_F(LexerTest, APLSymbols) {
    auto tokens = tokenize("× ÷ ⍴ ⍳");

    ASSERT_EQ(tokens.size(), 5);
    EXPECT_EQ(tokens[0].type, TOK_TIMES);
    EXPECT_EQ(tokens[1].type, TOK_DIVIDE);
    EXPECT_EQ(tokens[2].type, TOK_RESHAPE);
    EXPECT_EQ(tokens[3].type, TOK_IOTA);
}

// Test assignment and arrow
TEST_F(LexerTest, Assignment) {
    auto tokens = tokenize("x ← 5");

    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[0].type, TOK_NAME);
    EXPECT_EQ(tokens[1].type, TOK_ASSIGN);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER);
    EXPECT_EQ(tokens[3].type, TOK_EOF);
}

// Test parentheses and brackets
TEST_F(LexerTest, Delimiters) {
    auto tokens = tokenize("(a + b) [1 2 3] {x}");

    EXPECT_EQ(tokens[0].type, TOK_LPAREN);
    EXPECT_EQ(tokens[4].type, TOK_RPAREN);
    EXPECT_EQ(tokens[5].type, TOK_LBRACKET);
    EXPECT_EQ(tokens[9].type, TOK_RBRACKET);
    EXPECT_EQ(tokens[10].type, TOK_LBRACE);
    EXPECT_EQ(tokens[12].type, TOK_RBRACE);
}

// Test control flow keywords
TEST_F(LexerTest, ControlFlowKeywords) {
    auto tokens = tokenize(":If :While :For :Leave :Return");

    ASSERT_EQ(tokens.size(), 6);
    EXPECT_EQ(tokens[0].type, TOK_IF);
    EXPECT_EQ(tokens[1].type, TOK_WHILE);
    EXPECT_EQ(tokens[2].type, TOK_FOR);
    EXPECT_EQ(tokens[3].type, TOK_LEAVE);
    EXPECT_EQ(tokens[4].type, TOK_RETURN);
}

// Test simple expression
TEST_F(LexerTest, SimpleExpression) {
    auto tokens = tokenize("3 + 4 × 5");

    ASSERT_EQ(tokens.size(), 6);
    EXPECT_EQ(tokens[0].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[0].number, 3.0);
    EXPECT_EQ(tokens[1].type, TOK_PLUS);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[2].number, 4.0);
    EXPECT_EQ(tokens[3].type, TOK_TIMES);
    EXPECT_EQ(tokens[4].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[4].number, 5.0);
    EXPECT_EQ(tokens[5].type, TOK_EOF);
}

// Test reduction expression
TEST_F(LexerTest, ReductionExpression) {
    auto tokens = tokenize("+/1 2 3 4");

    EXPECT_EQ(tokens[0].type, TOK_PLUS);
    EXPECT_EQ(tokens[1].type, TOK_REDUCE);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER);
    EXPECT_EQ(tokens[3].type, TOK_NUMBER);
    EXPECT_EQ(tokens[4].type, TOK_NUMBER);
    EXPECT_EQ(tokens[5].type, TOK_NUMBER);
}

// Test comparison operators
TEST_F(LexerTest, ComparisonOperators) {
    auto tokens = tokenize("= ≠ < ≤ > ≥");

    EXPECT_EQ(tokens[0].type, TOK_EQUAL);
    EXPECT_EQ(tokens[1].type, TOK_NOT_EQUAL);
    EXPECT_EQ(tokens[2].type, TOK_LESS);
    EXPECT_EQ(tokens[3].type, TOK_LESS_EQUAL);
    EXPECT_EQ(tokens[4].type, TOK_GREATER);
    EXPECT_EQ(tokens[5].type, TOK_GREATER_EQUAL);
}

// Test logical operators
TEST_F(LexerTest, LogicalOperators) {
    auto tokens = tokenize("∧ ∨ ~");

    EXPECT_EQ(tokens[0].type, TOK_AND);
    EXPECT_EQ(tokens[1].type, TOK_OR);
    EXPECT_EQ(tokens[2].type, TOK_NOT);
}

// Test outer product
TEST_F(LexerTest, OuterProduct) {
    auto tokens = tokenize("∘.×");

    EXPECT_EQ(tokens[0].type, TOK_OUTER_PRODUCT);
    EXPECT_EQ(tokens[1].type, TOK_TIMES);
}

// Test line tracking
TEST_F(LexerTest, LineTracking) {
    auto tokens = tokenize("a\nb\nc");

    EXPECT_EQ(tokens[0].line, 1);
    EXPECT_EQ(tokens[1].type, TOK_NEWLINE);
    EXPECT_EQ(tokens[2].line, 2);
    EXPECT_EQ(tokens[3].type, TOK_NEWLINE);
    EXPECT_EQ(tokens[4].line, 3);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
