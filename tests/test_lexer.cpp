// Lexer tests

#include <gtest/gtest.h>
#include "lexer.h"
#include "token.h"

using namespace apl;

class LexerTest : public ::testing::Test {
protected:
    Lexer* lexer_ = nullptr;

    void TearDown() override {
        if (lexer_) {
            delete lexer_;
            lexer_ = nullptr;
        }
    }

    // Helper to tokenize entire string
    // Keeps lexer alive so name pointers remain valid
    std::vector<Token> tokenize(const char* input) {
        if (lexer_) delete lexer_;
        lexer_ = new Lexer(input);
        std::vector<Token> tokens;

        Token tok;
        do {
            tok = lexer_->next_token();
            tokens.push_back(tok);
        } while (tok.type != TOK_EOF && tok.type != TOK_ERROR);

        return tokens;
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

// Test all control flow keywords
TEST_F(LexerTest, AllControlFlowKeywords) {
    auto tokens = tokenize(":If :Else :ElseIf :EndIf :While :EndWhile :For :EndFor");

    EXPECT_EQ(tokens[0].type, TOK_IF);
    EXPECT_EQ(tokens[1].type, TOK_ELSE);
    EXPECT_EQ(tokens[2].type, TOK_ELSEIF);
    EXPECT_EQ(tokens[3].type, TOK_ENDIF);
    EXPECT_EQ(tokens[4].type, TOK_WHILE);
    EXPECT_EQ(tokens[5].type, TOK_ENDWHILE);
    EXPECT_EQ(tokens[6].type, TOK_FOR);
    EXPECT_EQ(tokens[7].type, TOK_ENDFOR);
}

// Test diamond statement separator
TEST_F(LexerTest, Diamond) {
    auto tokens = tokenize("a ⋄ b");

    EXPECT_EQ(tokens[0].type, TOK_NAME);
    EXPECT_EQ(tokens[1].type, TOK_DIAMOND);
    EXPECT_EQ(tokens[2].type, TOK_NAME);
}

// Test comments
TEST_F(LexerTest, Comments) {
    auto tokens = tokenize("a ⍝ this is a comment\nb");

    // Should skip comment
    ASSERT_EQ(tokens.size(), 4);  // a, newline, b, EOF
    EXPECT_EQ(tokens[0].type, TOK_NAME);
    EXPECT_STREQ(tokens[0].name, "a");
    EXPECT_EQ(tokens[1].type, TOK_NEWLINE);
    EXPECT_EQ(tokens[2].type, TOK_NAME);
    EXPECT_STREQ(tokens[2].name, "b");
}

// Test more APL symbols
TEST_F(LexerTest, MoreAPLSymbols) {
    auto tokens = tokenize("⍉ ↑ ↓ ⌿ ⍀ ¨ ∘ ⍨ →");

    EXPECT_EQ(tokens[0].type, TOK_TRANSPOSE);
    EXPECT_EQ(tokens[1].type, TOK_TAKE);
    EXPECT_EQ(tokens[2].type, TOK_DROP);
    EXPECT_EQ(tokens[3].type, TOK_REDUCE_FIRST);
    EXPECT_EQ(tokens[4].type, TOK_SCAN_FIRST);
    EXPECT_EQ(tokens[5].type, TOK_EACH);
    EXPECT_EQ(tokens[6].type, TOK_COMPOSE);
    EXPECT_EQ(tokens[7].type, TOK_COMMUTE);
    EXPECT_EQ(tokens[8].type, TOK_GOTO);
}

// Test brackets and braces
TEST_F(LexerTest, BracketsAndBraces) {
    auto tokens = tokenize("[ ] { } ; ( )");

    EXPECT_EQ(tokens[0].type, TOK_LBRACKET);
    EXPECT_EQ(tokens[1].type, TOK_RBRACKET);
    EXPECT_EQ(tokens[2].type, TOK_LBRACE);
    EXPECT_EQ(tokens[3].type, TOK_RBRACE);
    EXPECT_EQ(tokens[4].type, TOK_SEMICOLON);
    EXPECT_EQ(tokens[5].type, TOK_LPAREN);
    EXPECT_EQ(tokens[6].type, TOK_RPAREN);
}

// Test ravel
TEST_F(LexerTest, Ravel) {
    auto tokens = tokenize(",1 2 3");

    EXPECT_EQ(tokens[0].type, TOK_RAVEL);
    EXPECT_EQ(tokens[1].type, TOK_NUMBER);
}

// Test negative numbers
TEST_F(LexerTest, NegativeNumbers) {
    auto tokens = tokenize("-42 -3.14");

    EXPECT_EQ(tokens[0].type, TOK_MINUS);
    EXPECT_EQ(tokens[1].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[1].number, 42.0);
    EXPECT_EQ(tokens[2].type, TOK_MINUS);
    EXPECT_EQ(tokens[3].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[3].number, 3.14);
}

// Test mixed expression
TEST_F(LexerTest, MixedExpression) {
    auto tokens = tokenize("result ← +/⍳10");

    ASSERT_GE(tokens.size(), 6);
    EXPECT_EQ(tokens[0].type, TOK_NAME);
    EXPECT_STREQ(tokens[0].name, "result");
    EXPECT_EQ(tokens[1].type, TOK_ASSIGN);
    EXPECT_EQ(tokens[2].type, TOK_PLUS);
    EXPECT_EQ(tokens[3].type, TOK_REDUCE);
    EXPECT_EQ(tokens[4].type, TOK_IOTA);
    EXPECT_EQ(tokens[5].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[5].number, 10.0);
}

// Test whitespace handling
TEST_F(LexerTest, Whitespace) {
    auto tokens = tokenize("  a   b  \t  c  ");

    ASSERT_EQ(tokens.size(), 4);  // 3 names + EOF
    EXPECT_EQ(tokens[0].type, TOK_NAME);
    EXPECT_EQ(tokens[1].type, TOK_NAME);
    EXPECT_EQ(tokens[2].type, TOK_NAME);
}

// Test empty input
TEST_F(LexerTest, EmptyInput) {
    auto tokens = tokenize("");

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].type, TOK_EOF);
}

// Test only whitespace
TEST_F(LexerTest, OnlyWhitespace) {
    auto tokens = tokenize("   \t\t   ");

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].type, TOK_EOF);
}

// Test scientific notation edge cases
TEST_F(LexerTest, ScientificNotation) {
    auto tokens = tokenize("1e5 2E-3 3.5e+10");

    ASSERT_EQ(tokens.size(), 4);
    EXPECT_DOUBLE_EQ(tokens[0].number, 1e5);
    EXPECT_DOUBLE_EQ(tokens[1].number, 2E-3);
    EXPECT_DOUBLE_EQ(tokens[2].number, 3.5e+10);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
