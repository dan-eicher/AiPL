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

// Test basic number recognition (use newlines to separate)
TEST_F(LexerTest, Numbers) {
    auto tokens = tokenize("42\n3.14\n1.5e10\n2.5e-3");

    ASSERT_EQ(tokens.size(), 8);  // 4 numbers + 3 newlines + EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[0].number, 42.0);

    EXPECT_EQ(tokens[1].type, TOK_NEWLINE);

    EXPECT_EQ(tokens[2].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[2].number, 3.14);

    EXPECT_EQ(tokens[3].type, TOK_NEWLINE);

    EXPECT_EQ(tokens[4].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[4].number, 1.5e10);

    EXPECT_EQ(tokens[5].type, TOK_NEWLINE);

    EXPECT_EQ(tokens[6].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[6].number, 2.5e-3);

    EXPECT_EQ(tokens[7].type, TOK_EOF);
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
    auto tokens = tokenize("(a + b) [x] {y}");

    EXPECT_EQ(tokens[0].type, TOK_LPAREN);
    EXPECT_EQ(tokens[4].type, TOK_RPAREN);
    EXPECT_EQ(tokens[5].type, TOK_LBRACKET);
    EXPECT_EQ(tokens[6].type, TOK_NAME);
    EXPECT_EQ(tokens[7].type, TOK_RBRACKET);
    EXPECT_EQ(tokens[8].type, TOK_LBRACE);
    EXPECT_EQ(tokens[9].type, TOK_NAME);
    EXPECT_EQ(tokens[10].type, TOK_RBRACE);
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

// Test reduction expression (now with vector literal)
TEST_F(LexerTest, ReductionExpression) {
    auto tokens = tokenize("+/1 2 3 4");

    ASSERT_EQ(tokens.size(), 4);  // + / VECTOR EOF
    EXPECT_EQ(tokens[0].type, TOK_PLUS);
    EXPECT_EQ(tokens[1].type, TOK_REDUCE);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[2].vector_size, 4);
    EXPECT_DOUBLE_EQ(tokens[2].vector_data[0], 1.0);
    EXPECT_DOUBLE_EQ(tokens[2].vector_data[1], 2.0);
    EXPECT_DOUBLE_EQ(tokens[2].vector_data[2], 3.0);
    EXPECT_DOUBLE_EQ(tokens[2].vector_data[3], 4.0);
    EXPECT_EQ(tokens[3].type, TOK_EOF);
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

// Test arithmetic extension operators
TEST_F(LexerTest, ArithmeticExtensionOperators) {
    auto tokens = tokenize("| ⍟ !");

    EXPECT_EQ(tokens[0].type, TOK_STILE);
    EXPECT_EQ(tokens[1].type, TOK_LOG);
    EXPECT_EQ(tokens[2].type, TOK_FACTORIAL);
}

// Test outer product
TEST_F(LexerTest, OuterProduct) {
    auto tokens = tokenize("∘.×");

    EXPECT_EQ(tokens[0].type, TOK_OUTER_PRODUCT);
    EXPECT_EQ(tokens[1].type, TOK_TIMES);
}

// Test set function operators
TEST_F(LexerTest, SetFunctionOperators) {
    auto tokens = tokenize("∪ ~");

    ASSERT_EQ(tokens.size(), 3);  // ∪, ~, EOF
    EXPECT_EQ(tokens[0].type, TOK_UNION);
    EXPECT_EQ(tokens[1].type, TOK_NOT);
}

// Test rank operator
TEST_F(LexerTest, RankOperator) {
    auto tokens = tokenize("+⍤2");

    ASSERT_EQ(tokens.size(), 4);  // +, ⍤, 2, EOF
    EXPECT_EQ(tokens[0].type, TOK_PLUS);
    EXPECT_EQ(tokens[1].type, TOK_RANK);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER);
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

// Test inner product operator (dot)
TEST_F(LexerTest, InnerProduct) {
    auto tokens = tokenize("+.×");

    EXPECT_EQ(tokens[0].type, TOK_PLUS);
    EXPECT_EQ(tokens[1].type, TOK_DOT);
    EXPECT_EQ(tokens[2].type, TOK_TIMES);
}

// Test that +. without following operator produces correct tokens
TEST_F(LexerTest, DotAlone) {
    auto tokens = tokenize("+.");

    // Should be TOK_PLUS, TOK_DOT
    EXPECT_EQ(tokens[0].type, TOK_PLUS);
    EXPECT_EQ(tokens[1].type, TOK_DOT);
}

// Test diamond statement separator
TEST_F(LexerTest, Diamond) {
    auto tokens = tokenize("a ⋄ b");

    EXPECT_EQ(tokens[0].type, TOK_NAME);
    EXPECT_EQ(tokens[1].type, TOK_DIAMOND);
    EXPECT_EQ(tokens[2].type, TOK_NAME);
}

// Test column tracking with UTF-8 characters
TEST_F(LexerTest, UTF8ColumnTracking) {
    auto tokens = tokenize("× ÷ ⍴");

    // × is at column 0 (2 bytes in UTF-8)
    // space at byte 2
    // ÷ is at column 2 (should account for × being 1 char, not 2 bytes)
    // space at byte 5
    // ⍴ is at column 4
    EXPECT_EQ(tokens[0].column, 0);  // ×
    EXPECT_EQ(tokens[1].column, 2);  // ÷ (× took 1 column + 1 space)
    EXPECT_EQ(tokens[2].column, 4);  // ⍴ (× ÷ took 2 columns + 2 spaces)
    EXPECT_EQ(tokens[3].type, TOK_EOF);
}

// Test column tracking with keywords
TEST_F(LexerTest, KeywordColumnTracking) {
    auto tokens = tokenize(":EndWhile x");

    // :EndWhile is 9 characters
    // space at 9
    // x at column 10
    EXPECT_EQ(tokens[0].column, 0);   // :EndWhile
    EXPECT_EQ(tokens[1].column, 10);  // x (after :EndWhile + space)
}

// Test column tracking after comments
TEST_F(LexerTest, CommentColumnTracking) {
    auto tokens = tokenize("x ⍝ comment\ny");

    // x at column 0, line 1
    // newline after comment
    // y at column 0, line 2
    EXPECT_EQ(tokens[0].type, TOK_NAME);
    EXPECT_EQ(tokens[0].column, 0);
    EXPECT_EQ(tokens[0].line, 1);

    EXPECT_EQ(tokens[1].type, TOK_NEWLINE);
    EXPECT_EQ(tokens[1].line, 1);

    EXPECT_EQ(tokens[2].type, TOK_NAME);
    EXPECT_EQ(tokens[2].column, 0);
    EXPECT_EQ(tokens[2].line, 2);
}

// Test outer product is single token (not compose + dot)
TEST_F(LexerTest, OuterProductNotTwoTokens) {
    auto tokens = tokenize("∘.");

    // Should be ONE token (TOK_OUTER_PRODUCT), not two (TOK_COMPOSE + error)
    ASSERT_EQ(tokens.size(), 2);  // TOK_OUTER_PRODUCT + TOK_EOF
    EXPECT_EQ(tokens[0].type, TOK_OUTER_PRODUCT);
    EXPECT_EQ(tokens[1].type, TOK_EOF);
}

// Test compose alone still works
TEST_F(LexerTest, ComposeSeparate) {
    auto tokens = tokenize("∘");

    ASSERT_EQ(tokens.size(), 2);  // TOK_COMPOSE + TOK_EOF
    EXPECT_EQ(tokens[0].type, TOK_COMPOSE);
    EXPECT_EQ(tokens[1].type, TOK_EOF);
}

// Test string literals
TEST_F(LexerTest, StringLiterals) {
    auto tokens = tokenize("'hello' 'world'");

    ASSERT_EQ(tokens.size(), 3);  // 'hello', 'world', EOF
    EXPECT_EQ(tokens[0].type, TOK_STRING);
    EXPECT_STREQ(tokens[0].name, "hello");
    EXPECT_EQ(tokens[1].type, TOK_STRING);
    EXPECT_STREQ(tokens[1].name, "world");
}

// Test string literal with escaped quotes
TEST_F(LexerTest, StringLiteralsWithEscaping) {
    auto tokens = tokenize("'don''t'");

    ASSERT_EQ(tokens.size(), 2);  // 'don't', EOF
    EXPECT_EQ(tokens[0].type, TOK_STRING);
    EXPECT_STREQ(tokens[0].name, "don't");
}

// Test empty string
TEST_F(LexerTest, EmptyString) {
    auto tokens = tokenize("''");

    ASSERT_EQ(tokens.size(), 2);  // '', EOF
    EXPECT_EQ(tokens[0].type, TOK_STRING);
    EXPECT_STREQ(tokens[0].name, "");
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

// Test ravel (now with vector literal)
TEST_F(LexerTest, Ravel) {
    auto tokens = tokenize(",1 2 3");

    ASSERT_EQ(tokens.size(), 3);  // , VECTOR EOF
    EXPECT_EQ(tokens[0].type, TOK_RAVEL);
    EXPECT_EQ(tokens[1].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[1].vector_size, 3);
    EXPECT_EQ(tokens[2].type, TOK_EOF);
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

// Test scientific notation edge cases (use newlines to separate)
TEST_F(LexerTest, ScientificNotation) {
    auto tokens = tokenize("1e5\n2E-3\n3.5e+10");

    ASSERT_EQ(tokens.size(), 6);  // 3 numbers + 2 newlines + EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[0].number, 1e5);
    EXPECT_EQ(tokens[1].type, TOK_NEWLINE);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[2].number, 2E-3);
    EXPECT_EQ(tokens[3].type, TOK_NEWLINE);
    EXPECT_EQ(tokens[4].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[4].number, 3.5e+10);
    EXPECT_EQ(tokens[5].type, TOK_EOF);
}

// Test dfn tokens (braces and alpha/omega)
TEST_F(LexerTest, DfnTokens) {
    auto tokens = tokenize("{⍺+⍵}");

    ASSERT_EQ(tokens.size(), 6);  // { ⍺ + ⍵ } EOF
    EXPECT_EQ(tokens[0].type, TOK_LBRACE);
    EXPECT_EQ(tokens[1].type, TOK_ALPHA);
    EXPECT_EQ(tokens[2].type, TOK_PLUS);
    EXPECT_EQ(tokens[3].type, TOK_OMEGA);
    EXPECT_EQ(tokens[4].type, TOK_RBRACE);
    EXPECT_EQ(tokens[5].type, TOK_EOF);
}

// Test numeric vector literal (ISO 13751)
TEST_F(LexerTest, NumericVectorLiteral) {
    auto tokens = tokenize("1 2 3");

    ASSERT_EQ(tokens.size(), 2);  // VECTOR + EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[0].vector_size, 3);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[0], 1.0);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[1], 2.0);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[2], 3.0);
    EXPECT_EQ(tokens[1].type, TOK_EOF);
}

// Test numeric vector with decimals
TEST_F(LexerTest, NumericVectorDecimals) {
    auto tokens = tokenize("1.5 2.25 3.75");

    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[0].vector_size, 3);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[0], 1.5);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[1], 2.25);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[2], 3.75);
}

// Test numeric vector with scientific notation
TEST_F(LexerTest, NumericVectorScientific) {
    auto tokens = tokenize("1e5 2.5e-3 3.14e+2");

    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[0].vector_size, 3);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[0], 1e5);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[1], 2.5e-3);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[2], 3.14e+2);
}

// Test single number is NOT a vector
TEST_F(LexerTest, SingleNumberNotVector) {
    auto tokens = tokenize("42");

    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TOK_NUMBER);  // NOT TOK_NUMBER_VECTOR
    EXPECT_DOUBLE_EQ(tokens[0].number, 42.0);
}

// Test vector literal followed by operator
TEST_F(LexerTest, VectorLiteralWithOperator) {
    auto tokens = tokenize("1 2 3+4 5 6");

    ASSERT_EQ(tokens.size(), 4);  // VECTOR + VECTOR EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[0].vector_size, 3);
    EXPECT_EQ(tokens[1].type, TOK_PLUS);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[2].vector_size, 3);
}

// Test operator followed by vector (no space)
TEST_F(LexerTest, OperatorAdjacentToVector) {
    auto tokens = tokenize("+/1 2 3");

    // Should tokenize as: + / VECTOR EOF
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[0].type, TOK_PLUS);
    EXPECT_EQ(tokens[1].type, TOK_REDUCE);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[2].vector_size, 3);
    EXPECT_EQ(tokens[3].type, TOK_EOF);
}

// ============================================================================
// High Minus (¯) Tests
// ============================================================================

TEST_F(LexerTest, HighMinusScalar) {
    auto tokens = tokenize("¯3.14");

    ASSERT_EQ(tokens.size(), 2);  // NUMBER + EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[0].number, -3.14);
}

TEST_F(LexerTest, HighMinusInteger) {
    auto tokens = tokenize("¯42");

    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[0].number, -42.0);
}

TEST_F(LexerTest, HighMinusInVector) {
    auto tokens = tokenize("1 ¯2 3 ¯4.5");

    ASSERT_EQ(tokens.size(), 2);  // VECTOR + EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[0].vector_size, 4);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[0], 1.0);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[1], -2.0);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[2], 3.0);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[3], -4.5);
}

TEST_F(LexerTest, HighMinusMixedWithOperators) {
    auto tokens = tokenize("¯5+3");

    ASSERT_EQ(tokens.size(), 4);  // NUMBER + NUMBER + EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[0].number, -5.0);
    EXPECT_EQ(tokens[1].type, TOK_PLUS);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[2].number, 3.0);
}

TEST_F(LexerTest, HighMinusExponent) {
    auto tokens = tokenize("¯1.5e¯3");

    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TOK_NUMBER);
    // ¯1.5e¯3 should be -1.5e-3 = -0.0015
    EXPECT_DOUBLE_EQ(tokens[0].number, -1.5e-3);
}

// Test high minus in exponent within vector literals
TEST_F(LexerTest, HighMinusExponentInVector) {
    auto tokens = tokenize("1e2 ¯3e¯4 5e¯6");

    ASSERT_EQ(tokens.size(), 2);  // TOK_NUMBER_VECTOR + EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER_VECTOR);
    ASSERT_EQ(tokens[0].vector_size, 3);
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[0], 1e2);      // 100
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[1], -3e-4);    // -0.0003
    EXPECT_DOUBLE_EQ(tokens[0].vector_data[2], 5e-6);     // 0.000005
}

// Test reverse/rotate/tally tokens
TEST_F(LexerTest, ReverseRotateTally) {
    auto tokens = tokenize("⌽ ⊖ ≢");

    ASSERT_EQ(tokens.size(), 4);  // 3 tokens + EOF
    EXPECT_EQ(tokens[0].type, TOK_REVERSE);
    EXPECT_EQ(tokens[1].type, TOK_REVERSE_FIRST);
    EXPECT_EQ(tokens[2].type, TOK_TALLY);
}

// Test member of token (∊)
TEST_F(LexerTest, MemberOf) {
    auto tokens = tokenize("∊");

    ASSERT_EQ(tokens.size(), 2);  // 1 token + EOF
    EXPECT_EQ(tokens[0].type, TOK_MEMBER);
}

// Test grade tokens (⍋ ⍒)
TEST_F(LexerTest, GradeUpDown) {
    auto tokens = tokenize("⍋ ⍒");

    ASSERT_EQ(tokens.size(), 3);  // 2 tokens + EOF
    EXPECT_EQ(tokens[0].type, TOK_GRADE_UP);
    EXPECT_EQ(tokens[1].type, TOK_GRADE_DOWN);
}

// Test circular/random tokens (○ ?)
TEST_F(LexerTest, CircularRandom) {
    auto tokens = tokenize("○ ?");

    ASSERT_EQ(tokens.size(), 3);  // 2 tokens + EOF
    EXPECT_EQ(tokens[0].type, TOK_CIRCLE);
    EXPECT_EQ(tokens[1].type, TOK_QUESTION);
}

// Test encode/decode tokens (⊥ ⊤)
TEST_F(LexerTest, EncodeDecode) {
    auto tokens = tokenize("⊥ ⊤");

    ASSERT_EQ(tokens.size(), 3);  // 2 tokens + EOF
    EXPECT_EQ(tokens[0].type, TOK_DECODE);
    EXPECT_EQ(tokens[1].type, TOK_ENCODE);
}

// Test matrix/execute tokens (⌹ ⍎)
TEST_F(LexerTest, MatrixExecute) {
    auto tokens = tokenize("⌹ ⍎");

    ASSERT_EQ(tokens.size(), 3);  // 2 tokens + EOF
    EXPECT_EQ(tokens[0].type, TOK_DOMINO);
    EXPECT_EQ(tokens[1].type, TOK_EXECUTE);
}

// Test reserved nested array tokens (⊂ ⊃) and depth (≡)
TEST_F(LexerTest, NestedArrayAndDepthTokens) {
    auto tokens = tokenize("⊂ ⊃ ≡");

    ASSERT_EQ(tokens.size(), 4);  // 3 tokens + EOF
    EXPECT_EQ(tokens[0].type, TOK_ENCLOSE);   // Reserved (nested arrays)
    EXPECT_EQ(tokens[1].type, TOK_DISCLOSE);  // Reserved (nested arrays)
    EXPECT_EQ(tokens[2].type, TOK_MATCH);     // Implemented: depth
}

// String literal tests
TEST_F(LexerTest, StringLiteralSimple) {
    auto tokens = tokenize("'hello'");
    ASSERT_EQ(tokens.size(), 2);  // string + EOF
    EXPECT_EQ(tokens[0].type, TOK_STRING);
    EXPECT_STREQ(tokens[0].name, "hello");
}

TEST_F(LexerTest, StringLiteralEmpty) {
    auto tokens = tokenize("''");
    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TOK_STRING);
    EXPECT_STREQ(tokens[0].name, "");
}

TEST_F(LexerTest, StringLiteralWithSpaces) {
    auto tokens = tokenize("'hello world'");
    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TOK_STRING);
    EXPECT_STREQ(tokens[0].name, "hello world");
}

TEST_F(LexerTest, StringLiteralEscapedQuote) {
    // APL uses '' inside string to represent a single quote
    auto tokens = tokenize("'it''s'");
    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TOK_STRING);
    EXPECT_STREQ(tokens[0].name, "it's");
}

TEST_F(LexerTest, ExecuteToken) {
    auto tokens = tokenize("⍎");
    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TOK_EXECUTE);
}

// Test table token (⍪)
TEST_F(LexerTest, TableToken) {
    auto tokens = tokenize("⍪");
    ASSERT_EQ(tokens.size(), 2);  // ⍪ + EOF
    EXPECT_EQ(tokens[0].type, TOK_TABLE);
}

// Test zilde token (⍬) - empty vector
TEST_F(LexerTest, ZildeToken) {
    auto tokens = tokenize("⍬");
    ASSERT_EQ(tokens.size(), 2);  // ⍬ + EOF
    EXPECT_EQ(tokens[0].type, TOK_ZILDE);
}

// Test zilde in expressions
TEST_F(LexerTest, ZildeInExpression) {
    auto tokens = tokenize("x←⍬");
    ASSERT_EQ(tokens.size(), 4);  // x ← ⍬ EOF
    EXPECT_EQ(tokens[0].type, TOK_NAME);
    EXPECT_EQ(tokens[1].type, TOK_ASSIGN);
    EXPECT_EQ(tokens[2].type, TOK_ZILDE);
    EXPECT_EQ(tokens[3].type, TOK_EOF);
}

// Test format token (⍕)
TEST_F(LexerTest, FormatToken) {
    auto tokens = tokenize("⍕");
    ASSERT_EQ(tokens.size(), 2);  // ⍕ + EOF
    EXPECT_EQ(tokens[0].type, TOK_FORMAT);
}

// Test format in expression
TEST_F(LexerTest, FormatInExpression) {
    auto tokens = tokenize("⍕42");
    ASSERT_EQ(tokens.size(), 3);  // ⍕ 42 EOF
    EXPECT_EQ(tokens[0].type, TOK_FORMAT);
    EXPECT_EQ(tokens[1].type, TOK_NUMBER);
    EXPECT_DOUBLE_EQ(tokens[1].number, 42.0);
}

// Test dyadic format expression
TEST_F(LexerTest, DyadicFormatExpression) {
    auto tokens = tokenize("5 2⍕3.14");
    ASSERT_EQ(tokens.size(), 4);  // VECTOR ⍕ NUMBER EOF
    EXPECT_EQ(tokens[0].type, TOK_NUMBER_VECTOR);
    EXPECT_EQ(tokens[0].vector_size, 2);
    EXPECT_EQ(tokens[1].type, TOK_FORMAT);
    EXPECT_EQ(tokens[2].type, TOK_NUMBER);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
