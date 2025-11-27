// Token - Lexer token representation

#pragma once

namespace apl {

// Token types for APL lexer
enum TokenType {
    TOK_EOF,
    TOK_NUMBER,
    TOK_NAME,

    // Arithmetic operators
    TOK_PLUS,           // +
    TOK_MINUS,          // -
    TOK_TIMES,          // ×
    TOK_DIVIDE,         // ÷
    TOK_POWER,          // *

    // Array operators
    TOK_RESHAPE,        // ⍴
    TOK_RAVEL,          // ,
    TOK_TRANSPOSE,      // ⍉
    TOK_IOTA,           // ⍳
    TOK_TAKE,           // ↑
    TOK_DROP,           // ↓

    // Reduction and scan
    TOK_REDUCE,         // /
    TOK_REDUCE_FIRST,   // ⌿
    TOK_SCAN,           // \\ (backslash)
    TOK_SCAN_FIRST,     // ⍀

    // Higher-order operators
    TOK_EACH,           // ¨
    TOK_COMPOSE,        // ∘
    TOK_COMMUTE,        // ⍨
    TOK_OUTER_PRODUCT,  // ∘.

    // Comparison
    TOK_EQUAL,          // =
    TOK_NOT_EQUAL,      // ≠
    TOK_LESS,           // <
    TOK_LESS_EQUAL,     // ≤
    TOK_GREATER,        // >
    TOK_GREATER_EQUAL,  // ≥

    // Logical
    TOK_AND,            // ∧
    TOK_OR,             // ∨
    TOK_NOT,            // ~

    // Structural
    TOK_ASSIGN,         // ←
    TOK_LPAREN,         // (
    TOK_RPAREN,         // )
    TOK_LBRACKET,       // [
    TOK_RBRACKET,       // ]
    TOK_SEMICOLON,      // ;

    // Control flow
    TOK_IF,             // :If
    TOK_ELSE,           // :Else
    TOK_ELSEIF,         // :ElseIf
    TOK_ENDIF,          // :EndIf
    TOK_WHILE,          // :While
    TOK_ENDWHILE,       // :EndWhile
    TOK_FOR,            // :For
    TOK_IN,             // :In
    TOK_ENDFOR,         // :EndFor
    TOK_LEAVE,          // :Leave
    TOK_RETURN,         // :Return
    TOK_GOTO,           // →

    // Dfn delimiters
    TOK_LBRACE,         // {
    TOK_RBRACE,         // }

    // Special
    TOK_DIAMOND,        // ⋄ (statement separator)
    TOK_NEWLINE,
    TOK_COMMENT,

    TOK_ERROR           // Lexer error
};

// Token structure
struct Token {
    TokenType type;

    union {
        double number;      // For TOK_NUMBER
        const char* name;   // For TOK_NAME (points into LexerArena)
    };

    // Source location for error reporting
    int line;
    int column;

    // Constructors
    Token() : type(TOK_EOF), number(0.0), line(0), column(0) {}

    Token(TokenType t, int l = 0, int c = 0)
        : type(t), number(0.0), line(l), column(c) {}

    Token(double num, int l = 0, int c = 0)
        : type(TOK_NUMBER), number(num), line(l), column(c) {}

    Token(TokenType t, const char* n, int l = 0, int c = 0)
        : type(t), name(n), line(l), column(c) {}
};

// Helper function to convert token type to string (for debugging/error messages)
const char* token_type_name(TokenType type);

} // namespace apl
