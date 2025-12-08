// Token - Lexer token representation

#pragma once

#include <cstddef>  // for size_t

namespace apl {

// Token types for APL lexer
enum TokenType {
    TOK_EOF,
    TOK_NUMBER,
    TOK_NUMBER_VECTOR,  // Numeric vector literal "1 2 3" (ISO 13751)
    TOK_NAME,
    TOK_STRING,         // String literal 'hello'

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
    TOK_RANK,           // ⍤
    TOK_DOT,            // . (inner product)
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
    TOK_NAND,           // ⍲
    TOK_NOR,            // ⍱

    // Min/Max
    TOK_CEILING,        // ⌈
    TOK_FLOOR,          // ⌊

    // Arithmetic extensions
    TOK_STILE,          // | (magnitude/residue)
    TOK_LOG,            // ⍟ (logarithm)
    TOK_FACTORIAL,      // ! (factorial/binomial)

    // Reverse/Rotate
    TOK_REVERSE,        // ⌽ (reverse/rotate)
    TOK_REVERSE_FIRST,  // ⊖ (reverse first/rotate first)
    TOK_TALLY,          // ≢ (tally - count along first axis)

    // Search
    TOK_MEMBER,         // ∊ (member of)

    // Grade
    TOK_GRADE_UP,       // ⍋ (grade up)
    TOK_GRADE_DOWN,     // ⍒ (grade down)

    // Set functions
    TOK_UNION,          // ∪ (unique/union)

    // Circular/Random
    TOK_CIRCLE,         // ○ (pi times / circular functions)
    TOK_QUESTION,       // ? (roll / deal)

    // Encode/Decode
    TOK_DECODE,         // ⊥ (decode / base value)
    TOK_ENCODE,         // ⊤ (encode / representation)

    // Table
    TOK_TABLE,          // ⍸ (table - convert to matrix)

    // Matrix operations
    TOK_DOMINO,         // ⌹ (matrix inverse / divide)

    // Execute and Format
    TOK_EXECUTE,        // ⍎ (execute string as APL)
    TOK_FORMAT,         // ⍕ (format - convert to character)

    // Reserved (nested arrays - not yet implemented)
    TOK_ENCLOSE,        // ⊂ (enclose)
    TOK_DISCLOSE,       // ⊃ (disclose / pick)
    TOK_MATCH,          // ≡ (depth / match)

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

    // Dfn delimiters and arguments
    TOK_LBRACE,         // {
    TOK_RBRACE,         // }
    TOK_ALPHA,          // ⍺ (left argument)
    TOK_OMEGA,          // ⍵ (right argument)

    // Special
    TOK_DIAMOND,        // ⋄ (statement separator)
    TOK_NEWLINE,
    TOK_COMMENT,
    TOK_ZILDE,          // ⍬ (empty vector)

    TOK_ERROR           // Lexer error
};

// Token structure
struct Token {
    TokenType type;

    union {
        double number;      // For TOK_NUMBER
        const char* name;   // For TOK_NAME (points into LexerArena)
        double* vector_data; // For TOK_NUMBER_VECTOR (heap allocated)
    };

    // Vector size (only used for TOK_NUMBER_VECTOR)
    size_t vector_size;

    // Source location for error reporting
    int line;
    int column;

    // Constructors
    Token() : type(TOK_EOF), number(0.0), vector_size(0), line(0), column(0) {}

    Token(TokenType t, int l = 0, int c = 0)
        : type(t), number(0.0), vector_size(0), line(l), column(c) {}

    Token(double num, int l = 0, int c = 0)
        : type(TOK_NUMBER), number(num), vector_size(0), line(l), column(c) {}

    Token(TokenType t, const char* n, int l = 0, int c = 0)
        : type(t), name(n), vector_size(0), line(l), column(c) {}

    // Constructor for vector tokens
    Token(TokenType t, double* vec, size_t size, int l, int c)
        : type(t), vector_data(vec), vector_size(size), line(l), column(c) {}

    // Destructor to clean up vector data
    ~Token() {
        if (type == TOK_NUMBER_VECTOR && vector_data) {
            delete[] vector_data;
        }
    }

    // Copy constructor
    Token(const Token& other)
        : type(other.type), vector_size(other.vector_size),
          line(other.line), column(other.column) {
        if (type == TOK_NUMBER_VECTOR && other.vector_data) {
            vector_data = new double[vector_size];
            for (size_t i = 0; i < vector_size; i++) {
                vector_data[i] = other.vector_data[i];
            }
        } else {
            number = other.number;  // Copies union (name or number)
        }
    }

    // Move constructor
    Token(Token&& other) noexcept
        : type(other.type), vector_size(other.vector_size),
          line(other.line), column(other.column) {
        if (type == TOK_NUMBER_VECTOR) {
            vector_data = other.vector_data;
            other.vector_data = nullptr;
            other.vector_size = 0;
        } else {
            number = other.number;
        }
    }

    // Assignment operators
    Token& operator=(const Token& other) {
        if (this != &other) {
            // Clean up existing vector data
            if (type == TOK_NUMBER_VECTOR && vector_data) {
                delete[] vector_data;
            }

            type = other.type;
            vector_size = other.vector_size;
            line = other.line;
            column = other.column;

            if (type == TOK_NUMBER_VECTOR && other.vector_data) {
                vector_data = new double[vector_size];
                for (size_t i = 0; i < vector_size; i++) {
                    vector_data[i] = other.vector_data[i];
                }
            } else {
                number = other.number;
            }
        }
        return *this;
    }

    Token& operator=(Token&& other) noexcept {
        if (this != &other) {
            // Clean up existing vector data
            if (type == TOK_NUMBER_VECTOR && vector_data) {
                delete[] vector_data;
            }

            type = other.type;
            vector_size = other.vector_size;
            line = other.line;
            column = other.column;

            if (type == TOK_NUMBER_VECTOR) {
                vector_data = other.vector_data;
                other.vector_data = nullptr;
                other.vector_size = 0;
            } else {
                number = other.number;
            }
        }
        return *this;
    }
};

// Helper function to convert token type to string (for debugging/error messages)
const char* token_type_name(TokenType type);

} // namespace apl
