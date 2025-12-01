// APL Lexer - re2c specification
// Generates lexer.cpp with C++ Lexer class

#include "lexer.h"
#include <cstdlib>
#include <cstring>

namespace apl {

// Lexer constructor
Lexer::Lexer(const char* input)
    : cursor_(input)
    , marker_(input)
    , limit_(input + std::strlen(input))
    , line_(1)
    , column_(0)
{
    // arena_ is initialized by its default constructor
}

/*!re2c
    re2c:define:YYCTYPE = "unsigned char";
    re2c:encoding:utf8 = 1;
    re2c:define:YYCURSOR = cursor_;
    re2c:define:YYMARKER = marker_;
    re2c:define:YYLIMIT = limit_;
    re2c:yyfill:enable = 0;

    // Whitespace (skip)
    ws = [ \t\r]+;

    // Newline
    nl = "\n";

    // Digits
    digit = [0-9];
    digits = digit+;

    // Numbers (integer or float)
    number = digits ("." digits)? ([eE] [+\-]? digits)?;

    // Names (identifiers)
    alpha = [a-zA-Z_];
    alnum = [a-zA-Z0-9_];
    name = alpha alnum*;

    // String literals (APL uses single quotes, '' for escaped quote)
    string = "'" ([^'\n] | "''")* "'";

    // APL special characters (UTF-8)
    // Using actual Unicode characters (code points)
    times = "×";         // U+00D7
    divide = "÷";        // U+00F7
    reshape = "⍴";       // U+2374
    transpose = "⍉";     // U+2349
    iota = "⍳";          // U+2373
    take = "↑";          // U+2191
    drop = "↓";          // U+2193
    reduce_first = "⌿"; // U+233F
    scan_first = "⍀";   // U+2340
    each = "¨";          // U+00A8
    compose = "∘";       // U+2218
    commute = "⍨";       // U+2368
    assign = "←";        // U+2190
    goto_sym = "→";      // U+2192
    not_equal = "≠";     // U+2260
    less_eq = "≤";       // U+2264
    greater_eq = "≥";    // U+2265
    and_sym = "∧";       // U+2227
    or_sym = "∨";        // U+2228
    diamond = "⋄";       // U+22C4
    alpha_sym = "⍺";     // U+237A (left argument)
    omega_sym = "⍵";     // U+2375 (right argument)

    // Comments (⍝ to end of line)
    comment = "⍝" [^\n]*;  // U+235D
*/

Token Lexer::next_token() {
    while (true) {
        const char* token_start = cursor_;
        int token_line = line_;
        int token_column = column_;

        /*!re2c
            // End of input
            "\x00" { return Token(TOK_EOF, token_line, token_column); }

            // Whitespace - skip and continue
            ws { column_ += (cursor_ - token_start); continue; }

            // Newline
            nl {
                line_++;
                column_ = 0;
                return Token(TOK_NEWLINE, token_line, token_column);
            }

            // Comments - skip and continue
            comment { continue; }

        // Numbers
        number {
            double num = std::atof(token_start);
            column_ += (cursor_ - token_start);
            return Token(num, token_line, token_column);
        }

        // Control flow keywords
        ":If" { column_ += 3; return Token(TOK_IF, token_line, token_column); }
        ":Else" { column_ += 5; return Token(TOK_ELSE, token_line, token_column); }
        ":ElseIf" { column_ += 7; return Token(TOK_ELSEIF, token_line, token_column); }
        ":EndIf" { column_ += 6; return Token(TOK_ENDIF, token_line, token_column); }
        ":While" { column_ += 6; return Token(TOK_WHILE, token_line, token_column); }
        ":EndWhile" { column_ += 9; return Token(TOK_ENDWHILE, token_line, token_column); }
        ":For" { column_ += 4; return Token(TOK_FOR, token_line, token_column); }
        ":In" { column_ += 3; return Token(TOK_IN, token_line, token_column); }
        ":EndFor" { column_ += 7; return Token(TOK_ENDFOR, token_line, token_column); }
        ":Leave" { column_ += 6; return Token(TOK_LEAVE, token_line, token_column); }
        ":Return" { column_ += 7; return Token(TOK_RETURN, token_line, token_column); }

        // String literals (must come before names to avoid conflict)
        string {
            // Extract string content, handling '' escape sequences
            size_t len = cursor_ - token_start;
            column_ += len;

            // Skip opening and closing quotes
            const char* start = token_start + 1;
            const char* end = cursor_ - 1;

            // Count actual string length (handling '' -> ')
            size_t actual_len = 0;
            for (const char* p = start; p < end; p++) {
                actual_len++;
                if (*p == '\'' && p + 1 < end && *(p + 1) == '\'') {
                    p++;  // Skip second quote in '' pair
                }
            }

            // Build unescaped string in temporary buffer
            char* temp = new char[actual_len + 1];
            size_t dst = 0;
            for (const char* p = start; p < end; p++) {
                temp[dst++] = *p;
                if (*p == '\'' && p + 1 < end && *(p + 1) == '\'') {
                    p++;  // Skip second quote in '' pair
                }
            }
            temp[actual_len] = '\0';

            // Allocate in arena and copy
            char* str = arena_.allocate_string(temp, actual_len);
            delete[] temp;

            return Token(TOK_STRING, str, token_line, token_column);
        }

        // Names (must come after keywords and strings)
        name {
            size_t len = cursor_ - token_start;
            char* str = arena_.allocate_string(token_start, len);
            column_ += len;
            return Token(TOK_NAME, str, token_line, token_column);
        }

        // Single-character operators
        "+" { column_++; return Token(TOK_PLUS, token_line, token_column); }
        "-" { column_++; return Token(TOK_MINUS, token_line, token_column); }
        "*" { column_++; return Token(TOK_POWER, token_line, token_column); }
        "," { column_++; return Token(TOK_RAVEL, token_line, token_column); }
        "/" { column_++; return Token(TOK_REDUCE, token_line, token_column); }
        "\\" { column_++; return Token(TOK_SCAN, token_line, token_column); }
        "=" { column_++; return Token(TOK_EQUAL, token_line, token_column); }
        "<" { column_++; return Token(TOK_LESS, token_line, token_column); }
        ">" { column_++; return Token(TOK_GREATER, token_line, token_column); }
        "~" { column_++; return Token(TOK_NOT, token_line, token_column); }
        "(" { column_++; return Token(TOK_LPAREN, token_line, token_column); }
        ")" { column_++; return Token(TOK_RPAREN, token_line, token_column); }
        "[" { column_++; return Token(TOK_LBRACKET, token_line, token_column); }
        "]" { column_++; return Token(TOK_RBRACKET, token_line, token_column); }
        "{" { column_++; return Token(TOK_LBRACE, token_line, token_column); }
        "}" { column_++; return Token(TOK_RBRACE, token_line, token_column); }
        ";" { column_++; return Token(TOK_SEMICOLON, token_line, token_column); }

        // APL Unicode symbols
        times { column_++; return Token(TOK_TIMES, token_line, token_column); }
        divide { column_++; return Token(TOK_DIVIDE, token_line, token_column); }
        reshape { column_++; return Token(TOK_RESHAPE, token_line, token_column); }
        transpose { column_++; return Token(TOK_TRANSPOSE, token_line, token_column); }
        iota { column_++; return Token(TOK_IOTA, token_line, token_column); }
        take { column_++; return Token(TOK_TAKE, token_line, token_column); }
        drop { column_++; return Token(TOK_DROP, token_line, token_column); }
        reduce_first { column_++; return Token(TOK_REDUCE_FIRST, token_line, token_column); }
        scan_first { column_++; return Token(TOK_SCAN_FIRST, token_line, token_column); }
        each { column_++; return Token(TOK_EACH, token_line, token_column); }
        compose { column_++; return Token(TOK_COMPOSE, token_line, token_column); }
        commute { column_++; return Token(TOK_COMMUTE, token_line, token_column); }
        assign { column_++; return Token(TOK_ASSIGN, token_line, token_column); }
        goto_sym { column_++; return Token(TOK_GOTO, token_line, token_column); }
        not_equal { column_++; return Token(TOK_NOT_EQUAL, token_line, token_column); }
        less_eq { column_++; return Token(TOK_LESS_EQUAL, token_line, token_column); }
        greater_eq { column_++; return Token(TOK_GREATER_EQUAL, token_line, token_column); }
        and_sym { column_++; return Token(TOK_AND, token_line, token_column); }
        or_sym { column_++; return Token(TOK_OR, token_line, token_column); }
        diamond { column_++; return Token(TOK_DIAMOND, token_line, token_column); }
        alpha_sym { column_++; return Token(TOK_ALPHA, token_line, token_column); }
        omega_sym { column_++; return Token(TOK_OMEGA, token_line, token_column); }

        // Outer product (special two-character sequence)
        compose "." { column_ += 2; return Token(TOK_OUTER_PRODUCT, token_line, token_column); }

            // Unknown character - error
            * {
                column_++;
                return Token(TOK_ERROR, token_line, token_column);
            }
        */
    }
}

} // namespace apl
