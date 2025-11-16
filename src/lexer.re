// APL Lexer - re2c specification
// Generates lexer.cpp with lex_next_token() function

#include "token.h"
#include "lexer_arena.h"
#include <cstdlib>
#include <cstring>

namespace apl {

// Lexer state
struct LexerState {
    const char* cursor;     // Current position
    const char* marker;     // Backtrack marker
    const char* limit;      // End of input
    LexerArena* arena;      // Arena for string allocation
    int line;               // Current line number
    int column;             // Current column number
};

// Initialize lexer
LexerState* lexer_init(const char* input, LexerArena* arena) {
    LexerState* state = new LexerState();
    state->cursor = input;
    state->marker = input;
    state->limit = input + std::strlen(input);
    state->arena = arena;
    state->line = 1;
    state->column = 0;
    return state;
}

// Free lexer
void lexer_free(LexerState* state) {
    delete state;
}

// Forward declaration
Token lex_next_token(LexerState* state);

/*!re2c
    re2c:define:YYCTYPE = "unsigned char";
    re2c:encoding:utf8 = 1;
    re2c:define:YYCURSOR = state->cursor;
    re2c:define:YYMARKER = state->marker;
    re2c:define:YYLIMIT = state->limit;
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

    // Comments (⍝ to end of line)
    comment = "⍝" [^\n]*;  // U+235D
*/

Token lex_next_token(LexerState* state) {
    while (true) {
        const char* token_start = state->cursor;
        int token_line = state->line;
        int token_column = state->column;

        /*!re2c
            // End of input
            "\x00" { return Token(TOK_EOF, token_line, token_column); }

            // Whitespace - skip and continue
            ws { state->column += (state->cursor - token_start); continue; }

            // Newline
            nl {
                state->line++;
                state->column = 0;
                return Token(TOK_NEWLINE, token_line, token_column);
            }

            // Comments - skip and continue
            comment { continue; }

        // Numbers
        number {
            double num = std::atof(token_start);
            state->column += (state->cursor - token_start);
            return Token(num, token_line, token_column);
        }

        // Control flow keywords
        ":If" { state->column += 3; return Token(TOK_IF, token_line, token_column); }
        ":Else" { state->column += 5; return Token(TOK_ELSE, token_line, token_column); }
        ":ElseIf" { state->column += 7; return Token(TOK_ELSEIF, token_line, token_column); }
        ":EndIf" { state->column += 6; return Token(TOK_ENDIF, token_line, token_column); }
        ":While" { state->column += 6; return Token(TOK_WHILE, token_line, token_column); }
        ":EndWhile" { state->column += 9; return Token(TOK_ENDWHILE, token_line, token_column); }
        ":For" { state->column += 4; return Token(TOK_FOR, token_line, token_column); }
        ":EndFor" { state->column += 7; return Token(TOK_ENDFOR, token_line, token_column); }
        ":Leave" { state->column += 6; return Token(TOK_LEAVE, token_line, token_column); }
        ":Return" { state->column += 7; return Token(TOK_RETURN, token_line, token_column); }

        // Names (must come after keywords)
        name {
            size_t len = state->cursor - token_start;
            char* str = state->arena->allocate_string(token_start, len);
            state->column += len;
            return Token(TOK_NAME, str, token_line, token_column);
        }

        // Single-character operators
        "+" { state->column++; return Token(TOK_PLUS, token_line, token_column); }
        "-" { state->column++; return Token(TOK_MINUS, token_line, token_column); }
        "*" { state->column++; return Token(TOK_POWER, token_line, token_column); }
        "," { state->column++; return Token(TOK_RAVEL, token_line, token_column); }
        "/" { state->column++; return Token(TOK_REDUCE, token_line, token_column); }
        "\\" { state->column++; return Token(TOK_SCAN, token_line, token_column); }
        "=" { state->column++; return Token(TOK_EQUAL, token_line, token_column); }
        "<" { state->column++; return Token(TOK_LESS, token_line, token_column); }
        ">" { state->column++; return Token(TOK_GREATER, token_line, token_column); }
        "~" { state->column++; return Token(TOK_NOT, token_line, token_column); }
        "(" { state->column++; return Token(TOK_LPAREN, token_line, token_column); }
        ")" { state->column++; return Token(TOK_RPAREN, token_line, token_column); }
        "[" { state->column++; return Token(TOK_LBRACKET, token_line, token_column); }
        "]" { state->column++; return Token(TOK_RBRACKET, token_line, token_column); }
        "{" { state->column++; return Token(TOK_LBRACE, token_line, token_column); }
        "}" { state->column++; return Token(TOK_RBRACE, token_line, token_column); }
        ";" { state->column++; return Token(TOK_SEMICOLON, token_line, token_column); }

        // APL Unicode symbols
        times { state->column++; return Token(TOK_TIMES, token_line, token_column); }
        divide { state->column++; return Token(TOK_DIVIDE, token_line, token_column); }
        reshape { state->column++; return Token(TOK_RESHAPE, token_line, token_column); }
        transpose { state->column++; return Token(TOK_TRANSPOSE, token_line, token_column); }
        iota { state->column++; return Token(TOK_IOTA, token_line, token_column); }
        take { state->column++; return Token(TOK_TAKE, token_line, token_column); }
        drop { state->column++; return Token(TOK_DROP, token_line, token_column); }
        reduce_first { state->column++; return Token(TOK_REDUCE_FIRST, token_line, token_column); }
        scan_first { state->column++; return Token(TOK_SCAN_FIRST, token_line, token_column); }
        each { state->column++; return Token(TOK_EACH, token_line, token_column); }
        compose { state->column++; return Token(TOK_COMPOSE, token_line, token_column); }
        commute { state->column++; return Token(TOK_COMMUTE, token_line, token_column); }
        assign { state->column++; return Token(TOK_ASSIGN, token_line, token_column); }
        goto_sym { state->column++; return Token(TOK_GOTO, token_line, token_column); }
        not_equal { state->column++; return Token(TOK_NOT_EQUAL, token_line, token_column); }
        less_eq { state->column++; return Token(TOK_LESS_EQUAL, token_line, token_column); }
        greater_eq { state->column++; return Token(TOK_GREATER_EQUAL, token_line, token_column); }
        and_sym { state->column++; return Token(TOK_AND, token_line, token_column); }
        or_sym { state->column++; return Token(TOK_OR, token_line, token_column); }
        diamond { state->column++; return Token(TOK_DIAMOND, token_line, token_column); }

        // Outer product (special two-character sequence)
        compose "." { state->column += 2; return Token(TOK_OUTER_PRODUCT, token_line, token_column); }

            // Unknown character - error
            * {
                state->column++;
                return Token(TOK_ERROR, token_line, token_column);
            }
        */
    }
}

} // namespace apl
