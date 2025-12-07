// APL Lexer - re2c specification
// Generates lexer.cpp with C++ Lexer class

#include "lexer.h"
#include <cstdlib>
#include <cstring>

namespace apl {

// Parse an APL number from start to end, handling high minus (¯) in both
// the leading position and in the exponent (e.g., ¯1.5e¯3)
static double parse_apl_number(const char* start, const char* end) {
    const char* p = start;
    bool negative = false;

    // Check for leading high minus (¯ = U+00AF = 0xC2 0xAF in UTF-8)
    if (p < end - 1 && (unsigned char)p[0] == 0xC2 && (unsigned char)p[1] == 0xAF) {
        negative = true;
        p += 2;
    }

    // Check if there's a high minus in the exponent
    bool has_exp_minus = false;
    for (const char* q = p; q < end - 1; q++) {
        if ((unsigned char)q[0] == 0xC2 && (unsigned char)q[1] == 0xAF) {
            has_exp_minus = true;
            break;
        }
    }

    double num;
    if (has_exp_minus) {
        // Build temporary string with high minus replaced by regular minus
        size_t len = end - p;
        char* temp = new char[len + 1];
        size_t dst = 0;
        for (const char* q = p; q < end; ) {
            if (q < end - 1 && (unsigned char)q[0] == 0xC2 && (unsigned char)q[1] == 0xAF) {
                temp[dst++] = '-';
                q += 2;
            } else {
                temp[dst++] = *q++;
            }
        }
        temp[dst] = '\0';
        num = std::atof(temp);
        delete[] temp;
    } else {
        num = std::atof(p);
    }

    return negative ? -num : num;
}

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

    // High minus (macron) for negative literals
    high_minus = "¯";  // U+00AF

    // Numbers (integer or float)
    // APL uses ¯ (high minus) for negative literals: ¯3.14 and ¯1.5e¯10
    number = high_minus? digits ("." digits)? ([eE] ([+\-] | high_minus)? digits)?;

    // Numeric vector literals (ISO 13751)
    // Space-separated numbers: "1 2 3" → vector [1, 2, 3]
    // NOTE: Must be matched before single number pattern
    ws_inline = [ \t]+;  // Inline whitespace (not newline)
    number_vector = number (ws_inline number)+;

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
    rank = "⍤";          // U+2364
    assign = "←";        // U+2190
    goto_sym = "→";      // U+2192
    not_equal = "≠";     // U+2260
    less_eq = "≤";       // U+2264
    greater_eq = "≥";    // U+2265
    and_sym = "∧";       // U+2227
    or_sym = "∨";        // U+2228
    nand_sym = "⍲";      // U+2372
    nor_sym = "⍱";       // U+2371
    ceiling = "⌈";       // U+2308
    floor_sym = "⌊";     // U+230A
    log_sym = "⍟";       // U+235F
    reverse = "⌽";       // U+233D
    reverse_first = "⊖"; // U+2296
    tally = "≢";         // U+2262
    member_of = "∊";     // U+220A (small element of)
    grade_up = "⍋";      // U+234B (grade up)
    grade_down = "⍒";    // U+2352 (grade down)
    union_sym = "∪";     // U+222A (union/unique)
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

        // Numeric vector literal (ISO 13751) - must come before single number
        number_vector {
            // First pass: count numbers
            size_t count = 0;
            const char* p = token_start;
            while (p < cursor_) {
                while (p < cursor_ && (*p == ' ' || *p == '\t')) p++;
                if (p >= cursor_) break;
                count++;
                while (p < cursor_ && *p != ' ' && *p != '\t') p++;
            }

            // Allocate array
            double* vec_data = new double[count];

            // Second pass: parse numbers into array
            p = token_start;
            size_t idx = 0;
            while (p < cursor_ && idx < count) {
                while (p < cursor_ && (*p == ' ' || *p == '\t')) p++;
                if (p >= cursor_) break;
                const char* num_start = p;
                while (p < cursor_ && *p != ' ' && *p != '\t') p++;
                vec_data[idx++] = parse_apl_number(num_start, p);
            }

            column_ += (cursor_ - token_start);
            return Token(TOK_NUMBER_VECTOR, vec_data, count, token_line, token_column);
        }

        // Single number
        number {
            double num = parse_apl_number(token_start, cursor_);
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
        rank { column_++; return Token(TOK_RANK, token_line, token_column); }
        assign { column_++; return Token(TOK_ASSIGN, token_line, token_column); }
        goto_sym { column_++; return Token(TOK_GOTO, token_line, token_column); }
        not_equal { column_++; return Token(TOK_NOT_EQUAL, token_line, token_column); }
        less_eq { column_++; return Token(TOK_LESS_EQUAL, token_line, token_column); }
        greater_eq { column_++; return Token(TOK_GREATER_EQUAL, token_line, token_column); }
        and_sym { column_++; return Token(TOK_AND, token_line, token_column); }
        or_sym { column_++; return Token(TOK_OR, token_line, token_column); }
        nand_sym { column_++; return Token(TOK_NAND, token_line, token_column); }
        nor_sym { column_++; return Token(TOK_NOR, token_line, token_column); }
        ceiling { column_++; return Token(TOK_CEILING, token_line, token_column); }
        floor_sym { column_++; return Token(TOK_FLOOR, token_line, token_column); }
        log_sym { column_++; return Token(TOK_LOG, token_line, token_column); }
        "|" { column_++; return Token(TOK_STILE, token_line, token_column); }
        "!" { column_++; return Token(TOK_FACTORIAL, token_line, token_column); }
        reverse { column_++; return Token(TOK_REVERSE, token_line, token_column); }
        reverse_first { column_++; return Token(TOK_REVERSE_FIRST, token_line, token_column); }
        tally { column_++; return Token(TOK_TALLY, token_line, token_column); }
        member_of { column_++; return Token(TOK_MEMBER, token_line, token_column); }
        grade_up { column_++; return Token(TOK_GRADE_UP, token_line, token_column); }
        grade_down { column_++; return Token(TOK_GRADE_DOWN, token_line, token_column); }
        union_sym { column_++; return Token(TOK_UNION, token_line, token_column); }
        diamond { column_++; return Token(TOK_DIAMOND, token_line, token_column); }
        alpha_sym { column_++; return Token(TOK_ALPHA, token_line, token_column); }
        omega_sym { column_++; return Token(TOK_OMEGA, token_line, token_column); }

        // Outer product (special two-character sequence)
        compose "." { column_ += 2; return Token(TOK_OUTER_PRODUCT, token_line, token_column); }
        "." { column_++; return Token(TOK_DOT, token_line, token_column); }

            // Unknown character - error
            * {
                column_++;
                return Token(TOK_ERROR, token_line, token_column);
            }
        */
    }
}

} // namespace apl
