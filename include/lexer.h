// Lexer - APL tokenizer (C++ class)

#pragma once

#include "token.h"
#include "lexer_arena.h"

namespace apl {

// APL Lexer class
// Incrementally tokenizes input on-demand
class Lexer {
public:
    // Initialize lexer with input string
    // The input string must remain valid for the lifetime of the lexer
    explicit Lexer(const char* input);

    // Get next token from input
    // Returns TOK_EOF when input is exhausted
    Token next_token();

    // Get current line/column for error reporting
    int line() const { return line_; }
    int column() const { return column_; }

private:
    // re2c scanner state (must be accessible to generated code)
    const char* cursor_;    // Current position in input
    const char* marker_;    // Backtrack marker for re2c
    const char* limit_;     // End of input

    // String arena for identifier storage
    LexerArena arena_;

    // Source location tracking
    int line_;
    int column_;
};

} // namespace apl
