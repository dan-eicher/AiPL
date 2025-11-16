// Lexer - APL tokenizer interface

#pragma once

#include "token.h"
#include "lexer_arena.h"

namespace apl {

// Forward declaration from lexer.cpp
struct LexerState;

// Initialize lexer state with input string
// The input string must remain valid for the lifetime of the lexer
LexerState* lexer_init(const char* input, LexerArena* arena);

// Free lexer state
void lexer_free(LexerState* state);

// Get next token from input
// Returns TOK_EOF when input is exhausted
Token lex_next_token(LexerState* state);

} // namespace apl
