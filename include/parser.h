// APL Pratt Parser
// Manual Pratt parser with integrated on-demand lexing

#pragma once

#include "continuation.h"
#include "token.h"
#include "lexer.h"
#include <string>

namespace apl {

// Forward declaration
class APLHeap;

// Pratt parser that builds continuation graphs
// Integrates with Lexer for on-demand tokenization
class Parser {
public:
    Parser(APLHeap* heap) : heap_(heap), lexer_(nullptr), current_token_() {}

    // Parse an APL expression and return the continuation graph
    // Returns nullptr on parse failure
    Continuation* parse(const std::string& input);

    // Get the last error message (if parse failed)
    const std::string& get_error() const { return error_message_; }

private:
    APLHeap* heap_;
    std::string error_message_;
    std::string input_;  // Keep input alive for lexer

    // Integrated lexer (on-demand tokenization)
    Lexer* lexer_;
    Token current_token_;

    // Pratt parsing functions
    Continuation* parse_expression(int min_bp);
    Continuation* nud(const Token& token);  // Null denotation (prefix)
    Continuation* led(Continuation* left, const Token& token);  // Left denotation (infix)

    // Binding power lookup
    int get_binding_power(const Token& token);

    // Token stream helpers (unified with lexer)
    const Token& current() const { return current_token_; }
    void advance();  // Calls lexer->next_token()
    bool at_end() const { return current_token_.type == TOK_EOF; }
};

} // namespace apl
