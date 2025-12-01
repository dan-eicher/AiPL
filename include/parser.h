// APL Pratt Parser
// Manual Pratt parser with integrated on-demand lexing

#pragma once

#include "continuation.h"
#include "token.h"
#include "lexer.h"
#include <string>

namespace apl {

// Forward declarations
class Machine;

// Pratt parser that builds continuation graphs
// Integrates with Lexer for on-demand tokenization
class Parser {
public:
    Parser(Machine* machine) : machine_(machine), lexer_(nullptr), current_token_() {}

    // Parse APL input and return the continuation graph
    // - For single expressions: returns the expression continuation
    // - For multi-statement programs: returns SeqK wrapping all statements
    // Statements separated by newlines or diamonds (⋄)
    // Returns nullptr on parse failure
    Continuation* parse(const std::string& input);

    // Get the last error message (if parse failed)
    const std::string& get_error() const { return error_message_; }

private:
    Machine* machine_;
    std::string error_message_;
    std::string input_;  // Keep input alive for lexer

    // Integrated lexer (on-demand tokenization)
    Lexer* lexer_;
    Token current_token_;

    // Pratt parsing functions
    Continuation* parse_expression(int min_bp);
    Continuation* nud(const Token& token);  // Null denotation (prefix)
    Continuation* led(Continuation* left, const Token& token);  // Left denotation (infix)
    Continuation* led_juxtapose(Continuation* left, int bp);  // Juxtaposition (strand formation)

    // Statement parsing
    Continuation* parse_statement();  // Parse statement or expression

    // Binding power lookup
    int get_binding_power(const Token& token);

    // Helper functions
    Continuation* parse_dfn_body();  // Parse dfn body from { to }
    std::vector<Continuation*> parse_block_until(TokenType terminator);  // Parse statements until terminator
    Continuation* parse_if_statement();      // Parse :If/:Else/:EndIf
    Continuation* parse_while_statement();   // Parse :While/:EndWhile
    Continuation* parse_for_statement();     // Parse :For/:In/:EndFor
    Continuation* parse_return_statement();  // Parse :Return
    Continuation* parse_leave_statement();   // Parse :Leave

    // Token stream helpers (unified with lexer)
    const Token& current() const { return current_token_; }
    void advance();  // Calls lexer->next_token()
    bool at_end() const { return current_token_.type == TOK_EOF; }

    // Statement separator helpers (Phase 3.3)
    void skip_separators();  // Skip newlines, diamonds, and comments
    bool is_separator(const Token& token) const;
};

} // namespace apl
