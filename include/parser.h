// APL Pratt Parser
// Manual Pratt parser with binding powers for right-to-left evaluation

#pragma once

#include "continuation.h"
#include "token.h"
#include <string>
#include <vector>

namespace apl {

// Forward declaration
class APLHeap;

// Pratt parser that builds continuation graphs
class Parser {
public:
    Parser(APLHeap* heap) : heap_(heap), pos_(0) {}

    // Parse an APL expression and return the continuation graph
    // Returns nullptr on parse failure
    Continuation* parse(const std::string& input);

    // Get the last error message (if parse failed)
    const std::string& get_error() const { return error_message_; }

private:
    APLHeap* heap_;
    std::string error_message_;

    // Token stream and current position
    std::vector<Token> tokens_;
    size_t pos_;

    // Pratt parsing functions
    Continuation* parse_expression(int min_bp);
    Continuation* nud(const Token& token);  // Null denotation (prefix)
    Continuation* led(Continuation* left, const Token& token);  // Left denotation (infix)

    // Binding power lookup
    int get_binding_power(const Token& token);

    // Token stream helpers
    const Token& current() const;
    const Token& peek(int offset = 1) const;
    void advance();
    bool at_end() const;
};

} // namespace apl
