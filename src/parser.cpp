// APL Parser implementation using manual Pratt parsing

#include "parser.h"
#include "heap.h"
#include "continuation.h"
#include <stdexcept>
#include <cstdlib>

namespace apl {

// Binding powers for operators (higher = tighter binding)
// APL has uniform precedence, so all operators have the same binding power
// This gives us right-to-left evaluation
const int BP_NONE = 0;
const int BP_OPERATOR = 10;  // All dyadic operators have same precedence

// Parse entry point
Continuation* Parser::parse(const std::string& input) {
    error_message_.clear();

    // Keep input alive for lexer lifetime
    input_ = input;

    // Create lexer for on-demand tokenization
    lexer_ = new Lexer(input_.c_str());

    // Get first token
    current_token_ = lexer_->next_token();

    if (current_token_.type == TOK_ERROR) {
        error_message_ = "Lexer error";
        delete lexer_;
        lexer_ = nullptr;
        return nullptr;
    }

    // Parse expression starting with minimum binding power
    Continuation* result = parse_expression(BP_NONE);

    if (!result) {
        if (error_message_.empty()) {
            error_message_ = "Parse failed";
        }
        delete lexer_;
        lexer_ = nullptr;
        return nullptr;
    }

    // Check that we consumed all input
    if (!at_end()) {
        error_message_ = "Unexpected input after expression";
        delete lexer_;
        lexer_ = nullptr;
        return nullptr;
    }

    delete lexer_;
    lexer_ = nullptr;
    return result;
}

// Core Pratt parsing algorithm
Continuation* Parser::parse_expression(int min_bp) {
    if (at_end()) {
        error_message_ = "Unexpected end of input";
        return nullptr;
    }

    // Get the current token and parse its null denotation (prefix)
    Token token = current();
    advance();

    Continuation* left = nud(token);
    if (!left) {
        return nullptr;
    }

    // Now handle infix operators (led)
    // Continue while the next operator has higher binding power
    while (!at_end()) {
        Token next = current();
        int bp = get_binding_power(next);

        // APL is right-associative, so we use >= instead of >
        // This means: continue parsing if next_bp >= min_bp
        // For right-associativity, when we recurse, we pass bp (not bp+1)
        if (bp < min_bp) {
            break;
        }

        advance();
        left = led(left, next);
        if (!left) {
            return nullptr;
        }
    }

    return left;
}

// Null denotation: handles prefix position (literals, unary operators, etc.)
Continuation* Parser::nud(const Token& token) {
    switch (token.type) {
        case TOK_NUMBER: {
            // Token already contains the number value
            double value = token.number;
            LiteralK* lit = new LiteralK(value, nullptr);
            heap_->allocate_continuation(lit);
            return lit;
        }

        case TOK_MINUS: {
            // Unary minus: parse as negative literal if followed by number
            // Otherwise it's an error for now (monadic minus not yet implemented)
            if (!at_end() && current().type == TOK_NUMBER) {
                Token num = current();
                advance();
                double value = -num.number;
                LiteralK* lit = new LiteralK(value, nullptr);
                heap_->allocate_continuation(lit);
                return lit;
            }
            error_message_ = "Monadic operators not yet implemented";
            return nullptr;
        }

        default:
            error_message_ = std::string("Unexpected token in prefix position: ") + token_type_name(token.type);
            return nullptr;
    }
}

// Left denotation: handles infix position (binary operators)
Continuation* Parser::led(Continuation* left, const Token& token) {
    // Determine operator name
    const char* op_name = nullptr;

    switch (token.type) {
        case TOK_PLUS:
            op_name = "+";
            break;
        case TOK_MINUS:
            op_name = "-";
            break;
        case TOK_TIMES:
        case TOK_POWER:  // * is an alias for ×
            op_name = "×";
            break;
        case TOK_DIVIDE:
            op_name = "÷";
            break;
        default:
            error_message_ = std::string("Unexpected token in infix position: ") + token_type_name(token.type);
            return nullptr;
    }

    // For right-associative operators, we parse the right side with the SAME binding power
    // This is the key to right-to-left evaluation in Pratt parsing
    int bp = get_binding_power(token);
    Continuation* right = parse_expression(bp);

    if (!right) {
        return nullptr;
    }

    // Create binary operation continuation
    BinOpK* binop = new BinOpK(op_name, left, right);
    heap_->allocate_continuation(binop);

    return binop;
}

// Get binding power for a token
int Parser::get_binding_power(const Token& token) {
    // In APL, all operators have the same precedence
    // Right-to-left evaluation is achieved through right-associativity
    switch (token.type) {
        case TOK_PLUS:
        case TOK_MINUS:
        case TOK_TIMES:
        case TOK_POWER:  // * is an alias for ×
        case TOK_DIVIDE:
            return BP_OPERATOR;
        default:
            return BP_NONE;
    }
}

// Advance to next token (on-demand from lexer)
void Parser::advance() {
    if (lexer_ && !at_end()) {
        current_token_ = lexer_->next_token();
    }
}

} // namespace apl
