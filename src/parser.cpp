// APL Parser implementation using manual Pratt parsing

#include "parser.h"
#include "heap.h"
#include "continuation.h"
#include "lexer.h"
#include "lexer_arena.h"
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
    tokens_.clear();
    pos_ = 0;

    // Tokenize the input
    LexerArena arena;
    LexerState* lexer = lexer_init(input.c_str(), &arena);
    Token tok = lex_next_token(lexer);
    while (tok.type != TOK_EOF) {
        if (tok.type == TOK_ERROR) {
            error_message_ = "Lexer error";
            lexer_free(lexer);
            return nullptr;
        }
        tokens_.push_back(tok);
        tok = lex_next_token(lexer);
    }
    lexer_free(lexer);

    // Parse expression starting with minimum binding power
    Continuation* result = parse_expression(BP_NONE);

    if (!result) {
        if (error_message_.empty()) {
            error_message_ = "Parse failed";
        }
        return nullptr;
    }

    // Check that we consumed all tokens
    if (!at_end()) {
        error_message_ = "Unexpected input after expression";
        return nullptr;
    }

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

// Token stream helpers
const Token& Parser::current() const {
    static Token eof_token;  // Default constructor makes TOK_EOF
    if (pos_ >= tokens_.size()) {
        return eof_token;
    }
    return tokens_[pos_];
}

const Token& Parser::peek(int offset) const {
    static Token eof_token;  // Default constructor makes TOK_EOF
    size_t peek_pos = pos_ + offset;
    if (peek_pos >= tokens_.size()) {
        return eof_token;
    }
    return tokens_[peek_pos];
}

void Parser::advance() {
    if (pos_ < tokens_.size()) {
        pos_++;
    }
}

bool Parser::at_end() const {
    return pos_ >= tokens_.size();
}

} // namespace apl
