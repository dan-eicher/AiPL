// APL Parser implementation using manual Pratt parsing

#include "parser.h"
#include "machine.h"
#include "continuation.h"
#include <stdexcept>
#include <cstdlib>

namespace apl {

// Binding powers for operators (higher = tighter binding)
// APL has uniform precedence, so all operators have the same binding power
// This gives us right-to-left evaluation
const int BP_NONE = 0;
const int BP_OPERATOR = 10;      // All dyadic operators have same precedence
const int BP_JUXTAPOSE = 100;    // Juxtaposition (strands) binds tightest

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

    // Now handle infix operators (led) and juxtaposition (strands)
    // Continue while the next operator has higher binding power
    while (!at_end()) {
        Token next = current();
        int bp = get_binding_power(next);

        // Check for juxtaposition (implicit strand formation)
        // Juxtaposition occurs when we see a token that can start a value (fb-term)
        // This corresponds to the grammar rule: fbn-term ::= fb-term fbn-term
        bool is_juxtaposition = false;
        if (bp == BP_NONE) {
            // Not an operator - check if it can start a value (nud-able token)
            switch (next.type) {
                case TOK_NUMBER:
                case TOK_LPAREN:
                // TODO (Phase 3.4.1): TOK_MINUS is NOT included here because it's ambiguous
                // (could be monadic negate or dyadic subtraction). This is a phase ordering
                // issue - we need to implement monadic operators first. For now, negative
                // literals in strands must use parentheses: (-1) 2 (-3)
                // Alternatively, we could add APL's ¯ (high minus) for negative literals.
                case TOK_NAME:   // For variables (Phase 3.2.3)
                    is_juxtaposition = true;
                    bp = BP_JUXTAPOSE;
                    break;
                default:
                    break;
            }
        }

        // APL is right-associative, so we use >= instead of >
        // This means: continue parsing if next_bp >= min_bp
        // For right-associativity, when we recurse, we pass bp (not bp+1)
        if (bp < min_bp) {
            break;
        }

        if (is_juxtaposition) {
            // Juxtaposition: recursively parse the right operand and form a strand
            // For right-associativity, pass the same bp
            Continuation* right = parse_expression(bp);
            if (!right) {
                return nullptr;
            }

            // Combine left and right into a strand
            std::vector<Continuation*> elements;

            // If left is already a strand, extend it
            if (auto* strand = dynamic_cast<StrandK*>(left)) {
                elements = strand->elements;
            } else {
                elements.push_back(left);
            }

            // If right is a strand, append all its elements
            if (auto* strand = dynamic_cast<StrandK*>(right)) {
                elements.insert(elements.end(), strand->elements.begin(), strand->elements.end());
            } else {
                elements.push_back(right);
            }

            StrandK* new_strand = new StrandK(elements);
            machine_->heap->allocate_continuation(new_strand);
            left = new_strand;
        } else {
            // Regular infix operator
            advance();
            left = led(left, next);
            if (!left) {
                return nullptr;
            }
        }
    }

    return left;
}

// Null denotation: handles prefix position (literals, unary operators, etc.)
Continuation* Parser::nud(const Token& token) {
    switch (token.type) {
        case TOK_NUMBER: {
            // Single number - create LiteralK
            double value = token.number;
            LiteralK* lit = new LiteralK(value);
            machine_->heap->allocate_continuation(lit);
            return lit;
        }

        case TOK_LPAREN: {
            // Parenthesized expression: parse inner expression with minimum binding power
            Continuation* inner = parse_expression(BP_NONE);
            if (!inner) {
                return nullptr;
            }

            // Expect closing parenthesis
            if (at_end() || current().type != TOK_RPAREN) {
                error_message_ = "Expected ')' after expression";
                return nullptr;
            }
            advance();  // consume ')'

            return inner;
        }

        case TOK_MINUS:
        case TOK_PLUS:
        case TOK_TIMES:
        case TOK_POWER:
        case TOK_DIVIDE: {
            // Monadic operator in prefix position
            // Parse the operand and create MonadicK continuation

            // Determine operator name
            const char* op_name = nullptr;
            switch (token.type) {
                case TOK_PLUS:   op_name = "+"; break;
                case TOK_MINUS:  op_name = "-"; break;
                case TOK_TIMES:  op_name = "×"; break;
                case TOK_POWER:  op_name = "*"; break;
                case TOK_DIVIDE: op_name = "÷"; break;
                default: break;
            }

            // Look up the primitive function
            Value* op_val = machine_->env->lookup(op_name);
            if (!op_val || op_val->tag != ValueType::PRIMITIVE) {
                error_message_ = std::string("Unknown operator: ") + op_name;
                return nullptr;
            }

            PrimitiveFn* prim_fn = op_val->data.primitive_fn;

            // Check that it has a monadic form
            if (!prim_fn->monadic) {
                error_message_ = std::string("Operator has no monadic form: ") + op_name;
                return nullptr;
            }

            // Parse the operand with high binding power (monadic binds tighter than dyadic)
            Continuation* operand = parse_expression(BP_OPERATOR + 50);
            if (!operand) {
                return nullptr;
            }

            // Create MonadicK continuation
            MonadicK* monadic = new MonadicK(prim_fn, operand);
            machine_->heap->allocate_continuation(monadic);
            return monadic;
        }

        case TOK_NAME: {
            // Variable reference - create LookupK
            const char* name = token.name;
            LookupK* lookup = new LookupK(name);
            machine_->heap->allocate_continuation(lookup);
            return lookup;
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

    // Look up the primitive function from environment
    Value* op_val = machine_->env->lookup(op_name);
    if (!op_val || op_val->tag != ValueType::PRIMITIVE) {
        error_message_ = std::string("Unknown operator: ") + op_name;
        return nullptr;
    }

    PrimitiveFn* prim_fn = op_val->data.primitive_fn;

    // Create dyadic operation continuation
    DyadicK* dyadic = new DyadicK(prim_fn, left, right);
    machine_->heap->allocate_continuation(dyadic);

    return dyadic;
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

        // Closing delimiters should never be treated as infix
        // Give them negative binding power to stop parsing
        case TOK_RPAREN:
            return -1;

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
