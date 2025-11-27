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
const int BP_ASSIGN = 5;         // Assignment has lowest precedence
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
    // Assignment is handled naturally as a low-precedence infix operator
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

        // Phase 3.3: Stop if we hit a statement separator
        if (is_separator(next)) {
            break;
        }

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
            // Juxtaposition: recursively parse the right operand
            // For right-associativity, pass the same bp
            Continuation* right = parse_expression(bp);
            if (!right) {
                return nullptr;
            }

            // Simple strand formation - parser is type-agnostic
            // Per Grammar G2, all values/functions share same syntactic category
            // Juxtaposition ALWAYS means strand at parse time
            // The semantic interpretation (function application vs strand) happens at eval time
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
        case TOK_DIVIDE:
        case TOK_RESHAPE:
        case TOK_RAVEL:
        case TOK_IOTA: {
            // Monadic operator in prefix position
            // Parse the operand and create MonadicK continuation

            // Determine operator name
            const char* op_name = nullptr;
            switch (token.type) {
                case TOK_PLUS:    op_name = "+"; break;
                case TOK_MINUS:   op_name = "-"; break;
                case TOK_TIMES:   op_name = "×"; break;
                case TOK_POWER:   op_name = "*"; break;
                case TOK_DIVIDE:  op_name = "÷"; break;
                case TOK_RESHAPE: op_name = "⍴"; break;
                case TOK_RAVEL:   op_name = ","; break;
                case TOK_IOTA:    op_name = "⍳"; break;
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

        case TOK_IF: {
            // :If condition ... :Else ... :EndIf
            // Parse condition (expression until separator)
            skip_separators();
            Continuation* condition = parse_expression(BP_NONE);
            if (!condition) {
                return nullptr;
            }

            skip_separators();

            // Parse then-branch (statements until :Else or :EndIf)
            std::vector<Continuation*> then_stmts;
            while (!at_end() && current().type != TOK_ELSE && current().type != TOK_ENDIF) {
                Continuation* stmt = parse_expression(BP_NONE);
                if (!stmt) {
                    return nullptr;
                }
                then_stmts.push_back(stmt);
                skip_separators();
            }

            Continuation* then_branch = nullptr;
            if (!then_stmts.empty()) {
                then_branch = new SeqK(then_stmts);
                machine_->heap->allocate_continuation(then_branch);
            }

            // Check for :Else
            Continuation* else_branch = nullptr;
            if (!at_end() && current().type == TOK_ELSE) {
                advance();  // consume :Else
                skip_separators();

                // Parse else-branch (statements until :EndIf)
                std::vector<Continuation*> else_stmts;
                while (!at_end() && current().type != TOK_ENDIF) {
                    Continuation* stmt = parse_expression(BP_NONE);
                    if (!stmt) {
                        return nullptr;
                    }
                    else_stmts.push_back(stmt);
                    skip_separators();
                }

                if (!else_stmts.empty()) {
                    else_branch = new SeqK(else_stmts);
                    machine_->heap->allocate_continuation(else_branch);
                }
            }

            // Expect :EndIf
            if (at_end() || current().type != TOK_ENDIF) {
                error_message_ = "Expected :EndIf";
                return nullptr;
            }
            advance();  // consume :EndIf

            // Create IfK continuation
            IfK* if_k = new IfK(condition, then_branch, else_branch);
            machine_->heap->allocate_continuation(if_k);
            return if_k;
        }

        case TOK_WHILE: {
            // :While condition ... :EndWhile
            // Parse condition
            skip_separators();
            Continuation* condition = parse_expression(BP_NONE);
            if (!condition) {
                return nullptr;
            }

            skip_separators();

            // Parse loop body (statements until :EndWhile)
            std::vector<Continuation*> body_stmts;
            while (!at_end() && current().type != TOK_ENDWHILE) {
                Continuation* stmt = parse_expression(BP_NONE);
                if (!stmt) {
                    return nullptr;
                }
                body_stmts.push_back(stmt);
                skip_separators();
            }

            Continuation* body = nullptr;
            if (!body_stmts.empty()) {
                body = new SeqK(body_stmts);
                machine_->heap->allocate_continuation(body);
            }

            // Expect :EndWhile
            if (at_end() || current().type != TOK_ENDWHILE) {
                error_message_ = "Expected :EndWhile";
                return nullptr;
            }
            advance();  // consume :EndWhile

            // Create WhileK continuation
            WhileK* while_k = new WhileK(condition, body);
            machine_->heap->allocate_continuation(while_k);
            return while_k;
        }

        default:
            error_message_ = std::string("Unexpected token in prefix position: ") + token_type_name(token.type);
            return nullptr;
    }
}

// Left denotation: handles infix position (binary operators)
Continuation* Parser::led(Continuation* left, const Token& token) {
    // Handle assignment specially
    if (token.type == TOK_ASSIGN) {
        // Left side must be a LookupK (variable name)
        LookupK* lookup = dynamic_cast<LookupK*>(left);
        if (!lookup) {
            error_message_ = "Left side of assignment must be a variable name";
            return nullptr;
        }

        // Parse the right side (the value to assign)
        int bp = get_binding_power(token);
        Continuation* right = parse_expression(bp);

        if (!right) {
            return nullptr;
        }

        // Create AssignK continuation
        AssignK* assign = new AssignK(lookup->var_name.c_str(), right);
        machine_->heap->allocate_continuation(assign);

        return assign;
    }

    // Determine operator name for regular operators
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
        case TOK_RESHAPE:
            op_name = "⍴";
            break;
        case TOK_RAVEL:
            op_name = ",";
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
    // Assignment has lowest precedence
    switch (token.type) {
        case TOK_ASSIGN:
            return BP_ASSIGN;

        case TOK_PLUS:
        case TOK_MINUS:
        case TOK_TIMES:
        case TOK_POWER:  // * is an alias for ×
        case TOK_DIVIDE:
        case TOK_RESHAPE:
        case TOK_RAVEL:
        case TOK_IOTA:
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

// Check if token is a statement separator (Phase 3.3)
bool Parser::is_separator(const Token& token) const {
    return token.type == TOK_NEWLINE ||
           token.type == TOK_DIAMOND ||
           token.type == TOK_COMMENT;
}

// Skip statement separators (newlines, diamonds, comments)
void Parser::skip_separators() {
    while (!at_end() && is_separator(current_token_)) {
        advance();
    }
}

// Parse a multi-statement program (Phase 3.3)
Continuation* Parser::parse_program(const std::string& input) {
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

    // Skip any leading separators
    skip_separators();

    std::vector<Continuation*> statements;

    // Parse statements until EOF
    while (!at_end()) {
        // Parse one statement (expression)
        Continuation* stmt = parse_expression(BP_NONE);

        if (!stmt) {
            if (error_message_.empty()) {
                error_message_ = "Failed to parse statement";
            }
            delete lexer_;
            lexer_ = nullptr;
            return nullptr;
        }

        statements.push_back(stmt);

        // Skip statement separators
        skip_separators();
    }

    delete lexer_;
    lexer_ = nullptr;

    // Always wrap in SeqK for consistency
    // SeqK handles empty and single-statement cases efficiently
    auto* seq = new SeqK(statements);
    machine_->heap->allocate_continuation(seq);
    return seq;
}

} // namespace apl
