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

// Parse entry point - unified for both single expressions and multi-statement programs
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

    // Skip any leading separators
    skip_separators();

    std::vector<Continuation*> statements;

    // Parse statements until EOF
    while (!at_end()) {
        // Parse one statement (control flow or expression)
        Continuation* stmt = parse_statement();

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

    // Handle single statement case: return it directly (backward compatibility)
    // Handle multiple statements: wrap in SeqK
    if (statements.size() == 1) {
        return statements[0];
    } else {
        auto* seq = machine_->heap->allocate<SeqK>(statements);
        return seq;
    }
}

// Juxtaposition handler - forms strands from adjacent values
// Called when we see a token that can start a value in juxtaposition position
Continuation* Parser::led_juxtapose(Continuation* left, int bp) {
    // Parse the right operand (don't consume token - it's already current)
    Continuation* right = parse_expression(bp);
    if (!right) {
        return nullptr;
    }

    // Build strand from left and right
    // Per Grammar G2, all values/functions share same syntactic category
    // Juxtaposition ALWAYS means strand at parse time
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

    return machine_->heap->allocate<StrandK>(elements);
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
            // Juxtaposition: form a strand
            left = led_juxtapose(left, bp);
            if (!left) {
                return nullptr;
            }
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
            LiteralK* lit = machine_->heap->allocate<LiteralK>(value);
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
        case TOK_IOTA:
        case TOK_EQUAL: {
            // Monadic operator in prefix position
            // Parse the operand and create MonadicK continuation

            // Determine operator name and intern it
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
                case TOK_EQUAL:   op_name = "="; break;
                default: break;
            }

            // Intern the operator name
            const char* interned_name = machine_->string_pool.intern(op_name);

            // Parse the operand with high binding power (monadic binds tighter than dyadic)
            Continuation* operand = parse_expression(BP_OPERATOR + 50);
            if (!operand) {
                return nullptr;
            }

            // Create MonadicK continuation (lookup deferred to evaluation time)
            MonadicK* monadic = machine_->heap->allocate<MonadicK>(interned_name, operand);
            return monadic;
        }

        case TOK_NAME: {
            // Variable reference - create LookupK
            // Intern the name in the string pool
            const char* interned_name = machine_->string_pool.intern(token.name);
            LookupK* lookup = machine_->heap->allocate<LookupK>(interned_name);
            return lookup;
        }

        case TOK_ALPHA: {
            // ⍺ (alpha) - left argument in dfn
            LookupK* lookup = machine_->heap->allocate<LookupK>("⍺");
            return lookup;
        }

        case TOK_OMEGA: {
            // ⍵ (omega) - right argument in dfn
            LookupK* lookup = machine_->heap->allocate<LookupK>("⍵");
            return lookup;
        }

        case TOK_LBRACE: {
            // Dfn definition: {body}
            // NOTE: { has already been consumed by parse_expression before calling nud()
            Continuation* body = parse_dfn_body();
            if (!body) {
                return nullptr;
            }

            // Create ClosureLiteralK with the body
            ClosureLiteralK* closure_lit = machine_->heap->allocate<ClosureLiteralK>(body);
            return closure_lit;
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

        // Create AssignK continuation (var_name is already interned from LookupK)
        AssignK* assign = machine_->heap->allocate<AssignK>(lookup->var_name, right);

        return assign;
    }

    // Handle dfn application (e.g., "3 {⍺+⍵} 5")
    if (token.type == TOK_LBRACE) {
        // Parse the dfn body
        // NOTE: { has already been consumed by parse_expression before calling led()
        Continuation* body = parse_dfn_body();
        if (!body) {
            return nullptr;
        }

        // Create ClosureLiteralK for the dfn
        ClosureLiteralK* closure_lit = machine_->heap->allocate<ClosureLiteralK>(body);

        // Parse the right argument
        int bp = get_binding_power(token);  // Use operator binding power for dfns
        Continuation* right = parse_expression(bp);

        if (!right) {
            return nullptr;
        }

        // Create ApplyFunctionK for dyadic application: left {dfn} right
        ApplyFunctionK* apply = machine_->heap->allocate<ApplyFunctionK>(closure_lit, left, right);

        return apply;
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
        case TOK_EQUAL:
            op_name = "=";
            break;
        default:
            error_message_ = std::string("Unexpected token in infix position: ") + token_type_name(token.type);
            return nullptr;
    }

    // Intern the operator name
    const char* interned_name = machine_->string_pool.intern(op_name);

    // For right-associative operators, we parse the right side with the SAME binding power
    // This is the key to right-to-left evaluation in Pratt parsing
    int bp = get_binding_power(token);
    Continuation* right = parse_expression(bp);

    if (!right) {
        return nullptr;
    }

    // Create dyadic operation continuation (lookup deferred to evaluation time)
    DyadicK* dyadic = machine_->heap->allocate<DyadicK>(interned_name, left, right);

    return dyadic;
}

// Parse dfn body: parses statements from current position until }
// Assumes { has already been consumed
// Consumes the closing }
// Returns the body continuation (wrapped in SeqK if multiple statements)
Continuation* Parser::parse_dfn_body() {
    skip_separators();

    // Parse statements until } (dfns use expressions, not statements)
    std::vector<Continuation*> statements;
    while (!at_end() && current().type != TOK_RBRACE) {
        Continuation* stmt = parse_expression(BP_NONE);
        if (!stmt) {
            return nullptr;
        }
        statements.push_back(stmt);
        skip_separators();
    }

    // Expect closing brace
    if (current().type != TOK_RBRACE) {
        error_message_ = "Expected } to close dfn";
        return nullptr;
    }
    advance();  // consume }

    // Create body continuation
    Continuation* body;
    if (statements.empty()) {
        // Empty dfn body - return 0
        body = machine_->heap->allocate<LiteralK>(0.0);
    } else if (statements.size() == 1) {
        // Single expression - use directly
        body = statements[0];
    } else {
        // Multiple statements - wrap in SeqK
        body = machine_->heap->allocate<SeqK>(statements);
    }

    return body;
}

// Parse a block of statements until reaching a terminator token
// Returns a vector of parsed statements
// Does NOT consume the terminator token
std::vector<Continuation*> Parser::parse_block_until(TokenType terminator) {
    std::vector<Continuation*> statements;

    while (!at_end() && current().type != terminator) {
        Continuation* stmt = parse_statement();
        if (!stmt) {
            // Return empty vector on parse failure
            return std::vector<Continuation*>();
        }
        statements.push_back(stmt);
        skip_separators();
    }

    return statements;
}

// Parse :If statement
Continuation* Parser::parse_if_statement() {
    // :If has already been consumed by parse_statement()
    skip_separators();
    Continuation* condition = parse_expression(BP_NONE);
    if (!condition) {
        return nullptr;
    }

    skip_separators();

    // Parse then-branch (statements until :Else or :EndIf)
    // Can't use parse_block_until here because we need to stop at TWO different tokens
    std::vector<Continuation*> then_stmts;
    while (!at_end() && current().type != TOK_ELSE && current().type != TOK_ENDIF) {
        Continuation* stmt = parse_statement();
        if (!stmt) {
            return nullptr;
        }
        then_stmts.push_back(stmt);
        skip_separators();
    }

    Continuation* then_branch = nullptr;
    if (!then_stmts.empty()) {
        then_branch = machine_->heap->allocate<SeqK>(then_stmts);
    }

    // Check for :Else
    Continuation* else_branch = nullptr;
    if (!at_end() && current().type == TOK_ELSE) {
        advance();  // consume :Else
        skip_separators();

        // Parse else-branch (statements until :EndIf)
        std::vector<Continuation*> else_stmts = parse_block_until(TOK_ENDIF);
        if (else_stmts.empty() && !at_end() && current().type != TOK_ENDIF) {
            // parse_block_until returned empty due to error
            return nullptr;
        }

        if (!else_stmts.empty()) {
            else_branch = machine_->heap->allocate<SeqK>(else_stmts);
        }
    }

    // Expect :EndIf
    if (at_end() || current().type != TOK_ENDIF) {
        error_message_ = "Expected :EndIf";
        return nullptr;
    }
    advance();  // consume :EndIf

    // Create IfK continuation
    IfK* if_k = machine_->heap->allocate<IfK>(condition, then_branch, else_branch);
    return if_k;
}

// Parse :While statement
Continuation* Parser::parse_while_statement() {
    // :While has already been consumed by parse_statement()
    skip_separators();
    Continuation* condition = parse_expression(BP_NONE);
    if (!condition) {
        return nullptr;
    }

    skip_separators();

    // Parse loop body (statements until :EndWhile)
    std::vector<Continuation*> body_stmts = parse_block_until(TOK_ENDWHILE);
    if (body_stmts.empty() && !at_end() && current().type != TOK_ENDWHILE) {
        // parse_block_until returned empty due to error
        return nullptr;
    }

    Continuation* body = nullptr;
    if (!body_stmts.empty()) {
        body = machine_->heap->allocate<SeqK>(body_stmts);
    }

    // Expect :EndWhile
    if (at_end() || current().type != TOK_ENDWHILE) {
        error_message_ = "Expected :EndWhile";
        return nullptr;
    }
    advance();  // consume :EndWhile

    // Create WhileK continuation
    WhileK* while_k = machine_->heap->allocate<WhileK>(condition, body);
    return while_k;
}

// Parse :For statement
Continuation* Parser::parse_for_statement() {
    // :For has already been consumed by parse_statement()
    skip_separators();

    // Expect variable name
    if (at_end() || current().type != TOK_NAME) {
        error_message_ = "Expected variable name after :For";
        return nullptr;
    }
    // Intern the variable name in the string pool
    const char* var_name = machine_->string_pool.intern(current().name);
    advance();  // consume variable name

    skip_separators();

    // Expect :In
    if (at_end() || current().type != TOK_IN) {
        error_message_ = "Expected :In after variable name";
        return nullptr;
    }
    advance();  // consume :In

    skip_separators();

    // Parse array expression
    Continuation* array_expr = parse_expression(BP_NONE);
    if (!array_expr) {
        return nullptr;
    }

    skip_separators();

    // Parse loop body (statements until :EndFor)
    std::vector<Continuation*> body_stmts = parse_block_until(TOK_ENDFOR);
    if (body_stmts.empty() && !at_end() && current().type != TOK_ENDFOR) {
        // parse_block_until returned empty due to error
        return nullptr;
    }

    Continuation* body = nullptr;
    if (!body_stmts.empty()) {
        body = machine_->heap->allocate<SeqK>(body_stmts);
    }

    // Expect :EndFor
    if (at_end() || current().type != TOK_ENDFOR) {
        error_message_ = "Expected :EndFor";
        return nullptr;
    }
    advance();  // consume :EndFor

    // Create ForK continuation
    ForK* for_k = machine_->heap->allocate<ForK>(var_name, array_expr, body);
    return for_k;
}

// Parse :Return statement
Continuation* Parser::parse_return_statement() {
    // :Return has already been consumed by parse_statement()
    skip_separators();

    // Check if there's a return value
    Continuation* value_expr = nullptr;
    if (!at_end() && !is_separator(current())) {
        // Parse return value expression
        value_expr = parse_expression(BP_NONE);
        if (!value_expr) {
            return nullptr;
        }
    }

    // Create ReturnK continuation
    ReturnK* return_k = machine_->heap->allocate<ReturnK>(value_expr);
    return return_k;
}

// Parse :Leave statement
Continuation* Parser::parse_leave_statement() {
    // :Leave has already been consumed by parse_statement()
    // Create LeaveK continuation
    LeaveK* leave_k = machine_->heap->allocate<LeaveK>();
    return leave_k;
}

// Parse statement: checks for control flow keywords, otherwise falls through to parse_expression
Continuation* Parser::parse_statement() {
    // Check if current token is a control flow keyword
    switch (current().type) {
        case TOK_IF:
            advance();  // consume :If
            return parse_if_statement();

        case TOK_WHILE:
            advance();  // consume :While
            return parse_while_statement();

        case TOK_FOR:
            advance();  // consume :For
            return parse_for_statement();

        case TOK_RETURN:
            advance();  // consume :Return
            return parse_return_statement();

        case TOK_LEAVE:
            advance();  // consume :Leave
            return parse_leave_statement();

        default:
            // Not a statement keyword - parse as expression
            return parse_expression(BP_NONE);
    }
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
        case TOK_LBRACE:  // Dfns can be used as dyadic functions
            return BP_OPERATOR;

        // Closing delimiters should never be treated as infix
        // Give them negative binding power to stop parsing
        case TOK_RPAREN:
        case TOK_RBRACE:
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

} // namespace apl
