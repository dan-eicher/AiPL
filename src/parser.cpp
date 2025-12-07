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
const int BP_OUTER_PRODUCT = 10; // Outer product has low BP (doesn't grab array args)
const int BP_JUXTAPOSE = 20;     // Function application (juxtaposition)
const int BP_INNER_PRODUCT = 30; // Inner product has high BP (grabs function operands first)
const int BP_POSTFIX_OP = 50;    // Postfix operators (⍨, ¨, /, \) bind tightest

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
        auto* seq = machine->heap->allocate<SeqK>(statements);
        return seq;
    }
}

// Juxtaposition handler - implements G2 grammar juxtaposition
// Called when we see a token that can start a value in juxtaposition position
// G2 Grammar: fbn-term ::= fb-term fbn-term
// Semantics: if type(x₁) = bas then x₂(x₁) else x₁(x₂)
//
// IMPORTANT: Juxtaposition is LEFT-associative for G2 grammar!
// This ensures "2 f 3" parses as "(2 f) 3", giving correct argument order.
//
// Juxtaposition handler - implements G2 grammar juxtaposition
// ALL juxtaposition is function application (G2 rule)
// Strands are handled at lexer level for numeric literals only (ISO 13751)
Continuation* Parser::led_juxtapose(Continuation* left, int bp) {
    // G2 Rule 3: derived-operator fb → fb-term
    // If left is a derived operator from a DYADIC operator (like "." or "⍤"),
    // it should grab just fb (the function/value), not a full fbn-term.
    // This is distinct from Rule 1 (fb-term fbn-term) for general juxtaposition.
    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(left);
    if (derived && (strcmp(derived->op_name, ".") == 0 || strcmp(derived->op_name, "⍤") == 0)) {
        // Dyadic operator's derived form: parse right with high BP to get just the operand
        Continuation* right = parse_expression(BP_POSTFIX_OP);
        if (!right) {
            return nullptr;
        }
        return machine->heap->allocate<JuxtaposeK>(left, right);
    }

    // Parse the right operand
    // APL is right-associative: use bp (not bp+1) to continue parsing to the right
    // This builds right-associative structures for proper APL evaluation order
    Continuation* right = parse_expression(bp);
    if (!right) {
        return nullptr;
    }

    // G2 juxtaposition: fbn-term ::= fb-term fbn-term
    // Semantics: if type(x₁)=bas then x₂(x₁) else x₁(x₂)
    return machine->heap->allocate<JuxtaposeK>(left, right);
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

        // Check for juxtaposition (G2 grammar: fbn-term ::= fb-term fbn-term)
        // Juxtaposition occurs when we see a token that can start fb-term
        // G2: fb ::= identifier | ( expression )
        // Primitive functions are identifiers, so they trigger juxtaposition
        bool is_juxtaposition = false;

        // Special case: ∘. (outer product) always triggers juxtaposition
        // It has a binding power but should be treated as a prefix operator
        if (next.type == TOK_OUTER_PRODUCT) {
            is_juxtaposition = true;
            bp = BP_JUXTAPOSE;
        } else if (bp == BP_NONE) {
            // Tokens that can start fb: numbers, identifiers (including primitives), parentheses
            switch (next.type) {
                case TOK_NUMBER:
                case TOK_NUMBER_VECTOR:
                case TOK_LPAREN:
                case TOK_NAME:
                case TOK_ALPHA:  // ⍺ in dfns
                case TOK_OMEGA:  // ⍵ in dfns
                // Primitive function tokens are identifiers in G2 grammar
                case TOK_PLUS:
                case TOK_MINUS:
                case TOK_TIMES:
                case TOK_POWER:
                case TOK_DIVIDE:
                case TOK_RESHAPE:
                case TOK_RAVEL:
                case TOK_IOTA:
                case TOK_EQUAL:
                case TOK_NOT_EQUAL:
                case TOK_LESS:
                case TOK_LESS_EQUAL:
                case TOK_GREATER:
                case TOK_GREATER_EQUAL:
                case TOK_CEILING:
                case TOK_FLOOR:
                case TOK_AND:
                case TOK_OR:
                case TOK_NOT:
                case TOK_NAND:
                case TOK_NOR:
                case TOK_STILE:
                case TOK_LOG:
                case TOK_FACTORIAL:
                case TOK_TRANSPOSE:
                case TOK_TAKE:
                case TOK_DROP:
                case TOK_REVERSE:
                case TOK_REVERSE_FIRST:
                case TOK_TALLY:
                case TOK_MEMBER:
                case TOK_GRADE_UP:
                case TOK_GRADE_DOWN:
                case TOK_UNION:
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
            LiteralK* lit = machine->heap->allocate<LiteralK>(value);
            return lit;
        }

        case TOK_NUMBER_VECTOR: {
            // Numeric vector literal (ISO 13751): "1 2 3" → vector [1, 2, 3]
            // Map token data to Eigen vector and create Value
            Eigen::VectorXd vec = Eigen::Map<const Eigen::VectorXd>(
                token.vector_data, token.vector_size);
            Value* vec_val = machine->heap->allocate_vector(vec);
            // Create StrandK that holds this vector Value
            StrandK* strand = machine->heap->allocate<StrandK>(vec_val);
            return strand;
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
        case TOK_EQUAL:
        case TOK_NOT_EQUAL:
        case TOK_LESS:
        case TOK_LESS_EQUAL:
        case TOK_GREATER:
        case TOK_GREATER_EQUAL:
        case TOK_CEILING:
        case TOK_FLOOR:
        case TOK_AND:
        case TOK_OR:
        case TOK_NOT:
        case TOK_NAND:
        case TOK_NOR:
        case TOK_STILE:
        case TOK_LOG:
        case TOK_FACTORIAL:
        case TOK_TRANSPOSE:
        case TOK_TAKE:
        case TOK_DROP:
        case TOK_REVERSE:
        case TOK_REVERSE_FIRST:
        case TOK_TALLY:
        case TOK_MEMBER:
        case TOK_GRADE_UP:
        case TOK_GRADE_DOWN:
        case TOK_UNION: {
            // G2 Grammar: Primitive functions are identifiers (fb ::= identifier)
            // They are NOT special monadic operators in the grammar
            // Monadic behavior emerges from juxtaposition + runtime semantics
            // So ALWAYS create LookupK for primitive function tokens

            const char* op_name = nullptr;
            switch (token.type) {
                case TOK_PLUS:          op_name = "+"; break;
                case TOK_MINUS:         op_name = "-"; break;
                case TOK_TIMES:         op_name = "×"; break;
                case TOK_POWER:         op_name = "*"; break;
                case TOK_DIVIDE:        op_name = "÷"; break;
                case TOK_RESHAPE:       op_name = "⍴"; break;
                case TOK_RAVEL:         op_name = ","; break;
                case TOK_IOTA:          op_name = "⍳"; break;
                case TOK_EQUAL:         op_name = "="; break;
                case TOK_NOT_EQUAL:     op_name = "≠"; break;
                case TOK_LESS:          op_name = "<"; break;
                case TOK_LESS_EQUAL:    op_name = "≤"; break;
                case TOK_GREATER:       op_name = ">"; break;
                case TOK_GREATER_EQUAL: op_name = "≥"; break;
                case TOK_CEILING:       op_name = "⌈"; break;
                case TOK_FLOOR:         op_name = "⌊"; break;
                case TOK_AND:           op_name = "∧"; break;
                case TOK_OR:            op_name = "∨"; break;
                case TOK_NOT:           op_name = "~"; break;
                case TOK_NAND:          op_name = "⍲"; break;
                case TOK_NOR:           op_name = "⍱"; break;
                case TOK_STILE:         op_name = "|"; break;
                case TOK_LOG:           op_name = "⍟"; break;
                case TOK_FACTORIAL:     op_name = "!"; break;
                case TOK_TRANSPOSE:     op_name = "⍉"; break;
                case TOK_TAKE:          op_name = "↑"; break;
                case TOK_DROP:          op_name = "↓"; break;
                case TOK_REVERSE:       op_name = "⌽"; break;
                case TOK_REVERSE_FIRST: op_name = "⊖"; break;
                case TOK_TALLY:         op_name = "≢"; break;
                case TOK_MEMBER:        op_name = "∊"; break;
                case TOK_GRADE_UP:      op_name = "⍋"; break;
                case TOK_GRADE_DOWN:    op_name = "⍒"; break;
                case TOK_UNION:         op_name = "∪"; break;
                default: break;
            }

            const char* interned_name = machine->string_pool.intern(op_name);
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            return lookup;
        }

        case TOK_NAME: {
            // Variable reference - create LookupK
            // Intern the name in the string pool
            const char* interned_name = machine->string_pool.intern(token.name);
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            return lookup;
        }

        case TOK_ALPHA: {
            // ⍺ (alpha) - left argument in dfn
            const char* interned_name = machine->string_pool.intern("⍺");
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            return lookup;
        }

        case TOK_OMEGA: {
            // ⍵ (omega) - right argument in dfn
            const char* interned_name = machine->string_pool.intern("⍵");
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
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
            ClosureLiteralK* closure_lit = machine->heap->allocate<ClosureLiteralK>(body);
            return closure_lit;
        }

        case TOK_OUTER_PRODUCT: {
            // Outer product: ∘.f where f is a function
            // Parse the function operand with high binding power to get just the function
            Continuation* fn_operand = parse_expression(BP_POSTFIX_OP);
            if (!fn_operand) {
                error_message_ = "Outer product operator requires a function operand";
                return nullptr;
            }

            const char* interned_name = machine->string_pool.intern("∘.");
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(fn_operand, interned_name);
            return derived;
        }

        default:
            error_message_ = std::string("Unexpected token in prefix position: ") + token_type_name(token.type);
            return nullptr;
    }
}

// Left denotation: handles infix position (binary operators)
Continuation* Parser::led(Continuation* left, const Token& token) {
    switch (token.type) {
        case TOK_ASSIGN: {
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
            AssignK* assign = machine->heap->allocate<AssignK>(lookup->var_name, right);
            return assign;
        }

        case TOK_LBRACE: {
            // Handle dfn application (e.g., "3 {⍺+⍵} 5")
            // NOTE: { has already been consumed by parse_expression before calling led()
            Continuation* body = parse_dfn_body();
            if (!body) {
                return nullptr;
            }

            // Create ClosureLiteralK for the dfn
            ClosureLiteralK* closure_lit = machine->heap->allocate<ClosureLiteralK>(body);

            // Parse the right argument
            int bp = get_binding_power(token);  // Use operator binding power for dfns
            Continuation* right = parse_expression(bp);

            if (!right) {
                return nullptr;
            }

            // Create ApplyFunctionK for dyadic application: left {dfn} right
            ApplyFunctionK* apply = machine->heap->allocate<ApplyFunctionK>(closure_lit, left, right);
            return apply;
        }

        // Postfix monadic operators (¨, ⍨, /, \, ⌿, ⍀)
        // G2 Grammar: fb-term ::= fb-term monadic-operator
        // Evaluates as: x₂(x₁) - operator applied to operand
        case TOK_EACH:
        case TOK_COMMUTE:
        case TOK_REDUCE:
        case TOK_SCAN:
        case TOK_REDUCE_FIRST:
        case TOK_SCAN_FIRST: {
            const char* op_name = nullptr;
            switch (token.type) {
                case TOK_EACH:         op_name = "¨";  break;
                case TOK_COMMUTE:      op_name = "⍨";  break;
                case TOK_REDUCE:       op_name = "/";  break;
                case TOK_SCAN:         op_name = "\\"; break;
                case TOK_REDUCE_FIRST: op_name = "⌿";  break;
                case TOK_SCAN_FIRST:   op_name = "⍀";  break;
                default: break;
            }

            const char* interned_name = machine->string_pool.intern(op_name);

            // Create DerivedOperatorK: evaluate operand (left), then apply operator
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(left, interned_name);
            return derived;
        }

        case TOK_OUTER_PRODUCT: {
            // Outer product: A ∘.f B where A and B are arrays, f is a function
            // In "3 ∘.× 5", left=3 (array), we need to parse × (function)
            Continuation* fn_operand = parse_expression(BP_POSTFIX_OP);
            if (!fn_operand) {
                error_message_ = "Outer product operator requires a function operand";
                return nullptr;
            }

            const char* interned_name = machine->string_pool.intern("∘.");

            // Create derived operator from ∘. and the function
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(fn_operand, interned_name);

            // Now create juxtaposition: left (∘.f)
            // This applies ∘.f monadically to left, which will curry waiting for right argument
            JuxtaposeK* jux = machine->heap->allocate<JuxtaposeK>(left, derived);
            return jux;
        }

        case TOK_DOT: {
            // Inner product (.)
            // G2 Grammar Rule 4: fb-term dyadic-operator → derived-operator
            // The second operand will be delivered via juxtaposition (Rule 3: derived-operator fb → fb-term)
            const char* interned_name = machine->string_pool.intern(".");
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(left, interned_name);
            return derived;
        }

        case TOK_RANK: {
            // Rank operator (⍤)
            // f⍤k applies function f to k-cells of the argument(s)
            // Dyadic operator: first operand is function, second is rank specification
            const char* interned_name = machine->string_pool.intern("⍤");
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(left, interned_name);
            return derived;
        }

        default:
            // Unexpected token in infix position
            error_message_ = std::string("Unexpected token in infix position: ") + token_type_name(token.type);
            return nullptr;
    }
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
        body = machine->heap->allocate<LiteralK>(0.0);
    } else if (statements.size() == 1) {
        // Single expression - use directly
        body = statements[0];
    } else {
        // Multiple statements - wrap in SeqK
        body = machine->heap->allocate<SeqK>(statements);
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
        then_branch = machine->heap->allocate<SeqK>(then_stmts);
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
            else_branch = machine->heap->allocate<SeqK>(else_stmts);
        }
    }

    // Expect :EndIf
    if (at_end() || current().type != TOK_ENDIF) {
        error_message_ = "Expected :EndIf";
        return nullptr;
    }
    advance();  // consume :EndIf

    // Create IfK continuation
    IfK* if_k = machine->heap->allocate<IfK>(condition, then_branch, else_branch);
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
        body = machine->heap->allocate<SeqK>(body_stmts);
    }

    // Expect :EndWhile
    if (at_end() || current().type != TOK_ENDWHILE) {
        error_message_ = "Expected :EndWhile";
        return nullptr;
    }
    advance();  // consume :EndWhile

    // Create WhileK continuation
    WhileK* while_k = machine->heap->allocate<WhileK>(condition, body);
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
    const char* var_name = machine->string_pool.intern(current().name);
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
        body = machine->heap->allocate<SeqK>(body_stmts);
    }

    // Expect :EndFor
    if (at_end() || current().type != TOK_ENDFOR) {
        error_message_ = "Expected :EndFor";
        return nullptr;
    }
    advance();  // consume :EndFor

    // Create ForK continuation
    ForK* for_k = machine->heap->allocate<ForK>(var_name, array_expr, body);
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
    ReturnK* return_k = machine->heap->allocate<ReturnK>(value_expr);
    return return_k;
}

// Parse :Leave statement
Continuation* Parser::parse_leave_statement() {
    // :Leave has already been consumed by parse_statement()
    // Create LeaveK continuation
    LeaveK* leave_k = machine->heap->allocate<LeaveK>();
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
    // G2 Grammar: Primitive functions are identifiers, not operators
    // Operators (/, \, ¨, ⍨, ., ∘.) are distinct tokens with binding power
    // Primitive functions have no binding power - they trigger juxtaposition
    switch (token.type) {
        case TOK_ASSIGN:
            return BP_ASSIGN;

        // Primitive functions are identifiers in G2 - they have no binding power
        // Juxtaposition + currying produces infix behavior at runtime
        case TOK_PLUS:
        case TOK_MINUS:
        case TOK_TIMES:
        case TOK_POWER:
        case TOK_DIVIDE:
        case TOK_RESHAPE:
        case TOK_RAVEL:
        case TOK_IOTA:
        case TOK_EQUAL:
            return BP_NONE;  // Treat as identifiers

        case TOK_LBRACE:  // Dfns can be used as dyadic functions
            return BP_JUXTAPOSE;

        // Postfix monadic operators bind tighter than infix
        case TOK_EACH:
        case TOK_COMMUTE:
        case TOK_REDUCE:
        case TOK_SCAN:
        case TOK_REDUCE_FIRST:
        case TOK_SCAN_FIRST:
            return BP_POSTFIX_OP;

        // Infix dyadic operators (different binding powers)
        case TOK_DOT:
        case TOK_RANK:
            return BP_INNER_PRODUCT;  // High BP: grab function operands before juxtaposition
        case TOK_OUTER_PRODUCT:
            return BP_OUTER_PRODUCT;  // Low BP: don't steal array arguments

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
