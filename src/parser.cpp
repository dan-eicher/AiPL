// APL Parser implementation using manual Pratt parsing

#include "parser.h"
#include "machine.h"
#include "continuation.h"
#include <stdexcept>
#include <cstdlib>

namespace apl {

// ============================================================================
// Binding Powers
// ============================================================================

// Binding powers for operators (higher = tighter binding)
// APL has uniform precedence, so all operators have the same binding power
// This gives us right-to-left evaluation
const int BP_NONE = 0;
const int BP_ASSIGN = 5;         // Assignment has lowest precedence
const int BP_OUTER_PRODUCT = 10; // Outer product has low BP (doesn't grab array args)
const int BP_JUXTAPOSE = 20;     // Function application (juxtaposition)
const int BP_INNER_PRODUCT = 30; // Inner product has high BP (grabs function operands first)
const int BP_POSTFIX_OP = 50;    // Postfix operators (⍨, ¨, /, \) bind tightest

// ============================================================================
// Main Entry Point
// ============================================================================

Continuation* Parser::parse(const std::string& input) {
    error_message_.clear();

    // Keep input alive for lexer lifetime
    input_ = input;

    // Create lexer for on-demand tokenization
    lexer_ = new Lexer(input_.c_str());

    // Get first token
    current_token_ = lexer_->next_token();

    if (current_token_.type == TOK_ERROR) {
        set_error(std::string("unexpected character '") + current_token_.name + "'", current_token_);
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
        Continuation* stmt = parse_expression(BP_NONE);

        if (!stmt) {
            if (error_message_.empty()) {
                set_error("failed to parse expression");
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
    // If left is a derived operator from a DYADIC operator that needs a second OPERAND,
    // it should grab just fb (the function/value), not a full fbn-term.
    // This is distinct from Rule 1 (fb-term fbn-term) for general juxtaposition.
    //
    // Important distinction:
    // - Operators like "." (inner product) and "⍤" (rank) need a second OPERAND from parsing
    // - Operators like "/" (reduce) have dyadic APPLICATION (N f/ B) but don't need an operand
    // - User-defined dyadic operators (FF OP GG) need the second operand GG from parsing
    DerivedOperatorK* derived = dynamic_cast<DerivedOperatorK*>(left);
    if (derived) {
        // Check if this operator needs a second operand from parsing
        bool needs_second_operand = false;

        // Built-in operators that need second operand: "." and "⍤"
        if (strcmp(derived->op_name, ".") == 0 || strcmp(derived->op_name, "⍤") == 0) {
            needs_second_operand = true;
        } else {
            // Check for user-defined dyadic operators
            Value* op_val = machine->env->lookup(derived->op_name);
            if (op_val && op_val->is_defined_operator()) {
                needs_second_operand = op_val->data.defined_op_data->is_dyadic_operator;
            }
        }

        if (needs_second_operand) {
            // Dyadic operator's derived form: parse right with high BP to get just the operand
            Token jux_token = current_token_;  // Save location before parsing
            Continuation* right = parse_expression(BP_POSTFIX_OP);
            if (!right) {
                return nullptr;
            }
            JuxtaposeK* jux = machine->heap->allocate<JuxtaposeK>(left, right);
            jux->set_location(jux_token.line, jux_token.column);
            return jux;
        }
    }

    // Parse the right operand
    // APL is right-associative: use bp (not bp+1) to continue parsing to the right
    // This builds right-associative structures for proper APL evaluation order
    Token jux_token = current_token_;  // Save location before parsing
    Continuation* right = parse_expression(bp);
    if (!right) {
        return nullptr;
    }

    // G2 juxtaposition: fbn-term ::= fb-term fbn-term
    // Semantics: if type(x₁)=bas then x₂(x₁) else x₁(x₂)
    JuxtaposeK* jux = machine->heap->allocate<JuxtaposeK>(left, right);
    jux->set_location(jux_token.line, jux_token.column);
    return jux;
}

// Core Pratt parsing algorithm
Continuation* Parser::parse_expression(int min_bp) {
    if (at_end()) {
        set_error("unexpected end of input");
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

        // Check if TOK_NAME is a defined operator - operators have higher precedence
        if (next.type == TOK_NAME && bp == BP_NONE) {
            const char* interned = machine->string_pool.intern(next.name);
            Value* val = machine->env->lookup(interned);
            if (val && val->is_defined_operator()) {
                // Defined operator: use operator binding power, handled in led
                bp = BP_POSTFIX_OP;
            }
        }

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
                case TOK_STRING:
                case TOK_LPAREN:
                case TOK_NAME:
                case TOK_ALPHA:  // ⍺ in dfns
                case TOK_OMEGA:  // ⍵ in dfns
                case TOK_ALPHA_ALPHA:  // ⍺⍺ in defined operators
                case TOK_OMEGA_OMEGA:  // ⍵⍵ in defined operators
                case TOK_DEL:    // ∇ in recursive dfns
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
                case TOK_CIRCLE:
                case TOK_QUESTION:
                case TOK_DECODE:
                case TOK_ENCODE:
                case TOK_DOMINO:
                case TOK_EXECUTE:
                case TOK_FORMAT:
                case TOK_TABLE:
                case TOK_SQUAD:
                case TOK_MATCH:
                case TOK_LEFT_TACK:
                case TOK_RIGHT_TACK:
                case TOK_ZILDE:  // ⍬ (empty vector)
                case TOK_QUAD_NAME:  // ⎕IO, ⎕PP (system variables)
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
        // Also: if bp=0 and not juxtaposition, there's no led for this token
        if (bp < min_bp || (bp == 0 && !is_juxtaposition)) {
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

// ============================================================================
// Null Denotation (Prefix Handling)
// ============================================================================

Continuation* Parser::nud(const Token& token) {
    switch (token.type) {
        case TOK_NUMBER: {
            // Single number - create LiteralK
            double value = token.number;
            LiteralK* lit = machine->heap->allocate<LiteralK>(value);
            lit->set_location(token.line, token.column);
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
            strand->set_location(token.line, token.column);
            return strand;
        }

        case TOK_STRING: {
            // String literal: 'hello' → string value
            Value* str_val = machine->heap->allocate_string(token.name);
            StrandK* strand = machine->heap->allocate<StrandK>(str_val);
            strand->set_location(token.line, token.column);
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
                set_error("expected ')' after expression");
                return nullptr;
            }
            advance();  // consume ')'

            // Wrap in FinalizeK with finalize_gprime=false
            // This finalizes DYADIC_CURRY (like +/1 2 3) to values, but preserves
            // G_PRIME partial applications (like 2×) per ISO 13751 semantics
            FinalizeK* finalize = machine->heap->allocate<FinalizeK>(inner, false);
            finalize->set_location(token.line, token.column);
            return finalize;
        }

        case TOK_RAVEL: {
            // Special handling for ,[k] (catenate/laminate with axis)
            if (current_token_.type == TOK_LBRACKET) {
                advance();  // consume [
                Continuation* inner = parse_expression(0);
                if (!inner) {
                    set_error("expected axis expression after '['");
                    return nullptr;
                }
                if (current_token_.type != TOK_RBRACKET) {
                    set_error("expected ']' after axis expression");
                    return nullptr;
                }
                advance();  // consume ]
                // Wrap in FinalizeK to ensure the axis expression is fully evaluated
                Continuation* axis_cont = machine->heap->allocate<FinalizeK>(inner, true);
                // Create DerivedOperatorK with op_catenate_axis
                // axis_cont is the first (and only) operand
                const char* interned_name = machine->string_pool.intern(",⌷");
                DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(axis_cont, interned_name);
                derived->set_location(token.line, token.column);
                return derived;
            }
            // No axis - fall through to normal comma handling
            const char* interned_name = machine->string_pool.intern(",");
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            lookup->set_location(token.line, token.column);
            return lookup;
        }

        case TOK_MINUS:
        case TOK_PLUS:
        case TOK_TIMES:
        case TOK_POWER:
        case TOK_DIVIDE:
        case TOK_RESHAPE:
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
        case TOK_CIRCLE:
        case TOK_QUESTION:
        case TOK_DECODE:
        case TOK_ENCODE:
        case TOK_DOMINO:
        case TOK_EXECUTE:
        case TOK_FORMAT:
        case TOK_TABLE:
        case TOK_SQUAD:
        case TOK_MATCH:
        case TOK_LEFT_TACK:
        case TOK_RIGHT_TACK: {
            // G2 Grammar: Primitive functions are identifiers (fb ::= identifier)
            // They are NOT special monadic operators in the grammar
            // Monadic behavior emerges from juxtaposition + runtime semantics
            // So ALWAYS create LookupK for primitive function tokens

            const char* interned_name = machine->string_pool.intern(token_type_name(token.type));
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            lookup->set_location(token.line, token.column);
            return lookup;
        }

        case TOK_NAME: {
            // Variable reference or assignment
            // Intern the name in the string pool
            const char* interned_name = machine->string_pool.intern(token.name);

            // Check for assignment: NAME ← VALUE
            // Assignment binds only the immediate name, not larger expressions
            // So "1+X←5" parses as "1+(X←5)", not "(1+X)←5"
            if (current_token_.type == TOK_ASSIGN) {
                advance();  // consume ←
                Continuation* value = parse_expression(BP_ASSIGN);
                if (!value) {
                    return nullptr;
                }
                AssignK* assign = machine->heap->allocate<AssignK>(interned_name, value);
                assign->set_location(token.line, token.column);
                return assign;
            }

            // Just a variable reference
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            lookup->set_location(token.line, token.column);
            return lookup;
        }

        case TOK_ALPHA: {
            // ⍺ (alpha) - left argument in dfn
            dfn_uses_alpha = true;  // Track for niladic detection
            const char* interned_name = machine->string_pool.intern("⍺");
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            lookup->set_location(token.line, token.column);
            return lookup;
        }

        case TOK_OMEGA: {
            // ⍵ (omega) - right argument in dfn
            dfn_uses_omega = true;  // Track for niladic detection
            const char* interned_name = machine->string_pool.intern("⍵");
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            lookup->set_location(token.line, token.column);
            return lookup;
        }

        case TOK_ALPHA_ALPHA: {
            // ⍺⍺ - left operand in defined operator
            const char* interned_name = machine->string_pool.intern("⍺⍺");
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            lookup->set_location(token.line, token.column);
            return lookup;
        }

        case TOK_OMEGA_OMEGA: {
            // ⍵⍵ - right operand in defined operator
            const char* interned_name = machine->string_pool.intern("⍵⍵");
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            lookup->set_location(token.line, token.column);
            return lookup;
        }

        case TOK_DEL: {
            // ∇ (del) - self-reference in recursive dfn
            const char* interned_name = machine->string_pool.intern("∇");
            LookupK* lookup = machine->heap->allocate<LookupK>(interned_name);
            lookup->set_location(token.line, token.column);
            return lookup;
        }

        case TOK_ZILDE: {
            // ⍬ (zilde) - empty numeric vector
            Eigen::VectorXd empty_vec(0);
            Value* empty_val = machine->heap->allocate_vector(empty_vec);
            StrandK* strand = machine->heap->allocate<StrandK>(empty_val);
            strand->set_location(token.line, token.column);
            return strand;
        }

        case TOK_QUAD_NAME: {
            // System variable reference or assignment (⎕IO, ⎕PP, etc.)
            SysVarId var_id = lookup_sysvar(token.name, machine->sysvar_mask);
            if (var_id == SysVarId::INVALID) {
                set_error(std::string("unknown or disabled system variable: ⎕") + token.name, token);
                return nullptr;
            }

            // Check for assignment: ⎕IO ← VALUE
            if (current_token_.type == TOK_ASSIGN) {
                advance();  // consume ←
                Continuation* value = parse_expression(BP_ASSIGN);
                if (!value) {
                    return nullptr;
                }
                SysVarAssignK* assign = machine->heap->allocate<SysVarAssignK>(var_id, value);
                assign->set_location(token.line, token.column);
                return assign;
            }

            // Just a system variable read
            SysVarReadK* read = machine->heap->allocate<SysVarReadK>(var_id);
            read->set_location(token.line, token.column);
            return read;
        }

        case TOK_LBRACE: {
            // Dfn definition: {body}
            // NOTE: { has already been consumed by parse_expression before calling nud()
            ClosureLiteralK* closure_lit = parse_dfn();
            if (!closure_lit) {
                return nullptr;
            }
            closure_lit->set_location(token.line, token.column);
            return closure_lit;
        }

        case TOK_OUTER_PRODUCT: {
            // Outer product: ∘.f where f is a function
            // Parse the function operand with high binding power to get just the function
            Continuation* fn_operand = parse_expression(BP_POSTFIX_OP);
            if (!fn_operand) {
                set_error("outer product operator requires a function operand", token);
                return nullptr;
            }

            const char* interned_name = machine->string_pool.intern("∘.");
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(fn_operand, interned_name);
            derived->set_location(token.line, token.column);
            return derived;
        }

        case TOK_GOTO: {
            // Branch operator: →target
            // →0 or →⍬ exits the function
            Continuation* target = parse_expression(BP_NONE);
            if (!target) {
                set_error("branch operator requires a target expression", token);
                return nullptr;
            }
            BranchK* branch = machine->heap->allocate<BranchK>(target);
            branch->set_location(token.line, token.column);
            return branch;
        }

        // Control flow keywords can appear in dfn bodies too
        case TOK_IF: {
            Continuation* result = parse_if_statement();
            if (result) result->set_location(token.line, token.column);
            return result;
        }

        case TOK_WHILE: {
            Continuation* result = parse_while_statement();
            if (result) result->set_location(token.line, token.column);
            return result;
        }

        case TOK_FOR: {
            Continuation* result = parse_for_statement();
            if (result) result->set_location(token.line, token.column);
            return result;
        }

        case TOK_RETURN: {
            Continuation* result = parse_return_statement();
            if (result) result->set_location(token.line, token.column);
            return result;
        }

        case TOK_LEAVE: {
            Continuation* result = parse_leave_statement();
            if (result) result->set_location(token.line, token.column);
            return result;
        }

        case TOK_CONTINUE: {
            Continuation* result = parse_continue_statement();
            if (result) result->set_location(token.line, token.column);
            return result;
        }

        default:
            set_error(std::string("unexpected token in prefix position: ") + token_type_name(token.type), token);
            return nullptr;
    }
}

// ============================================================================
// Left Denotation (Infix Handling)
// ============================================================================

Continuation* Parser::led(Continuation* left, const Token& token) {
    switch (token.type) {
        case TOK_ASSIGN: {
            // Parse the right side (the value to assign)
            int bp = get_binding_power(token);
            Continuation* right = parse_expression(bp);

            if (!right) {
                return nullptr;
            }

            // Check for indexed assignment: A[I]←V
            // A[I] is parsed as JuxtaposeK(I, JuxtaposeK(⌷, A)) for proper curry handling
            JuxtaposeK* outer = dynamic_cast<JuxtaposeK*>(left);
            if (outer) {
                JuxtaposeK* inner = dynamic_cast<JuxtaposeK*>(outer->right);
                if (inner) {
                    LookupK* squad_lookup = dynamic_cast<LookupK*>(inner->left);
                    if (squad_lookup && strcmp(squad_lookup->var_name, "⌷") == 0) {
                        // Extract variable name from inner->right (the array)
                        LookupK* var_lookup = dynamic_cast<LookupK*>(inner->right);
                        if (!var_lookup) {
                            set_error("left side of indexed assignment must be a variable", token);
                            return nullptr;
                        }
                        // outer->left is the index, inner->right is the array variable
                        IndexedAssignK* indexed_assign = machine->heap->allocate<IndexedAssignK>(
                            var_lookup->var_name, outer->left, right);
                        indexed_assign->set_location(token.line, token.column);
                        return indexed_assign;
                    }
                }
            }

            // Check for operator definition: (FF OP) ← body or (FF OP GG) ← body
            // Parenthesized names produce FinalizeK wrapping JuxtaposeK of LookupKs
            // Due to right-associative parsing:
            //   (FF OP) → FinalizeK(JuxtaposeK(FF, OP))
            //   (FF OP GG) → FinalizeK(JuxtaposeK(FF, JuxtaposeK(OP, GG)))
            // Unwrap FinalizeK if present
            Continuation* inner_left = left;
            FinalizeK* finalize = dynamic_cast<FinalizeK*>(left);
            if (finalize) {
                inner_left = finalize->inner;
            }
            JuxtaposeK* jux = dynamic_cast<JuxtaposeK*>(inner_left);
            if (jux) {
                // Check for dyadic operator: (FF OP GG) = JuxtaposeK(FF, JuxtaposeK(OP, GG))
                LookupK* ff = dynamic_cast<LookupK*>(jux->left);
                JuxtaposeK* inner_jux = dynamic_cast<JuxtaposeK*>(jux->right);

                if (ff && inner_jux) {
                    LookupK* op_name = dynamic_cast<LookupK*>(inner_jux->left);
                    LookupK* gg = dynamic_cast<LookupK*>(inner_jux->right);

                    if (op_name && gg) {
                        // Dyadic operator header: (FF OP GG)
                        // Right side should be a closure body
                        ClosureLiteralK* closure = dynamic_cast<ClosureLiteralK*>(right);
                        if (!closure) {
                            set_error("operator body must be a dfn", token);
                            return nullptr;
                        }
                        // Create DefinedOperatorLiteralK for dyadic operator
                        DefinedOperatorLiteralK* def_op = machine->heap->allocate<DefinedOperatorLiteralK>(
                            closure->body, op_name->var_name, ff->var_name, gg->var_name);
                        def_op->set_location(token.line, token.column);
                        return def_op;
                    }
                }

                // Check for monadic operator: (FF OP) = JuxtaposeK(FF, OP)
                LookupK* op_name = dynamic_cast<LookupK*>(jux->right);

                if (ff && op_name) {
                    // Monadic operator header: (FF OP)
                    ClosureLiteralK* closure = dynamic_cast<ClosureLiteralK*>(right);
                    if (!closure) {
                        set_error("operator body must be a dfn", token);
                        return nullptr;
                    }
                    // Create DefinedOperatorLiteralK for monadic operator
                    DefinedOperatorLiteralK* def_op = machine->heap->allocate<DefinedOperatorLiteralK>(
                        closure->body, op_name->var_name, ff->var_name);
                    def_op->set_location(token.line, token.column);
                    return def_op;
                }
            }

            // Regular assignment: left side must be a LookupK (variable name)
            LookupK* lookup = dynamic_cast<LookupK*>(left);
            if (!lookup) {
                set_error("left side of assignment must be a variable name", token);
                return nullptr;
            }

            // Create AssignK continuation (var_name is already interned from LookupK)
            AssignK* assign = machine->heap->allocate<AssignK>(lookup->var_name, right);
            assign->set_location(token.line, token.column);
            return assign;
        }

        case TOK_LBRACE: {
            // Handle dfn application (e.g., "3 {⍺+⍵} 5")
            // NOTE: { has already been consumed by parse_expression before calling led()
            ClosureLiteralK* closure_lit = parse_dfn();
            if (!closure_lit) {
                return nullptr;
            }
            closure_lit->set_location(token.line, token.column);

            // Parse the right argument
            int bp = get_binding_power(token);  // Use operator binding power for dfns
            Continuation* right = parse_expression(bp);

            if (!right) {
                return nullptr;
            }

            // Create ApplyFunctionK for dyadic application: left {dfn} right
            ApplyFunctionK* apply = machine->heap->allocate<ApplyFunctionK>(closure_lit, left, right);
            apply->set_location(token.line, token.column);
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
            bool supports_axis = false;
            switch (token.type) {
                case TOK_EACH:         op_name = "¨";  break;
                case TOK_COMMUTE:      op_name = "⍨";  break;
                case TOK_REDUCE:       op_name = "/";  supports_axis = true; break;
                case TOK_SCAN:         op_name = "\\"; supports_axis = true; break;
                case TOK_REDUCE_FIRST: op_name = "⌿";  supports_axis = true; break;
                case TOK_SCAN_FIRST:   op_name = "⍀";  supports_axis = true; break;
                default: break;
            }

            const char* interned_name = machine->string_pool.intern(op_name);

            // Check for axis specification: f/[k] syntax
            Continuation* axis_cont = nullptr;
            if (supports_axis && current_token_.type == TOK_LBRACKET) {
                advance();  // consume [
                Continuation* inner = parse_expression(0);
                if (!inner) {
                    set_error("expected axis expression after '['");
                    return nullptr;
                }
                if (current_token_.type != TOK_RBRACKET) {
                    set_error("expected ']' after axis expression");
                    return nullptr;
                }
                advance();  // consume ]
                // Wrap in FinalizeK to ensure the axis expression is fully evaluated
                axis_cont = machine->heap->allocate<FinalizeK>(inner, true);
            }

            // Create DerivedOperatorK: evaluate operand (left), then apply operator
            // If axis_cont is present, it will be evaluated and applied via OPERATOR_CURRY
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(left, interned_name, axis_cont);
            derived->set_location(token.line, token.column);
            return derived;
        }

        case TOK_OUTER_PRODUCT: {
            // Outer product: A ∘.f B where A and B are arrays, f is a function
            // In "3 ∘.× 5", left=3 (array), we need to parse × (function)
            Continuation* fn_operand = parse_expression(BP_POSTFIX_OP);
            if (!fn_operand) {
                set_error("outer product operator requires a function operand", token);
                return nullptr;
            }

            const char* interned_name = machine->string_pool.intern("∘.");

            // Create derived operator from ∘. and the function
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(fn_operand, interned_name);
            derived->set_location(token.line, token.column);

            // Now create juxtaposition: left (∘.f)
            // This applies ∘.f monadically to left, which will curry waiting for right argument
            JuxtaposeK* jux = machine->heap->allocate<JuxtaposeK>(left, derived);
            jux->set_location(token.line, token.column);
            return jux;
        }

        case TOK_DOT: {
            // Inner product (.)
            // G2 Grammar Rule 4: fb-term dyadic-operator → derived-operator
            // The second operand will be delivered via juxtaposition (Rule 3: derived-operator fb → fb-term)
            const char* interned_name = machine->string_pool.intern(".");
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(left, interned_name);
            derived->set_location(token.line, token.column);
            return derived;
        }

        case TOK_RANK: {
            // Rank operator (⍤)
            // f⍤k applies function f to k-cells of the argument(s)
            // Dyadic operator: first operand is function, second is rank specification
            const char* interned_name = machine->string_pool.intern("⍤");
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(left, interned_name);
            derived->set_location(token.line, token.column);
            return derived;
        }

        case TOK_NAME: {
            // Defined operator: handle like primitive operators
            // At this point we already checked it's a defined operator in the parsing loop
            const char* interned_name = machine->string_pool.intern(token.name);
            DerivedOperatorK* derived = machine->heap->allocate<DerivedOperatorK>(left, interned_name);
            derived->set_location(token.line, token.column);
            return derived;
        }

        case TOK_LBRACKET: {
            // Bracket indexing: A[I] is equivalent to I⌷A
            // Use JuxtaposeK instead of DyadicK to go through proper curry handling
            Continuation* index_cont = parse_expression(0);
            if (!index_cont) {
                set_error("expected index expression after '['", token);
                return nullptr;
            }
            if (current_token_.type != TOK_RBRACKET) {
                set_error("expected ']' after index expression");
                return nullptr;
            }
            advance();  // consume ]

            // Build I⌷A as JuxtaposeK(I, JuxtaposeK(⌷, A))
            // This goes through DispatchFunctionK which handles curry finalization
            const char* squad_name = machine->string_pool.intern("⌷");
            LookupK* squad_lookup = machine->heap->allocate<LookupK>(squad_name);
            squad_lookup->set_location(token.line, token.column);
            JuxtaposeK* squad_array = machine->heap->allocate<JuxtaposeK>(squad_lookup, left);
            squad_array->set_location(token.line, token.column);
            JuxtaposeK* full_expr = machine->heap->allocate<JuxtaposeK>(index_cont, squad_array);
            full_expr->set_location(token.line, token.column);
            return full_expr;
        }

        default:
            // Unexpected token in infix position
            set_error(std::string("unexpected token in infix position: ") + token_type_name(token.type), token);
            return nullptr;
    }
}

// ============================================================================
// Dfn and Statement Parsing
// ============================================================================

ClosureLiteralK* Parser::parse_dfn() {
    // Save outer dfn's tracking state (for nested dfns)
    bool outer_uses_omega = dfn_uses_omega;
    bool outer_uses_alpha = dfn_uses_alpha;
    dfn_uses_omega = false;
    dfn_uses_alpha = false;

    skip_separators();

    // Parse statements until } (dfns use expressions, not statements)
    // Handles guards: {cond: result ⋄ cond2: result2 ⋄ default}
    // Guards are compiled to nested IfK continuations
    struct GuardedExpr {
        Continuation* condition;  // nullptr for default (unguarded) expression
        Continuation* result;
    };
    std::vector<GuardedExpr> guarded_exprs;

    while (!at_end() && current().type != TOK_RBRACE) {
        Continuation* expr = parse_expression(BP_NONE);
        if (!expr) {
            // Restore on error path
            dfn_uses_omega = outer_uses_omega;
            dfn_uses_alpha = outer_uses_alpha;
            return nullptr;
        }

        // Check for guard syntax: expr : result
        if (current().type == TOK_COLON) {
            advance();  // consume :
            Continuation* result = parse_expression(BP_NONE);
            if (!result) {
                dfn_uses_omega = outer_uses_omega;
                dfn_uses_alpha = outer_uses_alpha;
                return nullptr;
            }
            guarded_exprs.push_back({expr, result});
        } else {
            // Unguarded expression (default case or simple statement)
            guarded_exprs.push_back({nullptr, expr});
        }
        skip_separators();
    }

    // Expect closing brace
    if (current().type != TOK_RBRACE) {
        set_error("expected '}' to close dfn");
        dfn_uses_omega = outer_uses_omega;
        dfn_uses_alpha = outer_uses_alpha;
        return nullptr;
    }
    advance();  // consume }

    // Create body continuation
    Continuation* body;
    if (guarded_exprs.empty()) {
        // Empty dfn body - return 0
        body = machine->heap->allocate<LiteralK>(0.0);
    } else {
        // Build body from guarded expressions
        // Work backwards to create nested IfK for guards
        // Last expression becomes the else branch
        body = nullptr;
        for (int i = guarded_exprs.size() - 1; i >= 0; i--) {
            if (guarded_exprs[i].condition == nullptr) {
                // Unguarded expression
                if (body == nullptr) {
                    body = guarded_exprs[i].result;
                } else {
                    // Multiple unguarded - sequence them
                    std::vector<Continuation*> seq = {guarded_exprs[i].result, body};
                    body = machine->heap->allocate<SeqK>(seq);
                }
            } else {
                // Guarded expression: wrap in IfK
                body = machine->heap->allocate<IfK>(
                    guarded_exprs[i].condition,
                    guarded_exprs[i].result,
                    body  // else branch is the rest of the guards/default
                );
            }
        }
        if (body == nullptr) {
            body = machine->heap->allocate<LiteralK>(0.0);
        }
    }

    // A dfn is niladic if it uses neither ⍵ nor ⍺
    bool is_niladic = !(dfn_uses_omega || dfn_uses_alpha);

    // Restore outer dfn's state
    dfn_uses_omega = outer_uses_omega;
    dfn_uses_alpha = outer_uses_alpha;

    // Create and return the complete ClosureLiteralK
    return machine->heap->allocate<ClosureLiteralK>(body, is_niladic);
}

// Parse a block of statements until reaching a terminator token
// Returns a vector of parsed statements
// Does NOT consume the terminator token
std::vector<Continuation*> Parser::parse_block_until(TokenType terminator) {
    std::vector<Continuation*> statements;

    while (!at_end() && current().type != terminator) {
        Continuation* stmt = parse_expression(BP_NONE);
        if (!stmt) {
            // Return empty vector on parse failure
            return std::vector<Continuation*>();
        }
        statements.push_back(stmt);
        skip_separators();
    }

    return statements;
}

// ============================================================================
// Control Flow Parsing
// ============================================================================

Continuation* Parser::parse_if_statement() {
    // :If has already been consumed by nud()
    skip_separators();
    Continuation* condition = parse_expression(BP_NONE);
    if (!condition) {
        return nullptr;
    }

    skip_separators();

    // Parse then-branch (statements until :Else, :ElseIf, or :EndIf)
    std::vector<Continuation*> then_stmts;
    while (!at_end() && current().type != TOK_ELSE &&
           current().type != TOK_ELSEIF && current().type != TOK_ENDIF) {
        Continuation* stmt = parse_expression(BP_NONE);
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

    // Check for :ElseIf or :Else
    Continuation* else_branch = nullptr;
    if (!at_end() && current().type == TOK_ELSEIF) {
        // :ElseIf is syntactic sugar for :Else :If ... :EndIf
        // Recursively parse as nested if statement
        Token elseif_token = current();
        advance();  // consume :ElseIf
        Continuation* nested_if = parse_if_statement();
        if (!nested_if) {
            return nullptr;
        }
        nested_if->set_location(elseif_token.line, elseif_token.column);
        else_branch = nested_if;
        // The recursive call consumes :EndIf, so we're done
    } else if (!at_end() && current().type == TOK_ELSE) {
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

        // Expect :EndIf
        if (at_end() || current().type != TOK_ENDIF) {
            set_error("expected ':EndIf'");
            return nullptr;
        }
        advance();  // consume :EndIf
    } else {
        // Just :EndIf (no else branch)
        if (at_end() || current().type != TOK_ENDIF) {
            set_error("expected ':EndIf', ':Else', or ':ElseIf'");
            return nullptr;
        }
        advance();  // consume :EndIf
    }

    // Create IfK continuation
    IfK* if_k = machine->heap->allocate<IfK>(condition, then_branch, else_branch);
    return if_k;
}

// Parse :While statement
Continuation* Parser::parse_while_statement() {
    // :While has already been consumed by nud()
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
        set_error("expected ':EndWhile'");
        return nullptr;
    }
    advance();  // consume :EndWhile

    // Create WhileK continuation
    WhileK* while_k = machine->heap->allocate<WhileK>(condition, body);
    return while_k;
}

// Parse :For statement
Continuation* Parser::parse_for_statement() {
    // :For has already been consumed by nud()
    skip_separators();

    // Expect variable name
    if (at_end() || current().type != TOK_NAME) {
        set_error("expected variable name after ':For'");
        return nullptr;
    }
    // Intern the variable name in the string pool
    const char* var_name = machine->string_pool.intern(current().name);
    advance();  // consume variable name

    skip_separators();

    // Expect :In
    if (at_end() || current().type != TOK_IN) {
        set_error("expected ':In' after variable name");
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
        set_error("expected ':EndFor'");
        return nullptr;
    }
    advance();  // consume :EndFor

    // Create ForK continuation
    ForK* for_k = machine->heap->allocate<ForK>(var_name, array_expr, body);
    return for_k;
}

// Parse :Return statement
Continuation* Parser::parse_return_statement() {
    // :Return has already been consumed by nud()
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
    // :Leave has already been consumed by nud()
    // Create LeaveK continuation
    LeaveK* leave_k = machine->heap->allocate<LeaveK>();
    return leave_k;
}

// Parse :Continue statement
Continuation* Parser::parse_continue_statement() {
    // :Continue has already been consumed by nud()
    // Create ContinueK continuation
    ContinueK* continue_k = machine->heap->allocate<ContinueK>();
    return continue_k;
}

// Parse → (branch) statement
// →0 or →⍬ means exit function (like :Return)
// →N where N>0 is not supported (would require line numbers)
Continuation* Parser::parse_branch_statement() {
    // → has already been consumed by nud()
    // Parse the target expression
    Continuation* target_expr = parse_expression(BP_NONE);
    if (!target_expr) {
        return nullptr;
    }

    // Create BranchK continuation that will evaluate target and decide
    BranchK* branch_k = machine->heap->allocate<BranchK>(target_expr);
    return branch_k;
}

// ============================================================================
// Helper Functions
// ============================================================================

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
        case TOK_LBRACKET:  // Bracket indexing: A[I]
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
        case TOK_RBRACKET:
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

// Error formatting helpers - create consistent "SYNTAX ERROR [line:col]: message" format
void Parser::set_error(const std::string& message) {
    // Use current token's location
    set_error(message, current_token_);
}

void Parser::set_error(const std::string& message, const Token& token) {
    error_message_ = "SYNTAX ERROR [" + std::to_string(token.line) + ":" +
                     std::to_string(token.column) + "]: " + message;
}

} // namespace apl
