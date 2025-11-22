# The APL CEK Machine: A Unified Model of Parsing and Evaluation

## Abstract

We present the APL CEK Machine, a computational model that unifies syntactic parsing and semantic evaluation for the APL programming language. By integrating Pratt's top-down operator precedence parsing with the CEK machine's evaluation framework and adapting them to APL's unique characteristics, we create a single formal model that handles the inherent context-dependence of APL parsing. Building on insights from Georgeff et al.'s work on operator parsing, we incorporate explicit valency-polymorphic function representations, type-dependent precedence checking, and environment-preserving closures. This unified approach provides continuous transition from token stream to final result while correctly suspending evaluation for function definitions, offering benefits in incremental processing, error recovery, and semantic clarity for APL implementations.

## 1 Introduction

Traditional APL implementations face a fundamental challenge: the language's syntax is semantically ambiguous. Expressions like `x O1 H 3` cannot be parsed without knowing whether `O1` returns a basic value or function, as this determines whether `H` should be interpreted as monadic or dyadic. This forces either unbounded lookahead or evaluation during parsing, breaking the conventional compiler pipeline.

The APL CEK Machine resolves this tension by modeling both parsing and evaluation within a single continuation-passing framework. Unlike traditional implementations that treat parsing and evaluation as separate phases, our approach demonstrates they are fundamentally complementary computational processes that can be unified through careful design of grammar, function representation, and state transitions. By synthesizing the CEK framework with Georgeff et al.'s insights on operator parsing, we achieve a more robust and formally grounded model.

## 2 Background

### 2.1 The APL Parsing Problem

APL's combination of features creates unique parsing challenges:
- **Right-to-left evaluation** dictates unusual reduction order
- **Operator overloading** means identifiers lack fixed syntactic categories
- **Left-associating operators** with **right-associating functions** create complex precedence relationships
- **Function-valued operators** can return either values or functions

These characteristics mean that syntactic structure depends on semantic types, making purely syntactic parsing impossible.

### 2.2 The CEK Machine Framework

The CEK (Control-Environment-Continuation) machine models computation as state transitions between triples:
- **Control (C):** The current term or action to process
- **Environment (E):** The current variable bindings and grammar specifications
- **Continuation (K):** The "rest of the computation" to perform after the current task

This formulation makes evaluation context explicit and provides a formal foundation for control flow.

### 2.3 Prior Work on APL Operator Parsing

Georgeff, Fris, and Kautsky demonstrated that APL's parsing challenges can be addressed through:
- **Valency-polymorphic function representations** that dynamically dispatch based on argument presence
- **Curried grammars** that transform dyadic constructs into monadic forms
- **Type-dependent parsing decisions** that consult runtime type information
- **Environment-preserving closures** for proper handling of function definitions

Our work synthesizes these insights with the CEK machine framework to create a more unified and formally grounded model.

## 3 The APL CEK Machine

### 3.1 Unified State Representation

The machine state is a quadruple that encompasses both parsing and evaluation:

```
State = (Tokens, Control, Environment, Continuation)
```

Where:
- **Tokens:** The remaining token stream (identifiers, literals, operators, abstraction symbols)
- **Control:** During parsing: current parsing continuation; during evaluation: current expression or value
- **Environment:** Unified environment containing both grammar specifications and variable bindings
- **Continuation:** Stack of pending computations, mixing parsing and evaluation tasks

### 3.2 Curried Grammar for Syntactic Clarity

We adopt a curried grammar that eliminates syntactic ambiguity by transforming all dyadic constructs into monadic forms, following Georgeff et al.'s approach:

```
expression ::= fbn-term
fbn-term   ::= fb-term | fb-term fbn-term  
fb-term    ::= fb | fb-term monadic-operator | derived-operator fb
derived-operator ::= fb-term dyadic-operator
fb         ::= identifier | ( expression ) | abstraction
```

This grammar ensures that every expression has a unique parse tree regardless of semantic types, while maintaining APL's conventional notation.

### 3.3 Explicit Valency-Polymorphic Function Representation

To handle overloading without runtime type checking, we represent overloaded functions as unified operators that dynamically dispatch based on argument presence, extending Georgeff et al.'s representation:

```
g' = λx. λy.
    if null(y) then g₁(x)                 -- Monadic application
    else if isFunction(y) then y(g₁(x))   -- Monadic in left context  
    else g₂(x, y)                         -- Dyadic application
```

This representation defers valency decisions to application time while preserving APL's right-to-left evaluation order. The representation captures both monadic and dyadic interpretations within a single polymorphic closure.

### 3.4 Type-Dependent Precedence Parsing

Our parsing continuations incorporate type information from the environment to resolve precedence conflicts:

```
ParseExpr(bp, type_context)  -- Parse with binding power and type context
ParseFB(bp, expected_type)   -- Parse fb-term with type expectations
```

The precedence checking function consults both syntactic precedence and semantic types:

```
precedence_ok = syntactic_precedence(P, token) && 
                type_compatible(expected_type, infer_type(token, E))
```

This hybrid approach eliminates unbounded lookahead by using available type information to guide parsing decisions.

### 3.5 The Permute Operation for Argument Reordering

We incorporate Georgeff et al.'s `permute` operation to handle the argument reordering required by curried dyadic functions and operators:

```
permute(x₁, x₂, type_x₁, type_x₂) =
    if is_curried_dyadic(type_x₁) then (x₂, x₁)  -- Reorder for curried application
    else (x₁, x₂)                                -- Standard application order
```

This operation ensures that curried dyadic functions and operators receive their arguments in the correct order despite APL's right-to-left evaluation.

### 3.6 Environment-Preserving Closures

Function abstractions create closures that explicitly capture their definition environment:

```
closure = [body_ast, bound_variables, captured_environment]
```

When applied, these closures evaluate their body in the captured environment extended with the bound variables:

```
apply_closure(closure, arguments) =
    let extended_env = extend(captured_environment, bound_variables, arguments)
    in evaluate(body_ast, extended_env)
```

This ensures proper lexical scoping and handling of free variables in function definitions.

### 3.7 Grammar as Continuations

Parsing is modeled using Pratt-style parsing continuations adapted for APL's grammar:

- **ParseExpr(bp, type_ctx):** Parse an expression with minimum binding power and type context
- **ParseFBN(bp, expected_type):** Parse an fbn-term with binding power and type expectation
- **ParseFB(bp, type_info):** Parse an fb-term with binding power and type information
- **Nud(token, type_ctx):** Handle token in prefix position with type context
- **Led(token, left, left_type):** Handle token in infix position with left expression and type
- **CombineInfix(left, op, types):** Build AST node using permute for argument ordering
- **CompleteParse(result, final_type):** Finalize parsing with inferred type

## 4 State Transitions

The machine operates through continuous state transitions that interleave parsing and evaluation:

### 4.1 Core Parsing Transitions with Type Information

**Token Consumption with Type Context:**
```
([token | RestTokens], ParseFB(bp, expected_type), Env, K)
→ (RestTokens, Nud(token, expected_type), Env, K)
```

**Operator Application with Type-Dependent Precedence:**
```
(Tokens, Led(op, left, left_type), Env, K)  
→ (Tokens, ParseFB(bp(op), infer_type(op, Env)), Env, CombineInfix(left, op, left_type) · K)
```

**AST Construction with Argument Permutation:**
```
(Tokens, right, right_type, Env, CombineInfix(left, op, left_type) · K)
→ let (func, arg) = permute(left, right, left_type, right_type, op_type)
  in (Tokens, Apply(func, arg), Env, K)
```

### 4.2 Abstraction Handling with Environment Capture

**Abstraction Recognition:**
```
(['@f' | RestTokens], ParseFB(bp, type_ctx), Env, K)
→ (RestTokens, ParseExpr(0, function_type), Env, BuildAbstraction(Env) · K)
```

**Closure Creation with Environment Capture:**
```
(Tokens, body_ast, body_type, Env, BuildAbstraction(captured_env) · K)
→ (Tokens, [body_ast, bv, captured_env], Env, K)  // Environment-preserving closure
```

### 4.3 Evaluation Transitions with Valency Polymorphism

**Phase Transition:**
```
([], AST_Node, node_type, Env, K)
→ ([], AST_Node, Env, Eval(node_type) · K)
```

**Valency-Polymorphic Application:**
```
(Tokens, Apply(f, arg), Env, K)
→ (Tokens, f, Env, EvalArg(arg, infer_type(f, Env)) · K)
→ (Tokens, arg_val, arg_type, Env, ApplyValencyPoly(f, arg_type) · K)
```

The `ApplyValencyPoly` continuation implements the dynamic dispatch logic using the explicit valency-polymorphic representation.

## 5 Benefits of the Unified Model

### 5.1 Resolution of APL's Semantic Ambiguity

The machine elegantly resolves APL's fundamental ambiguity problem:
- The curried grammar ensures unambiguous syntactic structure
- Explicit valency-polymorphic functions handle overloading at application time
- Type-dependent precedence checking eliminates unbounded lookahead
- Continuation chains naturally capture the interdependent nature of parsing and evaluation

### 5.2 Incremental Processing

Traditional APL interpreters must parse entire expressions before evaluation can begin. The APL CEK Machine enables fine-grained interleaving:

```
(Tokens, ParseExpr(bp, type_ctx), Env, EvalContinuation)
→ (RestTokens, PartialAST, partial_type, Env, ParseContinuation)  
→ (RestTokens, IntermediateValue, value_type, Env, EvalContinuation)
```

This allows evaluation to begin before parsing completes, enabling streaming interpretation and responsive systems.

### 5.3 Unified Error Handling

Both parsing errors and evaluation exceptions flow through the same continuation mechanism:
- **Syntax errors** become exceptional values in the parsing phase
- **Type errors** and **domain errors** propagate through evaluation continuations
- **Error recovery** strategies can be uniformly applied across phases

### 5.4 Formal Semantics for APL

By modeling the entire APL computation process within the CEK framework, we gain:
- A formal operational semantics covering both parsing and evaluation
- Clear specification of APL's unique evaluation order
- Mathematical reasoning about parser and evaluator correctness
- Explicit treatment of valency polymorphism and environment capture

### 5.5 Suspendable Computation

The continuation-based approach naturally supports pausing and resuming:

```
Suspend: (Tokens, Control, Env, K) → SerializedState
Resume: SerializedState → (Tokens, Control, Env, K)
```

This enables:
- **Interactive development** with continuous parsing and evaluation
- **Debugging** that can step through the unified parse/eval process
- **Partial evaluation** of incomplete APL expressions

## 6 Example: Parsing and Evaluating `5 O2 6 O1 H 7`

Consider the expression where `H` is overloaded with both monadic and dyadic interpretations.

### Initial State:
```
Tokens = [5, O2, 6, O1, H, 7]
Control = ParseExpr(0, any_type)  
Env = {O2: (nud₂, led₂, 20, dyadic-op, o2_type), 
       O1: (nud₁, led₁, 10, monadic-op, o1_type),
       H: (nud_H, led_H, 30, overloaded-fn, h_poly_type)}
K = EvalAndPrint
```

### Unified Parse/Evaluate Steps:

1. **Parse `5 O2` with type checking:** Machine builds `Apply(O2, 5, 6)` through type-aware Pratt parsing
2. **Apply permute for curried operator:** Arguments are reordered according to operator type
3. **Parse `O1` application:** `Led(O1, Apply(O2, 5, 6), o2_result_type)` consults type information
4. **Parse `H 7` with valency polymorphism:** The overloaded `H` is parsed using its polymorphic type
5. **Build complete AST:** `Apply(O1, Apply(O2, 5, 6), Apply(H, 7))` with proper argument ordering
6. **Transition to evaluation:** Input exhausted, evaluation phase begins
7. **Evaluate using valency dispatch:** `ApplyValencyPoly` invokes the appropriate function branch based on runtime types
8. **Final application:** Results flow through the continuation chain to produce final value

Throughout this process, the machine maintains a single coherent state while leveraging type information and polymorphic dispatch to handle APL's complexities.

## 7 Implementation Considerations

### 7.1 Performance Optimizations

The continuation-passing style incurs overhead, but several optimizations are possible:
- **Selective eager evaluation** of primitive functions with known types
- **Type inference** to reduce runtime type checks
- **Continuation compression** for common parsing/evaluation patterns
- **Memoization** of grammar productions and function applications

### 7.2 Memory Management

The unified state requires careful resource management:
- **Grammar environment** can be shared and immutable across executions
- **Token streaming** avoids loading entire programs into memory
- **Continuation reuse** reduces allocation overhead for common patterns
- **Environment sharing** in closures minimizes duplication

## 8 Related Work

Our work synthesizes and extends several traditions:
- **Pratt parsing** provides the operator precedence foundation but lacked formal semantics and type integration
- **CEK machines** offer formal evaluation models but typically assume pre-parsed terms
- **Georgeff et al.'s operator parsing** provides practical solutions for APL's challenges but lacks unified formal foundations
- **APL implementation techniques** address practical concerns but often lack formal rigor

The APL CEK Machine is the first model to fully unify these approaches for APL's unique requirements, incorporating explicit valency polymorphism, type-dependent parsing, and environment preservation within a single continuation-passing framework.

## 9 Conclusion

The APL CEK Machine demonstrates that parsing and evaluation need not be separate computational phases, even for languages with complex, context-dependent syntax like APL. By unifying them within a single continuation-passing framework and incorporating insights from Georgeff et al.'s work on operator parsing, we achieve:

1. **Conceptual clarity** through a coherent computational model for APL
2. **Practical resolution** of APL's semantic ambiguity through valency polymorphism and type-dependent parsing
3. **Formal foundations** for both syntactic and semantic analysis
4. **Robust handling** of function definitions through environment-preserving closures
5. **Implementation flexibility** supporting various execution strategies

This integration suggests that the traditional compiler pipeline, while efficient for many languages, is fundamentally misaligned with APL's nature. The APL CEK Machine offers an alternative perspective where syntax and semantics coexist in a continuous computational process that respects APL's unique evaluation model while providing formal rigor and practical implementability.

## References

[1] Pratt, V. R. "Top Down Operator Precendence." POPL 1973.

[2] Felleisen, M. and Friedman, D. P. "Control Operators, the SECD Machine, and the λ-Calculus." 1986.

[3] Georgeff, M. P., Fris, I., & Kautsky, J. "Parsing and Evaluation of APL with Operators." 1981.

[4] Iverson, K. E. "Operators." ACM Transactions on Programming Languages and Systems, 1979.

[5] Strawn, G. O. "Does APL Really Need Run-time Parsing?" Software—Practice and Experience, 1977.
