# APL-Eigen CEK Machine Implementation Task List

## Project Overview
Building a continuation-based APL interpreter with direct Eigen integration, focusing on minimal viable implementation first, then extending with optimizations and plugins.

## Task List Updates (Latest Review)

**Key additions based on detailed specification review:**

1. **Phase 1 Enhancements:**
   - Added Completion Records as Phase 1.4 (early integration for control flow foundation)
   - Enhanced Continuation base class with GC and control flow methods
   - Detailed APLHeap with generational GC, scalar cache, and matrix pooling
   - Added Control register class as separate component
   - Expanded Machine core with completion handling details

2. **Phase 3 Parser Additions:**
   - Added Evaluation Context System (eCgv, eCts, eCtn, eCto, eCtf) as new Phase 3.1
   - Added specific parser continuations (ParseExprK, ParseStrandK, ParseTermK)
   - Added evaluation continuations (EvalStrandK, ApplyMonadicK, ApplyDyadicK, LookupK)
   - Enhanced scope lookup with ScopeCacheK for optimization

3. **Phase 4 Control Flow:**
   - Detailed control flow continuations (IfK, WhileK, CheckWhileCondK, ForK, ReturnK, LeaveK)
   - Added FunctionCallK with function boundary marking
   - Separated continuation implementation from parser integration

4. **Phase 6 Optimization:**
   - Added detailed Optimization Guard Continuations section
   - ShapeGuardK and TypeGuardK with profiling and deoptimization
   - Guard cache integration into Machine
   - Added semantic correctness testing based on formal rules
   - Added internal expression form debugging support

## Design Decisions (Pre-Phase 1)

The following design decisions were finalized before implementation:

1. **Testing Framework**: Google Test (gtest-devel on Fedora)
2. **C++ Standard**: C++17
3. **Build System**:
   - CMake with find_package() for system dependencies
   - re2c generates lexer.cpp at build time
   - Primary target: Linux (Fedora)
   - Dependencies: gtest-devel, re2c, eigen3-devel
4. **Lexer Token Lifetime**:
   - Lexer uses arena allocation (4KB blocks) for temporary token strings
   - Continuations intern strings into permanent string pool
   - Arena reset after parsing, pool persists with continuation graphs
5. **GC Strategy**:
   - Young generation: 4096 Value slots initially
   - Old generation: 16384 Value slots initially
   - Minor GC: Triggered when young generation full
   - Major GC: Every 10 minor GCs OR when old generation >75% full
   - Growth: 1.5x current size if >50% objects still live after GC
6. **Error Handling**:
   - APL errors (VALUE ERROR, DOMAIN ERROR, etc.) → THROW completion records
   - VM errors (out of memory, assertions) → C++ exceptions
7. **Continuation Memory Management**:
   - Continuations allocated with new/delete (manual management)
   - Continuations are NOT GC-managed (only Values are)
   - Continuation::mark() tells GC which Values the continuation references
   - Function cache stores reusable continuation graphs
   - No continuation object pooling initially (removed as over-engineered)
   - Future: If profiling shows allocation overhead, can add pooling via custom operator new/delete without changing call sites
8. **Scalar Cache**: Range -128 to 127 (256 cached values)
9. **Evaluation Order**: Right-to-left (APL standard)

## Phase 1: Core Foundation (Priority: CRITICAL)
*Estimated: 30-40% of budget*

### 1.1 Project Setup
- [ ] Create CMake project structure
  - [ ] Use find_package(Eigen3 REQUIRED)
  - [ ] Use find_package(GTest REQUIRED)
  - [ ] Use find_program() for re2c
  - [ ] Add custom command to generate lexer.cpp from lexer.re
- [ ] Set up directory structure: `src/`, `include/`, `tests/`
- [ ] Configure C++17 build with optimization flags (-O2, -std=c++17)
- [ ] Add Fedora package documentation (gtest-devel, re2c, eigen3-devel)
- [ ] Create basic test harness using Google Test

### 1.2 Lexer Implementation
- [ ] Implement LexerArena class
  - [ ] 4KB block allocation
  - [ ] allocate_string() for token strings
  - [ ] reset() to clear arena after parsing
- [ ] Implement StringPool class for interning
  - [ ] intern() method returns stable const char*
  - [ ] Used by continuations for long-term name storage
- [ ] Write re2c lexer specification (`lexer.re` ~100 LOC)
  - [ ] Define TokenType enum with all APL tokens
  - [ ] Implement number recognition (integers and floats)
  - [ ] Implement name/identifier recognition (allocate in arena)
  - [ ] Add all operator tokens (arithmetic, reduction, scan, each, etc.)
  - [ ] Add control flow tokens (if, while, for, leave, return)
- [ ] Create Token struct with type and value union
- [ ] Integrate lexer with arena (tokens point into arena)
- [ ] Write lexer test suite

### 1.3 Value System
- [ ] Implement base Value class with tagged union
  - [ ] SCALAR type with double storage
  - [ ] VECTOR type stored as Eigen column matrix (n×1)
  - [ ] MATRIX type with Eigen::MatrixXd storage
  - [ ] FUNCTION type with function pointer
  - [ ] OPERATOR type with operator pointer
- [ ] Implement zero-copy vector-as-matrix storage
- [ ] Add type checking methods (is_scalar, is_array, etc.)
- [ ] Implement Value factory methods
  - [ ] from_scalar(double)
  - [ ] from_vector(Eigen::VectorXd*) with zero-copy wrapping
  - [ ] from_matrix(Eigen::MatrixXd*)
- [ ] Add as_matrix() accessor with lazy scalar promotion
- [ ] Write comprehensive value system tests

### 1.4 Completion Records (Early Integration)
- [ ] Implement CompletionType enum (NORMAL, BREAK, CONTINUE, RETURN, THROW)
- [ ] Create APLCompletion struct with type, value, and target fields
- [ ] Add completion field to Control class
- [ ] Implement basic completion handling framework (will be expanded in Phase 4)

### 1.5 Continuation Base Classes
- [ ] Define abstract Continuation base class
- [ ] Implement Value* invoke(Machine*) interface
- [ ] Add virtual mark(APLHeap*) method for GC
- [ ] Add virtual is_function_boundary() method
- [ ] Add virtual is_loop_boundary() method
- [ ] Add virtual matches_label(const char*) method
- [ ] Create HaltK terminal continuation
- [ ] Implement ArgK for function arguments
- [ ] Implement FrameK for stack frames
- [ ] Add basic continuation chaining

### 1.6 Memory Management and GC
- [ ] Implement enhanced APLHeap with generational zones
  - [ ] Create young_objects array for short-lived allocations
  - [ ] Create old_objects array for long-lived objects
  - [ ] Add temp_zone for expression evaluation temporaries
  - [ ] Implement scalar_cache for common values (-128 to 127)
  - [ ] Add array_pool for reusable matrix buffers
  - [ ] Implement allocation strategy with cache checking
- [ ] Implement mark-and-sweep garbage collector
  - [ ] Implement mark phase with root set traversal
  - [ ] Add sweep phase for unreachable objects
  - [ ] Integrate GC trigger points (allocation threshold)
  - [ ] Implement minor_gc for young generation
  - [ ] Add promotion logic from young to old generation
- [ ] Create Eigen matrix pooling
  - [ ] Pool common matrix sizes
  - [ ] Add matrix recycling for temporaries
  - [ ] Implement matrix reuse from pool
- [ ] Implement Value reference tracking
  - [ ] Add GC metadata to Value class
  - [ ] Track inter-value references
- [ ] Add RAII wrappers for automatic cleanup
- [ ] Write memory management tests
  - [ ] Verify no leaks in basic operations
  - [ ] Test GC under memory pressure
  - [ ] Test generational promotion
  - [ ] Validate scalar cache hits
  - [ ] Test matrix pool reuse

### 1.7 Control Register
- [ ] Implement ExecMode enum (PARSE, EVAL)
- [ ] Create Control class with mode field
- [ ] Add input_ptr for parsing position
- [ ] Add current_token field
- [ ] Add value field for current evaluation result
- [ ] Add completion field for control flow
- [ ] Implement advance_token() method

### 1.8 CEK Machine Core
- [ ] Implement Machine class with C/E/K registers
  - [ ] Control register (Control class instance)
  - [ ] Environment register (Environment* env)
  - [ ] Kontinuation stack (std::vector<Continuation*>)
  - [ ] APLHeap heap instance
  - [ ] StringPool for interned names
- [ ] Add function_cache for continuation graphs (reusable parsed functions)
- [ ] Add guard_cache for optimization guards
- [ ] Integrate Machine with GC
  - [ ] Register machine roots with GC
  - [ ] Add GC safe points in trampoline
- [ ] Implement execute() trampoline loop
  - [ ] Pop and invoke continuations until stack empty
  - [ ] Check for completion records after each invoke
  - [ ] Call handle_completion() when needed
- [ ] Implement handle_completion() method
  - [ ] Handle NORMAL completions
  - [ ] Handle RETURN with stack unwinding to function boundary
  - [ ] Handle BREAK with stack unwinding to loop boundary
  - [ ] Handle CONTINUE with loop resumption
  - [ ] Handle THROW for error propagation
- [ ] Add push_kont() and pop_kont_and_invoke() helpers
- [ ] Add step() method for single-step execution
- [ ] Add basic error handling

## Phase 2: Essential Operations (Priority: HIGH)
*Estimated: 20-25% of budget*

### 2.1 Arithmetic Primitives
- [ ] Implement scalar-scalar fast path (no Eigen)
- [ ] Add scalar extension (broadcasting) using Eigen
- [ ] Implement core arithmetic operations:
  - [ ] prim_add (+)
  - [ ] prim_subtract (-)
  - [ ] prim_multiply (×)
  - [ ] prim_divide (÷)
  - [ ] prim_power (*)
- [ ] Add proper shape checking and error handling
- [ ] Write arithmetic operation tests

### 2.2 Array Operations
- [ ] Implement reshape (⍴)
- [ ] Implement ravel (,) - flatten to vector
- [ ] Implement transpose (⍉)
- [ ] Implement indexing operations
- [ ] Add iota (⍳) for index generation
- [ ] Implement take (↑) and drop (↓)

### 2.3 Reduction Operations
- [ ] Implement reduce (/) for vectors
- [ ] Extend reduce for matrices (row reduction)
- [ ] Add reduce-first (⌿) for column reduction
- [ ] Implement scan (\) for cumulative operations
- [ ] Add scan-first (⍀) for column scan

### 2.4 Environment and Binding
- [ ] Create Environment class with parent chain
- [ ] Implement variable lookup with scope chain traversal
- [ ] Add system namespace lookup as fallback
- [ ] Add bind() method for variable assignment
- [ ] Add assignment operation (←)
- [ ] Create lexical scoping
- [ ] Implement ScopeCacheK continuation for lookup optimization
  - [ ] Add cached_addr and cached_env fields
  - [ ] Implement cache validation logic
  - [ ] Handle cache miss with full lookup
- [ ] Add environment tests

## Phase 3: Parser Integration (Priority: HIGH)
*Estimated: 15-20% of budget*

### 3.1 Evaluation Context System
- [ ] Define EvalContext enum (eCgv, eCts, eCtn, eCto, eCtf)
  - [ ] eCgv: GetValue contexts (name resolution)
  - [ ] eCts: ToString contexts (string conversion)
  - [ ] eCtn: ToNumber contexts (numeric conversion)
  - [ ] eCto: ToArray contexts (array promotion)
  - [ ] eCtf: ToFunction contexts (operator/function resolution)
- [ ] Implement ContextK continuation
  - [ ] Add context field and next continuation
  - [ ] Implement context-aware value transformations
- [ ] Add determine_context() helper to Machine

### 3.2 Expression Parser
- [ ] Implement ParseExprK continuation
  - [ ] Add context-aware token dispatch
  - [ ] Integrate with ContextK for type-directed parsing
- [ ] Implement ParseStrandK for space-separated arrays (1 2 3)
- [ ] Implement ParseTermK for individual terms
- [ ] Add operator precedence parsing (APL: right-to-left evaluation)
- [ ] Handle parenthesized expressions
- [ ] Parse function trains
- [ ] Parse array literals

### 3.3 Statement Parser
- [ ] Parse assignment statements
- [ ] Parse control flow (if/while/for)
- [ ] Handle multi-line programs
- [ ] Add comment support

### 3.4 Evaluation Continuations
- [ ] Implement EvalStrandK for evaluating array strands
  - [ ] Add right-to-left evaluation of elements
  - [ ] Build result array from evaluated elements
- [ ] Implement ApplyMonadicK for unary function application
- [ ] Implement ApplyDyadicK for binary function application
  - [ ] Store left operand
  - [ ] Apply function when right operand available
- [ ] Implement LookupK for variable resolution
  - [ ] Use Environment lookup
  - [ ] Handle undefined names

### 3.5 Parse-Eval Integration
- [ ] Connect parser to continuation creation
- [ ] Implement ParseK continuation for deferred parsing
- [ ] Add runtime type-directed parsing
- [ ] Create continuation graph builder
- [ ] Add function_cache integration for caching parsed functions

## Phase 4: Control Flow (Priority: MEDIUM)
*Estimated: 10-15% of budget*

### 4.1 Control Flow Continuations
- [ ] Implement IfK continuation
  - [ ] Add then_branch and else_branch fields
  - [ ] Evaluate condition and select branch
- [ ] Implement WhileK continuation
  - [ ] Mark as loop boundary (is_loop_boundary() returns true)
  - [ ] Re-push self for next iteration
  - [ ] Integrate with CheckWhileCondK
- [ ] Implement CheckWhileCondK continuation
  - [ ] Test loop condition
  - [ ] Push body if condition true
  - [ ] Pop WhileK if condition false
- [ ] Implement ForK continuation for iteration
  - [ ] Mark as loop boundary
  - [ ] Handle iteration variable binding
- [ ] Implement ReturnK continuation
  - [ ] Create RETURN completion record
  - [ ] Set completion value and type
- [ ] Implement LeaveK continuation
  - [ ] Create BREAK completion record
  - [ ] Support optional label targeting

### 4.2 Control Structure Integration
- [ ] Connect parser to control flow continuations
- [ ] Test if-then-else with completion records
- [ ] Test while loops with BREAK completion
- [ ] Test for loops with iteration
- [ ] Test :Leave (break) with label targeting
- [ ] Test :Return with stack unwinding

### 4.3 Function Application and Caching
- [ ] Implement FunctionCallK continuation
  - [ ] Mark as function boundary (is_function_boundary() returns true)
  - [ ] Lookup cached continuation graph from function_cache
  - [ ] Create new environment for function scope
  - [ ] Bind ⍵ (omega) for right argument
  - [ ] Bind ⍺ (alpha) for left argument if present
  - [ ] Push function body continuation

### 4.4 Function Definition
- [ ] Parse function definitions (dfns)
- [ ] Implement function closures
- [ ] Add recursive function support
- [ ] Create function composition

## Phase 5: Advanced Features (Priority: MEDIUM)
*Estimated: 10-15% of budget*

### 5.1 Higher-Order Operations
- [ ] Implement each (¨) operator
- [ ] Add outer product (∘.)
- [ ] Implement inner product
- [ ] Add function composition (∘)
- [ ] Implement commute (⍨)

### 5.2 Eigen-Specific Functions
- [ ] Add norm functions (norm, norm1, norminf)
- [ ] Implement normalize function
- [ ] Add distance calculation
- [ ] Implement cross product
- [ ] Add dot product optimization
- [ ] Implement matrix decompositions (svd, qr, lu)

## Phase 6: Testing and Optimization (Priority: MEDIUM)
*Estimated: 5-10% of budget*

### 6.1 Test Suite
- [ ] Create comprehensive unit tests
- [ ] Add integration tests for complex expressions
- [ ] Implement performance benchmarks
- [ ] Add memory leak detection
- [ ] Create test coverage reports
- [ ] Add semantic correctness tests based on formal rules
  - [ ] Test scalar operations follow H,E,a+b semantics
  - [ ] Test array promotion rules
  - [ ] Test right-to-left evaluation order
  - [ ] Test operator application semantics
  - [ ] Verify function definition and caching

### 6.2 Optimization Guard Continuations
- [ ] Implement ShapeGuardK continuation
  - [ ] Add expected_rows and expected_cols fields
  - [ ] Add fast_path and slow_path continuations
  - [ ] Add hit_count and miss_count for profiling
  - [ ] Implement guard checking logic
  - [ ] Add deoptimization when miss_count too high
- [ ] Implement TypeGuardK continuation
  - [ ] Add expected_type field
  - [ ] Add fast_path and slow_path continuations
  - [ ] Add hit_count and miss_count tracking
  - [ ] Implement type checking and path selection
- [ ] Integrate guards into function_cache
- [ ] Add guard_cache to Machine for guard reuse
- [ ] Implement guard creation heuristics

### 6.3 General Optimizations
- [ ] Profile hot paths
- [ ] Optimize scalar operations
- [ ] Implement constant folding
- [ ] Add dead code elimination
- [ ] Optimize continuation allocation patterns

### 6.4 REPL and Tools
- [ ] Implement interactive REPL
- [ ] Add workspace save/load
- [ ] Create pretty-printing for arrays
- [ ] Add debugging commands
- [ ] Implement timing/profiling commands
- [ ] Add internal expression form display (@GetValue, @PromoteToArray, etc.)
  - [ ] Support debug traces showing semantic forms
  - [ ] Add optional IR dump for continuation graphs

## Phase 7: Plugin System (Priority: LOW)
*Estimated: 5-10% of budget - Only if budget allows*

### 7.1 Plugin Infrastructure
- [ ] Design plugin API header
- [ ] Implement ExternalType base class
- [ ] Create TypeRegistry
- [ ] Add PluginManager
- [ ] Implement dynamic library loading

### 7.2 Plugin Integration
- [ ] Extend Value system for external types
- [ ] Add method invocation support
- [ ] Extend existing GC for external types
  - [ ] Add external type marking to existing mark phase
  - [ ] Integrate external type cleanup in sweep phase
  - [ ] Hook external types into existing reference tracking
- [ ] Create example plugin
- [ ] Add plugin documentation

## Implementation Strategy for Budget Optimization

### Priority Guidelines
1. **CRITICAL**: Must have for basic functionality
2. **HIGH**: Core APL features needed for useful programs
3. **MEDIUM**: Important for production use
4. **LOW**: Nice-to-have extensions

### Claude Code Optimization Tips
1. **Batch Related Tasks**: Group similar files together to minimize context switches
2. **Use Templates**: Create file templates early to reduce repetitive code generation
3. **Incremental Testing**: Test as you go to avoid large debugging sessions
4. **Focus on MVP**: Get Phase 1-3 working before optimizations
5. **Defer Complex Features**: Plugin system can be added post-budget if needed

### Development Order
1. Start with lexer and value system (foundation)
2. **Implement GC immediately** - VM cannot function without memory management
3. Implement basic arithmetic to verify value system
4. Add minimal parser for testing
5. Build out array operations
6. Add control flow only when needed
7. Optimize only after profiling

### Critical Infrastructure (Non-negotiable)
- **Memory Management**: Must be implemented in Phase 1
  - Mark-and-sweep GC is not an "optimization" - it's required for basic operation
  - Even simple loops will crash without proper memory management
  - Build the actual GC from the start, not a temporary solution
- **Value Lifecycle**: Proper construction/destruction from day one
- **Continuation Pooling**: Reduces allocation pressure from the start

### File Structure Recommendation
```
apl-eigen-cek/
├── CMakeLists.txt
├── src/
│   ├── lexer.re          # re2c specification
│   ├── lexer.cpp         # generated
│   ├── value.cpp
│   ├── machine.cpp
│   ├── continuation.cpp
│   ├── primitives.cpp
│   ├── parser.cpp
│   └── main.cpp
├── include/
│   ├── value.h
│   ├── machine.h
│   ├── continuation.h
│   ├── primitives.h
│   └── parser.h
├── tests/
│   ├── test_lexer.cpp
│   ├── test_value.cpp
│   ├── test_primitives.cpp
│   └── test_integration.cpp
└── plugins/             # Only if budget allows
    └── example/
```

### Minimal Viable Product (MVP) Checklist
- [ ] Can tokenize APL expressions
- [ ] Can perform scalar and array arithmetic
- [ ] Can execute simple APL expressions
- [ ] Can define and call functions
- [ ] Has basic REPL for testing
- [ ] **No memory leaks** - GC properly manages all allocations
- [ ] Can run loops without memory exhaustion

### Post-MVP Extensions (After budget or with Pro plan)
- [ ] Full control flow implementation
- [ ] Advanced array operations
- [ ] Plugin system for external types
- [ ] Performance optimizations
- [ ] Comprehensive documentation
- [ ] Geometry nodes integration

## Success Metrics
- [ ] Lexer processes 1000+ tokens/sec
- [ ] Basic operations match APL semantics
- [ ] No memory leaks in core operations
- [ ] Can run factorial and fibonacci examples
- [ ] Can perform matrix operations efficiently

## Risk Mitigation
1. **Complexity**: Start simple, test often
2. **Budget**: Focus on MVP, defer nice-to-haves
3. **Debugging**: Add logging early
4. **Performance**: Profile before optimizing
5. **Integration**: Test Eigen operations standalone first

## Notes for Claude Code
- Keep context focused on one module at a time
- Provide clear function signatures and invariants
- Test each component before integration
- Use consistent naming conventions throughout
- Document tricky continuation flow clearly
