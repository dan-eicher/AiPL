# AiPL - An APL Interpreter

A continuation-based APL interpreter built on a CEK machine architecture with direct Eigen integration.

## Features

**Primitives**
- Arithmetic: `+` `-` `×` `÷` `*` `⌈` `⌊` `|` `⍟` `!` `○`
- Comparison: `=` `≠` `<` `>` `≤` `≥`
- Logical: `∧` `∨` `~` `⍲` `⍱`
- Structural: `⍴` `,` `⍉` `⌽` `⊖` `↑` `↓` `⊂` `⊃` `∊` `⍷` `≡` `≢`
- Selection: `⍳` `⍸` `⌷` `/` `⍋` `⍒` `∪` `∩` `~`
- Mathematical: `⌹` `⊥` `⊤` `?`
- Format/Execute: `⍕` `⍎`

**Operators**
- Reduce: `/` `⌿`
- Scan: `\` `⍀`
- Each: `¨`
- Commute: `⍨`
- Outer product: `∘.`
- Inner product: `.`
- Rank: `⍤`
- Axis specification: `f[k]`

**Control Flow**
- Conditionals: `:If` `:ElseIf` `:Else` `:EndIf`
- Loops: `:While` `:Until` `:For` `:EndWhile` `:EndFor`
- Control: `:Leave` `:Continue` `:Return` `→`
- Guards in dfns

**Other**
- Direct functions (dfns): `{⍵+1}` with `⍵` `⍺` `∇`
- Defined operators: `{⍺⍺/⍵}` with `⍺⍺` `⍵⍵`
- Nested arrays and strands
- N-dimensional arrays (rank 3+)
- System functions: `⎕IO` `⎕CT` `⎕PP` `⎕RL` `⎕NC` `⎕NL` `⎕EX` `⎕EA` `⎕ES` `⎕ET` `⎕EM` `⎕LX`
- ISO 13751 compliant error handling

## REPL

A minimal REPL is included for testing and demonstration. It only supports single-line input, so multi-line constructs like `:If`/`:While`/`:For` blocks are not usable interactively (though they are fully implemented and tested).

```
$ ./apl
APL Interpreter (ISO 13751)
Type )help for help, )quit to exit

      2 + 3 × 4
14
      ⍳5
1 2 3 4 5
      +/⍳100
5050
      fib ← {⍵≤1:⍵ ⋄ (∇⍵-1)+∇⍵-2}
      fib¨⍳10
1 1 2 3 5 8 13 21 34 55
      mat ← 2 2⍴1 2 3 4
1 2
3 4
      ⌹mat
¯2 1
1.5 ¯0.5
```

## Architecture

The interpreter uses a CEK (Control-Environment-Kontinuation) machine:

- **Control**: Current continuation being executed
- **Environment**: Variable bindings with lexical scoping
- **Kontinuation**: Stack of pending continuations

Parsing and evaluation are unified - the parser builds continuation graphs that are cached and reused. Array operations use Eigen for efficient linear algebra.

## Building

### Dependencies (Fedora)

```bash
sudo dnf install eigen3-devel re2c gtest-devel cmake gcc-c++
```

### Compile

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run Tests

```bash
cd build
ctest
```

## Requirements

- C++17 compiler (GCC 7+, Clang 5+)
- CMake 3.14+
- Eigen 3.4+
- re2c 3.0+
- Google Test 1.10+ (for tests)
