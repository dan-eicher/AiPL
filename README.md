# APL-Eigen CEK Machine

A continuation-based APL interpreter with direct Eigen integration.

## Overview

This project implements a CEK (Control-Environment-Kontinuation) machine for APL that operates directly on Eigen matrix types. The design integrates parsing and evaluation in a single runtime, building and caching continuation graphs for efficient execution.

See [APL-Eigen CEK Machine.md](APL-Eigen CEK Machine.md) for complete architecture documentation.

## Requirements

### Fedora Dependencies

Install required system packages:

```bash
sudo dnf install eigen3-devel re2c gtest-devel cmake gcc-c++
```

- **eigen3-devel**: Matrix library for array operations
- **re2c**: Lexer generator
- **gtest-devel**: Google Test framework
- **cmake**: Build system (≥3.14)
- **gcc-c++**: C++17 compiler

### Build Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+)
- CMake 3.14 or higher
- Eigen 3.4+
- Google Test 1.10+
- re2c 3.0+

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running Tests

```bash
cd build
ctest
# or
./tests/run_tests
```

