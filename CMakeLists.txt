cmake_minimum_required(VERSION 3.14)
project(apl-eigen-cek VERSION 0.1.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find required dependencies (Fedora system packages)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# replxx for REPL (git submodule in extern/replxx)
add_subdirectory(extern/replxx)

# Find re2c for lexer generation
find_program(RE2C_EXECUTABLE re2c)
if(NOT RE2C_EXECUTABLE)
    message(FATAL_ERROR "re2c not found. Install with: sudo dnf install re2c")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Generate lexer from re2c specification
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/src/lexer.cpp
    COMMAND ${RE2C_EXECUTABLE} --utf8 --input-encoding utf8 -o ${CMAKE_SOURCE_DIR}/src/lexer.cpp ${CMAKE_SOURCE_DIR}/src/lexer.re
    DEPENDS ${CMAKE_SOURCE_DIR}/src/lexer.re
    COMMENT "Generating lexer with re2c (UTF-8)"
)

# Source files
set(APL_SOURCES
    src/token.cpp
    src/lexer.cpp
    src/value.cpp
    src/continuation.cpp
    src/completion.cpp
    src/heap.cpp
    src/environment.cpp
    src/machine.cpp
    src/primitives.cpp
    src/operators.cpp
    src/parser.cpp
    src/sysvar.cpp
    src/optimizer.cpp
)

# Build APL VM as a library (build once, link everywhere)
add_library(apl_vm STATIC ${APL_SOURCES})
target_link_libraries(apl_vm PUBLIC Eigen3::Eigen)
target_include_directories(apl_vm PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Main executable (REPL with replxx)
add_executable(apl src/main.cpp)
target_link_libraries(apl apl_vm replxx)

# Enable testing
enable_testing()

# Add test subdirectory
add_subdirectory(tests)

# Optimizer timing benchmark (not a ctest — run manually)
add_executable(bench_optimizer src/bench_optimizer.cpp)
target_link_libraries(bench_optimizer apl_vm)

# Installation (optional)
# install(TARGETS apl-repl DESTINATION bin)
