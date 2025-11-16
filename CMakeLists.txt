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

# Find re2c for lexer generation
find_program(RE2C_EXECUTABLE re2c)
if(NOT RE2C_EXECUTABLE)
    message(FATAL_ERROR "re2c not found. Install with: sudo dnf install re2c")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files
set(APL_SOURCES
    # Will be added as we implement
)

# Generate lexer from re2c specification
# TODO: Add custom command once lexer.re is created
# add_custom_command(
#     OUTPUT ${CMAKE_SOURCE_DIR}/src/lexer.cpp
#     COMMAND ${RE2C_EXECUTABLE} -o ${CMAKE_SOURCE_DIR}/src/lexer.cpp ${CMAKE_SOURCE_DIR}/src/lexer.re
#     DEPENDS ${CMAKE_SOURCE_DIR}/src/lexer.re
#     COMMENT "Generating lexer with re2c"
# )

# Main executable (for REPL)
# add_executable(apl-repl ${APL_SOURCES})
# target_link_libraries(apl-repl Eigen3::Eigen)

# Enable testing
enable_testing()

# Add test subdirectory
add_subdirectory(tests)

# Installation (optional)
# install(TARGETS apl-repl DESTINATION bin)
