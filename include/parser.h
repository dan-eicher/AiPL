// APL Parser using Boost.Spirit X3
// Builds continuation graphs with right-to-left evaluation semantics

#pragma once

#include "continuation.h"
#include <string>

namespace apl {

// Forward declaration
class APLHeap;

// Parser class that builds continuation graphs
class Parser {
public:
    Parser(APLHeap* heap) : heap_(heap) {}

    // Parse an APL expression and return the continuation graph
    // Returns nullptr on parse failure
    Continuation* parse(const std::string& input);

    // Get the last error message (if parse failed)
    const std::string& get_error() const { return error_message_; }

private:
    APLHeap* heap_;
    std::string error_message_;
};

} // namespace apl
