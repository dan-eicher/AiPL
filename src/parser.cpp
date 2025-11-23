// APL Parser implementation using Boost.Spirit X3

#include "parser.h"
#include "heap.h"
#include "continuation.h"
#include <boost/spirit/home/x3.hpp>
#include <stdexcept>

namespace apl {

namespace x3 = boost::spirit::x3;

// Parser implementation
Continuation* Parser::parse(const std::string& input) {
    error_message_.clear();

    // Iterator types
    auto iter = input.begin();
    auto end = input.end();

    // For now, let's implement a simple expression parser
    // that handles: numbers and binary operations (+, -, ×, ÷)

    // We'll build this incrementally, starting with just literals

    // Parse a single number for now
    double value = 0.0;
    bool success = x3::phrase_parse(iter, end, x3::double_, x3::space, value);

    if (!success || iter != end) {
        error_message_ = "Parse failed";
        return nullptr;
    }

    // Create a LiteralK continuation
    HaltK* halt = new HaltK();
    LiteralK* lit = new LiteralK(value, halt);

    heap_->allocate_continuation(halt);
    heap_->allocate_continuation(lit);

    return lit;
}

} // namespace apl
