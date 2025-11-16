// Control implementation

#include "control.h"
#include "lexer.h"

namespace apl {

void Control::advance_token() {
    if (lexer_state) {
        current_token = lex_next_token(lexer_state);
    }
}

} // namespace apl
