// Token implementation

#include "token.h"

namespace apl {

const char* token_type_name(TokenType type) {
    switch (type) {
        case TOK_EOF: return "EOF";
        case TOK_NUMBER: return "NUMBER";
        case TOK_NUMBER_VECTOR: return "NUMBER_VECTOR";
        case TOK_NAME: return "NAME";
        case TOK_STRING: return "STRING";
        case TOK_PLUS: return "+";
        case TOK_MINUS: return "-";
        case TOK_TIMES: return "×";
        case TOK_DIVIDE: return "÷";
        case TOK_POWER: return "*";
        case TOK_RESHAPE: return "⍴";
        case TOK_RAVEL: return ",";
        case TOK_TRANSPOSE: return "⍉";
        case TOK_IOTA: return "⍳";
        case TOK_TAKE: return "↑";
        case TOK_DROP: return "↓";
        case TOK_REDUCE: return "/";
        case TOK_REDUCE_FIRST: return "⌿";
        case TOK_SCAN: return "\\";
        case TOK_SCAN_FIRST: return "⍀";
        case TOK_EACH: return "¨";
        case TOK_COMPOSE: return "∘";
        case TOK_COMMUTE: return "⍨";
        case TOK_RANK: return "⍤";
        case TOK_DOT: return ".";
        case TOK_OUTER_PRODUCT: return "∘.";
        case TOK_EQUAL: return "=";
        case TOK_NOT_EQUAL: return "≠";
        case TOK_LESS: return "<";
        case TOK_LESS_EQUAL: return "≤";
        case TOK_GREATER: return ">";
        case TOK_GREATER_EQUAL: return "≥";
        case TOK_AND: return "∧";
        case TOK_OR: return "∨";
        case TOK_NOT: return "~";
        case TOK_NAND: return "⍲";
        case TOK_NOR: return "⍱";
        case TOK_CEILING: return "⌈";
        case TOK_FLOOR: return "⌊";
        case TOK_STILE: return "|";
        case TOK_LOG: return "⍟";
        case TOK_FACTORIAL: return "!";
        case TOK_REVERSE: return "⌽";
        case TOK_REVERSE_FIRST: return "⊖";
        case TOK_TALLY: return "≢";
        case TOK_ASSIGN: return "←";
        case TOK_LPAREN: return "(";
        case TOK_RPAREN: return ")";
        case TOK_LBRACKET: return "[";
        case TOK_RBRACKET: return "]";
        case TOK_SEMICOLON: return ";";
        case TOK_IF: return ":If";
        case TOK_ELSE: return ":Else";
        case TOK_ELSEIF: return ":ElseIf";
        case TOK_ENDIF: return ":EndIf";
        case TOK_WHILE: return ":While";
        case TOK_ENDWHILE: return ":EndWhile";
        case TOK_FOR: return ":For";
        case TOK_IN: return ":In";
        case TOK_ENDFOR: return ":EndFor";
        case TOK_LEAVE: return ":Leave";
        case TOK_RETURN: return ":Return";
        case TOK_GOTO: return "→";
        case TOK_LBRACE: return "{";
        case TOK_RBRACE: return "}";
        case TOK_ALPHA: return "⍺";
        case TOK_OMEGA: return "⍵";
        case TOK_DIAMOND: return "⋄";
        case TOK_NEWLINE: return "NEWLINE";
        case TOK_COMMENT: return "COMMENT";
        case TOK_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

} // namespace apl
