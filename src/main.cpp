// APL REPL using replxx
// UTF-8 support for APL symbols, syntax highlighting, and completion

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <unistd.h>

#include "replxx.hxx"
#include "machine.h"
#include "value.h"

using Replxx = replxx::Replxx;
using namespace replxx::color;

// APL primitives for completion
static const std::vector<std::string> APL_PRIMITIVES = {
    // Arithmetic
    "+", "-", "×", "÷", "*", "⌈", "⌊", "|", "⍟", "!", "○",
    // Comparison
    "=", "≠", "<", ">", "≤", "≥",
    // Logical
    "∧", "∨", "~", "⍲", "⍱",
    // Array
    "⍴", ",", "⍉", "⍳", "↑", "↓", "⌽", "⊖", "≢", "∊", "⍋", "⍒", "∪", "⌷",
    // Special
    "?", "⊥", "⊤", "⌹", "⍎", "⍕", "⍸",
    // Operators
    "/", "⌿", "\\", "⍀", "¨", "⍨", ".", "∘.", "⍤",
    // Symbols
    "←", "⍬", "⍺", "⍵", "⋄",
};

// Color scheme for APL syntax highlighting
static std::unordered_map<std::string, Replxx::Color> APL_COLORS = {
    // Arithmetic - cyan
    {"+", Replxx::Color::BRIGHTCYAN},
    {"-", Replxx::Color::BRIGHTCYAN},
    {"×", Replxx::Color::BRIGHTCYAN},
    {"÷", Replxx::Color::BRIGHTCYAN},
    {"*", Replxx::Color::BRIGHTCYAN},
    {"⌈", Replxx::Color::BRIGHTCYAN},
    {"⌊", Replxx::Color::BRIGHTCYAN},
    {"|", Replxx::Color::BRIGHTCYAN},
    {"⍟", Replxx::Color::BRIGHTCYAN},
    {"!", Replxx::Color::BRIGHTCYAN},
    {"○", Replxx::Color::BRIGHTCYAN},
    // Comparison - blue
    {"=", Replxx::Color::BRIGHTBLUE},
    {"≠", Replxx::Color::BRIGHTBLUE},
    {"<", Replxx::Color::BRIGHTBLUE},
    {">", Replxx::Color::BRIGHTBLUE},
    {"≤", Replxx::Color::BRIGHTBLUE},
    {"≥", Replxx::Color::BRIGHTBLUE},
    // Logical - blue
    {"∧", Replxx::Color::BRIGHTBLUE},
    {"∨", Replxx::Color::BRIGHTBLUE},
    {"~", Replxx::Color::BRIGHTBLUE},
    {"⍲", Replxx::Color::BRIGHTBLUE},
    {"⍱", Replxx::Color::BRIGHTBLUE},
    // Array operations - green
    {"⍴", Replxx::Color::BRIGHTGREEN},
    {",", Replxx::Color::BRIGHTGREEN},
    {"⍉", Replxx::Color::BRIGHTGREEN},
    {"⍳", Replxx::Color::BRIGHTGREEN},
    {"↑", Replxx::Color::BRIGHTGREEN},
    {"↓", Replxx::Color::BRIGHTGREEN},
    {"⌽", Replxx::Color::BRIGHTGREEN},
    {"⊖", Replxx::Color::BRIGHTGREEN},
    {"≢", Replxx::Color::BRIGHTGREEN},
    {"∊", Replxx::Color::BRIGHTGREEN},
    {"⍋", Replxx::Color::BRIGHTGREEN},
    {"⍒", Replxx::Color::BRIGHTGREEN},
    {"∪", Replxx::Color::BRIGHTGREEN},
    {"⌷", Replxx::Color::BRIGHTGREEN},
    {"⍸", Replxx::Color::BRIGHTGREEN},
    // Special - magenta
    {"?", Replxx::Color::BRIGHTMAGENTA},
    {"⊥", Replxx::Color::BRIGHTMAGENTA},
    {"⊤", Replxx::Color::BRIGHTMAGENTA},
    {"⌹", Replxx::Color::BRIGHTMAGENTA},
    {"⍎", Replxx::Color::BRIGHTMAGENTA},
    {"⍕", Replxx::Color::BRIGHTMAGENTA},
    // Operators - red
    {"/", Replxx::Color::BRIGHTRED},
    {"⌿", Replxx::Color::BRIGHTRED},
    {"\\", Replxx::Color::BRIGHTRED},
    {"⍀", Replxx::Color::BRIGHTRED},
    {"¨", Replxx::Color::BRIGHTRED},
    {"⍨", Replxx::Color::BRIGHTRED},
    {".", Replxx::Color::BRIGHTRED},
    {"⍤", Replxx::Color::BRIGHTRED},
    {"∘", Replxx::Color::BRIGHTRED},
    // Assignment and special - yellow
    {"←", Replxx::Color::YELLOW},
    {"⍬", Replxx::Color::YELLOW},
    {"⋄", Replxx::Color::YELLOW},
    // Alpha/Omega - white bold
    {"⍺", Replxx::Color::WHITE},
    {"⍵", Replxx::Color::WHITE},
    // Brackets - magenta
    {"(", Replxx::Color::MAGENTA},
    {")", Replxx::Color::MAGENTA},
    {"[", Replxx::Color::MAGENTA},
    {"]", Replxx::Color::MAGENTA},
    {"{", Replxx::Color::MAGENTA},
    {"}", Replxx::Color::MAGENTA},
};

// Get UTF-8 character length
static int utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Syntax highlighting callback
void hook_color(const std::string& context, Replxx::colors_t& colors,
                const std::unordered_map<std::string, Replxx::Color>& word_color) {
    size_t i = 0;
    int color_idx = 0;

    while (i < context.length()) {
        int char_len = utf8_char_len(static_cast<unsigned char>(context[i]));
        std::string ch = context.substr(i, char_len);

        auto it = word_color.find(ch);
        if (it != word_color.end()) {
            colors[color_idx] = it->second;
        } else if (isdigit(context[i]) || context[i] == '.' ||
                   (char_len == 2 && context[i] == '\xc2' && context[i+1] == '\xaf')) {
            // Numbers and high minus (¯) - yellow
            colors[color_idx] = Replxx::Color::YELLOW;
        } else if (context[i] == '\'') {
            // String literals - green
            colors[color_idx] = Replxx::Color::GREEN;
        }

        i += char_len;
        ++color_idx;
    }
}

// Completion callback
Replxx::completions_t hook_completion(const std::string& context, int& contextLen,
                                       const std::vector<std::string>& examples) {
    Replxx::completions_t completions;

    // Find the last word/symbol
    size_t start = context.length();
    while (start > 0) {
        int len = 1;
        // Check for multi-byte UTF-8
        if (start >= 2) {
            unsigned char c = context[start - 2];
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
        }
        if (start >= (size_t)len) {
            std::string ch = context.substr(start - len, len);
            if (ch == " " || ch == "\t" || ch == "\n") break;
        }
        start -= len;
    }

    std::string prefix = context.substr(start);
    contextLen = 0;
    for (size_t j = 0; j < prefix.length(); ) {
        j += utf8_char_len(static_cast<unsigned char>(prefix[j]));
        ++contextLen;
    }

    for (const auto& e : examples) {
        if (prefix.empty() || e.find(prefix) == 0) {
            completions.emplace_back(e);
        }
    }

    return completions;
}

// Format a single number with APL conventions
static std::string format_number(double val) {
    if (std::isinf(val)) {
        return val > 0 ? "∞" : "¯∞";
    }
    if (std::isnan(val)) {
        return "NaN";
    }
    if (val == std::floor(val) && std::abs(val) < 1e15) {
        if (val < 0) {
            return "¯" + std::to_string(static_cast<long long>(-val));
        }
        return std::to_string(static_cast<long long>(val));
    }
    std::ostringstream oss;
    if (val < 0) {
        oss << "¯" << -val;
    } else {
        oss << val;
    }
    return oss.str();
}

// Format value for display
std::string format_value(apl::Value* v, apl::Machine* m) {
    if (!v) return "nil";

    if (v->tag == apl::ValueType::SCALAR) {
        return format_number(v->as_scalar());
    }

    if (v->tag == apl::ValueType::STRING) {
        return v->as_string();
    }

    if (v->is_array()) {
        std::ostringstream oss;
        const Eigen::MatrixXd* mat = v->as_matrix();
        if (v->is_vector()) {
            for (int i = 0; i < mat->rows(); ++i) {
                if (i > 0) oss << " ";
                oss << format_number((*mat)(i, 0));
            }
        } else {
            for (int i = 0; i < mat->rows(); ++i) {
                if (i > 0) oss << "\n";
                for (int j = 0; j < mat->cols(); ++j) {
                    if (j > 0) oss << " ";
                    oss << format_number((*mat)(i, j));
                }
            }
        }
        return oss.str();
    }

    return "[function]";
}

int main(int argc, char** argv) {
    // Create APL machine
    apl::Machine machine;

    // Init replxx with stdin/stdout
    Replxx rx(std::cin, std::cout, STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO);
    rx.install_window_change_handler();

    // History file
    std::string history_file = ".apl_history";
    {
        std::ifstream hf(history_file);
        rx.history_load(hf);
    }
    rx.set_max_history_size(1000);

    // Set callbacks using lambdas
    rx.set_completion_callback([](const std::string& ctx, int& len) {
        return hook_completion(ctx, len, APL_PRIMITIVES);
    });
    rx.set_highlighter_callback([](const std::string& ctx, Replxx::colors_t& colors) {
        hook_color(ctx, colors, APL_COLORS);
    });

    // Configure
    rx.set_word_break_characters(" \t\n");
    rx.set_completion_count_cutoff(128);
    rx.set_double_tab_completion(false);
    rx.set_complete_on_empty(true);
    rx.set_beep_on_ambiguous_completion(false);
    rx.set_no_color(false);

    // Welcome message
    std::cout << "APL Interpreter (ISO 13751)\n";
    std::cout << "Type )help for help, )quit to exit\n\n";

    // Prompt
    std::string prompt = "      ";  // 6 spaces, traditional APL indent

    // Main loop
    for (;;) {
        const char* input = nullptr;

        do {
            input = rx.input(prompt);
        } while (input == nullptr && errno == EAGAIN);

        if (input == nullptr) {
            // EOF (Ctrl-D)
            break;
        }

        std::string line(input);

        if (line.empty()) {
            continue;
        }

        // System commands
        if (line[0] == ')') {
            rx.history_add(line);

            if (line == ")quit" || line == ")exit" || line == ")off") {
                break;
            } else if (line == ")help") {
                std::cout << "APL System Commands:\n";
                std::cout << "  )help    - show this help\n";
                std::cout << "  )quit    - exit the interpreter\n";
                std::cout << "  )vars    - list defined variables\n";
                std::cout << "  )clear   - clear the screen\n";
                std::cout << "\nAPL Primitives: + - × ÷ * ⌈ ⌊ | ⍟ ! ○\n";
                std::cout << "Comparison: = ≠ < > ≤ ≥\n";
                std::cout << "Logical: ∧ ∨ ~ ⍲ ⍱\n";
                std::cout << "Array: ⍴ , ⍉ ⍳ ↑ ↓ ⌽ ⊖ ≢ ∊ ⍋ ⍒ ∪ ⌷ ⍸\n";
                std::cout << "Operators: / ⌿ \\ ⍀ ¨ ⍨ . ∘. ⍤\n";
                std::cout << "Special: ? ⊥ ⊤ ⌹ ⍎ ⍕ ← ⍬ ⋄\n";
            } else if (line == ")clear") {
                rx.clear_screen();
            } else if (line == ")vars") {
                std::cout << "Variables: ";
                bool first = true;
                for (const auto& [name, val] : machine.env->bindings) {
                    if (val && val->tag != apl::ValueType::PRIMITIVE &&
                        val->tag != apl::ValueType::OPERATOR) {
                        if (!first) std::cout << ", ";
                        std::cout << name;
                        first = false;
                    }
                }
                std::cout << "\n";
            } else {
                std::cout << "Unknown command: " << line << "\n";
            }
            continue;
        }

        // Evaluate APL expression
        rx.history_add(line);

        try {
            apl::Value* result = machine.eval(line);
            if (result) {
                std::cout << format_value(result, &machine) << "\n";
            }
        } catch (const apl::APLError& e) {
            std::cout << e.what() << "\n";
            if (!machine.error_stack.empty()) {
                std::cout << machine.format_stack_trace();
            }
        } catch (const std::exception& e) {
            std::cout << "ERROR: " << e.what() << "\n";
        }
    }

    // Save history
    {
        std::ofstream hf(history_file);
        rx.history_save(hf);
    }

    std::cout << "\n";
    return 0;
}
