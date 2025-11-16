// LexerArena - Fast arena allocator for temporary token strings
// Allocates strings in 4KB blocks, can be reset after parsing

#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

namespace apl {

class LexerArena {
public:
    static constexpr size_t BLOCK_SIZE = 4096;

    LexerArena() : current_block_(nullptr), block_offset_(BLOCK_SIZE) {
        // Start with block_offset_ = BLOCK_SIZE to force allocation on first use
    }

    ~LexerArena() {
        reset();
    }

    // Allocate a copy of the string in the arena
    // Returns pointer to arena-allocated null-terminated string
    char* allocate_string(const char* str, size_t len) {
        // Need space for string + null terminator
        size_t needed = len + 1;

        // If it doesn't fit in current block, allocate a new one
        if (block_offset_ + needed > BLOCK_SIZE) {
            allocate_new_block();
        }

        // Copy string into current block
        char* result = current_block_ + block_offset_;
        std::memcpy(result, str, len);
        result[len] = '\0';
        block_offset_ += needed;

        return result;
    }

    // Convenience overload for null-terminated strings
    char* allocate_string(const char* str) {
        return allocate_string(str, std::strlen(str));
    }

    // Reset arena - deallocates all blocks
    // Call this after parsing is complete and tokens are no longer needed
    void reset() {
        for (char* block : blocks_) {
            delete[] block;
        }
        blocks_.clear();
        current_block_ = nullptr;
        block_offset_ = BLOCK_SIZE;  // Force new allocation on next use
    }

    // Get statistics for testing/debugging
    size_t num_blocks() const { return blocks_.size(); }
    size_t bytes_used_in_current_block() const {
        return current_block_ ? block_offset_ : 0;
    }

private:
    void allocate_new_block() {
        current_block_ = new char[BLOCK_SIZE];
        blocks_.push_back(current_block_);
        block_offset_ = 0;
    }

    std::vector<char*> blocks_;
    char* current_block_;
    size_t block_offset_;

    // Disable copying
    LexerArena(const LexerArena&) = delete;
    LexerArena& operator=(const LexerArena&) = delete;
};

} // namespace apl
