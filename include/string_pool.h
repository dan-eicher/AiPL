// StringPool - String interning for continuation graphs
// Provides stable pointers to strings that live as long as the pool

#pragma once

#include <string>
#include <unordered_set>

namespace apl {

class StringPool {
public:
    StringPool() = default;
    ~StringPool() = default;

    // Intern a string - returns a stable pointer
    // Multiple calls with the same string return the same pointer
    // Pointer remains valid until the pool is destroyed
    const char* intern(const char* str) {
        auto [it, inserted] = pool_.insert(str);
        return it->c_str();
    }

    // Convenience overload for std::string
    const char* intern(const std::string& str) {
        auto [it, inserted] = pool_.insert(str);
        return it->c_str();
    }

    // Get statistics for testing/debugging
    size_t size() const { return pool_.size(); }
    bool contains(const char* str) const {
        return pool_.find(str) != pool_.end();
    }

    // Clear all interned strings (invalidates all previously returned pointers!)
    void clear() {
        pool_.clear();
    }

private:
    std::unordered_set<std::string> pool_;

    // Disable copying (pointers would be invalidated)
    StringPool(const StringPool&) = delete;
    StringPool& operator=(const StringPool&) = delete;
};

} // namespace apl
