// StringPool - String interning with GC support
// Owns String objects and provides stable pointers via interning

#pragma once

#include "value.h"  // For String class
#include <string_view>
#include <unordered_map>

namespace apl {

class StringPool {
public:
    StringPool() = default;

    ~StringPool() {
        for (auto& [key, str] : pool_) {
            delete str;
        }
    }

    // Intern a string - returns a GC-managed String*
    // Multiple calls with the same string return the same pointer
    String* intern(const char* str) {
        auto it = pool_.find(str);
        if (it != pool_.end()) return it->second;

        String* s = new String(str);
        pool_.emplace(std::string_view(s->c_str()), s);
        return s;
    }

    // Convenience overload for std::string
    String* intern(const std::string& str) {
        return intern(str.c_str());
    }

    // Remove dead strings after GC mark phase
    // Call this during sweep - deletes unmarked Strings
    void sweep_dead() {
        for (auto it = pool_.begin(); it != pool_.end(); ) {
            if (!it->second->marked) {
                delete it->second;
                it = pool_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Clear marks on all strings (call before mark phase)
    void clear_marks() {
        for (auto& [key, str] : pool_) {
            str->marked = false;
        }
    }

    // Get statistics for testing/debugging
    size_t size() const { return pool_.size(); }

    bool contains(const char* str) const {
        return pool_.find(str) != pool_.end();
    }

    // Clear all interned strings (invalidates all previously returned pointers!)
    void clear() {
        for (auto& [key, str] : pool_) {
            delete str;
        }
        pool_.clear();
    }

private:
    std::unordered_map<std::string_view, String*> pool_;

    // Disable copying (pointers would be invalidated)
    StringPool(const StringPool&) = delete;
    StringPool& operator=(const StringPool&) = delete;
};

} // namespace apl
