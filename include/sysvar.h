// System Variables - ⎕IO, ⎕PP, etc.
// Single source of truth for system variable definitions

#pragma once

#include <cstdint>
#include <string>

namespace apl {

// System variable identifiers
// Values must be sequential starting from 0 for bitmask to work
enum class SysVarId : uint8_t {
    IO,     // Index origin (0 or 1)
    PP,     // Print precision (1-17)
    CT,     // Comparison tolerance (nonnegative, default ~1e-14)
    RL,     // Random link (positive integer seed)
    ET,     // Event type (read-only, 2-element vector)
    EM,     // Event message (read-only, character vector)
    TS,     // Time stamp (read-only, 7-element vector) - ISO 13751 §11.4.1
    AV,     // Atomic vector (read-only, character vector) - ISO 13751 §11.4.2
    LC,     // Line counter (read-only, integer vector) - ISO 13751 §11.4.3
    LX,     // Latent expression (character vector) - ISO 13751 §12.2.5
    COUNT,  // Number of system variables (must be last)
    INVALID = 255
};

// Bitmask constants for enabling/disabling system variables
constexpr uint32_t SYSVAR_IO  = 1u << static_cast<uint8_t>(SysVarId::IO);
constexpr uint32_t SYSVAR_PP  = 1u << static_cast<uint8_t>(SysVarId::PP);
constexpr uint32_t SYSVAR_CT  = 1u << static_cast<uint8_t>(SysVarId::CT);
constexpr uint32_t SYSVAR_RL  = 1u << static_cast<uint8_t>(SysVarId::RL);
constexpr uint32_t SYSVAR_ET  = 1u << static_cast<uint8_t>(SysVarId::ET);
constexpr uint32_t SYSVAR_EM  = 1u << static_cast<uint8_t>(SysVarId::EM);
constexpr uint32_t SYSVAR_TS  = 1u << static_cast<uint8_t>(SysVarId::TS);
constexpr uint32_t SYSVAR_AV  = 1u << static_cast<uint8_t>(SysVarId::AV);
constexpr uint32_t SYSVAR_LC  = 1u << static_cast<uint8_t>(SysVarId::LC);
constexpr uint32_t SYSVAR_LX  = 1u << static_cast<uint8_t>(SysVarId::LX);
constexpr uint32_t SYSVAR_ALL = ~0u;
constexpr uint32_t SYSVAR_NONE = 0u;

// Check if a system variable is read-only
inline bool sysvar_is_readonly(SysVarId id) {
    return id == SysVarId::ET || id == SysVarId::EM ||
           id == SysVarId::TS || id == SysVarId::AV || id == SysVarId::LC;
}

// Lookup system variable by name
// Returns SysVarId::INVALID if name is unknown or disabled by mask
SysVarId lookup_sysvar(const std::string& name, uint32_t enabled_mask = SYSVAR_ALL);

// Get the name of a system variable (for error messages)
const char* sysvar_name(SysVarId id);

} // namespace apl
