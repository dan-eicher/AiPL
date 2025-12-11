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
    COUNT,  // Number of system variables (must be last)
    INVALID = 255
};

// Bitmask constants for enabling/disabling system variables
constexpr uint32_t SYSVAR_IO  = 1u << static_cast<uint8_t>(SysVarId::IO);
constexpr uint32_t SYSVAR_PP  = 1u << static_cast<uint8_t>(SysVarId::PP);
constexpr uint32_t SYSVAR_CT  = 1u << static_cast<uint8_t>(SysVarId::CT);
constexpr uint32_t SYSVAR_RL  = 1u << static_cast<uint8_t>(SysVarId::RL);
constexpr uint32_t SYSVAR_ALL = ~0u;
constexpr uint32_t SYSVAR_NONE = 0u;

// Lookup system variable by name
// Returns SysVarId::INVALID if name is unknown or disabled by mask
SysVarId lookup_sysvar(const std::string& name, uint32_t enabled_mask = SYSVAR_ALL);

// Get the name of a system variable (for error messages)
const char* sysvar_name(SysVarId id);

} // namespace apl
