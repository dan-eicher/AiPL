// System Variables implementation

#include "sysvar.h"

namespace apl {

SysVarId lookup_sysvar(const std::string& name, uint32_t enabled_mask) {
    SysVarId id = SysVarId::INVALID;

    // Lookup by name
    if (name == "IO") {
        id = SysVarId::IO;
    } else if (name == "PP") {
        id = SysVarId::PP;
    } else if (name == "CT") {
        id = SysVarId::CT;
    } else if (name == "RL") {
        id = SysVarId::RL;
    } else if (name == "ET") {
        id = SysVarId::ET;
    } else if (name == "EM") {
        id = SysVarId::EM;
    }

    // Check if disabled
    if (id != SysVarId::INVALID) {
        uint32_t bit = 1u << static_cast<uint8_t>(id);
        if ((enabled_mask & bit) == 0) {
            return SysVarId::INVALID;
        }
    }

    return id;
}

const char* sysvar_name(SysVarId id) {
    switch (id) {
        case SysVarId::IO: return "IO";
        case SysVarId::PP: return "PP";
        case SysVarId::CT: return "CT";
        case SysVarId::RL: return "RL";
        case SysVarId::ET: return "ET";
        case SysVarId::EM: return "EM";
        default: return "?";
    }
}

} // namespace apl
