#pragma once

#include <cstddef>
#include <string_view>

namespace menos::ir {

// Whether name matches [A-Za-z_][A-Za-z0-9_]* (ASCII): the names the circuit
// builder accepts for parameters and registers (docs/adr/0029 section 3.2).
[[nodiscard]] constexpr bool is_identifier(std::string_view name) noexcept {
    if (name.empty()) {
        return false;
    }
    for (std::size_t i = 0; i < name.size(); ++i) {
        const char c = name[i];
        const bool letter = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
        const bool digit = c >= '0' && c <= '9';
        if (!letter && !(digit && i > 0)) {
            return false;
        }
    }
    return true;
}

}  // namespace menos::ir
