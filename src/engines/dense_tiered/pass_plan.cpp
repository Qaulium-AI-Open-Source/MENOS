#include "engines/dense_tiered/pass_plan.h"

#include <algorithm>
#include <bit>
#include <type_traits>
#include <variant>

namespace menos::engines::dense_tiered {
namespace {

// The operands of a gate, or null for an operation that applies no matrix.
const std::vector<QubitId>* gate_qubits(const Operation& operation) {
    return std::visit(
        [](const auto& op) -> const std::vector<QubitId>* {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                return &op.qubits;
            } else {
                return nullptr;
            }
        },
        operation);
}

// The global operands of a gate as a mask over the global qubits: bit (q - L)
// for each operand q >= L.
std::uint64_t global_mask_of(const std::vector<QubitId>& qubits, std::uint32_t chunk_qubits) {
    std::uint64_t mask = 0;
    for (const QubitId q : qubits) {
        if (q >= chunk_qubits) {
            mask |= std::uint64_t{1} << (q - chunk_qubits);
        }
    }
    return mask;
}

}  // namespace

// Three qubit counts with distinct roles, named at every call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
PassPlan build_pass_plan(std::span<const Operation> operations,
                         std::uint32_t num_qubits,
                         std::uint32_t chunk_qubits,
                         std::uint32_t window_qubits) {
    PassPlan plan;
    plan.num_qubits = num_qubits;
    plan.chunk_qubits = chunk_qubits;
    plan.window_qubits = window_qubits;
    const int window_globals = static_cast<int>(window_qubits - chunk_qubits);

    // Two walks: the first counts, so the vector is reserved exactly and the
    // plan's own footprint is what the estimate charges for it.
    for (int walk = 0; walk < 2; ++walk) {
        bool open = false;
        std::uint64_t group = 0;
        std::size_t first = 0;
        std::size_t count = 0;
        const auto close = [&](std::size_t end) {
            if (walk == 1) {
                plan.passes.push_back(Pass{group, first, end});
            }
            ++count;
        };
        for (std::size_t i = 0; i < operations.size(); ++i) {
            const std::vector<QubitId>* qubits = gate_qubits(operations[i]);
            if (qubits == nullptr) {
                continue;  // belongs to whichever pass is open
            }
            const std::uint64_t mask = global_mask_of(*qubits, chunk_qubits);
            const int globals = std::popcount(mask);
            if (walk == 0) {
                plan.max_global_operands =
                    std::max(plan.max_global_operands, static_cast<std::uint32_t>(globals));
                if (globals > window_globals) {
                    plan.window_too_small = true;
                }
                if (globals > 0) {
                    ++plan.global_gates;
                }
            }
            if (!open) {
                open = true;
                group = mask;
                continue;
            }
            const std::uint64_t merged = group | mask;
            if (std::popcount(merged) <= window_globals) {
                group = merged;
            } else {
                close(i);
                first = i;
                group = mask;
            }
        }
        if (open) {
            close(operations.size());
        }
        if (walk == 0) {
            plan.passes.reserve(count);
        }
    }
    return plan;
}

std::optional<PassPlan> choose_pass_plan(std::span<const Operation> operations,
                                         std::uint32_t num_qubits,
                                         std::uint32_t window_qubits,
                                         bool needs_chunk_pair) {
    const std::uint32_t lowest =
        window_qubits > kPreferredMinChunkQubits ? kPreferredMinChunkQubits : 0;
    std::optional<PassPlan> best;
    // Larger chunks first, so a tie keeps the larger chunk. Chunks below the
    // preferred minimum are tried only when nothing above it fits the gates.
    for (std::uint32_t chunk_qubits = window_qubits;; --chunk_qubits) {
        if (chunk_qubits < lowest && best.has_value()) {
            break;
        }
        const std::uint32_t window_globals = window_qubits - chunk_qubits;
        if (!needs_chunk_pair || window_globals >= 1) {
            PassPlan plan = build_pass_plan(operations, num_qubits, chunk_qubits, window_qubits);
            if (!plan.window_too_small &&
                (!best.has_value() || plan.pass_count() < best->pass_count())) {
                best = std::move(plan);
            }
        }
        if (chunk_qubits == 0) {
            break;
        }
    }
    return best;
}

}  // namespace menos::engines::dense_tiered
