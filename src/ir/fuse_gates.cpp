#include "ir/fuse_gates.h"

#include <array>
#include <bit>
#include <limits>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "ir/fuse_diagonal.h"

namespace menos::ir {
namespace {

constexpr std::uint32_t kNoBlock = std::numeric_limits<std::uint32_t>::max();

// The operands of a gate as a bit mask, or nullopt for an operation that
// applies no matrix.
struct GateShape {
    bool is_gate = false;
    std::uint64_t mask = 0;
    bool diagonal = false;
};

GateShape shape_of(const Operation& operation) {
    return std::visit(
        [&](const auto& op) -> GateShape {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                GateShape shape;
                shape.is_gate = true;
                for (const QubitId q : op.qubits) {
                    if (q >= 64) {
                        throw common::InternalError("plan_fusion: an operand is beyond 63 qubits");
                    }
                    shape.mask |= std::uint64_t{1} << q;
                }
                shape.diagonal = is_diagonal_operation(operation);
                return shape;
            } else {
                return GateShape{};
            }
        },
        operation);
}

// Whether a gate of `gate` qubits may join an open block of `block` qubits
// (the containment rule in the header). The two masks and the two flags are
// the block's and the gate's, in that order, at the one call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
bool may_join(std::uint64_t block, bool block_diagonal, std::uint64_t gate, bool gate_diagonal) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    if (block == gate) {
        return true;
    }
    if ((block & gate) == block) {  // the gate is strictly wider
        return !gate_diagonal || block_diagonal;
    }
    if ((block & gate) == gate) {  // the block is strictly wider
        return !block_diagonal || gate_diagonal;
    }
    return false;
}

}  // namespace

std::uint64_t FusionPlan::heap_bytes() const {
    using common::checked_add;
    using common::checked_mul;
    using common::heap_block_bytes;
    std::uint64_t bytes = heap_block_bytes(checked_mul(order.size(), sizeof(std::uint32_t)));
    bytes = checked_add(bytes,
                        heap_block_bytes(checked_mul(block_begin.size(), sizeof(std::uint32_t))));
    bytes = checked_add(bytes, heap_block_bytes(checked_mul(block_qubits.size(), sizeof(QubitId))));
    bytes = checked_add(bytes,
                        heap_block_bytes(checked_mul(qubits_begin.size(), sizeof(std::uint32_t))));
    return bytes;
}

// Three counts with distinct roles, named at every call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::uint64_t FusionPlan::construction_peak_bytes(std::size_t operations,
                                                  std::size_t blocks,
                                                  std::size_t block_qubits) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    using common::checked_add;
    using common::checked_mul;
    using common::heap_block_bytes;
    // The working vectors of plan_fusion, every one reserved for `operations`
    // entries before the block count is known: the block of each operation,
    // and each block's qubit mask and diagonal flag.
    std::uint64_t bytes = heap_block_bytes(checked_mul(operations, sizeof(std::uint32_t)));
    bytes = checked_add(bytes, heap_block_bytes(checked_mul(operations, sizeof(std::uint64_t))));
    bytes = checked_add(bytes, heap_block_bytes(operations));
    // The counting sort's running fill of each block.
    bytes = checked_add(bytes, heap_block_bytes(checked_mul(blocks, sizeof(std::uint32_t))));
    // The finished vectors, which the working ones outlive by nothing: they
    // are all alive together just before plan_fusion returns.
    bytes = checked_add(bytes, heap_block_bytes(checked_mul(operations, sizeof(std::uint32_t))));
    bytes = checked_add(bytes, heap_block_bytes(checked_mul(blocks + 1, sizeof(std::uint32_t))));
    bytes = checked_add(bytes, heap_block_bytes(checked_mul(block_qubits, sizeof(QubitId))));
    bytes = checked_add(bytes, heap_block_bytes(checked_mul(blocks + 1, sizeof(std::uint32_t))));
    return bytes;
}

FusionPlan plan_fusion(std::span<const Operation> operations) {
    const std::size_t count = operations.size();
    common::ensure(count < kNoBlock, "plan_fusion: too many operations");

    // Working state, each vector reserved once for the most it can hold, so
    // the heap plan_fusion uses is exactly construction_peak_bytes().
    std::vector<std::uint32_t> block_of(count);
    std::vector<std::uint64_t> block_mask;
    std::vector<unsigned char> block_diagonal;
    block_mask.reserve(count);
    block_diagonal.reserve(count);
    // The newest open block holding each qubit.
    std::array<std::uint32_t, 64> newest{};
    newest.fill(kNoBlock);

    for (std::size_t i = 0; i < count; ++i) {
        const GateShape gate = shape_of(operations[i]);
        if (!gate.is_gate) {
            // A barrier: its own block, and every block before it is closed.
            block_of[i] = static_cast<std::uint32_t>(block_mask.size());
            block_mask.push_back(0);
            block_diagonal.push_back(0);
            newest.fill(kNoBlock);
            continue;
        }
        std::uint32_t dependency = kNoBlock;
        for (std::uint64_t rest = gate.mask; rest != 0; rest &= rest - 1) {
            const std::uint32_t b = newest[static_cast<std::size_t>(std::countr_zero(rest))];
            if (b != kNoBlock && (dependency == kNoBlock || b > dependency)) {
                dependency = b;
            }
        }
        std::uint32_t block = kNoBlock;
        if (dependency != kNoBlock && may_join(block_mask[dependency],
                                               block_diagonal[dependency] != 0,
                                               gate.mask,
                                               gate.diagonal)) {
            block = dependency;
            block_mask[block] |= gate.mask;
            block_diagonal[block] =
                static_cast<unsigned char>(block_diagonal[block] != 0 && gate.diagonal);
        } else {
            block = static_cast<std::uint32_t>(block_mask.size());
            block_mask.push_back(gate.mask);
            block_diagonal.push_back(gate.diagonal ? 1 : 0);
        }
        block_of[i] = block;
        for (std::uint64_t rest = gate.mask; rest != 0; rest &= rest - 1) {
            newest[static_cast<std::size_t>(std::countr_zero(rest))] = block;
        }
    }

    const std::size_t blocks = block_mask.size();
    FusionPlan plan;
    plan.block_begin.assign(blocks + 1, 0);
    for (std::size_t i = 0; i < count; ++i) {
        ++plan.block_begin[block_of[i] + 1];
    }
    for (std::size_t b = 0; b < blocks; ++b) {
        plan.block_begin[b + 1] += plan.block_begin[b];
    }
    plan.order.assign(count, 0);
    {
        // A stable counting sort of the operations by block: block_of is
        // reused as the running fill of each block.
        std::vector<std::uint32_t>& next = block_of;
        std::vector<std::uint32_t> fill(plan.block_begin.begin(), plan.block_begin.end() - 1);
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint32_t b = next[i];
            plan.order[fill[b]] = static_cast<std::uint32_t>(i);
            ++fill[b];
        }
    }
    plan.qubits_begin.assign(blocks + 1, 0);
    for (std::size_t b = 0; b < blocks; ++b) {
        plan.qubits_begin[b + 1] =
            plan.qubits_begin[b] + static_cast<std::uint32_t>(std::popcount(block_mask[b]));
    }
    plan.block_qubits.assign(plan.qubits_begin[blocks], 0);
    for (std::size_t b = 0; b < blocks; ++b) {
        std::uint32_t at = plan.qubits_begin[b];
        for (std::uint64_t rest = block_mask[b]; rest != 0; rest &= rest - 1) {
            plan.block_qubits[at++] = static_cast<QubitId>(std::countr_zero(rest));
        }
    }
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::uint32_t size = plan.block_begin[b + 1] - plan.block_begin[b];
        if (size >= 2) {
            ++plan.fused_blocks;
            plan.gates_absorbed += size - 1;
        }
    }
    return plan;
}

}  // namespace menos::ir
