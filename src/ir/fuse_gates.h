#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "menos/circuit.h"

namespace menos::ir {

// Gate fusion by containment (Phase 24 item 7, docs/adr/0028 section 17):
// the dense engines sweep the state once per BLOCK of gates instead of once
// per gate. A plan says which operations form each block and in what order
// the blocks run; the engines build each block's matrix as they reach it
// (engines/dense/fused_block.h) and apply it with the unchanged kernels.
//
// The grouping rule, read from the step-1 measurement (section 17.5): a gate
// joins the newest open block that shares a qubit with it only when one of
// the two qubit sets contains the other, so the fused block has the operand
// count of the wider of the two and the sweep count falls by one while no
// pass gets wider. A dense gate never joins a strictly wider diagonal block,
// and a dense block never takes a strictly wider diagonal gate, because that
// would turn a cheap streamed diagonal pass into a dense one. Every other
// pair is left apart. A gate whose qubits meet no open block opens a new one;
// an operation that applies no matrix (MeasureOp; ChannelOp never reaches a
// dense engine) is a barrier: a block of its own, after which every block is
// closed. Blocks are emitted in the order they opened, which is sound: a gate
// only ever joins the newest block that shares a qubit with it, and every
// block opened after that one is on qubits disjoint from the gate's, so it
// commutes with them.
//
// The rule reads nothing from the machine: the same operations give the same
// plan on every platform, thread count and backend, so the fused run's
// results are the same bytes everywhere the unfused run's were (sections 17.2
// and 17.6). It never adds arithmetic to a pass, which is why it is on by
// default at every size (section 17.6, Q2).
struct FusionPlan {
    // The operations' indices in execution order, blocks contiguous:
    // block b is order[block_begin[b] .. block_begin[b + 1]).
    std::vector<std::uint32_t> order;
    std::vector<std::uint32_t> block_begin;  // block count + 1 entries
    // Block b's operands, ascending: block_qubits[qubits_begin[b] ..
    // qubits_begin[b + 1]); empty for a barrier block.
    std::vector<QubitId> block_qubits;
    std::vector<std::uint32_t> qubits_begin;  // block count + 1 entries
    // Blocks of two or more gates, and the gates those blocks hold beyond
    // their first: what explain() reports as the fusion outcome.
    std::uint64_t fused_blocks = 0;
    std::uint64_t gates_absorbed = 0;

    [[nodiscard]] std::size_t block_count() const {
        return block_begin.empty() ? 0 : block_begin.size() - 1;
    }
    [[nodiscard]] std::span<const std::uint32_t> block_operations(std::size_t block) const {
        return std::span<const std::uint32_t>(order).subspan(
            block_begin[block], block_begin[block + 1] - block_begin[block]);
    }
    [[nodiscard]] std::span<const QubitId> block_operands(std::size_t block) const {
        return std::span<const QubitId>(block_qubits)
            .subspan(qubits_begin[block], qubits_begin[block + 1] - qubits_begin[block]);
    }

    // The heap the four vectors hold, each sized exactly, as the estimate
    // charges a plan that lives through the sweep (docs/adr/0025 section 3.4).
    [[nodiscard]] std::uint64_t heap_bytes() const;
    // The most heap plan_fusion holds at once while building the plan of
    // `operations` operations that ends with `blocks` blocks and
    // `block_qubits` operands in all: the finished vectors plus the working
    // ones, which are sized by the operation count before the block count
    // is known. Charged by an estimate as a transient that precedes the
    // state's allocation.
    [[nodiscard]] static std::uint64_t construction_peak_bytes(std::size_t operations,
                                                               std::size_t blocks,
                                                               std::size_t block_qubits);
};

// The fusion plan of `operations`, whose operands must all be below 64 (the
// dense engines' ceiling is 62 qubits; checked with common::ensure).
[[nodiscard]] FusionPlan plan_fusion(std::span<const Operation> operations);

}  // namespace menos::ir
