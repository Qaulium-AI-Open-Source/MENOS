#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

#include "common/error.h"
#include "ir/fuse_gates.h"
#include "ir/gate_table.h"
#include "menos/circuit.h"
#include "menos/simulation_engine.h"

// The sweep over a fusion plan's blocks (Phase 24 item 7, docs/adr/0028
// section 17), shared by the three dense engines: the CPU engine, the CUDA
// engine and the disk tier's chunked executor each apply a block's matrix
// with their own kernel, and the matrix itself is built here, once, by the
// CPU generic kernel, so it is the same bytes on every backend.
namespace menos::engines::dense {

// The matrix of block `block` of `plan`: the product of its gates in block
// order, on the block's ascending operands (bit j of the row and column
// index is block_operands(block)[j]). It is computed as the generic kernel's
// action on each basis vector of the block's 2^k-dimensional space
// (kernels::cpu::apply_matrix_generic, one thread), so the product obeys the
// kernels' own no-contraction arithmetic: the same bytes on Linux, Windows
// and both backends, and a fused run's results stay bitwise identical across
// thread counts, CPU/CUDA and the disk tier wherever the unfused run's were.
// Precondition: the block holds two or more gates.
[[nodiscard]] std::vector<std::complex<double>> fused_block_matrix(
    const ir::FusionPlan& plan, std::size_t block, std::span<const Operation> operations);

// The most heap one block of `plan` holds while it is built and applied, on
// `num_threads` threads, as the estimate charges it (docs/adr/0025 section
// 3.4): for a block of one GateOp its generated matrix beside the kernel's
// scratch, as before fusion; for a fused block the larger of the build (the
// block's matrix, one gate's generated matrix and its local operands) and the
// application (the block's matrix beside the kernel's scratch).
[[nodiscard]] std::uint64_t block_transient_bytes(const ir::FusionPlan& plan,
                                                  std::size_t block,
                                                  std::span<const Operation> operations,
                                                  std::uint32_t num_threads);

// The largest block_transient_bytes over every block of `plan`.
[[nodiscard]] std::uint64_t max_block_transient_bytes(const ir::FusionPlan& plan,
                                                      std::span<const Operation> operations,
                                                      std::uint32_t num_threads);

// Applies every block of `plan`, in order, through `apply(matrix, operands)`:
// a block of one GateOp or UnitaryOp with that gate's own matrix (bitwise the
// unfused sweep's for it), a fused block with fused_block_matrix. A barrier
// block (a MeasureOp) applies nothing. The gap between two blocks is the
// engine's safe boundary (Phase 11, docs/adr/0015): monitor, when set, is
// called after each block with the number of blocks completed. `apply` is a
// plain callable, resolved at compile time: no virtual call sits in the loop.
template <typename Apply>
void apply_blocks(const ir::FusionPlan& plan,
                  std::span<const Operation> operations,
                  const ExecutionMonitor* monitor,
                  Apply&& apply) {
    std::uint64_t completed = 0;
    for (std::size_t b = 0; b < plan.block_count(); ++b) {
        const std::span<const std::uint32_t> members = plan.block_operations(b);
        if (members.size() == 1) {
            // The gate's own operand order, not the block's ascending list:
            // bit j of its matrix is ITS operand j (cx 1,0 is not cx 0,1).
            std::visit(
                [&](const auto& op) {
                    using T = std::decay_t<decltype(op)>;
                    if constexpr (std::is_same_v<T, GateOp>) {
                        const std::vector<std::complex<double>> matrix =
                            ir::generate_matrix(op.opcode, op.params);
                        apply(std::span<const std::complex<double>>(matrix),
                              std::span<const QubitId>(op.qubits));
                    } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                        apply(std::span<const std::complex<double>>(op.matrix),
                              std::span<const QubitId>(op.qubits));
                    } else if constexpr (std::is_same_v<T, ChannelOp>) {
                        // Unreachable: estimate() refuses a noisy region.
                        throw common::InternalError(
                            "dense: a noise channel reached a pure-state engine");
                    } else {
                        static_assert(std::is_same_v<T, MeasureOp>);
                        // Outcomes are resolved by the sampling pass.
                    }
                },
                operations[members[0]]);
        } else {
            const std::vector<std::complex<double>> matrix =
                fused_block_matrix(plan, b, operations);
            apply(std::span<const std::complex<double>>(matrix), plan.block_operands(b));
        }
        ++completed;
        if (monitor != nullptr) {
            monitor->at_safe_boundary(completed);
        }
    }
}

}  // namespace menos::engines::dense
