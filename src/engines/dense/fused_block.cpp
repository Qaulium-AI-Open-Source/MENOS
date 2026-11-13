#include "engines/dense/fused_block.h"

#include <algorithm>
#include <utility>

#include "common/checked_math.h"
#include "common/memory_bounds.h"
#include "engines/dense/apply_matrix.h"
#include "kernels/dense_kernels.h"

namespace menos::engines::dense {
namespace {

using Complex = std::complex<double>;

// A gate's operands and matrix, the matrix generated for a GateOp and
// borrowed from a UnitaryOp.
struct GateView {
    const std::vector<QubitId>* qubits;
    std::vector<Complex> generated;
    std::span<const Complex> matrix;
};

GateView view_of(const Operation& operation) {
    return std::visit(
        [](const auto& op) -> GateView {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                GateView view{&op.qubits, ir::generate_matrix(op.opcode, op.params), {}};
                view.matrix = view.generated;
                return view;
            } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                return GateView{&op.qubits, {}, op.matrix};
            } else {
                throw common::InternalError("fused_block_matrix: a block member is not a gate");
            }
        },
        operation);
}

std::size_t operand_count(const Operation& operation) {
    return std::visit(
        [](const auto& op) -> std::size_t {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                return op.qubits.size();
            } else {
                return 0;
            }
        },
        operation);
}

std::uint64_t matrix_bytes(std::size_t k) {
    return common::heap_block_bytes(
        common::checked_mul(common::checked_pow2(static_cast<unsigned>(2 * k)), sizeof(Complex)));
}

}  // namespace

std::vector<Complex> fused_block_matrix(const ir::FusionPlan& plan,
                                        std::size_t block,
                                        std::span<const Operation> operations) {
    const std::span<const QubitId> block_qubits = plan.block_operands(block);
    const std::size_t k = block_qubits.size();
    const std::size_t dim = std::size_t{1} << k;
    // Column-major while it is built: column c is the generic kernel's state
    // vector, starting at the basis vector e_c, with every gate of the block
    // applied to it in block order.
    std::vector<Complex> columns(dim * dim, Complex{0.0, 0.0});
    for (std::size_t c = 0; c < dim; ++c) {
        columns[c * dim + c] = Complex{1.0, 0.0};
    }
    std::vector<std::uint32_t> local;
    for (const std::uint32_t index : plan.block_operations(block)) {
        const GateView gate = view_of(operations[index]);
        local.clear();
        for (const QubitId q : *gate.qubits) {
            const auto position = std::find(block_qubits.begin(), block_qubits.end(), q);
            common::ensure(position != block_qubits.end(),
                           "fused_block_matrix: an operand outside its block");
            local.push_back(static_cast<std::uint32_t>(position - block_qubits.begin()));
        }
        for (std::size_t c = 0; c < dim; ++c) {
            kernels::cpu::apply_matrix_generic(std::span<Complex>(columns).subspan(c * dim, dim),
                                               gate.matrix,
                                               local,
                                               /*num_threads=*/1);
        }
    }
    // Row-major, in place: entry (row, column) of the product is column
    // `column`'s amplitude `row`.
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = r + 1; c < dim; ++c) {
            std::swap(columns[r * dim + c], columns[c * dim + r]);
        }
    }
    return columns;
}

std::uint64_t block_transient_bytes(const ir::FusionPlan& plan,
                                    std::size_t block,
                                    std::span<const Operation> operations,
                                    std::uint32_t num_threads) {
    using common::checked_add;
    using common::checked_mul;
    using common::heap_block_bytes;
    const std::span<const std::uint32_t> members = plan.block_operations(block);
    const std::size_t k = plan.block_operands(block).size();
    if (members.size() == 1) {
        // As before fusion: a GateOp's generated matrix beside the kernel's
        // scratch; a UnitaryOp's matrix is borrowed.
        std::uint64_t bytes = apply_matrix_scratch_bytes(k, num_threads);
        if (std::holds_alternative<GateOp>(operations[members[0]])) {
            bytes = checked_add(bytes, matrix_bytes(k));
        }
        return bytes;
    }
    // The build: the block's matrix, the largest member's generated matrix
    // (a GateOp's; a UnitaryOp's is borrowed) and the local operand list.
    std::uint64_t widest_generated = 0;
    std::size_t widest = 0;
    for (const std::uint32_t index : members) {
        const std::size_t gate_k = operand_count(operations[index]);
        widest = std::max(widest, gate_k);
        if (std::holds_alternative<GateOp>(operations[index])) {
            widest_generated = std::max(widest_generated, matrix_bytes(gate_k));
        }
    }
    const std::uint64_t build =
        checked_add(checked_add(matrix_bytes(k), widest_generated),
                    heap_block_bytes(checked_mul(widest, sizeof(std::uint32_t))));
    // The application: the block's matrix beside the kernel's scratch.
    const std::uint64_t apply =
        checked_add(matrix_bytes(k), apply_matrix_scratch_bytes(k, num_threads));
    return std::max(build, apply);
}

std::uint64_t max_block_transient_bytes(const ir::FusionPlan& plan,
                                        std::span<const Operation> operations,
                                        std::uint32_t num_threads) {
    std::uint64_t bytes = 0;
    for (std::size_t b = 0; b < plan.block_count(); ++b) {
        bytes = std::max(bytes, block_transient_bytes(plan, b, operations, num_threads));
    }
    return bytes;
}

}  // namespace menos::engines::dense
