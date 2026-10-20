#include "ir/fuse_diagonal.h"

#include <algorithm>
#include <complex>
#include <type_traits>
#include <variant>

#include "common/error.h"
#include "ir/gate_table.h"

namespace menos::ir {
namespace {

using Complex = std::complex<double>;

bool diagonal_opcode(Opcode opcode) {
    switch (opcode) {
        case Opcode::Z:
        case Opcode::S:
        case Opcode::Sdg:
        case Opcode::T:
        case Opcode::Tdg:
        case Opcode::RZ:
        case Opcode::P:
        case Opcode::CZ:
        case Opcode::CP:
        case Opcode::RZZ:
            return true;
        default:
            return false;
    }
}

bool matrix_is_diagonal(const std::vector<Complex>& matrix, std::size_t dim) {
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            if (r != c &&
                (matrix[r * dim + c].real() != 0.0 || matrix[r * dim + c].imag() != 0.0)) {
                return false;
            }
        }
    }
    return true;
}

// The gate's operands and its diagonal entries, indexed by the gate's own
// operand-order bits (operand j is local bit j, as everywhere in KIR).
struct DiagonalGate {
    const std::vector<QubitId>* qubits;
    std::vector<Complex> diagonal;
};

DiagonalGate diagonal_of(const Operation& op) {
    return std::visit(
        [](const auto& o) -> DiagonalGate {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                const std::vector<Complex> m = generate_matrix(o.opcode, o.params);
                const std::size_t dim = std::size_t{1} << o.qubits.size();
                std::vector<Complex> d(dim);
                for (std::size_t i = 0; i < dim; ++i) {
                    d[i] = m[i * dim + i];
                }
                return DiagonalGate{&o.qubits, std::move(d)};
            } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                const std::size_t dim = std::size_t{1} << o.qubits.size();
                std::vector<Complex> d(dim);
                for (std::size_t i = 0; i < dim; ++i) {
                    d[i] = o.matrix[i * dim + i];
                }
                return DiagonalGate{&o.qubits, std::move(d)};
            } else {
                throw common::InternalError("fuse_diagonal_gates: not a diagonal gate");
            }
        },
        op);
}

std::size_t distinct_qubits_with(const std::vector<QubitId>& block,
                                 const std::vector<QubitId>& more) {
    std::size_t extra = 0;
    for (const QubitId q : more) {
        if (std::find(block.begin(), block.end(), q) == block.end()) {
            ++extra;
        }
    }
    return block.size() + extra;
}

// One UnitaryOp for a block of diagonal gates on the ascending qubit list
// `qubits`: entry [local] is the product, in block order, of each gate's
// diagonal entry at the sub-index its operands select from `local`.
UnitaryOp fuse_block(const std::vector<QubitId>& qubits, std::span<const Operation* const> gates) {
    const std::size_t k = qubits.size();
    const std::size_t dim = std::size_t{1} << k;
    std::vector<Complex> matrix(dim * dim, Complex{0.0, 0.0});
    std::vector<Complex> diagonal(dim, Complex{1.0, 0.0});
    for (const Operation* op : gates) {
        const DiagonalGate gate = diagonal_of(*op);
        // position_in_block[j] = which bit of `local` holds operand j.
        std::vector<std::size_t> position_in_block(gate.qubits->size());
        for (std::size_t j = 0; j < gate.qubits->size(); ++j) {
            const auto it = std::find(qubits.begin(), qubits.end(), (*gate.qubits)[j]);
            common::ensure(it != qubits.end(), "fuse_diagonal_gates: operand outside its block");
            position_in_block[j] = static_cast<std::size_t>(it - qubits.begin());
        }
        for (std::size_t local = 0; local < dim; ++local) {
            std::size_t sub = 0;
            for (std::size_t j = 0; j < position_in_block.size(); ++j) {
                sub |= ((local >> position_in_block[j]) & std::size_t{1}) << j;
            }
            diagonal[local] *= gate.diagonal[sub];
        }
    }
    for (std::size_t i = 0; i < dim; ++i) {
        matrix[i * dim + i] = diagonal[i];
    }
    return UnitaryOp{qubits, std::move(matrix)};
}

// Groups one run of diagonal operations [begin, end) and appends the result.
void fuse_run(std::span<const Operation> run,
              std::uint32_t max_qubits,
              std::vector<Operation>& out) {
    std::vector<bool> assigned(run.size(), false);
    for (std::size_t first = 0; first < run.size(); ++first) {
        if (assigned[first]) {
            continue;
        }
        std::vector<QubitId> block_qubits;
        std::vector<const Operation*> block;
        const auto try_add = [&](std::size_t i) {
            const std::vector<QubitId>* qp = std::visit(
                [](const auto& o) -> const std::vector<QubitId>* {
                    using T = std::decay_t<decltype(o)>;
                    if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                        return &o.qubits;
                    } else {
                        throw common::InternalError("fuse_diagonal_gates: not a gate");
                    }
                },
                run[i]);
            const std::vector<QubitId>& q = *qp;
            if (distinct_qubits_with(block_qubits, q) > max_qubits) {
                return;
            }
            for (const QubitId x : q) {
                if (std::find(block_qubits.begin(), block_qubits.end(), x) == block_qubits.end()) {
                    block_qubits.push_back(x);
                }
            }
            block.push_back(&run[i]);
            assigned[i] = true;
        };
        try_add(first);
        if (block.empty()) {  // wider than max_qubits on its own: left as it is
            assigned[first] = true;
            out.push_back(run[first]);
            continue;
        }
        for (std::size_t i = first + 1; i < run.size(); ++i) {
            if (!assigned[i]) {
                try_add(i);
            }
        }
        if (block.size() == 1) {
            out.push_back(*block[0]);
            continue;
        }
        std::sort(block_qubits.begin(), block_qubits.end());
        out.emplace_back(fuse_block(block_qubits, block));
    }
}

}  // namespace

bool is_diagonal_operation(const Operation& op) {
    return std::visit(
        [](const auto& o) -> bool {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                return diagonal_opcode(o.opcode);
            } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                return matrix_is_diagonal(o.matrix, std::size_t{1} << o.qubits.size());
            } else {
                return false;
            }
        },
        op);
}

std::vector<Operation> fuse_diagonal_gates(std::span<const Operation> operations,
                                           std::uint32_t max_qubits) {
    common::ensure(max_qubits >= 2 && max_qubits <= static_cast<std::uint32_t>(kMaxUnitaryQubits),
                   "fuse_diagonal_gates: max_qubits must be in [2, kMaxUnitaryQubits]");
    std::vector<Operation> out;
    out.reserve(operations.size());
    std::size_t i = 0;
    while (i < operations.size()) {
        if (!is_diagonal_operation(operations[i])) {
            out.push_back(operations[i]);
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < operations.size() && is_diagonal_operation(operations[j])) {
            ++j;
        }
        fuse_run(operations.subspan(i, j - i), max_qubits, out);
        i = j;
    }
    return out;
}

}  // namespace menos::ir
