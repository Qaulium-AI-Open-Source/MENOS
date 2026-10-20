#include "qasm3/modifiers.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include "common/error.h"
#include "ir/gate_table.h"

namespace menos::qasm3::detail {
namespace {

using Complex = std::complex<double>;

std::size_t dimension_of(std::size_t matrix_size) {
    std::size_t dim = 1;
    while (dim * dim < matrix_size) {
        dim *= 2;
    }
    common::ensure(dim * dim == matrix_size, "qasm3: a unitary matrix is not square");
    return dim;
}

GateOp inverse_of(const GateOp& op) {
    GateOp out = op;
    switch (op.opcode) {
        case Opcode::X:
        case Opcode::Y:
        case Opcode::Z:
        case Opcode::H:
        case Opcode::CX:
        case Opcode::CZ:
        case Opcode::SWAP:
        case Opcode::CCX:
            return out;
        case Opcode::S:
            out.opcode = Opcode::Sdg;
            return out;
        case Opcode::Sdg:
            out.opcode = Opcode::S;
            return out;
        case Opcode::T:
            out.opcode = Opcode::Tdg;
            return out;
        case Opcode::Tdg:
            out.opcode = Opcode::T;
            return out;
        case Opcode::SX:
            out.opcode = Opcode::SXdg;
            return out;
        case Opcode::SXdg:
            out.opcode = Opcode::SX;
            return out;
        case Opcode::RX:
        case Opcode::RY:
        case Opcode::RZ:
        case Opcode::P:
        case Opcode::CP:
        case Opcode::RZZ:
            out.params[0] = -op.params[0];
            return out;
        case Opcode::U:
            // U(t, p, l)^dagger = U(-t, -l, -p).
            out.params = {-op.params[0], -op.params[2], -op.params[1]};
            return out;
        case Opcode::Unitary:
        case Opcode::Measure:
            break;
    }
    common::ensure(false, "qasm3: a GateOp carries the Unitary or Measure opcode");
    return out;
}

UnitaryOp inverse_of(const UnitaryOp& op) {
    const std::size_t dim = dimension_of(op.matrix.size());
    UnitaryOp out{op.qubits, std::vector<Complex>(op.matrix.size())};
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col < dim; ++col) {
            out.matrix[col * dim + row] = std::conj(op.matrix[row * dim + col]);
        }
    }
    return out;
}

UnitaryOp controlled_unitary(QubitId control_qubit,
                             const std::vector<QubitId>& qubits,
                             std::span<const Complex> matrix,
                             std::string_view where) {
    if (qubits.size() + 1 > static_cast<std::size_t>(kMaxUnitaryQubits)) {
        throw common::UnsupportedFeature(
            "qasm3: a controlled gate on " + std::to_string(qubits.size() + 1) +
            " qubits has no gate-table form and exceeds the " + std::to_string(kMaxUnitaryQubits) +
            "-qubit limit of an explicit unitary" + std::string(where));
    }
    std::vector<QubitId> operands{control_qubit};
    operands.insert(operands.end(), qubits.begin(), qubits.end());
    return UnitaryOp{std::move(operands), controlled_matrix(matrix)};
}

}  // namespace

std::vector<Complex> controlled_matrix(std::span<const Complex> m) {
    const std::size_t dim = dimension_of(m.size());
    const std::size_t out_dim = 2 * dim;
    std::vector<Complex> out(out_dim * out_dim, Complex{0.0, 0.0});
    for (std::size_t row = 0; row < out_dim; ++row) {
        for (std::size_t col = 0; col < out_dim; ++col) {
            const std::size_t row_control = row & 1U;
            const std::size_t col_control = col & 1U;
            if (row_control != col_control) {
                continue;
            }
            if (row_control == 0) {
                out[row * out_dim + col] = row == col ? Complex{1.0, 0.0} : Complex{0.0, 0.0};
            } else {
                out[row * out_dim + col] = m[(row >> 1U) * dim + (col >> 1U)];
            }
        }
    }
    return out;
}

std::vector<Instr> invert(std::vector<Instr> body) {
    std::reverse(body.begin(), body.end());
    for (Instr& step : body) {
        std::visit(
            [](auto& s) {
                using T = std::decay_t<decltype(s)>;
                if constexpr (std::is_same_v<T, GlobalPhase>) {
                    s.angle = -s.angle;
                } else {
                    s = inverse_of(s);
                }
            },
            step);
    }
    return body;
}

std::vector<Instr> power(const std::vector<Instr>& body, std::int64_t k) {
    const std::vector<Instr> unit = k < 0 ? invert(body) : body;
    const std::int64_t count = k < 0 ? -k : k;
    std::vector<Instr> out;
    for (std::int64_t i = 0; i < count; ++i) {
        out.insert(out.end(), unit.begin(), unit.end());
    }
    return out;
}

std::vector<Instr> control(const std::vector<Instr>& body,
                           QubitId control_qubit,
                           std::string_view where) {
    std::vector<Instr> out;
    out.reserve(body.size());
    for (const Instr& step : body) {
        if (const auto* phase = std::get_if<GlobalPhase>(&step)) {
            out.emplace_back(GateOp{Opcode::P, {control_qubit}, {phase->angle}});
        } else if (const auto* unitary = std::get_if<UnitaryOp>(&step)) {
            out.emplace_back(
                controlled_unitary(control_qubit, unitary->qubits, unitary->matrix, where));
        } else {
            const auto& gate = std::get<GateOp>(step);
            std::vector<QubitId> qubits{control_qubit};
            qubits.insert(qubits.end(), gate.qubits.begin(), gate.qubits.end());
            switch (gate.opcode) {
                case Opcode::X:
                    out.emplace_back(GateOp{Opcode::CX, std::move(qubits), {}});
                    break;
                case Opcode::CX:
                    out.emplace_back(GateOp{Opcode::CCX, std::move(qubits), {}});
                    break;
                case Opcode::Z:
                    out.emplace_back(GateOp{Opcode::CZ, std::move(qubits), {}});
                    break;
                case Opcode::P:
                    out.emplace_back(GateOp{Opcode::CP, std::move(qubits), gate.params});
                    break;
                default: {
                    const std::vector<Complex> matrix =
                        ir::generate_matrix(gate.opcode, gate.params);
                    out.emplace_back(controlled_unitary(control_qubit, gate.qubits, matrix, where));
                    break;
                }
            }
        }
    }
    return out;
}

std::vector<Instr> negative_control(const std::vector<Instr>& body,
                                    QubitId control_qubit,
                                    std::string_view where) {
    std::vector<Instr> out;
    out.emplace_back(GateOp{Opcode::X, {control_qubit}, {}});
    std::vector<Instr> controlled = control(body, control_qubit, where);
    out.insert(out.end(),
               std::make_move_iterator(controlled.begin()),
               std::make_move_iterator(controlled.end()));
    out.emplace_back(GateOp{Opcode::X, {control_qubit}, {}});
    return out;
}

}  // namespace menos::qasm3::detail
