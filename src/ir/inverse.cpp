#include "ir/inverse.h"

#include <complex>
#include <cstddef>

#include "common/error.h"

namespace menos::ir {

Opcode inverse_opcode(Opcode opcode) {
    switch (opcode) {
        case Opcode::S:
            return Opcode::Sdg;
        case Opcode::Sdg:
            return Opcode::S;
        case Opcode::T:
            return Opcode::Tdg;
        case Opcode::Tdg:
            return Opcode::T;
        case Opcode::SX:
            return Opcode::SXdg;
        case Opcode::SXdg:
            return Opcode::SX;
        case Opcode::X:
        case Opcode::Y:
        case Opcode::Z:
        case Opcode::H:
        case Opcode::CX:
        case Opcode::CZ:
        case Opcode::SWAP:
        case Opcode::CCX:
        case Opcode::RX:
        case Opcode::RY:
        case Opcode::RZ:
        case Opcode::P:
        case Opcode::U:
        case Opcode::CP:
        case Opcode::RZZ:
            return opcode;
        case Opcode::Unitary:
        case Opcode::Measure:
            break;
    }
    common::ensure(false, "inverse: a named-gate inverse was asked of Unitary or Measure");
    return opcode;
}

GateOp inverse_of(const GateOp& op) {
    auto [opcode, params] = inverse_gate(op.opcode, op.params);
    return GateOp{opcode, op.qubits, std::move(params)};
}

UnitaryOp inverse_of(const UnitaryOp& op) {
    std::size_t dim = 1;
    while (dim * dim < op.matrix.size()) {
        dim *= 2;
    }
    common::ensure(dim * dim == op.matrix.size(), "inverse: a unitary matrix is not square");
    UnitaryOp out{op.qubits, std::vector<std::complex<double>>(op.matrix.size())};
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col < dim; ++col) {
            out.matrix[col * dim + row] = std::conj(op.matrix[row * dim + col]);
        }
    }
    return out;
}

}  // namespace menos::ir
