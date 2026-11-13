#pragma once

#include <utility>
#include <vector>

#include "menos/circuit.h"

namespace menos::ir {

// The exact inverse of a named gate, as the opcode of the inverse and its
// parameters: self-inverse gates (X Y Z H CX CZ SWAP CCX) unchanged;
// S <-> Sdg, T <-> Tdg, SX <-> SXdg; the angle negated for RX RY RZ P CP RZZ;
// U(t, p, l)^dagger = U(-t, -l, -p). Negation and reordering are exact, so
// the inverse is exact. P is double, or any type with unary minus (the
// circuit builder's parameter expressions, docs/adr/0029 section 3.4).
// opcode must be a named gate (not Unitary or Measure): InternalError
// otherwise, through common::ensure.
template <class P>
[[nodiscard]] std::pair<Opcode, std::vector<P>> inverse_gate(Opcode opcode, std::vector<P> params);

// The exact inverse of op, by inverse_gate.
[[nodiscard]] GateOp inverse_of(const GateOp& op);

// The exact inverse of op: its conjugate transpose.
[[nodiscard]] UnitaryOp inverse_of(const UnitaryOp& op);

// The opcode of a named gate's inverse; InternalError for Unitary/Measure.
[[nodiscard]] Opcode inverse_opcode(Opcode opcode);

template <class P>
std::pair<Opcode, std::vector<P>> inverse_gate(Opcode opcode, std::vector<P> params) {
    const Opcode inverse = inverse_opcode(opcode);
    switch (opcode) {
        case Opcode::RX:
        case Opcode::RY:
        case Opcode::RZ:
        case Opcode::P:
        case Opcode::CP:
        case Opcode::RZZ:
            params[0] = -params[0];
            break;
        case Opcode::U: {
            // U(t, p, l)^dagger = U(-t, -l, -p).
            std::vector<P> out;
            out.reserve(3);
            out.push_back(-params[0]);
            out.push_back(-params[2]);
            out.push_back(-params[1]);
            params = std::move(out);
            break;
        }
        default:
            break;
    }
    return {inverse, std::move(params)};
}

}  // namespace menos::ir
