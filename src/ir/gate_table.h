#pragma once

#include <complex>
#include <span>
#include <string_view>
#include <vector>

#include "menos/circuit.h"

namespace menos::ir {

using MatrixGenerator = std::vector<std::complex<double>> (*)(std::span<const double> params);

// opcode, name, arity, parameter count, matrix generator: the single
// source of truth for the 20 named-gate opcodes (docs/BUILD_PLAN.md
// Phase 1, 1.1). Matrices follow Qiskit's standard gate library;
// tests/ir's Qiskit-generated fixtures are the arbiter.
struct GateInfo {
    Opcode opcode;
    std::string_view name;
    int arity;
    int num_params;
    MatrixGenerator generator;
    // true for exactly X Y Z H S Sdg SX SXdg CX CZ SWAP (docs/BUILD_PLAN.md
    // Phase 4). No angle-based detection -- RZ(pi/2) is not recognized as
    // S, even though they agree up to global phase.
    bool is_clifford;
};

// Precondition: opcode is one of the 20 named gates (not Unitary or
// Measure, which have no table entry). Violating this is an internal
// invariant failure, not a user input error -- callers dispatch on
// Opcode before reaching here.
[[nodiscard]] const GateInfo& gate_info(Opcode opcode);

// nullptr if name matches no named gate (Unitary and Measure are not
// named gates and are never looked up this way). The single lookup the
// OpenQASM 2 parser uses to map a parsed gate name to an Opcode
// (docs/BUILD_PLAN.md Phase 7: "the parser does not define its own
// gate-name mapping") -- callers needing a typed error for an unknown
// name (UnsupportedFeature, naming the line) build it themselves from a
// null return, since this file has no notion of "line number".
[[nodiscard]] const GateInfo* gate_info_by_name(std::string_view name);

// gate_info(opcode).generator(params), with params.size() checked against
// gate_info(opcode).num_params.
[[nodiscard]] std::vector<std::complex<double>> generate_matrix(Opcode opcode,
                                                                std::span<const double> params);

}  // namespace menos::ir
