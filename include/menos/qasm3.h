#pragma once

#include <functional>
#include <map>
#include <string>
#include <string_view>

#include "menos/builder.h"
#include "menos/circuit.h"

namespace menos::qasm3 {

// Values for a program's `input` declarations, by name.
using InputBindings = std::map<std::string, double, std::less<>>;

// Parses an OpenQASM 3 program into a Circuit (docs/adr/0023).
//
// Accepted: an optional `OPENQASM 3;`/`OPENQASM 3.0;` header; `include
// "stdgates.inc";`; `qubit`/`bit` declarations (sized or not) and the older
// `qreg`/`creg`, several registers flattened in declaration order; the
// built-in gates `U` (the gate table's `u`) and `gphase`; the 32
// `stdgates.inc` names, each with Qiskit's matrix, global phase included
// (docs/adr/0023 section 4.1 explains why the file's own bodies are not used);
// user `gate` definitions; the modifiers `inv @`, `pow(k) @` with an integer k,
// `ctrl @`/`ctrl(n) @` and `negctrl @`/`negctrl(n) @`, applied exactly (a
// controlled gate with no gate-table form becomes a UnitaryOp of at most
// kMaxUnitaryQubits qubits); `const` and `input` numbers; constant
// expressions with + - * / ** and sin cos tan arcsin arccos arctan exp ln
// sqrt; `measure` into classical bits; and `barrier`, which produces no
// operation. Operations keep source order, measurements included.
//
// Every `input` must have a binding in `inputs` and every binding must name an
// `input`; a missing, unknown or non-finite binding is InvalidCircuit.
//
// Throws menos::common::UnsupportedFeature, naming the construct, line and
// column, for anything outside that subset (reset, control flow, subroutines,
// timing, calibration grammar, classical variables, register broadcast, ...).
// Throws menos::common::InvalidCircuit, with the same location, for a
// malformed program. Does not call menos::ir::validate(); the caller does.
[[nodiscard]] Circuit parse(std::string_view text, const InputBindings& inputs = {});

// Writes builder as an OpenQASM 3 program that parse() reads back into the
// identical operations and global phase (docs/adr/0029 section 3.6), given
// the same values for its inputs: `OPENQASM 3.0;`, `include "stdgates.inc";`,
// an `input float[64]` per parameter, `qubit[n]`/`bit[m]` per register in
// declaration order, `gphase(x);` for a non-zero global phase, then each
// operation by the importer's own name (`U` for u, `inv @ sx` for sxdg, and
// rzz through the definition `gate rzz(theta) a, b { cx a, b; rz(theta) b;
// cx a, b; }`, which parse() reads as one rzz). Numbers are written in their
// shortest round-trip form and expressions in their evaluation order, so the
// values are bitwise the same. A global phase of -0.0 is written as none.
//
// Throws menos::common::UnsupportedFeature, naming the operation's index,
// for an explicit unitary, a noise channel or a measurement with a readout
// error (OpenQASM 3 has no form for them), and naming the register or
// parameter whose name is an OpenQASM 3 reserved word or standard gate name
// or is shared by two of them. Throws menos::common::InvalidCircuit for a
// non-finite angle or global phase.
[[nodiscard]] std::string emit(const CircuitBuilder& builder);
// emit(CircuitBuilder::from_circuit(circuit)): registers q and c.
[[nodiscard]] std::string emit(const Circuit& circuit);

}  // namespace menos::qasm3
