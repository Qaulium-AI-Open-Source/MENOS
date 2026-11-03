"""The single source of truth for which named gates MENOS supports through
Qiskit, and how they map to ``menos._core.Opcode``. Both ``target.py``
(what MENOSBackend advertises) and ``convert.py`` (what it actually
executes) read this table, so the two can never drift apart
(docs/BUILD_PLAN.md Phase 2, 2.4: "generated from the same
supported-instruction table used by conversion").
"""

from __future__ import annotations

from dataclasses import dataclass

from menos._core import Opcode


@dataclass(frozen=True)
class GateSpec:
    opcode: Opcode
    arity: int
    num_params: int


# Verified in this session (docs/adr/0004) against
# qiskit.circuit.library.standard_gates.get_standard_gate_name_mapping():
# every name below exists there with a matching arity and parameter count.
GATE_TABLE: dict[str, GateSpec] = {
    "x": GateSpec(Opcode.X, 1, 0),
    "y": GateSpec(Opcode.Y, 1, 0),
    "z": GateSpec(Opcode.Z, 1, 0),
    "h": GateSpec(Opcode.H, 1, 0),
    "s": GateSpec(Opcode.S, 1, 0),
    "sdg": GateSpec(Opcode.Sdg, 1, 0),
    "t": GateSpec(Opcode.T, 1, 0),
    "tdg": GateSpec(Opcode.Tdg, 1, 0),
    "sx": GateSpec(Opcode.SX, 1, 0),
    "sxdg": GateSpec(Opcode.SXdg, 1, 0),
    "rx": GateSpec(Opcode.RX, 1, 1),
    "ry": GateSpec(Opcode.RY, 1, 1),
    "rz": GateSpec(Opcode.RZ, 1, 1),
    "p": GateSpec(Opcode.P, 1, 1),
    "u": GateSpec(Opcode.U, 1, 3),
    "cx": GateSpec(Opcode.CX, 2, 0),
    "cz": GateSpec(Opcode.CZ, 2, 0),
    "swap": GateSpec(Opcode.SWAP, 2, 0),
    "cp": GateSpec(Opcode.CP, 2, 1),
    "rzz": GateSpec(Opcode.RZZ, 2, 1),
    "ccx": GateSpec(Opcode.CCX, 3, 0),
}

# Dropped during conversion: KIR has no representation for them and their
# absence does not affect simulation (CLAUDE.md: "Barriers, id and delay
# are dropped by adapters and do not exist in KIR").
DROPPED_INSTRUCTION_NAMES = frozenset({"barrier", "id", "delay"})
