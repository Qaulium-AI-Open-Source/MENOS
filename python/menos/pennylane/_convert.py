"""PennyLane tape -> menos._core.Circuit / Observable, and MENOS results ->
PennyLane's conventions (docs/adr/0024 section 4).

Ordering: wire ``wire_order[k]`` is MENOS qubit ``k``. MENOS is little-endian
(bit k of an amplitude or matrix index is qubit k); PennyLane is big-endian
(the first wire is the most significant bit), for states, probabilities and
operator matrices alike.
"""

from __future__ import annotations

import numpy as np
from pennylane.operation import Channel

from menos._core import Circuit, Observable, Opcode, PauliString, kMaxUnitaryQubits

# PennyLane name -> MENOS opcode, each equal to PennyLane's matrix entrywise
# within 1e-12, global phase included (docs/adr/0024 section 4, gate table).
# Operand order is unchanged: controls first, as in both libraries.
GATE_MAP: dict[str, Opcode] = {
    "PauliX": Opcode.X,
    "PauliY": Opcode.Y,
    "PauliZ": Opcode.Z,
    "Hadamard": Opcode.H,
    "S": Opcode.S,
    "T": Opcode.T,
    "SX": Opcode.SX,
    "Adjoint(S)": Opcode.Sdg,
    "Adjoint(T)": Opcode.Tdg,
    "Adjoint(SX)": Opcode.SXdg,
    "RX": Opcode.RX,
    "RY": Opcode.RY,
    "RZ": Opcode.RZ,
    "PhaseShift": Opcode.P,
    "U3": Opcode.U,
    "CNOT": Opcode.CX,
    "CZ": Opcode.CZ,
    "SWAP": Opcode.SWAP,
    "ControlledPhaseShift": Opcode.CP,
    "IsingZZ": Opcode.RZZ,
    "Toffoli": Opcode.CCX,
}

# Operations that produce nothing in MENOS (GlobalPhase is folded into
# Circuit.global_phase instead).
NO_OP_NAMES = frozenset({"Identity", "Barrier"})


def is_supported_operation(op) -> bool:
    """The `decompose` stopping condition (docs/adr/0024 section 3.2)."""
    if op.name in GATE_MAP or op.name in NO_OP_NAMES or op.name == "GlobalPhase":
        return True
    if len(op.wires) > kMaxUnitaryQubits:
        return False
    if isinstance(op, Channel):
        return True
    return op.has_matrix


def _qubits(op, index) -> list[int]:
    return [index[w] for w in op.wires]


def _as_complex_list(matrix) -> list[complex]:
    return [complex(x) for x in np.asarray(matrix, dtype=complex).reshape(-1)]


def tape_to_circuit(operations, wire_order, *, measure: bool) -> Circuit:
    """Build a Circuit over ``len(wire_order)`` qubits.

    ``measure`` appends a measurement of qubit k into clbit k for every k.
    """
    index = {w: k for k, w in enumerate(wire_order)}
    n = len(wire_order)
    circuit = Circuit(n, n if measure else 0)
    phase = 0.0
    for op in operations:
        name = op.name
        if name in NO_OP_NAMES:
            continue
        if name == "GlobalPhase":
            # PennyLane's GlobalPhase(phi) multiplies the state by e^{-i phi}.
            phase -= float(op.data[0])
            continue
        if name in GATE_MAP:
            circuit.append(GATE_MAP[name], _qubits(op, index), [float(p) for p in op.data])
            continue
        # Big-endian PennyLane matrix -> MENOS operands in reverse wire order.
        qubits = _qubits(op, index)[::-1]
        if isinstance(op, Channel):
            circuit.append_channel(qubits, [_as_complex_list(k) for k in op.kraus_matrices()])
        else:
            circuit.append_unitary(qubits, _as_complex_list(op.matrix()))
    if measure:
        for k in range(n):
            circuit.append_measure(k, k)
    circuit.global_phase = phase
    return circuit


def pauli_sentence_to_observable(sentence, wire_order) -> tuple[Observable | None, complex]:
    """Split a PauliSentence into a MENOS Observable over the non-identity
    terms and the identity term's coefficient, which is added exactly in
    Python (a MENOS observable of identity terms only is refused by the
    core, docs/reference/limitations.md)."""
    index = {w: k for k, w in enumerate(wire_order)}
    n = len(wire_order)
    words = (n + 63) // 64
    identity = 0j
    terms = []
    for word, coeff in sentence.items():
        if len(word) == 0:
            identity += complex(coeff)
            continue
        x = [0] * words
        z = [0] * words
        for wire, letter in word.items():
            k = index[wire]
            bit = 1 << (k % 64)
            if letter in ("X", "Y"):
                x[k // 64] |= bit
            if letter in ("Z", "Y"):
                z[k // 64] |= bit
        pauli = PauliString()
        pauli.num_qubits = n
        # nanobind copies these lists: assign them whole.
        pauli.x_words = x
        pauli.z_words = z
        terms.append((pauli, complex(coeff)))
    if not terms:
        return None, identity
    observable = Observable()
    observable.terms = terms
    return observable, identity


def statevector_to_big_endian(state) -> np.ndarray:
    """MENOS's little-endian amplitudes in PennyLane's wire order."""
    state = np.asarray(state, dtype=complex)
    n = int(round(np.log2(state.size)))
    if n == 0:
        return state
    return state.reshape([2] * n).transpose(range(n - 1, -1, -1)).reshape(-1)


def counts_to_samples(counts: dict[str, int], n: int, rng: np.random.Generator) -> np.ndarray:
    """Expand MENOS's hex counts (bit k = clbit k = wire k) into PennyLane's
    per-shot rows (column k = wire k), in a random order drawn from ``rng``
    (docs/adr/0024 section 9, Q8)."""
    values = []
    for key, count in counts.items():
        values.extend([int(key, 16)] * count)
    values = np.asarray(values, dtype=np.int64)
    rng.shuffle(values)
    columns = np.arange(n, dtype=np.int64)
    return ((values[:, None] >> columns[None, :]) & 1).astype(np.int64)


def is_noisy(operations) -> bool:
    return any(isinstance(op, Channel) for op in operations)

