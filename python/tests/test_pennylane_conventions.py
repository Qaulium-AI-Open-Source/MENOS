"""docs/adr/0024 section 4: one test per conventions row, plus the gate
table and the UnitaryOp path (gate criteria 1 and 2)."""

from __future__ import annotations

import pytest

qml = pytest.importorskip("pennylane")

import numpy as np  # noqa: E402

from _pennylane_fixtures import TOL  # noqa: E402
from menos.pennylane import MENOSQubitDevice  # noqa: E402
from menos.pennylane._convert import GATE_MAP  # noqa: E402

_RNG = np.random.default_rng(20261016)

# One instance per GATE_MAP name, with random parameters.
_GATES = {
    "PauliX": lambda w: qml.PauliX(w[0]),
    "PauliY": lambda w: qml.PauliY(w[0]),
    "PauliZ": lambda w: qml.PauliZ(w[0]),
    "Hadamard": lambda w: qml.Hadamard(w[0]),
    "S": lambda w: qml.S(w[0]),
    "T": lambda w: qml.T(w[0]),
    "SX": lambda w: qml.SX(w[0]),
    "Adjoint(S)": lambda w: qml.adjoint(qml.S(w[0])),
    "Adjoint(T)": lambda w: qml.adjoint(qml.T(w[0])),
    "Adjoint(SX)": lambda w: qml.adjoint(qml.SX(w[0])),
    "RX": lambda w: qml.RX(0.731, w[0]),
    "RY": lambda w: qml.RY(-1.234, w[0]),
    "RZ": lambda w: qml.RZ(2.468, w[0]),
    "PhaseShift": lambda w: qml.PhaseShift(-0.377, w[0]),
    "U3": lambda w: qml.U3(0.37, -1.21, 2.05, wires=w[0]),
    "CNOT": lambda w: qml.CNOT(w[:2]),
    "CZ": lambda w: qml.CZ(w[:2]),
    "SWAP": lambda w: qml.SWAP(w[:2]),
    "ControlledPhaseShift": lambda w: qml.ControlledPhaseShift(0.83, wires=w[:2]),
    "IsingZZ": lambda w: qml.IsingZZ(1.37, wires=w[:2]),
    "Toffoli": lambda w: qml.Toffoli(w[:3]),
}


def _unitary_from_device(op) -> np.ndarray:
    """The matrix MENOS applies for ``op``, column by column: each basis
    state is prepared with X gates and evolved through the device."""
    wires = list(op.wires)
    k = len(wires)
    dev = MENOSQubitDevice(wires=wires)
    columns = []
    for b in range(2**k):
        bits = [(b >> (k - 1 - j)) & 1 for j in range(k)]  # big-endian
        prep = [qml.PauliX(w) for w, bit in zip(wires, bits) if bit]
        tape = qml.tape.QuantumScript([*prep, op], [qml.state()])
        columns.append(qml.execute([tape], dev)[0])
    return np.stack(columns, axis=1)


def test_gate_map_covers_every_name():
    assert set(_GATES) == set(GATE_MAP)


@pytest.mark.parametrize("name", sorted(GATE_MAP))
def test_gate_table_matches_pennylane_matrix(name):
    op = _GATES[name](["w0", "w1", "w2"])
    assert op.name == name
    np.testing.assert_allclose(_unitary_from_device(op), op.matrix(), rtol=0, atol=TOL)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_unitary_path_matches_pennylane_matrix(k):
    z = _RNG.normal(size=(2**k, 2**k)) + 1j * _RNG.normal(size=(2**k, 2**k))
    q, r = np.linalg.qr(z)
    u = q * (np.diagonal(r) / np.abs(np.diagonal(r)))
    op = qml.QubitUnitary(u, wires=[f"q{j}" for j in range(k)][::-1])
    np.testing.assert_allclose(_unitary_from_device(op), op.matrix(), rtol=0, atol=TOL)


def test_unitary_path_for_unmapped_named_gate():
    op = qml.CRX(0.61, wires=["c", "t"])
    np.testing.assert_allclose(_unitary_from_device(op), op.matrix(), rtol=0, atol=TOL)


def test_global_phase_sign():
    # PennyLane's GlobalPhase(phi) multiplies the state by e^{-i phi}.
    dev = MENOSQubitDevice(wires=1)
    tape = qml.tape.QuantumScript([qml.GlobalPhase(0.4)], [qml.state()])
    np.testing.assert_allclose(
        qml.execute([tape], dev)[0], [np.exp(-0.4j), 0], rtol=0, atol=TOL
    )


def test_wire_to_qubit_with_device_wires():
    # Device wires given: the state covers every device wire, in device order.
    dev = MENOSQubitDevice(wires=["b", 3, "a"])
    tape = qml.tape.QuantumScript([qml.PauliX("a")], [qml.state()])
    state = qml.execute([tape], dev)[0]
    assert state.shape == (8,)
    assert abs(state[0b001] - 1) < TOL  # "a" is the last, least significant wire


def test_wire_to_qubit_without_device_wires():
    # Device wires None: the tape's own wire order.
    dev = MENOSQubitDevice()
    tape = qml.tape.QuantumScript([qml.PauliX(5), qml.Hadamard(2)], [qml.state()])
    ref = qml.device("default.qubit")
    np.testing.assert_allclose(
        qml.execute([tape], dev)[0], qml.execute([tape], ref)[0], rtol=0, atol=TOL
    )


def test_state_is_big_endian():
    dev = MENOSQubitDevice(wires=3)
    tape = qml.tape.QuantumScript([qml.PauliX(0)], [qml.state()])
    state = qml.execute([tape], dev)[0]
    assert abs(state[0b100] - 1) < TOL


def test_probs_follow_the_listed_wire_order():
    dev = MENOSQubitDevice(wires=3)
    ops = [qml.PauliX(2), qml.RY(0.9, 0)]
    for wires in ([2, 0], [0, 2], [1, 2, 0]):
        tape = qml.tape.QuantumScript(ops, [qml.probs(wires=wires)])
        ref = qml.device("default.qubit", wires=3)
        np.testing.assert_allclose(
            qml.execute([tape], dev)[0], qml.execute([tape], ref)[0], rtol=0, atol=TOL
        )


def test_counts_and_samples_put_the_first_wire_leftmost():
    dev = MENOSQubitDevice(wires=["x", "y"], seed=1)
    ops = [qml.PauliX("x")]
    tape = qml.tape.QuantumScript(ops, [qml.counts(), qml.sample()], shots=10)
    counts, samples = qml.execute([tape], dev)[0]
    assert dict(counts) == {"10": 10}
    assert np.array_equal(samples, np.tile([1, 0], (10, 1)))


def test_identity_and_barrier_are_no_operations():
    dev = MENOSQubitDevice(wires=2)
    ops = [qml.Identity(0), qml.Barrier(wires=[0, 1]), qml.Hadamard(1)]
    tape = qml.tape.QuantumScript(ops, [qml.state()])
    np.testing.assert_allclose(
        qml.execute([tape], dev)[0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], rtol=0, atol=TOL
    )


def test_channel_operand_order():
    # An asymmetric two-wire channel: a mixture of the identity and a CNOT
    # whose control is the first wire. A reversed operand order would make
    # the second wire the control.
    p = 0.3
    kraus = [np.sqrt(1 - p) * np.eye(4), np.sqrt(p) * qml.CNOT.compute_matrix()]
    ops = [qml.PauliX("c"), qml.QubitChannel(kraus, wires=["c", "t"])]
    tape = qml.tape.QuantumScript(ops, [qml.expval(qml.Z("t"))])
    dev = MENOSQubitDevice(wires=["c", "t"])
    ref = qml.device("default.mixed", wires=["c", "t"])
    value = qml.execute([tape], dev)[0]
    assert abs(value - qml.execute([tape], ref)[0]) < TOL
    assert abs(value - (1 - 2 * p)) < TOL


def test_seeds_reproduce_and_differ():
    def run(seed):
        dev = MENOSQubitDevice(wires=3, seed=seed)
        tape = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.Hadamard(1), qml.Hadamard(2)], [qml.sample()], shots=50
        )
        # Two executions on one device: the generator advances between them.
        return qml.execute([tape], dev)[0], qml.execute([tape], dev)[0]

    a1, a2 = run(7)
    b1, b2 = run(7)
    assert np.array_equal(a1, b1) and np.array_equal(a2, b2)
    assert not np.array_equal(a1, a2)
    c1, _ = run(8)
    assert not np.array_equal(a1, c1)
