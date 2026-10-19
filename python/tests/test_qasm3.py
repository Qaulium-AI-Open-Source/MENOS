"""OpenQASM 3 import from Python (Phase 19, docs/adr/0023).

Gate criteria 3 (the 32 stdgates.inc names and U against Qiskit's matrices),
5 (differential against Qiskit's own import of the same text, and seeded
counts against MENOSBackend) and 8 (menos.from_qasm3), plus the section 4.1
measurement of Qiskit's vendored stdgates.inc kept as a regression test.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import qiskit
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import library as lib
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, Statevector

import menos
from menos import from_qasm3
from menos.exceptions import InvalidCircuitError, UnsupportedFeatureError
from menos.qiskit import MENOSBackend

TOL = 1e-12
ANGLES = (0.37, -1.21, 2.05, 0.83)

# name -> (Qiskit gate class, number of parameters, number of qubits)
STDGATES = {
    "p": (lib.PhaseGate, 1, 1),
    "x": (lib.XGate, 0, 1),
    "y": (lib.YGate, 0, 1),
    "z": (lib.ZGate, 0, 1),
    "h": (lib.HGate, 0, 1),
    "s": (lib.SGate, 0, 1),
    "sdg": (lib.SdgGate, 0, 1),
    "t": (lib.TGate, 0, 1),
    "tdg": (lib.TdgGate, 0, 1),
    "sx": (lib.SXGate, 0, 1),
    "rx": (lib.RXGate, 1, 1),
    "ry": (lib.RYGate, 1, 1),
    "rz": (lib.RZGate, 1, 1),
    "cx": (lib.CXGate, 0, 2),
    "cy": (lib.CYGate, 0, 2),
    "cz": (lib.CZGate, 0, 2),
    "cp": (lib.CPhaseGate, 1, 2),
    "crx": (lib.CRXGate, 1, 2),
    "cry": (lib.CRYGate, 1, 2),
    "crz": (lib.CRZGate, 1, 2),
    "ch": (lib.CHGate, 0, 2),
    "swap": (lib.SwapGate, 0, 2),
    "ccx": (lib.CCXGate, 0, 3),
    "cswap": (lib.CSwapGate, 0, 3),
    "cu": (lib.CUGate, 4, 2),
    "CX": (lib.CXGate, 0, 2),
    "phase": (lib.PhaseGate, 1, 1),
    "cphase": (lib.CPhaseGate, 1, 2),
    "id": (lib.IGate, 0, 1),
    "u1": (lib.U1Gate, 1, 1),
    "u2": (lib.U2Gate, 2, 1),
    "u3": (lib.U3Gate, 3, 1),
}


def _statevector(circuit: menos.Circuit) -> np.ndarray:
    options = menos.RunOptions()
    options.shots = 0
    options.want_statevector = True
    options.seed = 0
    return np.asarray(menos.run(circuit, options).statevector)


def _unitary(program_body: str, num_qubits: int, prelude: str = 'include "stdgates.inc";\n') -> np.ndarray:
    """The num_qubits-qubit unitary of program_body, one column per basis state.

    Column k is the statevector after preparing |k> with x gates, which is
    exact. Bit q of an index is qubit q, as in Qiskit.
    """
    dim = 1 << num_qubits
    columns = []
    for k in range(dim):
        prep = "".join(f"U(pi, 0, pi) q[{q}];\n" for q in range(num_qubits) if (k >> q) & 1)
        text = f"OPENQASM 3.0;\n{prelude}qubit[{num_qubits}] q;\n{prep}{program_body}"
        columns.append(_statevector(from_qasm3(text)))
    return np.stack(columns, axis=1)


def _call(name: str, num_params: int, num_qubits: int) -> str:
    params = f"({', '.join(str(a) for a in ANGLES[:num_params])})" if num_params else ""
    operands = ", ".join(f"q[{q}]" for q in range(num_qubits))
    return f"{name}{params} {operands};\n"


@pytest.mark.parametrize("name", sorted(STDGATES))
def test_each_stdgates_name_has_qiskits_matrix(name):
    gate_class, num_params, num_qubits = STDGATES[name]
    mine = _unitary(_call(name, num_params, num_qubits), num_qubits)
    reference = Operator(gate_class(*ANGLES[:num_params])).data
    assert np.max(np.abs(mine - reference)) <= TOL


def test_builtin_u_is_qiskits_ugate():
    mine = _unitary("U(0.37, -1.21, 2.05) q[0];\n", 1, prelude="")
    assert np.max(np.abs(mine - Operator(lib.UGate(0.37, -1.21, 2.05)).data)) <= TOL


# docs/adr/0023 section 4.1: Qiskit's vendored stdgates.inc, read literally
# (here by MENOS itself, as user gate definitions with U = Qiskit's UGate),
# disagrees with the gates its names stand for. If Qiskit ships a corrected
# file this test fails, and section 4.1 and the built-in choice are revisited.
LITERAL_EXACT = {"p", "z", "rz", "cz", "cp", "crz", "CX", "phase", "cphase", "id", "u1"}
LITERAL_PHASE_ONLY = {"x", "y", "h", "rx", "ry", "u2", "u3"}
LITERAL_DIFFERENT = {"cx", "cy", "crx", "cry", "ch", "swap", "ccx", "cswap", "cu"}
# The bodies of s, sdg, t, tdg and sx use pow(0.5), which the importer
# refuses (only integer powers are exact here).
LITERAL_NEEDS_FRACTIONAL_POW = {"s", "sdg", "t", "tdg", "sx"}


def test_qiskits_vendored_stdgates_bodies_disagree_with_their_names():
    vendored = Path(qiskit.__file__).parent / "qasm" / "libs" / "stdgates.inc"
    definitions = vendored.read_text(encoding="utf-8")
    exact, phase_only, different, refused = set(), set(), set(), set()
    for name, (gate_class, num_params, num_qubits) in STDGATES.items():
        try:
            mine = _unitary(_call(name, num_params, num_qubits), num_qubits, prelude=definitions)
        except UnsupportedFeatureError:
            refused.add(name)
            continue
        reference = Operator(gate_class(*ANGLES[:num_params])).data
        if np.max(np.abs(mine - reference)) <= TOL:
            exact.add(name)
            continue
        index = np.unravel_index(np.argmax(np.abs(reference)), reference.shape)
        phase = mine[index] / reference[index]
        if np.max(np.abs(mine - phase * reference)) <= 1e-9:
            phase_only.add(name)
        else:
            different.add(name)
    assert refused == LITERAL_NEEDS_FRACTIONAL_POW
    assert exact == LITERAL_EXACT
    assert phase_only == LITERAL_PHASE_ONLY
    assert different == LITERAL_DIFFERENT


# Qiskit 2.5.2's exporter cannot write gate modifiers (it refuses an
# AnnotatedOperation as a "non-unitary subroutine call"), so the modifier
# fixture is written by hand and compared with Qiskit's import of it. It
# controls and inverts a user gate whose body has a gphase, where the phase
# becomes observable.
MODIFIER_PROGRAM = """OPENQASM 3.0;
include "stdgates.inc";
gate g(a) x, y { h x; cx x, y; rz(a) y; gphase(0.3); }
qubit[4] q;
h q[0]; h q[1]; h q[2]; h q[3];
ctrl @ ry(0.3) q[0], q[1];
inv @ s q[2];
pow(3) @ t q[1];
ctrl(2) @ h q[0], q[1], q[2];
negctrl @ x q[3], q[0];
ctrl @ inv @ g(0.7) q[3], q[1], q[2];
inv @ pow(2) @ g(-0.4) q[0], q[3];
negctrl(2) @ rx(1.1) q[1], q[2], q[0];
ctrl @ ctrl @ x q[2], q[0], q[3];
ctrl @ gphase(0.5) q[1];
"""


# qiskit-qasm3-import 0.6.0 (converter.py, _apply_gate_modifier) still calls
# Gate.control(annotated=None), which Qiskit 2.3+ deprecates. The call is in the
# reference importer, not in MENOS; drop this when that package is updated.
@pytest.mark.filterwarnings(
    r"ignore:.*Gate\.control\(\).*annotated.*:DeprecationWarning"
)
def test_modifier_program_equals_qiskits_import():
    mine = _statevector(from_qasm3(MODIFIER_PROGRAM))
    theirs = Statevector(qasm3.loads(MODIFIER_PROGRAM)).data
    assert np.max(np.abs(mine - theirs)) <= TOL


def _stdgates_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h([0, 1, 2])
    for name, (gate_class, num_params, num_qubits) in STDGATES.items():
        if name in ("CX", "phase", "cphase", "u1", "u2", "u3"):
            continue  # Qiskit exports these under their canonical names
        qc.append(gate_class(*ANGLES[:num_params]), list(range(num_qubits)))
    qc.sxdg(1)
    qc.rzz(0.4, 0, 2)
    qc.u(0.1, 0.2, 0.3, 1)
    return qc


def _differential_fixtures() -> list[tuple[str, QuantumCircuit]]:
    fixtures = [("stdgates", _stdgates_circuit())]
    for seed in range(12):
        fixtures.append(
            (f"random-{seed}", random_circuit(4, depth=8, max_operands=3, seed=1000 + seed))
        )
    return fixtures


@pytest.mark.parametrize("circuit", [c for _, c in _differential_fixtures()], ids=[n for n, _ in _differential_fixtures()])
def test_statevector_equals_qiskits_import_of_the_same_text(circuit):
    text = qasm3.dumps(circuit)
    mine = _statevector(from_qasm3(text))
    theirs = Statevector(qasm3.loads(text)).data
    assert np.max(np.abs(mine - theirs)) <= TOL, text
    # The exporter drops the circuit's global phase, so against the original
    # circuit the states agree up to one phase factor.
    original = Statevector(circuit).data
    overlap = np.vdot(original, mine)
    assert abs(abs(overlap) - 1.0) <= TOL
    assert np.max(np.abs(mine - (overlap / abs(overlap)) * original)) <= TOL


@pytest.mark.parametrize("seed", [3, 17, 2026])
def test_seeded_counts_equal_menos_backend(seed):
    circuit = random_circuit(4, depth=6, max_operands=2, measure=True, seed=seed)
    text = qasm3.dumps(circuit)
    options = menos.RunOptions()
    options.shots = 500
    options.seed = seed
    counts = menos.run(from_qasm3(text), options).counts
    backend_counts = MENOSBackend().run(circuit, shots=500, seed_simulator=seed).result().get_counts()
    width = circuit.num_clbits
    mine = {format(int(key, 16), f"0{width}b"): value for key, value in counts.items()}
    assert mine == backend_counts


def test_inputs_bind_by_name_and_none_means_no_inputs():
    text = 'OPENQASM 3.0;\ninclude "stdgates.inc";\ninput float theta;\nqubit q;\nrx(theta) q;\n'
    state = _statevector(from_qasm3(text, {"theta": math.pi}))
    assert abs(abs(state[1]) - 1.0) <= TOL
    assert from_qasm3('OPENQASM 3.0;\nqubit q;\n', None).num_qubits() == 1
    with pytest.raises(InvalidCircuitError, match="input 'theta' has no binding"):
        from_qasm3(text)
    with pytest.raises(InvalidCircuitError, match="names no 'input' declaration"):
        from_qasm3(text, {"theta": 1.0, "other": 2.0})


def test_refused_constructs_raise_unsupported_feature_with_their_location():
    with pytest.raises(UnsupportedFeatureError, match=r"'reset'.*line 3, column 1"):
        from_qasm3("OPENQASM 3.0;\nqubit q;\nreset q;\n")
    with pytest.raises(UnsupportedFeatureError, match="'if'"):
        from_qasm3("OPENQASM 3.0;\nqubit q;\nbit c;\nif (c) U(0, 0, 0) q;\n")
