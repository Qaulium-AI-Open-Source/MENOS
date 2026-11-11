"""Builder circuits agree with the same circuits built in Qiskit (docs/adr/0029
Q9 criterion 6).

Each fixture is written twice: once with menos.CircuitBuilder, run through
menos.run, and once with qiskit.QuantumCircuit, run through MENOSBackend. The
statevectors agree within 1e-12, global phase included (which bounds every
expectation value), both agree with Qiskit's own Statevector, and the seeded
counts are identical.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter as QiskitParameter
from qiskit.quantum_info import Statevector

import menos

ABS_TOL = 1e-12
SHOTS = 2000
SEED = 1234


def _options(**fields):
    options = menos.RunOptions()
    for name, value in fields.items():
        setattr(options, name, value)
    return options


def bell_with_phase():
    b = menos.CircuitBuilder(2)
    b.global_phase = 0.4
    b.h(0).cx(0, 1).s(1)
    q = QuantumCircuit(2)
    q.global_phase = 0.4
    q.h(0)
    q.cx(0, 1)
    q.s(1)
    return b, q


def every_gate():
    b = menos.CircuitBuilder(3)
    b.x(0).y(1).z(2).h(0).s(1).sdg(2).t(0).tdg(1).sx(2).sxdg(0)
    b.rx(0.3, 1).ry(-0.7, 2).rz(1.1, 0).p(0.9, 1).u(0.1, 0.2, 0.3, 2)
    b.cx(0, 1).cz(1, 2).swap(0, 2).cp(0.5, 2, 1).rzz(-1.25, 0, 1).ccx(1, 2, 0)
    q = QuantumCircuit(3)
    q.x(0)
    q.y(1)
    q.z(2)
    q.h(0)
    q.s(1)
    q.sdg(2)
    q.t(0)
    q.tdg(1)
    q.sx(2)
    q.sxdg(0)
    q.rx(0.3, 1)
    q.ry(-0.7, 2)
    q.rz(1.1, 0)
    q.p(0.9, 1)
    q.u(0.1, 0.2, 0.3, 2)
    q.cx(0, 1)
    q.cz(1, 2)
    q.swap(0, 2)
    q.cp(0.5, 2, 1)
    q.rzz(-1.25, 0, 1)
    q.ccx(1, 2, 0)
    return b, q


def registers_and_parameters():
    """Named registers and a bound parameter, built by register in both."""
    theta_value = 0.8125
    b = menos.CircuitBuilder()
    a = b.add_qreg("a", 2)
    r = b.add_qreg("r", 2)
    theta = menos.Parameter("theta")
    b.h(a[0]).ry(theta, r[1]).cx(a[0], r[0]).rzz(theta, a[1], r[1]).cp(theta, r[0], a[1])
    b = _rebuild(b, {"theta": theta_value})

    qa = QuantumRegister(2, "a")
    qr = QuantumRegister(2, "r")
    q = QuantumCircuit(qa, qr)
    t = QiskitParameter("theta")
    q.h(qa[0])
    q.ry(t, qr[1])
    q.cx(qa[0], qr[0])
    q.rzz(t, qa[1], qr[1])
    q.cp(t, qr[0], qa[1])
    q = q.assign_parameters({t: theta_value})
    return b, q


def inverse_composition():
    body = menos.CircuitBuilder(3)
    body.h(0).u(0.3, -0.2, 0.9, 1).cx(1, 2).rzz(0.4, 0, 2).sx(1)
    b = menos.CircuitBuilder(3)
    b.ry(0.6, 0).ry(-0.2, 1).compose(body).compose(body.repeat(2).inverse())

    qbody = QuantumCircuit(3)
    qbody.h(0)
    qbody.u(0.3, -0.2, 0.9, 1)
    qbody.cx(1, 2)
    qbody.rzz(0.4, 0, 2)
    qbody.sx(1)
    q = QuantumCircuit(3)
    q.ry(0.6, 0)
    q.ry(-0.2, 1)
    q.compose(qbody, inplace=True)
    q.compose(qbody.repeat(2).decompose().inverse(), inplace=True)
    return b, q


def _rebuild(b, values):
    """A parameter-free builder with b's registers and b's operations bound."""
    out = menos.CircuitBuilder()
    for reg in b.qregs():
        out.add_qreg(reg.name, reg.size)
    for reg in b.cregs():
        out.add_creg(reg.name, reg.size)
    out.compose(menos.CircuitBuilder.from_circuit(b.bind(values)))
    return out


FIXTURES = [bell_with_phase, every_gate, registers_and_parameters, inverse_composition]


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f.__name__ for f in FIXTURES])
def test_statevectors_agree(fixture, backend):
    builder, qc = fixture()
    ours = menos.run(builder, _options(shots=0, want_statevector=True, seed=SEED)).statevector
    via_backend = np.asarray(
        backend.run(qc, shots=0, want_statevector=True).result().get_statevector(qc)
    )
    reference = Statevector(qc).data
    assert np.allclose(ours, via_backend, atol=ABS_TOL, rtol=0)
    assert np.allclose(ours, reference, atol=ABS_TOL, rtol=0)


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f.__name__ for f in FIXTURES])
def test_seeded_counts_are_identical(fixture, backend):
    builder, qc = fixture()
    builder.measure_all()
    qc = qc.copy()
    qc.measure_all()
    ours = menos.run(builder, _options(shots=SHOTS, seed=SEED)).counts
    theirs = backend.run(qc, shots=SHOTS, seed_simulator=SEED).result().data(qc)["counts"]
    assert ours == theirs


def test_measure_all_register_matches_qiskit():
    builder = menos.CircuitBuilder(2, 1)
    builder.measure_all()
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(1, "c"))
    qc.measure_all()
    assert [(r.name, r.size) for r in builder.cregs()] == [(r.name, r.size) for r in qc.cregs]
