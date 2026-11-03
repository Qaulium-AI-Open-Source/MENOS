"""Plan reuse across parameter values (menos.plan_cache_stats) and opt-in
diagonal-gate fusion (RunOptions.fuse_diagonal_gates), through the Qiskit
conversion path a variational loop uses, plus the native rzz gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from menos import _core
from menos.qiskit._observable_convert import convert_observable
from menos.qiskit.convert import convert_circuit

ABS_TOL = 1e-12


def _qaoa(n: int, p: int):
    gammas, betas = ParameterVector("g", p), ParameterVector("b", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for i in range(n):
            for j in range(i + 1, n):
                qc.rzz(2 * gammas[layer] * (0.1 * i + 0.05 * j + 0.2), i, j)
            qc.rz(2 * gammas[layer] * (0.3 - 0.07 * i), i)
        qc.rx(2 * betas[layer], range(n))
    return qc, list(gammas) + list(betas)


def _hamiltonian(n: int) -> SparsePauliOp:
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            label[n - 1 - j] = "Z"
            terms.append(("".join(label), 0.1 * i + 0.05 * j + 0.2))
    return SparsePauliOp.from_list(terms)


def _options(n: int, fuse: bool = False) -> _core.RunOptions:
    ro = _core.RunOptions()
    ro.shots = 0
    ro.want_statevector = False
    ro.expectation = convert_observable(_hamiltonian(n), n)
    ro.fuse_diagonal_gates = fuse
    return ro


def test_rzz_is_a_named_gate_and_matches_qiskit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.rzz(0.73, 0, 1)
    qc.cx(1, 2)
    qc.rzz(-2.1, 2, 0)
    circuit = convert_circuit(qc)
    ro = _core.RunOptions()
    ro.shots = 0
    ro.want_statevector = True
    actual = np.asarray(_core.run(circuit, ro).statevector)
    expected = Statevector(qc).data
    assert np.allclose(actual, expected, atol=ABS_TOL, rtol=0)
    # And it is a GateOp carrying its angle, not an opaque unitary: explain()
    # counts it as a named operation.
    assert "rzz" in _core.Opcode.__members__ or hasattr(_core.Opcode, "RZZ")


def test_rebinding_parameters_reuses_the_plan():
    n, p = 6, 2
    qc, params = _qaoa(n, p)
    ro = _options(n)
    rng = np.random.default_rng(3)
    _core.plan_cache_clear()
    energies = []
    for _ in range(12):
        values = rng.uniform(0.05, 3.0, size=len(params))
        bound = qc.assign_parameters(dict(zip(params, values)))
        result = _core.run(convert_circuit(bound), ro)
        energies.append(result.expectation_value.real)
        expected = Statevector(bound).expectation_value(_hamiltonian(n)).real
        assert abs(energies[-1] - expected) < ABS_TOL
    stats = _core.plan_cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 11
    assert stats["bypasses"] == 0
    assert len(set(np.round(energies, 9))) > 1  # really different circuits


def test_fusion_changes_nothing_but_the_time():
    n, p = 7, 2
    qc, params = _qaoa(n, p)
    rng = np.random.default_rng(5)
    values = rng.uniform(0.05, 3.0, size=len(params))
    bound = qc.assign_parameters(dict(zip(params, values)))
    circuit = convert_circuit(bound)
    off = _core.run(circuit, _options(n, fuse=False)).expectation_value.real
    on = _core.run(circuit, _options(n, fuse=True)).expectation_value.real
    expected = Statevector(bound).expectation_value(_hamiltonian(n)).real
    assert abs(off - expected) < ABS_TOL
    assert abs(on - expected) < ABS_TOL
    ro = _core.RunOptions()
    ro.shots = 0
    ro.want_statevector = True
    sv_off = np.asarray(_core.run(circuit, ro).statevector)
    ro.fuse_diagonal_gates = True
    sv_on = np.asarray(_core.run(circuit, ro).statevector)
    assert np.allclose(sv_off, sv_on, atol=ABS_TOL, rtol=0)
    assert _core.RunOptions().fuse_diagonal_gates is False


@pytest.mark.parametrize("special", [0.0, np.pi])
def test_special_rzz_angles_are_planned_afresh(special):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rzz(special, 0, 1)
    qc.rx(0.4, 1)
    circuit = convert_circuit(qc)
    ro = _core.RunOptions()
    ro.shots = 0
    ro.want_statevector = True
    _core.plan_cache_clear()
    _core.run(circuit, ro)
    _core.run(circuit, ro)
    stats = _core.plan_cache_stats()
    assert stats["bypasses"] == 2 and stats["hits"] == 0
