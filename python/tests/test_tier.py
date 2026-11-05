"""RunOptions.tier and RunOptions.scratch_directory through the Python API
(Phase 23, docs/adr/0027): the disk tier reached the public way, a memory
budget the dense state does not fit, and forced; its counts and expectation
values are bit-for-bit the in-memory run's; explain() names the tier; a forced
tier never falls back.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import menos
from menos._core import Backend, Circuit, Observable, Opcode, RunOptions, Tier
from menos._core import run as menos_run

_N = 14
_SMALL_BUDGET = 256 * 1024  # the 256 KiB state and its outputs do not fit; a window does


def _entangled(num_qubits: int, layers: int = 3) -> Circuit:
    # Non-Clifford and entangled across the register, so the dense algorithm
    # is the only candidate (the stabilizer and MPS engines cannot take it).
    circuit = Circuit(num_qubits, num_qubits)
    half = num_qubits // 2
    for _ in range(layers):
        for q in range(num_qubits):
            circuit.append(Opcode.H, [q])
            circuit.append(Opcode.RX, [q], [0.3 + 0.1 * q])
        for q in range(half):
            circuit.append(Opcode.CX, [q, q + half])
        for q in range(num_qubits - 1):
            circuit.append(Opcode.CX, [q, q + 1])
    for q in range(num_qubits):
        circuit.append_measure(q, q)
    return circuit


def _pauli(num_qubits: int, x_word: int, z_word: int) -> menos._core.PauliString:
    pauli = menos._core.PauliString()
    pauli.num_qubits = num_qubits
    pauli.x_words = [x_word]
    pauli.z_words = [z_word]
    return pauli


def _three_terms(num_qubits: int) -> Observable:
    everything = (1 << num_qubits) - 1
    observable = Observable()
    observable.terms = [
        (_pauli(num_qubits, 0, everything), 1.0),
        (_pauli(num_qubits, everything, 0), 0.5),
        (_pauli(num_qubits, 1 << (num_qubits - 1), 1), 0.25),
    ]
    return observable


def _options(**fields) -> RunOptions:
    options = RunOptions()
    options.shots = 500
    options.seed = 11
    options.backend = Backend.Cpu
    for name, value in fields.items():
        setattr(options, name, value)
    return options


def _scratch_files(directory: pathlib.Path) -> set[str]:
    return {p.name for p in directory.iterdir() if p.name.startswith("menos-scratch-")}


def test_defaults_are_unset():
    options = RunOptions()
    assert options.tier is None
    assert options.scratch_directory is None
    assert {Tier.InMemory, Tier.Disk, Tier.HostOffloaded} == set(Tier.__members__.values())
    assert menos.Tier is Tier


def test_a_state_over_the_budget_runs_on_the_disk_tier_with_the_same_outputs(tmp_path):
    circuit = _entangled(_N)
    observable = _three_terms(_N)
    expected = menos_run(circuit, _options(expectation=observable))
    assert "tier" not in expected.explain()
    before = _scratch_files(tmp_path)
    actual = menos_run(
        circuit,
        _options(expectation=observable, memory_budget_bytes=_SMALL_BUDGET, scratch_directory=tmp_path),
    )
    assert actual.counts == expected.counts
    assert actual.expectation_value == expected.expectation_value
    assert actual.peak_disk_bytes == 16 * 2**_N
    assert actual.peak_host_bytes <= _SMALL_BUDGET
    assert expected.peak_disk_bytes == 0
    report = actual.explain()
    assert " tier=disk passes=" in report
    assert "engine=dense backend=cpu tier=disk" in report
    assert f'scratch_directory="{os.fspath(tmp_path)}"' in report
    assert f" peak_disk_bytes={16 * 2**_N}" in report
    assert _scratch_files(tmp_path) == before


def test_a_forced_disk_tier_runs_a_fitting_job_on_it(tmp_path):
    circuit = _entangled(10)
    forced = menos_run(circuit, _options(tier=Tier.Disk, scratch_directory=tmp_path))
    plain = menos_run(circuit, _options())
    assert "strategy   direct_dense (exact)" in plain.explain()
    assert forced.counts == plain.counts
    report = forced.explain()
    assert "tier       disk (requested)" in report
    assert "engine=dense backend=cpu tier=disk" in report
    assert " has no disk tier" in report
    assert forced.peak_disk_bytes == 16 * 2**10
    assert _scratch_files(tmp_path) == set()


def test_a_forced_tier_never_falls_back(tmp_path):
    circuit = _entangled(6)
    with pytest.raises(menos.ResourceExceededError):
        menos_run(circuit, _options(tier=Tier.Disk, scratch_directory=tmp_path / "missing"))
    statevector = _options(tier=Tier.Disk, scratch_directory=tmp_path)
    statevector.shots = 0
    statevector.want_statevector = True
    with pytest.raises(menos.UnsupportedFeatureError):
        menos_run(_entangled(6, 1), statevector)
    with pytest.raises(menos.UnsupportedFeatureError):
        menos_run(circuit, _options(tier=Tier.HostOffloaded))
    with pytest.raises(menos.ResourceExceededError) as refused:
        menos_run(_entangled(_N), _options(tier=Tier.InMemory, memory_budget_bytes=_SMALL_BUDGET))
    assert "disk" not in str(refused.value)


def test_a_statevector_request_over_the_budget_is_refused_without_the_tier():
    options = _options(memory_budget_bytes=_SMALL_BUDGET)
    options.shots = 0
    options.want_statevector = True
    with pytest.raises(menos.ResourceExceededError) as refused:
        menos_run(_entangled(_N), options)
    assert "direct dense simulation requires" in str(refused.value)
