"""Explicit backend selection from Python (Phase 17, docs/adr/0021 section 2).

RunOptions.backend and the MENOSBackend run option backend. A forced CPU run
must be the unforced run plus one report line; a forced CUDA request is either
honoured or refused with UnsupportedFeatureError, never run on the CPU.
"""

from __future__ import annotations

import re

import pytest
from qiskit import QuantumCircuit

import menos
from menos._core import Circuit, Opcode, RunOptions
from menos._core import run as menos_run
from menos.exceptions import UnsupportedFeatureError


def _bell() -> Circuit:
    circuit = Circuit(2, 2)
    circuit.append(Opcode.H, [0])
    circuit.append(Opcode.CX, [0, 1])
    circuit.append_measure(0, 0)
    circuit.append_measure(1, 1)
    return circuit


def _without_timings(report: str) -> str:
    return re.sub(
        r"(execute_s|sample_s|plan_s|replan_s)=\S+",
        r"\1=<t>",
        report,
    )


def _options(**fields) -> RunOptions:
    options = RunOptions()
    options.shots = 100
    options.seed = 1
    for name, value in fields.items():
        setattr(options, name, value)
    return options


def test_backend_is_exported_and_defaults_to_none():
    assert menos.Backend.Cpu != menos.Backend.Cuda
    options = RunOptions()
    assert options.backend is None
    options.backend = menos.Backend.Cpu
    assert options.backend == menos.Backend.Cpu
    options.backend = None
    assert options.backend is None


def test_forced_cpu_changes_only_the_report_line():
    unforced = menos_run(_bell(), _options())
    forced = menos_run(_bell(), _options(backend=menos.Backend.Cpu))
    assert forced.counts == unforced.counts
    assert "backend    cpu (requested)\n" in forced.explain()
    # execute_s and sample_s are wall-clock times, so they are blanked first.
    forced_report = _without_timings(forced.explain()).replace("backend    cpu (requested)\n", "")
    assert forced_report == _without_timings(unforced.explain())
    assert "(requested)" not in unforced.explain()


def test_forced_cuda_is_honoured_or_refused_never_run_on_the_cpu():
    try:
        result = menos_run(_bell(), _options(backend=menos.Backend.Cuda))
    except UnsupportedFeatureError as error:
        # A build without CUDA support, or no usable device.
        assert "backend=cuda" in str(error)
        return
    assert "engine=dense backend=cuda" in result.explain()


def _bell_qc() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_the_qiskit_backend_defaults_to_no_backend_request(backend):
    assert backend.options.backend is None


def test_the_qiskit_backend_accepts_cpu(backend):
    reference = backend.run(_bell_qc(), shots=128, seed_simulator=3).result()
    forced = backend.run(_bell_qc(), shots=128, seed_simulator=3, backend="cpu").result()
    assert forced.get_counts() == reference.get_counts()


def test_the_qiskit_backend_refuses_or_honours_cuda(backend):
    try:
        result = backend.run(_bell_qc(), shots=128, seed_simulator=3, backend="cuda").result()
    except UnsupportedFeatureError as error:
        assert "backend=cuda" in str(error)
        return
    assert sum(result.get_counts().values()) == 128


@pytest.mark.parametrize("value", ["gpu", "CPU", "", 1])
def test_the_qiskit_backend_rejects_an_unknown_backend(backend, value):
    with pytest.raises(ValueError, match="backend must be None or one of"):
        backend.run(_bell_qc(), shots=16, backend=value)
