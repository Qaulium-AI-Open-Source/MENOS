"""time_budget_seconds through the Python API (Phase 11, docs/adr/0015).

Phase 10 added the field in C++ but deliberately did not expose it here, on the
evidence that no Python caller needed it. Phase 11 changes that evidence: the
budget is what triggers runtime monitoring and replanning, so leaving it
C++-only would have put the whole feature out of reach of the primary entry
point.

Most tests go through menos._core directly. Phase 17 (docs/adr/0021 section 3)
lifted the deferral that kept the Qiskit adapter from setting the field:
MENOSBackend now takes a time_budget_seconds run option, tested at the end.
"""

from __future__ import annotations

import pytest

from menos._core import Circuit, Opcode, RunOptions
from menos._core import run as menos_run
from menos.exceptions import ResourceExceededError


def _layered(num_qubits: int, layers: int) -> Circuit:
    circuit = Circuit(num_qubits, num_qubits)
    for layer in range(layers):
        for q in range(num_qubits):
            circuit.append(Opcode.H, [q])
        circuit.append(Opcode.T, [0])
        for q in range(layer % 2, num_qubits - 1, 2):
            circuit.append(Opcode.CX, [q, q + 1])
    for q in range(num_qubits):
        circuit.append_measure(q, q)
    return circuit


def test_time_budget_seconds_round_trips():
    options = RunOptions()
    assert options.time_budget_seconds is None
    options.time_budget_seconds = 2.5
    assert options.time_budget_seconds == pytest.approx(2.5)
    options.time_budget_seconds = None
    assert options.time_budget_seconds is None


def test_a_generous_budget_does_not_change_the_answer():
    circuit = _layered(6, 3)

    unbudgeted = RunOptions()
    unbudgeted.shots = 256
    unbudgeted.seed = 3
    reference = menos_run(circuit, unbudgeted)

    budgeted = RunOptions()
    budgeted.shots = 256
    budgeted.seed = 3
    budgeted.time_budget_seconds = 3600.0  # no real machine needs an hour here
    result = menos_run(circuit, budgeted)

    assert result.counts == reference.counts


def test_an_unmeetable_budget_raises_resource_exceeded():
    # Small enough that no candidate's estimate can fit, so this is refused at
    # plan time. The run-time refusal (observed rather than estimated) needs a
    # budget straddling a wrong prediction, which is machine-dependent and is
    # covered by the C++ suite instead.
    circuit = _layered(8, 4)
    options = RunOptions()
    options.shots = 16
    options.seed = 3
    options.time_budget_seconds = 1e-12
    with pytest.raises(ResourceExceededError):
        menos_run(circuit, options)


def test_the_budget_is_reported_in_explain():
    circuit = _layered(6, 3)
    options = RunOptions()
    options.shots = 16
    options.seed = 3
    options.time_budget_seconds = 3600.0
    report = menos_run(circuit, options).explain()
    assert "time_budget" in report
    # No replan happened, so the report carries no replan section at all.
    assert "replan" not in report


def _bell_qc():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(3), range(3))
    return qc


# Phase 17 (docs/adr/0021 section 3, owner Q3). This replaces the Phase 11 test
# test_the_qiskit_backend_still_exposes_no_time_budget, which pinned the
# deferral this phase lifts: it asserted that the option did NOT exist.


def test_the_qiskit_backend_defaults_to_no_time_budget(backend):
    assert backend.options.time_budget_seconds is None


# A budget ranks candidates by predicted seconds (Phase 24 item 4, owner
# option (b), docs/adr/0028 section 21), so a budgeted job may run a different
# exact plan than the unbudgeted one -- here direct stabilizer instead of
# factorized -- and draw a different sample for the same seed. Phase 17's
# same-counts check is therefore replaced by a check that the counts are a
# valid sample of the Bell state (owner, 2026-11-11, section 21.6).
def test_the_qiskit_backend_honours_a_generous_budget(backend):
    qc = _bell_qc()
    budgeted = backend.run(qc, shots=256, seed_simulator=5, time_budget_seconds=3600.0).result()
    counts = budgeted.get_counts()
    assert set(counts) <= {"000", "011"}
    assert sum(counts.values()) == 256
    # No attempt was abandoned, so nothing is added to the metadata.
    assert "menos_replans" not in getattr(budgeted.results[0], "metadata", {})


def test_the_qiskit_backend_refuses_an_unmeetable_budget(backend):
    with pytest.raises(ResourceExceededError):
        backend.run(_bell_qc(), shots=16, time_budget_seconds=1e-12)


@pytest.mark.parametrize("budget", [0.0, -1.0, float("nan"), float("inf")])
def test_the_qiskit_backend_rejects_an_invalid_budget(backend, budget):
    from menos.exceptions import InvalidCircuitError

    with pytest.raises(InvalidCircuitError):
        backend.run(_bell_qc(), shots=16, time_budget_seconds=budget)


@pytest.mark.parametrize("budget", [0.0, -1.0, float("nan"), float("inf")])
def test_run_options_reject_an_invalid_budget(budget):
    from menos.exceptions import InvalidCircuitError

    options = RunOptions()
    options.shots = 16
    options.seed = 3
    options.time_budget_seconds = budget
    with pytest.raises(InvalidCircuitError):
        menos_run(_layered(4, 1), options)


def test_a_result_without_a_replan_has_no_replans():
    options = RunOptions()
    options.shots = 16
    options.seed = 3
    options.time_budget_seconds = 3600.0
    assert menos_run(_layered(4, 1), options).replans == []


def test_a_replan_record_reaches_the_qiskit_metadata():
    """A run-time replan cannot be forced from Python on every machine (see the
    plan-time refusal test above), so the adapter's metadata step is tested on a
    stand-in result carrying one event, shaped as Result.replans returns it."""
    from types import SimpleNamespace

    from menos.qiskit.result_adapter import build_qiskit_result

    event = {
        "sequence": 1,
        "threshold_seconds": 0.5,
        "observed_seconds": 0.6,
        "predicted_seconds": 0.1,
        "boundaries_crossed": 12,
        "engine": "dense",
        "backend": "cpu",
        "reason": "observed elapsed time exceeded the budget",
    }
    stand_in = SimpleNamespace(
        counts={"0x0": 16},
        statevector=None,
        execute_seconds=0.0,
        sample_seconds=0.0,
        replans=[event],
    )
    result = build_qiskit_result(
        circuit=_bell_qc(),
        menos_result=stand_in,
        shots=16,
        seed_simulator=1,
        backend_name="menos",
        backend_version="0",
        job_id="j",
        want_statevector=False,
    )
    assert result.results[0].metadata == {"menos_replans": [event]}
