"""Result.peak_host_bytes and Result.peak_device_bytes through the Python API
(Phase 21, docs/adr/0025 section 3.2): the most host and CUDA device memory
the execution held, measured by the runtime's memory ledger, and their place
on explain()'s "actual" line.
"""

from __future__ import annotations

from menos._core import Backend, Circuit, Opcode, RunOptions
from menos._core import run as menos_run

_AMPLITUDE_BYTES = 16


def _entangled(num_qubits: int) -> Circuit:
    circuit = Circuit(num_qubits, 0)
    for _ in range(3):
        for q in range(num_qubits):
            circuit.append(Opcode.H, [q])
            circuit.append(Opcode.T, [q])
        for q in range(num_qubits - 1):
            circuit.append(Opcode.CX, [q, q + 1])
    return circuit


def _statevector_on_cpu() -> RunOptions:
    options = RunOptions()
    options.shots = 0
    options.want_statevector = True
    options.backend = Backend.Cpu
    return options


def test_a_dense_statevector_job_reports_exactly_its_state():
    result = menos_run(_entangled(12), _statevector_on_cpu())
    # The state, moved into the result rather than copied.
    assert result.peak_host_bytes == _AMPLITUDE_BYTES << 12
    assert result.peak_device_bytes == 0
    estimated = next(
        line for line in result.explain().splitlines() if line.startswith("estimated ")
    )
    estimated_bytes = int(estimated.split("state_bytes=")[1].split()[0])
    assert result.peak_host_bytes <= estimated_bytes


def test_explain_ends_the_actual_line_with_both_peaks():
    result = menos_run(_entangled(8), _statevector_on_cpu())
    actual = next(
        line for line in result.explain().splitlines() if line.startswith("actual ")
    )
    assert actual.endswith(
        f" peak_host_bytes={result.peak_host_bytes} peak_device_bytes=0"
    )
