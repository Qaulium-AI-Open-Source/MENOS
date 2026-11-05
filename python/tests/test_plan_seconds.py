"""Result.plan_seconds and Result.replan_seconds through the Python API
(Phase 24, docs/adr/0028 section 16.7 Q3): the measured wall-clock time of the
initial planner call, the replans' planning time summed, and their place on
explain()'s "actual" line.
"""

from __future__ import annotations

import math

from menos._core import Circuit, Opcode, RunOptions
from menos._core import run as menos_run


def _bell_counts() -> tuple[Circuit, RunOptions]:
    circuit = Circuit(2, 2)
    circuit.append(Opcode.H, [0])
    circuit.append(Opcode.CX, [0, 1])
    circuit.append_measure(0, 0)
    circuit.append_measure(1, 1)
    options = RunOptions()
    options.shots = 16
    options.want_statevector = False
    return circuit, options


def test_a_run_reports_its_planning_time_and_no_replanning_time():
    circuit, options = _bell_counts()
    result = menos_run(circuit, options)
    assert result.plan_seconds > 0.0
    assert math.isfinite(result.plan_seconds)
    assert result.replan_seconds == 0.0
    assert result.replans == []


def test_explain_puts_the_planning_times_with_the_other_wall_clock_fields():
    circuit, options = _bell_counts()
    actual = next(
        line
        for line in menos_run(circuit, options).explain().splitlines()
        if line.startswith("actual ")
    )
    assert actual.index("sample_s=") < actual.index(" plan_s=")
    assert actual.index(" plan_s=") < actual.index(" peak_host_bytes=")
    # Nothing was replanned, so the report carries no replanning field.
    assert " replan_s=" not in actual
