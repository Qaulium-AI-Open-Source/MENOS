# Time Budgets

`time_budget_seconds` constrains how long a job may take. It is a **constraint, not a hint**: MENOS will refuse a job rather than knowingly exceed it.

## Setting one

The native API exposes it on `RunOptions`, the Qiskit `MENOSBackend` takes it as a run option, and the CLI takes `--time-budget SECONDS` (both since Phase 17). It must be finite and greater than 0; anything else raises `InvalidCircuitError`.

```python
from menos import Circuit, Opcode, RunOptions, run

circuit = Circuit(2, 2)
circuit.append(Opcode.H, [0])
circuit.append(Opcode.CX, [0, 1])
circuit.append_measure(0, 0)
circuit.append_measure(1, 1)

options = RunOptions()
options.shots = 100
options.seed = 1
options.time_budget_seconds = 5.0
report = run(circuit, options).explain()
print(next(line for line in report.splitlines() if line.startswith("time_budget")))
assert "time_budget" in report
```

With `MENOSBackend`, pass it like any other option: `backend.run(qc, time_budget_seconds=5.0)`. An attempt the job abandoned is reported in `result.results[0].metadata["menos_replans"]`. `MENOSEstimator` and `MENOSSampler` do not take a budget.

## What it bounds: one attempt

::: warning Read this
**`time_budget_seconds` bounds one execution attempt, not the total wall time of a job that replans.**
:::

If an attempt overruns, it is abandoned and the job is restarted under a new plan, which gets a fresh budget. A job that replans has therefore spent up to one budget on the abandoned attempt and then runs the replacement, so its total can approach several budgets. The abandoned attempt's time is reported in the plan report and is never hidden.

The total is bounded by roughly `(number of attempts) × (budget + one boundary) + planning`, and the number of attempts is at most one per distinct single-region plan (six) plus one. If you need a hard bound on *total* wall time, this option does not provide it.

Bounding the sum was considered and rejected: a crossing means the budget is already spent, so every replan would cross immediately and no replan could ever complete.

Planning and replanning themselves are not charged against the budget. They are cheap (see [Plan](/execution/plan)), and since Phase 24 their cost is reported: `plan_seconds` and `replan_seconds` on the result, and `plan_s` (and after a replan `replan_s`) on the `actual` line of the plan report.

## Two places it applies

### 1. At plan time

Each candidate's estimated seconds are computed from [calibration](/planning/cost-model#calibration). A candidate whose estimate exceeds the budget is infeasible, with a reason:

```text
estimated runtime 0.011243 s exceeds the job's time budget of 0.000000 s
```

If none fits, planning throws `ResourceExceeded` and nothing is run.

### 2. While running

Because the estimate can be wrong, the run is [monitored](/runtime/monitoring). An overrun is abandoned at a safe boundary and [replanned](/runtime/replanning).

## A refusal is final

If the budget is crossed and **no plan remains that fits**, MENOS throws `ResourceExceeded`, **even if the abandoned attempt was nearly complete**. A near-finished run that overran is discarded, not returned late. This is what "strict" means, and it affects only jobs that set a budget.

## What the report shows

A job with a budget prints a `time_budget` line and an `est_seconds` value on each feasible candidate. A job without a budget prints neither, so its report does not depend on the machine's speed. See [Plan Explanation](/planning/explain).

## Not set

A job that sets no time budget is never monitored and is never replanned, whatever it costs.

## An example of each outcome

```python
from menos import Circuit, Opcode, RunOptions, run
from menos.exceptions import ResourceExceededError

circuit = Circuit(2, 2)
circuit.append(Opcode.H, [0])
circuit.append(Opcode.CX, [0, 1])
circuit.append_measure(0, 0)
circuit.append_measure(1, 1)

def attempt(budget):
    options = RunOptions()
    options.shots = 10
    options.seed = 1
    options.time_budget_seconds = budget
    return run(circuit, options)

assert attempt(10.0).exact                  # fits: runs normally
try:
    attempt(1e-12)                          # no candidate can meet this: refused up front
except ResourceExceededError as error:
    assert "time budget" in str(error)
else:
    raise AssertionError("expected a refusal")
```

## Related

[Monitoring](/runtime/monitoring) · [Replanning](/runtime/replanning) · [ResourceExceeded](/runtime/resource-exceeded) · [Time budget example](/examples/time-budget)
