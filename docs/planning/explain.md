# Plan Explanation

`Result.explain()` returns a plain-text report of how a job was planned and what happened. It is available from the [native API](/build/native-api) and the [command line](/build/command-line).

## An annotated report

```text
MENOS EXECUTION REPORT
circuit    qubits=4 clbits=4 operations=8 depth=3 global_phase=0
light_cone qubits 4->4 operations 8->8
candidate 0strategy=direct_dense feasible=true winner=false peak_bytes=256 work_units=224
candidate 1strategy=direct_stabilizer feasible=true winner=false peak_bytes=72 work_units=56
candidate 2strategy=direct_mps feasible=true winner=false peak_bytes=256 work_units=68
candidate 3strategy=factorized feasible=true winner=true peak_bytes=40 work_units=32
strategy   factorized (exact)
region 0   qubits=[0,1] engine=stabilizer backend=cpu why="stabilizer (pauli rejected: …)"
region 1   qubits=[2,3] engine=stabilizer backend=cpu why="stabilizer (pauli rejected: …)"
estimated  state_bytes=40 work_units=32
actual     state_bytes=40 execute_s=0.000006 sample_s=0.000010
```

| Line | Meaning |
|---|---|
| `circuit` | Facts about the **original** circuit: qubits, classical bits, operation count, depth, global phase |
| `noise` | `channels=… readout_errors=…` in the original circuit. **Only present when the job contains noise** ([Noise Models](/build/noise-models)) |
| `light_cone` | Qubits and operations before and after [light-cone reduction](/observables/light-cone-reduction) |
| `time_budget` | The budget in seconds. **Only present when the job set one** |
| `tier` | `disk (requested)` or `in_memory (requested)`. **Only present when the job set `RunOptions.tier`** (since Phase 23) |
| `candidate i` | One per enumerated candidate: strategy, feasible, winner, peak bytes, work units. A candidate on the GPU also shows `device_bytes`; one on the disk tier shows `tier=disk passes=… chunk_qubits=… disk_bytes=…` (since Phase 23). A refused or [pruned](#pruned-candidates) candidate shows `reason="…"` instead. With a time budget each feasible one also shows `est_seconds` |
| `strategy` | The chosen strategy and `(exact)` or `(approximate)` |
| `region i` | A region's qubits, engine and backend, and `tier=disk` when it runs on the disk tier; for factorized plans `why=` explains the engine choice |
| `cut` | Present for `cut_and_recombine`: cuts, fragments, branches, distinct evaluations, peak bytes |
| `estimated` | The plan's estimated peak bytes, an upper bound, and its work units. When the plan puts anything on the GPU, also `device_available_bytes`, the free device memory it was checked against. When it uses the disk tier, also `disk_available_bytes`, `scratch_directory` and the directory's `disk_write_bandwidth` and `disk_read_bandwidth` followed by `(measured)`, or `disk_bandwidth=unmeasured` |
| `replan i` | Present only after a replan; see below |
| `actual` | What happened: the engine's state bytes, seconds executing and sampling, and the measured `peak_host_bytes` and `peak_device_bytes` (since Phase 21), with `peak_disk_bytes` when it is not 0 (since Phase 23) |
| `expectation` | The expectation value, when one was requested |

Extra detail can appear on a `region` line: `accumulated_error_bound` for an approximating MPS run, and `actual_peak_bytes` for a tensor-network region.

::: info A formatting detail
Labels are padded to 11 characters. Labels that are already 11 or more characters (`candidate 0`, `time_budget`) therefore run straight into their value: `candidate 0strategy=…`, `time_budget5 s`. That is the format as shipped.
:::

## Determinism

The report is byte-identical across runs of the same job, with two exceptions:

- the two wall-clock fields, `execute_s` and `sample_s`;
- for a plan that uses the GPU, `device_available_bytes`, which depends on what other processes hold;
- for a plan that uses the disk tier, `disk_available_bytes`, which depends on what else is on the volume, and the two bandwidth figures, which are measured once per process.

Estimated seconds are machine-dependent, so `est_seconds` and the `time_budget` line are printed only when a time budget was set. Without one, a report does not depend on the machine's speed. Byte figures count the standard library's container sizes, so they can differ slightly between platforms.

## Why losers are listed

A plan report keeps every candidate, including infeasible ones, with the reason:

```text
candidate 0strategy=direct_dense feasible=false winner=false reason="checked_pow2: n = 100 exceeds the maximum of 62"
candidate 1strategy=direct_stabilizer feasible=false winner=false reason="operation 0 (ry) is not a Clifford gate"
```

That is usually all you need to understand why a large job was planned the way it was.

## Pruned candidates

::: info Not in a published release yet
Pruning is on the development branch (up to Phase 24, planner performance) and is not in a published release. A released version costs every candidate in full.
:::

Costing a tensor-network or cut-and-recombine candidate is itself a search. Once a cheaper feasible candidate is known, the planner stops such a search as soon as its running total passes that candidate's work units. A candidate stopped this way could never have won: the winner is the feasible candidate with the fewest work units. Its line shows where it was stopped instead of its full figures:

```text
candidate 4strategy=direct_tensor_network feasible=false winner=false reason="pruned by branch-and-bound: costing stopped before any elimination search: a lower bound of 20857968 work units over its 127 observable terms (from each term's network size) is already more than the 2105344 of a cheaper feasible alternative"
```

The same applies to the engine choice inside each `factorized` region. An engine stopped there is listed in the region's `why=` as `(tensor_network rejected: pruned by branch-and-bound: …)`.

To give these searches a bound early, the cheaper candidates are costed first. The table still lists candidates in the [fixed order](/execution/plan#candidates-in-a-fixed-order). With a time budget, pruning works the same way on the CPU: only a candidate that is also within the budget can be the bound. When the CUDA backend can be chosen, every candidate is costed in full.

Pruning never changes the chosen plan or its estimate. It only shortens planning and changes the losing lines.

## After a replan

If a run overran its time budget and was restarted, the report gains a block before the `actual` line. The threshold, the observed time, the prediction, the reason and the old and new plans are all there:

```text
replan 0   threshold=0.2 s crossed at elapsed=0.224734 s
           abandoned after 194 safe boundaries on engine=mps backend=cpu
           was  strategy=direct_mps predicted=0.000835369 s
           reason="observed runtime exceeded the time budget; mps on cpu is excluded by an observed floor of 0.224734 s"
           now  strategy=direct_dense
```

This is real output from the [replanning example](/examples/replanning). A run that never replans produces a report with no such block, byte-identical to one produced without a budget apart from the budget lines. See [Replanning](/runtime/replanning).

## Reading it from Python

`explain()` returns a string. To pick out a value, search for the line you want:

```python
from menos import Circuit, Opcode, RunOptions, run

circuit = Circuit(2, 2)
circuit.append(Opcode.H, [0])
circuit.append(Opcode.CX, [0, 1])
circuit.append_measure(0, 0)
circuit.append_measure(1, 1)

options = RunOptions()
options.shots = 10
options.seed = 1
report = run(circuit, options).explain()

strategy = next(line for line in report.splitlines() if line.startswith("strategy"))
print(strategy)
assert "direct_stabilizer" in strategy
```

The text format is meant for people. There is no structured `Plan` object in Python.

## Related

[Execution Plans](/planning/execution-plans) · [Cost Model](/planning/cost-model) · [Plan Explanation example](/examples/planner-explain)
