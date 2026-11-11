# Why MENOS

A single simulation algorithm is the right choice for some circuits and a poor one for others. A dense statevector needs memory that doubles with every qubit. A stabilizer simulator is cheap but only handles Clifford circuits. A tensor-network method can simulate hundreds of qubits when entanglement stays low and stalls when it does not. No fixed choice is best everywhere, and picking by hand requires knowing things about the circuit that are tedious to work out.

MENOS turns that choice into a measured, inspectable decision.

## What is different

### The decision is a cost comparison, not a heuristic

Every candidate is priced in one unit, the *work unit*, and the cheapest feasible candidate wins. A job with a time budget is instead ranked by predicted seconds, from each engine's own calibrated time model. There is no rule such as "Clifford circuits go to the stabilizer engine". A 200-qubit GHZ circuit is entirely Clifford, yet MENOS plans it onto the matrix-product-state engine because that is cheaper (3,400 work units against 361,000 for the stabilizer engine). See [Cost Model](/planning/cost-model).

### Infeasible is a first-class answer

A candidate that cannot run, because it would not fit in memory, does not support the requested output, or does not support a gate, is reported as infeasible *with the reason*. The plan report keeps the losers:

```text
candidate 0strategy=direct_dense feasible=false winner=false reason="checked_pow2: n = 100 exceeds the maximum of 62"
```

### Independent regions are simulated independently

If a circuit's qubits split into groups that never interact, MENOS simulates each group on its own and recombines the results exactly, using the cheapest eligible engine per group. See [Execution Plans](/planning/execution-plans).

### Work the output does not depend on is removed

If you ask for one expectation value, only the operations in the causal past of that observable can matter. [Light-cone reduction](/observables/light-cone-reduction) drops the rest before planning.

### Budgets are constraints, and they are enforced

A memory budget is checked before any allocation. A time budget is checked against the plan's estimate *and* against the run as it executes. When the estimate turns out to be wrong, the run is stopped at a safe boundary and [replanned from what was observed](/runtime/replanning). Nothing is silently allowed to overrun.

### Exact results, reproducibly

Every engine reachable from Python is exact. The same circuit, options and seed produce the same counts and the same plan. On a GPU, statevectors are bitwise identical to the CPU's. See [Determinism](/concepts/determinism) and [Exactness](/concepts/exactness).

### Memory is accounted for, not guessed

Every memory estimate is an upper bound on what the run really holds, checked by measurement, and every result reports its measured peak in RAM, GPU memory and disk. GPU memory is reused across jobs from MENOS's own pool, and a dense state too large for the memory budget can still run for counts or expectation values from a scratch file, with the same bits. See [Memory Architecture](/planning/memory-architecture).

### You can see what happened

`explain()` returns a plain-text report of the circuit summary, every candidate, the chosen strategy, the estimates, and what was actually measured. After a replan it also reports the threshold, the observation, the prediction and the reason. See [Plan Explanation](/planning/explain).

## Trade-offs

These are real and worth knowing before you adopt MENOS.

- **Planning has a cost.** It is small next to a non-trivial simulation (most jobs plan in well under 11 ms; see [How Planning Stays Fast](/planning/planner-performance)), but for a very small circuit planning can cost as much as executing it.
- **Calibration has a one-time cost per process.** The first job that needs calibrated seconds (any job on a CUDA build, a budgeted job on a CPU build) measures the machine first: about 1.5–2 s on the development machine. Later jobs in the same process do not pay it again; a new process does.
- **The estimate can be wrong.** Every CPU engine's time prediction is measured to be within 2× typically and 4× at the 90th percentile on the development machine, but predictions are for this machine's calibration and can miss elsewhere, for example when the operating system moves a one-thread job to a slower core. That is why time budgets are monitored while running rather than trusted at plan time. See [Monitoring](/runtime/monitoring).
- **Two platforms are verified:** Linux on x86-64 (with the CUDA backend) and Windows x64 (CPU only). See [Supported Platforms](/introduction/supported-platforms).
- **The Python surface is deliberately narrow.** Accuracy policy and plan inspection beyond `explain()` are C++ concepts that Python does not expose. See [Limitations](/reference/limitations).
