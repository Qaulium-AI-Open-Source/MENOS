# How MENOS Executes

Every job goes through the same stages, whether it is a two-qubit circuit or a hundred-qubit one.

```text
1. Convert     Qiskit, PennyLane, OpenQASM, CircuitBuilder  →  KIR
2. Validate    KIR well-formed, measurements terminal, output requested
3. Reduce      light-cone reduction: drop operations outside the output's causal past
4. Analyze     interaction graph → connected components; cut search (expectation only)
5. Estimate    each candidate engine prices the job: feasible?, peak bytes, work units, time counts
               (a repeated structure with new angles reuses its cached plan; expensive searches are pruned)
6. Plan        cheapest feasible candidate wins (predicted seconds under a time budget);
               dense then chooses CPU or CUDA, and the disk tier only if the state does not fit
7. Execute     run each region on its engine under a memory ledger; recombine exactly
8. Monitor     (only with a time budget) compare elapsed time with the budget at safe boundaries
9. Replan      (only on an overrun) observe → same planner → restart, or refuse
10. Report     Result, with explain()
```

Stages 1 to 6 happen before any simulation work. Stages 8 and 9 exist only for jobs that set a time budget; a job without one runs stage 7 unmonitored and pays nothing for them.

## A worked example

Take the 100-qubit brickwork circuit from the [MPS example](/examples/mps), asked for counts.

1. **Convert and validate.** 100 qubits, 100 classical bits, measurements at the end.
2. **Reduce.** Counts on every qubit: the light cone covers everything, so nothing is dropped.
3. **Analyze.** One connected component.
4. **Estimate.** The dense engine is refused (`checked_pow2: n = 100 exceeds the maximum of 62`). The stabilizer engine is refused (`operation 0 (ry) is not a Clifford gate`). MPS is feasible with a small peak memory.
5. **Plan.** `direct_mps` is the only feasible candidate, so it wins.
6. **Execute.** The MPS engine runs the gates and samples 32 shots.
7. **Report.** `exact=True`; 49,792 bytes actually used.

## What changes the path

| If… | Then… |
|---|---|
| the circuit splits into independent qubit groups | a `factorized` plan runs each group on its own engine |
| you request only an expectation value | Pauli, tensor-network and cut strategies become candidates |
| you request a statevector | only the dense engine is feasible |
| a memory budget is too small for every candidate | planning fails with `ResourceExceeded` before any allocation |
| a time budget is set | candidates are ranked by predicted seconds instead of work units, and the run is monitored and may be replanned |
| a CUDA device is usable and faster by the calibrated model | the dense region runs on it |
| a backend is forced (`RunOptions.backend`) | the dense region runs there or the job is refused; it never falls back |
| the dense state does not fit the memory budget, and you asked for counts or an expectation value | the [disk tier](/planning/memory-architecture#tiers-where-the-dense-state-lives) runs it from a scratch file |
| the same circuit structure runs again with new angles | the cached plan is reused |

## Next

[Analyze](/execution/analyze), [Estimate](/execution/estimate), [Plan](/execution/plan), [Execute](/execution/execute), then [Monitor](/runtime/monitoring) and [Replan](/runtime/replanning).
