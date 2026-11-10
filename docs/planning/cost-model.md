# Cost Model

The cost model is what makes candidates comparable. It answers two different questions with two different currencies.

| Question | Currency | Used for |
|---|---|---|
| *Which algorithm is cheapest for this job?* | **work units** | Choosing the strategy |
| *How long will it take, in seconds?* | **seconds**, from calibration | Time budgets, CPU vs CUDA |

## Work units

A work unit is an analytic count of basic operations, computed from structural facts about the job only: qubit count, the number of operations and how many qubits each touches, and which outputs are requested. It involves no measurement and is identical on every machine.

All engines count in the same spirit so that their numbers can be ranked against each other for *the same job*. The units are **not** claimed to be wall-clock equivalent: a stabilizer word operation and a dense complex multiply-add are different things. What the unit promises is a consistent ranking.

```text
Dense       2ⁿ  +  Σ_gates 2^(n+k)  +  output terms             (exponential in n)
Stabilizer  n  +  gates·n  +  2·n²·⌈n/64⌉  (+ sampling)       (polynomial in n)
```

Pauli propagation, MPS, tensor network and cut estimates follow the same principle with their own structural inputs: the worst-case term count, the rigorous bond-dimension bound, the contraction order's largest tensor, and the sum over fragment evaluations. All use **checked arithmetic**, so an unrepresentable quantity makes a candidate infeasible with a reason and never wraps.

## When the choice depends on timing

- **A job with no time budget:** the planner picks the algorithm by work units alone. Timing never enters that decision, which keeps the choice deterministic and identical across machines.
- **A job with a time budget** (since Phase 24): the planner picks the feasible candidate with the fewest predicted seconds, because the budget is a statement about seconds. Every candidate is then costed in full, with no branch-and-bound pruning, because pruning bounds work units and a candidate with more work units can be the faster one.

## Calibration

Seconds come from a one-time calibration, performed the first time a job is planned in a process and then cached:

| Measured | How | Used for |
|---|---|---|
| Streaming memory bandwidth | Timed large copy | A floor on any candidate's runtime |
| Work units per second (CPU) | A real circuit through the real dense engine, dividing its own work units by elapsed time | Converting work units to seconds |
| Host↔device transfer rates | Pageable host memory, the path the backend really takes | CUDA estimates |
| Kernel-launch overhead | A tiny kernel, including constant upload and synchronize | CUDA estimates |
| Device work rate | A real circuit through the CUDA engine, launch overhead subtracted | CUDA estimates |
| Seconds per time count, per CPU engine | Each engine runs a few small fixed circuits on one thread. A non-negative least-squares fit then gives what one count of each of the engine's time terms costs | CPU estimates |

Calibration measures this machine's speed on the code that will actually run, instead of assuming a constant. It runs the first time a job needs a calibrated profile in a process, and then costs microseconds.

That first run took 1.0–1.1 s on Windows and 1.8–2.5 s on Linux with the CUDA calibration included (development machine, Phase 24). Before the per-engine time model it was about 0.1 s.

An unbudgeted CPU-only job never runs it.

### Seconds for a CPU candidate

```text
seconds = max( Σ_engines Σ_c  time_count[engine][c] · seconds_per_count[engine][c] ,
               floor_bytes / memory_bandwidth )
```

Each CPU engine's estimate reports up to four **time counts**: structural counts of what its runtime is made of. For example:
- dense: passes, work units and shots;
- stabilizer: gates, tableau updates and word operations;
- MPS: SVDs, flops, tensor elements and sampled elements.

Calibration measures each count's cost per engine. The candidate is priced as the sum over the engines it uses, so each engine is timed in its own unit, and no constant relates one engine's work to another's. The second term is a physical floor: a candidate that materializes `floor_bytes` cannot finish sooner than that many bytes take to stream. The floor is the peak bytes, except for Pauli propagation, where it is the unbranched term list's bytes.

If an engine has no calibrated figures, its candidate falls back to `work_units / work_units_per_second` at the dense rate.

### Seconds for a CUDA candidate

```text
seconds = kernel_launches · launch_overhead
        + host_to_device_bytes / h2d_rate
        + work_units / device_work_rate
        + device_to_host_bytes / d2h_rate
```

The launch term is what makes small circuits slower on the GPU: each launch costs a fixed amount however little it does. A model without it would claim the GPU always wins.

## How accurate is it?

This was measured on the development machine (Phase 24, `docs/adr/0028` section 20.5).

**Method.**
- Each engine's actual over predicted seconds was measured on easy, representative and stress workloads.
- Every job carries a time budget.
- The prediction and the run are in the same process.
- There were three processes per platform.

"Typical error" is the median of |ln(actual ÷ predicted)|, expressed as a factor. "P90" is the 90th percentile of the same figure.

::: warning Disclaimer
These figures come from benchmarks run on the developer's own machine. Treat them as indicative; on your hardware they will differ.
:::

| Engine | Linux: typical error | Linux: P90 | Windows: typical error | Windows: P90 |
|---|---|---|---|---|
| Dense (CPU) | 1.16× | 1.37× | 1.69× | 3.57× |
| Dense (CUDA) | 1.14× | 1.21× | not built | not built |
| Density matrix | 1.03× | 1.36× | 1.17× | 2.78× |
| Factorized | 1.11× | 1.46× | 1.53× | 2.19× |
| Tensor network | 1.10× | 1.17× | 1.14× | 1.25× |
| Trajectory | 1.24× | 1.42× | 1.26× | 1.65× |
| MPS | 1.55× | 1.78× | 2.12× | 3.09× |
| Pauli propagation | 1.64× | 2.15× | 2.26× | 2.95× |
| Stabilizer | 1.70× | 3.09× | 2.04× | 3.99× |
| Cut and recombine | not measured | not measured | not measured | not measured |

Before the per-engine time model, the typical error on Linux was 28× for MPS, 18× for stabilizer, 11× for tensor network and 7.2× for Pauli propagation.

What the table says:
- **Every engine is within a factor of about two typically, and of four at the 90th percentile.** On Windows the MPS, Pauli and stabilizer medians are slightly over two.
- **Pauli propagation is optimistic,** more so with each non-Clifford rotation. The time model charges a run in which no term branches.
- **The stabilizer model is optimistic on 1000-qubit random Clifford circuits and pessimistic on 2000-qubit graph states.** Calibration uses tableaux of at most 512 qubits.
- **Dense on Windows is pessimistic at 20–22 qubits.** Calibration runs on one thread, and large jobs run on every thread.
- **Cut and recombine has no measured rows.** Under the seconds ranking, the planner chose a direct plan on every cut workload.

The disk tier's prediction (since Phase 24): the work at the dense rate measured on every hardware thread, plus the bytes written at the scratch directory's write bandwidth and the bytes read at its read bandwidth. Measured over predicted on the Phase 23 gate circuits:
- 0.81 at 30 qubits on Linux;
- 1.17 at 31 qubits on Windows.

Calibration itself varies about 1.5× between processes on Windows, and under 5% on Linux.

These facts are why a time budget is enforced while the run executes, not just at plan time. See [Time Budgets](/runtime/time-budgets) and [Monitoring](/runtime/monitoring).

## Memory is exact

By contrast, predicted peak bytes equalled actual peak bytes in every workload measured, because each engine allocates exactly what it estimates.

## Related

[Execution Plans](/planning/execution-plans) · [Backend Selection](/planning/backend-selection) · [Estimate](/execution/estimate)
