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

## Why the choice does not depend on timing

The planner picks the algorithm by work units alone. Timing never enters that decision, which keeps algorithm choice deterministic and identical across machines.

## Calibration

Seconds come from a one-time calibration, performed the first time a job is planned in a process and then cached:

| Measured | How | Used for |
|---|---|---|
| Streaming memory bandwidth | Timed large copy | A floor on any candidate's runtime |
| Work units per second (CPU) | A real circuit through the real dense engine, dividing its own work units by elapsed time | Converting work units to seconds |
| Host↔device transfer rates | Pageable host memory, the path the backend really takes | CUDA estimates |
| Kernel-launch overhead | A tiny kernel, including constant upload and synchronize | CUDA estimates |
| Device work rate | A real circuit through the CUDA engine, launch overhead subtracted | CUDA estimates |

Calibration measures this machine's speed on the code that will actually run, instead of assuming a constant. It costs on the order of 0.1 s the first time in a process (measured at roughly 86 to 99 ms on the development machine) and microseconds thereafter.

### Seconds for a CPU candidate

```text
seconds = max( work_units / work_units_per_second ,  peak_bytes / memory_bandwidth )
```

The first term comes from calibration on a cache-resident circuit, so it is optimistic for very large states. The second is a physical floor: a candidate that materializes `peak_bytes` cannot finish sooner than that many bytes take to stream.

### Seconds for a CUDA candidate

```text
seconds = kernel_launches · launch_overhead
        + host_to_device_bytes / h2d_rate
        + work_units / device_work_rate
        + device_to_host_bytes / d2h_rate
```

The launch term is what makes small circuits slower on the GPU: each launch costs a fixed amount however little it does. A model without it would claim the GPU always wins.

## How accurate is it?

Measured on the development machine (Phase 24, `docs/adr/0028` sections 16.10 and 16.12): actual over predicted seconds for every engine. Each engine ran easy, representative and stress workloads, with the prediction and the run in the same process, in three processes per platform. The table is Linux (WSL, GCC), the reproducible platform. On Windows the same models read further off for dense, density-matrix and factorized jobs, because the calibration there is slower and varies more between processes.

::: warning Disclaimer
These figures come from benchmarks run on the developer's own machine. Treat them as indicative; on your hardware they will differ.
:::

| Engine | Median actual ÷ predicted | Range | Typical error |
|---|---|---|---|
| Dense (CPU) | 1.2 | 0.58 to 1.75 | 1.3× |
| Dense (CUDA) | 0.92 | 0.80 to 0.98 | 1.1× |
| Factorized | 1.5 | 0.91 to 2.05 | 1.5× |
| Trajectory | 1.5 | 1.47 to 1.59 | 1.5× |
| Density matrix | 0.40 | 0.37 to 1.21 | 2.5× |
| Cut and recombine | 3.8 | 2.6 to 5.1 | 3.7× |
| Tensor network | 4.9 | 4.6 to 5.1 | 4.9× |
| Pauli propagation | 0.40 | 0.013 to 9.3 | 7.4× |
| **Stabilizer** | **10** | **0.79 to 30** | **7.9×** |
| **MPS** | **21** | **0.78 to 58** | **12×** |

"Typical error" is the median of |ln(actual ÷ predicted)| expressed as a factor.

What the table says:
- **Dense, CUDA dense, factorized and trajectory** predictions are within a factor of about two.
- **The MPS model is optimistic,** worst on low-entanglement circuits, where every bond is small. Its per-gate cost is dominated by a fixed cost of each SVD and its allocations, which the χ³ term does not see.
- **The stabilizer model is optimistic on long Clifford circuits** and close on graph states.
- **The Pauli model charges the worst-case branching.** Its prediction grows about 4× per non-Clifford rotation while the measured time stays flat.

These models are scheduled for a follow-on phase with targets set from this table.

The disk tier's prediction (since Phase 24): the work at the dense rate measured on every hardware thread, plus the bytes written at the scratch directory's write bandwidth and the bytes read at its read bandwidth. Measured over predicted on the Phase 23 gate circuits:
- 0.81 at 30 qubits on Linux;
- 1.17 at 31 qubits on Windows.

Calibration itself varies about 1.5× between processes on Windows, and under 5% on Linux.

These facts are why a time budget is enforced while the run executes, not just at plan time. See [Time Budgets](/runtime/time-budgets) and [Monitoring](/runtime/monitoring).

## Memory is exact

By contrast, predicted peak bytes equalled actual peak bytes in every workload measured, because each engine allocates exactly what it estimates.

## Related

[Execution Plans](/planning/execution-plans) · [Backend Selection](/planning/backend-selection) · [Estimate](/execution/estimate)
