# Architecture

MENOS is a pipeline. A circuit enters through one of several front ends, is planned once, and leaves as a result with a report of how it ran. Every stage in between is a separate component with one job.

```text
 Qiskit QuantumCircuit   PennyLane tape   OpenQASM 2 / 3   CircuitBuilder   Circuit.append
        │                      │                │                │               │
        └──────────── converters / importers / bind ─────────────┴───────────────┘
                                         ↓
                                        KIR            the circuit in MENOS's own representation, validated
                                         ↓
                               Light-cone reduction    drops operations outside the output's causal past
                                         ↓
                                     Plan cache        a repeated structure with new angles reuses its plan
                                         ↓
                                      Analysis         interaction graph, connected components, cut search
                                         ↓
                                     Cost model        each candidate: feasible?, peak bytes (host, device, disk),
                                                       work units, time counts; expensive searches pruned
                                         ↓
                                      Planner          cheapest feasible candidate (work units, or predicted seconds
                                                       under a time budget), then CPU or CUDA, then the tier
                                         ↓
                                   Execution plan      strategy, regions, engine / backend / tier per region
                                         ↓
                                      Runtime          runs each region under a memory ledger, recombines, reports
                                         ↓
     ┌──────────────┬──────────────┬─────────────────┬──────────────────────┬────────────────────┐
     ↓              ↓              ↓                 ↓                      ↓                    ↓
 Dense (CPU)   Dense (CUDA)   Dense (disk tier)  Stabilizer / Pauli   MPS / Tensor network   Density matrix /
 fused, AVX2   pooled VRAM    scratch file                            / Cut                  Trajectory
                                         ↓
                                     Monitoring        compares elapsed time with the time budget at safe boundaries
                                         ↓
                                     Replanning        on an overrun: observe, replan with the same planner, restart
```

## The components

**Front ends.** Every way in produces the same native `Circuit`:
- `menos.qiskit.convert` lowers a Qiskit `QuantumCircuit`, used by `MENOSBackend`, `MENOSEstimator` and `MENOSSampler`;
- the PennyLane device `menos.qubit` lowers a tape;
- `menos.from_qasm3` and the command line read OpenQASM 3 (static subset) and OpenQASM 2;
- `CircuitBuilder` writes a circuit by hand, with parameters bound by `bind` ([Writing Circuits](/build/circuit-builder));
- `Circuit.append` builds one operation at a time.

Gates in the [supported table](/reference/supported-gates) map to opcodes, other gates with a matrix on up to five qubits become dense unitaries, and anything else is refused with an explicit error rather than approximated.

**KIR.** The [Kernel Intermediate Representation](/concepts/kir) is a flat list of gate, unitary, measurement and noise operations over numbered qubits and classical bits. It is validated once, at the boundary, so nothing downstream needs to re-check it.

**Plan cache.** A job whose circuit has the same structure and options as an earlier one, differing only in gate angles, reuses that job's Direct, in-memory plan. A variational loop therefore plans once. See [How Planning Stays Fast](/planning/planner-performance#the-plan-cache).

**Analysis.** Builds the interaction graph of the circuit and its connected components, runs [light-cone reduction](/observables/light-cone-reduction), and searches for [wire cuts](/engines/cut-and-recombine).

**Cost model.** Each engine reports, for a given request, whether it can serve it and what it would cost:
- **peak bytes**, an upper bound on what it holds, per memory space;
- **work units**, a currency shared by all engines;
- **time counts**, the structural counts its runtime is made of (passes, gates, SVDs, term updates…), which calibration turns into seconds per engine.

Arithmetic is checked, so an unrepresentable estimate is reported as infeasible and never wraps around. Expensive searches (tensor network, cut, per-region choice) stop as soon as they cannot beat a cheaper feasible candidate (branch-and-bound). See [Cost Model](/planning/cost-model) and [How Planning Stays Fast](/planning/planner-performance).

**Planner.** `plan_job` is the only planner. It:
1. enumerates candidates in a fixed order and discards infeasible ones;
2. picks the fewest work units, or, for a job with a time budget, the fewest predicted seconds, with ties going to the earlier candidate;
3. for dense regions, chooses CPU or CUDA by calibrated seconds;
4. puts a dense state on the disk tier only when it does not fit in memory.

See [Plan](/execution/plan) and [Backend Selection](/planning/backend-selection).

**Runtime.** Executes the plan. A multi-region plan runs its regions one at a time and recombines them exactly. Every attempt runs under a **memory ledger** that records the peak bytes in host memory, device memory and on disk. See [How MENOS Executes](/execution/how-menos-executes) and [Memory Architecture](/planning/memory-architecture).

**Engines.** An *engine* is an algorithm, a *backend* is the hardware it runs on, and a *tier* is where its state lives:
- **Dense** is the only algorithm with more than one backend or tier: CPU, CUDA, and the disk tier.
- On the CPU it fuses neighbouring gates into blocks (on by default) and applies them with AVX2 kernels where the processor has them.
- Every path is bit for bit the generic kernel on every thread count.
- On CUDA the state comes from MENOS's own memory pool.

The other engines (stabilizer, Pauli propagation, MPS, tensor network, cut and recombine, density matrix, trajectory) run on the CPU in memory. See [Simulation Engines](/engines/dense).

**Monitoring and replanning.** When a job carries a time budget, engines report safe boundaries to a monitor. Crossing the budget abandons the attempt and records what the engine (or, in a factorized plan, each region) actually cost. It then calls the same planner again. See [Runtime](/runtime/monitoring).

## Dependency direction

The C++ layers depend on one another in one direction only, and the build enforces it:

```text
api → runtime → engines → kernels
api → planner → cost → analysis → ir → common
```

Nothing lower knows about anything higher. In particular the runtime never plans and never interprets the time-budget monitor; both decisions belong to the layer above it.

## Source layout

| Directory | What it holds |
|---|---|
| `include/menos/` | The public C++ headers, including `builder.h` (the circuit builder) |
| `src/ir/` | KIR, validation, the gate table, the builder, parameters, exact inverses, the gate-fusion passes |
| `src/qasm2/`, `src/qasm3/` | The OpenQASM importers and the OpenQASM 3 exporter |
| `src/analysis/` | Interaction graph, components, light cone, cuts, the bitset elimination search |
| `src/cost/`, `src/plan/` | The cost model, hardware profile and calibration, the time model, the planner, the plan cache |
| `src/api/` | The public entry points, and the replanning loop around `plan_job` and the runtime |
| `src/runtime/` | Execution of a plan and the exact recombination of its regions |
| `src/common/` | Shared pieces, including the memory ledger, checked arithmetic and the SVD |
| `src/engines/` | One directory per engine; `dense_cuda` and `dense_tiered` (the disk tier) beside `dense` |
| `src/kernels/` | CPU kernels (scalar and AVX2) and CUDA kernels, with the device memory pool |
| `src/mitigation/` | Readout-error mitigation |

## Where the Python package fits

```text
menos.qiskit     MENOSBackend, MENOSEstimator, MENOSSampler, MENOSJob, converter, target
menos.pennylane  MENOSQubitDevice (registered as "menos.qubit")
menos            Circuit, CircuitBuilder, Parameter, Opcode, RunOptions, Result, Backend, Tier,
                 run, run_batch, from_qasm3, release_device_memory, readout mitigation, exceptions
menos._core      the nanobind extension that wraps the C++ library (internal)
```

`menos.qiskit` and `menos.pennylane` contain no simulation logic. They convert, call `menos.run`, and adapt the result. `CircuitBuilder` and `Parameter` are bindings of the C++ types, so the two languages behave identically.
