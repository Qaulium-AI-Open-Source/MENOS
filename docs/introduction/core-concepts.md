# Core Concepts

A short glossary of the ideas the rest of the documentation builds on.

## Circuit, KIR, operation

A circuit is an ordered list of *operations* over numbered qubits and classical bits. An operation is a named **gate**, a **unitary** given as an explicit matrix (up to five qubits), or a **measurement**. See [KIR](/concepts/kir).

## Job and output request

A *job* is a circuit plus what you want out of it: measurement **counts** (`shots > 0`), the exact **statevector**, or an **expectation value** of an observable. A job must request at least one output. The request shapes the plan: only the dense engine can produce a statevector, and only some engines can produce an expectation value.

## Engine and backend

An **engine** is an algorithm (dense, stabilizer, Pauli propagation, MPS, tensor network, density matrix, trajectory). A **backend** is the hardware it runs on (CPU or CUDA). They are chosen separately: the algorithm by work units (or, under a time budget, by predicted seconds), then the backend by calibrated seconds. You can also force a backend; a forced backend never falls back.

## Tier

Where a dense state lives: **in memory** (RAM, or VRAM on CUDA) or on **disk**, in a scratch file applied in passes. The planner uses the disk tier only when the in-memory state does not fit the memory budget, never for speed, and its counts and expectation values are bit for bit the in-memory engine's. See [Memory Architecture](/planning/memory-architecture#tiers-where-the-dense-state-lives).

## Plan, strategy, region

A **plan** is the fully decided way to run a job. Its **strategy** is one of `direct_dense`, `direct_stabilizer`, `direct_pauli`, `direct_mps`, `direct_tensor_network`, `factorized`, or `cut_and_recombine`. A **region** is a set of qubits run together on one engine. A direct plan has one region; a factorized or cut plan has several.

## Candidate

Every strategy the planner considered, feasible or not. The plan report lists all of them. See [Plan Explanation](/planning/explain).

## Work unit

The cost model's shared currency. A work unit is comparable across engines, which is what lets the planner choose between them. It is *not* seconds; seconds come from calibration. See [Cost Model](/planning/cost-model).

## Time count and calibration

Each CPU engine's estimate also reports **time counts**: what its runtime is made of (passes, gates, tableau updates, SVDs, term updates, shots). **Calibration** measures what one count costs on this machine, engine by engine, the first time a process needs it. Seconds are counts times those costs. See [Cost Model](/planning/cost-model).

## Pruned candidate

A tensor-network, cut or per-region candidate whose costing was stopped because it had already passed a cheaper feasible candidate. It could not have won, so the plan is the same as if it had been costed in full. See [How Planning Stays Fast](/planning/planner-performance).

## Feasible and infeasible

A candidate is **infeasible** if it cannot serve the request: it would exceed the memory budget, it does not support a gate or an output, or it would exceed the time budget. Infeasible candidates carry a reason string.

## Budget

A constraint on a run. A **memory budget** bounds the bytes a plan may need. A **time budget** bounds the wall-clock seconds of *one execution attempt*. Both are checked at plan time; the time budget is also checked during execution. See [Memory Constraints](/planning/memory-constraints) and [Time Budgets](/runtime/time-budgets).

## Memory ledger and peaks

While a job runs, a ledger adds up the buffers that grow with the problem and records the peak in host memory, device memory and on disk: `peak_host_bytes`, `peak_device_bytes`, `peak_disk_bytes`. Every estimate is an upper bound on them. See [Memory Architecture](/planning/memory-architecture).

## Fused block

The dense engines group neighbouring gates into **blocks** and sweep the state once per block instead of once per gate (gate fusion, on by default). A gate joins a block only when one qubit set contains the other, so no pass gets wider. See [Dense](/engines/dense).

## Circuit builder and parameters

`CircuitBuilder` writes a circuit with named gate methods, registers and composition. A `Parameter` is a named angle left open until `bind` gives it a value; a rebinding loop reuses its plan. See [Writing Circuits](/build/circuit-builder).

## Exact

A run is exact if no approximation was made. Every engine reachable from Python is exact, and `Result.exact` reports it. See [Exactness](/concepts/exactness).

## Replan

When a run overruns its time budget it is stopped and planned again using what was observed. The abandoned attempt is reported. See [Replanning](/runtime/replanning).

## Deterministic

The same circuit, options and seed always produce the same counts and the same plan. See [Determinism](/concepts/determinism).
