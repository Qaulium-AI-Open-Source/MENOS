---
layout: doc
title: MENOS
aside: false
---

# MENOS

A quantum circuit simulation compiler and runtime. MENOS analyzes a circuit, estimates candidate execution strategies, chooses a plan, and runs it on the engine that fits the workload.

<div class="menos-cards">
  <a href="/get-started/installation"><strong>Get started</strong><span>Install MENOS and run your first circuit.</span></a>
  <a href="/build/menos-backend"><strong>Build with MENOS</strong><span>Backends, jobs, results, estimator and sampler.</span></a>
  <a href="/execution/how-menos-executes"><strong>How it works</strong><span>Analyze, estimate, plan, execute, replan.</span></a>
  <a href="/engines/dense"><strong>Simulation engines</strong><span>Dense, stabilizer, Pauli, MPS, tensor network, cutting.</span></a>
  <a href="/gpu/cuda-installation"><strong>GPU support</strong><span>CUDA execution, chosen by a calibrated cost model.</span></a>
  <a href="/reference/python-api"><strong>API reference</strong><span>Python API, configuration and exceptions.</span></a>
</div>

## What is MENOS?

MENOS is a high-performance quantum circuit simulator built as a **compiler and runtime system**. Instead of pushing every circuit through one fixed algorithm, it:

1. **Analyzes** the circuit (structure, entanglement, locality, what output you asked for),
2. **Estimates** what each candidate execution strategy would cost,
3. **Plans** by choosing the cheapest feasible strategy, and
4. **Executes** that plan, monitoring it against your time budget and replanning if reality disagrees with the estimate.

It is also a **Qiskit `BackendV2`** and **`EstimatorV2`**, so existing Qiskit code runs against it without modification.

<div class="menos-flow">
  <span>Install</span><span class="arrow">→</span>
  <span>Create a QuantumCircuit</span><span class="arrow">→</span>
  <span>Create MENOSBackend</span><span class="arrow">→</span>
  <span>Run</span><span class="arrow">→</span>
  <span>Inspect the result</span>
</div>

## Your first circuit

This is the complete program. It is an executable file in the repository's `examples/` directory and is run by the test suite.

<<< @/../examples/bell_state.py

Continue with [Installation](/get-started/installation), or go straight to [Your First Circuit](/get-started/first-circuit).

## Where to go next

| If you want to… | Read |
|---|---|
| Install MENOS and check that it works | [Installation](/get-started/installation), [Verify Installation](/get-started/verify-installation) |
| Run circuits from Qiskit | [MENOSBackend](/build/menos-backend), [Running Jobs](/build/running-jobs) |
| Compute expectation values | [Expectation Values](/build/expectation-values), [EstimatorV2](/build/estimator) |
| Understand how a plan is chosen | [How MENOS Executes](/execution/how-menos-executes), [Cost Model](/planning/cost-model) |
| Use a GPU | [CUDA Installation](/gpu/cuda-installation), [CPU vs CUDA](/gpu/cpu-vs-cuda) |
| Set a time or memory budget | [Time Budgets](/runtime/time-budgets), [Memory Constraints](/planning/memory-constraints) |
| Look something up | [Python API](/reference/python-api), [Supported Gates](/reference/supported-gates) |
| Fix a problem | [Troubleshooting](/troubleshooting/common-errors) |

::: info Status of this release
MENOS is at version 1.3.1. Phases 1 to 17 are complete, Phase 18 adds native Windows (x64, MSVC), and Phase 26 adds the CUDA backend there. The verification suites run on Linux x86-64 and on Windows x64; see [Supported Platforms](/introduction/supported-platforms) for exactly what that means.
:::
