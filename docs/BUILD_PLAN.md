# MENOS Build Plan

## Current status

| Phase | Name | Status |
|---|---|---|
| 0 | Foundation | Complete |
| 1 | KIR, dense CPU engine, Plan IR, runtime, result and explain | Complete |
| 2 | Python package and Qiskit BackendV2 | Complete |
| 3 | Factorization, work-unit cost model, first real planner decision | Complete |
| 4 | Stabilizer engine and engine interface | Complete |
| 5 | Observables, light-cone reduction, Pauli propagation | Complete |
| 6 | MPS engine and accuracy policy | Complete |
| 7 | CLI and OpenQASM 2 | Complete |
| 8 | Tensor network engine | Closed; do not reopen |
| 9 | Cut and recombine | Complete; verified |
| 10 | First GPU backend, primitive API, calibration | **Complete — acceptance gate passed 2026-09-30** |
| 11 | Runtime monitoring and replanning | **COMPLETE — officially closed; acceptance gate passed 2026-09-30 (implementation gate `322eca5`, documentation closure `9a0f0f7`)** |
| 12 | Noise channels and exact noisy simulation (density matrix) | **COMPLETE — acceptance gate passed 2026-10-02 (final code commit `96a7cd0`; criterion 10 re-verified on `e546581`)** |
| 13 | Noisy trajectory simulation | **COMPLETE — acceptance gate passed 2026-10-03 (evidence: `docs/adr/0017` section 10; `scripts/verify.sh` exit 0 on `847cb44`)** |
| 14 | Error mitigation and batch execution | **COMPLETE — acceptance gate passed 2026-10-07 (evidence: `docs/adr/0018` section 10; `scripts/verify.sh` exit 0 on `75ad45f`)** |
| 15 | Dense CPU engine performance | **COMPLETE — acceptance gate passed 2026-10-08 (evidence: `docs/adr/0019` sections 10.9–10.10; `scripts/verify.sh` exit 0 on `a01052a`)** |
| 16 | Specialised CUDA kernels | **COMPLETE — acceptance gate passed 2026-10-09 (evidence: `docs/adr/0020` section 13; merged to `main` in `c6b9c87`)** |
| 17 | API surface: recorded deferrals | **COMPLETE — acceptance gate passed 2026-10-10 (evidence: `docs/adr/0021` section 11; `scripts/verify.sh` exit 0 on `1041893`)** |
| 18 | Native Windows (MSVC) | Started 2026-10-13 — ADR `docs/adr/0022` |
| 19 | OpenQASM 3 import (frontend) | **COMPLETE — acceptance gate passed 2026-10-16 (evidence: `docs/adr/0023` section 10; `scripts/verify.sh` exit 0 on `bb720d8`)** |
| 20 | PennyLane device (frontend) | **COMPLETE — gate recorded passed 2026-10-18 (evidence: `docs/adr/0024` section 10; `scripts/verify.sh` exit 0 on `5da898e`)** |
| 21 | Memory accounting and copy removal | **COMPLETE — acceptance gate passed 2026-11-02 on the owner's ruling (evidence: `docs/adr/0025` sections 10.5–10.6; `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0 on `9b8833c`)** |
| 22 | CUDA memory efficiency | **COMPLETE — acceptance gate accepted 2026-11-02 (evidence: `docs/adr/0026` sections 10.1–10.6; `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0 on `e3619b3`)** |
| 23 | Tiered memory across VRAM, RAM and disk | **COMPLETE — acceptance gate passed 2026-11-05 on the owner's ruling (evidence: `docs/adr/0027` sections 10.1–10.4; `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0 on `a20c58d`)** |
| 24 | Planner performance; planner accuracy, planning cost and dense-engine throughput | Started 2026-11-03 on a contributor's fork as "Phase 21" (steps 0–2 done there, `docs/adr/0028`); merged with Phases 21–23 on 2026-11-05 (`docs/adr/0028` section 15); gate OPEN; the remaining items are a proposal |
| 25 | Stim engine for noisy Clifford circuits (backend) | Proposed (2026-10-07) — not approved |
| 26 | Cirq frontend | Proposed (2026-10-07) — not approved |
| 27 | Adjoint differentiation | Proposed (2026-10-07) — not approved |
| 28 | Dynamic circuits | Proposed (2026-10-07) — not approved |
| 29 | NVIDIA cuQuantum engines (backend) | Proposed (2026-10-07) — not approved |
| 30 | Further frontends and engines (candidates) | Proposed (2026-10-07) — not approved |

**Current position.** Phase 11 is COMPLETE and officially closed: `322eca5` is the final implementation and gate commit (`scripts/verify.sh` exit 0) and `9a0f0f7` is the documentation closure commit, on branch `phase-11-monitoring-replanning`. Phase 12 is NOT STARTED. On 2026-10-01 the owner set the scope of Phases 12–14: noise models, delivered in three phases (exact density-matrix simulation, then trajectories, then error mitigation). The specifications below are the plan of record. The owner recorded the Phase 12 decisions on 2026-10-01 and the branch `phase-12-noise-density-matrix` exists. The owner approved the Phase 12 plan and ADR 0016 the same day. Phase 12 is implemented on branch `phase-12-noise-density-matrix` (final code commit `96a7cd0`) and awaits the owner's gate decision; see the Phase 12 verification record. On 2026-10-02 the owner made that decision: Phase 12 closes once criterion 10 is re-verified with GPU tests that actually execute. CUDA 12.x support (commit `e546581`, `docs/adr/0014` section 17) made that possible on this machine, the re-verification passed, and **Phase 12 is COMPLETE** (see the Phase 12 gate record). Phase 13 followed: the owner answered ADR 0017's open questions and approved the plan the same day, Phase 13 was implemented and the owner accepted its gate on 2026-10-03: **Phase 13 is COMPLETE**. Phase 14 is not started. On 2026-10-05 the owner added the performance roadmap (Phases 15–17), which comes after Phase 14 and does not renumber it. Phases 14–17 are COMPLETE (gates accepted 2026-10-07, 2026-10-08, 2026-10-09 and 2026-10-10). On 2026-10-13 the owner started **Phase 18, native Windows (MSVC)**, as the next phase, and renumbered the proposed interoperability roadmap from Phases 18–25 to Phases 19–26 with its content unchanged. On 2026-10-14 the owner started **Phase 19, OpenQASM 3 import**, in parallel with Phase 18, and accepted its gate on 2026-10-16: **Phase 19 is COMPLETE**. On 2026-10-16 the owner started **Phase 20, PennyLane device**, whose gate was recorded passed on 2026-10-18: **Phase 20 is COMPLETE**. The owner accepted the Phase 18 gate on 2026-10-18: **Phase 18 is COMPLETE**. The same day the owner placed the memory roadmap (Phases 21–23: memory accounting, CUDA memory efficiency, tiered memory) next, and renumbered the remaining proposed interoperability phases to 24–29. On 2026-10-20 the owner started **Phase 21, memory accounting and copy removal** (`docs/adr/0025`); it is implemented through `72d8f27`. On 2026-10-31 the owner deferred the final verification and the gate decision. The verification ran on `9b8833c` on 2026-11-02, and the owner passed the gate the same day, ruling that the criterion 4 benchmark misses do not count against it: **Phase 21 is COMPLETE**. On 2026-10-31 the owner also started **Phase 22, CUDA memory efficiency** (`docs/adr/0026`), on branch `phase-22-cuda-memory-efficiency` from Phase 21's head; `phase-21-memory-accounting` was merged into it on 2026-11-02. Its verification ran on `e3619b3` and the owner accepted its gate the same day: **Phase 22 is COMPLETE**. The same day the owner started **Phase 23, tiered memory across VRAM, RAM and disk** (`docs/adr/0027`), on branch `phase-23-tiered-memory` from Phase 22's head. It was implemented, measured and verified on 2026-11-04 (`scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0 on `a20c58d`), and on 2026-11-05 the owner passed its gate, ruling that the one criterion 3 miss in WSL does not count against it: **Phase 23 is COMPLETE**. Meanwhile, on 2026-11-03, a contributor's fork started planner-performance work under the name "Phase 21" and brought it to `main` through pull requests #12 and #13; on 2026-11-05 that line and the completed memory roadmap were merged (branch `integrate-memory-roadmap`), the planner work was recorded as the start of **Phase 24** with its ADR renumbered to `docs/adr/0028`, and Phases 21–23 kept their numbers and ADRs 0025–0027. Phase 24's gate is OPEN; the merge's decisions and verification are `docs/adr/0028` section 15.

### Phase 10 frozen record

Verified environment: CUDA 13.4.92, GCC 15.2, NVIDIA driver 595.91.07, RTX 3050 Laptop GPU, compute capability 8.6, approximately 3770 MiB device memory. CPU dev/release: 466/466. CUDA dev/release: 483/483. Python: 69 passed, 1 deselected. End-to-end CUDA crossover: n=12. `scripts/verify.sh`: exit 0.

Deferred from Phase 10:
1. heterogeneous per-region CPU/CUDA backend selection;
2. specialized 1Q/2Q CUDA kernels;
3. Python `time_budget_seconds`;
4. explicit Python CUDA selection;
5. multi-GPU, ROCm, other architectures and unscheduled platform work.

### Phase 11 frozen record

Verified on the Phase 11 implementation and gate commit `322eca5` (documentation closure commit `9a0f0f7`): CPU dev/release 496/496. CUDA dev/release 528/528 (CUDA 13.4.92, GCC 15.2, driver 595.91.07, RTX 3050 Laptop GPU, compute capability 8.6). Python: 74 passed, 1 deselected. `scripts/verify.sh`: exit 0. Design, measurements and consequences: `docs/adr/0015`.

Owner decisions confirmed at closure:
1. `time_budget_seconds` bounds **one execution attempt**, not the total job. A job that replans can take several times its budget in total; abandoned attempts are reported, not hidden.
2. If no feasible plan remains after an overrun, MENOS throws `ResourceExceeded`, even if the abandoned attempt was near completion.
3. The CUDA-overrun → CPU recovery fix (`361ab12`) is part of Phase 11.

Deferred from Phase 11 (all intentional):
1. state migration/checkpointing (restart measured cheap for executions of about a second or more: one replan 0.05–17 ms mean, 85 ms worst sample; not small relative to millisecond-scale jobs, `docs/adr/0015` section 10; requires measurement plus an ADR);
2. observed-runtime floors for `Factorized`/`CutAndRecombine` plans (elapsed time is not attributed per region, so such an overrun is refused); *lifted for `Factorized` in Phase 24 (per-region and plan floors, `docs/adr/0028` sections 16.4 and 16.8); a `CutAndRecombine` overrun is still refused*;
3. per-operation boundaries in the tensor-network engine (only per observable term);
4. per-shot monitoring of the dense sampler;
5. a CLI time-budget flag;
6. heterogeneous per-region CPU/CUDA backend selection (carried from Phase 10);
7. specialized 1Q/2Q CUDA kernels (carried from Phase 10);
8. explicit Python CUDA selection (carried from Phase 10);
9. multi-GPU, ROCm, other architectures and unscheduled platform work.

Phase 10 deferral 3 (Python `time_budget_seconds`) was **lifted** in Phase 11 (`f9f908a`); the Qiskit `BackendV2` adapter still does not set it.

Acceptance evidence by gate criterion:

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Prediction monitored during real execution | Pass | `ReplanTest.TheMonitorFiresOnRealHardware…` |
| 2 | Threshold deterministically triggers a replan | Pass | `ReplanTest.AnInjectedClockCrossesTheBudget…` |
| 3 | Replanning uses the existing planner | Pass | `plan_job` is the only planner; `src/runtime/` has no `plan_job` call |
| 4 | New plan uses observed runtime information | Pass | `RuntimeObservationTest.*`; mutation check: disabling the floor fails 10 of 22 tests |
| 5 | Restart/re-execution correctness | Pass | `ReplanTest.ARestartedRunIsIdenticalToRunningTheNewPlanFromCold`; `CudaExecutionMonitorTest.AGpuOverrunIsAbandonedOnTheDeviceAndRestartedOnTheCpu` |
| 6 | No migration without measured justification | Pass | none implemented; `docs/adr/0015` section 10 |
| 7 | `explain()` reports all seven items | Pass | `ReplanTest.ExplainReports…` |
| 8 | No-replan `explain()` unchanged | Pass | `ReplanTest.ARunWithNoReplanEmitsNoReplanTextAtAll`; existing explain tests and CLI goldens |
| 9 | CPU and CUDA verification green | Pass | 496/496 and 528/528 |
| 10 | `scripts/verify.sh` passes | Pass | exit 0 on `322eca5` |
| 11 | No later-phase features | Pass | see deferrals above |

## Phase 11: Runtime monitoring and replanning (COMPLETE — officially closed; acceptance gate passed 2026-09-30)

### Goal

Observe runtime quantities predicted by the plan, detect a documented threshold crossing, safely stop/restart when required, and replan using observed information. `explain()` reports predicted versus actual quantities and every replan with its reason.

### Required behavior

1. Identify quantities already predicted by the current plan/cost model that can be measured during execution.
2. Define a documented, testable threshold for at least one real runtime quantity.
3. On threshold crossing, stop at a safe execution boundary and request a new plan.
4. Replanning must use the existing planner/cost model; do not create a second planner.
5. Preserve CPU/CUDA selection and strict time-budget semantics.
6. `explain()` reports original prediction, observed value, threshold, reason, old plan, new plan and replan sequence.
7. No-replan behavior preserves existing explain output.
8. Do not implement checkpointing/state migration unless measurement demonstrates restart is materially costly; if justified, create an ADR first.
9. Correctness and exactness/accuracy semantics must remain intact.

### Phase 11 acceptance gate

1. At least one existing plan prediction is monitored during real execution.
2. A documented threshold deterministically triggers a replan in a controlled test.
3. Replanning uses the existing planner/cost model.
4. The new plan uses observed runtime information.
5. Restart/re-execution correctness is demonstrated.
6. No migration/checkpointing without measured justification and ADR.
7. `explain()` reports prediction, observation, threshold, reason, old plan, new plan and sequence.
8. No-replan explain behavior remains unchanged.
9. Existing CPU and CUDA verification remains green.
10. `scripts/verify.sh` passes.
11. No later-phase features are implemented.

### Phase 11 process

Before implementation Claude must:
- inspect runtime, planner, Plan IR, explain and CPU/CUDA selection;
- identify quantities already predicted by the current plan;
- measure at least one realistic workload and restart cost;
- draft an ADR for the monitoring/replanning mechanism;
- enter plan mode with exact files, API changes, telemetry model, threshold semantics, safe boundary, replan path, explain changes, tests, risks and ambiguities;
- wait for owner approval.

Prefer restart/re-execution initially. Do not build speculative migration infrastructure.

## Noise roadmap (Phases 12–14)

The owner set this roadmap on 2026-10-01. Noise support comes in three phases, and each one is validated against the one before it. This follows how the project was built: an exact reference comes first, and cheaper approximate methods are checked against it.

1. **Phase 12** adds noise to the circuit representation and simulates it **exactly** with a density matrix. The result is exact for the requested noise model: nothing is truncated or sampled. This is a small-qubit tool, but it is the correctness reference for everything that follows.
2. **Phase 13** adds **trajectory** (Monte-Carlo) simulation, validated against Phase 12. Trajectory results carry statistical error and are never reported as exact. The planner chooses between density-matrix and trajectory plans by cost.
3. **Phase 14** adds **error mitigation**. Mitigation runs many circuits and combines their results, so it needs batch execution that MENOS does not have yet.

Invariants every noise phase must preserve:
- **Never silently approximate.** A noisy result is either exact for its stated model or flagged, with its error reported. A noise model is never applied unless the caller asked for one.
- **A job with no noise model behaves exactly as before.** Its plan, results and `explain()` output are unchanged.
- **`plan_job` is the only planner.** Noisy candidates compete through the existing cost model and its feasibility rules.
- **Determinism.** The same circuit, noise model, options and seed give the same results, including across a replan.
- Memory and time budgets are enforced for noisy plans exactly as they are for noiseless ones.

## Phase 12: Noise channels and exact noisy simulation (COMPLETE — acceptance gate passed 2026-10-02)

### Phase 12 verification record

`scripts/verify.sh` was run on `96a7cd0` in WSL Ubuntu 26.04 (GCC 15.2, CMake 4.2.3, clang 21.1.8, uv-managed CPython 3.14.7, CUDA toolkit 13.3.1, RTX 4060 Laptop GPU, Windows driver 566.14). Results:

- **Pass:** format, clang-tidy, dev **551/551**, release **551/551**, benchmarks, Python **195 passed / 1 deselected**, wheel smoke test, CUDA clang-tidy and builds, cuda-release **584/584**. The docs build was run on Windows and passes.
- **Fails:** the `cuda-dev` test stage. All 93 failing processes passed their tests, then LeakSanitizer reported a leak inside WSL's `libcuda.so` (`cuDriverGetVersion`), with no MENOS frame. The pre-Phase-12 commit `b77133a` fails identically.
- **Not verified:** no GPU test could execute, because the Windows driver supports CUDA 12.7 and the toolkit is CUDA 13. Phase 12 adds no CUDA kernels.

Per-criterion evidence: `docs/adr/0016` section 10. Measured planner accuracy for the new engine: section 5.

### Phase 12 gate record (2026-10-02)

The owner's decision: close Phase 12 once criterion 10 is re-verified with GPU tests that execute. Two changes made that possible on the same WSL machine, neither touching Phase 12 code (no file under `src/`, `include/`, `tests/`, `python/`, `cli/` or `benchmarks/` changed between `96a7cd0` and `e546581`):

- CUDA 12.0+ support, with `cuda12-dev` / `cuda12-release` presets on CUDA 12.4, which driver 566.14 runs (`docs/adr/0014` section 17).
- An owner-approved LeakSanitizer suppression for the WSL driver shim only, `scripts/lsan-wsl-libcuda.supp`, matched by full path (`docs/adr/0014` section 17).

`scripts/verify.sh` on `e546581`: **all stages passed, exit 0**. dev 551/551, release 551/551, cuda-dev 584/584, cuda-release 584/584, cuda12-dev 584/584, cuda12-release 584/584. Python, wheel and benchmark stages pass. The docs build was skipped in WSL (no Node). In the CUDA 12 builds the GPU tests execute: `CpuCudaDifferentialTest`, `CudaExecutionMonitorTest`, the CUDA backend-selection tests and Phase 12's `CudaEngineNoiseRefusalTest` all pass rather than skip. One CUDA test skips by design on this 8 GB device (`docs/adr/0014` section 17). Criterion 10: **Pass**. Phase 12 is **COMPLETE**.

### Goal

Represent noise in KIR and simulate noisy circuits exactly with a density-matrix engine, planned and costed like every other candidate.

### Required behavior

1. KIR gains an explicit channel operation (Kraus operators on at most `kMaxUnitaryQubits` qubits). It is a member of the `Operation` variant, like `UnitaryOp`, not a separate mechanism. Validation rejects a channel that is not completely positive and trace preserving, within a tolerance recorded in the ADR.
2. A Qiskit Aer `NoiseModel` given to the Qiskit adapter lowers into channel operations and readout errors in Python, before conversion to KIR. The C++ core receives only explicit channel operations and readout errors; Phase 12 adds no C++ noise-model type. Circuits without a noise model contain no channel operations.
3. A density-matrix engine executes channels exactly on the CPU. It produces counts and exact Pauli expectation values. Whether it also returns the density matrix itself is decided in the ADR.
4. Light-cone reduction and factorization must stay correct when channels are present, or refuse with a stated reason. Each has a written argument in the ADR and regression tests.
5. The cost model gets density-matrix formulas (work units and peak bytes, with checked arithmetic). A noisy job on an engine that cannot run channels is infeasible, with a reason. The noiseless engines are never used silently in place of a noisy run.
6. `explain()` reports the noise model's presence and the chosen engine. A job with no noise model has byte-identical `explain()` output.
7. Phase 11 time-budget monitoring works with the density-matrix engine (it needs safe boundaries).
8. Readout error is a per-measurement assignment matrix. It changes sampled counts only, never expectation values, matching Qiskit Aer.
9. Results match `AerSimulator(method="density_matrix", noise_model=...)`, including where Aer applies an error attached to `measure`: before the measurement. This was measured on 2026-10-01 to differ from `qiskit_aer.utils.insert_noise`, which places it after.

### Out of scope for Phase 12

Trajectory sampling, CUDA density-matrix support, noisy stabilizer, Pauli, MPS or tensor-network engines, error mitigation, `SamplerV2`, mid-circuit measurement and reset, and fitting noise models to device calibration data.

### Owner decisions (recorded 2026-10-01)

1. **Density matrix first.** Confirmed: Phase 12 is the exact density-matrix phase and trajectories are Phase 13.
2. **Noise-model format: Qiskit Aer compatible.** Users give a `qiskit_aer.noise.NoiseModel`. `qiskit-aer` becomes an optional dependency, needed only when a noise model is used. Compatibility was checked on 2026-10-01: `qiskit-aer==0.17.2` installs and runs with `qiskit==2.5.2` on Python 3.14.
3. **Noise types.** The first verified set is readout error, depolarizing and amplitude damping. Further noise types are added incrementally, each with its own verification against Aer.
4. **Meaning of `Result.exact` for noisy runs: `true`** for a density-matrix run, meaning exact for the requested model. The documentation is restated to match.
5. **Validation tolerance:** `common::kDefaultAbsTol` (1e-12), the same as the unitarity check, for trace preservation and readout rows.
6. **Unverified Aer noise types are rejected** with `UnsupportedFeatureError` until each is verified. The exact rule is ADR 0016 section 9.1. The native `append_channel` accepts any valid CPTP channel.
7. **Verification environment:** WSL Ubuntu 26.04 on the owner's machine, CPU and CUDA (RTX 4060 Laptop GPU, CC 8.9; CUDA 13.3.1, the newest published for WSL). Details are in ADR 0016 section 9.

### Phase 12 acceptance gate

1. Channel operations exist in KIR, are validated (CPTP within the ADR's tolerance), and have regression tests.
2. Density-matrix results match an independent reference (for example Qiskit Aer's density-matrix method) on a documented fixture set. The tolerance is recorded in the ADR.
3. A noiseless circuit run through the density-matrix engine matches the dense statevector engine.
4. The planner selects the density-matrix engine for noisy jobs through `plan_job`, and every other engine reports noisy jobs as infeasible with a reason.
5. Memory-budget refusal works for density-matrix plans (4ⁿ state size).
6. Light-cone reduction and factorization with channels are correct, or refuse, as the ADR specifies.
7. Jobs with no noise model are unchanged, including `explain()` and the CLI golden outputs.
8. Results are deterministic under a fixed seed.
9. The `Result.exact` semantics match the owner's decision and are documented.
10. CPU and CUDA verification stay green, and `scripts/verify.sh` passes.
11. No Phase 13 or 14 features are implemented.

### Phase 12 process

Before implementation Claude must:
- inspect KIR, validation, light-cone reduction, factorization, the cost model, the planner, the dense engine and `explain()`;
- draft ADR 0016 (noise representation, density-matrix engine, cost formulas, the `exact` semantics, validation tolerances);
- enter plan mode with exact files, API changes, tests, risks and ambiguities;
- wait for owner approval.

## Phase 13: Noisy trajectory simulation (COMPLETE — acceptance gate passed 2026-10-03)

**Status (2026-10-02):** Phase 12 is closed. The owner answered `docs/adr/0017` section 9 the same day: explicit opt-in (Q1), `expectation_precision` as proposed (Q2), noisy stabilizer and Pauli propagation deferred (Q3), and the section 6 thresholds as proposed (Q4). The owner approved the implementation plan the same day. Phase 13 is implemented on branch `phase-13-noisy-trajectories` (commits `58d89df` to `53de60b`, plus documentation). Every gate criterion has evidence in `docs/adr/0017` section 10. **Gate record (2026-10-03):** the owner accepted the gate on that evidence; all six criteria pass, and `scripts/verify.sh` exits 0 on `847cb44` (dev/release 578/578, pytest 207 passed, every CUDA stage 611/611 with GPU tests executing in the CUDA 12 builds). Phase 13 is COMPLETE.

### Goal

Simulate noisy circuits by sampling trajectories, which reaches much larger qubit counts than a density matrix. Validate it against Phase 12's exact results, and let the planner choose between the two methods by cost.

### Required behavior

1. Trajectory execution over the dense statevector engine. Noisy stabilizer (Pauli noise on Clifford circuits) and Pauli-propagation channel support are candidates whose inclusion is decided in the ADR.
2. Trajectory results are never reported as exact. Expectation values carry a reported statistical error, and EstimatorV2 `stds` stop being hard-coded to zero.
3. The caller's requested precision determines the trajectory count through a formula documented in the ADR. The cost model uses the same count.
4. Each trajectory has its own documented RNG stream, derived from the job seed, so results stay deterministic across replans.
5. "Between trajectories" is a Phase 11 safe monitoring boundary, and replanning from a density-matrix plan to a trajectory plan uses the existing planner.

### Phase 13 acceptance gate

1. Trajectory results agree with Phase 12 density-matrix results within their reported statistical error, checked with a documented statistical test on a fixture set.
2. Reported errors are genuine standard errors, and non-exact results are flagged in `Result` and `explain()`.
3. The planner chooses between density-matrix and trajectory plans by cost, and a controlled test demonstrates the choice.
4. Results are deterministic under a fixed seed, including after a replan.
5. Phase 12 behavior is unchanged, and `scripts/verify.sh` passes.
6. No Phase 14 features are implemented.

### Phase 13 process

ADR 0017 (trajectory sampling, precision semantics, RNG stream derivation) is written first. Then plan mode and owner approval.

## Phase 14: Error mitigation and batch execution (COMPLETE — acceptance gate passed 2026-10-07)

**Status (2026-10-07):** implemented on branch `phase-14-mitigation-batch` (commits `48d75f2` to `75ad45f`). Every gate criterion has evidence in `docs/adr/0018` section 10, and `scripts/verify.sh` exits 0 on `75ad45f`. **Gate record (2026-10-07):** the owner accepted the gate on that evidence, including criterion 6. All eight criteria pass. Phase 14 is COMPLETE.

### Goal

Provide error-mitigation techniques for noisy simulation, built on a batch execution layer that runs and combines many related circuits.

### Required behavior

1. A batch or job-level execution layer that runs several circuits and combines their results. It is not a second planner: each circuit is still planned by `plan_job`.
2. A Qiskit `SamplerV2` primitive, if the owner confirms it is in scope.
3. At least one mitigation technique. The owner chooses which, for example readout-error mitigation or zero-noise extrapolation.
4. Mitigation is explicit opt-in and never applied silently. Mitigated results are never labelled exact, and their error or uncertainty is reported.

### Owner decisions (recorded 2026-10-07)

1. **Technique: readout-error mitigation** (tensored assignment-matrix inversion from calibration circuits). Zero-noise extrapolation is not in this phase.
2. **`SamplerV2` is in scope** (`MENOSSampler`).
3. Calibration circuits run under the full noise model. The reported error includes target shot noise (exact) and calibration noise (first-order delta method). Mitigated data goes in `PubResult.metadata["readout_mitigation"]`. The thresholds are those in `docs/adr/0018` section 6. Design: `docs/adr/0018`. Branch: `phase-14-mitigation-batch`.

### Phase 14 acceptance gate

1. `run_batch` plans every circuit through `plan_job`, and result i equals `run(circuit_i)` under seed_i bit for bit. Results are deterministic under a fixed seed.
2. On the `docs/adr/0018` section 6 fixture set (6 circuits × {readout only, readout + depolarizing}), the mitigated distribution's total-variation distance to the noiseless reference is strictly smaller than the raw distribution's on every fixture.
3. Under readout-only noise, every mitigated outcome agrees with the noiseless reference within its reported standard error (Bonferroni, family-wise 0.001).
4. Reported standard errors are genuine: the 400-seed coverage test of `docs/adr/0018` section 6 passes.
5. Mitigated results are never labelled exact; their errors, γ and calibration are reported in the result and in `explain()`.
6. `MENOSSampler` follows the `SamplerV2` contract (checked against `StatevectorSampler`), and its noisy counts agree with Aer's `SamplerV2` by a chi-square test at 0.001.
7. Jobs without mitigation are unchanged: every pre-existing test, explain output and CLI golden file passes unmodified.
8. `scripts/verify.sh` passes. No Phase 15 features are implemented.

## Performance roadmap (Phases 15–17)

The owner added this roadmap on 2026-10-05, ordered after Phase 14. It comes from the benchmark dashboard (`tools/dashboard/`, commits `d27d530` and `33cb657`). Measured on an RTX 4060 Laptop GPU and an Intel Core Ultra 7 155H, the dashboard shows two things:

- The dense CPU engine is only **level with Qiskit's NumPy reference simulator** (`qiskit.quantum_info.Statevector`) on the crossover circuit family. Two runs gave a MENOS/Qiskit ratio of 0.98× and 1.52× at n = 22.
- The CUDA backend is 28× faster than the CPU at n = 24 for statevector output and 32× for expectation output, with the crossover at n = 14. `docs/adr/0014` section 16.3 records that the shipped generic kernel gives up about 4–8× of the GPU's available advantage.

Each phase follows the usual process: an ADR, then plan mode, then owner approval, on a `phase-<N>-<slug>` branch. Numeric speed-up targets are **not** set here. The owner sets them at each phase's kickoff from a fresh baseline measurement (CLAUDE.md: thresholds are not guessed).

Invariants every performance phase must preserve:
- **Correctness before performance.** Every result still agrees with the existing reference within the single tolerance policy (`src/common/tolerance.h`), and the existing differential tests stay unchanged.
- **The existing kernel stays as the oracle.** A faster path is added beside the generic one, never in place of it, and tests compare the two.
- **Determinism.** The same circuit, options and seed give the same results. Gate application that is independent per amplitude must stay bitwise identical to the single-threaded engine. Any reduction (expectation sums, norms, the sampling cumulative distribution) either keeps a fixed summation order or has an ADR that justifies and documents the change, because counts depend on it.
- **The CUDA bitwise-identity property** (`-fmad=false`, `docs/adr/0014` section 10) holds for every new kernel.
- **`plan_job` stays the only planner.** Faster kernels change the calibrated work-unit rates, so calibration is re-run. Any planner decision that changes as a result (for example the CPU/CUDA crossover) is measured and recorded, never left implicit.
- **No optimisation without a benchmark that shows the bottleneck.** The dashboard and `menos_dense_gate_application_benchmark` are the starting baseline.

## Phase 15: Dense CPU engine performance (COMPLETE — acceptance gate passed 2026-10-08)

**Status (2026-10-08):** the owner started the phase on 2026-10-07. Design: `docs/adr/0019`. Branch: `phase-15-dense-cpu-performance`. Step 1 (benchmark hygiene) is implemented and its baseline measured: `docs/adr/0019` section 10, data in `docs/benchmarks/phase-15-baseline/`. On 2026-10-08 the owner set the targets (N = 4, M = 5, plus three conditions) and the gate below is written. Steps 2–5 need their own implementation plan and owner approval. **Gate record (2026-10-08):** steps 2–5 are implemented on the same branch. Every gate criterion has evidence in `docs/adr/0019` sections 10.9–10.10, and `scripts/verify.sh` exits 0 on `a01052a` (dev/release 619/619, pytest 236 passed, every CUDA stage 652/652 with GPU tests executing in the CUDA 12 builds). The owner accepted the gate on that evidence, ruling that the benchmark-tool test extension flagged under criterion 5 does not count against it. All eight criteria pass. Phase 15 is COMPLETE.

### Owner decisions (recorded 2026-10-07)

1. The gate machine is the Intel Core Ultra 7 155H, measured in its WSL clone. The dashboard uses 10 repeats per point and reports min–max and interquartile range beside the median.
2. Threading first used OpenMP. On 2026-10-08 it was revised to a MENOS-owned thread pool, after OpenMP's default waiting policy measurably stalled every gate (`docs/adr/0019` section 3.1). The default is all hardware threads, and a new public `RunOptions::max_threads` (C++ and Python) caps it.
3. The specialised 1- and 2-qubit kernels are bitwise identical to the generic kernel (revised the same day from "within tolerance", to keep the CPU/CUDA bitwise identity of `docs/adr/0014` section 10). Different thread counts must also give bitwise-identical results.
4. Expectation sums and the sampling cumulative distribution stay single-threaded.
5. Step 5 may change the calibration method, by an amendment to `docs/adr/0019` backed by measurements.
6. Gate fusion is out of Phase 15.

### Goal

Make the dense CPU engine materially faster than Qiskit's reference simulator on the dashboard's circuit family and on the existing dense benchmarks, without changing any result.

### Required behavior

1. **Benchmark hygiene comes first.** Before any optimisation, the dashboard records the run-to-run spread (more repeats, and the spread reported next to the median) and the machine's power state, so that a change in speed can be told apart from noise. This step produces the baseline the gate is measured against.
2. **Multi-threaded gate application.** `src/kernels/cpu/dense_kernels.cpp` is single-threaded today. The threading mechanism (OpenMP or a hand-rolled pool), the thread-count policy, and how this interacts with the memory and time budgets are decided in the ADR.
3. **Specialised 1- and 2-qubit kernels**, kept beside the generic k-operand `apply_matrix`. The generic path gathers every amplitude group into a temporary vector and expands each index bit by bit; that path remains the oracle.
4. **Gate fusion is a candidate**, decided in the ADR: neighbouring gates combined into one matrix before execution, so the state is swept fewer times. If it is included, the fused circuit must give results equal to the unfused one within tolerance, and `explain()` reports that fusion was applied.
5. Re-run calibration, and record its effect on planner decisions.

### Phase 15 acceptance gate

The owner set the targets on 2026-10-08 from the step-1 baseline (`docs/adr/0019` section 10). "Baseline" below means the medians in section 10.2 (dashboard, runs C and D) and section 10.3 (dense benchmark, runs E and F). A speed-up is baseline median ÷ new median.

**How it is measured.** The gate measurements use the same machine (Intel Core Ultra 7 155H, WSL clone), on AC with the power check passing at every step, the same Windows power scheme as the baseline ("Balanced"), release builds, and the default thread count. Two complete runs are made, and **both** must meet every numeric criterion; a target met in only one run fails.

1. **Dashboard speed-up, N = 4.** MENOS's end-to-end median on the dashboard circuit family is at least **4×** faster than the baseline at n = 22 and at n = 24.
2. **Dense gate speed-up, M = 5.** The dense benchmark's 1-qubit and 2-qubit gate medians are at least **5×** faster than the baseline at n = 22, 24 and 26.
3. **No small-circuit regression.** No dashboard point from n = 4 to n = 18 is more than 10% slower than the baseline median.
4. **The specialised kernels deliver on their own.** With `max_threads = 1`, the dense 1-qubit and 2-qubit gate medians are at least 2× faster than the baseline at n = 22, 24 and 26.
5. **Results unchanged.** The specialised 1- and 2-qubit kernels are bitwise identical to the generic kernel. Every thread count gives bitwise-identical states for the dense, density-matrix and trajectory engines. The CPU/CUDA bitwise identity (`docs/adr/0014` section 10) still holds. Every pre-existing test, `explain()` output and CLI golden file passes unmodified.
6. **Recalibration recorded.** Calibration is re-run. The before/after rates, the cost-model correlation and every planner decision that changes (including the CPU/CUDA crossover, n = 14 at baseline) are recorded in `docs/adr/0019`. Any change to the calibration method follows `docs/adr/0019` section 7.
7. **Thread interaction checked.** The Python path is measured with and without `OPENBLAS_NUM_THREADS=1`, and the result is recorded.
8. `scripts/verify.sh` passes. No Phase 16 features are implemented.

### Phase 15 process

An ADR first (threading model, determinism of reductions, fusion yes or no). Then plan mode and owner approval.

## Phase 16: Specialised CUDA kernels (COMPLETE — acceptance gate passed 2026-10-09)

**Status (2026-10-08):** the owner accepted the Phase 15 gate and started Phase 16 the same day. Branch: `phase-16-specialised-cuda-kernels`. Design: `docs/adr/0020`. The owner answered its section 9 on 2026-10-08: the RTX 4060 Laptop GPU in WSL (`cuda12`) is the gate machine; the probe is rebuilt and re-measured; the specialised kernels are general 1- and 2-qubit matrix kernels, bitwise identical to the generic kernel; calibration may change by amendment; and, as in Phase 15, step 1 (probe, per-gate profile, baseline) comes first and the owner sets the targets from it. Step 1 (benchmark-only probe, per-gate profile, baseline) is implemented and measured: `docs/adr/0020` section 10, data in `docs/benchmarks/phase-16-baseline/` (runs a and b). The owner then answered Q4 ((a) in scope; one synchronisation per gate kept) and set the gate targets. The owner approved the implementation plan for steps 2–4 the same day. Step 2 (specialised kernels without per-gate copies, `f495da6`) showed that bitwise-identical kernels are FP64-bound on this GPU (`docs/adr/0020` section 10.7), and the owner reset the targets below to the measured figures. **Gate record (2026-10-08):** steps 2–4 are implemented. The gate runs and `scripts/verify.sh` (exit 0 on `da0ad8b`, GPU tests executing in the cuda12 stages) are recorded in `docs/adr/0020` sections 12–13. Every criterion has evidence and passes, except criterion 1 at one point: n = 8 in run b, 1.39× against 1.4×, inside that point's spread, with the GPU throttled in that run. **The owner accepted the gate on 2026-10-09**, ruling that this one miss does not count against it (`docs/adr/0020` section 13.1). Merged to `main` in `c6b9c87`. Phase 16 is COMPLETE.

### Goal

Reclaim the GPU advantage that the generic CUDA kernel gives up, as recorded in `docs/adr/0014` section 16.3.

### Required behavior

1. Re-measure the specialised H/CX probe first. The probe was scratch code and is not in the repository. Before any specialised kernel is accepted, the new measurement must reproduce the recorded figures (crossover n ≈ 10 and 57–134× at n = 22 on the RTX 3050), or explain why it does not.
2. Specialised kernels for common 1- and 2-qubit gates, added beside the generic k-operand kernel, which stays as the oracle.
3. Every specialised kernel holds the `-fmad=false` bitwise-identity property against the CPU primitive, because counts depend on it.
4. Re-run calibration, and re-measure and record the CPU/CUDA crossover.

### Phase 16 acceptance gate

The owner set the targets on 2026-10-08 from the step-1 baseline, and reset them the same day after step 2 (`docs/adr/0020` sections 10–11).

**How it is measured.** Every measurement is made on the gate machine: the RTX 4060 Laptop GPU in the WSL clone, `cuda12-release`, on AC with the power check passing at every step, and the Windows power scheme "Balanced". A speed-up is a **same-run** ratio, generic median ÷ new median. The generic side is `kernels::cuda::apply_matrix_generic`, the unchanged generic kernel kept as the oracle. The new side is the shipped `kernels::cuda::apply_matrix`. Both are measured in the same run of `menos_cuda_kernel_benchmark` on the crossover circuit family, 10 repeats each. A same-run ratio is used because the generic kernel's speed moved 25–40% between the two baseline runs (`docs/adr/0020` section 10.4). Two complete runs are made, and **both** must meet every numeric criterion; a target met in only one run fails.

1. **End-to-end speed-up: at least 1.4× at every n from 4 to 14, and at least 1.15× at n = 22, 24 and 26.** End to end means allocate, initialise, apply every gate, copy the state back to the host and free, as `CudaDenseEngine` runs a statevector job.
2. **Per-gate speed-up at n = 22, 24 and 26 (host wall time): at least 1.3× for 1-qubit gates, and at least 0.95× for 2-qubit gates (no regression beyond measurement noise; owner, 2026-10-08).**
3. **Bitwise identity.** The specialised 1- and 2-qubit kernels are bitwise identical to the generic kernel, on random states and random unitaries, on every operand position and order (2-qubit reversed and non-adjacent), compared by exact equality. `CpuCudaDifferentialTest` is unmodified and passes, with its GPU tests executing.
4. **No small-circuit regression.** No point of the same-run end-to-end comparison from n = 4 to n = 18 is more than 10% slower than the generic kernel, in either run.
5. **The probe is explained.** The step-1 measurement of the probe and the reasons its recorded 4–8× gap is not reproduced are recorded (`docs/adr/0020` section 10.1).
6. **Recalibration and crossover recorded.** Calibration is re-run. The before/after device rate, the launch overhead and every planner decision that changes, including the CPU/CUDA crossover, are recorded in `docs/adr/0020`. Any change to the calibration method is an amendment with measurements (owner Q5).
7. **`scripts/verify.sh` exits 0**, with GPU tests that execute in the CUDA 12 builds on this machine.
8. **Documentation.** `docs/adr/0020`, this file and the README reflect the shipped behaviour, and every new symbol has a production caller and tests.

### Phase 16 process

An ADR (or an amendment to `docs/adr/0014` section 16.3) first. Then plan mode and owner approval.

## Phase 17: API surface: recorded deferrals (COMPLETE — acceptance gate passed 2026-10-10)

**Status (2026-10-09):** the owner started Phase 17 on 2026-10-09 and put all four candidates below in scope. Branch: `phase-17-api-deferrals`. Design: `docs/adr/0021`, whose section 9 the owner answered on 2026-10-09: `backend = cuda` refuses rather than falls back; backend selection is exposed in C++/Python `RunOptions`, the Qiskit `MENOSBackend` and the CLI; the Qiskit time budget goes on `MENOSBackend` only; and item 4 is measured first. The gate below is written. Implementation needs its own plan and owner approval. Implemented on 2026-10-09: item 2 (`d49d1c5`), item 3 (`199efd4`), item 1 (`8acebe0`), step 4a (`docs/adr/0021` section 10) and item 4b for Factorized plans (`e214c27`), with the owner's rulings on criteria 1, 4 and 5. **Gate record (2026-10-09):** every criterion has evidence in `docs/adr/0021` section 11, and `scripts/verify.sh` exits 0 on `1041893` with GPU tests executing in the cuda12 builds. **The owner accepted the gate on 2026-10-10** (`docs/adr/0021` section 11.1). Merged to `main` (fast-forward) at `0af513e`. Phase 17 is COMPLETE.

### Goal

Lift API deferrals recorded in earlier phases, where a later phase or the dashboard has shown they are needed. The owner confirms which candidates are in scope at kickoff.

### Candidates

1. **Explicit backend selection from Python.** The Python API is backend-agnostic by design (`docs/adr/0014`). The dashboard had to use the C++ crossover benchmark to force CPU against CUDA, which shows the need.
2. **`time_budget_seconds` in the Qiskit `BackendV2` adapter.** It is exposed in the Python API (Phase 11) but the adapter does not set it.
3. **A time-budget flag for the CLI.**
4. **Heterogeneous per-region CPU/CUDA selection** for `Factorized` and `CutAndRecombine` plans (`docs/adr/0014` section 16.2). This changes planner decisions, so if it is chosen it needs its own measurement and ADR, and it may become a separate phase.

### Phase 17 acceptance gate

Written on 2026-10-09 from `docs/adr/0021` (owner answers in its section 9).

1. **Backend selection (item 1).**
   - `RunOptions::backend` (C++), `RunOptions.backend` (Python), the `MENOSBackend` run option `backend` and the CLI flag `--backend cpu|cuda` each have tests showing that a forced backend is the one that executes (the executed region's backend is asserted).
   - `backend = cuda` without CUDA raises `UnsupportedFeature`, and when nothing fits it raises `ResourceExceeded`; never a CPU fallback.
   - A forced backend survives a replan.
   - Forcing CPU or CUDA gives bitwise-identical statevectors and counts whenever both runs use dense simulation (tested on a device). Forcing CUDA may change the algorithm relative to the planner's choice (to dense), which `explain()` shows. *Reworded by the owner on 2026-10-09: the original wording (identical to the unforced run) cannot hold when the algorithm changes; see `docs/adr/0021` section 2.*
2. **Qiskit time budget (item 2).** `MENOSBackend` accepts `time_budget_seconds`. Tests cover a budget that is met (normal result), one that cannot be met (`ResourceExceededError`), and a replan whose record appears in the result metadata.
3. **CLI time budget (item 3).** `--time-budget SECONDS` on `menos run` and `menos explain`. Tests cover a valid value, each invalid form (non-numeric, zero, negative, non-finite) and a refusal.
4. **Per-region selection (item 4).** Step 4a's measurement is recorded in `docs/adr/0021` section 10. *Owner decision (2026-10-09): 4b is built for `Factorized` plans only; `CutAndRecombine` keeps the CPU (measured, not adopted).* Target, measured with `menos_cuda_factorized_backend_benchmark` as a same-run ratio (the all-CPU assignment's median ÷ the planner's own assignment's median), in two complete runs on AC, both of which must pass:
   - at least **1.2×** on at least three of the five measured circuits (12 + 22, 16 + 20, 20 + 22, 10 + 16 + 22 and 14 + 18 + 22 qubits of scrambling blocks) in each run. *Reset by the owner on 2026-10-09 from "each of the five", after both gate runs met 1.2× on three circuits and measured 0.95–1.07× on 20 + 22 and 14 + 18 + 22, with every GPU assignment 30–50% slower than in step 4a while the all-CPU times were unchanged: the GPU's run-to-run speed variation (`docs/adr/0020` section 10.7), which a calibration taken once per process cannot follow (`docs/adr/0021` section 10.3).*
   - no circuit more than 10% slower than all-CPU;
   - every per-region choice appears in `explain()` (`backend=cuda` on the region line), and results are unchanged: each region runs dense on either backend, so counts are bitwise identical.
5. **Defaults unchanged.** Every pre-existing test and every CLI golden file passes unmodified, and `explain()` for a job that sets none of the new options is byte-identical to before (asserted by a test). *Owner ruling (2026-10-09): the Phase 11 test `test_the_qiskit_backend_still_exposes_no_time_budget`, which asserted the deferral item 2 lifts, is replaced by tests of the new option and does not count against this criterion.*
6. **`scripts/verify.sh` exits 0**, with GPU tests executing in the `cuda12` builds.
7. **Documentation.** The README, the Python and CLI reference pages, `docs/adr/0021` and this file describe every new option, with its default and refusals. Every new symbol has a production caller and tests.

### Phase 17 process

An ADR for each item that changes public API or planner behaviour. Then plan mode and owner approval.

## Phase 18: Native Windows (MSVC) — COMPLETE (acceptance gate passed 2026-10-18)

**Status (2026-10-13):** the owner started Phase 18 on 2026-10-13 as the next phase, ahead of the proposed interoperability roadmap, which was renumbered to Phases 19–26. Branch: `phase-18-windows-native`. Design: `docs/adr/0022`, whose section 9 records the owner's answers of 2026-10-13. The owner approved the plan the same day. The gate below is written. Gate evidence on `7fa38ce` (local Windows, GitHub Actions run 38581342797, Linux `verify.sh`) is recorded in `docs/adr/0022` section 10. The owner accepted the gate on that evidence on 2026-10-18 (section 10.4).

### Goal

Build, test and ship MENOS natively on Windows x64 with MSVC: the CPU engines, planner, runtime, CLI, Python package and wheels. Until now the source refused to build on Windows (`src/plan/hardware_profile.cpp`), and Windows users had to use WSL.

### Required behavior

1. **Host-memory probe.** `detect_hardware_profile()` reads total physical memory with `GlobalMemoryStatusEx` on Windows. The planner's default memory budget (half of it) is unchanged.
2. **MSVC toolchain.** The library, CLI, tests, benchmarks and Python extension build with MSVC at `/W4 /WX`. Floating-point contraction stays off (`/fp:precise`, never `/fp:contract`), the MSVC counterpart of `-ffp-contract=off`.
3. **LAPACK on Windows.** `LAPACKE_zgesdd` comes from OpenBLAS's official x64 binary release, pinned by version and SHA-256, LP64 only.
4. **Windows wheels.** CPython 3.10–3.14, `win_amd64`, CPU-only, with the OpenBLAS DLL and its licence inside the wheel.
5. **CUDA on Windows is refused at configure time** with a message naming the deferral; it is never silently disabled.
6. **Linux and macOS behaviour is unchanged.**

### Owner decisions (recorded 2026-10-13)

1. Phase number 18; the proposed interoperability roadmap becomes Phases 19–26, content unchanged.
2. CPU only. CUDA on Windows is a recorded deferral.
3. Windows wheels for CPython 3.10–3.14, the same range as Linux.
4. Sanitizers: MSVC `/fsanitize=address` in the Windows Debug preset. UBSan and LSan coverage stays with the Linux `scripts/verify.sh`, which remains part of the gate.
5. LAPACK: the official OpenBLAS x64 binary release, pinned by version and SHA-256, shipped inside the wheel.
6. A Windows test that fails only through a last-bit numeric difference from Linux (the MSVC runtime's `sin`/`cos`/`exp`, or OpenBLAS against reference LAPACK) is brought to the owner. No tolerance or expected value is changed without an owner ruling.

### Out of scope for Phase 18

CUDA on Windows; Windows ARM64; MinGW and clang-cl; publishing to PyPI; clang-tidy and clang-format on Windows (they stay in the Linux `verify.sh`); processor groups beyond 64 logical CPUs; CPU-name detection in the Windows benchmark environment.

### Phase 18 acceptance gate

Written on 2026-10-13 from `docs/adr/0022` (owner answers in its section 9).

1. **Native build.** The `windows-dev` (MSVC, `/fsanitize=address`, Debug) and `windows-release` presets configure and build at `/W4 /WX`, and `ctest` passes 100% on both. Counts are recorded and every skipped test is listed.
2. **Memory probe.** The Windows branch uses `GlobalMemoryStatusEx`. A regression test bounds the result by `GetPhysicallyInstalledSystemMemory`, an independent source, and the planner's default-budget tests pass.
3. **LAPACK.** OpenBLAS is pinned by version and SHA-256, found by CMake, and checked to be LP64. `svd_test` and every MPS and Schmidt-rank test pass on Windows with no tolerance or expected value changed.
4. **Python (editable).** The full Python suite passes on CPython 3.14 on Windows, with the same deselects as `scripts/verify.sh`.
5. **Wheels.** `cibuildwheel` builds cp310–cp314 `win_amd64` wheels locally; each, installed in a clean environment, passes its test command; each contains the OpenBLAS DLL and its licence; `delvewheel` reports no unresolved dependency.
6. **CUDA refusal.** Configuring with MSVC and `MENOS_ENABLE_CUDA=ON` fails with the deferral message.
7. **Linux unchanged.** `scripts/verify.sh` exits 0 in the WSL clone on the final commit, and the Linux test counts change only by the tests this phase adds.
8. **Documentation and CI.** The README, the platform, requirements, installation, compatibility, limitations, troubleshooting and changelog pages, `docs/adr/0022` and this file describe Windows support and its limits. The wheel workflow has a Windows job. Every new symbol has a production caller and tests.

**Gate record (2026-10-18):** every criterion has evidence in `docs/adr/0022` section 10, on `7fa38ce`: `scripts\verify-windows.ps1 -Wheels` exits 0 locally, GitHub Actions run 38581342797 succeeds, and `scripts/verify.sh` exits 0 in the WSL clone. The owner accepted the gate on that evidence. All eight criteria pass. Phase 18 is COMPLETE.

### Phase 18 process

ADR `docs/adr/0022` first, then plan mode and owner approval (both done on 2026-10-13). Windows verification runs through `scripts/verify-windows.ps1`; Linux verification stays `scripts/verify.sh` in the WSL clone.

## Memory roadmap (Phases 21–23) — Phases 21–23 COMPLETE

**Status (2026-10-18):** the owner placed three memory phases next on the roadmap, ahead of the remaining interoperability phases, which were renumbered from Phases 21–26 to Phases 24–29 with their content unchanged. The order is decided. Each phase is NOT STARTED: it starts only on the owner's instruction, and then follows the usual process (an ADR, owner answers to its section 9, plan mode and owner approval, a `phase-<N>-<slug>` branch). The scope, thresholds and gate of each are set at its kickoff; the text below is the proposal. Problem statement and proposal for the owner: the "MENOS Memory: Problem and Proposed Roadmap" document (2026-10-18). **Update (2026-10-31):** the owner deferred Phase 21's final verification and gate decision, and started Phase 22 (`docs/adr/0026`) on branch `phase-22-cuda-memory-efficiency` from Phase 21's head, so Phase 22 merges after Phase 21. This sets aside "current phase only" for the two memory phases, as it was for Phases 18 and 19. Neither gate is passed. **Update (2026-11-02):** the owner placed a planner phase (Phase 24, "Planner and dense-engine roadmap" below) after Phase 23, so the interoperability phases became Phases 25–30.

### Why

The planner checks each candidate's `estimated_peak_bytes` against the host budget (`memory_budget_bytes`, else half of total RAM) and, for CUDA, against free device memory read once per process. Several paths allocate more than they estimate, and no run measures its real peak:

| Path | Estimated | Allocated | Where |
|---|---|---|---|
| Dense CPU, statevector requested | 16·2^n | 32·2^n (the result is a second copy) | `src/engines/dense/dense_engine.cpp` |
| CUDA dense, statevector or counts | 16·2^n host | 32·2^n host (device → host buffer → result) | `src/engines/dense_cuda/cuda_dense_engine.cpp` |
| Counts through `CumulativeDistribution` (multi-region Factorized, CUDA, density matrix, trajectories) | 8·2^n | at least 48·2^n (a double, a `BitString` and its heap word per basis state) [DERIVED from the type layout, not measured] | `src/engines/dense/cumulative_distribution.h` |
| MPS | site tensors | plus SVD copies and per-gate theta buffers | `src/common/svd.cpp`, `src/engines/mps/apply_gate.cpp` |

`EngineExecutionResult::actual_bytes` is computed from the same container sizes, so tests comparing it with the estimate cannot detect this. Example: on 32 GiB of RAM (default budget 16 GiB), a 30-qubit dense job asking for the statevector is estimated at 16 GiB, accepted, and allocates 32 GiB. This is a correctness defect under prime directive 1, so accounting comes first.

### Invariants every memory phase must preserve

- `plan_job` stays the only planner. A memory tier is a candidate plan with a cost and a feasibility rule, never a runtime decision.
- An estimate is an upper bound on measured use. A plan that cannot keep its memory promise is infeasible, never attempted.
- Results stay bitwise identical to the in-memory engines, which remain the oracles.
- A job that does not use a new tier keeps its plan, results and `explain()` output, except where a measured accounting fix changes feasibility; every such change is recorded.
- The Phase 11 invariants hold (no tuning constants; an observation only raises a cost; state migration needs a measurement and an ADR first). The Phase 15 and 16 invariants hold (bitwise-identical kernels, `-fmad=false`, one synchronisation per gate).
- New dependencies are optional, pinned (ADR 0001), behind a CMake option, and added only after a measurement shows they beat what MENOS does without them.

### Phase 21: Memory accounting and copy removal — COMPLETE (acceptance gate passed 2026-11-02)

*The owner started this phase on 2026-10-20. Design: `docs/adr/0025`; the owner answered its section 9 the same day, taking every recommended option. Branch: `phase-21-memory-accounting`, from `main` at `1f89192`. Implemented in `25fcef9`–`72d8f27` (steps 1–4 of `docs/adr/0025` section 6, evidence in its sections 10.1–10.4). On 2026-10-31 the owner deferred step 5 and the gate decision, and started Phase 22 on a branch from `72d8f27`.*

**Gate record (2026-11-02):** step 5 ran on `9b8833c`, which changes only documentation after `72d8f27` (`docs/adr/0025` section 10.5). Both verification scripts exit 0. `scripts/verify.sh`: dev/release 749/749, every CUDA stage 819/819 with GPU tests executing in the cuda12 builds, pytest 556 passed. `scripts/verify-windows.ps1`: `windows-dev`/`windows-release` 750/750, pytest 556 passed. Criterion 4 gate run 3 missed 7 of the 36 dense ratios: three gate-application ratios that vary by 6–15% between rounds, and `DenseSample` at n = 20–26, 1.021–1.042. The MPS benchmark passed at 0.995. Run alone over six rounds, `DenseSample` is 1.010–1.023 slower; alignment accounts for at most about 1%, and the rest of the cause is not identified (section 10.6). **The owner ruled on 2026-11-02 that these misses do not count against criterion 4, and the gate is passed.** The 2% and 5% figures are unchanged. Every other criterion has evidence and passes. Phase 21 is COMPLETE.

**Goal.** Every estimate is an upper bound on the memory the job really uses, shown by measurement, and the avoidable copies are removed. No new dependency.

**Required behavior (proposed).**
1. Each engine's `estimated_peak_bytes` counts result copies, the counts distribution, MPS SVD workspaces and per-gate buffers, and the global statevector of Factorized plans.
2. Peak process memory (RSS on Linux and Windows) and device memory in use are measured around a run; a test asserts estimate ≥ measured for each engine and output kind.
3. Copies removed: the statevector is moved into the result, device memory is copied straight into the result buffer, and counts outcomes are stored as basis indices, with `BitString`s built only for outcomes drawn.
4. `explain()` prints device bytes as well as host bytes, and `Result` carries the measured peak beside the estimate.

**Owner decisions needed.** Default budget from half of total RAM (deterministic, today) or from free RAM; whether free VRAM is refreshed before each plan; whether an accounting fix may make a job that is feasible today infeasible, or must land together with the copy removal that keeps it feasible.

**Acceptance gate (proposed).** Estimate ≥ measured peak on a fixture set covering every engine and output kind; the peak of the statevector and counts paths reduced by a measured amount; every count, statevector and expectation bitwise identical to before; `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0.

### Phase 22: CUDA memory efficiency — COMPLETE (gate accepted 2026-11-02)

*The owner started this phase on 2026-10-31, with the Phase 21 gate open. Design: `docs/adr/0026`; the owner answered its section 9 the same day, taking every recommended option, and approved the implementation plan. Branch: `phase-22-cuda-memory-efficiency`, from `phase-21-memory-accounting` at `9b8833c` (code unchanged since `72d8f27`). The owner decisions and the gate below supersede the proposal text where they differ.*

**Owner decisions (2026-10-31).** Q1–Q6 of `docs/adr/0026` section 9: the pool persists across jobs, regions, replans and batch circuits in one process; its release threshold is the device's total memory and the planner counts its retained bytes as available; the device→host path is decided by measurement; the gate as refined there; an explicit pool, not the device default pool; Phase 21 is verified on its own branch before the Phase 22 gate. After step 1 measured (section 10.1), Q7–Q9: **item 1 is built** (a pooled cycle is 215–45 000× cheaper than a direct one, worth 1.2–4.5% of a 20-qubit execution and 2–20% at 16–18 qubits); **item 2 is recorded, not built** (0.1–2.8% of a 100-term job; the cost is the synchronising copy, not the allocation); **item 3 is recorded, not adopted** (no page-locked path beat the pageable copy in five runs on this machine).

**Phase 22 acceptance gate** (`docs/adr/0026` section 9, Q4 as amended by Q9):
1. The step-1 measurements recorded as same-run ratios with spread.
2. Device allocation + zero fill + free per execution at least 10× cheaper from the pool than direct at n = 20, 22 and 24, same run (`menos_cuda_memory_benchmark`).
3. Page-locked transfers: the record of their non-adoption with the measurement.
4. The pool's retained memory counted in the device feasibility check and printed by `explain()`; `release_device_memory()` works; the fallback works.
5. Every count, statevector and expectation in the Phase 21 fixture set bitwise identical to the `1f89192` baseline.
6. The planner change (recalibrated device rate, any crossover shift) recorded.
7. GPU tests execute in the `cuda12-*` presets; `scripts/verify.sh` (WSL) and `scripts/verify-windows.ps1` exit 0 on the final commit; documentation updated; every new symbol has a production caller and tests.

**Verification record (2026-11-02).** Steps 2–6 of `docs/adr/0026` section 6 are done. The pool, the single zero fill, the accounting, `explain()`'s two new fields and `release_device_memory()` are implemented in `2f02d10` and merged with Phase 21's closure in `5bf355a`; `e3619b3` adds `release_device_memory` to the `python-api.md` export snippet, the one test failure of the first `verify.sh` run, and is the final code. Evidence: section 10.2 (the code, its tests and callers), 10.3 (criterion 2: the allocation cycle is 1 082–1 156× cheaper from the pool at n = 20, 1 637–2 722× at 22 and 9 799–10 339× at 24, three runs), 10.4 (criterion 6: the device rate recalibrates 9–35% higher; the forced and the planned crossover do not move), 10.5 (criterion 7: `scripts/verify.sh` exit 0 in WSL with the GPU tests executing in both `cuda12-*` presets, `scripts/verify-windows.ps1` exit 0, both on `e3619b3`) and 10.6 (every criterion with its evidence). **The owner accepted the gate on 2026-11-02 (`docs/adr/0026` section 10.7): Phase 22 is COMPLETE.**

**Goal.** Less time spent allocating and copying on the GPU, using only CUDA runtime features MENOS already links (CUDA 12.0 and later). No new library.

**Required behavior (proposed).**
1. A stream-ordered memory pool (`cudaMallocAsync`) for the statevector and scratch buffers, reused across jobs, regions and replans instead of a `cudaMalloc` per execution.
2. The expectation partial-sum buffer allocated once per execution, not once per observable term.
3. Device-to-host copies through page-locked host memory, counted against the host budget.

**Owner decisions needed.** Whether the pool persists across jobs in one process, and how much free VRAM it may hold.

**Acceptance gate (proposed).** Measured reductions in allocation and device-to-host time as same-run ratios (`docs/adr/0020`: the GPU's clock state varies about 2× between runs); pool memory counted in the device feasibility check; bitwise-identical results; GPU tests that actually execute in the CUDA 12 presets; `scripts/verify.sh` exits 0.

### Phase 23: Tiered memory across VRAM, RAM and disk — COMPLETE (gate passed 2026-11-05)

*The owner started this phase on 2026-11-02, the day Phase 22's gate was accepted. Design: `docs/adr/0027`, whose section 9 (Q1–Q10: tiers in scope, the scratch file's location and limits, when the disk may be measured, the materiality rule for the host-offloaded tier, when the planner may choose a tier, transfer/compute overlap, the gate's factor, the carried memory-review items, the gate, the chunk seam) the owner answered the same day, taking the recommended option for every question (ADR 0027, "Owner answers"), and approved the implementation plan. Branch: `phase-23-tiered-memory`, from `phase-22-cuda-memory-efficiency` at `0282ded`. The text below is the proposal the ADR refines; the outcome follows it.*

**Outcome (2026-11-04).** Step 1 measured first (`88294be`, ADR section 10.1): on the owner's rule the host-offloaded CUDA tier is recorded as measured and not built, unified memory is measured only, and the disk tier is built with unbuffered I/O. Steps 2 and 3 (`82d1cdf`–`bee5263`, section 10.2) built it: an execution engine is now `(Engine, Backend, Tier)`; one chunked pass executor applies gates on a window's local qubits chunk by chunk with the unchanged kernels and pairs chunks for gates on global qubits, in the fewest passes; `FileBackingStore` holds the state in a scratch file that is gone on any exit; the planner prices the tier from the circuit's pass count and the directory's measured bandwidth and reaches for it only when the in-memory state does not fit (or when `RunOptions::tier` forces it, which never falls back); `explain()` names the tier and the scratch figures; `RunOptions::scratch_directory` (5%/8 GiB reserve), `Result::peak_disk_bytes()`, `menos.Tier`, `--tier`, `--scratch-dir`; the memory-accounting fixtures cover the tier. Step 5 (`30266fd`–`0e7242e`, section 10.3): at n = 31 on Windows (a 32 GiB file on drive C:) the planner chose the disk tier unaided, 4 passes, 440–765 s against predictions of 826–857 s (0.51–1.17× with the first run), the file gone afterwards; at n = 29 counts and the expectation value are bitwise the in-memory engine's on both platforms; in WSL at n = 30 the measured 155.6 s is 0.35× the predicted 439.4 s, outside the factor of 2 on the pessimistic side (the work rate is calibrated single-threaded; a Phase 24 item); the device race (Q8(i)) is a `ResourceExceeded` naming the bytes, verified on the GPU in WSL. Step 6 (this documentation and the two verification scripts on the final commit, `a20c58d`, both exit 0) is recorded in section 10.4. The owner passed the gate on 2026-11-05, ruling that criterion 3's one WSL miss does not count against it (section 10.5).

**Goal.** A dense state larger than the fastest memory still runs, by keeping it in a slower tier and streaming chunks of 2^L amplitudes to where the work happens. Consecutive gates on the L local qubits are applied in one pass; a gate on a global qubit pairs chunks or is preceded by a qubit-index swap. `plan_job` prices each tier with calibrated bandwidths and chooses; `explain()` names the tier.

**Candidate plans (proposed).**
1. Host-offloaded CUDA dense: the state in pinned RAM, processed on the GPU chunk by chunk over PCIe; reaches the host budget instead of free VRAM.
2. Unified memory (managed memory oversubscribing VRAM), compared by measurement with explicit chunking.
3. Out-of-core CPU dense: the state in a memory-mapped file (`mmap` / `CreateFileMapping`); reaches the disk limit. [DERIVED] One pass over a 34-qubit state (256 GiB) at about 3 GB/s takes about 90 s, so this is useful only with deep gate grouping.

**Owner decisions needed.** Which candidate plans are in scope; the disk location and size limit; whether NVIDIA cuStateVec (Phase 29, licence decision) is used here.

**Candidates carried from the memory review of 2026-11-02** (`docs/adr/0026` section 11; each needs an owner decision at kickoff and is in Phase 23 only if the owner puts it there): a separately reported planning-memory peak; budget reservation at the allocation sites, as a runtime guarantee behind the estimates; a configurable cap on the retained device pool, and `ResourceExceeded` rather than `InternalError` when another process takes device memory between planning and allocation; a batch-level budget or a streaming `run_batch`. The thread-local ledger is an invariant to preserve, not work.

**Acceptance gate (proposed).** Each tier runs a circuit too large for the tier above it; results bitwise identical to the in-memory dense engine on overlapping sizes; predicted time within a factor the owner sets of measured time; the planner picks a tier only when it is the cheapest feasible plan, shown in `explain()`; `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0.

## Planner and dense-engine roadmap (Phase 24) — STARTED (planner performance merged 2026-11-05; gate OPEN)

**Status (2026-11-02):** the owner placed one planner phase after the memory roadmap, on the assessment of a planner review the same day, and ahead of the interoperability phases, which were renumbered from Phases 24–29 to Phases 25–30 with their content unchanged. Its first part, planner performance (items 2 and 5 below), was started on a contributor's fork on 2026-11-03 and merged on 2026-11-05 (`docs/adr/0028`); the rest starts only on the owner's instruction, and then follows the usual process (an ADR or an extension of `docs/adr/0028`, owner answers to its section 9, plan mode and owner approval, branch `phase-24-planner-accuracy`). The scope, thresholds and gate are set at its kickoff; the text below is the proposal. **Update (2026-11-02, later the same day):** after a measured comparison with Qiskit Aer and PennyLane Lightning (recorded under "Why" below), the owner added two dense-engine items to this phase, gate fusion (item 7) and SIMD kernels (item 8), and extended item 2 with a bound on the tensor-network elimination search. The phase's name became "Planner accuracy, planning cost and dense-engine throughput"; its branch name is unchanged.

### Phase 24: Planner performance; planner accuracy, planning cost and dense-engine throughput — STARTED (on a contributor's fork, 2026-11-03; merged 2026-11-05; gate OPEN)

#### Planner accuracy: items 1, 2, 3 and 9 (started 2026-11-05)

**Status (2026-11-06).** The owner started items 1 (measure the cost model), 2 (report and bound planning cost), 3 (per-region observations) and 9 (threaded calibration and the disk link per direction) on 2026-11-05, on branch `phase-24-planner-accuracy` from `main` at `65e9c6c`, and answered the design questions the same day (`docs/adr/0028` sections 16–16.9, with the termination proof of per-region replanning in 16.8 and the benchmark method in 16.9). Step 1 measured every engine's actual over predicted seconds on Linux and Windows, planning cost and the dense rate by size and thread count (16.10). At checkpoint A (16.11) the owner decided: the threaded rate prices the disk tier only, with reads and writes at their own bandwidths; the MPS formula is unchanged (per-gate bonds would move it the wrong way; a measured per-gate fixed cost is a follow-on item); no search cap (the 4.4 s case now plans in 2–6 ms); the accuracy gate is no regression on Linux plus the disk tier within 2× on the Phase 23 circuits, with per-engine model fixes a follow-on phase. Implemented: `Result::plan_seconds()`/`replan_seconds()` and `plan_s=`/`replan_s=` in `explain()` (item 2); the disk tier's pricing (item 9); per-region and plan floors so that a Factorized overrun replans instead of being refused, with a cut overrun still refused (item 3). The final measurement and verification are in `docs/adr/0028` section 16.12. **Gate: OPEN** (the owner's).

#### Item 7, gate fusion in the dense CPU engine (started 2026-11-09)

**Status (2026-11-09).** After the planner-accuracy items were verified on both platforms (`docs/adr/0028` section 16.13), the owner asked for the remaining items to be completed in the order recorded below, so item 7 started the same day on branch `phase-24-dense-fusion`, from `phase-24-planner-accuracy`. Its kickoff is `docs/adr/0028` section 17: what the engine does today (17.1), the proposed design (17.2: one `ir::fuse_gates` pass applied by the CPU dense engine, the CUDA dense engine and the disk tier, a structural grouping rule whose per-size costs come from the step-1 measurement and never from run-time calibration, the fused matrix computed by the kernels themselves so the fused run keeps every bitwise identity the unfused run has, the cost model charging the blocks, `explain()` naming the outcome, `RunOptions::fuse_gates`), the step-1 method (17.3, `benchmarks/dense_fusion_probe.cpp`) and the owner's questions (17.4: the window, the default, the Phase 21 baseline, which engines, the cost model, the gate). Step 1 was measured the same day on both platforms (`docs/benchmarks/phase-24/fusion-step1/`, ADR section 17.5): on every thread at 20–24 qubits a 2-qubit pass costs 1.2–1.9× a 1-qubit pass but a 3-qubit pass 3.5–4.7× and a 5-qubit pass 8–10×, on one thread the 2-qubit and generic kernels are 2–10× and 4–30× a 1-qubit pass (the kernels, item 8's subject, set the window), diagonal passes are cheap at every width, a window of 2 gains 2.0–2.5× on every thread and 1.2–1.95× on one thread on the brickwork family while windows 4 and 5 lose, and fusion's rounding is at most 2.3 × 10⁻¹⁷. Section 17.5.5 recommends a window of 2 for dense blocks under a containment rule (a gate joins a block only when one qubit set contains the other), 5 for diagonal blocks, on by default at every size. **The owner took every recommendation the same day** (ADR section 17.6), with four conditions: numerical correctness, memory accounting, cross-engine consistency and performance predictability are not compromised; the last one refined the diagonal window to the same containment rule (the single-thread measurement does not support merging diagonal gates on disjoint qubits, which stays the opt-in `fuse_diagonal_gates`). **Built the same day** (section 17.7): `ir::plan_fusion` and `engines::dense::fused_block_matrix`, applied by the CPU dense engine, the CUDA dense engine and the disk tier; every dense estimate charges the blocks, the plan and the block transient; `explain()` prints `fusion=containment blocks= absorbed=` on the region line when a gate was absorbed; `RunOptions::fuse_gates` (default on; Python `fuse_gates`, CLI `--fusion on|off`); the Phase 21 fixtures keep their baseline with fusion off and gain fused copies pinned beside them; the golden corpus is regenerated on both platforms with one winner change (`counts_factorized.small`, MPS to factorized, the fused dense regions being cheaper). Gate evidence (section 17.7): through `menos::run` the fused sweep is 2.12–2.47× (Linux) and 2.20–2.25× (Windows) the unfused one on every thread at 20–24 qubits, against the 2.0× the owner set from step 1; the largest amplitude difference is 2.2 × 10⁻¹⁷; the fused sweep is bitwise identical across thread counts and between the in-memory engine and the disk tier. Verification (section 17.8): on the final commit `195fc0b`, `scripts/verify.sh` (WSL, the `cuda12` GPU tests executing) and `scripts/verify-windows.ps1` both exit 0, on 2026-11-10. The planner-accuracy branch is verified the same way on `05056f8` (section 16.13). **Gate: OPEN** (the owner's).

#### Stabilizer sampler fix (owner addition, 2026-11-09)

**Status (2026-11-09).** The owner placed a correctness fix in this phase: the stabilizer engine's counts sampler returned impossible outcomes on some Clifford circuits (seen on 2026-11-05 at 40–100 qubits in the cross-simulator comparison, reproduced by `docs/benchmarks/cross-simulator-2026-11-05/harness/stabilizer_repro.{py,json}`, and on 2026-11-07 from 8 qubits up: a qubit whose exact marginal is 0.5 came out 1 in every shot). Cause (`docs/adr/0028` section 18): the sampler combined each selected generator into the base outcome with OR instead of XOR, so a base bit of 1 could never flip to 0; the tableau, `reduce()` and the expectation path were right, which is why single-qubit measurements and expectation values agreed with other simulators. Fixed the same day: `BitString::operator^=` and the sampler flipping with it; pinned by the minimal anti-correlated pair (X, H, CX), the owner's 50-qubit reproducer, and the random-Clifford differential test now drawing from the engine's own sampler and checking every drawn outcome against the dense oracle (it had enumerated `reduce()`'s subspace itself, which is why the bug stayed hidden). Counts from `direct_stabilizer` plans change wherever the bug fired, which includes the two Phase 21 stabilizer counts fixtures: their baseline entries had pinned the buggy sampler's outcomes and are regenerated (the same digests on both platforms, as before, the sampler being integer arithmetic over the same seed); the stabilizer expectation fixture and every other entry are unchanged.

#### Planner performance: the work merged from the contributor's fork (2026-11-05)

**Status (2026-11-05).** Steps 0–2 below were done on a contributor's fork between 2026-11-03 and 2026-11-05 under the name "Phase 21, planner performance" (branch `phase-21-planner-performance` there, from `main` at `d89dde3`), with the design in the fork's "ADR 0025", and reached `main` through pull requests #12 and #13. At the merge with the memory roadmap on 2026-11-05 the work was recorded here, as the start of this phase, and the ADR became `docs/adr/0028` (its section numbering unchanged). Steps 3–6 and the owner's gate are open. What the merge kept, decided and retired, and its verification, are `docs/adr/0028` section 15; the sentence-level record of the fork's work is that ADR's sections 10–13.

**Scope (owner, 2026-11-03).** The latency of one `plan::plan_job` call (`src/plan/planner.cpp`), and also the one-time hardware calibration (`plan::calibrated_hardware_profile`, `src/plan/hardware_profile.cpp`). The execution engines are out of scope.

**Steps.**
0. Discovery, read-only except for documentation: `docs/ARCHITECTURE_NOTES.md` (call graph, hypotheses), this entry, and the draft ADR.
1. Measure before changing anything: a release build with symbols, a reproducible planner benchmark (small/medium/large inputs, fixed seeds, every candidate family, plus cold calibration), and a profile (time per plan, allocation count, peak memory, hotspots). The result is recorded in `bench/BASELINE.md`, and every later claim is compared with it.
2. Algorithmic and data-structure changes that the profile justifies. Each is its own commit with a before/after measurement, and a change that does not measurably help is reverted.
3. A survey of applicable open-source engine techniques (`docs/ENGINE_TECHNIQUES.md`, credited, no copied code). Only techniques the step-1 profile justifies are implemented.
4. A Rust kernel behind a C ABI, built only if profiling justifies it, and only with owner approval (the "Rust service layer, C ABI" item under Deferred work needs an ADR). Kept only if it is clearly faster than optimised C++ after FFI overhead.
5. Correctness: the golden planner corpus (`tests/plan/golden/planner_corpus`, 45 fixed jobs) and the Phase 21 bitwise baseline; ASan/UBSan (and TSan, since planning runs under a mutex-guarded cache); `scripts/verify.sh` and `scripts/verify-windows.ps1`. *The fork's frozen-`d89dde3` differential test was retired at the merge: its reference predates the Phase 21 accounting, which changed every estimate by design (`docs/adr/0028` section 15).*
6. Release: README benchmark table (before/after), `CHANGELOG.md`, a semver bump, CI, and draft release notes. Nothing is pushed, tagged or published; the owner runs the commands.

**Invariants (proposed, `docs/adr/0028` section 3).**
- `plan_job` stays the only planner.
- For every job, its result (strategy, regions, backends, every candidate's feasibility, reason, bytes and work units) and `explain()` output stay byte-identical to the planner without pruning, except that a losing candidate may be reported as pruned (`docs/adr/0028` section 13); the golden corpus pins this.
- No tuning constant is introduced. Calibration still yields figures from the same measurements.
- The Phase 11 replanning, Phase 14 batch and Phase 17 forced-backend invariants hold, and the plan cache (`78bbd49`) still reuses only Direct plans whose structure matches.
- New dependencies are optional, pinned, behind a CMake option, and approved by the owner first.

**Acceptance gate (planner performance part).** To be set by the owner from the step-1 baseline, as in Phases 15 and 16. It is proposed to include: planning-time ratios against `bench/BASELINE.md` on the fixed corpus; byte-identical plans and `explain()` against the golden corpus, up to pruning; sanitizer runs; and `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0.

**Why.** The planner ranks feasible candidates by work units alone and converts them to seconds only to enforce a time budget (`docs/planning/cost-model.md`; `src/plan/planner.cpp`, selection loop). Its time models were last measured against runs in Phase 11 (`docs/adr/0015`): actual over predicted was 0.77–0.88 for dense, 1.4–1.6 for stabilizer, 0.70–1.33 for factorized and 42–1,167 for MPS. Phase 21 replaced the MPS bond estimate, which that measurement used, with a replay of the actual bond dimensions (`docs/adr/0025` Q7), but the MPS time ratio has not been re-measured since, the cost-correlation benchmark does not cover MPS, and the MPS work formula charges every gate at the peak bond cubed although the replay knows each gate's bond. `docs/planning/cost-model.md` still prints the Phase 11 table and the explanation written for the old estimate. Runtime observations are one floor per (engine, backend) and reach only single-engine candidates, because the runtime does not time regions, so a Factorized or CutAndRecombine overrun is refused rather than replanned (Phase 11 deferral). Planning time is reported nowhere, and planning memory is outside the ledger (`docs/adr/0026` section 11.2). Factorization uses the interaction graph's connected components; the cut search allows at most two cuts (`analysis::kMaxCuts`), expectations only, on the CPU.

**Measured on 2026-11-02** (Intel Core Ultra 7 155H, 22 threads, 32 GB, Windows 11, CPU only; MENOS built from this branch against Qiskit Aer 0.17.2 and PennyLane Lightning 0.45.0, every simulator at its defaults, median of three runs after a warm-up; the comparison script is not checked in). Correctness: exact statevectors agreed with both to fidelity 1.000000000000, a 50-term expectation to 5e-16, and noisy counts at 200,000 shots were within sampling distance of Aer's density-matrix probabilities. Structure: MENOS was 5–19× faster than Aer on Clifford circuits of 50–1,000 qubits, 190× faster on a circuit of two independent 12-qubit blocks (the factorized plan), and ran a 40-qubit low-entanglement circuit as MPS in 0.23 s where Aer's automatic method chose statevector and failed on memory. Two gaps, with different causes:
- *Planning cost.* A 20-qubit, depth-10 expectation of 50 Pauli terms took 4.7 s against Aer's 0.27 s, of which execution was 0.33 s and planning 4.4 s: the tensor-network candidate's elimination search ran until it hit the job's memory budget and was then refused. The same circuit as a counts job planned in 2 ms. This is item 2.
- *Dense throughput.* On dense random circuits of 20–24 qubits and depth 20, MENOS was 1.5–2× slower than Aer with every thread, and 6× slower single-threaded at 22 qubits (21.4 s against 3.5 s). Planning was 2 ms there. Aer's binary applies cost-based gate fusion (active in every dense run) and an AVX2 kernel; MENOS applies every gate as its own pass over the state, compiled for baseline x86-64 with no SIMD path. These are items 7 and 8; fusion is a recorded Phase 15 deferral (`docs/adr/0019` section 6).

**Order of the remaining items (owner, 2026-11-06).** After items 1, 2, 3 and 9 (above), the remaining items go in this order. Each item stays a proposal until the owner's kickoff, with its own ADR (or an extension of `docs/adr/0028`), answers to its questions, and plan approval.

1. **Item 7, gate fusion in the dense CPU engine.**
   - **Why first:** it is the largest measured user-visible gap: dense 20–24-qubit circuits ran 1.5–2× slower than Aer on every thread and 6× slower single-threaded.
   - **Why before 8 and the model fixes:** it changes which kernels matter.
   - **Kickoff decisions:**
     - the fusion window k;
     - whether fusion is on by default or opt-in;
     - whether the Phase 21 bitwise baseline is regenerated.
   - **Step 1** must first show that the number of passes over the state is what limits speed.
2. **Item 8, SIMD kernels.**
   - **Why after 7:** fusion creates the k ≥ 3 blocks where SIMD helps most.
   - **Constraint:** results must stay bitwise identical to the generic kernel on every thread count.
   - **Kickoff decision:** whether an AVX-512 path is built at all.
3. **A new item, the time-model fixes found in step 1** (`docs/adr/0028` sections 16.10 and 16.11):
   - **MPS:** a measured per-gate fixed cost (actual 20–57× the prediction);
   - **stabilizer:** a cost per gate (up to 30×);
   - **Pauli:** replace the worst-case branching charge (predictions up to 75× too slow);
   - **tensor network:** about 5×;
   - **density matrix:** about 2.5×;
   - **dense:** the gap between the calibration circuit and real gate mixes at 20–24 qubits.

   **Why after 7 and 8:** they change the dense rate and kernels, so the dense model would otherwise be fixed twice. Targets are set from the step-1 baseline.
4. **Item 4, the ranking rule.**
   - **Why after the model fixes:** ranking by predicted seconds is only sound once predictions are. With today's errors of 5–30× on several engines, ranking by seconds would choose worse plans than ranking by work units.

**Parked: items 5 and 6,** until a workload needs them:
- **Item 5, planning budget and reuse.** Planning measured 2–106 ms on Linux, and the plan cache exists.
- **Item 6, structural plans.** The planner chose a cut plan on no step-1 workload except the dedicated cut fixture.

**Candidate items (proposed), in the originally recommended order.**
1. **Measure before modelling.** Add MPS to `cost_model_correlation_benchmark` over several circuit families (not only the H/T/CX layers of Phase 11), on the current bond replay; record actual over predicted per engine in the ADR; refine the MPS work formula to per-gate bonds from the replay only if the measurement calls for it; replace the Phase 11 table in `docs/planning/cost-model.md` with the new one.
2. **Report and bound planning cost.** `plan_seconds` on `Result` and on the `explain()` `actual` line, measured around `plan_job`; a separately named planning-memory peak (`docs/adr/0026` section 11.2) if the owner places it here rather than in Phase 23. *Done on the fork (`docs/adr/0028` section 13):* the tensor-network elimination search, the cut costing and the per-region engine choice stop as soon as their running total exceeds the cheapest feasible candidate already in hand, and `explain()` reports the candidate as pruned with the figure it reached. Still open: a node-count or time cap derived from measurement, and the `plan_seconds` report.
3. **Per-region observations.** Time each region in the factorized and cut executors, attribute the observation per (engine, backend), and let a multi-region overrun replan through the existing Phase 11 path instead of being refused. Observations can only raise a figure, as today.
4. **Ranking rule** (owner decision, see below). Options: (a) work units, as today; (b) predicted seconds when the job carries a time budget, work units otherwise; (c) predicted seconds always. Under (c) the chosen plan varies with calibration, which differs about 1.5× between processes. An uncertainty band on a prediction is in scope only with a measured basis per engine; no invented constant.
5. **Planning budget and reuse.** A planning time budget, if a caller needs one, and cached structural analysis for repeated circuits (as in `run_batch`): estimates may be cached, feasibility is re-checked against the fresh memory figures every `plan_job`. *Done on the fork, in part:* the plan cache (`src/plan/plan_cache.h`) reuses a Direct, in-memory plan across calls that differ only in gate angles; since the merge its key carries the hardware profile's memory figures, so a stale memory figure can never serve a cached plan.
6. **Structural plans.** Separator discovery beyond connected components and more than two cuts, only if a workload needs them, with the search's own time and memory counted in the candidate's cost.
7. **Gate fusion in the dense CPU engine** (added 2026-11-02; deferred from Phase 15, `docs/adr/0019` section 6, whose four requirements stand). Combine runs of neighbouring gates acting on at most k qubits into one k-qubit matrix before the sweep, so the state is streamed once per fused block instead of once per gate. The window k is the owner's and is measured, not assumed (Aer's default is 5; in MENOS k ≥ 3 runs on the generic kernel). Fusion changes the rounding, so fused results equal the unfused path within `src/common/tolerance.h`, not bitwise: the Phase 21 bitwise baseline (`tests/memory/data`) is regenerated for the fused path, or fusion is opt-in, an owner decision at kickoff. The cost model charges the fused circuit (fewer sweeps, denser matrices), the Phase 11 safe boundary becomes the fused block and `completed` counts fused blocks, `explain()` prints the fusion outcome (blocks, window, gates absorbed), and the unfused path stays as the oracle. Step 1 is a benchmark showing the sweep count is the bound before any fusion code, as the Phase 15 process required.
8. **SIMD kernels for the dense CPU engine** (added 2026-11-02; after item 7, because fusion changes which matrix sizes matter). The CPU kernels (`src/kernels/cpu/dense_kernels.cpp`) are built for baseline x86-64 with no `-march`/`/arch` flag and no intrinsics. Vectorise across amplitudes, several state indices per instruction, never across a sum, with contraction kept off (`src/kernels/CMakeLists.txt`), so every lane performs the same correctly rounded operations in the same order as the scalar code and results stay bitwise identical to the generic kernel, pinned by a test on every thread count. Runtime dispatch on `cpuid` between the baseline, AVX2 and optionally AVX-512 paths, each in its own translation unit compiled with that target's flag (MSVC `/arch:AVX2` only on that unit), because the wheels must still run on any x86-64. Expected, to be measured in step 1: once the state exceeds the last-level cache the sweep is memory-bandwidth bound, so the gain is largest for small n and for the k ≥ 3 blocks fusion creates, and smaller at 20+ qubits.
9. **A threaded calibration for the dense work rate, and the disk link per direction** (added 2026-11-04 from Phase 23's gate runs, `docs/adr/0027` section 10.3). `work_units_per_second` is measured on a 13-qubit circuit that the CPU kernels run on one thread, while every large dense job runs on every hardware thread: on Linux the disk tier's measured time was 0.35× its prediction, on Windows 0.51–1.17×, and the same rate prices every in-memory dense candidate. Calibrate on a state large enough to be threaded (and record the single-thread rate beside it), and price the disk tier's writes and reads at their own measured bandwidths rather than the slower of the two. The gate's factor for the disk tier is then re-measured on the Phase 23 gate record.

**Owner decisions needed.** The ranking rule (item 4); whether items 5 and 6 are in scope; the factor each time model must meet against measurement, per engine; which circuit families the gate measures; which of the `docs/adr/0026` section 11 items land here; for item 7, the fusion window k, whether fusion is on by default or opt-in, and whether the Phase 21 bitwise baseline is regenerated; for item 8, whether an AVX-512 path is built and tested at all (no machine in the verification set has been shown to have it).

**Acceptance gate (proposed).** Every engine's time model is measured on the correlation benchmarks over the owner's circuit families and actual over predicted is within the owner's factor, recorded in the ADR and in `docs/planning/cost-model.md`; `plan_seconds` is reported and tested, and the 50-term expectation case above plans in under a bound the owner sets from the measurement; fused dense results are within `src/common/tolerance.h` of the unfused path on the correlation benchmarks and the Phase 21 baseline circuits, and meet a dense speed-up target the owner sets from the step-1 baseline as same-run ratios (as in Phase 16); the SIMD kernels are bitwise identical to the generic kernel on every thread count; a factorized overrun under a time budget replans with per-region observations rather than being refused, pinned by an injected-clock test; every existing `explain()` golden is unchanged for a job with no time budget (unless the owner chooses ranking (c), in which case the changed goldens are listed); results stay bitwise identical; `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0.

**Invariants to preserve.** `plan_job` is the only planner and the runtime never plans; an observation can only raise a cost figure; no tuning constant without a measured derivation; a job with no time budget keeps today's deterministic plan choice unless the owner decides otherwise at kickoff; the generic kernel stays the oracle, floating-point contraction stays off, every thread count gives bitwise-identical results (Phase 15), and the density-matrix and trajectory engines, which share the kernel, keep passing their tests unmodified.

## Interoperability roadmap (Phases 19–20 complete; 25–30 proposed)

**Status (2026-10-07): proposal only.** The owner asked for MENOS to work with other open-source quantum SDKs, both as frontends (other SDKs submit circuits to MENOS) and as backends (MENOS plans over external engines). Nothing in this section is approved. The owner sets the order, the scope and every threshold at each phase's kickoff. Every phase follows the usual process: an ADR, then plan mode, then owner approval, on a `phase-<N>-<slug>` branch. The roadmap comes after Phase 17 and does not renumber it. *Renumbered by the owner on 2026-10-13 from Phases 18–25 to Phases 19–26, when native Windows became Phase 18; the content is unchanged. Renumbered again by the owner on 2026-10-18: the memory roadmap became Phases 21–23, so the proposed Phases 21–26 became Phases 24–29 with their content unchanged (Stim 24, Cirq 25, adjoint differentiation 26, dynamic circuits 27, cuQuantum 28, further candidates 29). Renumbered a third time by the owner on 2026-11-02: a planner phase became Phase 24, after the memory roadmap, so Phases 24–29 became Phases 25–30 with their content unchanged (Stim 25, Cirq 26, adjoint differentiation 27, dynamic circuits 28, cuQuantum 29, further candidates 30). The Phase 19 and 20 records and their ADRs are kept as written: their "Phase 24" (dynamic circuits, `reset`) meant Phase 27 after the second renumbering and now means Phase 28.*

### Why

MENOS's distinguishing feature is not a single fast engine. It is the planner: analyse the circuit, choose among engines under memory, time and accuracy constraints, replan on overrun, and explain the choice. Interoperability strengthens that in two ways:

- **Frontends** bring MENOS to users of other SDKs: variational algorithms and quantum machine learning (PennyLane), error correction (Stim), Google-style circuits (Cirq), and anything that exports OpenQASM 3.
- **Backends** give the planner more engines to choose from. Each added engine makes every frontend more capable, without changing any frontend.

### Invariants every interoperability phase must preserve

- **`plan_job` stays the only planner.** A frontend adapter translates circuits and results, and never chooses an engine. An external engine is a `SimulationEngine` (`include/menos/simulation_engine.h`) and competes through the existing cost model and its feasibility rules, with calibrated work-unit rates.
- **Never silently approximate, never silently drop.** An adapter that meets a construct it cannot represent exactly (an operation, a noise channel, a measurement form, a classical control construct) refuses it with `UnsupportedFeature` and names the construct. It never ignores it, decomposes it approximately, or substitutes something else.
- **Conventions are written down and tested.** Each adapter's ADR has a conventions table: qubit and wire ordering, bitstring endianness, global phase, parameter conventions (for example `RX(θ)` = exp(−iθX/2)), measurement and reset semantics, and shot semantics. Each row has a test.
- **Differential testing against the SDK's own simulator.** A frontend is accepted only when its results agree with that SDK's reference simulator on a fixture set, within `src/common/tolerance.h` for exact outputs and by a chi-square test for counts (0.001 was used in Phases 13 and 14; the owner confirms it per phase).
- **The existing engine stays as the oracle.** An external engine that overlaps an existing MENOS engine (for example Stim and `StabilizerEngine`) is tested against it, and the MENOS engine is not removed.
- **Optional dependencies only.** The core C++ build and the core Python package never require an external SDK. Python frontends are optional extras (for example `pip install menos[pennylane]`). C++ engine dependencies are pinned by tag or commit (ADR 0001) behind a CMake option. `scripts/verify.sh` must still exercise every integration that is enabled, so a disabled integration is never reported as verified.
- **Licences.** An Apache-2.0, MIT or BSD dependency can be proposed in its ADR. Any other licence (for example NVIDIA cuQuantum's) needs an explicit owner decision before code.
- **Determinism.** The same circuit, options and seed give the same results through any frontend. Where an external engine has its own RNG, the ADR defines how MENOS seeds map to it and whether results are bitwise reproducible or only statistically equal.
- **Jobs that do not use an integration are unchanged.** Plans, results, `explain()` output and CLI golden files stay the same, unless a new engine wins a plan on cost. Any such planner change is measured and recorded, never left implicit.
- **Abstractions follow the second implementation.** The Qiskit adapter is the first frontend. Shared frontend code (a common conversion layer, a plugin registry) may be extracted only when the second frontend (proposed: PennyLane) shows what is truly shared, not before.

### Phase 19: OpenQASM 3 import (frontend) — COMPLETE (acceptance gate passed 2026-10-16)

*The owner started this phase on 2026-10-14 as the first interoperability phase, in parallel with Phase 18 (native Windows, branch `phase-18-windows-native`), whose gate was still open; the owner's instruction set aside "current phase only" for this one case. Design: `docs/adr/0023`. Branch: `phase-19-openqasm3-import`, rebased onto `phase-18-windows-native` on 2026-10-14, so it merges after Phase 18.*

**Goal.** Accept OpenQASM 3 programs in the CLI and the C++/Python APIs. OpenQASM 3 is the common exchange format of Qiskit, Braket, tket and others, so one importer gives MENOS the widest reach per unit of work.

**Required behavior.**
1. Import the static-circuit subset: `qubit`/`bit` declarations, the standard gate library (`stdgates.inc`), user `gate` definitions, gate modifiers that are exact (`inv`, `pow` with integer exponent, `ctrl`, `negctrl`), constant parameter expressions, `measure`, `barrier`, and `input` parameters bound at submission. The exact scope is `docs/adr/0023` section 2.
2. Everything else is refused with its source location: `reset`, classical control flow, `while`, subroutines with classical return values, timing and `delay` semantics, `defcal`/pulse grammar, and `extern`. Dynamic circuits, `reset` included, are Phase 24 (old numbering 23).
3. A program that is also valid OpenQASM 2 gives the same KIR circuit through both importers.

**Owner decisions (2026-10-14).** A hand-rolled parser; `reset` refused and deferred to Phase 24; `input` parameters in scope; a modifier with no exact gate-table equivalent becomes an exact `UnitaryOp` of at most `kMaxUnitaryQubits` qubits. Q5–Q8 (same day): the OpenQASM 2 importer keeps source order; the CLI chooses the importer from the header; Python `menos.from_qasm3(text, inputs=None)`; the `stdgates.inc` names as built-in gates with Qiskit's matrices (first answered "embed verbatim", revised after `docs/adr/0023` section 4.1 measured that the file's bodies, taken literally, disagree with the gates they name). Q9, the gate, is settled with the implementation plan.

### Phase 19 acceptance gate

Approved by the owner with the implementation plan on 2026-10-14 (`docs/adr/0023` section 9, Q9). It refines the proposal above it ("statevectors equal to Qiskit's within tolerance") after `qiskit.qasm3.dumps` was measured to drop a circuit's global phase: the oracle is Qiskit's own import of the same text.

1. **Subset.** Every construct in `docs/adr/0023` section 2.1 is imported; each has a C++ test.
2. **Refusals.** Every construct in section 2.2 is refused with `UnsupportedFeature` naming it, its line and its column; each has a test.
3. **Built-in gates.** All 32 `stdgates.inc` names and `U` equal Qiskit's matrices entrywise within 1e-12, global phase included.
4. **Modifiers.** Every `inv`/`pow`/`ctrl`/`negctrl` rule is exact, entrywise within 1e-12, including the `UnitaryOp` path and the refusal above 5 qubits.
5. **Differential.** For every fixture exported with `qiskit.qasm3.dumps`, MENOS's statevector equals that of `qiskit.qasm3.loads` of the same text within 1e-12, global phase included, and equals the original circuit's up to one global phase factor; seeded counts equal `MENOSBackend`'s exactly.
6. **OpenQASM 2/3 equivalence.** The paired fixtures give identical circuits.
7. **OpenQASM 2 measure order.** `qasm2::parse` keeps source order, with a regression test; the existing OpenQASM 2 tests and CLI golden tests pass unmodified.
8. **Entry points.** CLI header dispatch and `--input`, and Python `from_qasm3`, each tested.
9. **Verification.** `scripts/verify.sh` exits 0 in the WSL clone on the final commit; documentation updated; every new symbol has a production caller and tests.

**Gate record (2026-10-16):** every criterion has evidence in `docs/adr/0023` section 10, and `scripts/verify.sh` exits 0 on `bb720d8` (dev/release 679/679, pytest 313 passed, every CUDA stage 734/734 with GPU tests executing in the CUDA 12 builds); `scripts\verify-windows.ps1` also exits 0 there. The owner accepted the gate on that evidence. All nine criteria pass. Phase 19 is COMPLETE.

### Phase 20: PennyLane device (frontend) — COMPLETE 2026-10-18

*The owner started this phase on 2026-10-16, while Phase 18's gate is still open, and answered `docs/adr/0024` section 9 the same day with every recommended option. Branch: `phase-20-pennylane-device`, from `main` at `887b987`. The owner decisions and the gate below supersede the proposal text above them where they differ.*

**Owner decisions (2026-10-16).**
- `pennylane==0.45.1`, exact-pinned, CPython 3.11–3.14 only.
- Noise channels are in scope, through PennyLane's own Kraus matrices.
- A non-exact result under `shots=None` is always refused.
- `MENOSBackend.max_circuits` stays 1.
- Each required run is one `menos.run` call, so item 3 reads "each circuit planned alone by `plan_job`" instead of "through `run_batch`".
- Analytic `var` is ⟨O²⟩ − ⟨O⟩².
- Seeds come from one NumPy generator per device.
- Finite-shot measurements follow PennyLane's semantics.
- Refusals raise an exception that is both `UnsupportedFeatureError` and `DeviceError`, if measured feasible.
- No shared frontend code is extracted.

**Phase 20 acceptance gate** (`docs/adr/0024` section 9, Q11):
1. Conventions: one test per row.
2. Gate table and the `UnitaryOp` path within 1e-12 of PennyLane's matrices, global phase included.
3. Analytic `state`/`probs`/`expval`/`var` within 1e-12 of `default.qubit`.
4. Finite-shot chi-square at a family-wise 0.001; shot vectors; seeded determinism.
5. The parameter-shift VQE gradient within 1e-12 of `default.qubit`'s.
6. Noise against `default.mixed` within 1e-12, with chi-square for counts.
7. Every refusal by name.
8. PennyLane's device test suite passes for the supported features, with every skip listed.
9. The entry point works from the installed wheel.
10. Existing suites unchanged; `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0 on the final commit; documentation updated; every new symbol has a production caller and tests.

**Gate record (2026-10-18):** every criterion has evidence in `docs/adr/0024` section 10. On `b942fd6`, two refusals of criterion 7 fell short: `backprop`/`adjoint` were refused by PennyLane rather than the device, and program capture and `qml.qjit` had no refusal. At the owner's request they were fixed in `5da898e`, with no C++ change (section 10.6). `scripts/verify.sh` (dev/release 679/679, pytest 554 passed, every CUDA stage 734/734) and `scripts\verify-windows.ps1` (680/680, pytest 554 passed) both exit 0 on `5da898e`. The owner had instructed on 2026-10-17 that the gate be marked passed once everything passed. Phase 20 is COMPLETE.

**Goal.** A PennyLane device (`menos.qubit`, registered by an entry point) built on PennyLane's `qml.devices.Device` API, so PennyLane users can run circuits, variational algorithms and gradients through MENOS's planner.

**Required behavior.**
1. Measurements: `expval` and `var` of Pauli-sum observables (through the existing observable path, so light-cone reduction and Pauli propagation stay available), `probs`, `sample`, `counts` and `state`.
2. `shots=None` returns exact values only when the chosen plan is exact. A truncated MPS or a trajectory plan is never returned as an analytic value without an explicit opt-in (decision below).
3. Batches of tapes go through `run_batch`, so parameter-shift gradients (computed by PennyLane) are planned circuit by circuit with no second planner.
4. Wire labels map to MENOS qubits by an explicit, documented rule. PennyLane operations without an exact MENOS equivalent are refused, or decomposed by PennyLane's own exact decomposition, never approximated by MENOS.
5. Optionally, PennyLane noise channels map to the Phase 12 channels where they are mathematically identical.

**Owner decisions needed.** The minimum PennyLane version; whether noise channels are in scope; how a non-exact plan behaves under `shots=None` (refuse, or allow behind an option that reports the error); whether `MENOSBackend.max_circuits` (Phase 14 deferral) is lifted alongside.

**Acceptance gate (proposed).** Every measurement type agrees with `default.qubit` on a fixture set within tolerance (exact) or by chi-square (counts); the parameter-shift gradient of a VQE fixture agrees with `default.qubit`; PennyLane's device test suite passes for the supported features; unsupported operations are refused by name; the Qiskit adapter's tests pass unmodified; `scripts/verify.sh` passes with the extra installed.

### Phase 25: Stim engine for noisy Clifford circuits (backend) — PROPOSED

**Goal.** Add Stim (Apache-2.0, C++) as a `SimulationEngine` for Clifford circuits with Pauli noise. Noisy stabilizer simulation was deferred in Phase 13 (`docs/adr/0017`). With this engine the planner can run error-correction-scale noisy Clifford circuits (thousands of qubits, millions of shots), which the density-matrix and trajectory engines cannot.

**Required behavior.**
1. A `StimEngine` that accepts a circuit only when every operation is Clifford and every noise channel is a Pauli channel Stim represents exactly (depolarizing, Pauli, bit- and phase-flip, readout flip). Anything else is infeasible for this engine, and the planner chooses another one.
2. Counts and Pauli expectation outputs. The noiseless `StabilizerEngine` stays as the oracle for noiseless circuits.
3. Calibrated work-unit rates, so Stim competes on cost. Every planner decision that changes is recorded.
4. A documented mapping from MENOS seeds to Stim's RNG. Results are deterministic for a fixed seed but not bitwise equal to the trajectory engine; agreement with it is statistical.

**Owner decisions needed.** Linking the C++ library (FetchContent, pinned) or calling Stim from Python (which would put the engine outside `plan_job`, so it is not recommended); whether Stim circuit import (`.stim`, with `DETECTOR`/`OBSERVABLE_INCLUDE`) and detector sampling are in this phase or a later frontend (detector samples would be a new output kind).

**Acceptance gate (proposed).** On small fixtures, Stim counts agree with the Phase 12 density matrix by chi-square; noiseless results agree exactly with `StabilizerEngine`; non-Clifford or non-Pauli circuits are never sent to Stim; the planner chooses Stim for a large noisy-Clifford fixture that no other engine can run within budget, and `explain()` says why; `scripts/verify.sh` passes.

### Phase 26: Cirq frontend — PROPOSED

**Goal.** A Cirq simulator (`menos.cirq.MENOSSimulator`, implementing `cirq.SimulatesSamples` and `cirq.SimulatesExpectationValues`) that converts Cirq circuits directly. Cirq's own QASM export is OpenQASM 2, which loses some gates and parameters, so a direct converter is more faithful.

**Required behavior.** Line and grid qubits map to MENOS qubits by a documented rule; `cirq.measure` keys map to result records; Cirq noise channels map to Phase 12 channels only where they are identical; anything else is refused by name.

**Acceptance gate (proposed).** Agreement with `cirq.Simulator` on a fixture set (tolerance for exact outputs, chi-square for samples); refused constructs tested; `scripts/verify.sh` passes with the extra installed.

### Phase 27: Adjoint differentiation — PROPOSED

**Goal.** Gradients of expectation values in one forward and one backward pass, for the dense CPU and CUDA engines. Parameter-shift needs 2 circuits per parameter. Adjoint differentiation is what users of PennyLane's `lightning.qubit` expect, and it makes MENOS practical for variational workloads.

**Required behavior.** A public gradient entry point (C++ and Python), used by the PennyLane device as `diff_method="adjoint"`; refusal for noisy circuits, shot-based execution and non-dense plans, with the reason given; parameter-shift remains the oracle.

**Acceptance gate (proposed).** Adjoint gradients agree with parameter-shift and with finite differences within tolerance on a fixture set; the speed-up over parameter-shift is measured and recorded; `scripts/verify.sh` passes.

### Phase 28: Dynamic circuits — PROPOSED

**Goal.** Mid-circuit measurement, reset, and classical feed-forward (`if` on measured bits). Qiskit, OpenQASM 3, PennyLane (`qml.measure`, `qml.cond`) and Cirq all express these, and the frontends above must refuse them until this phase.

**Required behavior.** KIR support for classical bits and conditioned operations; engine feasibility rules (dense and density-matrix engines branch or sample exactly; engines that cannot represent a construct are infeasible); per-shot semantics documented; the frontends of Phases 19, 20 and 25 lift their refusals for the supported subset.

**Owner decisions needed.** The supported subset (single-bit conditions, register comparisons, loops); whether exact branch-weighted outputs or sampled outputs are required; how replanning interacts with partially executed shots.

**Acceptance gate (proposed).** Agreement with Aer's dynamic-circuit support on a fixture set (teleportation, active reset, repeat-until-success); existing static-circuit results unchanged; `scripts/verify.sh` passes.

### Phase 29: NVIDIA cuQuantum engines (backend) — PROPOSED

**Goal.** cuStateVec and cuTensorNet as additional CUDA engines, competing with MENOS's own CUDA dense engine and CPU tensor-network engine on cost. Phase 8's engine stays closed and stays as the oracle; this adds engines beside it and does not reopen it.

**Required behavior.** Owner decision on the cuQuantum licence first. Then: engines behind a CMake option; calibrated rates; a measurement showing where each engine wins; bitwise or tolerance agreement with the existing engines, as the ADR defines.

**Acceptance gate (proposed).** Differential agreement with the existing engines; measured crossovers recorded; planner changes recorded; GPU tests that actually execute; `scripts/verify.sh` passes.

### Phase 30: Further frontends and engines (candidates) — PROPOSED

The owner chooses from these at kickoff, using what Phases 19–29 have shown:

1. **pytket** (Quantinuum TKET) backend.
2. **Amazon Braket** local-simulator interface.
3. **MQT** (Munich Quantum Toolkit) integration, for example its decision-diagram simulator as an engine for highly structured circuits.
4. **Stim circuit import with detector sampling**, if it was not in Phase 25.
5. **A plugin registry for third-party engines and frontends**, only if at least two frontends and two external engines exist and show a common shape.

### Contributing a frontend or an engine

The owner plans to write some frontends and engines directly. This checklist is the contract each one meets before its gate.

**Frontend adapter.**
1. ADR: the target SDK and minimum version; its licence; the conventions table; the exact supported subset, and the refusal for everything else.
2. Code in `python/menos/<sdk>/` (or `src/<format>/` for a file format), translating to the existing API. It does not call engines or plan.
3. Tests: one per conventions-table row; differential tests against the SDK's own simulator; a refusal test for each unsupported construct; determinism under a fixed seed.
4. Documentation in the README and a runnable example covered by `test_examples.py`.

**Backend engine.**
1. ADR: the library, version pin and licence; which circuits the engine is feasible for, stated as a rule the planner can check; its error model (exact, bounded, or statistical); its seeding; and the measurement showing a workload where it beats every existing engine. Without that measurement it is not added (prime directive 2).
2. A `SimulationEngine` subclass in `src/engines/<name>/`, a cost-model entry and calibrated rates.
3. Tests: differential tests against the existing MENOS engine that covers the same circuits; feasibility tests (it is never chosen for a circuit it cannot run exactly); a planner test showing when it wins, with the `explain()` output.
4. Every planner decision that changes is recorded in the ADR.

## Deferred work

Rust service layer, C ABI, sparse engine, distributed/multi-GPU, ROCm, lineage cache, cloud API and other unscheduled platform work remain deferred and require owner approval plus an ADR. Density matrix and noise support moved out of this list into the Phase 12–14 noise roadmap on 2026-10-01; noisy MPS and tensor-network engines, CUDA density-matrix support, and noise-model fitting from device calibration data stay deferred. On 2026-10-07 the PennyLane device and OpenQASM 3 moved out of this list into the proposed interoperability roadmap (Phases 19–26), which is not yet approved. On 2026-10-13 native Windows (x64, MSVC, CPU) moved out of this list into Phase 18; CUDA on Windows, Windows ARM64 and other Windows toolchains stay deferred. On 2026-10-18 the owner placed memory accounting, CUDA memory pools and pinned memory, and tiered VRAM/RAM/disk execution in the memory roadmap (Phases 21–23); multi-GPU and the sparse engine stay deferred, and the proposed interoperability phases are now numbered 24–29. On 2026-11-03 planner performance became Phase 21, the memory roadmap Phases 22–24 and the interoperability proposal Phases 25–30.

## Phase 11 kickoff (historical: used; Phase 11 is complete)

```text
We are starting Phase 11: Runtime monitoring and replanning.

Read CLAUDE.md completely, then read docs/BUILD_PLAN.md and the Phase 11 specification.

Phase 10 is complete and frozen. Do not reopen Phase 10 except for a compatibility regression that directly blocks Phase 11.

START IN PLAN MODE. Do not write implementation code yet.

Inspect the current runtime, planner, Plan IR, explain(), execution engines, and CPU/CUDA backend selection.

Report:
1. exact files to add/change;
2. safe runtime execution boundaries for monitoring;
3. quantities already predicted by the plan/cost model that are measurable;
4. candidate monitoring signals;
5. measured evidence from a realistic workload showing prediction deviation;
6. measured restart/re-execution cost;
7. proposed threshold semantics;
8. telemetry/replan data model;
9. exact replan path through the existing planner;
10. explain() changes;
11. public API changes, if genuinely required;
12. required ADRs;
13. acceptance criterion → test/command mapping;
14. risks, ambiguities and any point requiring a guess.

Do not implement checkpointing, state migration, heterogeneous per-region CPU/CUDA assignment, specialized CUDA kernels, or other future work merely because it may be useful later.

The initial implementation should prefer restart/re-execution unless measured evidence proves restart overhead is materially expensive.

Do not create a second planner inside the runtime.

Preserve:
- CPU/CUDA backend selection;
- HardwareProfile calibration;
- strict time_budget semantics;
- exactness/accuracy semantics;
- existing explain output when no replan occurs.

Do not change the public API unless the architecture genuinely requires it.

WAIT FOR MY APPROVAL after the plan.

After approval:
- implement in small coherent commits;
- add regression tests with the implementation;
- run scripts/verify.sh;
- walk every Phase 11 acceptance criterion with concrete evidence;
- do not start later phases.

Do not modify CLAUDE.md's Current phase line to claim Phase 11 completion. The owner controls the gate.

End with:

Changes:
Verification:
Not verified:
Open questions:
Acceptance:
```
