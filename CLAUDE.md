# CLAUDE.md

Project: **MENOS** (Qaulium AI). A quantum circuit simulation runtime.

## What this project is

MENOS takes a quantum circuit (primary entry point: a Qiskit `BackendV2`), analyzes its structure, estimates candidate execution plans, and runs a plan satisfying requested output, accuracy, memory and time constraints. It is a compiler, planner, runtime and simulation-engine system; it has no ML/LLM component and is not a general quantum SDK.

The authoritative plan is `docs/BUILD_PLAN.md`. Read the current phase specification before starting work.

## Current phase

**Phase 20 is COMPLETE: the owner instructed on 2026-10-17 that its gate be marked passed once every criterion passed, and it was recorded passed on 2026-10-18, on the evidence in `docs/adr/0024` section 10 (`scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0 on `5da898e`), recorded in section 10.7. Phase 19 is COMPLETE (gate accepted 2026-10-16, `docs/adr/0023` section 10.4). Phase 18 (native Windows) is COMPLETE (gate accepted 2026-10-18 on the evidence on `7fa38ce`, `docs/adr/0022` section 10.4). Phases 12–17 are COMPLETE. On 2026-10-18 the owner placed the memory roadmap (Phases 21–23: memory accounting and copy removal, CUDA memory efficiency, tiered VRAM/RAM/disk memory) next in `docs/BUILD_PLAN.md`; Phase 21 is NOT STARTED and starts only on the owner's instruction. The rest of the interoperability roadmap, renumbered to Phases 24–29, is a proposal only, and the owner sets its order and scope.**

Platform note (2026-10-02, `docs/adr/0014` section 17): the CUDA backend supports toolkits 12.0 and later. On the WSL machine (driver 566.14), only the `cuda12-dev`/`cuda12-release` presets reach the GPU; `scripts/lsan-wsl-libcuda.supp` suppresses, by path, a leak inside WSL's CUDA driver shim. Git inside WSL has no identity configured; commits use the repository's existing author.

The rest of this section is the Phase 11 closure record, kept as written.

Phase 11 was verified on its implementation/gate commit `322eca5`: CPU dev/release 496/496; CUDA dev/release 528/528 (CUDA 13.4.92, RTX 3050 Laptop GPU, compute capability 8.6, driver 595.91.07); Python 74 passed, 1 deselected; `scripts/verify.sh` exit 0. Full evidence is in `docs/adr/0015` and `docs/BUILD_PLAN.md`.

Repository state at closure: branch `phase-11-monitoring-replanning`; `322eca5` is the final Phase 11 implementation and gate commit; `9a0f0f7` ("Close Phase 11 documentation and record final verification") is the documentation closure commit; no implementation, source, test or build file has changed since `322eca5`; working tree clean. No Phase 12 branch, ADR, design or code exists.

Owner decisions confirmed at closure:
- `time_budget_seconds` bounds **one execution attempt**, not the total job. A job that replans can take several times its budget in total; the abandoned attempt is reported in `Result::replans()` and `explain()`, never hidden.
- If no feasible plan remains after an overrun, MENOS throws `ResourceExceeded`, even if the abandoned attempt was near completion.
- The CUDA-overrun → CPU recovery fix (`361ab12`) is part of Phase 11.

Phase 11 deferrals are intentional: state migration/checkpointing (restart was measured cheap for executions of about a second or more, see `docs/adr/0015` section 10; requires measurement plus an ADR first); observed-runtime floors for `Factorized`/`CutAndRecombine` plans (elapsed time is not attributed per region, so such an overrun is refused); per-operation boundaries in the tensor-network engine (its only boundary is per observable term); per-shot monitoring of the dense sampler; a CLI time-budget flag; heterogeneous per-region CPU/CUDA selection; specialized 1Q/2Q CUDA kernels; explicit Python CUDA selection; multi-GPU/ROCm/other architectures and other unscheduled platform work. The Phase 10 deferral "Python exposure of `time_budget_seconds`" was **lifted** in Phase 11 (`f9f908a`); the Qiskit `BackendV2` adapter still does not set it.

Update this Current phase line only after the next phase's gate is confirmed by the owner. Do not mark a phase complete yourself.

## Prime directives

1. Correctness before performance.
2. No speculative code.
3. Abstractions follow the second implementation.
4. Current phase only.
5. Never silently approximate.
6. Evidence before claims.
7. Do not weaken acceptance criteria.

## Working agreement

- Tasks touching >2 files or public API start in plan mode.
- Plan mode identifies exact files, APIs, acceptance evidence, risks and ambiguities.
- If mathematics, numerical thresholds or externally observable API behavior requires guessing, stop and ask.
- Use `phase-<N>-<slug>` branches.
- Small coherent commits; relevant verification after each.
- Never force-push/reset --hard/delete work without owner approval.
- Do not claim a phase complete until every criterion has evidence and the owner confirms the gate.
- Do not modify the Current phase line after completing a phase; owner controls the transition.

## Phase 11 record (frozen)

**Runtime monitoring and replanning.** MENOS observes wall-clock time against a job's `time_budget_seconds` at safe execution boundaries (between gates, observable terms and sampled shots). On a crossing it abandons the run, records what the running engine actually cost as a lower bound, re-runs the existing `plan_job` with it, and restarts under the new plan or throws `ResourceExceeded` if none fits. `Result::explain()` reports each replan's threshold, observation, prediction, reason, old plan and new plan.

Invariants later work must preserve:
- `plan_job` is the only planner; the runtime never plans and never interprets the monitor.
- The threshold is the caller's own budget. No tuning constant is introduced.
- An observation can only raise a cost figure, never lower it.
- A job with no time budget installs no monitor, and its behaviour and `explain()` output are unchanged.
- Checkpointing/state migration requires a measurement showing restart is materially costly, and an ADR first.

## Phase 12

**Noise channels and exact density-matrix simulation.** The owner set its scope on 2026-10-01, together with the Phase 12–14 noise roadmap, and approved its plan and `docs/adr/0016` the same day. Specification, owner decisions and acceptance gate: `docs/BUILD_PLAN.md`, Phase 12. Branch: `phase-12-noise-density-matrix`. Final code commit `96a7cd0`. **COMPLETE (frozen): the owner closed the gate on 2026-10-02** after criterion 10 was re-verified on `e546581` (`docs/adr/0016` section 12). Builds and `scripts/verify.sh` run in the WSL clone `~/MENOS`, with `UV_PYTHON_PREFERENCE=only-managed`.

## Phase 13

**Noisy trajectory simulation.** Specification and gate: `docs/BUILD_PLAN.md`, Phase 13. Design: `docs/adr/0017`, whose section 9 the owner answered on 2026-10-02 (explicit opt-in; `expectation_precision` as proposed; noisy stabilizer and Pauli propagation deferred; the section 6 thresholds as proposed). Branch: `phase-13-noisy-trajectories`. The owner approved the implementation plan on 2026-10-02 (also deciding: `MENOSEstimator(seed=None)` draws a fresh seed per run and records it; precision opts in only the expectation, so counts need `NoiseMethod::Auto`). Implemented in commits `58d89df`–`53de60b`, documented in `847cb44` and `63cf08e`. **COMPLETE (frozen): the owner accepted the gate on 2026-10-03**; evidence in `docs/adr/0017` section 10.

## Phase 14

**Readout-error mitigation, batch execution and SamplerV2.** Specification and gate: `docs/BUILD_PLAN.md`, Phase 14. Design: `docs/adr/0018`. The owner started the phase and answered section 9 on 2026-10-07 (readout mitigation, not ZNE; `SamplerV2` in scope; calibration under the full noise model; target plus first-order calibration uncertainty; mitigated data in `PubResult.metadata["readout_mitigation"]`; the section 6 thresholds). Branch: `phase-14-mitigation-batch`, merged to `main` in `bdc0d8e`. Implemented in commits `48d75f2`–`75ad45f`, evidence in `96def14`. **COMPLETE (frozen): the owner accepted the gate on 2026-10-07**, including the added criterion 6 (Sampler against Aer `SamplerV2`). Invariants: mitigation is opt-in only; `run_batch` never plans (each circuit goes through `run`/`plan_job`); single-circuit `run` and `MENOSBackend.run` keep their seeds; `MENOSBackend.max_circuits` stays 1 (list submission is a recorded deferral).

## Phase 15

**Dense CPU engine performance.** Specification and gate: `docs/BUILD_PLAN.md`, Phase 15. Design: `docs/adr/0019`. The owner started the phase on 2026-10-07, answered section 9 the same day and set the gate targets on 2026-10-08 (N = 4, M = 5, plus three conditions). Branch: `phase-15-dense-cpu-performance`. **COMPLETE (frozen): the owner accepted the gate on 2026-10-08**, on the evidence in `docs/adr/0019` sections 10.9–10.10, ruling that the extension of the benchmark-tool test `python/tests/test_dashboard.py` does not count against criterion 5. Invariants: the specialised 1- and 2-qubit kernels stay bitwise identical to the generic kernel, which remains the oracle; every thread count gives bitwise-identical results; expectation sums and the sampling cumulative distribution stay single-threaded; the default thread count is every hardware thread, capped by `RunOptions::max_threads`; MENOS does not set `OPENBLAS_NUM_THREADS`; gate fusion is deferred.

## Phase 16

**Specialised CUDA kernels.** Specification and gate: `docs/BUILD_PLAN.md`, Phase 16. Design: `docs/adr/0020`. The owner started the phase on 2026-10-08, answered section 9 the same day, and set the gate targets from the step-1 baseline. After step 2 showed that bitwise-identical kernels are FP64-bound on this GPU, the owner reset them to same-run ratios against the generic kernel. Branch `phase-16-specialised-cuda-kernels`, merged to `main` in `c6b9c87`. **COMPLETE (frozen): the owner accepted the gate on 2026-10-09**, ruling that the single miss (criterion 1, n = 8, gate run b, 1.39× against 1.4×) does not count against it (`docs/adr/0020` section 13.1).

Invariants:
- `kernels::cuda::apply_matrix` sends k = 1 and 2 to specialised kernels that are bitwise identical to `apply_matrix_generic`, which remains the oracle and serves k ≥ 3.
- The specialised kernels take the matrix as a kernel argument (no per-gate copy).
- `-fmad=false` stays.
- One synchronisation per gate stays (the Phase 11 monitor relies on it).
- Diagonal and permutation kernels, asynchronous launches and a calibration-method change are recorded deferrals, each needing an ADR first.
- On this GPU the gate kernels are FP64-bound; their speed-ups are measured as same-run ratios, because the GPU's clock state varies about 2× between runs.

## Phase 17

**API surface: recorded deferrals.** Specification and gate: `docs/BUILD_PLAN.md`, Phase 17. Design: `docs/adr/0021`. The owner started the phase on 2026-10-09 with all four candidates in scope and answered section 9 the same day. Branch `phase-17-api-deferrals`, fast-forwarded into `main` at `0af513e`. **COMPLETE (frozen): the owner accepted the gate on 2026-10-10** (`docs/adr/0021` section 11.1).

Invariants:
- A forced backend (`RunOptions::backend`, `MENOSBackend` `backend`, CLI `--backend`) never falls back: `backend = cuda` raises `UnsupportedFeature` or `ResourceExceeded`, and survives a replan.
- An unset backend leaves planning and `explain()` byte-identical.
- `MENOSBackend` accepts `time_budget_seconds`; the CLI takes `--time-budget SECONDS` on `run` and `explain`; a non-finite or non-positive budget is refused.
- Per-region CPU/CUDA selection applies to `Factorized` plans only, uses the planner's calibrated comparison with ties to the CPU, and counts stay bitwise identical. `CutAndRecombine` regions stay on the CPU (measured, not adopted).
- Recalibrating to follow the GPU's run-to-run speed variation is not part of Phase 17.

## Phase 18

**Native Windows (MSVC).** Specification and gate: `docs/BUILD_PLAN.md`, Phase 18. Design: `docs/adr/0022`. The owner started the phase on 2026-10-13 as the next phase, renumbered the proposed interoperability roadmap to Phases 19–26, answered section 9 and approved the plan the same day. Branch `phase-18-windows-native`, merged to `main` in `8742433`. **COMPLETE (frozen): the owner accepted the gate on 2026-10-18** on the evidence in `docs/adr/0022` section 10 (`7fa38ce`), recorded in section 10.4.

Scope decisions: Windows x64, MSVC, CPU only (CUDA on Windows is refused at configure time); wheels for CPython 3.10–3.14 `win_amd64`; MSVC ASan in the `windows-dev` preset, with UBSan/LSan coverage staying in the Linux `verify.sh`; LAPACK from the official OpenBLAS 0.3.34 x64 (LP64) release, pinned by SHA-256 and shipped in the wheel; any Windows-only last-bit numeric failure goes to the owner, never a tolerance change. Windows verification: `scripts/verify-windows.ps1`; Linux verification stays `scripts/verify.sh` in the WSL clone.

## Phase 19

**OpenQASM 3 import.** Specification and gate: `docs/BUILD_PLAN.md`, Phase 19. Design: `docs/adr/0023`. The owner started the phase on 2026-10-14, in parallel with Phase 18 (native Windows, branch `phase-18-windows-native`, gate still open), and answered `docs/adr/0023` section 9 Q1–Q8 the same day: a hand-rolled parser; `reset` refused and deferred to Phase 24; `input` parameters in scope; a modifier with no exact gate-table equivalent becomes an exact `UnitaryOp`; the OpenQASM 2 importer keeps measurements in source order; the CLI chooses the importer from the header; Python `menos.from_qasm3`; the `stdgates.inc` names as built-in gates with Qiskit's matrices (revised from "embed verbatim" after the section 4.1 measurement). Branch `phase-19-openqasm3-import`, rebased onto `phase-18-windows-native` (`7fa38ce`) at the owner's decision, so it merges after Phase 18. The owner approved the implementation plan and the gate (Q9) the same day. **COMPLETE (frozen): the owner accepted the gate on 2026-10-16** on the evidence in `docs/adr/0023` section 10 (`bb720d8`), recorded in section 10.4.

## Phase 20

**PennyLane device.** Specification and gate: `docs/BUILD_PLAN.md`, Phase 20. Design: `docs/adr/0024`. The owner started the phase on 2026-10-16 and answered section 9 the same day, taking every recommended option. Branch `phase-20-pennylane-device`, from `main` at `887b987`. Implemented in `b942fd6`; the criterion 7 refusals were fixed in `5da898e` at the owner's request (`docs/adr/0024` section 10.6). **COMPLETE (frozen): gate recorded passed on 2026-10-18** on the owner's instruction (section 10.7).

Invariants:
- `menos.qubit` (`menos.pennylane.MENOSQubitDevice`) is registered by the `pennylane.plugins` entry point. The extra is exact-pinned to `pennylane==0.45.1`, CPython 3.11 or later.
- Each required run is one `menos.run`, planned alone by `plan_job`; the device never chooses an engine.
- Under `shots=None`, every result is exact or refused; there is no opt-in.
- Every refusal raises `MENOSDeviceUnsupportedError` (both `menos.UnsupportedFeatureError` and PennyLane's `DeviceError`) naming the construct, including `backprop`/`adjoint`/`device` derivatives, program capture and `qml.qjit`.
- PennyLane's own Kraus matrices are used for channels; there is no per-channel convention table.
- No shared frontend code is extracted, and `MENOSBackend.max_circuits` stays 1.

## Definition of done

1. `scripts/verify.sh` exits 0 in the current session.
2. New behavior has regression coverage.
3. New symbols have production callers/public entry points and tests.
4. Public headers are documented.
5. README/ADRs/BUILD_PLAN are updated when behavior or plan changes.
6. Every acceptance criterion has concrete evidence.
7. Final git status and commit list are reported.

## Required end-of-task report

**Changes** — files and behavior.
**Verification** — exact commands/results.
**Not verified** — coverage gaps.
**Open questions** — genuine owner decisions only.
**Acceptance** — every criterion with evidence.

Do not start the next phase unless explicitly instructed by the owner.
