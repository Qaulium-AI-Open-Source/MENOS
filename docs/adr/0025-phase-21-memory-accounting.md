# ADR 0025: Phase 21 — Memory accounting and copy removal

## Status

**Accepted; section 9 answered (2026-10-20).** The owner started Phase 21 on 2026-10-20 as the first phase of the memory roadmap (`docs/BUILD_PLAN.md`, "Memory roadmap (Phases 21–23)"), and the same day took the recommended option for every question in section 9. Branch `phase-21-memory-accounting`, from `main` at `1f89192`. No implementation, source, test or build file has changed yet; the implementation plan goes through plan mode and owner approval next.

Labels follow `docs/adr/0014`–`0024`: **[MEASURED]**, **[DERIVED]**, **[MENOS]** (a proposed design decision) and **[OPEN]** (an owner decision, section 9). In section 1 every figure is **[DERIVED] from reading the source at `1f89192`**: x64 release layouts (`sizeof(std::vector)` = 24, a `std::map` node header = 32 bytes), with malloc's per-block overhead not included. None of them has been measured yet. Step 1 of section 6 measures them before any code changes, and section 10 will record the results.

---

## 1. Context

`docs/BUILD_PLAN.md`, Phase 21: every estimate must be an upper bound on the memory the job really uses, shown by measurement, and the avoidable copies are removed. The planner checks each candidate's `estimated_peak_bytes` against the host budget (`memory_budget_bytes`, otherwise half of total RAM, `docs/adr/0003`). For a CUDA candidate it also checks `device_bytes` against the free device memory. Neither figure is compared with what a run really allocates.

### 1.1 Nothing measures memory today

- No code in `src`, `tests`, `benchmarks`, `python` or `tools` reads process memory: there is no `getrusage`, `/proc/self/status`, `GetProcessMemoryInfo`, `psutil` or `tracemalloc`. The only memory query is `cudaMemGetInfo` (`src/kernels/cuda/device_info.cpp:22`). It is read once per process and cached (`:42`), so the device figure is the free memory at first use.
- `EngineExecutionResult::actual_bytes` is documented as "a genuine measurement of this region's own peak allocation" (`include/menos/simulation_engine.h:177-178`). In practice it is one of three things. It may be the size of the main buffer (dense `dense_engine.cpp:158`, CUDA `cuda_dense_engine.cpp:226`, density matrix `density_matrix_engine.cpp:278`). It may repeat the estimate's own formula (stabilizer `stabilizer_engine.cpp:260-264`; the trajectory distribution term `trajectory_engine.cpp:390`). It may be a final or single-tensor size (MPS `mps_engine.cpp:216`; tensor network `tensor_network_engine.cpp:195`). For a Factorized plan the runtime ignores it and reports `global + region.estimated_state_bytes` (`runtime.cpp:314-315`). So `actual ≤ estimate` tests (`tensor_network_engine_test.cpp:301`, `cut_differential_test.cpp:321`, `tensor_network_scale_test.cpp:221`) compare a formula with itself.
- `Result::actual_state_bytes()` is documented as "as tracked by the engine (not process RSS)" (`include/menos/result.h:203-205`). `explain()` prints it as `actual state_bytes=` (`result_impl.cpp:128-131`).

### 1.2 Where each estimate falls short

| Engine / path | Estimate | Live at the peak, not counted | Where |
|---|---|---|---|
| Dense CPU, statevector | 16·2^n | A second 16·2^n: the result is copied from the state, which is still alive | `dense_engine.cpp:144`, `:162` |
| Dense CPU, counts (sole region) | 16·2^n | `uniforms`, 8·S; the counts map, (72 + 8W)·min(S, 2^m) | `sample.cpp:35`, `:29` |
| CUDA dense, statevector or counts | 16·2^n host | A host `StateVector`, then a second host copy into the result: 32·2^n host. The 16 KiB expectation scratch on the device | `cuda_dense_engine.cpp:188`, `:243-249`, `:151-154` |
| `CumulativeDistribution` (multi-region counts; density matrix with readout or in a region; every trajectory shot) | 8·2^m where charged | (40 + 8W) per basis state: a double, a 32-byte `BitString` and its heap word, plus 2^m small mallocs | `cumulative_distribution.h:55-56`, `.cpp:16-31`, `include/menos/bitstring.h:52-53` |
| Factorized, counts | max over regions | Every region's sampler is held until the end of sampling (`runtime.cpp:320-323`), so the bytes add up across regions | `cost_model.cpp:194` |
| Factorized, statevector | max(region) + 16·2^n | Each region's own state plus its statevector copy, alongside the global buffer | `cost_model.cpp:198-201` |
| Density matrix | 16·4^n + superoperator (+ 8·2^n for counts) | Readout path: the probability vector plus a sampler at 48 B/state, held together. Kernel scratch 24·4^k per thread for k ≥ 2 channels. Gate matrix and its conjugate | `density_matrix_engine.cpp:252-256`, `:282-305` |
| Trajectory | 16·2^n + Σ K†K (+ 8·2^n for counts) | A full `CumulativeDistribution` is built per shot for the first S trajectories, to draw one outcome; the counts map; the reduced density matrix per channel | `trajectory_engine.cpp:318-322`, `:388-393` |
| Stabilizer | 16w·n + 8w (bit words only) | About 56 B of vector headers and 2 mallocs per row (about 4.5× at w = 1). `reduce()` and `expectation_sign()` each make a working copy of the tableau | `stabilizer_engine.cpp:215-218`, `tableau.cpp:124-214`, `:236-329` |
| Pauli propagation | max_terms·(16w + 16) | About 72 + 16w bytes per term. While a gate splits terms, the old and new lists and the growth slack are all live. A deep copy of every operation. The estimate also overcounts, by assuming every term's branches are live at once when terms are propagated one at a time | `pauli_propagation_engine.cpp:553-566`, `:262-268`, `:575` |
| MPS | Σ 32·L_i·R_i (site tensors) | Around a k = 2 block: `theta` (kept until return), `theta2`, the SVD input copy, LAPACKE's row-major transposes and workspace, about 736·χ² bytes in all. The counts sampler takes a deep copy of the whole state (`mps_engine.cpp:220`, `sampling.cpp:15`), and a non-sole region returns a further copy. The counts map | `mps_engine.cpp:110-131`, `apply_gate.cpp:139-235`, `svd.cpp:45-73` |
| MPS, Exact policy | Bond ≤ Schmidt-rank bound per cut | **Unconfirmed:** Exact keeps numerically zero singular values (`apply_gate.cpp:89-90`). When a SWAP-gathered gate is undone, a bond can be left above `bound_per_cut`. Hand trace: n = 6, CX(0,5) leaves bond 4 at cut 1, where the bound is 2. Step 1 tests this | `apply_gate.cpp:277-320` |
| Tensor network | 16·2^(w+1), one tensor | During each `contract_pair`, the result and both inputs are live (≥ 3×). Absorbed slots are not freed until `contract()` returns. The initial network, two copies of the timelines, the line graph, and the min-fill sets | `tensor_network_engine.cpp:124-161`, `contraction.cpp:126-188`, `:411-413` |
| Runtime, direct path | — | The counts map is **copied** out of the engine result (`result.counts.value_or(...)` on an lvalue), and the `Plan` is copied | `runtime.cpp:145`, `:136` |
| `run_batch` | Per circuit | Every `Result` is held, so the peak is the sum of the batch's statevectors | `src/api/batch.cpp:19-25` |
| Python | — | Each `.statevector` access makes a new NumPy copy (`bindings.cpp:35-46`). PennyLane's `_convert.py:136-142` makes a further transpose copy | |

W = `ceil(bitstring_width/64)`, w = `ceil(n/64)`, S = shots, m = region qubits, k = operands. **Example (BUILD_PLAN):** on 32 GiB of RAM (default budget 16 GiB), a 30-qubit dense statevector job is estimated at 16 GiB, accepted, and allocates 32 GiB.

### 1.3 Other wrinkles

- When CUDA wins, `plan.estimated_state_bytes` and the region's bytes still come from the CPU dense estimate (`planner.cpp:883-893`). Only the candidate row carries the CUDA figures.
- The default budget is computed in two places (`planner.cpp:148-149`, `mitigation.cpp:204-206`).
- `StateVector`'s budget check (`state_vector.cpp:19-24`) and `DeviceBuffer`'s (`device_buffer.h`) cover only the main buffer.
- `StateVector` uses a 64-byte-aligned allocator (`state_vector.h:62`). `Result`'s statevector is a plain `std::vector<std::complex<double>>`, so the state cannot be moved into the result without changing one of the two. No kernel relies on the alignment (no `assume_aligned`, no aligned intrinsics); only `state_vector_test.cpp:23` checks it.
- LAPACKE allocates its row-major transposes and workspace with `malloc` (`LAPACKE_malloc`). A tracker that replaces `operator new` cannot see them. **[DERIVED]**, from the netlib LAPACKE design; not yet checked against the shipped OpenBLAS 0.3.34.

## 2. Scope — [MENOS], subject to section 9

1. **Accounting.** Each engine's `estimated_peak_bytes` is an upper bound on everything it holds at once: the main state, scratch, temporaries, SVD workspace, samplers, counts maps, result copies, and anything still held when it returns. The cost model sums held samplers across Factorized regions. Each bound is a closed form computed from the engine's own types (`sizeof` and documented container growth), not a tuned constant.
2. **Measurement**, in two independent forms (section 3.1): a counting `operator new` in a dedicated test executable, and the peak RSS of a child process. For CUDA, the bytes allocated on the device are counted and cross-checked against `cudaMemGetInfo`.
3. **Copy removal**, keeping every result bitwise identical:
   - The statevector is moved into the result.
   - The CUDA device state is copied straight into the result buffer, and the sampler reads that buffer.
   - Counts outcomes are stored as basis indices; a `BitString` is built only for an outcome that is drawn.
   - Trajectory shots draw from a single cumulative pass, with no per-shot table.
   - The MPS sampler uses the state itself, not a copy.
   - The tensor network frees absorbed tensors.
   - The runtime moves the counts map.
   - The SVD calls `zgesdd_` column-major with workspace it owns (Q6).
4. **Reporting.** `Result` carries the measured peak host and device bytes beside the estimates. `explain()` prints device bytes and the measured peak.
5. **Device budget.** Free device memory per Q3.

## 3. Design — [MENOS]

### 3.1 Measurement

- **Heap tracker (gate assertion).** A new test executable, `memory_accounting_test`, replaces the global `operator new`/`operator delete` (all forms, aligned forms included) with counters that track live and peak bytes. The counters are atomic, so they work under the engines' threads. A fixture runs `menos::run`, or a single `SimulationEngine::execute`, between `reset_peak()` and `peak_bytes()`, and asserts that the estimate is at least the tracked peak above the starting live bytes. The executable is built only in the non-sanitizer presets, because ASan and the MSVC ASan runtime own `operator new` there. The other presets still run every fixture as an ordinary test, with the tracker assertions skipped and the skip listed.
- **Peak RSS (cross-check, which also sees `malloc`, LAPACK and the allocator's overhead).** A small CLI, `tools/memory_probe`, runs one fixture in a fresh process. It prints the RSS before the run and the high-water mark after it: `VmHWM` from `/proc/self/status` on Linux, `PeakWorkingSetSize` from `GetProcessMemoryInfo` on Windows. A fresh process needs no reset. The gate compares `RSS_peak − RSS_before` with `estimate + B`. B is the same measurement for a 1-qubit run of the same engine, taken in the same session. B is measured each time, never a constant in the code (Q4).
- **Device.** `DeviceBuffer` and the expectation scratch report their bytes to the ledger (section 3.2). The CUDA fixtures also compare the drop in free memory from `cudaMemGetInfo` against `device_bytes`, allowing the driver's allocation granularity, which is measured in step 1. These fixtures run only in the `cuda12-*` presets, where the GPU is reachable (`docs/adr/0014` section 17).

### 3.2 What `Result` reports (Q2)

`Result::peak_host_bytes()` and `Result::peak_device_bytes()` come from a `MemoryLedger` scoped to one execution attempt. Every allocation whose size grows with the problem goes through `common::LedgerAllocator<T>`, which updates the live and peak figures of the ledger installed for that execution. These allocations are state buffers, samplers, counts maps, MPS tensors and SVD workspace, tensor-network tensors, tableau rows, Pauli term lists, and device buffers. The heap-tracker gate test checks two things: the ledger's peak is at most the tracked peak, and the gap is only allocations that do not grow with n. `explain()`'s `actual` line keeps `state_bytes=` and appends `peak_host_bytes=` and `peak_device_bytes=`. `actual_state_bytes()` is unchanged, so existing tests and outputs stay valid.

### 3.3 Copy removal, each with a bitwise check

- **Statevector move (Q5).** Recommended: `StateVector` stores a plain `std::vector<std::complex<double>>` and `DenseEngine` moves it into the result. Kernel results do not depend on alignment, so they stay bitwise identical. Speed is checked as a same-run ratio on the Phase 15 benchmark.
- **CUDA.** `copy_to_host` writes into the result vector itself. `sample` and `CumulativeDistribution` take a `std::span<const std::complex<double>>` instead of a `StateVector`. For counts-only jobs the buffer is temporary.
- **Counts by index.** `CumulativeDistribution` keeps only `cumulative_`. `sample_one` maps the index it finds to a `BitString` using the stored `measure_targets` and width. The same doubles are summed in the same order and `upper_bound` is the same, so the draws are bitwise identical. This brings the structure to 8·2^m, which is what the estimate already charges.
- **Trajectory shots.** Each shot walks the cumulative sum once, with the same doubles and order. The first index whose cumulative sum is above u is the same index that `upper_bound` finds, including the fallback to the last index.
- **MPS sampler.** `MPSCountsSampler` stops taking `MPSState` by value. It sweeps the center to site 0 on the engine's own state, which is not used after sampling except for the expectation; the expectation is then computed first. The returned `counts_sampler` for a non-sole region takes ownership by move. **Risk:** the center sweep moves the gauge, and the expectation contraction then sees different tensors, so its last bits could change. Hence the expectation is computed before sampling (section 8).
- **Tensor network.** `live[root_b]` is released when it is absorbed (`contraction.cpp:412`).
- **Runtime.** `std::move(result.counts).value_or(...)`.

### 3.4 Accounting changes per engine (formulas fixed in step 3 from the step-1 measurements)

Each engine gets one `peak_bytes` function, used by both `estimate()` and the test. The terms are listed below.

- **Dense:** 16·2^n, plus 8·S and the counts-map bound for counts.
- **CUDA:** host 16·2^n only when a statevector or counts are requested; device 16·2^n plus the scratch.
- **Density matrix:** add the kernel scratch per thread and the sampler.
- **Trajectory:** add the cumulative pass's buffer, which is none after 3.3, and the counts map.
- **Stabilizer:** the real per-row size and the working copies.
- **Pauli:** the real per-term size, and both lists during a split.
- **MPS:** the largest live set around one block (`theta`, `theta2`, SVD input, outputs and workspace), and the per-cut bond given by the code's own growth rule (Q7).
- **Tensor network:** the result plus both inputs, plus the initial network.
- **Factorized:** the sum of held samplers, plus the largest region's transient peak.

The counts map is charged min(S, 2^m) nodes. Each node is charged `sizeof(std::pair<const BitString, std::uint64_t>)` + 4·`sizeof(void*)` + 8W. The heap tracker checks this bound on both platforms; malloc's per-block overhead is left to the RSS cross-check.

### 3.5 Unchanged

- `plan_job` stays the only planner.
- The work units are unchanged, and so are the Phase 11 monitor boundaries.
- `-fmad=false` stays, and so do the Phase 15 and 16 kernels.
- A job's plan changes only where a corrected estimate changes feasibility. Each such change is listed (Q8).

## 4. Testing — [MENOS]

- **Fixture set:** every engine × output kind.
  - Dense CPU: sv, counts, exp, all three. CUDA: the same.
  - Density matrix: counts, counts with readout, exp. Trajectory: counts, exp.
  - Stabilizer: counts, exp. Pauli: exp.
  - MPS: counts, exp; Exact and Controlled. Tensor network: exp.
  - Factorized with mixed engines: sv, counts, exp. Cut: exp.
  - The mitigated run.
  - Each at two or three sizes, so the terms that grow with n dominate.
- **Bitwise baseline.** Before any change, the fixture outputs on `1f89192` are recorded to `tests/data/memory_accounting_baseline.json`: counts maps, statevector bytes as hex, expectation bit patterns. After the changes they must match exactly.
- **MPS Exact bond test** (Q7), written in step 1, so it shows whether the hand trace holds.
- **Device-budget refresh test** (if Q3 = a).
- **`explain()` / `Result` tests** for the new fields; the Python exposure.

## 5. API changes

- `Result::peak_host_bytes()`, `Result::peak_device_bytes()`, and the Python properties `peak_host_bytes` / `peak_device_bytes`.
- `explain()` appends to the candidate line (`device_bytes=`, CUDA candidates only) and the `actual` line.
- Plan-level `estimated_state_bytes` takes the chosen backend's host figure.
- Nothing is removed.

## 6. Order of work

1. **Measure first, change nothing.**
   - Add the heap tracker, the probe and the device counter, plus the bitwise baseline.
   - Record estimate vs tracked vs RSS on `1f89192`'s behaviour in section 10, with the MPS bond test and the LAPACKE allocations.
   - Stop and report to the owner if a measurement contradicts section 1.
2. **Copy removal** (section 3.3), one commit per path, each with its bitwise check and a re-measurement.
3. **Accounting** (section 3.4), one commit per engine, each turning that engine's fixtures from expected failure to passing. Each lists any change in feasibility.
4. **Ledger, `Result`, `explain()` and Python** (sections 3.2 and 5).
5. **Device budget** (Q3); docs (README, the limitations and changelog pages, BUILD_PLAN, this ADR); `scripts/verify.sh` (WSL) and `scripts/verify-windows.ps1`, both on the final commit.

## 7. Out of scope

- Memory pools, pinned memory, and allocation and transfer time (Phase 22).
- Tiers and out-of-core execution (Phase 23).
- Zero-copy NumPy views of `Result` buffers (they change aliasing semantics; recorded deferral).
- `run_batch` peak accounting across circuits (each circuit is planned alone; the batch total is documented, not planned).
- cgroup/container memory limits.
- Reducing the Pauli estimate's overcount.

## 8. Risks

- **Replacing `operator new`** is per executable. A second test binary doing the same would conflict. ASan presets cannot run the tracker (section 3.1).
- **OpenBLAS** allocates per-thread buffers on first use, which could dominate the RSS of small MPS fixtures. B is measured per engine, and step 1 reports the size.
- **The map-node bound** depends on the standard library. The tracker test checks it on libstdc++ and MSVC STL; libc++ (macOS) is not verified.
- **MPS sampler reuse** changes where the gauge is when the expectation is computed. The order "expectation, then sampling" is required, and the bitwise baseline catches any change.
- **Feasibility changes.** A corrected estimate can refuse a job that runs today, for example dense counts with a very large S. Q8 decides whether such fixes land only together with the copy removal for that path.
- **The device figure is stale** today. Refreshing it (Q3) makes plans depend on other processes' GPU use at planning time. That figure is printed, so the plan is explainable but not reproducible when other processes' GPU use changes.

## 9. Owner decisions

- **Q1 — Default host budget.**
  - (a, recommended) Unchanged: half of total RAM (`docs/adr/0003`). It is deterministic, so the same job gets the same plan on the same machine.
  - (b) Half of available RAM at planning time. That reflects other processes, but plans then vary from run to run.
- **Q2 — What `Result` reports as measured.**
  - (a, recommended) The ledger peak of section 3.2. It is exact for every allocation that grows with the problem, the same on Linux and Windows, and costs nothing beyond the allocation itself. The heap-tracker test shows that it bounds everything that matters.
  - (b) The process's peak RSS around the run. On Windows the peak cannot be reset within a process, so a run that does not raise the process's previous peak reports nothing new. On Linux, resetting it writes to `/proc/self/clear_refs`, which affects the whole process.
  - (c) No `Result` figure. The measurement exists only in tests, and BUILD_PLAN item 4's second half is dropped.
- **Q3 — Free device memory.**
  - (a, recommended) Re-read with `cudaMemGetInfo` at the start of each `plan_job`. The figure used is printed in `explain()`. Today's cached figure can promise memory that is no longer free.
  - (b) Unchanged: read once per process.
- **Q4 — The gate's comparison.**
  - (a, recommended) Two checks. First, the heap tracker: estimate ≥ tracked peak, with **no slack**. Second, RSS: `RSS_peak − RSS_before ≤ estimate + B`, where B is the same engine's 1-qubit run measured in the same session. Both run for every fixture on Linux (WSL) and Windows.
  - (b) Heap tracker only.
  - (c) RSS only, with a slack you set.
- **Q5 — Moving the statevector.**
  - (a, recommended) `StateVector` uses `std::vector`'s default allocator, which gives 16-byte alignment. The 64-byte alignment test is replaced by a bitwise check. The Phase 15 benchmark's same-run ratio must stay within the gate's tolerance (Q9).
  - (b) Keep the 64-byte alignment and change `Result`'s statevector type. That breaks the public C++ API.
  - (c) Keep the copy and only count it (32·2^n), so jobs near the budget stay infeasible.
- **Q6 — SVD.**
  - (a, recommended) Call `zgesdd_` directly, column-major, with the transposes and the workspace (taken from its own query) in vectors MENOS owns. That is what `LAPACKE_zgesdd_work` does internally, so the inputs are identical and the results bitwise identical. The workspace is then countable and visible to the tracker.
  - (b) Keep LAPACKE; charge its workspace from the query, and leave it to the RSS check.
- **Q7 — The MPS Exact bond, if step 1 confirms it exceeds the Schmidt-rank bound.**
  - (a, recommended) Change the estimate to the bond that the code's growth rule actually produces. Results stay bitwise identical, and some MPS candidates become costlier or infeasible (listed).
  - (b) Under Exact, drop singular values below `common::numerical_rank`'s cutoff. That bounds the bond by the true rank but changes MPS results in the last bits, which the gate forbids unless you allow it.
- **Q8 — An accounting fix that makes a job feasible today infeasible.**
  - (a, recommended) Allowed, but only after that path's copy removal has landed. Every remaining change is listed in section 10 with its job and the old and new figures.
  - (b) Never: every such fix must be paired with a copy removal that keeps the job feasible, or the fix waits.
- **Q9 — The gate.** As BUILD_PLAN's proposal, refined as:
  1. The measurement of Q4 on the fixture set of section 4, for every engine and output kind, on Linux and Windows. CUDA runs in a `cuda12-*` preset.
  2. Peak reduced, measured by the heap tracker, with these targets:
     - Dense CPU statevector at n = 24: from ≥ 32·2^n to ≤ 16·2^n + 1 MiB.
     - CUDA host statevector at n = 24: from ≥ 32·2^n to ≤ 16·2^n + 1 MiB.
     - Factorized and density-matrix counts samplers: from (40 + 8W)·2^m to ≤ 8·2^m + 1 KiB per region.
     - MPS counts: the state held once, not twice.
     - The 1 MiB and 1 KiB allowances are for the non-scaling allocations; you set them or replace them.
  3. Every count, statevector and expectation in the fixture set is bitwise identical to the `1f89192` baseline.
  4. No benchmark slowdown: on the Phase 15 dense benchmark the same-run ratio is within 2%, and on the MPS benchmark within 5%. You set both figures.
  5. Every feasibility change is listed (Q8).
  6. `Result` and `explain()` carry the new fields (Q2), with tests, including Python.
  7. `scripts/verify.sh` (WSL) and `scripts/verify-windows.ps1` exit 0 on the final commit. Documentation is updated, and every new symbol has a production caller and tests.

### Owner answers (2026-10-20)

The owner took the recommended option, (a), for every question:

- **Q1:** the default host budget stays half of total RAM.
- **Q2:** `Result` reports the ledger peak of section 3.2.
- **Q3:** free device memory is re-read at the start of each `plan_job` and printed in `explain()`.
- **Q4:** the heap tracker with no slack, plus the RSS check against B measured in the same session, on Linux and Windows.
- **Q5:** `StateVector` uses the default allocator and the statevector is moved into the result.
- **Q6:** the SVD calls `zgesdd_` directly, with workspace MENOS owns.
- **Q7:** if step 1 confirms the Exact bond growth, the estimate follows the code's growth rule; results are not changed.
- **Q8:** an infeasibility change is allowed only after that path's copy removal, and each is listed in section 10.
- **Q9:** the gate as written in Q9, with the proposed figures: 1 MiB and 1 KiB allowances, a 2% dense and a 5% MPS benchmark ratio.

**Q7 revisited (2026-10-20), after step 1 measured the bond growth (section 10.1).** The owner was shown that Exact bonds reach 16–32 where the bound is 4–8, which makes both the MPS memory and time estimates low. The owner kept (a): results stay bitwise identical, and the MPS estimates (memory and work units) follow the bond the code actually builds.

## 10. Gate evidence

### 10.1 Step 1: the measurements before any change — [MEASURED], 2026-10-20

Measured with the step-1 tools (`tests/memory`) and with the engines and runtime unchanged from `1f89192`. Linux: the WSL clone, `release` and `cuda12-release` presets (g++, RTX 4060 Laptop GPU, driver 566.14). Windows: `windows-release` (MSVC 14.50). Tracked = the heap-tracker peak above the live bytes at entry. RSS Δ = `VmHWM` after the run minus `VmRSS` before it, in a fresh process, after a high-water reset (`/proc/self/clear_refs`). B = the same figure for the engine's 1-qubit fixture. Bytes.

| Fixture | Estimate | Reported actual | Tracked, Linux | Tracked, Windows | RSS Δ, Linux | B, Linux |
|---|---|---|---|---|---|---|
| dense_statevector (n = 18) | 4 194 304 | 4 194 304 | 8 389 600 | 8 389 855 | 8 900 608 | 847 872 |
| dense_counts (S = 20 000) | 4 194 304 | 4 194 304 | 4 712 304 | 4 712 415 | 5 615 616 | 782 336 |
| dense_expectation | 4 194 304 | 4 194 304 | 4 194 560 | 4 194 560 | 4 804 608 | 835 584 |
| dense_all | 4 194 304 | 4 194 304 | 8 906 608 | 8 906 758 | 9 678 848 | 794 624 |
| dense_region_counts (n = 16, sampler) | 1 048 576 | 1 048 576 | 4 194 360 | 4 194 438 | 6 197 248 | 778 240 |
| dense_statevector_24 (probe only) | 268 435 456 | 268 435 456 | — | — | 537 571 328 | 847 872 |
| dense_cuda_statevector | 4 194 304 (device 4 194 304) | 4 194 304 | 8 388 608 | n/a (no CUDA) | 8 732 672 | 782 336 |
| dense_cuda_counts | 4 194 304 | 4 194 304 | 4 712 304 | n/a | 5 115 904 | 700 416 |
| dense_cuda_expectation | 0 (device 4 194 304) | 4 194 304 | 16 432 | n/a | 720 896 | 720 896 |
| dense_cuda_all | 4 194 304 | 4 194 304 | 8 906 656 | n/a | 8 982 528 | 667 648 |
| dense_cuda_region_counts | 1 048 576 | 1 048 576 | 4 194 360 | n/a | 5 984 256 | 761 856 |
| dense_cuda_statevector_24 (probe only) | 268 435 456 | 268 435 456 | — | n/a | 536 940 544 | 688 128 |
| density_matrix_counts (n = 8) | 1 050 880 | 1 050 624 | 1 228 064 | 1 228 175 | 1 691 648 | 774 144 |
| density_matrix_counts_readout | 1 050 880 | 1 050 624 | 1 083 400 | 1 083 615 | 1 499 136 | 790 528 |
| density_matrix_expectation | 1 048 832 | 1 048 576 | 1 049 096 | 1 049 096 | 1 515 520 | 724 992 |
| density_matrix_region_counts | 1 050 880 | 1 050 624 | 1 063 000 | 1 063 039 | 1 409 024 | 806 912 |
| trajectory_counts (n = 14, N = S = 300) | 396 800 | 396 800 | 1 074 384 | 1 074 654 | 2 002 944 | 786 432 |
| trajectory_expectation | 265 728 | 265 728 | 269 272 | 269 344 | 782 336 | 774 144 |
| stabilizer_counts (n = 256) | 16 416 | 16 416 | 2 145 464 | 2 143 685 | 3 219 456 | 524 288 |
| stabilizer_expectation | 16 416 | 16 416 | 78 032 | 78 149 | 610 304 | 524 288 |
| stabilizer_region_counts | 16 416 | 16 416 | 2 096 416 | 2 096 527 | 3 096 576 | 475 136 |
| pauli_expectation (n = 30) | 24 576 | 64 | 25 824 | 25 863 | 663 552 | 458 752 |
| mps_counts (n = 40, Exact) | 37 504 | **535 168** | 1 235 496 | 1 237 964 | 4 390 912 | 778 240 |
| mps_expectation (Exact) | 37 504 | **535 168** | 607 704 | 609 225 | 3 604 480 | 856 064 |
| mps_counts_controlled (χ ≤ 16) | 273 024 | 253 824 | 673 016 | 675 340 | 3 817 472 | 815 104 |
| mps_region_counts (Exact) | 37 504 | **535 168** | 1 123 383 | 1 126 161 | 4 014 080 | 798 720 |
| tensor_network_expectation (n = 14) | 8 192 | 1 024 | 117 584 | 111 417 | 757 760 | 737 280 |
| run_dense_statevector | 4 194 304 | 4 194 304 | 8 408 179 | 8 408 344 | see below | |
| run_factorized_statevector | 4 202 496 | 4 202 496 | 4 330 624 | 4 316 309 | see below | |
| run_factorized_counts | 8 192 | 8 192 | 998 592 | 998 340 | see below | |
| run_factorized_expectation | 8 192 | 8 192 | 395 316 | 407 452 | see below | |
| run_cut_expectation | 512 | 512 | 39 836 | 40 120 | see below | |
| run_mitigated (3 runs, estimates summed) | 16 785 568 | 16 785 408 | 16 915 399 | 16 914 309 | see below | |

What these show:

- **Every fixture's tracked peak is above its estimate, on both platforms.** The two platforms agree to within a few hundred bytes. Every figure in section 1.2 is confirmed in direction, and the largest gaps are where section 1.2 put them:
  - The statevector copy: dense and CUDA dense are 2.00× at n = 18. At n = 24 the RSS Δ is 537.6 MB against an estimate of 268.4 MB.
  - The counts sampler: 4.0× its estimate for a 16-qubit region.
  - The stabilizer's working copies and counts: 130× the bit-word estimate for counts at n = 256.
  - The per-shot trajectory table: 2.7×.
- **The MPS bond exceeds the Schmidt-rank bound under Exact (Q7: confirmed).** The engine's own `actual_bytes` for the final state is 535 168 against an estimate of 37 504. A diagnostic run of the fixture's circuit (40 qubits, a 5-layer nearest-neighbour brick) through `apply_gate` with `AccuracyPolicy::Exact` gave these bonds against `compute_schmidt_rank_bound`:
  - cuts 3–35 alternate 16 against a bound of 4 and 32 against a bound of 8;
  - cuts 0–2 and 36–38 equal their bounds (2, 4, 8).
  - The hand trace of section 1.2 also holds: H(0) then CX(0,5) on 6 qubits leaves bonds 2, 4, 4, 4, 2 against bounds that are all 2.

  Cause: under Exact every split keeps min(m, n) singular values, numerically zero ones included (`apply_gate.cpp:89-90`). So a bond grows to twice its neighbour's, not to twice its own true rank. The existing consistency suite (`tests/engines/mps/schmidt_rank_bound_consistency_test.cpp`, Phase 6 criterion 2: "the rank bound is never smaller than the bond dimension actually reached") passes only because its circuits are short enough that the two coincide. The same bond figures feed the MPS work estimate (χ³), so MPS time estimates are low by the same factor.
- **The RSS check needs B.** B is 0.5–0.9 MB on Linux and 0.08–0.16 MB on Windows for every engine fixture. It is measured, not constant: a fresh process's startup and the first allocations of the C++ runtime.
- **Job fixtures on Windows:** `PeakWorkingSetSize` cannot be reset, and the planner's one-time CPU calibration (about 67 MB of RSS on Linux, measured before the reset was added) is larger than every job fixture's own peak. So the probe reports `masked=1` for all six job fixtures on Windows, and their RSS check runs on Linux only. Their heap-tracker check runs on both platforms.
- **Cross-platform bits:** counts and strategies are bitwise identical between Linux and Windows on all 34 fixtures. Statevectors and expectation values differ in their last bits on 11 of them. Engines and runtime are unchanged from `1f89192`, so this predates Phase 21; the cause was not investigated. The bitwise baseline is therefore kept per platform (`tests/memory/data/baseline_linux.json`, `baseline_windows.json`). Each is deterministic run to run, and the Linux one is identical between `release`, `cuda12-release` and `dev` (Debug, ASan, UBSan).
- **CUDA versus CPU:** counts and statevectors are bitwise identical. Expectation values differ in their last bits, as documented in Phase 17.

### 10.2 Step 2: copy removal — [MEASURED], 2026-10-20

All of section 3.3 is done. Every fixture's outputs are bitwise identical to its platform's step-1 baseline on Linux `release`, `cuda12-release` and Windows `windows-release`. The full suites pass: 735/735 `release`, 801/801 `cuda12-release` (three skips that depend on the device, as before), 736/736 `windows-release`. The direct `zgesdd` call matches `LAPACKE_zgesdd` bit for bit on 158 shapes (`tests/common/svd_lapacke_reference_test.cpp`), on both the Linux LAPACK and Windows OpenBLAS.

The tracked heap peak, Linux, before → after (bytes):

| Fixture | Before | After | Removed |
|---|---|---|---|
| dense_statevector | 8 389 600 | 4 195 552 | the result copy, 16·2^n |
| dense_all | 8 906 608 | 4 712 304 | the result copy |
| dense_region_counts | 4 194 360 | 1 573 056 | 2^m `BitString`s; the sampler is now 8·2^m |
| dense_cuda_statevector | 8 388 608 | 4 194 304 | the host `StateVector` |
| dense_cuda_region_counts | 4 194 360 | 1 573 056 | as dense |
| density_matrix_counts_readout | 1 083 400 | 1 073 224 | the sampler's `BitString`s |
| density_matrix_region_counts | 1 063 000 | 1 052 832 | as above |
| trajectory_counts | 1 074 384 | 288 200 | the per-shot table |
| mps_counts | 1 235 496 | 698 728 | the sampler's copy of the state |
| mps_region_counts | 1 123 383 | 698 888 | as above |
| mps_counts_controlled | 673 016 | 417 592 | as above |
| run_dense_statevector | 8 408 179 | 4 227 090 | the result copy |
| run_factorized_counts | 998 592 | 957 792 | region samplers' `BitString`s |
| run_mitigated | 16 915 399 | 16 874 519 | the sampler's `BitString`s |

At the ADR's Q9 size, n = 24, the RSS Δ of a statevector job fell from 537 571 328 to 268 996 608 bytes (CPU), and from 536 940 544 to 268 378 112 bytes (CUDA host).

As built, differing from section 3.3:

- **MPS order.** The MPS engine now computes the expectation before sampling, because sampling consumes the state. The expectation reads the same state in the same gauge as before. The Phase 11 boundaries are unchanged in kind and number: the expectation loop has none.
- **SVD.** The direct call releases its input and workspace before building the row-major outputs, so `a_copy` and LAPACKE's internal copies are gone. This is reflected in `mps_*` as part of the "after" figures.
- **Trajectory `actual_bytes`.** It no longer adds the 8·2^n per-shot table, which is no longer allocated. Its estimate is corrected in step 3.
