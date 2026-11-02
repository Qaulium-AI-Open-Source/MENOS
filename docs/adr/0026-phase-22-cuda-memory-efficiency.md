# ADR 0026: Phase 22 — CUDA memory efficiency

## Status

**Accepted; section 9 answered (2026-10-31).** The owner took the recommended option for every question (answers recorded at the end of section 9). The owner started Phase 22 on 2026-10-31 as the second phase of the memory roadmap (`docs/BUILD_PLAN.md`, "Memory roadmap (Phases 21–23)"), while the Phase 21 gate is still open: its final verification was deferred the same day (`docs/adr/0025`, Status). Branch `phase-22-cuda-memory-efficiency`, from `phase-21-memory-accounting` at `9b8833c` (code unchanged since `72d8f27`), because every change here builds on Phase 21's device ledger, its fresh free-memory read and its `device_bytes` accounting. Phase 22 therefore merges after Phase 21. The owner approved the implementation plan on 2026-10-31. Step 1 of section 6 (measure first, change nothing) is done and recorded in section 10.1; no engine, planner or runtime code has changed. The owner answered Q7–Q9 at the end of section 10.1 on 2026-10-31, taking every recommended option: item 1 is built, item 2 and item 3 are recorded as measured and not adopted, and the gate's criterion 2 is the 10× same-run allocation ratio.

Labels follow `docs/adr/0014`–`0025`: **[MEASURED]**, **[DERIVED]**, **[MENOS]** (a proposed design decision) and **[OPEN]** (an owner decision, section 9). Every figure in section 1 is **[DERIVED] from reading the source at `72d8f27`** or quoted from an earlier ADR's measurement, and says which. Nothing in this ADR has been measured for Phase 22 yet; step 1 of section 6 measures first, and section 10 will record the results.

Machine note: the GPU work of this phase runs in the WSL clone, on the RTX 4060 Laptop GPU (8188 MiB, driver 566.14), which reaches the GPU only through the `cuda12-dev`/`cuda12-release` presets (`docs/adr/0014` section 17). Nsight Systems does not work there (`docs/adr/0020` section 10.3), so timings come from CUDA events and host timers. Windows is CPU-only and unchanged.

---

## 1. Context

`docs/BUILD_PLAN.md`, Phase 22: less time spent allocating and copying on the GPU, using only CUDA runtime features MENOS already links (CUDA 12.0 and later), no new library. Phase 21 made the CUDA engine's memory honest: its device estimate equals its device ledger peak on every fixture (`docs/adr/0025` section 10.4), the host state is copied from the device straight into the result (section 10.2), and free device memory is re-read before each `plan_job` (Q3 there). What it did not change is **how** the engine allocates and transfers. This section records that, from the source.

### 1.1 Where the CUDA engine allocates, copies and frees today

| What | Where | Per | Notes |
|---|---|---|---|
| `cudaMalloc` of the 16·2^n state, then `cudaMemset` to zero | `src/kernels/cuda/device_buffer.cpp:26`, `:31` | execution | `DeviceBuffer` is constructed once per `CudaDenseEngine::execute` (`src/engines/dense_cuda/cuda_dense_engine.cpp:265`). |
| A second `cudaMemset` of the whole state | `src/kernels/cuda/dense_kernels.cu:275` | execution | `initialize()` zeroes the buffer again before writing element 0, so a fresh state is written twice. The comment at `device_buffer.cpp:28` says the constructor's fill is "so initialize() only has to write element 0", which is not what `initialize()` does. One of the two is redundant. |
| `cudaMemcpy` host→device of 16 bytes (the global phase) + `cudaDeviceSynchronize` | `dense_kernels.cu:281-283` | execution | Negligible bytes; one synchronising call. |
| `cudaFree` of the state | `device_buffer.cpp:37` | execution | In the destructor, at the end of `execute()` or during unwinding. |
| `cudaMalloc` of the expectation partial sums (`kExpectationMaxBlocks` × 16 B = 16 KiB), `cudaMemcpy` device→host of them, `cudaFree` | `dense_kernels.cu:378`, `:396`, `:397` | **observable term** | For an observable with T terms: T allocations, T frees, T synchronising copies, and T host `std::vector` allocations (`:393`). The ledger charges the 16 KiB per term (`:379`). |
| `cudaMemcpy` device→host of the 16·2^n state into a pageable `std::vector` | `device_buffer.cpp:59`, called at `cuda_dense_engine.cpp:303` | execution with a statevector or counts | The only large transfer. The calibration measures pageable rates deliberately, "because that is what the backend actually transfers" (`src/kernels/cuda/device_benchmarks.h:49-51`). |
| Two `cudaMemcpyToSymbol` per gate (matrix and operands) | `dense_kernels.cu:350-353` | gate with k ≥ 3 | Phase 16 moved k = 1 and 2 to kernel arguments; the generic path keeps the copies. Not a memory allocation; out of scope here (section 7). |

So one execution makes 1 + T device allocations and 1 + T device frees, writes the state twice before the first gate, and transfers the state back through pageable memory.

### 1.2 How often an execution happens

- **Every job** planned onto the CUDA backend is one `execute()`: one `cudaMalloc`, two memsets, one `cudaFree`.
- **A Factorized plan with CUDA regions** runs its regions one after another (`src/runtime/runtime.cpp:254`); each CUDA region constructs and destroys its own `DeviceBuffer`, so a plan with R CUDA regions allocates and frees R times. Phase 21 made the plan's `device_bytes` the largest region's, "since regions run one after another and each frees its device state before the next" (`docs/adr/0025` section 10.5). Phase 22 must keep that statement true from the planner's point of view even when freed memory is retained in a pool (section 3.4).
- **A replan** (Phase 11) abandons the attempt, which destroys its `DeviceBuffer`, and the new attempt allocates again from nothing.
- **`run_batch`** (Phase 14) runs each circuit through `run`, so a batch of B CUDA circuits allocates and frees B times.
- **Calibration** (`src/plan/cuda_calibration.cpp`) constructs several `DeviceBuffer`s per `measure_cuda_calibration`, and the work probe times `engine.execute()` end to end, so the measured device rate includes the allocation, the two memsets and the free. If Phase 22 makes those cheaper, the calibrated `device_work_units_per_second` rises; that is a planner change to record, not a tuning constant.

### 1.3 What the costs are, from earlier measurements

- `docs/adr/0014` (Phase 10) and `docs/adr/0020` section 10.3 (Phase 16) measured per-gate costs, not allocation: an empty launch plus synchronize is 36–67 µs, the two constant-memory copies 74–109 µs. Nothing in any ADR measures `cudaMalloc`, `cudaMemset` or `cudaFree` at the state sizes MENOS uses. The comment at `dense_kernels.cu:33` ("cudaMalloc/cudaFree per gate costs single-digit microseconds") is for a small buffer and was the reason the gate matrix went to constant memory; it is not a measurement of a 16·2^n allocation.
- `docs/adr/0025` section 10.2 measured the host side of the device→host path (the statevector job's RSS at n = 24 fell from 537 MB to 268 MB) but not its time.
- The transfer rates the planner uses are pageable rates measured on a 16 MiB buffer (`device_benchmarks.cpp:18`, `:33-45`). Pinned transfers are faster on every PCIe system the CUDA documentation describes, because a pageable copy is staged through a driver-owned pinned buffer; by how much on this machine, in WSL, is **unmeasured** and is the first thing step 1 settles.

**[DERIVED]** From the figures above, at n = 24 (256 MiB state) one execution writes 512 MiB of device memory before the first gate; a 100-term observable makes 100 device allocations and 100 synchronising 16 KiB copies. Whether any of this is material against the gates' own time is exactly what step 1 measures, with the stated rule (section 6) that an item whose measured cost is not material is recorded and **not** built.

### 1.4 What the CUDA runtime already offers, without a new library

- **Stream-ordered allocation**: `cudaMallocAsync` / `cudaFreeAsync` (CUDA 11.2 and later, so within MENOS's 12.0 floor), which serve allocations from a memory pool and let the pool keep freed memory for reuse. A device reports support through `cudaDevAttrMemoryPoolsSupported`; MENOS must query it, not assume it (section 8).
- **Pools**: the device's default pool (`cudaDeviceGetDefaultMemPool`) or an explicit one (`cudaMemPoolCreate`, `cudaMallocFromPoolAsync`). A pool's `cudaMemPoolAttrReleaseThreshold` is 0 by default, which returns freed memory to the driver at the next synchronisation. MENOS synchronises once per gate (a frozen invariant), so with the default threshold a pool would retain nothing across executions; the threshold has to be set for the pool to do anything, and how much it may hold is the owner's decision (Q2). The pool reports `cudaMemPoolAttrReservedMemCurrent` and `cudaMemPoolAttrUsedMemCurrent`, so its retained-but-free bytes are a queryable figure, not a guess. `cudaMemPoolTrimTo` releases them.
- **Page-locked host memory**: `cudaMallocHost` / `cudaHostAlloc` for a buffer MENOS owns, or `cudaHostRegister` to page-lock memory it already has (the result's `std::vector`), with `cudaMemcpyAsync` for a transfer that can overlap a host-side copy.

## 2. Scope — [MENOS], subject to section 9

As `docs/BUILD_PLAN.md` Phase 22, items 1–3, each conditioned on step 1's measurement:

1. **One state allocation served from a pool**, reused across jobs, regions, replans and batch circuits in one process, instead of a `cudaMalloc`/`cudaFree` per execution. The state is written to zero once, not twice.
2. **The expectation partial-sum buffer allocated once per execution**, on the device and on the host, not once per observable term.
3. **Device→host copies through page-locked host memory**, with the page-locked bytes counted against the host budget and in the host ledger.
4. **Accounting and reporting** (so Phase 21's promises stay true): the planner's device feasibility check counts the pool's retained memory as available to MENOS; `explain()` prints it; `Result::peak_device_bytes()` is unchanged in meaning (bytes the execution held, pooled or not); and a public call releases the pool.

Not in scope: anything in section 7.

## 3. Design — [MENOS]

### 3.1 The device memory pool (item 1)

- A `kernels::cuda::DevicePool` owns one explicit CUDA memory pool for device 0, created on first use with `cudaMemPoolCreate` (Q5: explicit, not the device default pool, so MENOS never changes the behaviour of another library sharing the process, such as a Python user's PyTorch) and with `cudaMemPoolAttrReleaseThreshold` set per Q2. It is function-local-static like `device_info()`: written once, read-only afterwards except through the CUDA runtime's own thread-safe calls. If `cudaDevAttrMemoryPoolsSupported` is 0 on the device, `DevicePool` falls back to plain `cudaMalloc`/`cudaFree` and reports `pooled = false`, which `explain()` shows; the planner then counts no retained memory. The fallback is the code that exists today, not new speculative code.
- `DeviceBuffer` allocates with `cudaMallocFromPoolAsync` on the legacy default stream (the stream every kernel already uses) and frees with `cudaFreeAsync` on it. Because `apply_matrix` synchronises after every gate and `initialize()` synchronises after the phase write, no allocation is ever outstanding when host code reads the state. The ledger `Charge(bytes, Space::Device)` is made exactly as today, so `peak_device_bytes` is unchanged.
- The constructor's `cudaMemset` is removed and `initialize()`'s kept, so the state is zeroed once. A pooled block is not zero on arrival, which is why `initialize()` is the one that stays. Result: bitwise identical, since the state after `initialize()` is the same bytes.
- **Replans and regions** need no new code: the abandoned attempt's or the previous region's `DeviceBuffer` is destroyed before the next is constructed, and the pool serves the next allocation from the block it just took back when the size fits.
- **Release.** `menos::release_device_memory()` (C++) and `menos.release_device_memory()` (Python) call `cudaMemPoolTrimTo(pool, 0)`. Nothing releases automatically other than the threshold.

### 3.2 The expectation scratch (item 2)

`expectation_of_term` takes the partial-sum buffers as arguments: a device buffer of `expectation_scratch_bytes()` allocated once per `execute()` (through `DeviceBuffer`, so it is pooled and ledgered once) and a host `std::vector` of `kExpectationMaxBlocks` sized once. The kernel, the block count, the fixed-order tree reduction and the host summation order are untouched, so every expectation is bitwise identical to today's CUDA result. The ledger then charges the scratch once per execution instead of once per term, which is what the estimate already assumes (`cuda_dense_engine.cpp`, `device_bytes = state_bytes + expectation_scratch_bytes()`); the device estimate does not change.

### 3.3 Page-locked device→host transfer (item 3, Q3)

The result's statevector is a plain `std::vector<std::complex<double>>` (Phase 21, Q5), and the device state is copied straight into it. A page-locked path must not reintroduce the copy Phase 21 removed. Three candidates, measured in step 1 at n = 20, 22, 24 and 26 against today's pageable `cudaMemcpy`:

- **(a) Register in place.** `cudaHostRegister` the result vector's memory, `cudaMemcpy` into it, `cudaHostUnregister`. No extra host bytes. Its cost is the page-locking of 16·2^n bytes, which is proportional to the size and may cancel the bandwidth gain; measured, not assumed.
- **(b) Pinned staging.** A `cudaHostAlloc`ed staging area of two chunks of C bytes, owned by `DevicePool` and retained like the device pool. `cudaMemcpyAsync` fills one chunk while the host `memcpy`s the other into the result vector. Extra host bytes 2C, counted in `host_peak_bytes` and charged to the host ledger; C set from the step-1 measurement and recorded. It changes no planner decision, only an accounting term that Phase 21's heap tracker and RSS check then verify.
- **(c) Keep pageable**, if neither (a) nor (b) is materially faster on this machine. That is a legitimate outcome and is recorded as "measured, not adopted", as Phase 17 item 4b was.

Whichever is adopted, the bytes that land in the result are the same bytes, so the statevector and counts stay bitwise identical.

### 3.4 Accounting the pool (item 4)

- `kernels::cuda::device_available_bytes()` returns `cudaMemGetInfo` free + (`ReservedMemCurrent` − `UsedMemCurrent`) of MENOS's pool: what is free on the device plus what MENOS already holds and can reuse. Without this, a pool that retains a 256 MiB state would make the next 256 MiB job look infeasible by exactly the memory that is waiting for it.
- `explain()`'s `estimated` line, which Phase 21 gave `device_available_bytes=…` when a CUDA candidate or region exists, appends `device_pool_bytes=…` (the retained-but-free figure) so the sum is auditable. A plan with no CUDA in it prints exactly what it does today.
- The Phase 21 statement that regions free their device state before the next runs stays true for the ledger (the `Charge` is released) and for the planner (the retained block is counted as available). The `cudaMemGetInfo` drop that `tests/api/cuda_peak_memory_test.cpp` compares with `device_bytes` is replaced, for pooled runs, by the pool's `UsedMemHigh` over the run; the assertion `ledger peak ≤ estimate` stays as it is. This is a change of instrument, not of criterion, and section 10 lists it.
- Calibration (`measure_cuda_calibration`) runs before any job and allocates through the same pool, so the first job of a process may already find a warm block. The recorded crossover (`cuda_crossover_benchmark`) is re-measured and the change in `device_work_units_per_second` is recorded in section 10.

### 3.5 Unchanged

- `plan_job` is the only planner; the pool is an allocation mechanism and makes no decision.
- One `cudaDeviceSynchronize` per gate; `-fmad=false`; the Phase 15 and 16 kernels, bitwise; the Phase 11 monitor boundaries.
- Every host estimate of Phase 21 remains an upper bound; the staging bytes of 3.3(b), if adopted, are added to it, never left outside it.
- Windows: CUDA stays refused at configure time; `scripts/verify-windows.ps1` must still exit 0 with no CUDA code compiled.

## 4. Testing — [MENOS]

- **Step-1 measurement tool**, `benchmarks/cuda_memory_benchmark.cpp` (a benchmark, not a test, like `cuda_kernel_benchmark.cpp`): in one run, (i) `cudaMalloc` + `cudaMemset` + `cudaFree` of 16·2^n for n = 16…26 against `cudaMallocFromPoolAsync` + `cudaFreeAsync` from a warm pool, same sizes; (ii) one 16 KiB `cudaMalloc`/copy/`cudaFree` against a reused scratch, × 100 terms; (iii) device→host at n = 20…26 pageable, registered-in-place and pinned-staged, with the registration cost shown separately; (iv) a whole `execute()` for the crossover circuit family, statevector and expectation outputs. Medians and spread per `benchmark_spread.h`; the GPU's clock state varies about 2× between runs (`docs/adr/0020` section 10.4), so every comparison is a **same-run ratio**. JSON output through an environment variable, as the other CUDA benchmarks do.
- **Reuse test** (GPU, `cuda12-*` presets): two consecutive same-size statevector jobs; the second's `cudaMemGetInfo` free does not fall, the pool's `UsedMemHigh` equals `device_bytes`, and `device_available_bytes()` equals free + retained.
- **Release test:** after `release_device_memory()`, the pool's reserved bytes are 0 and free memory is back within the driver's granularity measured in Phase 21.
- **Fallback test:** with `cudaDevAttrMemoryPoolsSupported` reported 0 (a test seam at the `DevicePool` constructor, not in production code paths), `DeviceBuffer` still works and `explain()` says `pooled=false`.
- **Expectation scratch:** the ledger's device peak for `dense_cuda_expectation` equals `state + 16 KiB` (it already does; the test keeps it), and a 100-term observable gives bitwise the same terms as `1f89192`'s recorded baseline.
- **Bitwise:** the Phase 21 baseline `tests/data/memory_accounting_baseline.json` CUDA fixtures and `tests/api/cpu_cuda_differential_test.cpp` pass unchanged.
- **Accounting:** `cuda_peak_memory_test` fixtures keep `ledger peak ≤ estimate`; the Factorized-with-CUDA-regions test keeps its bound; the heap tracker and RSS checks of Phase 21 cover the staging bytes of 3.3(b) if adopted (the tracker cannot see `cudaHostAlloc`, so the ledger charges it explicitly and the RSS check is what sees it).
- **Explain and Python:** `device_pool_bytes=` and `pooled=` golden tests; `menos.release_device_memory()`.
- **Replan:** the existing `cuda_execution_monitor_test` replan path, with an added check that the second attempt's `peak_device_bytes` is reported as before.

## 5. API changes

- `menos::release_device_memory()` and Python `menos.release_device_memory()`; no-ops without CUDA.
- `explain()` appends `device_pool_bytes=` and `pooled=` on the `estimated` line, CUDA plans only.
- Nothing is removed; `Result` gains no field.

## 6. Order of work

1. **Measure first, change nothing.** Build `cuda_memory_benchmark`; record (i)–(iv) of section 4 in section 10.1; query `cudaDevAttrMemoryPoolsSupported` in WSL. **Rule:** an item whose measured cost is below 2% of the smallest end-to-end execution it affects at n ≥ 20 is recorded and not built; the owner may override (Q4). Stop and report if a measurement contradicts section 1.
2. **Pool + single zero fill** (3.1), with the reuse, release and fallback tests and the bitwise check.
3. **Expectation scratch once per execution** (3.2).
4. **Device→host path** per the step-1 result and Q3 (3.3).
5. **Accounting, `explain()`, release API, Python** (3.4, 5); recalibration measured and recorded.
6. Docs (README, `docs/reference/limitations.md`, `changelog.md`, BUILD_PLAN, this ADR); `scripts/verify.sh` (WSL, GPU tests executing in `cuda12-dev`) and `scripts/verify-windows.ps1`, both on the final commit.

## 7. Out of scope

- Asynchronous kernel launches and fewer synchronisations (`docs/adr/0020` Q4(b), a recorded deferral that would change what the frozen Phase 11 monitor observes).
- Gate fusion; diagonal and permutation kernels (Phase 16 deferrals, each needing its own ADR).
- The generic kernel's two `cudaMemcpyToSymbol` per gate for k ≥ 3.
- Host→device pinned transfers: the only host→device bytes are 16 per execution and the gate matrices, which go through constant memory or kernel arguments.
- Tiered memory, host-offloaded or unified-memory states (Phase 23); multi-GPU; cuQuantum (Phase 28, licence decision first); ROCm; CUDA on Windows.
- A CPU-side allocation pool; any change to the CPU engines.

## 8. Risks

- **WSL support for memory pools or page-locked memory may be partial.** Step 1 queries the attribute and measures; if pools are unsupported, item 1 reduces to the fallback and the ADR records it; if page-locking is slow or capped in WSL, 3.3(c) is the result.
- **A retained pool is memory other processes cannot use.** Bounded by Q2; released by the public call; always visible in `explain()`.
- **Pool allocations are not zero-filled**, which 3.1 handles by keeping `initialize()`'s memset. A test that reads a state before `initialize()` would see stale bytes; none exists, and `DeviceBuffer`'s header will say so.
- **Calibration shifts.** A cheaper allocation raises the calibrated device rate and may move the CPU/CUDA crossover by a qubit. Recorded, not hidden; `explain()` output of CPU-only plans is unchanged.
- **Phase 21's open gate.** Phase 22 branches from Phase 21's head. Its own gate cannot be claimed before Phase 21's criterion 4 re-run and verification have been made (Q6).

## 9. Owner decisions

- **Q1 — Does the pool persist across jobs in one process?**
  - (a, recommended) Yes: retained between jobs, regions, replans and batch circuits, released only by `release_device_memory()` or process exit. This is what gives `run_batch`, parameter-shift gradients through PennyLane and Factorized plans their saving.
  - (b) Per job: trimmed to zero at the end of each `run`. Saves within a job only (regions, replans, expectation terms).
- **Q2 — How much free VRAM may the pool hold?**
  - (a, recommended) Whatever the largest state it has served needed: the release threshold is set to the device's total memory and the pool is never trimmed automatically. The planner counts the retained bytes as available, so no MENOS job is refused because of them, and `release_device_memory()` exists for other processes' sake.
  - (b) A fraction of total device memory you set; above it, freed memory returns to the driver at the next synchronisation.
  - (c) Zero (the CUDA default): no reuse across executions. Item 1 would then only remove the double zero fill.
- **Q3 — The device→host path.**
  - (a, recommended) Decide from the step-1 measurement among 3.3(a) register-in-place, (b) pinned staging and (c) pageable, adopting the fastest whose host bytes stay inside the estimate; the choice and the figures are recorded in section 10 and shown to you before step 4.
  - (b) Choose one now.
- **Q4 — The gate.** As BUILD_PLAN's proposal, refined as:
  1. Section 4's measurements recorded on this machine, as same-run ratios with spread, before and after.
  2. Per execution, device allocation + zero fill + free time reduced by a factor you set from the step-1 baseline (as Phases 15 and 16 set their targets), at n = 20, 22, 24; the expectation path at 100 terms likewise.
  3. Device→host time at n = 24 reduced by a factor you set, or 3.3(c) recorded with the measurement that justifies it.
  4. The pool's retained memory counted in the device feasibility check and printed by `explain()`; the release call works; the fallback works.
  5. Every count, statevector and expectation in the Phase 21 fixture set bitwise identical to the `1f89192` baseline.
  6. The planner change (recalibrated device rate, any crossover shift) recorded.
  7. GPU tests actually execute in the `cuda12-*` presets; `scripts/verify.sh` (WSL) and `scripts/verify-windows.ps1` exit 0 on the final commit; documentation updated; every new symbol has a production caller and tests.

  The 2% rule of section 6 step 1 is yours to confirm or replace.
- **Q5 — Explicit pool or the device default pool?** (a, recommended) explicit, so no other library in the process is affected; (b) the device default pool, fewer lines.
- **Q6 — Order of the two open gates.** (a, recommended) Phase 21's verification and criterion 4 re-run are made on the Phase 21 branch first, then Phase 22 rebases if needed; (b) one verification on Phase 22's final commit covers both, with each phase's criteria listed separately in its own ADR. Either way neither gate is marked passed without your confirmation.

### Owner answers (2026-10-31)

The owner took the recommended option, (a), for every question:

- **Q1:** the pool persists across jobs, regions, replans and batch circuits in one process; only `release_device_memory()` or process exit empties it.
- **Q2:** the release threshold is the device's total memory; the pool is never trimmed automatically, and the planner counts its retained bytes as available.
- **Q3:** the device→host path is decided from the step-1 measurement among register-in-place, pinned staging and pageable; the figures are shown to the owner before step 4.
- **Q4:** the gate as written in Q4; the reduction factors of criteria 2 and 3 are set by the owner from the step-1 baseline; the 2% materiality rule of section 6 step 1 is confirmed.
- **Q5:** an explicit pool, not the device default pool.
- **Q6:** Phase 21's verification and criterion 4 re-run are made on the Phase 21 branch, before the Phase 22 gate is claimed; Phase 22 work proceeds meanwhile, as the owner instructed on 2026-10-31, and rebases onto Phase 21's final commit if that commit differs from `9b8833c`.

## 10. Gate evidence

### 10.1 Step 1: the measurements before any change — [MEASURED], 2026-10-31

Tool: `benchmarks/cuda_memory_benchmark.cpp` (`menos_cuda_memory_benchmark`, section 4), built in the WSL clone with the `cuda12-release` preset (g++-13, CUDA 12.4 runtime, driver API 12.7, driver 566.14) against the engines unchanged from `72d8f27`. RTX 4060 Laptop GPU, 8 585 216 000 bytes total. Five runs, JSON in `docs/benchmarks/phase-22-baseline/cuda-memory-run1.json` to `run5.json`: runs 1–3 with earlier drafts of the benchmark that lacked some of the section (2) rows (the rows they have are the same measurements), runs 4–5 with the committed source. Every figure is the median of 10 repetitions after a warm-up; interquartile ranges are in the JSON. Both the GPU's clock state and the host-side driver costs moved between runs by up to 2.4×, so each conclusion below is drawn from same-run ratios and is stated only where all five runs agree.

**Facts (0).** `cudaDevAttrMemoryPoolsSupported` = 1, `cudaDevAttrHostRegisterSupported` = 1, the default pool's release threshold = 0 bytes (so an unconfigured pool would retain nothing across MENOS's per-gate synchronisations). With the threshold set to the device's total memory, the explicit pool held 1 073 741 824 bytes reserved after the sweep: the configuration of section 3.1 works on this machine.

**(1) Allocation**, microseconds per cycle, direct = `cudaMalloc` + `cudaMemset` + `cudaFree` + synchronise (what `DeviceBuffer` does today), pooled = `cudaMallocFromPoolAsync` + `cudaFreeAsync` + stream synchronise from a warm pool; runs 1 / 2 / 3 / 4 / 5:

| n | bytes | direct | memset alone | pooled |
|---|---|---|---|---|
| 16 | 1 048 576 | 2 321 / 1 653 / 655 / 535 / 801 | 115 / 61 / 32 / 30 / 82 | 2.1 / 1.8 / 1.2 / 0.9 / 1.0 |
| 18 | 4 194 304 | 1 858 / 1 767 / 735 / 644 / 658 | 144 / 97 / 37 / 42 / 36 | 1.5 / 1.4 / 1.1 / 0.9 / 3.1 |
| 20 | 16 777 216 | 3 117 / 2 399 / 1 370 / 1 096 / 1 287 | 294 / 280 / 64 / 45 / 59 | 1.4 / 1.4 / 1.1 / 0.9 / 0.9 |
| 22 | 67 108 864 | 6 508 / 6 419 / 3 635 / 2 707 / 3 275 | 961 / 928 / 443 / 334 / 426 | 2.7 / 2.2 / 6.1 / 0.9 / 0.9 |
| 24 | 268 435 456 | 22 714 / 22 358 / 15 849 / 9 462 / 12 117 | 2 193 / 1 742 / 1 639 / 1 134 / 1 627 | 2.2 / 1.4 / 0.9 / 1.3 / 2.0 |
| 26 | 1 073 741 824 | 50 160 / 54 142 / 63 387 / 39 830 / 42 972 | 6 468 / 6 527 / 6 369 / 6 206 / 4 973 | 2.8 / 2.5 / 1.1 / 0.9 / 1.0 |

The same-run ratio direct / pooled is between 215× and 45 000× in every row of every run. A pooled cycle is 1–6 µs at every size; a direct cycle is 1–3 ms at n = 20, 9–23 ms at n = 24 and 40–63 ms at n = 26. The zero fill is 3–15% of the direct cycle; `cudaMalloc` + `cudaFree` are the rest.

**(2) The expectation scratch**, microseconds per term at n = 20, 100 terms; runs 1 / 2 / 3 / 4 / 5 (— = not in that draft):

| Row | Figures |
|---|---|
| bookkeeping today (16 KiB `cudaMalloc`, synchronising copy back, `cudaFree`, host vector) | 75 / 109 / 38 / 40 / 44 |
| the same with the allocation from the pool | — / — / 41 / 41 / 42 |
| the copy alone, into buffers allocated once | 67 / 48 / 36 / 28 / 35 |
| whole `expectation_of_term`, kernel included, state \|0…0⟩ | 308 / 305 / 401 / 255 / 290 |
| whole call on the state the crossover circuit leaves | — / — / 395 / 236 / 288 |
| whole call right after an H gate, the gate's own time subtracted | — / — / 382 / 233 / 294 |
| whole call in `execute()`'s own sequence (fresh buffer, initialise, gates, 100 terms) | — / — / — / 223 / 286 |

What item 2 as written removes is "bookkeeping today" minus "the copy alone": 8 / 61 / 2 / 12 / 9 µs per term, 2.6 / 20.1 / 0.5 / 4.6 / 3.0% of a term, and over 100 terms 0.4 / 2.8 / 0.1 / 0.9 / 0.6% of the 100-term n = 20 execution of table (4) (median 0.6%). The 16 KiB allocation itself costs the same from the pool as from `cudaMalloc` (41 vs 38–44 µs), so item 1 does not help a term. The larger part of the bookkeeping is the synchronising 16 KiB copy (28–67 µs); the kernel is the rest of the term. The state's contents and a preceding gate make no difference (rows 5–6 equal row 4 within the run's spread).

**(3) Device→host of the state**, GB/s of the median (higher is better); pageable = `cudaMemcpy` into a `std::vector`, as `copy_to_host` does today; register = `cudaHostRegister` + copy + `cudaHostUnregister`, all three timed; staged = the best of two-chunk pinned staging at 4, 16 and 64 MiB; runs 1 / 2 / 3 / 4 / 5:

| n | pageable | register, total | staged, best |
|---|---|---|---|
| 20 | 3.12 / 3.13 / 3.16 / 6.24 / 6.22 | 1.77 / 1.48 / 2.06 / 3.29 / 3.63 | 2.64 / 2.57 / 2.91 / 5.20 / 5.41 |
| 22 | 3.20 / 3.17 / 3.23 / 6.32 / 6.37 | 1.84 / 1.74 / 2.80 / 3.66 / 4.10 | 2.78 / 2.98 / 3.10 / 3.10 / 5.83 |
| 24 | 3.19 / 3.22 / 3.25 / 6.39 / 3.25 | 1.90 / 1.83 / 2.50 / 2.50 / 2.52 | 3.07 / 3.11 / 3.17 / 3.19 / 3.21 |
| 26 | 3.16 / 3.16 / 3.24 / 3.25 / 3.24 | 2.01 / 2.04 / 2.25 / 2.41 / 2.52 | 3.18 / 3.18 / 3.23 / 3.22 / 3.22 |

Pageable is never beaten, in any run at any size. Registering the destination costs 5.7–18.9 ms at n = 24 on its own, and the copy into the registered pages is then *slower* than the pageable copy (90–104 ms against 42–84 ms). Staging matches pageable within 3% at n ≥ 24 and loses below. The pageable path itself is bimodal on this machine (3.2 or 6.3 GB/s), which is the WSL transfer path's own behaviour and not something MENOS controls. The copy is 3.2–7.4% of a statevector execution and cannot be made cheaper here.

**(4) End to end**, `CudaDenseEngine::execute` on the crossover circuit, seconds; runs 1 / 2 / 3 / 4 / 5:

| n | statevector | expectation, 1 term | expectation, 100 terms |
|---|---|---|---|
| 16 | 0.0147 / 0.0138 / 0.0128 / 0.0089 / 0.0086 | 0.0116 / 0.0139 / 0.0136 / 0.0083 / 0.0082 | 0.0273 / 0.0273 / 0.0257 / 0.0180 / 0.0165 |
| 18 | 0.0227 / 0.0316 / 0.0331 / 0.0186 / 0.0186 | 0.0237 / 0.0224 / 0.0318 / 0.0189 / 0.0176 | 0.1395 / 0.1828 / 0.1268 / 0.0848 / 0.0861 |
| 20 | 0.0732 / 0.0727 / 0.1192 / 0.0640 / 0.0622 | 0.0686 / 0.0693 / 0.1121 / 0.0626 / 0.0711 | 0.2087 / 0.2184 / 0.2545 / 0.1327 / 0.1521 |
| 22 | 0.318 / 0.307 / 0.502 / 0.279 / 0.327 | 0.253 / 0.256 / 0.456 / 0.234 / 0.306 | 0.435 / 0.441 / 0.706 / 0.361 / 0.449 |
| 24 | 1.279 / 1.263 / 2.129 / 1.145 / 1.434 | 1.043 / 1.032 / 1.031 / 1.040 / 1.362 | 1.523 / 1.434 / 1.280 / 1.349 / 1.731 |
| 26 | 5.565 / 5.575 / 5.053 / 6.616 / 6.535 | 4.327 / 4.250 / 4.413 / 5.807 / 5.625 | 5.525 / 6.703 / 6.502 / 7.036 / 6.911 |

The share of an execution that item 1 would remove (the direct cycle of table (1) against the same run's execution): statevector at n = 16: 15.8 / 12.0 / 5.1 / 6.0 / 9.3%; n = 18: 8.2 / 5.6 / 2.2 / 3.5 / 3.5%; n = 20: 4.3 / 3.3 / 1.2 / 1.7 / 2.1%; n = 22: 2.1 / 2.1 / 0.7 / 1.0 / 1.0%; n = 24: 1.8 / 1.8 / 0.7 / 0.8 / 0.8%; n = 26: 0.9 / 1.0 / 1.3 / 0.6 / 0.7%. One-term expectation at n = 20: 4.5 / 3.5 / 1.2 / 1.8 / 1.8%. The redundant second zero fill alone is 0.05–0.4% at n = 20.

**A note on method.** The difference of the 100-term and 1-term medians divided by 99 is *not* a per-term cost: it gives 0.7–1.5 ms at n = 20 but 0.08–0.16 ms at n = 16, below any isolated call, because the two medians are taken in separate windows between which the GPU's clock state moves. The per-term figures of this record are the direct ones in table (2), whose last row times the terms inside `execute()`'s own sequence.

#### Conclusions under section 6, step 1

1. **Item 1 (the pool) is on the materiality line at n = 20 and clearly above it below n = 20.** Its saving is 1.2–4.5% (median 1.8%) of the smallest n ≥ 20 execution, 2.2–20% at n = 16–18, and 0.6–2.6% at n ≥ 22. The primitive it replaces is 215–45 000× slower than its replacement. The sizes where it matters most are exactly where Factorized regions, `run_batch` circuits and PennyLane parameter-shift circuits run on the device. **Recommended: build it** (section 3.1, including the removal of the duplicate zero fill, which costs nothing extra to remove). The owner decides (Q7).
2. **Item 2 (the scratch allocated once) is immaterial as written:** 0.1–2.8% (median 0.6%) of a 100-term n = 20 execution, and the allocation it removes is not where a term's bookkeeping goes; the synchronising copy is. **Recommended: recorded, not built.** The variant that would address the copy, one copy back for all terms at the end of the sweep (bitwise identical, the same partials summed in the same order on the host), would save about 99 × (28–67) µs ≈ 3–7 ms, 1.3–2.7% at n = 20 with 100 terms; it is a change to the kernel-call structure that this ADR did not propose. **Recommended: not in Phase 22, recorded for a later ADR.** The owner decides (Q8).
3. **Item 3 (page-locked device→host) is not adopted: 3.3(c).** No page-locked path beat the pageable copy in any of five runs at any size on this machine. Recorded as "measured, not adopted". Whether native Linux behaves differently is unmeasured and stays so in this phase.
4. **The section 3.4 accounting** (the pool's retained bytes counted as available and printed) is needed if and only if item 1 is built.

#### Owner decisions from step 1

- **Q7 — Item 1.** (a, recommended) Build the pool and remove the duplicate zero fill, as section 3.1. (b) Do not build it: the measured share at n ≥ 20 is at the 2% line.
- **Q8 — Item 2.** (a, recommended) Not built, recorded; the one-copy-for-all-terms variant recorded for a later ADR. (b) Build the one-copy variant in this phase. (c) Build item 2 as written despite the measurement.
- **Q9 — The gate's figures (Q4 criteria 2 and 3).** (a, recommended) Criterion 2: the per-execution device allocation + zero fill + free, measured as a same-run ratio by `menos_cuda_memory_benchmark` table (1), is at least 10× cheaper from the pool than direct at n = 20, 22 and 24 (measured 215–45 000×); the expectation clause of criterion 2 is dropped with Q8(a). Criterion 3: replaced by the record of 3.3(c) above. (b) Figures of your own.

**Owner answers (2026-10-31):** (a) to Q7, Q8 and Q9. Item 1 is built (steps 2 and 5 of the plan); items 2 and 3 are recorded, not built; the gate is Q4 with criterion 2 as the 10× ratio and criterion 3 as the record of 3.3(c).

### 10.2 Steps 2 and 5: the pool, the single zero fill, the accounting and the release call — [MENOS], 2026-11-02

Built in `2f02d10` on the code of `72d8f27`, and merged with Phase 21's closure (`aa14168`, documentation only) in `5bf355a`, which is the code under the gate. Sections 3.1 and 3.4 are built as written, with one addition noted below. Items 2 and 3 are not built (Q8, Q9).

**Production code.**

- `src/kernels/cuda/device_pool.{h,cpp}`: `DevicePool`, one explicit pool for device 0 (`cudaMemPoolCreate`, pinned device-local memory, no export handle), release threshold set to the device's total memory (Q2). `allocate()` is `cudaMallocFromPoolAsync` on the legacy default stream; `release()` is `cudaFreeAsync` on it and never throws; `reusable_bytes()` is `ReservedMemCurrent − UsedMemCurrent` read after a stream synchronisation, so a release still in flight is counted; `trim()` synchronises and calls `cudaMemPoolTrimTo(pool, 0)`. The fallback (no device, `cudaDevAttrMemoryPoolsSupported` = 0, or a failed create or attribute set) is `cudaMalloc`/`cudaFree` with `pooled() == false` and `reusable_bytes() == 0`. `device_pool()` is a function-local static like `device_info()`. The constructor takes `use_pool` so the fallback has a test on a device that supports pools; production constructs only `device_pool()`.
- `DeviceBuffer` allocates from `device_pool()` and releases to it; its `cudaMemset` is gone, so `initialize()` is the one zero fill. The ledger `Charge` is made exactly as before, so `peak_device_bytes()` keeps its meaning.
- `kernels::cuda::device_available_bytes()` is now free + `device_pool_bytes()` (new), the figure `current_hardware_profile()` reads into `HardwareProfile::cuda_device_available_bytes`; the profile also carries `cuda_device_pool_bytes` and `cuda_device_pooled`.
- `Plan` gains `device_pool_bytes` and `device_pooled`, set exactly when `device_available_bytes` is (a CUDA candidate or region exists); `explain()` appends ` device_pool_bytes=… pooled=…` to the `estimated` line. A plan with no CUDA in it prints byte for byte what it did.
- `menos::release_device_memory()` calls `device_pool().trim()`; without CUDA it is an empty function. Python exports `menos.release_device_memory`.

**One addition to section 3.1.** `allocate()` waits (`cudaStreamSynchronize`) after `cudaMallocFromPoolAsync` before returning the pointer. Section 3.1 relied on the per-gate synchronisation alone; the explicit wait makes a pooled pointer usable by the very next host call, exactly as a `cudaMalloc`ed one is, whoever the caller is. Its cost is inside the `DeviceBuffer` cycle of section 10.3 (1.1–2.4 µs including the ledger charge).

**Change of instrument in two existing tests** (section 3.4, last bullet). A released `DeviceBuffer` no longer raises the driver's free figure, so `tests/api/cuda_peak_memory_test.cpp` and `tests/api/cuda_execution_monitor_test.cpp` compare `device_available_bytes()` (free plus the pool's reusable bytes) where they compared `cudaMemGetInfo` free. The assertions are the same: the 256 MiB control still has to lower the figure by well over 200 MiB and give it back, ten abandons still leave it within 32 MiB, and `ledger peak ≤ estimate` is untouched. The `explain()` golden strings now carry the two new fields, and the CPU-only plan test asserts that neither appears.

**New tests**, all of which skip loudly without a device and so execute only in the `cuda12-*` presets on this machine:

- `tests/engines/dense/cuda_device_pool_test.cpp` (`CudaDevicePoolTest`): the process pool is in use on a device that supports pools; a freed 64 MiB block is retained (`reusable_bytes() ≥ 64 MiB`), the next allocation of that size takes it (`reusable_bytes()` falls by the block, the driver's free figure does not), and `trim()` empties it; the fallback (`use_pool = false`) allocates from the driver, is usable by `initialize()`, retains nothing and trims without throwing; `device_available_bytes()` counts the retained block on top of the driver's free figure; a block that comes back from the pool holding a 12-qubit Hadamard state is `|0…0⟩` with the requested phase after `initialize()`, every amplitude checked.
- `tests/api/cuda_memory_pool_test.cpp` (`CudaMemoryPoolTest`, 20-qubit jobs forced to CUDA): the second job of a size is served from the pool (the driver's free figure does not fall by a state, both results report `peak_device_bytes() == 16 MiB`, and the two statevectors are equal); the plan counts the retained pool and `explain()` prints ` device_available_bytes=… device_pool_bytes=… pooled=true`; `release_device_memory()` empties the pool, returns the state to the driver and is idempotent; a `preview()` after a job sees at least the availability it saw before it (the retained block is counted).
- `python/tests/test_peak_memory.py`: `menos.release_device_memory` is exported and returns `None` twice in a CPU build.

**New symbols and their callers.** `DevicePool`/`device_pool()`: `DeviceBuffer`, `device_pool_bytes()`, `current_hardware_profile()`, `release_device_memory()`. `device_pool_bytes()`: `device_available_bytes()`, `current_hardware_profile()`. `HardwareProfile::cuda_device_pool_bytes`/`cuda_device_pooled` and `Plan::device_pool_bytes`/`device_pooled`: `plan_job()` and `Plan::explain()`. `release_device_memory()`: the public header, the Python binding, and `tests/api/cuda_memory_pool_test.cpp`.

**Documentation** (step 6): `README.md`, `docs/reference/limitations.md` (the retained pool and the pageable copies), `docs/reference/changelog.md`, `docs/reference/python-api.md`, `docs/BUILD_PLAN.md`, `CLAUDE.md` and this ADR.
