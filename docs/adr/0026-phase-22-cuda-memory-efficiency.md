# ADR 0026: Phase 22 — CUDA memory efficiency

## Status

**Proposed; section 9 awaits the owner's answers.** The owner started Phase 22 on 2026-10-31 as the second phase of the memory roadmap (`docs/BUILD_PLAN.md`, "Memory roadmap (Phases 21–23)"), while the Phase 21 gate is still open: its final verification was deferred the same day (`docs/adr/0025`, Status). Branch `phase-22-cuda-memory-efficiency`, from `phase-21-memory-accounting` at `9b8833c` (code unchanged since `72d8f27`), because every change here builds on Phase 21's device ledger, its fresh free-memory read and its `device_bytes` accounting. Phase 22 therefore merges after Phase 21. No implementation, source, test or build file has changed yet; the implementation plan goes through plan mode and owner approval next.

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

## 10. Gate evidence

To be written step by step, as `docs/adr/0025` section 10 was. Nothing is recorded yet.
