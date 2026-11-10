# Changelog

MENOS was built in phases, each passing an acceptance gate before the next began. The package version is **1.1.0**. The repository has no release tags, so this page records the development history by phase rather than by tagged release.

## Phase 24 (in progress): planner accuracy, planning cost and dense-engine throughput

- **Gate fusion in the dense engines, on by default.** The CPU dense engine, the CUDA dense engine and the disk tier sweep the state once per fused block instead of once per gate: a gate joins the newest block that shares a qubit with it when one qubit set contains the other, so every merge removes a pass and widens none, and a diagonal pass is never turned into a dense one. A block's matrix is the product of its gates, computed by the kernel that applies it, so a fused run is exact up to rounding (within `1e-12` of the gate-by-gate run) and keeps every bit-for-bit identity the unfused run has: the same bytes on every thread count, on CPU and CUDA, in memory and on the disk tier. The planner charges the blocks, and `explain()` names the outcome on the region line (`fusion=containment blocks=… absorbed=…`). **New `RunOptions.fuse_gates`** (default `True`; CLI `--fusion on|off`): `False` applies every gate alone, with results and reports byte-identical to before. Measured on the development machine (`docs/adr/0028` section 17.5): on random brickwork circuits the sweep is 2.0–2.5× faster on every thread at 20–24 qubits. The opt-in `fuse_diagonal_gates` is unchanged.
- **AVX2 kernels for the dense CPU engine.** The 1-qubit, 2-qubit and diagonal gate kernels use AVX2 where the processor has it, chosen at load time on every compiler (MSVC included, which had no vector code before). Plain x86-64 loops serve processors without AVX2. Both paths are bit for bit the generic kernel: no fused multiply-add, the same operations in the same order, tested on every operand placement and thread count. Results, plans and reports are unchanged. Measured on the development machine (`docs/adr/0028` section 19.8), against the engine before the change:
  - the 2-qubit kernel is 2.9–4.8× faster on one thread at 16–20 qubits, on Linux and Windows;
  - fused brickwork circuits through `menos.run` are 3.3–4.4× faster on one thread and 1.2–2.4× faster on every thread at 20–24 qubits;
  - the 1-qubit kernel is 1.9–3.4× faster on Windows on one thread, and on Linux matches the compiler-vectorised loops it replaces (0.90–1.50×).
- **Stabilizer counts are correct on every Clifford circuit.** The stabilizer engine's counts sampler combined a generator into the base outcome with OR instead of XOR, so on some circuits a qubit whose exact marginal is 0.5 came out 1 in every shot (seen from 8 qubits up; expectation values and single-qubit measurements were right). Fixed; counts from `direct_stabilizer` plans change wherever the bug fired (`docs/adr/0028` section 18).
- **Planning time is reported**: `Result.plan_seconds` and `Result.replan_seconds` (C++ and Python), `plan_s=` and `replan_s=` on the `actual` line.
- **A Factorized plan that overruns its time budget replans** with per-region observations instead of being refused; a cut plan's overrun is still refused.
- **The disk tier is priced at the dense rate measured on every thread**, with reads and writes at their own bandwidths: measured over predicted 0.81 on Linux and 1.17 on Windows on the Phase 23 gate circuits.
- **The cost model's accuracy is measured per engine** (`docs/planning/cost-model`), replacing the Phase 11 table.
- **Every CPU engine has its own calibrated time model.** Each engine's estimate reports what its runtime is made of: passes, gates, tableau updates, SVDs, term updates, shots. Calibration measures what each of those costs on this machine, engine by engine.
  - **Before:** every engine's work was priced at the dense engine's rate, so a faster dense kernel made every other engine look faster.
  - **Measured on the development machine** (`docs/adr/0028` section 20.5): the typical error on Linux fell from 28× to 1.55× for MPS, from 18× to 1.70× for stabilizer, from 11× to 1.10× for tensor network and from 7.2× to 1.64× for Pauli propagation. Every engine is now within 2.3× typically and 4× at the 90th percentile, on Linux and Windows.
  - **Calibration now costs about 1–2.5 s** the first time a process needs it. Unbudgeted CPU-only jobs still never run it.
- **A job with a time budget picks the plan predicted to be fastest.** Every other job still picks the fewest work units, so unbudgeted plans and their `explain()` output are unchanged. For a budgeted job, every candidate is costed in full; none is reported as pruned (`docs/adr/0028` section 21). A budgeted job may therefore run a different exact plan than the same job without a budget, and draw different counts for the same seed. Because the prediction comes from this process's calibration, a budgeted job whose best two plans are predicted close can also choose differently in another process.

## Phase 23: tiered memory across VRAM, RAM and disk

- **A dense state larger than the memory budget still runs, from a scratch file.** When the in-memory dense state does not fit `memory_budget_bytes` and no statevector is wanted, the planner prices a **disk tier**: the state lives in a scratch file of `16 · 2ⁿ` bytes and gates are applied in passes over a window that fits the budget, with gates on the window's qubits applied chunk by chunk by the unchanged kernels. A job that was refused for memory now runs if the disk has the room. Measured on the development machine (`docs/adr/0027` section 10.3): 31 qubits, a 32 GiB file against a 15.85 GiB budget, 100 gates and 1 000 shots in 7–13 minutes. Counts and expectation values are bit for bit the in-memory engine's, verified at 29 qubits and across chunk sizes, gate shapes and thread counts in the test suite.
- **The tier is priced and shown.** `explain()` names it: `tier=disk passes=… chunk_qubits=… disk_bytes=…` on the candidate line, `tier=disk` on the region line, and the scratch directory, its available space and its measured write and read bandwidth on the `estimated` line. The planner reaches for the tier only when the in-memory state does not fit (never as a speed choice), only for counts and expectation values, only for noiseless circuits, and only as a job's sole region. A job that fits in memory plans and reports exactly as before.
- **New `RunOptions.tier`** (`menos.Tier`; CLI `--tier disk|in_memory`) forces the disk tier or forbids it, and a forced tier never falls back. **New `RunOptions.scratch_directory`** (CLI `--scratch-dir PATH`) names where the file goes; the default is the system's temporary directory, and 5% of the volume or 8 GiB, whichever is larger, is left free. **New `Result.peak_disk_bytes`** (C++ and Python), printed on the `actual` line as `peak_disk_bytes=` when it is not 0. The file is removed when the run ends, including after an exception.
- **The scratch directory's bandwidth is measured once per process**, with at most 1 GiB written and read, and only for a job that could reach the tier. Under a time budget, a directory whose bandwidth is unmeasured is a refusal, never a guess.
- **A device allocation that fails because another process took GPU memory after planning is now `ResourceExceeded`**, naming the bytes asked for, the bytes free and the bytes MENOS's pool retains. It was `InternalError`.
- **Measured and not built:** a host-offloaded CUDA tier (the state in RAM, computed on the GPU window by window over PCIe) did not meet the owner's rule of beating the in-memory CPU engine at 29 qubits on the development machine, so `Tier.HostOffloaded` is a name that is refused with `UnsupportedFeatureError`; unified memory was measured only; overlapping transfers with compute is deferred (`docs/adr/0027` sections 10.1 and 9).
- **Known:** the disk tier's time prediction is pessimistic on Linux with many threads (0.35× measured over predicted in one WSL run, against 0.51–1.17× in four Windows runs), because the dense work rate is calibrated on a single-threaded 13-qubit circuit while the tier runs its 8 GiB windows on every thread. A threaded calibration is a Phase 24 item. The prediction errs towards refusing a time budget it would have met, never towards admitting one it would overrun.

## Phase 22: CUDA memory efficiency

- **GPU states come from a memory pool.** A finished job's device state stays in MENOS's own CUDA memory pool and serves the next job, region, replan or batch circuit of that size, instead of a `cudaMalloc` and `cudaFree` per execution. Measured on the development GPU, the allocation cycle went from 1–3 ms at 20 qubits and 9–23 ms at 24 to 1–6 µs (`docs/adr/0026` section 10). The state is zero-filled once, not twice. Results are unchanged, bitwise.
- **The retained memory is counted and shown.** The planner counts it as available device memory, so no job is refused for memory that is waiting for it, and `explain()` prints it as `device_pool_bytes` beside `device_available_bytes`, with `pooled=false` on a device without memory-pool support (which falls back to `cudaMalloc`).
- **New `release_device_memory()`** (C++ and Python) returns the pool's memory to the driver, for when another process needs the GPU.
- **Measured and not adopted:** allocating the expectation partial sums once per execution saves 0.1–2.8% of a 100-term job; no page-locked device-to-host path beat the pageable copy on the development machine (WSL). Both are recorded in `docs/adr/0026` section 10.1.
- **Documented:** what `memory_budget_bytes` and the two peak figures mean (execution buffers of one attempt, not process memory), and the memory review of 2026-11-02 (`docs/adr/0026` section 11): planning memory, budget reservation, the thread-local ledger, a cap on the retained pool and the plan-to-allocation race, and batch budgeting are recorded for a later phase.

## Phase 21: memory accounting and copy removal

- **Every memory estimate is an upper bound on use, shown by measurement.** Each engine's estimate now counts everything it holds at once. Before, several paths allocated more than they estimated: a dense statevector job up to twice its estimate, and a sampler about six times. Some jobs that were feasible on a tight budget are now refused; each change is listed in `docs/adr/0025` section 10.
- **Avoidable copies removed, with every result bitwise identical:**
  - the statevector is moved into the result, not copied;
  - the CUDA state is copied straight into the result;
  - counts samplers hold 8 bytes per basis state instead of about 48;
  - the MPS sampler uses the state itself;
  - the tensor network frees absorbed tensors;
  - the SVD owns its workspace.

  A 24-qubit dense statevector job's peak fell from 537.6 MB to 269.0 MB.
- **New `Result.peak_host_bytes` and `Result.peak_device_bytes`** (C++ and Python): the most memory the execution really held, measured by a memory ledger. `explain()` appends them to the `actual` line. It also shows each GPU candidate's `device_bytes` and, for a plan that uses the GPU, the `device_available_bytes` it was checked against.
- **Free GPU memory is read again before each job**, not once per process.
- **Known cost:** sampling counts from a dense CPU state of 20–26 qubits measured about 1–2% slower than before this phase. Gate application and expectation values are unchanged within run-to-run variation. The cause is not identified, and the owner accepted it at the gate (`docs/adr/0025` section 10.6).
## Phase 24: planner performance (in progress, not released)

Planning is faster; what it chooses is not. Every plan is pinned by golden files on 45 fixed jobs (`tests/plan/golden/planner_corpus`). This work was done as "Phase 21" on a contributor's fork while Phases 21–23 below were built, and was renumbered to Phase 24 (`docs/adr/0028`) when the two lines were merged on 2026-11-05; the frozen-planner comparison it was verified against predates the memory phases' accounting and was retired at the merge.

- Faster analysis inside planning: the tensor-network cost search (min-fill elimination order) and the MPS bond-dimension bound. The data structures are described for maintainers in `docs/ENGINE_TECHNIQUES.md`.
- Expensive candidates (tensor network, cut-and-recombine, and the tensor network inside each factorized region) are stopped once they cannot beat a cheaper feasible candidate, and the cheaper candidates are costed first. With a time budget this now also applies on the CPU.
- **`explain()` change:** a candidate stopped this way shows `reason="pruned by branch-and-bound: …"` instead of its full figures, and a factorized region's `why=` lists such an engine as pruned. Only losing lines change; see [Pruned candidates](/planning/explain#pruned-candidates).
- A plan-cache hit is cheaper than planning the job again. Since the merge the cache key also carries the hardware profile's memory figures (read afresh for every job since Phase 21), so a cached plan is reused only under the memory it was planned for; a plan on the disk tier is never cached.
- **Not kept at the merge:** batching Z-only observable terms through one Walsh–Hadamard transform in the dense CPU engine. It sums in a different order, so it is not bitwise the per-term kernel, which the Phase 21 baseline and the Phase 23 disk tier require (`docs/adr/0028` section 15).

## Phase 20: PennyLane device

- New PennyLane device **`menos.qubit`** (`menos.pennylane.MENOSQubitDevice`), registered through PennyLane's `pennylane.plugins` entry point and installed with the `pennylane` extra (PennyLane 0.45.1, CPython 3.11 or later).
- **Analytic results:** `expval` and `var` of Pauli-sum observables, `probs` and `state`, all exact and equal to `default.qubit`'s within 1e-12, global phase included. **With shots:** `expval`, `var`, `probs`, `sample` and `counts`, including shot vectors.
- Noise channels are simulated exactly through their own Kraus matrices; results equal `default.mixed`'s within 1e-12.
- Parameter-shift gradients run through the device, each circuit planned on its own.
- Constructs the device cannot run exactly are refused by name with `MENOSDeviceUnsupportedError`, which is both `menos.UnsupportedFeatureError` and PennyLane's `DeviceError`.

## Phase 19: OpenQASM 3 import

- New OpenQASM 3 importer for the static-circuit subset: `qubit`/`bit` declarations and several registers, `U`, `gphase`, every `stdgates.inc` gate, user `gate` definitions, the `inv`, `pow`, `ctrl` and `negctrl` modifiers, `const` and `input` parameters, constant expressions, and all three `measure` forms. Constructs outside it are refused with their line and column.
- The `stdgates.inc` gates have Qiskit's matrices, global phase included: the official file's own definitions, read literally, disagree with the gates they name (measured, `docs/adr/0023` section 4.1).
- The CLI chooses the importer from the program's header and binds OpenQASM 3 inputs with `--input NAME=VALUE`. Python gains `menos.from_qasm3(text, inputs)`.
- **Behaviour change:** the OpenQASM 2 importer keeps measurements in source order. It used to move every measurement to the end, so a program that measured a qubit and then acted on it ran with the gate first, silently. Such a program is now refused, as MENOS refuses mid-circuit measurement everywhere; programs whose measurements are all terminal are unchanged.

## Phase 18: native Windows

- MENOS builds, tests and ships natively on **Windows x64 with MSVC**, CPU only: the engines, planner, runtime, CLI and Python package. Before, the source refused to build on Windows.
- The host-memory probe uses `GlobalMemoryStatusEx` on Windows; the default memory budget is unchanged (half of physical memory).
- LAPACKE on Windows comes from the official OpenBLAS 0.3.34 x64 (LP64) release, downloaded and hash-checked by `scripts/fetch-openblas-windows.ps1` and shipped inside the Python package.
- New presets `windows-dev` (Debug, AddressSanitizer) and `windows-release`, and `scripts/verify-windows.ps1`.
- Windows wheels for CPython 3.10–3.14 (`win_amd64`), built by `cibuildwheel` and the wheel workflow; each wheel runs the Python suite.
- Refused at configure time: CUDA with MSVC, and any compiler other than MSVC on Windows.
- The `noise` extra (`qiskit-aer` 0.17.2) is restored, and `menos.__version__` matches the package version (1.1.0).

## Phase 17: API surface

- `RunOptions.backend` (C++ and Python, `menos.Backend`), the `MENOSBackend` run option `backend` and the CLI flag `--backend cpu|cuda` force the CPU or CUDA backend. A CUDA request that cannot be honoured is refused, never run on the CPU.
- `MENOSBackend` takes `time_budget_seconds`; abandoned attempts appear in the result metadata as `menos_replans`, and `Result.replans` exposes them in Python.
- The CLI takes `--time-budget SECONDS`.
- Each dense region of a `factorized` plan runs on CPU or CUDA, whichever the calibrated model says is faster (measured 1.31–1.63× faster on factorized plans of dense blocks than keeping every region on the CPU). Cut-plan fragments stay on the CPU.
- A time budget that is not finite and greater than 0 now raises `InvalidCircuitError`. Before, NaN was silently ignored.

## Phase 16: specialised CUDA kernels

- Specialised 1- and 2-qubit CUDA kernels, bitwise identical to the generic kernel, which stays for larger gates. They make no host-to-device copy per gate.
- Small CUDA circuits run about 1.6–2× faster end to end. Large ones (22–26 qubits) run about 1.2× faster, because the GPU's double-precision rate is the limit.

## Phase 15: dense CPU engine performance

- Multi-threaded gate application on a MENOS-owned thread pool, used by the dense, density-matrix and trajectory engines. Results are bitwise identical for every thread count.
- Specialised 1- and 2-qubit dense kernels, bitwise identical to the generic kernel.
- `RunOptions.max_threads` (C++ and Python) caps the thread count; the default is every hardware thread.
- The benchmark dashboard records 10 repeats per point, the spread beside the median, and the machine's power state.

## Phase 14: readout mitigation, batch execution and SamplerV2

- `run_batch`: runs several circuits, each planned by `plan_job`, under independent derived seeds.
- Opt-in readout-error mitigation (`run_readout_mitigated`): calibration circuits, exact tensored inversion over all outcomes, a reported standard error per outcome and the overhead gamma. Never labelled exact.
- `MENOSSampler`, a Qiskit `SamplerV2` with `noise_model`, `seed` and `readout_mitigation`.

## Phase 13: noisy trajectory simulation

- The trajectory engine: Monte-Carlo sampling of noisy circuits, one pure state per trajectory, with exact Kraus branch probabilities. Results are flagged not exact, and expectation values carry a measured standard error.
- Opt-ins: `RunOptions.expectation_precision` for expectation values and `RunOptions.noise_method = NoiseMethod.Auto` for counts. Without them, noisy jobs plan exactly as in Phase 12.
- The planner chooses between the density matrix and trajectories by cost; a density-matrix overrun of a time budget can replan to trajectories.
- `Result.expectation_standard_error`; `MENOSEstimator` returns genuine `stds` and takes a `seed`; `MENOSBackend` takes `noise_method`.
- CUDA toolkits 12.0 and later are supported.

## Phase 12: exact noisy simulation

- Noise channels and readout errors in KIR, validated (trace preservation within `1e-12`).
- The density-matrix engine: exact noisy simulation, planned only for jobs that contain noise.
- Qiskit Aer `NoiseModel` support in `MENOSBackend` and `MENOSEstimator` (`noise_model=`), for verified noise types.
- `Circuit.append_channel` and a `readout_error` argument on `Circuit.append_measure` in the native API.

## 0.2.0, the state after Phase 11

| Phase | Delivered |
|---|---|
| 0 | Foundation: build system, toolchain, verification script, error types |
| 1 | KIR, dense CPU engine, plan IR, runtime, result and `explain()` |
| 2 | Python package and a standards-conforming Qiskit `BackendV2` |
| 3 | Factorization, the work-unit cost model, and the first real planner decision |
| 4 | Stabilizer engine and the common engine interface |
| 5 | Observables, light-cone reduction, Pauli propagation, and expectation values |
| 6 | MPS engine and the accuracy policy (exact by default) |
| 7 | Command-line tool and OpenQASM 2 parser |
| 8 | Tensor-network engine (expectation values) |
| 9 | Cut and recombine: exact wire cutting for expectation values |
| 10 | CUDA dense backend, the shared primitive API, hardware calibration, backend selection by measured seconds, and strict time budgets at plan time |
| 11 | Runtime monitoring and replanning; `time_budget_seconds` exposed in Python |

## Notable behaviours introduced along the way

- **Exactness by default** (Phase 6): every engine reachable from Python is exact.
- **Arbitrary-width counts** (Phase 3): classical registers have no 64-bit ceiling.
- **Estimator** (Phase 5): `MENOSEstimator` for exact expectation values.
- **Bitwise-identical CUDA results** (Phase 10): kernels are built without fused multiply-add so they match the CPU exactly.
- **Budgets enforced while running** (Phase 11): an overrun is abandoned at a safe boundary and replanned. The budget bounds one attempt.

## Known deferrals

Recorded, not scheduled: checkpointing and state migration; per-fragment CPU/CUDA selection for cut plans; CUDA kernels specialised to diagonal or permutation gates; asynchronous CUDA gate launches; multi-GPU, ROCm and Metal; CUDA on Windows, Windows ARM64 and non-MSVC Windows toolchains; OpenQASM 3. See [Limitations](/reference/limitations).

## Design records

The reasoning behind each phase is preserved as architecture decision records in the repository's `docs/adr/` directory.
