# ADR 0027: Phase 23 — Tiered memory across VRAM, RAM and disk

## Status

**Accepted; section 9 answered (2026-11-02); step 1 measured (2026-11-04), awaiting the owner's step-1 decisions.** The owner started Phase 23 on 2026-11-02 as the third phase of the memory roadmap (`docs/BUILD_PLAN.md`, "Memory roadmap (Phases 21–23)"), the same day Phase 22's gate was accepted (`docs/adr/0026` section 10.7). Branch `phase-23-tiered-memory`, from `phase-22-cuda-memory-efficiency` at `0282ded`. The owner answered section 9 the same day, taking the recommended option for every question (recorded at the end of section 9), and approved the implementation plan. Step 1 of section 6 (measure first) is done: the benchmarks are `88294be`, the measurements and the predictions are section 10.1, and nothing under `src/` has changed. Section 10.1 ends with the decisions the owner takes before step 2: Q4 (the host-offloaded tier, recommended "measured, not built"), the disk tier under the Q7 rule (passes), and two corrections to section 1.

Labels follow `docs/adr/0014`–`0026`: **[MEASURED]**, **[DERIVED]**, **[MENOS]** (a proposed design decision) and **[OPEN]** (an owner decision, section 9). Every figure in section 1 is **[DERIVED] from reading the source at `0282ded`** or quoted from an earlier ADR's measurement; nothing in this document has been measured for this phase yet. Step 1 of section 6 measures before anything is built.

---

## 1. Context

`docs/BUILD_PLAN.md`, Phase 23: a dense state larger than the fastest memory still runs, by keeping it in a slower tier and streaming chunks of 2^L amplitudes to where the work happens. This section records what bounds a dense job today and which facts the design rests on.

### 1.1 What bounds a dense job today

- **Host.** `StateVector` holds `std::vector<std::complex<double>>` sized 16·2^n bytes, refused against `budget_bytes` before it is allocated (`src/engines/dense/state_vector.cpp:16-28`). The budget defaults to half of physical memory (`kDefaultMemoryBudgetFraction = 0.5`, `src/plan/hardware_profile.h:76`): 16 GB on the 32 GB development machine, so n = 30 (16 GiB) is the last size the CPU dense engine accepts there, and n = 29 the last that leaves room for a counts histogram or an expectation term's bookkeeping.
- **Device.** `DeviceBuffer` is budgeted against `device_available_bytes()` (free VRAM plus Phase 22's retained pool): about 7.4 GB free on the RTX 4060 Laptop GPU, so n = 28 (4 GiB) is the last size the CUDA engine accepts, and n = 29 (8 GiB) is refused.
- **The CUDA path requires the CPU candidate to be feasible first.** The planner costs the CUDA engine only for a `DirectDense` candidate that already fits the host budget (`src/plan/planner.cpp:745`, `dense_candidate.feasible`), because the CUDA engine's host peak includes the statevector copy and the sampler. A job whose output is an expectation alone, with a tiny host peak, is still refused at n > 30.
- **Outputs.** The statevector output is the state itself, moved into `Result` (Phase 21): it needs 16·2^n bytes of host memory by definition. Counts are drawn by one single-threaded pass over the amplitudes in increasing index order (`src/engines/dense/sample.cpp:22-74`), with the shots' uniforms sorted first, which is a streaming computation. An expectation term reads the pairs m and m ⊕ x (`src/kernels/cpu/dense_kernels.cpp:217-232`), single-threaded, in increasing m.
- **Kernels.** `kernels::cpu::apply_matrix(span state, matrix, operands, threads)` and `kernels::cuda::apply_matrix(DeviceState, matrix, operands)` address amplitudes by bit position inside the span they are given, with bit j of the index being qubit j (little-endian). They know nothing about where the span came from. The specialised 1- and 2-qubit kernels are bitwise identical to the generic one (Phases 15 and 16), and every thread count gives bitwise-identical results.
- **Boundaries.** Both dense engines call `monitor->at_safe_boundary(completed)` after every operation (Phase 11); the CUDA engine synchronises after every kernel launch (Phases 16 and 22, frozen). A replan restarts from scratch.
- **No chunking, out-of-core, pinned, managed-memory or qubit-permutation code exists in `src/`**; the only reordering pass, the light cone, drops operations and keeps program order. Gate fusion does not exist and is proposed for Phase 24 (item 7), after this phase.

### 1.2 What a pass costs — [DERIVED]

Let the state have n qubits and the fast tier hold 2^F amplitudes at once. Split the state into 2^(n−L) chunks of 2^L amplitudes, with qubits 0…L−1 **local** (the index bits inside a chunk) and qubits L…n−1 **global** (the chunk number). A gate whose operands are all local is applied to each chunk independently by the existing kernel, unchanged: the chunk is a span of 2^L amplitudes and the operand bit positions are below L. A gate with g global operands pairs the 2^g chunks that differ in those bits: loaded together into one contiguous buffer of 2^(L+g) amplitudes, the global operands sit at buffer bit positions L…L+g−1, and the same kernel applies the gate with the operands remapped. The arithmetic per output amplitude is the same sum in the same order as the in-memory engine performs it, so the result is **bitwise identical** by construction (section 3.1).

A **pass** streams every chunk through the fast tier once: 16·2^n bytes read and 16·2^n bytes written across the tier link. Several consecutive gates can be applied to a chunk group while it is resident, so the traffic is per pass, not per gate; how many gates a pass absorbs is a property of the circuit and the layout (section 3.2), computable at planning time with no execution.

Per pass, at the bandwidths MENOS has measured or can measure:

| link | bandwidth | 2^29 (8 GiB) | 2^31 (32 GiB) | 2^33 (128 GiB) |
|---|---|---|---|---|
| PCIe under WSL, pageable, both directions (`docs/adr/0026` section 10.1, calibration: 3.1 / 3.2 GB/s) | 3.1 GB/s | 5.5 s | 22 s | 88 s |
| host RAM, in-memory sweep (`memory_bandwidth_bytes_per_second`, 32 MiB copy) | measured per calibration, order 10 GB/s | ~1.7 s | ~7 s | — |
| disk, sequential | **unmeasured**; step 1 | ? | ? | ? |

For comparison, the in-memory CPU engine on the crossover circuit spent 2.0–2.2 s on 147 gates at n = 24 and the CUDA engine 1.1–1.2 s (`docs/adr/0026` section 10.4), that is about 14 ms and 8 ms per gate; each doubling of n doubles it, so a gate at n = 31 costs about 1.8 s on the CPU and 1.0 s on the device. **A host-offloaded CUDA pass over PCIe at n = 31 (22 s) therefore pays for itself only if it carries more than about 25 gates that the CPU would otherwise apply in memory.** Below n = 31 the CPU engine runs in memory anyway, so on this machine candidate 1 of the proposal is a speed question that the measurement of section 6 step 1 decides, not a reach question. Candidate 3 (disk) is the reach: n = 31–33 on the 232 GB free on drive C: (a 2^33 state is 128 GiB), or n = 34 on the WSL virtual disk (906 GB reported free, backed by the same physical drive), at a cost per pass the disk's sequential bandwidth sets.

### 1.3 Facts that constrain the design

- **Counts can be streamed bitwise.** `sample()` sums `std::norm(amps[i])` in increasing i and assigns the sorted uniforms as the running sum passes them. Fed the chunks in increasing order, the same loop produces the same bits. The `CumulativeDistribution` used for multi-region counts holds 8·2^m bytes and is built from a span; regions are small, so it is unaffected.
- **An expectation term pairs chunks.** For x_mask with a non-zero global part h = x ≫ L, chunk c pairs with chunk c ⊕ h; the running sum over m stays single-threaded and in increasing m, so each chunk c needs its partner resident when c's turn comes. With a two-chunk window the partner is read again when its own turn comes: twice the read traffic for such terms, charged by the cost model, and the same bits as today.
- **The statevector output of a disk-tier job is a contradiction** (the state does not fit host memory by premise) and is refused at planning with that reason. A host-offloaded CUDA job's statevector output is the host copy itself, as today.
- **Pinned host memory is invisible to the heap tracker** (`docs/adr/0026` section 4): if it is used, the ledger charges it explicitly and the RSS check is what sees it. Phase 22 measured that page-locked device→host copies were never faster than pageable ones on this machine at up to 64 MiB chunks, in five runs (`docs/adr/0026` section 10.1, table (3)); chunk-sized transfers of 1–4 GiB, and transfer overlapped with compute on a second stream, were not measured.
- **The CUDA kernels synchronise after every launch**, and the Phase 11 monitor relies on it. Overlapping a chunk's transfer with another chunk's kernels needs a second stream and asynchronous launches, which are a recorded deferral needing an ADR (`docs/adr/0020`). Section 3.4 keeps the invariant and section 9 asks whether to lift it for the chunked path only.
- **Unified memory under WSL.** `cudaMallocManaged` with oversubscription of VRAM relies on on-demand page migration; whether WSL 2 supports it for this driver, and at what speed, is unknown and is a step-1 measurement, not an assumption.
- **The heap tracker and the RSS check of Phase 21** verify every host estimate on the fixture set. A memory-mapped file's resident pages count in RSS; explicit chunk buffers count in both. Section 3.5 chooses explicit buffers so that both checks keep their meaning.

## 2. Scope — [MENOS], subject to section 9

As `docs/BUILD_PLAN.md` Phase 23, each candidate conditioned on step 1's measurement (section 6):

1. **A chunked pass executor for the dense state** (section 3.1–3.2): the one piece every tier needs, built once, with the backing store and the compute backend behind two small interfaces. Results bitwise identical to the in-memory engine for counts and statevectors, and for expectations on the CPU; expectations computed on the device stay within `src/common/tolerance.h` of the CPU, as the CUDA engine's are today.
2. **Out-of-core CPU dense** (proposal candidate 3): the state in a scratch file, chunks read and written explicitly, compute by the CPU kernels. Reaches the scratch disk's limit. Works on Windows and Linux.
3. **Host-offloaded CUDA dense** (proposal candidate 1): the state in host memory, chunks copied to a `DeviceBuffer` of 2^(L+g) amplitudes, compute by the CUDA kernels. Reaches the host budget instead of free VRAM. Built only if step 1 shows it beats the in-memory CPU engine on at least one circuit family at a size the device cannot hold (Q4).
4. **Unified memory** (proposal candidate 2): measured in step 1 against item 3 and adopted only if it is both faster and accountable; otherwise recorded and not built.
5. **Planner and accounting**: a tier on every dense candidate and region, priced from calibrated bandwidths and the circuit's pass count, chosen only when cheapest among the feasible plans; `explain()` names the tier; the ledger and `Result` report the tier's bytes; a tier is never chosen silently (prime directive 5).
6. **The items carried from the memory review** (`docs/adr/0026` section 11), each in or out by Q8.

Not in scope: section 7.

## 3. Design — [MENOS]

### 3.1 The chunked pass executor

A new engine-side component, `engines::dense::ChunkedExecutor` (name final at plan time), owned by a new `SimulationEngine` per tier (section 3.6) and used by nothing else. It is parameterised by two interfaces that have exactly one implementation each per tier ("abstractions follow the second implementation": the second implementation of each is the second tier, so the interface is justified only once both are in scope; if Q4 removes item 3, the backing store is a concrete class):

- **`BackingStore`**: `read(chunk_index, span<complex> into)`, `write(chunk_index, span<const complex> from)`, `chunk_count()`, `size_bytes()`. Implementations: `FileBackingStore` (a scratch file, section 3.3) and `HostBackingStore` (a host `std::vector`, section 3.4).
- **`ChunkCompute`**: `apply(span<complex> buffer, matrix, operands, threads)` on a resident buffer and `expectation_partial(...)`. Implementations: the CPU kernels on the buffer in place, or a `DeviceBuffer` the buffer is copied into and back out of (section 3.4).

The executor owns one resident buffer of 2^(L+g_max) amplitudes (the **window**), where g_max is the largest number of global operands any single gate of the circuit has (at most `kMaxUnitaryQubits` = 5, and 0 when every gate is local). It applies the circuit as a sequence of **passes**:

- A pass has a set G of global qubits, |G| ≤ g_max. For each group of 2^|G| chunks that differ exactly in the bits of G, in increasing order of the remaining global bits, the executor reads the group into the window (chunk c at window offset rank(c) · 2^L), applies every gate of the pass with its operands remapped (local operand q → q; global operand q ∈ G → L + rank of q in G), and writes the group back. A pass with G = ∅ is one chunk at a time.
- The gates of a pass are a maximal run of consecutive operations whose global operands all lie in G, with G chosen greedily: start G from the first gate's global operands and extend it with the global operands of following gates while |G| ≤ g_max; a gate that would exceed g_max closes the pass. Section 3.2 gives the pass count this yields and why a larger window buys little.
- **Bitwise identity.** Each output amplitude of each gate is computed by the same kernel from the same inputs in the same matrix order as in the in-memory engine; only the addresses differ. The window's thread count follows `RunOptions::max_threads` as today, and every thread count is bitwise identical (Phase 15). Counts come from the existing `sample()` loop fed the chunks in increasing order; a two-chunk expectation window keeps the running sum in increasing m (section 1.3). The proof obligation is a test, not an argument: section 4.
- **The safe boundary is the end of a pass**, when the backing store holds a complete, normalised state and the window holds nothing the store does not. The monitor is called once per pass with `completed` counting operations, as the engines' `completed` does today, so `Result::replans()` reports the same figure. A monitor that throws unwinds through the RAII store and window; a replan restarts from scratch, as today (checkpointing stays a deferral).
- **Initialisation** writes chunk 0 with amplitude 0 set (and the global phase) and every other chunk as zeros, one write per chunk: one half-pass.
- **Outputs.** Counts: one read pass through `sample()`. Expectation: one read pass per term, two-chunk window where the term's x_mask has global bits. Statevector: refused at planning for the disk tier; for the host tier it is the host vector moved into the result, no copy.

### 3.2 How many passes a circuit needs — [DERIVED]

With L local qubits and the greedy rule above, the pass count is a function of the circuit and L, computed at planning by walking the operations once: it is what the cost model charges (section 3.6), and `explain()` prints it. Two facts bound it:

- Every gate on local qubits only is free (absorbed into the current pass). For a layered circuit on n qubits with L local, a fraction about (n − L)/n of the 1-qubit gates and about 2(n − L)/n of the 2-qubit gates touch a global qubit. At n = 31, L = 28 that is 10% and 19%; at n = 33, L = 28 it is 15% and 30%.
- A pass closes when the next global gate's operands do not fit G. With g_max = 2 (2-qubit gates, no `UnitaryOp` above 2 qubits) and a layered circuit, G fills with the first two distinct global qubits touched, and the next gate on a third global qubit closes the pass. The number of passes is then about the number of distinct-global-qubit changes along the circuit: for the crossover circuit family (six layers of H on all qubits, T on q0, CX on alternating pairs), about one pass per layer per (n − L) ⁄ 2, which at n = 31, L = 28 is 2 passes per layer, 12 passes for the whole circuit; at n = 33 it is about 3 per layer, 18 in all.

**Why not relabel qubits instead.** A qubit-index swap (exchange a global qubit with a local one by one extra pass) makes the next run of gates on that qubit local, and a scheduler could choose the layout from the upcoming gates. For a layered circuit it does not reduce the pass count below the greedy rule's (every layer touches every qubit, so every layer needs the global qubits once either way), and it adds a permutation state that the safe boundary, `explain()`, the outputs and the tests must track. It is recorded as a deferral for circuits whose structure rewards it (section 7), to be measured first.

**Why the window is small.** The window holds 2^(L+g_max) amplitudes in the fast tier. On this machine the disk tier's fast tier is host memory under the host budget (16 GB): L + g_max ≤ 29, so L = 27 with g_max = 2, 2 GiB chunks, 8 GiB windows. The host-offloaded tier's fast tier is free VRAM (7.4 GB): L + g_max ≤ 28, L = 26 with g_max = 2. Both the chunk size and the window are set from the fast tier's budget at planning, printed, and overridable for tests (section 4: a 2^10 chunk makes a 14-qubit job run 16 chunks through every path).

### 3.3 The disk tier: `FileBackingStore`

- One scratch file of 16·2^n bytes, created with `std::filesystem` in the directory of `RunOptions::scratch_directory` if set, else `std::filesystem::temp_directory_path()` (Q2), named uniquely per execution, opened for unbuffered sequential reads and writes of whole chunks at explicit offsets (POSIX `pread`/`pwrite`; Windows `ReadFile`/`WriteFile` with an `OVERLAPPED` offset, the file opened with `FILE_FLAG_SEQUENTIAL_SCAN`), and deleted by the destructor, also on exceptions. No `mmap`: with explicit chunks the page cache would only double-count the resident window in RSS and make the cost unpredictable; step 1 measures `mmap` anyway (Q1) so the choice is on evidence.
- Feasibility: `std::filesystem::space(scratch_directory).available` must exceed the file size plus the free-space reserve the owner sets (Q2), read at every `plan_job` like free VRAM; the window plus the sampler's bookkeeping must fit the host budget. The file size is charged to the ledger in a new `Space::Disk` and reported as `Result::peak_disk_bytes()` and in `explain()` (`disk_bytes=`), so that "estimate is an upper bound on the peak" holds per space.
- Calibration: the scratch disk's sequential read and write bandwidth, measured on first need (not at process start: a 1 GiB write to the user's disk is not something every job should do), with a file of the chunk size the plan would use or 1 GiB, whichever is smaller, cached per process and per directory, printed by `explain()`. The measurement itself is charged and reported as part of the job that triggered it. Q3 asks whether a measurement that writes to disk may happen without the caller asking for it.

### 3.4 The host-offloaded tier: `HostBackingStore` and device compute

- The state lives in a host `std::vector` of 16·2^n bytes, charged to `Space::Host`, which is also the statevector output when one is wanted. The compute side holds one `DeviceBuffer` of 2^(L+g_max) amplitudes from Phase 22's pool and copies each window in (`cudaMemcpy` host→device), applies the pass's gates with the existing device kernels, one synchronisation per launch as today, and copies the window out.
- Pinned memory: not used unless step 1 shows chunk-sized pinned transfers beat pageable ones (Phase 22 found no gain at ≤ 64 MiB). If adopted, the vector is page-locked in place with `cudaHostRegister` once per execution, charged explicitly, and the RSS check verifies it.
- Overlap: none in the first implementation; one synchronisation per launch stays (frozen invariant). Q6 asks whether this phase may add a second stream for the chunk copies, which would be the ADR that the asynchronous-launch deferral requires.
- Expectation: computed on the device per window as the CUDA engine does today (partial sums reduced on the host in fixed order), so it is within tolerance of the CPU, not bitwise, exactly as the in-memory CUDA engine's expectation is (`docs/adr/0014`). Counts: the host vector is the state, so `sample()` runs on it directly and is bitwise identical to the CPU engine, as the CUDA engine's counts are today.

### 3.5 Accounting

- `MemoryLedger::Space` gains `Disk`; `Charge(bytes, Space::Disk)` is made by `FileBackingStore` for the file; `Result::peak_disk_bytes()` reports it; `explain()` prints `disk_bytes=` on candidates and regions that use the disk tier and `disk_available_bytes=` when a plan has the disk tier in it, mirroring Phase 21's device figures. The estimate for every tier is state + window + outputs + bookkeeping per space, and the heap tracker, RSS and ledger checks of Phase 21 run on new fixtures per tier (section 4).
- Nothing changes for a job that uses no tier: every existing estimate, `explain()` line and baseline figure is byte for byte what it was.

### 3.6 Planner, `Plan` and `explain()`

- A new field on `Region` and `CandidateSummary`: `Tier tier` with `enum class Tier { InMemory, HostOffloaded, Disk }` (names final at plan time). Every existing plan has `InMemory`, and `explain()` prints `tier=` only when it is not `InMemory`, so existing goldens are unchanged.
- The dense candidate is costed per (backend, tier) combination that is feasible: (Cpu, InMemory) and (Cuda, InMemory) as today; (Cuda, HostOffloaded) when the state fits the host budget but not the device; (Cpu, Disk) when the state fits neither the host budget nor the device but the window fits the host budget and the file fits the scratch disk. A tier is considered only when the in-memory state does not fit the memory it would otherwise use (Q5), so no job that runs today changes its plan.
- Seconds for a tier = passes × 2 × 16·2^n ⁄ link bandwidth + work ⁄ compute rate + per-pass fixed costs (launches, syscalls), with the pass count from section 3.2's walk; the link bandwidth is the calibrated PCIe figure (host-offloaded) or the measured disk figure (disk). The tier candidate competes with the other feasible candidates on the same footing as today's backend choice (predicted seconds, ties to the simpler plan), and the winner is still selected by the existing rule (work units among feasible candidates; seconds only for the backend/tier choice and the time budget). Q5 asks whether a tier may be chosen only when nothing else is feasible (reach only) or also when it is predicted faster (reach and speed).
- `RunOptions::tier` (optional, like `backend`) forces a tier for tests and benchmarks, and a forced tier never falls back (the Phase 17 rule): infeasible is `ResourceExceeded` with the reason. Python mirrors it; the CLI takes `--tier` and `--scratch-dir`.
- Observed floors: `RuntimeObservations` keys floors on (engine, backend); a tier adds a third key component so an overrun on the disk tier does not raise the in-memory floor.

### 3.7 Unchanged

- `plan_job` is the only planner; the executor makes no decision beyond the pass grouping, which is deterministic from the circuit and L and is printed.
- The kernels, their bitwise identity across thread counts and backends, `-fmad=false`, one synchronisation per launch (unless Q6 lifts it for the chunk copies only).
- Every existing plan, estimate, `explain()` line and the Phase 21 bitwise baselines.
- The density-matrix, trajectory, stabilizer, Pauli, MPS and tensor-network engines: no tier (section 7).

## 4. Testing — [MENOS]

- **Step-1 measurement tool**, `benchmarks/tiered_memory_benchmark.cpp` (a benchmark, not a test): (i) sequential read and write bandwidth of the scratch directory for 256 MiB, 1 GiB and 2 GiB chunks through the explicit I/O of section 3.3 and through `mmap`/`CreateFileMapping`, on Windows and in WSL; (ii) PCIe host→device and device→host at 1 and 2 GiB, pageable and pinned, and a two-stream overlapped variant; (iii) `cudaMallocManaged` at 2^29 and 2^30 amplitudes on WSL: whether it allocates, whether a gate sweep completes, and its time against (ii); (iv) the pass count of section 3.2 for the crossover circuit family and the correlation-benchmark families at n = 28–34 and L = 24–28 (analysis only); (v) the in-memory CPU and CUDA per-gate time at n = 26–30 as the competitor. JSON output through an environment variable; medians and spread per `benchmark_spread.h`; same-run ratios.
- **Chunk seam.** The chunk size is an engine parameter settable by tests (`RunOptions::tier_chunk_qubits` as an undocumented test knob, or an internal request field; plan-time decision), so every tier path runs at n = 10–16 in milliseconds.
- **Bitwise tests, per tier** (`tests/engines/dense/tiered_*_test.cpp`): counts and statevector (host tier) equal the in-memory engine's bit for bit on random circuits at n = 10–16 with chunk sizes 2^4–2^12, including gates with 1, 2 and 5 global operands (`UnitaryOp`), gates straddling local and global, expectation terms whose x_mask has 0, 1 and several global bits, every thread count, and a circuit where every gate is global. Expectation on the host-offloaded tier within tolerance; on the disk tier bitwise.
- **Pass grouping**: unit tests of the walk (section 3.2) on hand-written circuits with the expected pass count and groups; the crossover family's pass counts pinned as goldens.
- **Monitor**: the chunked engine calls the monitor once per pass; a throwing monitor abandons mid-circuit, deletes the scratch file, releases the window, and the next attempt is correct (the Phase 11 pattern).
- **Scratch file**: created in the requested directory, unique, deleted on success and on exception; refused when the directory lacks space; refused with the reason when the statevector output is requested on the disk tier.
- **Accounting**: new memory fixtures per tier under `tests/memory/` with `ledger peak ≤ estimate` per space, the heap tracker and RSS checks (a page-locked vector, if adopted, is covered by RSS), and `peak_disk_bytes()` equal to the file size.
- **Planner**: a tier is chosen only when the in-memory state does not fit (goldens with an injected hardware profile: small host budget, small device); a forced tier never falls back; the tier appears in `explain()`; every existing `explain()` golden unchanged.
- **Gate-scale runs** (criterion 1 of the gate): the disk tier at n = 31 on this machine (32 GiB file, 16 GB budget) and the host-offloaded tier at n = 29 (8 GiB state, 7.4 GB free VRAM), each compared with the in-memory engine at the largest overlapping size for bitwise identity of counts, in `docs/benchmarks/phase-23-gate/`, not in the test suite.
- **Windows**: the disk tier runs under `windows-dev` and `windows-release` at test sizes; the CUDA tier is refused at configure time as all CUDA is.

## 5. API changes

- `RunOptions::scratch_directory` (optional path), `RunOptions::tier` (optional forced tier), and the test knob of section 4; `Plan`/`Region`/`CandidateSummary::tier`; `Result::peak_disk_bytes()`; `explain()` lines `tier=`, `passes=`, `chunk_qubits=`, `disk_bytes=`, `disk_available_bytes=`, `disk_bandwidth=`; Python and CLI mirrors (`menos.Tier`, `--tier`, `--scratch-dir`). Nothing is removed.

## 6. Order of work

1. **Measure first, change nothing.** Build `tiered_memory_benchmark`; record (i)–(v) of section 4 in section 10.1 on Windows and WSL. Report to the owner with the pass counts and the predicted seconds per tier for the gate-scale circuits (Q4, Q7). **Rule:** a tier whose predicted time for a 100-gate circuit at the smallest size it reaches exceeds the figure the owner sets in Q7 is recorded and not built. Stop and report if a measurement contradicts section 1.
2. **The chunked executor on an in-memory backing store** (sections 3.1–3.2), the chunk seam and the bitwise tests of section 4: the mechanism proven at small n before any tier.
3. **The disk tier** (3.3), its accounting (3.5), planner candidate and `explain()` (3.6), tests, Windows.
4. **The host-offloaded tier** (3.4) if Q4 keeps it, with the pinned and overlap decisions from step 1.
5. **Gate-scale runs** and the calibration record; the carried items of Q8 that the owner placed here.
6. Docs (README, `docs/reference/limitations.md`, `docs/planning/memory-constraints.md`, `changelog.md`, `python-api.md`, BUILD_PLAN, this ADR); `scripts/verify.sh` (WSL, GPU tests executing in `cuda12-dev`) and `scripts/verify-windows.ps1`, both on the final commit.

## 7. Out of scope

- Qubit relabelling / swap passes (section 3.2): recorded; needs a measurement showing a circuit family where the greedy pass count is the bound.
- Checkpointing or migrating a tiered state across a replan (restart from scratch stays; the Phase 11 deferral is unchanged).
- Tiers for the density-matrix, trajectory, MPS and tensor-network engines, and for `CutAndRecombine` or `Factorized` regions (regions are chosen because they are small).
- Multi-GPU, ROCm, CUDA on Windows, NVIDIA cuStateVec (Phase 29, licence decision) and unified memory as a shipped path unless step 1 says otherwise.
- Gate fusion and SIMD kernels (Phase 24, items 7 and 8). The pass executor applies gates one at a time inside a window so that results stay bitwise identical; it does not fuse matrices.
- Compression of the state on disk, and any approximation.

## 8. Risks

- **Disk writes on the user's machine.** A 32 GiB scratch file and a bandwidth measurement that writes 1 GiB are visible side effects; Q2 and Q3 set where and when. The file is always deleted, including on exceptions and on a thrown monitor, and never left behind on a normal exit; a crash can leave it, and the name pattern is documented.
- **The WSL virtual disk grows** with every write and does not shrink on its own; the step-1 measurement and the gate runs are run on drive C: from Windows where possible, and the WSL VHD size is recorded before and after.
- **Page cache double counting**: the RSS check of a disk-tier fixture would see cached file pages; explicit I/O with `FILE_FLAG_NO_BUFFERING`/`O_DIRECT` is an option step 1 measures; otherwise the RSS fixture for the disk tier is reported, not asserted, as the Windows RSS fixtures are.
- **Test time**: every tier test runs at n ≤ 16 through the chunk seam; the gate-scale runs are benchmarks, not tests.
- **Expectation on the device is not bitwise** (as today); the gate criterion for it is tolerance, stated in Q9.
- **Calibration variance**: the device rate varies about 2× between runs (`docs/adr/0020`); predictions are judged as same-run ratios with the factor of Q7.

## 9. Owner decisions

- **Q1 — Which tiers are in scope?** (a, recommended) The chunked executor and the disk tier (items 1, 2 and 5 of section 2) are in; the host-offloaded CUDA tier (item 3) is in **subject to step 1** (Q4); unified memory is measured only. (b) Disk tier only. (c) Host-offloaded CUDA only (no disk). (d) All three built regardless of measurement.
- **Q2 — The scratch file's location and limits.** (a, recommended) `RunOptions::scratch_directory`, defaulting to the platform temporary directory; feasibility requires the file plus a reserve of 5% of the volume or 8 GiB, whichever is larger, to be available at `plan_job`; no absolute size cap beyond that. (b) A cap in `RunOptions` (`disk_budget_bytes`) with a default you set. (c) No default location: the disk tier exists only when the caller names a directory.
- **Q3 — When may MENOS measure the disk?** (a, recommended) Lazily, the first time a `plan_job` has a disk-tier candidate to price, writing a file of at most 1 GiB to the scratch directory, cached per process; `explain()` prints the figure and that it was measured. (b) Only when the caller forces the tier or names a directory (never for an unforced job). (c) Never: the disk tier is priced from a figure the caller supplies, else not offered.
- **Q4 — The materiality rule for the host-offloaded tier.** (a, recommended) Built only if step 1 predicts, and a step-1 run shows, that at a size the device cannot hold (n = 29 here) it beats the in-memory CPU engine end to end on at least one circuit family of the correlation benchmarks; otherwise recorded as measured and not built. (b) Built regardless, as a reach option for machines whose RAM exceeds their VRAM by more than this one's.
- **Q5 — When may the planner choose a tier?** (a, recommended) Only when the in-memory state does not fit the memory that tier's backend would use (reach only): no job that runs today changes its plan or its `explain()`. (b) Also when the tier is predicted faster (reach and speed), accepting that some jobs between n = 26 and 30 would change plan when a device is present.
- **Q6 — Overlap of chunk transfer and compute.** (a, recommended) Not in this phase: one synchronisation per launch, copies on the legacy stream, as the frozen invariant says; the overlapped variant is measured in step 1 and recorded. (b) This ADR is the one the asynchronous-launch deferral requires: a second stream for the chunk copies, the per-gate synchronisation kept for the kernels, built if step 1 shows at least a 1.3× end-to-end gain at n = 29.
- **Q7 — The gate's prediction factor and the time limit.** (a, recommended) Predicted seconds within 2× of measured for every tier run of the gate (the device rate itself varies about 2× between runs), measured as a same-run ratio; and the rule of section 6 step 1 uses 30 minutes for a 100-gate circuit at the smallest size the tier reaches on this machine. (b) Figures of your own.
- **Q8 — The items carried from `docs/adr/0026` section 11.** For each, in this phase or deferred: (i) `ResourceExceeded` instead of `InternalError` when another process takes device memory between planning and allocation — (a, recommended) in, it is small and the tier fallback paths need the right exception type; (ii) a separately reported planning-memory peak — (a, recommended) deferred to Phase 24 item 2, where the other session's proposal already places it; (iii) budget reservation at the allocation sites — (a, recommended) deferred, own ADR; (iv) a cap on the retained device pool — (a, recommended) deferred until a workload needs it; (v) a batch-level budget or streaming `run_batch` — (a, recommended) deferred, own ADR.
- **Q9 — The gate.** As BUILD_PLAN's proposal, refined as:
  1. Each built tier runs a circuit too large for the tier above it on this machine (disk: n = 31 against the 16 GB host budget; host-offloaded: n = 29 against 7.4 GB of free VRAM), recorded in `docs/benchmarks/phase-23-gate/` with the scratch disk, the file size, the pass count and the wall time.
  2. Counts and statevectors (where the output exists) bitwise identical to the in-memory dense engine on the overlapping sizes, in the test suite at n ≤ 16 through the chunk seam and in the gate runs at the largest overlapping size; expectations bitwise on the disk tier and within `src/common/tolerance.h` on the device tier.
  3. Predicted seconds within the factor of Q7 of measured, same run.
  4. The planner picks a tier only under Q5's rule, shown in `explain()` with `tier=`, `passes=` and the tier's bytes; a forced tier never falls back; every existing `explain()` golden unchanged.
  5. Every estimate is an upper bound on the ledger peak per space, verified by the Phase 21 checks on new fixtures; `peak_disk_bytes()` equals the file size; the scratch file is gone after every run, including abandoned ones.
  6. GPU tests execute in the `cuda12-*` presets; the disk tier's tests execute on Windows; `scripts/verify.sh` (WSL) and `scripts/verify-windows.ps1` exit 0 on the final commit; documentation updated; every new symbol has a production caller and tests.
  
  The step-1 materiality rule of section 6 is yours to confirm or replace.
- **Q10 — The chunk seam's surface.** (a, recommended) An internal field of the engine request set by tests and the benchmark, not on `RunOptions`, so the public API carries nothing a user should not set. (b) A documented `RunOptions::tier_chunk_qubits` for users who want to tune the chunk size.

### Owner answers (2026-11-02)

**(a) to every question, Q1–Q10.** In the owner's words: the chunked executor and the disk tier are in scope, host-offloaded execution is conditional on the Q4 benchmark, unified memory is measurement-only (Q1); `RunOptions::scratch_directory`, defaulting to the system temporary directory, with a reservation of 5% of the volume's capacity or 8 GiB (Q2); lazy disk measurement, at most 1 GiB written, exposed through `explain()` (Q3); the host-offloaded tier must beat the in-memory CPU engine at n = 29 on a correlation-benchmark family (Q4); reach-only tier selection, and no job that runs today changes its plan (Q5); transfer/compute overlap deferred, measured and recorded only (Q6); prediction within 2× of measurement on the same run, and the step-1 rule at 100 gates in 30 minutes at the smallest size a tier reaches (Q7); `ResourceExceeded` on the device race is in, the planning-memory peak moves to Phase 24, and reservation, the pool cap and the batch budget are deferred (Q8); the gate is the six criteria of Q9 (Q9); the chunk seam is an internal engine field, not on `RunOptions` (Q10).

## 10. Gate evidence

### 10.1 Step 1: the measurements (2026-11-03 to 2026-11-04) — [MEASURED]

**Record.** `docs/benchmarks/phase-23-baseline/`: the JSON and logs of every run, the run scripts (`scripts/phase23-step1.sh`, `scripts/phase23-step1-ext4.sh`, `scripts/phase23-step1.ps1`), the pass-count driver `scripts/pass_counts_at_gate_sizes.cpp` with its output in `analysis/`, and `scripts/summarise.py`, which regenerates every table below from the JSON. Benchmarks `benchmarks/tiered_memory_benchmark.cpp` and `benchmarks/cuda_tiered_memory_benchmark.cpp` at `88294be` (the Windows tree also carried two uncommitted comment-only edits to `src/common/observable_restrict.cpp` and `src/plan/hardware_profile.h`, recorded in each `commit.txt`; neither is compiled into the benchmark's work). Medians within a run; the spread below is min–max across runs; every ratio is same-run.

**Machine.** Intel Core Ultra 7 155H (22 threads), 32 GB; NVIDIA GeForce RTX 4060 Laptop GPU, 8.59 GB total, 7.44 GB free (WSL: driver 566.14, runtime 12.4, GCC 13.4 for the CUDA tree and GCC 15.2 for the CPU tree, both Release); Samsung PM9A1 NVMe 1 TB (`MZVL21T0HDLU`), drive C: NTFS, 723 GiB, 250 GiB free; the WSL ext4 virtual disk on the same drive. Host budget: 15.85 GiB on Windows (half of 31.7 GiB physical), 11.73 GiB in WSL (half of its 23.5 GiB). AC power at every checkpoint of every run that counts; one Windows run was invalidated by the AC rule (the charger came out mid-run) and re-run.

**Runs.** WSL `wsl-run-a/b/c` (CUDA side then CPU side, 28–31 min each); Windows `win-run-a/b/c` (CPU side, 13–14 min each, scratch on drive C:); WSL `wsl-ext4-run-a/b/c` (CPU side again with the scratch on the ext4 volume). The first three WSL CPU runs let the benchmark default to `std::filesystem::temp_directory_path()`, which this WSL mounts as a 12 GB **tmpfs**: their disk rows are RAM-backed and are kept as the memory-copy reference; the ext4 runs are the WSL disk figures.

#### (i) Disk: a 2 GiB scratch file written and read in chunks (GB/s)

Windows, NTFS, drive C: (3 runs; the read is after the write of the same file, and Windows has no cache-drop call, so reads through the cache may be cached; unbuffered reads bypass it):

| chunk | path | write | write + flush | read |
|---|---|---|---|---|
| 256 MiB | explicit buffered | 1.44–1.57 | 1.41–1.53 | 4.80–4.97 |
| 256 MiB | explicit unbuffered | 4.69–4.74 | 4.68–4.72 | 6.07–6.09 |
| 256 MiB | mapped | 2.20 | 1.46 | 15.7–16.6 (page cache) |
| 1 GiB | explicit buffered | 1.52–1.55 | 1.51–1.54 | 4.86–5.03 |
| 1 GiB | explicit unbuffered | 4.60–4.71 | 4.59–4.70 | 5.98–6.03 |
| 2 GiB | explicit buffered | 1.51–1.55 | 1.50–1.54 | 4.83–5.05 |
| 2 GiB | explicit unbuffered | 4.62–4.66 | 4.60–4.64 | 5.98–6.00 |

WSL, ext4 virtual disk (3 runs; reads after `posix_fadvise(DONTNEED)`):

| chunk | path | write | write + flush (`fsync`) | read |
|---|---|---|---|---|
| 256 MiB | explicit buffered | 4.51–4.67 | 2.34–2.39 | 4.33–4.40 |
| 256 MiB | explicit unbuffered (`O_DIRECT`) | 4.47–4.56 | 4.42–4.50 | 6.41–6.49 |
| 256 MiB | mapped | 4.48–4.63 | 2.32–2.37 | 18.6–19.0 (page cache) |
| 1 GiB | explicit buffered | 4.48–4.56 | 2.33–2.35 | 3.71–4.29 |
| 1 GiB | explicit unbuffered | 4.77–4.81 | 4.71–4.75 | 6.58–6.75 |
| 2 GiB | explicit buffered | 4.50–4.55 | 2.33–2.36 | 3.15–4.39 |
| 2 GiB | explicit unbuffered | 4.86–4.93 | 4.81–4.87 | 6.78–6.81 |

WSL, tmpfs (3 runs, RAM-backed, the memory-copy reference): explicit write 3.7–4.1, read 5.0–5.3; mapped write 2.6–2.7, read 18.6–18.9.

Findings: on NTFS the unbuffered explicit path (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH`) writes **3.0–3.3× faster** than the buffered one, whose cost is the cache manager's copy, and reads 20–25% faster; on ext4 the buffered write lands in the page cache at the same rate as the unbuffered one but its `fsync` doubles the cost (2.3–2.4 GB/s to durable), and the buffered read after the cache is dropped is 30–50% slower than `O_DIRECT`. The mapped path is the slowest durable writer on both and its read figure is the page cache, not the disk. The unbuffered bandwidth is flat from 256 MiB to 2 GiB chunks (within 3% on NTFS, 8% on ext4). **Decision for step 2 (plan decision 7's condition is met): the scratch file uses unbuffered explicit I/O on both platforms, with chunks of at least 256 MiB.** At these figures a pass at n = 31 (32 GiB written, 32 GiB read) costs **13.1–13.2 s on Windows and 12.3–12.5 s on the WSL ext4 volume**, against a tmpfs (memory-copy) floor of 15–16 s for the buffered path's two copies.

#### (ii) PCIe under WSL: chunk transfers, windows, overlap (3 runs)

| chunk | source | host → device GB/s | device → host GB/s |
|---|---|---|---|
| 1 GiB | pageable | 3.05 | 3.14 |
| 1 GiB | pinned (`cudaHostAlloc`) | 3.05–3.13 | 3.18–3.19 |
| 1 GiB | registered (`cudaHostRegister`) | 2.99–3.02 | 3.17 |
| 2 GiB | pageable | 3.08–3.09 | 3.17–3.19 |
| 2 GiB | pinned | 3.10–3.19 | 3.23 |
| 2 GiB | registered | 3.04–3.06 | 3.22 |

Pinned memory gains 0–3% and registered memory loses 1–2% against pageable under WSL: **not adopted** (the same finding as `docs/adr/0026` section 10.1 for the device→host direction). A 2 GiB window copied in and out costs 1.33–1.45 s, which is 21 one-qubit gates on the resident window when the gate figure includes the first launch (k = 1) and 48–51 when it does not (k = 10, 25); a 1 GiB window costs 0.66–0.76 s, 19–20 and 35–54 gates. Overlap: a 1 GiB pinned copy on a second stream against eight memsets on the first ran in 343–347 ms concurrent against 393–396 ms sequential, that is the whole 50 ms of work hid under the copy and the copy is the bound; recorded under Q6, not built. A pass of the host-offloaded tier at n = 29 (8 GiB each way) costs **5.48–5.50 s**; at n = 31, 21.9–22.0 s, which is section 1.2's 22 s.

#### (iii) `cudaMallocManaged` beyond device memory under WSL (3 runs)

n = 29 (8 GiB on a device with 7.44 GB free): allocates and a one-qubit sweep completes, in 4.26–4.40 s against 0.227–0.235 s for the resident 2^28 sweep scaled by 2, that is **18.7–19.0× slower than resident**; a windowed pass (5.5 s) is cheaper than one managed sweep whenever a pass carries two or more gates. n = 30 exceeds the WSL host budget and was not attempted. The device reports `concurrentManagedAccess = 0` and `pageableMemoryAccess = 0` under WSL. Measurement-only per Q1; nothing is built on it.

#### (iv) Pass counts of the greedy rule (analysis, no execution)

The benchmark's own table (n = 28, 30, 32, 34; `wsl-run-*/cpu.log` section 2) and the driver's rows at the gate sizes (`analysis/passes-gate-sizes.txt`; its n = 28 rows equal the benchmark's). Every family has g_max = 2. Fewest passes over L and w with L + w within the window the tier can hold:

| family | n | gates | global gates | passes, whole circuit | passes, first 100 operations |
|---|---|---|---|---|---|
| crossover | 29 | 264 | 45 | 13 (L = 24, w = 4) | 5 |
| crossover | 31 | 282 | 54 | 18 (L = 25, w = 4) | 6 |
| ladder | 29 | 342 | 60 | 18 (L = 24, w = 4) | 4 |
| ladder | 31 | 366 | 72 | 20 (L = 25, w = 4) | 5 |
| random pairs | 29 | 200 | 46 | 5 (L = 24, w = 4) | 2 |
| random pairs | 31 | 200 | 43 | 6 (L = 25, w = 4) | 2 |
| QFT-like | 29 | 435 | 135 | 33 (L = 24, w = 4) | 4 |
| QFT-like | 31 | 496 | 171 | 42 (L = 25, w = 4) | 5 |

(Window limits: L + w ≤ 28 for the host-offloaded tier, a 4 GiB window in 7.44 GB of free VRAM; L + w ≤ 29 for the disk tier, an 8 GiB window beside the outputs in the 15.85 GiB budget.) A wider window with smaller chunks always wins here: at n = 31 the crossover family needs 24 passes with (L = 27, w = 2), 22 with (26, 3) and 18 with (25, 4). Section 3.2's estimate of 12 passes for the crossover family at n = 31, L = 28 measures 15 at w = 2 (and 1 at w = 3). **The chunk-size rule of plan decision 6 should therefore prefer the smallest L whose window fills the budget, not the largest L**: the executor chooses (L, w) by minimum pass count under the window limit, which the pass plan computes exactly.

#### (v) The in-memory engines on the crossover circuit (per-gate seconds, circuit minus empty circuit over gates)

| n | CPU, Windows (3 runs) | CPU, WSL (3 runs) | CUDA in VRAM, WSL (3 runs) |
|---|---|---|---|
| 26 | 0.0338–0.0344 | 0.0340–0.0360 | 0.0176–0.0180 |
| 27 | 0.0658–0.0676 | 0.0680–0.0759 | 0.0351–0.0481 |
| 28 | 0.1300–0.1343 | 0.1397–0.1546 | 0.0701–0.0971 |
| 29 | 0.2631–0.2692 | 0.2683–0.2974 | does not fit (8 GiB) |
| 30 | does not fit the budget | does not fit the budget | does not fit |

The CPU engine moves 60–66 GB/s of state per gate on both platforms; the device is 1.6–2.0× faster per gate, same run (its clock varies between runs, as `docs/adr/0020` records). **Correction to section 1.1:** n = 30 (a 16 GiB state) does not fit the 15.85 GiB default budget on this machine, so n = 29 is the largest in-memory dense job on both platforms, and the smallest size the disk tier reaches is n = 30, not 31. The gate run of criterion 1 stays at n = 31 (the stricter figure, chosen by the owner in Q9); the Q7 rule is applied below at n = 31, which bounds n = 30 from above.

#### Predictions — [DERIVED] from the figures above, same run

**Q4, the host-offloaded tier at n = 29.** Model: passes × (8 GiB / host→device + 8 GiB / device→host, 2 GiB pageable) + gates × (2 × the n = 28 in-VRAM per-gate figure), against the measured in-memory CPU crossover circuit at n = 29 of the same run, and gates × the CPU per-gate figure for the other families.

| family | predicted seconds (runs a, b, c) | CPU in memory (same runs) | ratio |
|---|---|---|---|
| crossover | 123, 108, 108 | 79, 82, 74 (measured) | 1.32–1.54× |
| ladder | 165, 147, 147 | 98, 102, 92 | 1.44–1.68× |
| random pairs | 66, 55, 55 | 57, 59, 54 | 0.93–1.15× |
| QFT-like | 266, 242, 242 | 125, 129, 117 | 1.87–2.13× |

A pass must carry 35–59 gates to pay for its transfer (5.5 s against 0.15–0.19 s saved per gate); the families carry 6–40. The tier is predicted **slower than the in-memory CPU engine on three families and indistinguishable from it on random pairs** (0.93–1.15×, inside the run-to-run spread of the device figure). Q4(a) requires both a prediction and a step-1 run that show a win; neither does. **Recommendation: record the host-offloaded tier as measured and not build it (Q4), unless the owner takes Q4(b) for machines whose RAM exceeds their VRAM by more.** Section 1.2's conclusion stands with worse numbers for the tier: the measured break-even is 35–59 gates per pass, not 25.

**Q7, the disk tier at n = 31 (100 gates in 30 minutes).** Model: passes × (32 GiB / write+flush + 32 GiB / read, 1 GiB unbuffered chunks) + gates × (4 × the n = 29 CPU per-gate figure), same run. Applied three ways: the family's first 100 operations, the family's passes scaled to 100 gates, and one pass per gate (the worst case the rule can meet).

| platform | seconds per pass | seconds per gate at n = 31 | 100 gates, families (first 100 ops / scaled) | 100 gates, one pass per gate |
|---|---|---|---|---|
| Windows (3 runs) | 13.1–13.2 | 1.05–1.08 | 2.2–3.1 min / 2.4–3.7 min | 23.6–23.8 min |
| WSL ext4 (3 runs) | 12.3–12.5 | 1.07–1.08 | 2.2–3.0 min / 2.4–3.6 min | 22.3–22.6 min |

Whole family circuits at n = 31 on Windows: crossover 8.9–9.0 min (282 gates, 18 passes), ladder 10.8–11.0 min, random pairs 4.8–4.9 min, QFT-like 17.9–18.1 min. With buffered I/O the pass would cost 29–30 s and the worst case 50–51 min, which is why the unbuffered path is the decision above. **The disk tier passes the Q7 rule on every family and in the one-pass-per-gate worst case; it is built (step 2).** These same-run predictions are the criterion-3 figures the gate run compares against (within 2×).

**Section 1 checked.** PCIe 3.05–3.23 GB/s against section 1.2's 3.1; a host-offloaded pass at n = 31 of 22 s as stated; a CPU gate at n = 31 extrapolates to 1.05–1.19 s (section 1.2 said about 1.8) and a device gate to 0.56–0.78 s (said about 1.0); the disk's sequential bandwidth, which section 1.2 left unmeasured, is 4.6–4.7 GB/s written and 6.0–6.1 GB/s read on NTFS. No measurement contradicts section 1; the corrections are the budget bound of (v) and the pass-count preference of (iv).

**WSL virtual disk.** `ext4.vhdx` was 57 993 592 832 bytes before and after the three Windows runs (they write to NTFS) and **62 255 005 696 bytes after the three ext4 runs** (`analysis/wsl-vhd-size.txt`): the scratch files were deleted and ext4 reports the space free, but the virtual disk does not shrink, so a disk-tier job run inside WSL grows the `.vhdx` by up to the file size the first time that space is touched. A limitation to document in step 6; the disk tier's intended home on this machine is drive C: from native Windows.

**Stop.** Step 1 ends here. The owner decides Q4 (the recommendation is "measured, not built"), confirms the disk tier under the Q7 rule, and confirms the two corrections before step 2 begins.
