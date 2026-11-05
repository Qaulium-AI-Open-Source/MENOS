# ADR 0027: Phase 23 — Tiered memory across VRAM, RAM and disk

## Status

**Accepted; implemented; gate PASSED (2026-11-05); section 9 answered (2026-11-02); step 1 measured and its decisions taken (2026-11-04); steps 2 and 3 implemented and verified (2026-11-04, section 10.2); step 5 measured (2026-11-04, section 10.3: criteria 1–3 met on Windows, criterion 3 missed once in WSL on the pessimistic side, for the owner's ruling); step 6 done (2026-11-05, section 10.4: the documentation, and `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0 on `a20c58d`). The owner passed the gate on 2026-11-05, ruling that criterion 3's one WSL miss does not count against it (section 10.5). Phase 23 is COMPLETE.** The owner started Phase 23 on 2026-11-02 as the third phase of the memory roadmap (`docs/BUILD_PLAN.md`, "Memory roadmap (Phases 21–23)"), the same day Phase 22's gate was accepted (`docs/adr/0026` section 10.7). Branch `phase-23-tiered-memory`, from `phase-22-cuda-memory-efficiency` at `0282ded`. The owner answered section 9 the same day, taking the recommended option for every question (recorded at the end of section 9), and approved the implementation plan. Step 1 of section 6 (measure first) is done: the benchmarks are `88294be`, the measurements and the predictions are section 10.1, and nothing under `src/` has changed. Section 10.1 ends with the decisions the owner took on 2026-11-04, each the recommended option: the host-offloaded tier is measured and not built (Q4), the disk tier is built with unbuffered I/O under the Q7 rule, and the two corrections to section 1 stand (section 10.1, "Decisions"). Steps 2 and 3 (the mechanism, the disk tier, the planner and the API) were implemented and verified the same day (section 10.2); step 4 of section 6 is not built; the gate-scale runs and the device-race refusal (step 5) are measured and recorded in section 10.3; the documentation and the verification scripts on the final commit (step 6) are recorded in section 10.4, which ends with the gate's criterion table.

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

**Decisions (owner, 2026-11-04).** The owner took every recommendation above: (1) Q4: the host-offloaded tier is recorded as measured and is not built; step 4 of section 6 is dropped from the implementation plan, and the `HostOffloaded` tier remains a name in the design for a machine whose RAM exceeds its VRAM by more than this one's, to be revisited only with a new measurement; (2) the disk tier is built under the Q7 rule, with unbuffered explicit I/O (`O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING` on Windows) at chunks of at least 256 MiB; (3) the two corrections to section 1 stand: the largest in-memory dense state under the 15.85 GiB default budget on this machine is n = 29, so the smallest size the disk tier reaches here is n = 30, and the gate stays at n = 31; and the executor chooses the chunk size by the minimum pass count rather than the largest chunk that fits (plan decision 6 is revised accordingly in step 2d). Step 2 began the same day.

### 10.2 Steps 2 and 3: the mechanism, the disk tier, the planner and the API (2026-11-04) — [MENOS]

**Commits.** Step 2: `82d1cdf` (2a, the tier as a third coordinate of an engine), `c5aa24d` (2b, `Space::Disk` and `peak_disk_bytes()`), `d14c24c` (2c, the chunk-accumulating expectation and `sample_streamed`), `b60d84f` (2d, the pass plan), `90b7b04` (2e, `FileBackingStore`), `478de0a` and `59765f0` (2f, the chunked executor and `DiskDenseEngine`), `b308631` (clang-tidy findings, platform-independent test circuits). Step 3: `338fe55` (3a, the scratch directory in the hardware profile), `1af2cc8` (3b, the planner), `59c11ba` (3c, `RunOptions::tier`, `scratch_directory`, Python and CLI), `f5e157e` (3d, memory fixtures), `5e91bc2` (the profile stays a literal type; two stale assertions), `d062551` (clang-tidy findings; `verify.sh` leaves the docs record sources out), `bee5263` (a state beyond the dense ceiling is refused before any pass plan). Step 4 (the host-offloaded tier) is not built (section 10.1, the owner's decision 1).

**What was built, as the plan said, with these refinements found while building.**

- *Identity.* An execution engine is `(Engine, Backend, Tier)`; `DiskDenseEngine` is `(Dense, Cpu, Disk)` in `execution_engines()` only, never a candidate of its own: the planner reaches it through the `DirectDense` candidate, which keeps its work units and gains the tier (section 3.6). `find_execution_engine` defaults the tier to `InMemory`; observed floors are keyed by the triple (`RuntimeObservations`).
- *Bitwise identity.* `kernels::cpu::expectation_of_term` is now one call of `expectation_of_term_accumulate` over the whole state plus `expectation_term_phase`, and the executor accumulates chunk by chunk through the same function (each chunk paired with the chunk its X mask's high bits select; the Z mask's high bits give one sign per chunk), so the sum is the same products in the same order. `engines::dense::sample_streamed` walks the same sort-then-walk over chunks read in increasing order. Both are pinned bit for bit against the whole-span functions for every chunk size from one amplitude to the whole state (`tests/engines/dense/chunked_kernels_test.cpp`), and the engine against `DenseEngine` on random circuits at n = 8–14 with one-, two-, three- and five-operand gates (`UnitaryOp`), local, global and straddling operands, every chunk size and window, every thread count, a non-zero global phase, and terms with 0, 1 and many global X bits (`tests/engines/dense_tiered/disk_dense_engine_test.cpp`). The test circuits are drawn from `std::mt19937_64`'s standardised output, not the standard distributions, which differ between libstdc++ and the MSVC STL.
- *Pass plan* (`src/engines/dense_tiered/pass_plan.{h,cpp}`): the greedy rule of section 3.2 as a pure function; `choose_pass_plan` picks, in a window of W qubits, the chunk size with the fewest passes among L = 24..W (larger chunk on a tie; every L down to 0 when the window is below 2^24 or nothing above it fits the gates), the correction of section 10.1 (iv). The step-1 figures for the crossover and ladder families at n = 31 and 33 are pinned as goldens; the benchmark's `count_passes` is now a view of the same plan.
- *Chunk addressing* (`chunked_executor.cpp`): a pass with global set G of g qubits runs over 2^(n−L−g) groups; group t's chunks are `deposit(t, ~G) | deposit(s, G)` for slot s (a portable PDEP), a local operand keeps its bit and global qubit g of G becomes window bit L + rank(g). The monitor is called once at the end of each pass with the operations applied so far.
- *The window* is one `std::vector` with one sector of slack, so a 4096-byte-aligned span of 2^W amplitudes exists wherever the heap puts it; the estimate charges the vector. The host peak is window + max(one gate's transient, the outputs) + the pass vector, with the pass vector bounded by one `Pass` per operation so that an estimate and an execution derive the same window from the same budget.
- *`FileBackingStore`* (section 3.3): `O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC`, unlinked on open, `pread`/`pwrite`; `CreateFileW` with `FILE_FLAG_DELETE_ON_CLOSE`, `ReadFile`/`WriteFile` with `OVERLAPPED` offsets in pieces of at most 1 GiB. Unbuffered (`O_DIRECT`; `FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH`) when the chunk is a whole number of sectors, as the owner decided; a file system that refuses `O_DIRECT` (a tmpfs) gets buffered I/O and `unbuffered()` says so. The file is sized up front (`ftruncate` / `SetEndOfFile`) and written in full by `initialize_store` before any read.
- *The seams* `EngineEstimateRequest`/`EngineExecutionRequest::tier_chunk_qubits` and `tier_window_qubits` (internal, Q10) also carry the planner's choice to the executor: `Region::tier_cost` holds the pass plan's chunk and window and the runtime hands them to the engine, so the plan the planner priced is the plan that runs, with no second choice from the execution budget.
- *Reach only* (`src/plan/planner.cpp`, Q5): the dense candidate moves to the disk tier when its in-memory estimate is infeasible for memory and the job wants no statevector, is noiseless and does not force CUDA; the engine is given the budget less the runtime's measurement bookkeeping, the file must leave `scratch_reserve_bytes` (5% of the volume or 8 GiB) available, and the CUDA comparison is skipped for a candidate on the tier. A forced tier makes every other candidate infeasible with `tier=disk requested: <strategy> has no disk tier` and never falls back; `Tier::HostOffloaded`, a statevector request, noise and `backend=cuda` with a forced disk tier are `UnsupportedFeature`. Seconds on the tier are work ⁄ `work_units_per_second` + `link_bytes` ⁄ min(write, read bandwidth); an unmeasured link under a time budget is a refusal (`the scratch directory's bandwidth is unmeasured`), never a guess.
- *Profile and measurement* (`current_hardware_profile(scratch_directory, measure_scratch)`): the directory's capacity and available space at every plan; its bandwidth from `scratch_bandwidth_for` (256 MiB chunks, at most 1 GiB, once per directory per process, Q3) only when `plan::may_reach_disk_tier` says the job could use the tier (a forced disk tier, or the in-memory dense estimate of the whole circuit over the budget). The profile keeps only the figures, so it stays a literal type; the job spec names the directory.
- *Report.* The candidate line gains ` tier=disk passes= chunk_qubits= disk_bytes=`, the region line ` tier=disk`, a requested tier prints `tier       disk (requested)`, and the estimated line ` disk_available_bytes= scratch_directory="…" disk_write_bandwidth= disk_read_bandwidth= (measured)` or ` disk_bandwidth=unmeasured`, only when the plan has the tier in it; the actual line gains ` peak_disk_bytes=` only when it is not 0. A 3-qubit non-Clifford circuit run with `--tier disk` on this machine:

```text
tier       disk (requested)
candidate 0strategy=direct_dense feasible=true winner=true peak_bytes=9152 work_units=160 tier=disk passes=1 chunk_qubits=3 disk_bytes=128
candidate 1strategy=direct_stabilizer feasible=false winner=false reason="operation 1 (rx) is not a Clifford gate"
candidate 2strategy=direct_mps feasible=false winner=false reason="tier=disk requested: direct_mps has no disk tier"
strategy   direct_dense (exact)
region 0   qubits=[0,1,2] engine=dense backend=cpu tier=disk
estimated  state_bytes=9152 work_units=160 disk_available_bytes=257393086464 scratch_directory="…" disk_write_bandwidth=4.52766e+09 disk_read_bandwidth=5.90646e+09 (measured)
actual     state_bytes=4224 execute_s=0.000901 sample_s=0.000000 peak_host_bytes=5984 peak_device_bytes=0 peak_disk_bytes=128
```

- *One report change for jobs that run today.* A job whose in-memory dense state does not fit the budget and that some other engine already runs (the tensor-network scale tests at n = 31, for instance) keeps its plan, but its dense candidate is now feasible on the disk tier when the scratch directory has the room, so its report shows the tier's figures and the estimated line the scratch figures, and such a job pays the one-off bandwidth measurement. Two tests that asserted the dense candidate infeasible now assert it never wins (`5e91bc2`).
- *API.* `RunOptions::tier`, `RunOptions::scratch_directory`, `Result::peak_disk_bytes()`, `menos.Tier`, `--tier disk|in_memory`, `--scratch-dir PATH`; documented in `docs/reference/python-api.md`.
- *Accounting* (`tests/memory`): `dense_disk_counts` and `dense_disk_expectation` at n = 18 (chunks of 2^12, a window of 2^15) and the `dense_disk_b` baseline; the ledger's disk peak must equal the estimate's file size; the baselines pin the outputs as the dense fixtures' own digests.

**Verification.**

- Windows, `windows-release` (MSVC 14.50): `cmake --build --preset windows-release` and `ctest --preset windows-release` on `bee5263`: 815 tests, 100% passed (the RSS probes reported, not asserted, on Windows as in Phase 21). The tiered benchmark builds in the benchmark tree against the moved pass plan.
- WSL, `dev` (GCC, AddressSanitizer and UndefinedBehaviorSanitizer): `cmake --build --preset dev` and `ctest --preset dev` on `bee5263`: 814 tests, exit 0 (the RSS probes skipped under the sanitizer, as always). UBSan caught one defect on the way (`bee5263`): the disk estimate of a 100-qubit job built a pass plan, shifting a global-qubit mask by more than 63 bits, before the `2^n` overflow refused the size.
- `clang-tidy -p build/dev` over every `.cpp` this phase changed: no `bugprone-*` finding; what remains are two pre-existing `readability-identifier-naming` notes on constexpr `n` and one `performance-inefficient-vector-operation` note in a test.
- `clang-format --dry-run --Werror` clean on every changed file.
- Python, WSL (`uv`, managed CPython 3.14, editable install): `python -m pytest python/tests --deselect python/tests/test_wheel_smoke.py` on `5e91bc2`: 562 passed, 2 deselected (`python/tests/test_tier.py` and the `docs/reference/python-api.md` snippet among them). The C++ changes after that commit are the literal-type profile, the tidy notes and the overflow guard, none of which touch the bindings.
- The whole of `scripts/verify.sh` and `scripts/verify-windows.ps1` run on the final commit in step 6, as the plan says.

**Where this leaves the phase.** Steps 2 and 3 of section 6 are done; step 4 is not built; step 5 (the gate-scale runs at n = 31 on drive C: from native Windows, criterion 3's prediction check within 2×, and the Q8(i) device-race refusal) and step 6 (documentation and the two verification scripts on the final commit) remain. The plan's stop point after step 3 stands: the owner sees the first report with a tier before the gate runs, which write a 32 GiB scratch file and need the machine on AC.

### 10.3 Step 5: the gate-scale runs and the device race (2026-11-04) — [MEASURED]

**Record.** `docs/benchmarks/phase-23-gate/`: `win-run-a`, `win-run-b`, `win-run-c` (Windows, the scratch file on drive C:, `scripts/gate.ps1`), `win-run-a0` (the first attempt, see below), `wsl-run-a` (WSL, ext4, `scripts/gate.sh`), each with `gate.json` (every figure below, the plans' `explain()` text, the counts' digest and the expectation value's bits), `gate.log`, `commit.txt`, `power.log` (on AC at every checkpoint) and the disks before and after; `scripts/summarise.py` prints the tables. The driver is `benchmarks/tiered_gate_benchmark.cpp` (`30266fd`, `f451a7f`), through the public API: 1 000 shots, seed 2311, a 20-term observable, and a 100-operation circuit of the random-pairs family (a CX chain, then RX on a random qubit and CX on a random pair alternating, from `std::mt19937_64`'s own output so it is the same on every platform). Random pairs, not the crossover family of section 10.1: a hundred operations of a layered circuit are shallow enough for the MPS engine to take the job in memory at any size, and the first attempt (`win-run-a0`, random pairs without the chain) left the interaction graph in several components, so the in-memory plan was Factorized rather than the one dense state; neither is the comparison the gate asks for. The Windows runs are on `30266fd`/`f451a7f` (the driver's circuit changed between `a0` and `a`; the library did not) and the WSL run on `5341849` (GCC fixes in the driver and the device-race test; the library did not change); `1ccc66b` is the record itself and `0e7242e` the device-race test's final form.

**Overlap, n = 29 (criterion 2).** The largest size the in-memory dense engine runs on this machine (section 10.1's correction: 8 GiB of state under the 15.85 GiB budget). The same job on the in-memory engine (`tier = InMemory`, `backend = Cpu`) and on the disk tier forced:

| run | n | in-memory s | disk s | passes | chunk / window | counts | expectation | file gone |
|---|---|---|---|---|---|---|---|---|
| win-run-a | 29 | 124.2 | 90.8 | 1 | 29 / 29 | bitwise | bitwise | yes |
| win-run-a0 | 29 | 13.6 | 94.3 | 1 | 29 / 29 | DIFFERENT | DIFFERENT | yes |
| win-run-b | 29 | 121.1 | 169.0 | 1 | 29 / 29 | bitwise | bitwise | yes |
| win-run-c | 29 | 112.2 | 101.5 | 1 | 29 / 29 | bitwise | bitwise | yes |
| wsl-run-a | 29 | 61.7 | 72.9 | 1 | 29 / 29 | bitwise | bitwise | yes |

The counts (1 000 shots, 29 clbits) and the expectation value are the same bits on the two paths in every run. With the whole 8 GiB state inside the budget the forced tier's window holds it all, so the run is one pass over the file (write, read and write, the sampling read, and the terms' reads); the chunk pairing is exercised at n = 31 below and in the test suite.

**Reach, n = 31 (criteria 1 and 3).** The smallest size the in-memory engine cannot run here (16 × 2³¹ = 34.4 GB of state against the 15.85 GiB budget), the planner left to itself with the default budget:

| run | n | tier | passes | chunk / window | file bytes | predicted s | measured s | measured / predicted | within 2x | peak host | peak disk | file gone |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win-run-a | 31 | disk | 4 | 25 / 29 | 34359738368 | 825.9 | 479.7 | 0.58 | yes | 8590026448 | 34359738368 | yes |
| win-run-a0 | 31 | disk | 4 | 25 / 29 | 34359738368 | 781.5 | 915.4 | 1.17 | yes | 8590026688 | 34359738368 | yes |
| win-run-b | 31 | disk | 4 | 25 / 29 | 34359738368 | 856.9 | 765.2 | 0.89 | yes | 8590026448 | 34359738368 | yes |
| win-run-c | 31 | disk | 4 | 25 / 29 | 34359738368 | 857.4 | 439.5 | 0.51 | yes | 8590026448 | 34359738368 | yes |
| wsl-run-a | 30 | disk | 3 | 26 / 29 | 17179869184 | 439.4 | 155.6 | 0.35 | NO | 8590006208 | 17179869184 | yes |

Every run: the planner chose the disk tier (`tier=disk`), 4 passes with chunks of 2²⁵ amplitudes in a window of 2²⁹ (8 GiB), a 34 359 738 368-byte scratch file that was gone afterwards, `peak_disk_bytes` equal to the file, the host peak inside the budget, and the measured wall time within the factor of 2 of the plan's own predicted seconds on the same run (criterion 3; the Q7 factor). The prediction is the work term plus 40 crossings of the file at the slower measured direction (the initial write, two per pass, the sampling read, and one or two per term); it is pessimistic by 1.1–2× on these runs, because the terms' reads and the passes' writes run nearer the faster direction than the model's single figure. The spread of the measured times (440–765 s on the same circuit) is the disk's and the machine's, not the plan's: the three predictions agree within 4%.

The n = 31 counts' digest and the expectation value's bits are identical across the three Windows runs (`gate.json`: `counts_digest`, `expectation_real_bits`), so the chunked executor is deterministic at the gate size as it is bitwise at the test sizes.

**WSL (ext4), `wsl-run-a`.** The same driver in the WSL clone, once: WSL's budget (50% of 23.5 GiB) stops the in-memory engine at n = 29, so the reach run is n = 30 (a 16 GiB file). The overlap is bitwise as on Windows. The reach run chose the disk tier (3 passes, chunks of 2²⁶ in a 2²⁹ window), the file was gone afterwards and `peak_disk_bytes` equals it, but the measured 155.6 s is 0.35× the predicted 439.4 s: **outside the factor of 2, the one miss of criterion 3**, on the side of a pessimistic prediction. The cause is in the record: the link term (38 crossings of 16 GiB at the slower measured direction, 149 s) is close to the I/O the run did (about 110 s with reads at the read figure), and the rest of the prediction is the work term, 378 G work units at this machine's calibrated 1.3 G/s, 290 s, against roughly 45 s of compute. The calibrated rate is measured on a 13-qubit circuit (`src/plan/hardware_profile.cpp`), which the CPU kernels run on one thread (below `kParallelMinGroups`), while the executor applies the gates to 8 GiB windows on every hardware thread; on Linux the threaded kernels are several times faster per work unit than that single-thread rate, on Windows less so (the four Windows runs' compute is 0.4–1.0× the work term). The same single-thread rate prices every in-memory dense candidate since Phase 10, so this is the time model's known optimism inverted at large n with many threads, not something the tier introduced; the prediction errs towards refusing a time budget it would have met, never towards admitting one it would overrun. Pricing the work with a rate calibrated on a threaded state size, and the link per direction, belongs to Phase 24 (planner accuracy, `docs/BUILD_PLAN.md`). The WSL virtual disk did not grow this time (63 802 703 872 bytes before and after, `scripts/gate.ps1`'s figure read after the run): ext4 reused blocks the virtual disk had already been given.

The scratch bandwidth each plan was priced with:

| run | write GB/s | read GB/s | available bytes |
|---|---|---|---|
| win-run-a | 4.66 | 6.11 | 253087670272 |
| win-run-a0 | 4.70 | 6.16 | 253507432448 |
| win-run-b | 4.26 | 5.53 | 252915048448 |
| win-run-c | 4.14 | 5.60 | 252562518016 |
| wsl-run-a | 4.39 | 6.13 | 971950301184 |

**The device race (section 9, Q8(i)).** `DevicePool::allocate` maps `cudaErrorMemoryAllocation` to `ResourceExceeded` naming the bytes asked for, the bytes free and the bytes the pool retains, and clears the sticky error (`30266fd`). `tests/engines/dense/cuda_device_pool_test.cpp` fills the device with a direct driver allocation outside the pool, asks for a 256 MiB state, and checks the refusal's text and that the next allocation works once the memory is back. Verified in WSL on `0e7242e`: `cuda12-dev` (CUDA 12.4; the RTX 4060 Laptop GPU executing) built, and `ctest -R CudaDevicePool` passed 6 of 6, the race test three times in a row. The test's first form (`30266fd`: one 256 MiB request with 32 MiB left) failed there, and the reason is a property of WSL's driver worth recording: on a WDDM device the driver backs device memory with shared system memory, so an allocation past the free figure is granted rather than refused. Measured with a probe outside MENOS (not checked in): after a `cudaMalloc` of every free byte, further `cudaMalloc`s succeeded for another 16 GiB (61 blocks of 256 MiB); a single `cudaMallocFromPoolAsync` of 256 MiB, 1, 2, 4 or 8 GiB from a pool with MENOS's properties succeeded past zero free, while a 512 MiB one was refused; in the test's own shape (the hog to 32 MiB, then 256 MiB pooled blocks) the refusal came at the second block in two runs. The grant varies between calls, so the test now asks for states until the refusal comes, bounded at 64 GiB, and checks that it is the named refusal (`0e7242e`); on a native Linux driver the first request is refused. The mapping itself, `cudaErrorMemoryAllocation` to `ResourceExceeded`, is the same on every driver; what WSL changes is only when the driver says no. `device_available_bytes` stays the device's own memory: the shared system memory the GPU reaches over the bus is not device memory, MENOS does not count it, and this phase does not change that.

**Criteria after step 5.** 1: met (this section). 2: met (the overlap runs here; the test suite at n ≤ 16, section 10.2). 3: met on the four Windows runs (0.51–1.17×); missed once, in WSL (0.35×, the prediction pessimistic), with the cause above, for the owner's ruling at the gate. 4 and 5: met (section 10.2). 6: step 6.

### 10.4 Step 6: the documentation and the verification scripts on the final commit — [MEASURED], 2026-11-04 to 2026-11-05

**The documentation (`55ff527`).** The user-facing record of the tier: `README.md` (the planner and dense-engine rows); `docs/reference/changelog.md` (the Phase 23 section); `docs/reference/limitations.md` (the disk-tier engine bullet, a "Memory tiers" section with the scratch file's name pattern, the crash case, the WSL virtual-disk note and the absence of a statevector on disk, the device race as `ResourceExceeded` with the WSL over-grant, `peak_disk_bytes`, the CLI flag); `docs/planning/memory-constraints.md` (a "The disk tier" section: the checks the planner makes, an `explain()` sample and a Python snippet that `python/tests/test_docs_snippets.py` executes); `docs/planning/explain.md` (the `tier` line and the candidate, region, estimated and actual fields); `docs/reference/configuration.md` (`tier`, `scratch_directory`); `docs/build/command-line.md` (`--tier`, `--scratch-dir`, the usage lines); `docs/engines/dense.md`, `docs/gpu/gpu-memory.md`, `docs/reference/exceptions.md`, `docs/troubleshooting/memory-errors.md`, `docs/runtime/resource-exceeded.md`, `docs/get-started/requirements.md` (the 32-qubit row and the scratch-disk paragraph); `docs/BUILD_PLAN.md` (the Phase 23 outcome paragraph and the Phase 24 calibration item); and the `CLAUDE.md` Phase 23 section, which states that the gate is not yet confirmed. `docs/reference/python-api.md`'s export list already carried `Tier` from step 3. The documentation site's record directories (`docs/benchmarks/`, like `docs/adr/`) are excluded from the site's navigation check and build (`docs/scripts/check-nav.mjs`, `docs/.vitepress/config.mts`), because the step-1 and step-5 records carry analysis pages that are not site pages. `npm run build` in `docs/` (the navigation check, the VitePress build and the anchor check) passed on Windows at `55ff527`; no documentation page changed after it.

**The two fixes the first runs of `scripts/verify.sh` asked for.** Run 1 (`55ff527`) failed at the format stage: the stage formats every tracked C++ source, including the step-1 record's analysis program under `docs/benchmarks/phase-23-baseline/scripts/`, which `4f9f2db` formats. Run 2 (`4f9f2db`) passed the format, clang-tidy, `dev` (814/814), `release` (814/814) and benchmark-build stages and failed at clang-tidy over `benchmarks/`, where `bugprone-*` findings are errors: the gate driver `benchmarks/tiered_gate_benchmark.cpp` (step 5) read three `std::optional` plan figures without checking them and let an exception escape `main`. `a20c58d` gives them `value_or` defaults (the figures are present whenever the plan has the disk tier, which the driver already checks) and a `try`/`catch` around `main` that prints the error and exits 1. It is the only change under `benchmarks/` after step 5 and nothing under `src/`, `include/`, `tests/`, `cli/` or `python/` changed after `0e7242e`.

**`scripts/verify.sh` on `a20c58d`** (WSL clone `~/MENOS`, `UV_PYTHON_PREFERENCE=only-managed`, log `~/phase23-verify.log`): **exit 0, "all stages passed"**. Tool versions: GCC 15.2.0 for the CPU presets, CMake 4.2.3, clang-format and clang-tidy 21.1.8, Python 3.14.4 (the venv 3.14.7), uv 0.12.21; CUDA 13.3 (V13.3.73) for the `cuda-*` presets and CUDA 12.4 (V12.4.131) for the `cuda12-*` presets; RTX 4060 Laptop GPU, compute capability 8.9, driver 566.14.

- `dev` 814/814 (skips: `WallTimeReportTest.TwentyFourQubitsTwoHundredOperations` and the RSS fixtures under AddressSanitizer, the two `dense_disk_*` ones among them) and `release` 814/814 (no skips), against 749 in Phase 22. The Phase 23 suites ran in both: `PassPlanTest` (12), `FileBackingStoreTest` (7), `DiskDenseEngineTest` (8), `ChunkedKernelsTest` (3), `DiskTierSelectionTest` (11), `TierRequestTest` (5), the extended `MemoryLedgerTest` (9), and the `dense_disk_counts` and `dense_disk_expectation` fixtures of `MemoryFixtureTest.OutputsMatchTheBaselineAndMemoryIsReported` (bitwise against `tests/memory/data/baseline_linux.json`, the ledger checks asserted) and, in `release`, of `MemoryRssTest`.
- Python 563 passed, 2 deselected (8 min 22 s; 557 in Phase 22), including `test_tier.py` and the documentation snippets of `docs/planning/memory-constraints.md`; the wheel built, installed into a clean venv, and its smoke test passed 2/2. The documentation site was not built in the WSL clone (`node`/`npm` are not installed there, as in every run since Phase 21); it was built on Windows, above.
- `cuda-dev` 894/894 and `cuda-release` 894/894 with every GPU test skipping, because driver 566.14 does not run a CUDA 13 build.
- `cuda12-dev` 894/894 (skips: the wall-time test, the RSS fixtures under AddressSanitizer, `CudaExecutionMonitorTest.TenAbandonsOfAnOversubscribingTotalNeverExhaustTheDevice` and `CudaBackendRequestTest.ForcedCudaWithoutAUsableDeviceIsUnsupported`) and `cuda12-release` 894/894 (the latter two skips only): **the GPU tests execute**. In both presets the six `CudaDevicePoolTest` cases report `Passed`, `ADeviceThatRanOutSinceThePlanIsARefusalNamingTheBytes` (the device race, section 10.3) among them, with the `CpuCudaDifferentialTest` suite and the `dense_cuda_*` fixtures.

**`scripts/verify-windows.ps1` on `a20c58d`** (worktree `C:\Users\janak\menos-phase23`, MSVC 14.50.35717 from Visual Studio 18 Build Tools, CMake 4.2.3-msvc3, Ninja 1.12.1, uv 0.12.21, OpenBLAS 0.3.34; log `C:\Users\janak\verify-windows-a20c58d.log`): **exit 0, "all stages passed"** on the third run. `windows-dev` (MSVC with AddressSanitizer) 815/815 and `windows-release` 815/815 (the skips in both are the RSS fixtures, which are reported, not asserted on Windows since Phase 21, plus the wall-time test in `windows-dev`; 750 in Phase 22): **the disk tier's tests execute on Windows**. The 46 cases of `PassPlanTest`, `FileBackingStoreTest` (`ChunksOfWholeSectorsAreUnbufferedOnARealDisk`, on drive C:, among them), `DiskDenseEngineTest`, `ChunkedKernelsTest`, `DiskTierSelectionTest` and `TierRequestTest`, and the `dense_disk_counts` and `dense_disk_expectation` fixtures of `MemoryFixtureTest` (against `tests/memory/data/baseline_windows.json`), report `Passed` in both presets. Python 563 passed, 2 deselected (2 min 45 s); the wheel built and its smoke test passed 2/2 in a clean venv. The first two runs stopped before the pytest stage for reasons outside the repository: the first was launched inside a PowerShell output redirection, which under `$ErrorActionPreference = "Stop"` turns a native command's standard error into a terminating error, so `uv venv`'s progress text ended the process at the venv stage after `windows-dev` and `windows-release` had passed 815/815; the second, launched as the Phase 22 runs were (`Start-Process` with its own redirections), passed both presets 815/815 again and failed at `uv venv` because the ended run had left an empty `.venv` directory in the worktree. That directory was removed and the third run is the one above. No file in the repository changed between the three runs (logs `verify-windows-a20c58d-aborted-run1.log` and `verify-windows-a20c58d-run2-venv.log` beside the final one).

**The gate, criterion by criterion (section 9, Q9).**

| Criterion | Evidence | Status |
|---|---|---|
| 1. Each built tier runs a circuit too large for the tier above it (disk: n = 31 against the 16 GB host budget), recorded in `docs/benchmarks/phase-23-gate/` with the scratch disk, the file size, the pass count and the wall time | Section 10.3: four Windows runs at n = 31 (drive C:, a 34 359 738 368-byte file, 4 passes, 440–915 s) and one WSL run at n = 30 (ext4, 16 GiB, 3 passes, 156 s); the host-offloaded tier is measured and not built (section 10.1, Q4) | met |
| 2. Counts and statevectors bitwise identical to the in-memory dense engine on the overlapping sizes, in the test suite at n ≤ 16 through the chunk seam and in the gate runs at the largest overlapping size; expectations bitwise on the disk tier | Section 10.2: `DiskDenseEngineTest` and `ChunkedKernelsTest` (counts and expectations bit for bit across chunk sizes, thread counts and gate shapes), the `dense_disk_*` fixtures against the Phase 21 baselines; section 10.3: the n = 29 overlap runs bitwise on both platforms, and the n = 31 digests identical across runs | met |
| 3. Predicted seconds within the factor of Q7 (2×) of measured, same run | Section 10.3: met on the four Windows runs at n = 31 (0.51, 0.58, 0.89, 1.17×); **missed once, in WSL at n = 30 (0.35×, the prediction pessimistic)**, with the cause recorded there (the single-threaded work-rate calibration, a Phase 24 item) | met on Windows; one WSL miss for the owner's ruling |
| 4. The planner picks a tier only under Q5's rule, shown in `explain()` with `tier=`, `passes=` and the tier's bytes; a forced tier never falls back; every existing `explain()` golden unchanged | Section 10.2: `DiskTierSelectionTest` (reach-only, forced, refused, time-budget and observed-floor paths) and `TierRequestTest`; the existing goldens in `tests/api/backend_request_test.cpp`, `tests/api/explain_test.cpp`, `tests/cli/golden_test.cpp` and `tests/plan/` pass unchanged | met |
| 5. Every estimate an upper bound on the ledger peak per space, verified by the Phase 21 checks on new fixtures; `peak_disk_bytes()` equals the file size; the scratch file gone after every run, including abandoned ones | Section 10.2: the `dense_disk_*` fixtures' ledger checks (host and disk), `DiskDenseEngineTest` (the file is absent after a completed run and after a monitor abandons one), `FileBackingStoreTest` (gone after destruction and after an exception); section 10.3: `peak_disk_bytes` equal to the file and the file gone after every gate run | met |
| 6. GPU tests execute in the `cuda12-*` presets; the disk tier's tests execute on Windows; `scripts/verify.sh` (WSL) and `scripts/verify-windows.ps1` exit 0 on the final commit; documentation updated; every new symbol has a production caller and tests | This section: both scripts exit 0 on `a20c58d`, the GPU tests executing in `cuda12-dev` and `cuda12-release`, the Phase 23 suites and the `dense_disk_*` fixtures executing in `windows-dev` and `windows-release`; the documentation above; section 10.2 (the callers and tests of every new symbol) | met |

The gate decision is the owner's, including the ruling on criterion 3's WSL miss. Nothing here marks the phase complete.

### 10.5 Gate accepted — [OWNER], 2026-11-05

The owner passed the Phase 23 gate on 2026-11-05, on sections 10.1–10.4 as recorded, ruling that criterion 3's one WSL miss (section 10.3, `wsl-run-a`: 0.35× measured over predicted at n = 30, on the pessimistic side, from the single-threaded work-rate calibration that Phase 24 takes up) does not count against it. The six criteria of section 9 (Q9) are met on that ruling; `scripts/verify.sh` (WSL, the GPU tests executing in `cuda12-dev` and `cuda12-release`) and `scripts/verify-windows.ps1` (the disk tier's tests executing) exit 0 on `a20c58d`, the final code. The host-offloaded tier stays measured and not built (section 10.1, Q4), and the deferrals of section 9 (Q6, Q8) and the calibration item go to Phase 24's kickoff as recorded in `docs/BUILD_PLAN.md`. **Phase 23 is COMPLETE.**
