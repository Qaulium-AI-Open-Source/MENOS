# ADR 0025: Phase 21 — Planner performance

## Status

**Draft (2026-11-03), step 2 in progress.** Section 9 is answered (items 2–6
on 2026-11-03, item 1 and the step-2 order after the step-1 report the same
day). The owner started Phase 21 on 2026-11-03. Its scope is `plan_job`
latency plus the one-time hardware calibration, and it is placed ahead of the
memory roadmap (now Phases 22–24). Step 1 added the benchmark, the fixed
corpus, the golden plan files and `bench/BASELINE.md`. perf was unavailable
(`kernel.perf_event_paranoid=4`), so the profile was taken with a gdb sampling
profiler (`bench/tools/`). Step 2 results are in section 10.

Labels follow `docs/adr/0014`–`0024`: **[MEASURED]**, **[DERIVED]**,
**[MENOS]** (a proposed design decision) and **[OPEN]** (an owner decision,
section 9). A claim with no measurement in this phase is **[HYPOTHESIS]**.

---

## 1. Context

The owner asked for planning to be made much faster, and for a release to
follow. It is done in steps 0–6 (`docs/BUILD_PLAN.md`, Phase 21), each
reviewed by the owner before the next one starts.

What "planning" is, derived from the source (`docs/ARCHITECTURE_NOTES.md`):
`plan::plan_job` (`src/plan/planner.cpp:147`) enumerates a fixed list of
candidate strategies. It costs each one through the engines' `estimate()`
methods, applies the memory and time budgets, and picks the fewest work
units. The cost lies in the analyses behind some estimates:
- operator-Schmidt-rank SVDs for MPS;
- greedy min-fill elimination per observable term for the tensor network;
- the wire-cut search;
- repeated per-component costing for Factorized.

Calibration (`plan/hardware_profile.cpp:233`) runs deadline-bounded
microbenchmarks once per process.

### 1.1 Prior measurements — [MEASURED] in earlier phases, not re-run

| Figure | Value | Source |
|---|---|---|
| One `plan_job` call, four workloads | 0.052–17.2 ms mean; worst sample 85 ms | `docs/adr/0015` §10 |
| Tensor-network planning overhead, expectation, ≤20 qubits, depth 8 | 0.65–10.6 ms | `docs/adr/0011` |
| Planning relative to execution | 0.0005% (statevector n=22) to ~300% (expectation n=16, 0.5 ms run) | `docs/adr/0015` §10 |
| Calibration | "well under a second" (stated in a comment, not measured) | `hardware_profile.cpp:66` |

Planning matters most for short executions and for workflows that plan many
times: variational loops (partly served by the plan cache, `78bbd49`),
batches, and replans.

## 2. Scope

In scope:
- the latency of `plan_job` for every candidate family;
- the plan-cache lookup and key;
- the one-time calibration (`calibrate_hardware`, `measure_cuda_calibration`).

Out of scope:
- execution engines and kernels;
- the cost model's *figures* (what each estimate returns);
- the selection rule;
- `Plan`/`Region` public layout changes (those overlap Phase 22 and need the
  owner's sign-off).

## 3. Invariants — [MENOS], proposed

1. `plan_job` stays the only planner. The runtime never plans.
2. **Byte-identical plans.** For every job, the returned `Plan` (strategy,
   regions, backends, every candidate's feasibility, reason, bytes and work
   units) and `explain()` output are identical to the pre-phase planner's. A
   faster search is acceptable only when it gives exactly the same answer:
   the same tie-breaks, the same pruning outcome, the same reason text.
3. No tuning constant is introduced. Calibration figures still come from the
   same measurements. Any change to a calibration deadline or circuit is an
   accuracy trade-off, and needs its own measurement and an owner decision.
4. The Phase 11 invariants hold (an observation only raises a figure; an
   unbudgeted job installs no monitor). So do the Phase 17 forced-backend
   rules and the plan cache's structural-match rule.
5. Determinism: identical input gives an identical plan, independent of
   thread count or hash order.
6. New dependencies are optional, pinned (ADR 0001), behind a CMake option,
   and added only after owner approval and a measurement showing they help.

## 4. Hypotheses — [HYPOTHESIS], to be measured in step 1

Ranked by expected impact; details and file:line references are in
`docs/ARCHITECTURE_NOTES.md` §4.

1. Tensor-network costing repeats an identical min-fill search for every
   observable term with the same support. The search also selects the next
   vertex by a linear scan (O(V²) overall).
2. The Schmidt-rank bound recomputes the same SVD for every cut a gate spans,
   and generates a matrix for every operation, single-qubit ones included.
3. Factorized costs every component two or three times
   (`estimate_factorized`, `build_factorized_regions`, and the CUDA
   per-region step). `describe_engine_choice` re-estimates every losing
   engine without the prune bound.
4. Each plan makes four or more full copies of the circuit (about 2 heap
   allocations per operation per copy).
5. Cut search: ≤648 evaluations × O(ops). Bounded already.
6. Calibration: about 60–100 ms fixed CPU cost per process (four deadline
   passes of ≥15 ms each, plus a 64 MiB buffer), plus CUDA context and probe.

## 5. Measurement plan (step 1) — [MENOS], proposed

- Build: `release` preset with `-g` (RelWithDebInfo-equivalent) and
  `-DMENOS_BUILD_BENCHMARKS=ON`.
- New benchmark `benchmarks/planner_benchmark.cpp` (Google Benchmark, already
  a dependency). It calls `plan_job` directly against a fixed
  `HardwareProfile`, so calibration noise is excluded. Calibration is timed
  separately, cold, in a fresh process. The fixture matrix uses fixed seeds:
  - random circuits at small/medium/large (n, depth) for statevector, counts
    and expectation (1, 10 and 100+ Pauli terms);
  - Clifford;
  - factorizable (several components);
  - cut-friendly (two blocks joined by one or two wires);
  - noisy (density matrix and trajectory).
- Profile: `perf record -g` for hotspots and `perf stat` for cache misses.
  Allocation count and peak memory come from a counting `operator new` hook
  built only into the benchmark binary (valgrind/heaptrack are not installed
  here; see §9).
- Output: `bench/BASELINE.md`, with machine, compiler, commit, the raw
  numbers, and the spread over repetitions.

## 6. Equivalence testing — [MENOS], built before step 2

Two oracles, both chosen by the owner (section 9, Q2):

- **Golden files** (step 1): `tests/plan/golden/planner_corpus/`, one per
  fixed-corpus job, recorded from the unchanged d89dde3 planner and checked
  by `tests/plan/planner_golden_test.cpp`. A file is `explain()` plus the plan
  fields `explain()` does not print (`tests/plan/plan_fingerprint.h`), or the
  refusal.
- **Frozen planner and randomised differential test**
  (`tests/plan/frozen_d89dde3/`, `tests/plan/planner_differential_test.cpp`).
  CMake extracts the d89dde3 tree with `git archive` at configure time: the
  public headers and every library `plan_job` reaches, without CUDA. It
  compiles that tree with the token `menos` defined as `menos_frozen_d89dde3`,
  which renames the namespace, so both planners link into one test binary
  without colliding. No copy is checked in, so the copy cannot drift. A check
  of the frozen library's strong symbols (`nm --defined-only`) found none
  outside the renamed namespace. Option `MENOS_BUILD_FROZEN_PLANNER` is off by
  default, because a build from an sdist has no git history. The Linux
  presets that `scripts/verify.sh` runs switch it on.
  - The test first checks the oracle: the frozen planner must reproduce all
    45 golden files.
  - It then plans `random_job(seed)` (`tests/plan/random_planner_jobs.h`)
    with both planners and requires identical fingerprints: 2000 seeds in
    Release (about 17 s) and 60 in Debug ASan (200 took 105 s there). The seed range can be changed with
    `MENOS_DIFFERENTIAL_FIRST_SEED`/`MENOS_DIFFERENTIAL_SEEDS`.
  - The jobs draw circuit families (gate soup with every named gate, CCX,
    diagonal unitaries and mid-circuit measurement; Clifford; independent
    blocks; one-wire cut; long-range pairs; wide brickwork; noise), outputs,
    accuracy policies, memory and time budgets, noise method, expectation
    precision, thread cap, diagonal fusion and a forced CPU backend.
  - Over the 2000 Release seeds, every strategy wins at least once. Refusals
    occur too, and 447 jobs record a branch-and-bound prune.
  - CUDA is never present or forced, because the frozen copy has no CUDA.
  - **[MEASURED] sensitivity:** changing the live min-fill tie-break from `<`
    to `<=` makes the test fail at seeds 8, 27 and 41.

## 7. Out of scope

Execution performance; the cost model's figures; new candidate strategies;
multi-threaded planning (it would add a TSan obligation and nondeterminism
risk with no measurement yet).

## 8. Risks

- **A faster search giving a different answer.** For example, a heap that
  breaks ties differently from the linear scan, or memoisation keyed on too
  little. This is mitigated by invariant 2 and the §6 equivalence corpus.
- **Calibration changes silently shifting every budgeted plan.** Mitigated by
  invariant 3.
- **Overlap with Phase 22** (copy removal). Copy reduction here is limited to
  planner-internal scratch unless the owner decides otherwise.

## 9. Owner decisions

**Answered by the owner on 2026-11-03** (items 2–6 below; item 1 stays open until the baseline exists):

- **Q2, equivalence oracle:** both. A frozen, test-only copy of the planner from `d89dde3`, in a separate namespace and not part of the shipped library, for randomized differential testing of plans and `explain()`. Plus golden `explain()` files for the fixed benchmark corpus (recorded in step 1: `tests/plan/golden/planner_corpus/`, checked by `tests/plan/planner_golden_test.cpp`).
- **Q3, tools:** valgrind, heaptrack and ninja may be installed as system tools, never as project dependencies.
- **Q5, calibration:** nothing changes in step 1; cold calibration is measured first. Any later change must keep the derived constants within an accuracy bound agreed after the baseline, and must keep plans identical. Caching the result is preferred over shortening probe windows.
- **Q6, release:** the version is decided after the work: 1.2.1 if the API is unchanged and nothing is added, 1.3.0 only if something user-visible is added. CI builds and tests on every push and pull request; the benchmark runs on demand only.
- **Build location:** natively on Linux in this repository; the stale WSL note in CLAUDE.md is updated.
- **Q4, Rust:** deferred unless the profile shows a tight numeric kernel.
- **Step 2 order of attack**, to be confirmed by the profile: (a) memoise the min-fill search per qubit set; (b) replace the linear-scan vertex choice with a heap (lazy deletion or indexed), with tie-breaking identical to the current scan; (c) cache the SVD per cut and skip gate-matrix construction for single-qubit ops. Circuit-copy removal is left to Phase 22. Each change is its own commit with before/after numbers; anything that does not measurably help is reverted.

**Answered by the owner on 2026-11-03, after the step-1 report** (item 1 and
the step-2 order):

- **Step 2 order:** (d) `BitAdjacency::common()` over the non-zero words only
  (or sparse neighbour lists), then (b) the vertex-choice heap with
  tie-breaking identical to the scan, then (e) no re-costing in
  `describe_engine_choice`, then (c) the per-cut SVD cache and no gate matrix
  for single-qubit operations. (a) stays deferred. Each is its own commit with
  before/after numbers; the golden test and the differential test against the
  frozen d89dde3 planner run after every change; anything that does not help
  is reverted.
- **Investigate, do not implement, report first:** costing cheaper candidates
  before the tensor network so the pruning bound exists (does `explain()` stay
  byte-identical?); why a time budget switches branch-and-bound off, and
  whether pruning can stay on while honouring the budget; why a plan-cache
  hit (115 µs) is slower than planning (94.5 µs) on
  statevector_brickwork/medium.
- **Q1, gate:** median speed-up >= 5x on expectation_ising_budgeted/large and
  expectation_factorized/large; no corpus job slower than baseline by more
  than max(5%, 2x its baseline run-to-run spread); all 45 golden plans
  byte-identical; the differential test against the frozen planner passes on
  randomised inputs.
- **Q5, calibration:** the measurement does not change in this phase. Only
  caching or lazy initialisation may be considered, and the constants must
  not become noisier than the baseline. Separately: report whether the
  constants' run-to-run variation (1.18-1.7x) can change which plan is chosen.
- **Tools:** cache-miss counters are optional (the owner sets
  `perf_event_paranoid` if wanted); nothing blocks on them. Rust stays
  deferred.

The questions as originally asked (2026-11-03):

1. **Gate targets.** Set from the step-1 baseline, as in Phases 15 and 16.
2. **Equivalence oracle (§6).** A frozen test-only copy of the pre-phase
   planner, or `explain()`/`Plan` golden files recorded at `d89dde3`.
3. **Profiling tools.** May valgrind and heaptrack be installed on this
   machine (system tools, not project dependencies)? The alternative is the
   counting-allocator hook alone.
4. **Rust and mimalloc (step 4).** "Rust service layer, C ABI" is a recorded
   deferral needing owner approval plus an ADR. Evaluate it only if the
   profile shows one self-contained kernel dominates?
5. **Calibration.** May its deadlines or probe sizes change at all? If yes,
   only with a measurement showing the figures stay within a bound the owner
   sets.
6. **Release (step 6).** The version number: 1.3.0 is proposed if no public
   API changes. Also: CI that builds, tests and benchmarks on every push (it
   is workflow_dispatch wheels only today), or on demand.

## 10. Step 2 results — [MEASURED]

Method, for every change:
- Build with `cmake --preset release -B build/release-syms -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer" -DMENOS_BUILD_BENCHMARKS=ON`, as for the baseline.
- Copy the benchmark binary built before the change and the one built after it.
- Run the full benchmark four times, alternating before, after, before, after (`--benchmark_repetitions=5 --benchmark_min_time=0.2s`), so that drift in the machine's state falls on both sides.
- `bench/tools/compare.py` pools each side's 10 repetitions and reports the medians. A job counts as slower only if its median got worse by more than the gate's limit, max(5%, 2 × the before side's spread), where spread is max/min − 1 over the before repetitions of the same session.
- Raw files and comparisons: `bench/raw/step2/`.
- After the change, the golden test (45/45) and the differential test against the frozen planner pass in Release.

### 10.1 (d) `common()` over the non-zero words only

`compute_fill(v)` sums |N(a) ∩ N(v)| over the neighbours a of v. Every word
of v's row outside its non-zero span is zero, so its AND with any other row
is zero too. Restricting the popcount to that span gives the same integer
without reading the rest of the row, which for a graph of V nodes is V/64
words.

| Variant | Result |
|---|---|
| Word list: collect the indices of v's non-zero words into a reused vector, popcount over those | Rejected. Large expectation 1.48–1.50x, pairs_controlled/large 1.76x, factorized/large 1.21x. But expectation_factorized/small was 0.87x (limit 15%) and expectation_cut/small 0.94x (limit 5%), both past the gate's limit. `d_word_list.compare.txt` |
| **Word range**: popcount over the contiguous span [first, last) of v's non-zero words | **Kept.** No job slower than the limit. `d_word_range.compare.txt` |

Word range, selected jobs (median of 10 repetitions each side):

| Job | Before | After | Speed-up |
|---|---|---|---|
| expectation_ising_budgeted/large (gate) | 1853 ms | 1072 ms | 1.73x |
| expectation_factorized/large (gate) | 13068 ms | 7869 ms | 1.66x |
| expectation_{z_all,ising,heisenberg,random_paulis}/large | 78–79 ms | 45–46 ms | 1.71–1.72x |
| the same, medium | 8.2 ms | 6.0 ms | 1.37x |
| expectation_factorized/medium | 348 ms | 258 ms | 1.35x |
| expectation_cut/large | 30.5 ms | 19.1 ms | 1.59x |
| expectation_random_pairs_controlled/large | 379 ms | 335 ms | 1.13x |
| Jobs with no tensor-network candidate (counts, statevector, noise) | — | — | 0.99–1.02x |

The long-range random_pairs job gains less from the span than from the list
(1.13x against 1.76x), because a long-range neighbour widens the span. (b)
attacks the other share of the same loop.

Equivalence: golden 45/45; differential test 2000 seeds (Release), 60 seeds
(Debug ASan), plus one extra sweep of 50,000 seeds (10000–59999) in Release,
all identical to the frozen planner.

### 10.2 (b) Heap for the min-fill vertex choice

Each elimination step chose its vertex by scanning every remaining vertex
for the smallest fill, keeping the lowest index on ties: O(V) per step, O(V²)
per search. The dense search now keeps a min-heap of (fill, vertex) pairs
(`std::priority_queue` with `std::greater<>`). A pair compares fill first and
then vertex, which is exactly the scan's order, ties included.
- A vertex whose fill changes gets a new entry. No entry is ever updated in
  place.
- A popped entry is skipped if its vertex has been eliminated, or if its fill
  is no longer the vertex's current fill.
- Every remaining vertex therefore always has one entry holding its current
  fill, so the first entry that survives is the vertex the scan picked.

The sparse fallback (over 23000 nodes) keeps the linear scan. That makes the
existing `DenseAndSparseMinFillAgreeOnRandomGraphs` unit test an independent
oracle for the heap, and it passes.

**Machine noise.** During this measurement, an unrelated process on the
machine used about 65% of a core (a uvicorn server, not part of MENOS), and
the desktop was busy too. The first alternating run had before-side spreads
of 100–350% (`b_heap_noisy.compare.txt`). The re-run, pinned to one core with
`taskset -c 6` (`b_heap.compare.txt`), still varies by 5–9% on jobs this
change cannot affect: statevector and counts jobs never run min-fill. Their
0.91–1.00x are noise. The figures below come from the pinned re-run. They
agree with the noisy run's fastest repetitions: 2.27x, 1.45x and 1.5x on the
same jobs.

| Job | Before ((d)) | After | Speed-up |
|---|---|---|---|
| expectation_factorized/large (gate) | 8942 ms | 3900 ms | 2.29x |
| expectation_ising_budgeted/large (gate) | 1115 ms | 792 ms | 1.41x |
| expectation_{z_all,ising,heisenberg,random_paulis}/large | 47–49 ms | 34–35 ms | 1.37–1.41x |
| expectation_factorized/medium | 270 ms | 181 ms | 1.49x |
| expectation_random_pairs_controlled/large | 410 ms | 303 ms | 1.35x |
| expectation_cut/large | 20.6 ms | 15.4 ms | 1.34x |

No job is slower than the gate's limit.

Equivalence:
- golden 45/45;
- analysis unit tests 50/50, including the dense-versus-sparse oracle and
  the constructed tie case;
- the differential test, 2000 seeds, plus a sweep of 50,000 seeds
  (60000–109999) — all identical to the frozen planner.

### 10.3 (e) No re-costing in `describe_engine_choice`

For each Factorized region, `build_factorized_regions` picks the cheapest
engine (`pick_cheapest_engine`). `describe_engine_choice` then re-estimated
every losing engine, without a prune bound, only to report the infeasible
ones. `pick_cheapest_engine` now optionally hands back the estimate it made
for each engine, and `describe_engine_choice` reuses them.

Why the reported reasons cannot change:
- An engine costed under a prune bound gives exactly the unbounded estimate
  unless it was stopped. The contract of
  `EngineEstimateRequest::prune_above_work_units` allows only one deviation:
  an infeasible result whose reason starts with `kPrunedReasonPrefix`.
- Only the tensor network reads the bound. Its bounded and unbounded searches
  are the same search, except that the bounded one can stop.
- So only an engine pruned there is costed again without the bound. Every
  other engine's estimate is reused as it is.

In expectation_factorized jobs the costly engine, the tensor network, is
usually the pruned one, so it is still re-costed and those jobs gain nothing.

| Job | Before ((b)) | After | Speed-up | Runs |
|---|---|---|---|---|
| counts_factorized/medium | 949 µs | 806 µs | 1.18x | focused, 30 reps each side (`e_reuse_estimates_focused.compare.txt`); full run 1.16x |
| counts_factorized/large | 4045 µs | 3575 µs | 1.13x | focused; full run 1.13x |
| every other job | | | 0.97–1.04x | within noise (`e_reuse_estimates.compare.txt`) |

The gain is small and confined to one family. It was kept because it repeats
across three runs (full, focused, and the fastest repetitions of the full
run). No job is slower than the gate's limit.

Equivalence:
- golden 45/45, cost tests 13/13;
- the differential test, 2000 seeds, plus a sweep of 50,000 seeds
  (110000–159999) — all identical to the frozen planner.

## 11. Investigations the owner asked for (2026-11-03) — [MEASURED] / [DERIVED], nothing implemented

### 11.1 Costing cheaper candidates before the tensor network

In expectation_factorized/large the tensor-network candidate (index 4) is
costed before Factorized (index 5). No feasible candidate exists yet when it
is costed, so its bound is unbounded, and the baseline profile put 85% of the
job's time there.
- Costing Factorized first, while keeping the candidates' printed order, would
  give a bound of 2105344 work units. The tensor network's running total would
  cross that bound early, and the candidate would be **pruned**.
- **`explain()` would not stay byte-identical.** Today the golden file prints
  `candidate 4strategy=direct_tensor_network feasible=true ...
  work_units=3580198848`. Pruned, that line would carry a "pruned by
  branch-and-bound" reason instead. Printing the full figure means computing
  it, and that is the cost being saved.
- The winner, and every other candidate line, would be unchanged.
- **[DERIVED]:** this is possible only by relaxing invariant 2 for pruned
  candidates' lines. That is an owner decision.

### 11.2 Why a time budget turns branch-and-bound off

From `planner.cpp:379–394` and `apply_time_budget` (`planner.cpp:316`):
- **Why it is off.** The time check runs after enumeration. A candidate with
  fewer work units can still be rejected on seconds, so the work-unit
  incumbent at enumeration time may not survive. Pruning a later candidate
  against it would then be unsound.
- **Pruning can stay on soundly in a CPU-only job.** There,
  `apply_time_budget` depends only on the candidate's own figures and on
  observations already known when planning starts (`floor_for`). So each
  candidate's time check can be applied as soon as it is pushed, and the
  incumbent taken over the candidates that are feasible *and* within the
  budget.
  - Selection is fewest work units among feasible candidates, so a
    candidate stopped above such an incumbent can never win.
  - The plan and winner would be identical.
- **Where it stays off.** Two figures are set only after enumeration: the
  CUDA dense-backend step and Factorized's per-region mixed seconds. Jobs
  where either applies (CUDA calibrated, or backend = cuda) keep pruning off.
- **Cost:** as in 11.1, `explain()` changes. Pruned candidates print the
  pruned reason in place of their figures and `est_seconds`. For example, in
  expectation_ising_budgeted/large the cut_and_recombine line, and the tensor
  network's memory reason, would become pruned reasons.
- Again an owner decision on invariant 2.

### 11.3 Why a plan-cache hit costs more than planning (statevector_brickwork/medium)

- **Method:** gdb sampling of `cache_hit/statevector_brickwork/medium`, 635
  samples (`bench/raw/profiles/cache_hit_statevector_brickwork_medium.stacks.gz`).
- **Breakdown of a hit:**

  | Share | Where |
  |---|---|
  | 22.5% | `Plan plan = cached` (`planner.cpp:1147`): copies the cached plan, including its operation list and reduced circuit |
  | 17.5% | lines 1148–1149: overwrites the copies just made with the fresh operations and circuit, and destroys the stale ones |
  | 20.9% | `compute_light_cone`: the fresh reduced circuit (needed) |
  | ~17% | `std::operator+`, `std::to_string` and the string destructor: the `common::ensure` message, built eagerly for every operation in the structural loop (`planner.cpp:1140–1142`) even when nothing differs |
  | 3.9% | `plan_cache_key` |
  | <1% | hash lookup |
  | 16.1% | destroying the returned plan, outside `plan_job_cached` (a fresh plan pays this too) |

- **[DERIVED]:** about 57% of a hit is work that does not need to happen:
  - the eager error message;
  - the copy of the stale operation lists, which are immediately overwritten.
- Removing both changes no plan and no `explain()` output. That would bring
  the hit to roughly 50 µs against 96 µs for planning (an estimate, not
  measured).
- The plan-cache lookup is in scope (section 2). The second fix is a copy
  removal, which the owner left to Phase 22. Whether it belongs here is the
  owner's decision.

### 11.4 Can calibration's run-to-run variation change the chosen plan?

**Where calibration is used.** The planner reads it in two places only:
- estimated seconds, which matter only against a time budget;
- the CPU-versus-CUDA comparisons, in a CUDA build.

A job with no time budget in a CPU build is never affected. Its plan does
not depend on calibration at all, and such a job does not calibrate.

**Measurement.** `benchmarks/calibration_sensitivity.cpp` plans every
fixed-corpus job once per measured calibration sample, under 8 time budgets
(none, and 1 ms to 1000 s): 360 (job, budget) cases.

| Samples | Cases whose outcome depends on the sample | What changes |
|---|---|---|
| 20 CPU samples (`calibration_cpu.csv`) | 6 of 360 | plan ↔ `ResourceExceeded` only: a candidate near the budget is in for some samples and out for others |
| 20 CUDA samples (`calibration_cuda.csv`, CUDA build) | 11 of 360 | plan ↔ `ResourceExceeded` only (5 of them are dense-on-CUDA jobs at a 10 s budget) |

Raw results: `bench/raw/calibration_sensitivity_{cpu,cuda}.txt`.

**Findings:**
- In this corpus, no case changes strategy or CPU/CUDA backend between
  samples, and no unbudgeted case varies.
- **[DERIVED]:** in general, though, a change of strategy or backend is
  possible. It needs two candidates whose estimated seconds (or CPU and CUDA
  times) are within the calibration spread of each other, or of the budget.
  The corpus simply has no such pair.
