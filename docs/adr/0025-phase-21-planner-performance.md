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

### 10.4 (c) Schmidt-rank bound: no matrix for single-qubit operations, ranks cached

`compute_schmidt_rank_bound` (the MPS candidate's bond-dimension bound)
generated a gate matrix for every operation before checking whether it
crosses a cut. It then ran an SVD (LAPACK `zgesdd`) for every cut a
multi-qubit operation spans.

Two changes:
- **Operands first.** Each operation's operands are now checked before its
  matrix is generated, so a single-qubit operation or a measurement never
  builds one. A noise channel still throws the same `InternalError`, whatever
  its size.
- **Rank cache.** Operator Schmidt ranks are cached for the duration of one
  bound computation. The key is the operand count, which operands lie left of
  the cut, and the exact bytes of the matrix.
  - The rank is an integer computed deterministically from exactly those
    inputs, so a repeated key returns the integer the SVD would return again.
  - Keying on bytes is conservative: equal matrices with different
    representations (0.0 and −0.0) are simply computed separately.
  - Gates repeat (every CX on neighbouring qubits is the same matrix and the
    same split), so most SVDs disappear.
  - A 2-qubit gate splits its operands the same way at every cut it spans,
    so each further cut it spans is a cache hit too.

The "skip when every spanned cut is already at its cap" shortcut was
considered and left out. It was not asked for, and on invalid input it would
skip a matrix-generation error the old code raised.

Focused, pinned, alternating runs on top of (b)+(e), 30 repetitions per side
(`c_rank_cache_focused.compare.txt`; full run in `c_rank_cache.compare.txt`):

| Job | Before | After | Speed-up |
|---|---|---|---|
| counts_clifford/small, medium, large | 0.18 / 2.94 / 38.4 ms | 0.05 / 0.73 / 12.6 ms | 3.54x / 4.01x / 3.05x |
| counts_random_pairs/small, medium, large | 0.13 / 0.79 / 3.13 ms | 0.05 / 0.13 / 0.48 ms | 2.62x / 5.95x / 6.52x |
| counts_factorized/small, medium, large | 0.12 / 0.81 / 3.49 ms | 0.05 / 0.29 / 1.06 ms | 2.24x / 2.82x / 3.28x |
| expectation_ising/small | 311 µs | 222 µs | 1.40x |
| expectation_random_pairs_controlled/large | 293 ms | 268 ms | 1.09x |

In the full run, expectation_random_pairs_controlled/large looked 0.88x
slower on fastest repetitions; the focused run gives 1.09x, so that did not
reproduce. expectation_factorized/large was not re-measured: its full-run
0.94x is inside that run's 25% limit, and the focused run shows its medium
size at 1.02x. No job is slower than the gate's limit.

Equivalence:
- golden 45/45;
- the differential test, 2000 seeds, plus a sweep of 50,000 seeds
  (160000–209999) — all identical to the frozen planner.

### 10.5 Gate status after (d), (b), (e), (c): not met

This is a direct measurement against the pre-phase planner, not a product of
the per-step ratios: d89dde3 against c8f9083, pinned to one core,
alternating, 15 repetitions × 2 rounds per side (`gate_head.compare.txt`).

| Job | d89dde3 | c8f9083 | Speed-up | Gate |
|---|---|---|---|---|
| expectation_ising_budgeted/large | 2423 ms | 1022 ms | 2.37x | 5x: not met |
| expectation_factorized/large | 16447 ms | 5294 ms | 3.11x | 5x: not met |

The machine was slower and noisier than in step 1: the baseline's absolute
times are 1.3x step 1's, and the budgeted family's before-side spread is
29–65%. The ratios agree with the chained per-step estimates (about 2.4x and
3.5x). The other gate conditions hold: no job is slower than its limit,
golden 45/45, and the differential sweeps are identical. The candidates
not yet tried either change explain() for pruned candidates (11.1, 11.2) or
are (a), which is deferred. Both are owner decisions. Whether any of them
reaches 5x has not been measured.

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

## 12. Step 2, pass 2: audit, harden, strengthen (owner request 2026-11-04)

The owner asked for three things, in order: (A) an audit of the three data
structures added in step 2; (B) targeted tests and microbenchmarks for each;
(C) one measured strengthening change per commit against `c8f9083`.
Changes that do not help are reverted, and their numbers are kept here.

### 12.1 Audit of the step-2 structures — [MENOS], read from the diffs

| | (d) non-zero-span `common()` | (b) vertex-choice heap | (c) Schmidt-rank cache |
|---|---|---|---|
| Commit | `0a5ec7c` | `bba745c` | `c8f9083` |
| File | `src/analysis/treewidth.cpp`, `BitAdjacency` (anonymous namespace) | same file, `estimate_treewidth_min_fill_dense` | `src/analysis/schmidt_rank_bound.cpp`, `compute_schmidt_rank_bound` |
| Structure | One `uint64_t` bitset row per vertex; `WordRange{first, last}`, the span of v's non-zero words, computed once per `compute_fill(v)`; `common(a, v, range)` popcounts `row(a) & row(v)` over that span only; `compute_fill` carries `target_clones("popcnt", "default")` | `std::priority_queue<std::pair<size_t, size_t>, std::vector<…>, std::greater<>>` of (fill, vertex), lazy deletion | `std::unordered_map<std::string, std::uint64_t>`, a local of one `compute_schmidt_rank_bound` call, plus one reused `std::string` key buffer |
| Replaced | A popcount over every word of the row | A linear scan of every remaining vertex on each elimination step | One SVD (`zgesdd`) per (operation, spanned cut) |
| Order / key | The same integer as the full popcount: the words outside v's span are zero | `pair` compares fill, then vertex, so ties go to the lowest index, exactly as the scan's strict `<` | `char(k)`, then `char(p)` for each left position p, then `'|'`, then the matrix's raw bytes |
| Stale entries | — | A popped entry is skipped if its vertex is eliminated or its fill is no longer the current one. An entry is pushed only when a vertex's fill changes, so every remaining vertex always has one current entry | None: the cache dies with the call |
| Empty row | `nonzero_range` returns `{0, 0}`, so no word is read; `k*(k-1)/2` with k = 0 is 0. No UB | — | — |
| Size bound | Fixed: n·⌈n/64⌉ words, as before | n + the number of fill changes over the whole search (no decrease-key) | One entry per distinct (matrix, split); each key holds a full copy of the matrix. No cap, no eviction |
| Thread safety | Every structure is function-local. `plan_job` shares no mutable state across calls here, so concurrent planning cannot race on them, and TSan adds nothing for these three | (same) | (same) |
| Invalidation | — | — | Not needed: the cache is built and dropped within one call over one operation list |

**Risks found:**
1. **(b) heap growth [MENOS].** Without decrease-key, the heap holds up to
   n + Σ(fill changes) entries of 16 bytes. Each step can change the fill of
   every vertex within distance 2 of the eliminated one, so the worst case
   is O(n²). That is hundreds of MB near the dense path's n ≈ 23000 limit.
   It is not counted in any memory estimate. 12.2 measures the real peak
   per corpus job.
2. **(c) unbounded memory [MENOS].** Each distinct key copies the whole
   16·4^k-byte matrix. A wide `UnitaryOp` spanning k − 1 cuts with distinct
   splits keeps k − 1 copies until the call returns: for k = 12, about
   2.8 GB. The SVD workspace for such a matrix is transient; the cache's
   copies accumulate across operations.
3. **(c) key encoding [MENOS].** The key is injective only while every
   position byte stays below `'|'` (124) and k ≤ 127 fits the `char` cast.
   Both always hold in practice, since a matrix with k ≥ 124 cannot be
   allocated, but nothing checks it. The fix is fixed-width fields.
4. **(c) collisions and ±0 [MENOS].** `std::unordered_map` compares whole
   keys, so a hash collision costs time, never correctness. +0.0 and −0.0
   give distinct keys: correct, with less sharing. A NaN's bytes are
   deterministic, so equal bytes still mean equal inputs to the SVD.

No correctness defect was found. The two memory risks are fixed in C.

### 12.2 Hardening: targeted tests, microbenchmarks, measured sizes — [MEASURED]

**Internal headers.** The three structures moved out of anonymous namespaces
into `src/analysis/` headers in namespace `menos::analysis::detail`, so they
can be tested alone. They are not public API, and the code is unchanged:
- `bit_adjacency.h`: `BitAdjacency`, `compute_fill`;
- `min_fill_queue.h`: `MinFillQueue`, the lazy heap of (b) behind
  `set_fill` / `pop_min`;
- `schmidt_rank_internal.h`: `operator_schmidt_rank`, `SchmidtRankCache`
  with hit/miss/byte counters.

Golden 45/45 and the 2000-seed differential test pass on the extraction.

**Targeted tests** (`tests/analysis/`):
- `bit_adjacency_test.cpp`: `common`, `nonzero_range`,
  `for_each_neighbor_in` and `compute_fill` against a plain
  `std::vector<bool>` oracle. Covers n ∈ {1, 2, 63, 64, 65, 127, 128, 129,
  200, 1000}, densities 0, 1%, 10%, 50% and 100%, bits on both sides of
  each word boundary, single-bit rows and emptied rows.
- `min_fill_queue_test.cpp`: the pick sequence against the strict-`<`
  linear scan. 500 random runs, most with fills in {0..3}, so nearly every
  pick is a tie. Also covers a fill changing and changing back (duplicate
  live entries) and an empty queue.
- `schmidt_rank_cache_test.cpp`:
  - a hit equals a fresh `operator_schmidt_rank` for every split of CX, CZ,
    SWAP, CP, RZZ, CCX and random dense and product matrices for k = 2..4;
  - known ranks (CX 2, SWAP 4, product 1);
  - distinct splits are distinct entries; separately built copies share
    one; ±0.0 are distinct keys with the same rank;
  - k = 2 and k = 3 keys do not collide.

**Mutation checks.** Each one was applied by hand, built, run, and then
reverted:

| Mutation | Result |
|---|---|
| (b) ties go to the highest index | 4 of 5 `MinFillQueue` tests fail |
| (d) `common` skips the first word of the span | 3 of 4 `BitAdjacency` tests fail |
| (c) the key omits the left positions | 2 of 6 `SchmidtRankCache` tests fail (hit ≠ recompute; splits merged) |

**Sanitizers.** In the dev preset (ASan + UBSan), the new tests, the
treewidth, Schmidt-rank, golden and differential tests pass: 44 of 44.
TSan was not run. All three structures are locals of one function call, and
`plan_job` shares none of them between threads (12.1).

**Microbenchmarks** (`benchmarks/planner_structures_benchmark.cpp`), pinned,
c8f9083-equivalent code:

| Benchmark | Time |
|---|---|
| `compute_fill` over all vertices, n = 256: 1% / 10% / 50% density | 4.6 / 25.6 / 103 µs |
| same, n = 2048 | 1.54 / 11.9 / 45.7 ms |
| dense min-fill, brickwork 16×8 / 24×12 / 32×16 (528 / 1176 / 2080 nodes) | 0.68 / 3.41 / 9.32 ms |
| dense min-fill, random pairs 24×6 / 40×8 (624 / 1360 nodes) | 2.21 / 15.4 ms |
| rank: SVD alone / cache miss / cache hit, CX | 2635 / 2782 / 25.6 ns |
| same, SWAP | 4061 / 4374 / 27.1 ns |
| same, CCX | 5118 / 5548 / 36.8 ns |

**Measured sizes per corpus job.** These come from temporary
instrumentation, not committed; the script is in the session scratchpad.
- **(b) heap.** The peak is at most **2.64 × n** entries: 17,737 entries
  for 7,266 nodes, about 277 KiB, on expectation_factorized/large. The
  O(n²) worst case of 12.1, risk 1, does not occur in the corpus. It
  remains possible in principle.
- **(c) cache.** The cache peaks at **1.0 KiB** (counts_random_pairs). The
  hit rate is 81–100% on every job that computes a rank: for example
  counts_clifford/large has 10,299 hits and 1 miss, and
  expectation_factorized/large has 913 hits and 19 misses. Risk 2 of 12.1
  needs a wide `UnitaryOp`, which the corpus does not contain.

### 12.3 Strengthening, one change per commit — [MEASURED]

**Method.**
- The machine was busier than in step 1: a qstudio job used about 7 cores,
  with a load average of 6–9 on 16 hardware threads, and every core was
  16–63% busy.
- Runs are pinned to CPU 13 (the least-loaded core pair) and compare CPU
  time, which excludes time spent descheduled.
- Runs alternate before/after over 3 rounds × 5 repetitions. Each table
  gives the pooled median ratio and the fastest-repetition ratio.
- "Before" is the previous commit's binary.
- Raw files: `bench/raw/step2_pass2/`.

#### C.1a (d): each row's non-zero word span is maintained, not scanned — KEPT

`nonzero_range` scanned the row from both ends on every `compute_fill`.
In the re-profile at 46b2d93 it took 21% of expectation_factorized/large
(12.1). `BitAdjacency` now keeps each row's exact span:
- `set` widens it;
- a `reset` that empties the word at either end narrows it past every
  empty word there;
- `clear_row` empties it.

`degree`, `for_each_neighbor` and `clear_row` also walk the span only. The
span is exact, so every popcount and neighbour visit is the same as before.

A new test interleaves 20,000 sets, resets and clears per size and checks
the span and degree after each one. Dropping the narrowing makes it fail.

| Job | 46b2d93 | C.1a | Median | Fastest |
|---|---|---|---|---|
| expectation_factorized/large | 5303 ms | 3100 ms | 1.71x | 1.64x |
| expectation_ising_budgeted/large | 1027 ms | 794 ms | 1.29x | 1.27x |
| expectation_z_all/large | 45.8 ms | 34.9 ms | 1.31x | 1.19x |
| expectation_ising/large | 46.2 ms | 38.1 ms | 1.21x | 1.25x |
| expectation_cut/large | 17.4 ms | 15.1 ms | 1.16x | 1.14x |
| expectation_factorized/small | 1.75 ms | 1.94 ms | 0.90x | 1.02x |

The other jobs in the run were 1.02–1.22x. None was slower than its limit:
factorized/small's spread was 28%, so its limit is 56%.

C.1a equivalence: golden 45/45; differential 2000 seeds plus a 50,000-seed
sweep (210000–259999), all identical.

#### C.1b (d): bit tests instead of popcounts for rows sparser than their span — KEPT (third variant)

When v has fewer neighbours k than words in its span, counting
|N(a) ∩ N(v)| by k bit tests per neighbour is cheaper than one popcount per
span word. Both methods count the same integer.

Three variants were measured, each against C.1a:

| Variant | random_pairs_controlled/large | Other jobs | Gate jobs (budgeted/large, factorized/large) | Verdict |
|---|---|---|---|---|
| 1: `degree()` pass to decide, then the chosen path | 2.18x (fastest 2.57x) | 0.91–1.12x, mostly 0.92–0.94x | 0.91x, 1.00x | rejected: gate job slower |
| 2: always collect the neighbours into a vector, take k from it | 2.25x (2.42x) | 0.89–0.97x | 0.97x, 0.95x | rejected: every dense job slower |
| 3: `BitAdjacency` keeps each row's degree as a counter (`set`, `reset` and `clear_row` update it); dense rows run exactly C.1a's loop | **2.05x (2.47x)** | 0.96–1.03x | 1.01x, 1.01x | **kept** |

Variant 3 also gives random_pairs_controlled/medium 1.19x (1.28x).
- The branch is a direct cost comparison (k bit tests against `span`
  popcounts), not a tuned threshold.
- A new test checks both paths against the plain oracle on graphs of
  density 0.2–30%.
- The interleaved test checks the degree counter after every operation.

#### C.1c (d): AVX2 popcount with runtime dispatch — KEPT

On dense rows with a span of at least one vector (4 words),
`compute_fill` counts `popcount(a & v)` four words at a time:
- each byte's count is looked up from its two nibbles (`vpshufb`);
- the byte counts are summed into 64-bit lanes (`vpsadbw`);
- the tail is scalar.

This is the standard nibble-lookup method, written here without copied
code. It is exact integer arithmetic, so the counts are the same.

Dispatch and portability:
- x86-64 Linux with GCC or Clang only. `__builtin_cpu_supports("avx2")`,
  after `__builtin_cpu_init`, is evaluated once.
- Any other platform or CPU, including MSVC, keeps the portable
  `std::popcount` loop.
- This CPU has AVX2 but no AVX-512, so `vpopcntq` was not an option.

The oracle tests at n = 640 and 1000 with 5–30% density take this path.

| Job | C.1b | C.1c | Median | Fastest |
|---|---|---|---|---|
| expectation_ising_budgeted/large | 731 ms | 669 ms | 1.09x | 1.11x |
| expectation_ising_budgeted/medium | 172 ms | 149 ms | 1.16x | 1.14x |
| expectation_z_all/large | 31.5 ms | 27.9 ms | 1.13x | 1.09x |
| expectation_ising/large | 30.3 ms | 27.8 ms | 1.09x | 1.12x |
| expectation_cut/large | 13.5 ms | 11.8 ms | 1.14x | 1.16x |
| expectation_factorized/large | 2876 ms | 2827 ms | 1.02x | 1.02x |

The other jobs were 0.96–1.04x; none was slower than its limit.

C.1b equivalence: differential 50,000-seed sweep (260000–309999), identical.

#### C.2a (b): indexed heap with packed integer keys — KEPT

`MinFillQueue` is now an indexed binary min-heap:
- **Size.** One entry per remaining vertex, plus a position table, so
  `set_fill` moves a vertex's entry up or down in place. The heap never
  holds more than n entries, which removes risk 1 of 12.1. The lazy heap
  peaked at 2.64 n.
- **Key.** Each entry is one integer, `(fill << 32) | vertex`. Integer order
  is fill, then vertex: the scan's tie-break, compared in one instruction.
- **Bounds.** The constructor checks that n and C(n − 1, 2), the largest
  possible fill, both fit in 32 bits. The dense path's size limit keeps
  them far below.

Mutation check: reversing the vertex half of the key, so ties go to the
highest index, fails 6 tests.

| Job | C.1c | C.2a | Median | Fastest |
|---|---|---|---|---|
| expectation_factorized/large | 2781 ms | 2623 ms | 1.06x | 1.07x |
| expectation_factorized/medium | 159 ms | 142 ms | 1.12x | 1.09x |
| expectation_ising_budgeted/large | 654 ms | 645 ms | 1.01x | 1.02x |
| expectation_ising_budgeted/medium | 146 ms | 135 ms | 1.08x | 1.12x |
| expectation_ising/large | 26.4 ms | 24.6 ms | 1.07x | 1.10x |

The other jobs were 0.90–1.15x on medians and 1.02–1.15x on fastest
repetitions; none was slower than its limit.

#### C.2b (b): exact incremental fill update — KEPT

After eliminating v, the search recomputed `compute_fill` for every
vertex within two steps of v. Most of those fills cannot change.
fill(u) counts the non-adjacent pairs among u's neighbours, which gives
two cases [DERIVED]:
- **u a neighbour of v.** u loses v and gains the other neighbours of v,
  so its fill is recomputed in full, as before.
- **Any other remaining u.** u was not adjacent to v, so it keeps exactly
  its neighbourhood: every new fill-in edge joins two neighbours of v, and
  the only removed edges are v's own. So the only change among u's
  neighbours is the new edges that have both ends adjacent to u. fill(u)
  drops by exactly that count.

So the search now:
1. records the new edges as it adds them (it already tested each pair);
2. for each new edge (a, b), walks the common neighbours of a and b, using
   `BitAdjacency::for_each_common_neighbor` over the overlap of the two
   spans;
3. subtracts each such vertex's count from its fill;
4. recomputes only v's neighbours in full.

When fill(v) is 0, which is common under min-fill, no vertex outside N(v)
is touched at all.

Every step is integer counting, so every fill is the integer a full
recompute gives. The sparse search still recomputes fills from scratch and
scans every vertex, and it serves as the oracle:
- the existing test covers 200 random graphs of n ≤ 90;
- a new test covers 16 random graphs of n = 70–250 (2–4 words per row) and
  12 circuit line graphs.

Mutation check: dropping the outside-N(v) update fails both tests.

| Job | C.2a | C.2b | Median | Fastest |
|---|---|---|---|---|
| expectation_ising_budgeted/large | 665 ms | 271 ms | **2.46x** | 2.46x |
| expectation_factorized/large | 2651 ms | 1719 ms | **1.54x** | 1.54x |
| expectation_ising_budgeted/medium | 138 ms | 75.7 ms | 1.83x | 1.86x |
| expectation_factorized/medium | 166 ms | 88.8 ms | 1.87x | 1.77x |
| expectation_random_pairs_controlled/large | 200 ms | 43.9 ms | 4.55x | 3.92x |
| expectation_z_all/large | 27.2 ms | 11.5 ms | 2.36x | 2.26x |
| expectation_ising/large | 28.0 ms | 12.3 ms | 2.28x | 2.17x |
| expectation_cut/large | 10.7 ms | 6.87 ms | 1.56x | 1.79x |

Every job in the run is faster: 1.13–4.55x on medians.

C.1c and C.2a equivalence: one differential 50,000-seed sweep at C.2a
(310000–359999), covering both: identical. C.1c had no sweep of its own.

#### C.3 (c): fixed-width cache key, a byte cap, and counters — KEPT (safety; speed neutral)

Three changes:
- **Key.** Four-byte fields: k, the number of left operands, each left
  position. Then the matrix bytes. Every field has a fixed width and k fixes
  the matrix length, so two different (k, split, matrix) triples can never
  give the same key. This removes risk 3 of 12.1. A field that does not fit
  32 bits is refused with `InternalError`.
- **Cap.** At most 16 MiB of stored key bytes (`kDefaultCapBytes`; the
  constructor takes a cap for tests).
  - An insertion that would cross the cap first clears the cache. This is
    deterministic, and correct because a hit equals a recompute.
  - A single key larger than the cap (a k ≥ 10 `UnitaryOp`) is computed
    and not stored.
  - This removes risk 2 of 12.1. The cap does not change any result, only
    how often the SVD runs, so it is not a tuning constant of the planner.
- **Counters.** `SchmidtRankCacheStats` counts hits, misses, evictions,
  uncached keys and peak bytes.
  - A new CMake option, `MENOS_PLANNER_COUNTERS` (OFF in every preset, so
    timed and shipped builds never carry it), compiles a thread-local total
    into `menos_analysis` (`src/analysis/planner_counters.h`).
  - `menos_planner_benchmark` then reports them per job.

New tests:
- splits that differ only in their last position are distinct;
- a reached cap evicts, and every rank after the eviction is still exact;
- an over-cap key is computed and not stored.

**Hit rate per corpus job** (counters build,
`bench/raw/step2_pass2/c3_rank_cache_counters.json`). The cache is earning
its memory: 81–100% hits on every job that computes a rank, a peak of at
most 1,072 bytes, and no eviction or uncached key anywhere in the corpus.

| Job | Hits | Misses | Hit rate | Evictions | Uncached | Peak cache bytes |
|---|---|---|---|---|---|---|
| counts_brickwork/small | 15 | 1 | 93.8% | 0 | 0 | 268 |
| counts_brickwork/medium | 63 | 1 | 98.4% | 0 | 0 | 268 |
| counts_brickwork/large | 143 | 1 | 99.3% | 0 | 0 | 268 |
| expectation_z_all/small | 30 | 2 | 93.8% | 0 | 0 | 268 |
| expectation_z_all/medium | 126 | 2 | 98.4% | 0 | 0 | 268 |
| expectation_z_all/large | 286 | 2 | 99.3% | 0 | 0 | 268 |
| expectation_ising/small | 30 | 2 | 93.8% | 0 | 0 | 268 |
| expectation_ising/medium | 126 | 2 | 98.4% | 0 | 0 | 268 |
| expectation_ising/large | 286 | 2 | 99.3% | 0 | 0 | 268 |
| expectation_heisenberg/small | 30 | 2 | 93.8% | 0 | 0 | 268 |
| expectation_heisenberg/medium | 126 | 2 | 98.4% | 0 | 0 | 268 |
| expectation_heisenberg/large | 286 | 2 | 99.3% | 0 | 0 | 268 |
| expectation_random_paulis/small | 30 | 2 | 93.8% | 0 | 0 | 268 |
| expectation_random_paulis/medium | 126 | 2 | 98.4% | 0 | 0 | 268 |
| expectation_random_paulis/large | 286 | 2 | 99.3% | 0 | 0 | 268 |
| expectation_ising_budgeted/small | 240 | 16 | 93.8% | 0 | 0 | 268 |
| expectation_ising_budgeted/medium | 2016 | 32 | 98.4% | 0 | 0 | 268 |
| expectation_ising_budgeted/large | 6864 | 48 | 99.3% | 0 | 0 | 268 |
| counts_clifford/small | 54 | 1 | 98.2% | 0 | 0 | 268 |
| counts_clifford/medium | 879 | 1 | 99.9% | 0 | 0 | 268 |
| counts_clifford/large | 10299 | 1 | 100.0% | 0 | 0 | 268 |
| counts_random_pairs/small | 32 | 4 | 88.9% | 0 | 0 | 1072 |
| counts_random_pairs/medium | 252 | 4 | 98.4% | 0 | 0 | 1072 |
| counts_random_pairs/large | 1020 | 4 | 99.6% | 0 | 0 | 1072 |
| expectation_random_pairs_controlled/small | 32 | 4 | 88.9% | 0 | 0 | 1072 |
| expectation_random_pairs_controlled/medium | 252 | 4 | 98.4% | 0 | 0 | 1072 |
| expectation_random_pairs_controlled/large | 1020 | 4 | 99.6% | 0 | 0 | 1072 |
| counts_factorized/small | 23 | 5 | 82.1% | 0 | 0 | 268 |
| counts_factorized/medium | 181 | 9 | 95.3% | 0 | 0 | 268 |
| counts_factorized/large | 883 | 17 | 98.1% | 0 | 0 | 268 |
| expectation_factorized/small | 26 | 6 | 81.2% | 0 | 0 | 268 |
| expectation_factorized/medium | 189 | 10 | 95.0% | 0 | 0 | 268 |
| expectation_factorized/large | 913 | 19 | 98.0% | 0 | 0 | 268 |
| expectation_cut/small | 24 | 2 | 92.3% | 0 | 0 | 268 |
| expectation_cut/medium | 108 | 2 | 98.2% | 0 | 0 | 268 |
| expectation_cut/large | 340 | 2 | 99.4% | 0 | 0 | 268 |

**Speed** (vs C.2b, rank-heavy jobs, 3 × 5 repetitions). This run had the
worst noise of the pass (before-side spreads up to 255%). Fastest
repetitions are 0.96–1.04x and medians 0.87–1.18x: neutral within noise.

#### C.2c (b): bucket queue — NOT TRIED

A bucket queue needs small integer keys. Fills range up to C(n − 1, 2),
and C.2b's updates can raise a fill as well as lower it. After C.2b,
`MinFillQueue` takes 9–11% of the gate jobs (C.5 profile), so the possible
gain is small. Not attempted.

#### C.4 (c): exact Clifford Schmidt rank by GF(2) linear algebra — INVESTIGATED, NOT IMPLEMENTED

**Claim [DERIVED, then MEASURED].** Take a Clifford U on k qubits, split
into sides A and B, with symplectic matrix S (U's conjugation action on
Pauli (x, z) vectors). Let S_BA be the block that maps the Paulis on A to
the B coordinates of their images. Then the operator Schmidt rank across
A | B is 2^rank_GF2(S_BA).

Why: U's Pauli expansion has a support that is a coset of a subspace, with
coefficients of equal magnitude. The A ⊗ B coefficient matrix is then a
scaled character table of the quotient, so its rank is a power of two set
by how much of A's Pauli group reaches B.

**Check.** The scratch prototype (`bench/raw/step2_pass2/gf2_rank_prototype.cpp.txt`,
not built in the repository) compares this formula with the planner's own
`operator_schmidt_rank` (the SVD, through `schmidt_rank_internal.h`). It
covers:
- all 11,520 two-qubit Cliffords up to phase, generated by breadth-first
  search over {H, S, CX}. 11,520 is the correct group order. Ranks 1, 2 and
  4 occur for 576, 5,184 and 5,760 of them;
- 2,400 (unitary, split) cases of random 3-qubit Clifford words;
- 5,600 cases of random 4-qubit Clifford words, with ranks 1 to 16.

| Result | Count |
|---|---|
| Cases where 2^rank_GF2(S_BA) ≠ SVD rank | **0 of 19,520** |
| Cases where rank(S_AB) ≠ rank(S_BA) | 0 |
| Smallest ratio of the smallest to largest non-zero Pauli coefficient | 1 − 2·10⁻¹⁶ (flat) |

**Exactness and tolerance.** The GF(2) rank is exact integer arithmetic.
The SVD counts singular values above σ_max · max(m, n) · ε_rank
(`common::numerical_rank`). For a Clifford every non-zero singular value is
equal, which is what "flat" means above, so the smallest kept value sits at
σ_max. The rounding error is about 10⁻¹⁶ relative, and zero values come out
near 10⁻¹⁶ σ_max, both many orders of magnitude from the cutoff. So on
Clifford matrices the two methods cannot disagree because of tolerance.
None of the 19,520 cases did.

**Feasibility.** Feasible: detect Clifford gates (the named gates H, S,
Sdg, X, Y, Z, CX, CZ, SWAP, or a symplectic check of a unitary), build S
from conjugation, and take a GF(2) rank. The values are exact and equal to
today's.

**Recommendation: do not implement now.** The 73% figure from step 1
predates the rank cache (c). Re-profiled at c8f9083-equivalent code,
counts_clifford/large does 1 SVD and 10,299 cache hits, and the SVD is
**1.9%** of its time. The whole rank bound is 12.5%, mostly key building
and matrix generation. GF(2) would save at most about 2%. It would pay only
for circuits with many distinct Clifford matrices, such as Clifford
`UnitaryOp`s, which the corpus does not contain.

#### C.5: allocation — MEASURED, ARENA NOT IMPLEMENTED

Re-profiled at C.3 (`bench/raw/step2_pass2/prof/`), inclusive shares:

| | expectation_factorized/large | expectation_ising_budgeted/large |
|---|---|---|
| `malloc` / `free` | 2.7% / 2.5% | 2.2% / 0.9% |
| `BitAdjacency` constructor (zeroing a fresh n × ⌈n/64⌉-word bitset) | 5.8% | 4.3% |
| min-fill search, total | 84.0% | 81.5% |
| `compute_fill` | 44.3% | 49.2% |
| `MinFillQueue` | 10.8% | 8.6% |
| `describe_engine_choice` (re-costing, section D) | 38.4% | 0% |

Allocator calls are 3–5%, at the threshold set for trying an arena.
A per-plan `std::pmr` arena cannot remove most of that:
- each min-fill call makes a handful of large allocations, so an arena
  scoped to one call allocates just as often;
- the larger visible cost is zeroing a fresh bitset (6.6 MB per call on
  factorized/large, 1,145 calls), which an arena does not avoid.

Removing it means reusing buffers across calls (thread-local or
caller-owned scratch). That is a state-ownership change, and it overlaps
the memory roadmap (Phase 22). Not implemented; listed as an open question.

C.2b and C.3 equivalence: one differential 50,000-seed sweep at C.3
(360000–409999), covering both: identical.

### 12.4 Where pass 2 leaves the gate jobs — [MEASURED], before section E

The head is the C.3 code, 31dd27b; later commits change documentation only.
Pinned to CPU 13, CPU time, 3 rounds × 5 repetitions per side, alternating.

**Gate jobs against the pre-phase planner** (`passC_gate_vs_d89dde3.compare.txt`):

| Job | d89dde3 | Head | Median | Fastest | Gate target |
|---|---|---|---|---|---|
| expectation_ising_budgeted/large | 2129 ms | 300 ms | **7.11x** | 8.00x | ≥ 5x |
| expectation_factorized/large | 15000 ms | 1696 ms | **8.84x** | 9.38x | ≥ 5x |

**Against c8f9083**, the start of this pass (`passC_vs_c8f9083.compare.txt`),
the whole corpus:

| Jobs | Median | Fastest |
|---|---|---|
| expectation_ising_budgeted/large | 5.31x | 3.94x |
| expectation_factorized/large | 4.02x | 2.56x |
| Large expectation jobs (z_all, ising, heisenberg, random_paulis) | 4.6–5.2x | 3.5–3.7x |
| expectation_random_pairs_controlled/large | 13.4x | 8.2x |
| expectation_cut/large | 2.36x | 2.73x |
| Counts, statevector and noisy jobs, which do no min-fill search | 0.87–2.1x | 0.96–1.14x |

No job is slower than its limit, max(5%, 2 × spread).

This run was very noisy: before-side spreads were 52–407%, and the medians
of the c8f9083 side are 1.3–2x higher than in quieter runs. The ratios
against c8f9083 are therefore indicative. The d89dde3 gate run had spreads
of 23–36%. Section E's gate run should repeat it with 30 or more
repetitions, on a quiet machine if one is confirmed.

Equivalence at every kept commit: golden 45/45 and the 2000-seed
differential test. The 50,000-seed sweeps, recorded above, cover
seeds 210000–409999 with no differences.

## 13. Section D: pruning order, budgeted pruning, re-costing, cache hits (owner request 2026-11-04) — [MEASURED]

The owner approved the four section 11 findings for implementation on
2026-11-04, with one relaxation of invariant 2: **explain() may change only
where it reflects different pruning**, each golden update in its own commit.
The chosen plan and its cost stay identical. Each change below was measured
against the commit before it: pinned to CPU 13, CPU time, 3 rounds × 5
repetitions per side, alternating (`bench/raw/step2_pass2/sectionD/`).

### D.0 What "identical up to pruning" means (`tests/plan/prune_equivalence.h`)

The golden test still compares byte for byte, against re-recorded files. The
differential test compares the live planner with the frozen d89dde3 planner,
and the frozen planner with every golden file, under this rule:
- the two outcomes have the same lines in the same order;
- a differing **candidate** line must be pruned in the live plan, must not
  be the reference winner, and, if the reference costed it as feasible, its
  work units must be strictly above the winner's (a prune needs a cheaper
  feasible bound; ties go to the earlier candidate);
- a differing **region** line may differ only in engines the live plan lists
  as pruned: such an engine may be absent from the reference (feasible but
  costlier there) or carry any reason;
- everything else (the strategy, the estimate, every region's fields, every
  refusal) is byte-identical.

`prune_equivalence_test.cpp` checks the rule accepts and rejects what it
should (11 cases).

### D.1a The tensor network is costed after the cheaper candidates — KEPT

`direct_tensor_network` is still listed after `direct_mps`. It is now
costed after `direct_density_matrix`, `direct_trajectory` and `factorized`,
and then moved into its listed place, so its min-fill search gets a bound
from them (section 11.1).

| Job | Before (C.3 code) | After | Median | Fastest |
|---|---|---|---|---|
| expectation_factorized/medium | 76.7 ms | 36.8 ms | 2.08x | 2.09x |
| expectation_factorized/large | 1357 ms | 648 ms | 2.09x | 2.09x |
| expectation_ising_budgeted, expectation_cut, factorized/small | | | 0.99–1.01x | |

Golden change (`cce76b0`): the tensor-network candidate of
expectation_factorized/{small,medium,large} now carries the pruned reason
(small: it was already pruned; only the reason's figures change).

### D.1b Branch-and-bound stays on under a time budget, CPU-only — KEPT

Section 11.2: in a job where neither the CUDA dense-backend step nor
Factorized's per-region backends can apply (`backend` is not `cuda`, and not
CUDA-and-CPU-both-calibrated with `backend` unset), each costed candidate is
held to the time budget before it may serve as the bound, so the bound is
the fewest work units among candidates feasible in memory and time. The
budget check is idempotent, so it still runs once more for every candidate
after enumeration, exactly as before. CUDA-capable jobs still cost every
candidate in full.

| Job | Before (D.1a) | After | Median | Fastest |
|---|---|---|---|---|
| expectation_ising_budgeted/small | 2.95 ms | 0.161 ms | 18.3x | 18.1x |
| expectation_ising_budgeted/medium | 65.9 ms | 2.07 ms | 31.9x | 31.9x |
| expectation_ising_budgeted/large | 228 ms | 10.3 ms | 22.1x | 22.1x |
| expectation_factorized, expectation_cut | | | 0.99–1.00x | |

On the large job the tensor network still ends on its memory-budget reason:
its first term's elimination search hits the budget before the per-term
bound check is reached. The saving is the cut-and-recombine costing, now
pruned in fragment 1 of 2.

Golden change (`297bf24`): expectation_ising_budgeted/{small,medium,large}.
`cut_and_recombine` (all three) and `direct_tensor_network` (small, medium)
print the pruned reason in place of their figures and `est_seconds`.

Tests changed with it: `WithATimeBudgetTheTensorNetworkIsCostedInFull`
became `...IsStillPruned`; new `ACandidateOverTheTimeBudgetIsNotTheBound`;
the observed-floor test also covers pruned candidates.

### D.2 A pruned engine in a region's "why" is reported as pruned — KEPT

`describe_engine_choice` re-costed every engine that `pick_cheapest_engine`
had pruned, without the bound, to print its unbounded reason (section 10.3).
That was 38% of expectation_factorized/large after pass 2. It now prints
the engine's own pruned reason.

| Job | Before (D.1b) | After | Median | Fastest |
|---|---|---|---|---|
| expectation_factorized/small | 0.915 ms | 0.136 ms | 6.75x | 6.79x |
| expectation_factorized/medium | 37.5 ms | 0.997 ms | 37.6x | 38.3x |
| expectation_factorized/large | 658 ms | 4.47 ms | 147x | 149x |
| counts_factorized, expectation_cut | | | 0.96–1.02x | (spread 4–9%) |

Golden change (`444c4f7`): each Factorized region of
expectation_factorized/{small,medium,large} also lists
`(tensor_network rejected: pruned by branch-and-bound: ...)`.

### D.3 Plan-cache hit: lazy error message, light cone without copies — KEPT

Section 11.3. `reparameterize_direct_plan` built its "operation i differs"
message for every operation of every hit; it is now built only on failure.
`compute_light_cone` copied each operation's qubit list on its backward
walk; it now reads the list in place, which also speeds every plan. The
copies of the cached plan itself are left to Phase 22 (owner). No golden
change.

| Benchmark | Before (D.2) | After | Median | Fastest |
|---|---|---|---|---|
| cache_hit/statevector_brickwork/medium | 108 µs | 86.5 µs | 1.25x | 1.28x |
| cache_hit/counts_brickwork/large | 330 µs | 260 µs | 1.27x | 1.30x |
| cache_hit/expectation_ising/medium | 114 µs | 91.0 µs | 1.26x | 1.26x |
| plan_job statevector_brickwork, counts_brickwork, counts_clifford | | | 1.04–1.07x | |

A hit on statevector_brickwork/medium (86.5 µs) is now cheaper than planning
it (94.5 µs), which section 11.3 found reversed.

### D equivalence

- Golden 45/45 against the re-recorded files at every code commit's golden
  commit; the frozen d89dde3 planner reproduces all 45 up to pruning.
- Differential, 2000 seeds, at every commit: nothing beyond pruning (151
  seeds differ only by pruning after D.1a, 238 after D.2).
- Planner, plan-cache, light-cone and analysis tests under ASan+UBSan (dev):
  102/102.
- 50,000-seed sweep after D.3 (seeds 410000–459999, Release): nothing beyond
  pruning (`sectionD/sweep_410000_459999.txt`).
