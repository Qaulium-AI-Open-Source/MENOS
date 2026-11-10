# ADR 0028: Phase 24 — Planner performance

## Status

**Renumbered at the merge (2026-11-05).** This ADR was written on a
contributor's fork as "ADR 0025, Phase 21 — Planner performance", with the
memory roadmap renumbered there to Phases 22–24. On this repository's own
line the memory roadmap kept its numbers (Phases 21–23, ADRs 0025–0027) and
passed its gates while the fork's work was done, and the fork's work matches
items 2 and 5 of the Phase 24 proposal (`docs/BUILD_PLAN.md`, "Planner and
dense-engine roadmap"). When the two lines were merged on 2026-11-05 the
fork's work became the start of Phase 24 and this ADR became 0028. Its
sections 1–14 are the fork's record, kept as written apart from the
renumbering: "Phase 21" in them is this phase, "Phases 22–24" are the memory
phases 21–23, and "the owner" is the fork's owner. Section 15 is the merge:
what was kept, what was decided and what was retired, with its verification.

**Fork status (2026-11-04), step 2 in progress.** Section 9 is answered (items 2–6
on 2026-11-03, item 1 and the step-2 order after the step-1 report the same
day). The fork started the phase on 2026-11-03. Its scope is `plan_job`
latency plus the one-time hardware calibration. Step 1 added the benchmark, the fixed
corpus, the golden plan files and `bench/BASELINE.md`. perf was unavailable
(`kernel.perf_event_paranoid=4`), so the profile was taken with a gdb sampling
profiler (`bench/tools/`). Step 2 results are in section 10.

Labels follow `docs/adr/0014`–`0024`: **[MEASURED]**, **[DERIVED]**,
**[MENOS]** (a proposed design decision) and **[OPEN]** (an owner decision,
section 9). A claim with no measurement in this phase is **[HYPOTHESIS]**.

---

## 1. Context

The owner asked for planning to be made much faster, and for a release to
follow. It is done in steps 0–6 (`docs/BUILD_PLAN.md`, Phase 24), each
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

## 14. Section E: the gate run — [MEASURED]

**Conditions.** No quiet machine was available (owner, 2026-11-04). CPU
governor `performance` (already set). Pinned to CPU 13, CPU time, 6 rounds
× 5 repetitions per side (30 per side), before and after alternating each
round, the whole corpus plus the cache-hit benchmarks. Load average (1 min)
logged before every round: 1.26–4.33, mean 2.63; one long-running
single-threaded test process of another project was running throughout and
was left alone. Head binary: the D.3 code (`300fcb3`); every later commit
changes documentation or evidence only. Raw runs, compare tables and load
logs: `bench/raw/step2_pass2/sectionE/`.

**Gate criterion 1, median speed-up ≥ 5x — MET.**

| Job | d89dde3 | Head | Median | Worst case | Spread before / after |
|---|---|---|---|---|---|
| expectation_ising_budgeted/large | 1992 ms | 10.8 ms | **184.6x** | 159.8x | 24% / 20% |
| expectation_factorized/large | 14041 ms | 4.44 ms | **3164x** | 2605x | 18% / 19% |

"Worst case" is the fastest d89dde3 repetition over the slowest head
repetition: even that is far above 5x, so the spread cannot put the result
in doubt. Against c8f9083 (the start of pass 2): 73.0x and 837x median,
57.6x and 668x worst case. No cost-model change was needed.

**Gate criterion 2, no corpus job slower than baseline by more than
max(5%, 2 × spread) — MET.** Every one of the 45 jobs and the 3 cache-hit
benchmarks is faster at the median than d89dde3; the smallest ratio is
statevector_brickwork/small, 1.07x. Against c8f9083 the smallest is
counts_factorized/small, 1.02x. Per-job spread ranges 4–53% (before side)
and is listed per job in `vs_d89dde3.compare.txt`.

| Family (vs d89dde3, median) | Small | Medium | Large |
|---|---|---|---|
| statevector_brickwork | 1.07x | 1.08x | 1.07x |
| counts_brickwork | 2.56x | 2.63x | 2.33x |
| expectation_z_all / ising / heisenberg / random_paulis | 2.05–2.12x | 4.04–4.07x | 7.39–7.97x |
| expectation_ising_budgeted | 37.8x | 139x | 185x |
| counts_clifford | 3.81x | 4.22x | 3.17x |
| counts_random_pairs | 2.78x | 6.71x | 7.53x |
| expectation_random_pairs_controlled | 2.22x | 5.38x | 13.0x |
| counts_factorized | 2.31x | 3.43x | 3.96x |
| expectation_factorized | 11.4x | 383x | 3164x |
| expectation_cut | 2.01x | 3.21x | 6.13x |
| counts_noisy_density_matrix / trajectory | 1.10x / 1.10x | 1.11x / 1.08x | 1.09x / 1.10x |
| cache_hit (3 benchmarks) | | 1.32–1.35x | |

**Gate criteria 3 and 4, golden and differential — MET** on the final head:
golden 45/45 (re-recorded only in the D golden commits, each
reproduced by the frozen d89dde3 planner up to pruning); differential 2000
seeds per build in `scripts/verify.sh`, and the 50,000-seed sweeps of
sections 12 and 13 (seeds 210000–459999), nothing beyond pruning.

Invariant 2 was relaxed by the owner for section D only: `explain()`
differs from d89dde3 in six golden files, in losing candidate lines and
region reasons that are now reported as pruned (section 13).

---

## 15. The merge with the memory roadmap (2026-11-05) — [MENOS] / [MEASURED]

The owner asked for every branch to be merged into `main`, choosing the best
resolution where the two lines collided. `main` held this phase's work (pull
requests #12 and #13); branch `phase-23-tiered-memory` held the completed
memory roadmap, Phases 21–23 (`docs/adr/0025`–`0027`), from `main` at
`1f89192`. The merge was made on branch `integrate-memory-roadmap` from
`main`, merging `phase-23-tiered-memory`; 25 files conflicted. The rules
applied, in order: a frozen, gate-passed invariant of either line is kept;
where both lines added a mechanism, both are kept and made to compose; where
the two cannot both hold, correctness (bitwise results, exact memory upper
bounds) wins over speed, and the loser is recorded here as a deferral.

### 15.1 Numbering

Phases 21–23 keep their numbers, ADRs and gate records: they were closed by the
owner with evidence, and their ADR numbers appear in those records and in
the code. This phase's work, open on the fork, is renumbered to Phase 24,
the planner roadmap it belongs to, and this ADR to 0028 (0025–0027 are the
memory ADRs). Every "Phase 21" / "docs/adr/0025" that the fork wrote in
code comments, benchmarks, `bench/`, `docs/ENGINE_TECHNIQUES.md`,
`docs/ARCHITECTURE_NOTES.md` and the changelog now reads "Phase 24" /
"docs/adr/0028"; the memory line's references are untouched. The section
numbers cited in code comments are this ADR's own, unchanged.

### 15.2 Kept from both lines, composed

- **The estimate request.** `EngineEstimateRequest` carries the memory
  line's `shots`, `bitstring_width`, `is_sole_region` and tier fields and
  this phase's `prune_above_work_units`, last. `cost::estimate_direct` takes
  `trajectories, shots, prune_above_work_units = max`; `cost::estimate_cut`
  takes the memory line's `circuit` and returns its `CutEstimate` (the
  evaluations' engines), with the prune bound as its last argument, and a
  pruned cut is a `CutEstimate` with no evaluations. Every call site uses the
  memory line's order; the positional initialisers were checked, and none
  reaches past `observable`.
- **The planner.** The disk-tier block (Phase 23) runs right after the dense
  candidate, then the branch-and-bound incumbent is defined; the
  tensor-network candidate is costed after the cheaper candidates with the
  bound and `shots`, and rotated back into its listed place, as the fork had
  it; the forced-disk-tier refusals follow. `reparameterize_direct_plan` and
  `may_reach_disk_tier` both stay.
- **The hardware profile.** `plan::hardware_profile_for(time_budgeted)`
  (section 11.1: an unbudgeted CPU-only job never runs the calibration) and
  the memory line's `current_hardware_profile(scratch_directory,
  measure_scratch)` (`docs/adr/0025` Q3: the device's free memory, and the
  scratch directory, read again for every job) compose through a new overload
  `current_hardware_profile(base, scratch_directory, measure_scratch)`. The
  api layer passes `hardware_profile_for(time_budgeted || measure_scratch)`
  as the base: a job that could reach the disk tier is planned against the
  calibrated profile, because the planner prices the tier's crossings at the
  calibrated rates (`docs/adr/0027` section 3.5) and Phase 23's gate was
  measured that way. The reach test itself reads memory facts only, so it is
  asked of the detected profile and never triggers the calibration.
- **The tensor-network engine.** Pruning inside the term loop (section 13)
  and the memory line's exact per-term peak (`term_peak_bytes`) both stay;
  git merged the loop itself.
- **The treewidth search.** The bitset min-fill search (section 12.3) is the
  one every job runs (the `std::set` search serves graphs above 23 000
  nodes), so it now reports `EliminationOrder::search_peak_bytes` as the
  sparse one does: the bit rows, spans and degrees, the fill counts, the
  queue's three arrays, the result's two lists, the flags and drop counts,
  sized once, plus the four lists grown by `push_back` at the capacity the
  search reached. The search is deterministic, so the search `execute()`
  runs again on the same graph holds exactly that; the memory fixtures
  (`tests/memory`) hold the tensor-network estimate to the heap tracker.
- **The dense engine.** The streaming diagonal kernels (bitwise identical to
  the generic kernel, pinned by `specialised_kernels_test`) and opt-in
  fusion stay; the statevector is still moved into the result last.
- **The region builder** passes the memory line's request shape and collects
  the per-engine estimates for `describe_engine_choice` (section 13, D.2).
- **The runtime request** carries `fuse_diagonal_gates`, `scratch_directory`
  and the tier's chunk and window.
- **Public API, Python, docs.** `plan_cache_stats`/`plan_cache_clear` and
  `release_device_memory`; `RunOptions.fuse_diagonal_gates`, `tier` and
  `scratch_directory`; both lines' documentation pages and changelog entries;
  the site excludes `benchmarks/`, `ARCHITECTURE_NOTES.md` and
  `ENGINE_TECHNIQUES.md`.

### 15.3 Decided at the merge

1. **The plan cache keys on the hardware profile.** The fork's key was the
   job's structure and options, which was complete while the profile was
   fixed per process. Since `docs/adr/0025` Q3 the device's free memory is
   read again for every job, and `hardware_profile_for` gives the detected
   or the calibrated profile, so a cached plan could outlive the memory it
   was planned under. `plan_job_cached` appends every figure `plan_job` reads
   (total memory, the two rates, the five device figures and the three CUDA
   rates) to the key; the key also carries `JobSpec::tier` and the scratch
   directory (a forbidden tier is printed by `explain()`). The scratch
   figures are left out, because an in-memory plan never reads them and
   they change with every write to the disk; a plan on the disk tier is
   never stored instead (`on_disk_tier`). The key format version is 2.
2. **Pruning is off under a forced disk tier.** `plan_job` refuses every
   candidate but the dense one after enumeration, with the reason
   `explain()` prints (Phase 23); a candidate pruned before that step would
   keep the pruned reason instead. The bound stays at its default there, as
   it does for a forced CUDA backend and a calibrated CPU/CUDA comparison.
3. **The frozen `d89dde3` differential test is retired.** Its reference is
   the planner before Phase 21 (memory accounting), whose estimates differ
   from every candidate's by design (`docs/adr/0025` section 10 lists jobs
   whose feasibility changed), so "the same plan up to pruning" against it
   cannot hold and would have to be weakened to "up to memory" to pass. The
   golden corpus (45 fixed jobs, hand-built profiles) is regenerated once
   against the merged planner and pins plans from here on, recorded per
   platform (`tests/plan/golden/planner_corpus/{linux,windows}/`) because
   the memory estimates count standard containers, whose sizes differ
   between libstdc++ and the MSVC STL (`std::set` is 48 bytes against 16):
   the first `scripts/verify.sh` run on `073c22b` failed only there, in
   peak-byte figures of stabilizer and factorized plans, with every
   strategy, winner, reason and work-unit figure the same; `tests/plan/frozen_d89dde3/`, `planner_differential_test.cpp`,
   `planner_differential.h`, `random_planner_jobs.h`, `prune_equivalence.h`
   and its test, and `MENOS_BUILD_FROZEN_PLANNER` are removed. A randomised
   equivalence test needs a reference planner that includes the memory
   phases; none exists, and building one is a deferral for the owner.
4. **The batched Z-term expectation is retired.** The fork's
   `expectation_of_diagonal_terms` summed the Z-only terms of an observable
   through one in-place Walsh–Hadamard transform (butterfly order, within
   tolerance of the per-term kernel, consuming the state). Two frozen
   invariants require the per-term kernel's summation order bit for bit:
   the Phase 21 baseline (`tests/memory/data`, every fixture result bitwise
   the `1f89192` result) and the Phase 23 disk tier (its chunked
   accumulation is bitwise the in-memory engine's). In the merged engine it
   also overwrote a state that `want_statevector` was still to return. The
   dense engine computes every term with `expectation_of_term` as the memory
   line did; the kernel and its test are removed. Re-adding it needs the
   disk tier to apply the same transform in the same order, or an owner
   decision to relax the bitwise invariants, and an ADR.
5. **Windows.** The fork built and verified on Linux only; the golden test's
   `std::getenv` takes the `_CRT_SECURE_NO_WARNINGS` guard the benchmarks
   already use, so `windows-dev`/`windows-release` build with `/WX`.

### 15.4 Verification — [MEASURED]

Development machine: Intel Core Ultra 7 155H, 22 threads, 32 GB, Windows 11;
MSVC 14.50 for the native build, Ubuntu 26.04 in WSL (GCC 15.2, CUDA 13.3.1
toolkit, driver 566.14, so GPU tests skip there) for `scripts/verify.sh`.

- **Merge commit `991ec2e`** (branch `integrate-memory-roadmap`, parents
  `f5341e0` = `main` and `373636c` = `phase-23-tiered-memory`); `fee38b9`
  restores one namespace comment that the format stage rejected (dropped
  by `373636c` on the memory line); `84c5cc7`, `b3d67af`, `dd07c97`,
  `073c22b`, `bbac1ef` and `295de4f` are the fixes the two scripts found,
  each named below; `295de4f` is the commit both scripts were judged on,
  and the documentation closure follows it.
- **During the merge**, `windows-release` was built and its full ctest run
  repeated until clean. The failures on the way, each fixed in the merge
  commit: the frozen-planner and golden tests' `std::getenv` under `/WX`
  (guarded); the golden plans differing from the fork's recordings in
  every candidate's `peak_bytes` (the Phase 21 accounting; regenerated,
  45 files, winners and pruning reasons unchanged) and in line endings
  (read line-ending-neutral); the plan cache missing on an identical job
  because `std::optional<Backend>` (the fork's key) and
  `std::optional<Tier>` (this merge's) were serialised as raw bytes; and
  the `run_mitigated` memory fixture exceeding its estimate by the
  calibration's 64 MiB, because `hardware_profile_for` had moved the
  first calibration into the fixture's tracked window through
  `run_readout_mitigated`'s budget (now the detected profile's figure).
  The first `scripts/verify-windows.ps1` run then failed only on the golden
  test in `windows-dev`: the MSVC debug STL makes `std::vector` 32 bytes
  instead of 24 and the memory estimates count it (257 histogram nodes ×
  8 bytes = the 2 056-byte difference in every counts plan), so on MSVC
  the golden plans are checked in optimised builds only (`84c5cc7`,
  `b3d67af`); the memory fixtures still hold each build's estimate to
  its heap tracker.
  Final subset on the merge commit: plan-cache, golden and memory-fixture
  tests, ctest exit 0; the preceding full run on the same code apart from
  the two optional fixes: 861/863, the two plan-cache tests being those
  fixes.
- **`scripts/verify-windows.ps1` on `295de4f`: exit 0** (2026-11-05, 15:55–16:22):
  `windows-dev` (MSVC ASan) 863/863, `windows-release` 863/863, the benchmarks
  build, pytest 574 passed / 2 deselected on the editable install, the wheel
  built and its smoke test 2 passed in a clean venv. Earlier runs on the way:
  `b3d67af` stopped at the benchmarks build (`malloc_usable_size`, fixed in
  `dd07c97` with `_msize`); `dd07c97` stopped at pytest on the
  `compatibility.md` snippet (`menos.__version__` 1.2.0 against the package's
  1.2.1, one from each line; `073c22b` sets 1.2.1, the release); `073c22b`
  passed every stage.
- **`scripts/verify.sh` (WSL) on `295de4f`:** exit 0 on 2026-11-05, every
  stage: `dev` 862/862, `release` 862/862, the benchmarks build and
  clang-tidy, pytest 574 passed / 2 deselected on the editable install,
  the wheel built and its smoke test 2 passed in a clean venv, `cuda-dev`
  942/942, `cuda-release` 942/942, `cuda12-dev` 942/942, `cuda12-release`
  942/942 (the docs stage skipped: no node in the WSL clone). Earlier:
  `fee38b9`'s run was cut off with the session at clang-tidy; `073c22b`'s
  failed only the golden test in the dev stage (the per-platform recording
  above, `bbac1ef` and `295de4f`), 861/862 otherwise.

Not verified here: the fork's planning-time figures (sections 10–14) were
not re-measured after the merge; the memory accounting adds to every
estimate but not to the search. The CUDA presets' GPU tests skip on this
driver, as in Phases 21–23.

---

## 16. Kickoff: planner accuracy (items 1, 2, 3 and 9) — [MENOS] / [OPEN]

**Started by the owner on 2026-11-05** on branch `phase-24-planner-accuracy`
from `main` at `65e9c6c`. Items 1, 2, 3 and 9 of the `docs/BUILD_PLAN.md`
proposal are in scope:
- item 1, measure the cost model;
- item 2, report and bound planning cost;
- item 3, per-region observations;
- item 9, threaded calibration and the disk link per direction.

Items 4–8 are not started. The process is the usual one. The owner answers
this section's questions (16.6) first. A plan is then approved. Step 1
measures before any code changes. Nothing below is built.

### 16.1 What the code does today — [MENOS], read from the source

- **Cost-model benchmark.**
  `benchmarks/cost_model_correlation_benchmark.cpp` covers the dense engine
  only, on one family: one H layer plus CX on even pairs, n = 10–20.
  - It times the whole public `menos::run`, planning included.
  - It prints a Spearman rank correlation, not actual over predicted.
  - The Phase 11 table came from a different harness that is not in the
    repository. The table is in `docs/adr/0015` lines 25–46, and is repeated
    in `docs/planning/cost-model.md` lines 69–74 and
    `docs/reference/limitations.md` line 50.
  - Stabilizer, density-matrix and trajectory have their own correlation
    programs. MPS, factorized, cut and tensor network have none.
- **MPS work formula.** `MPSEngine::estimate`
  (`src/engines/mps/mps_engine.cpp` lines 97–153) charges every gate's local
  updates at `model.max_bond`³. The bond replay
  (`src/engines/mps/memory_model.cpp`, `Replay`) knows each gate's bonds
  while it runs, but only `max_bond` and `final_bonds` leave it.
- **Planning time** is not measured anywhere.
  - `api::validate_and_plan` (`menos.cpp` line 117) calls
    `plan::plan_job_cached`, which calls `plan_job` at `plan_cache.cpp`
    lines 235 and 250. A cache hit, at line 244, does not plan.
  - The replan controller calls `plan_job` again
    (`replan_controller.cpp` line 172).
  - `run_batch` plans through `run`.
  - `replan_controller.h` lines 120–123 record that planning time is not
    charged against the budget (0.05–85 ms measured).
- **The tensor-network search has only two bounds:** the job's memory
  budget and the branch-and-bound stop predicate (`analysis/treewidth.h`
  lines 185–244).
  - If no feasible cheaper candidate is in hand, the incumbent is "none"
    (`planner.cpp` lines 562–576). The search then runs until it completes or
    exceeds the memory budget.
  - The 4.4 s case in `docs/BUILD_PLAN.md` ("Measured on 2026-11-02") has
    not been re-measured since pruning was merged.
- **Multi-region overruns.**
  - `run_with_replanning` refuses any plan with more than one region
    (`replan_controller.cpp` lines 155–161).
  - Observations are one floor per `(Engine, Backend, Tier)`
    (`runtime_observations.h`), and a floor is only ever raised.
  - The factorized executor (`runtime.cpp` lines 199–412) knows each
    region's engine, backend and tier at its `engine.execute` call
    (line 313).
  - The cut executor (lines 421–548) runs fragment assignments on
    `(engine, Cpu, InMemory)`.
  - Neither executor catches `ReplanRequested`. One timer spans each whole
    loop.
  - The termination argument in `replan_controller.cpp` lines 92–138
    assumes single-region plans.
- **Calibration.** `work_units_per_second` is measured on a 13-qubit,
  2-layer circuit (`hardware_profile.cpp` lines 88–92 and 147–215).
  - At 13 qubits a gate has at most 2^12 groups. The kernels thread only from
    `kParallelMinGroups` = 2^16 groups (`src/kernels/dense_kernels.h`
    line 99), so the calibration runs on one thread.
  - The source comment's "under 1% across 12–16 qubits" is therefore a
    statement about the single-thread rate only.
  - Every dense job above about 17 qubits runs on every hardware thread.
- **Disk-tier pricing.** The disk tier is priced at
  `work / rate + link_bytes / min(write, read)` (`planner.cpp`
  lines 422–445). The write and read bandwidths are already measured
  separately (`hardware_profile.cpp` lines 307–356).

### 16.2 Item 1: measure the cost model — [MENOS], proposed

- **The program.** A new program, `benchmarks/cost_model_accuracy_benchmark.cpp`,
  measures every engine. It does not use the correlation program above.
- **Choosing the engine.** It forces each engine where forcing exists.
  Otherwise it uses a circuit the planner gives to that engine, and checks
  that the planner did.
- **Measured time.** It times `execute_seconds + sample_seconds` from
  `Result`. Planning is excluded.
- **Predicted time.** The plan's `estimated_seconds`, from the calibrated
  profile in the same process.
- **What it prints.** Actual over predicted, as the median of 5 runs after a
  warm-up.
- **Engines covered.** Dense (CPU), MPS, stabilizer, factorized, cut and
  tensor network. The circuit families are set by Q1.
- **Where the output goes.** A table written to `docs/benchmarks/phase-24/`,
  copied into this ADR and into `docs/planning/cost-model.md`, where it
  replaces the Phase 11 table.
- **The MPS formula.** It changes to per-gate bonds only if the measurement
  shows the peak-bond charge is the error (Q2). The change is then measured
  the same way.

### 16.3 Item 2: report and bound planning cost — [MENOS], proposed

- **What `Result::plan_seconds()` reports.** C++ `Result::plan_seconds()`,
  Python `plan_seconds`. It is the wall time of the initial `plan_job_cached`
  call plus every replan's `plan_job` call.
- **How it is measured.** With `steady_clock`, in `validate_and_plan` and in
  the replan controller. A cache hit reports its own (small) time.
- **`explain()`.** It appends `plan_s=` to the `actual` line, after
  `sample_s=`.
- **Tests.** The timing normalisers in `tests/cli/golden_test.cpp`,
  `tests/cli/qasm3_test.cpp` and `tests/api/explain_test.cpp` learn the new
  field. The planner corpus goldens carry no `actual` line and are unchanged.
- **`preview()`.** It does not execute and reports nothing (Q3).
- **A cap on the tensor-network elimination search** is proposed only if
  step 1 shows a case that still plans slowly after pruning.
  - Step 1 re-runs the 20-qubit, depth-10, 50-term case and the planner
    corpus, with `plan_seconds` reported.
  - If a cap is needed, it is a node count derived from the measurement.
  - It is never a time cap. A time cap would make the plan depend on machine
    load and break deterministic plan choice.
  - A capped search refuses the candidate with a named reason. It never
    guesses the candidate's cost (Q4).

### 16.4 Item 3: per-region observations — [MENOS], proposed

- **Timing in the factorized executor.** It times each region's
  `engine.execute`. On `ReplanRequested`, before rethrowing, it records:
  - the elapsed time of every completed region;
  - the elapsed time of the running region up to the crossing.
- **What the controller raises.** It raises the floor of the overrunning
  region's `(Engine, Backend, Tier)` to that region's own elapsed time,
  never to the job's total. The floor stays a sound lower bound for that
  engine on that region.
- **Keying floors by region.**
  - Today a floor is per engine, not per circuit. A region's floor applied to
    another region of a different size is not a sound bound.
  - Proposed: key the floor additionally by the region's qubit set, which
    identifies the region within this job.
  - The planner applies the floor to candidates whose region set contains
    that same region.
  - This keeps both "an observation only raises a figure" and soundness
    (Q5).
- **Termination.** The termination argument (`replan_controller.cpp`
  lines 92–138) is extended, and written out in this ADR before code:
  - each replan raises at least one floor to at least the budget;
  - each floor has one of finitely many keys, (region, engine, backend,
    tier);
  - once its floor reaches the budget, that candidate is infeasible;
  - so the number of replans is bounded by the number of keys.
- **Cut plans.** Fragments run inside a recombination sum, and their elapsed
  time can be attributed per fragment assignment in the same way. They are
  in scope only if Q6 says so. Otherwise a cut overrun stays refused, with
  the message unchanged.
- **Tests.** The existing `ScriptedClock` (`tests/api/replan_test.cpp`)
  drives an injected crossing in a two-region factorized job. The test checks
  the replan, the recorded per-region observation and `explain()`'s replan
  lines.

### 16.5 Item 9: threaded calibration and the disk link per direction — [MENOS], proposed

- **The threaded rate.** `work_units_per_second` is measured on a state
  large enough to be threaded (proposed: 20 qubits, which gives 2^19 groups
  per single-qubit gate; Q7).
  - It uses the job's thread count and is deadline-bounded, as today.
  - The 13-qubit single-thread rate is kept as a second field,
    `work_units_per_second_single_thread`.
- **Which rate prices a candidate.** The planner prices a dense candidate at
  the rate for the thread count the kernel will actually use at that size:
  - one thread below `kParallelMinGroups` groups;
  - the threaded rate above it.

  No interpolation constant is introduced. Both rates are used as measured.
- **Calibration cost.** It rises: a 20-qubit state is 16 MiB, and its
  circuit is timed to the same deadline. The rise is measured. The profile
  stays cached once per process.
- **Disk-tier pricing.**
  - New formula: `work / rate + bytes_written / write + bytes_read / read`.
  - This needs `tier_cost` to carry written and read bytes separately, in
    place of today's `link_bytes`.
  - The chunked pass executor reads and writes each chunk once per pass, so
    the two byte counts are equal per pass. The change matters only when the
    two bandwidths differ.
  - It is re-measured on the Phase 23 gate circuits
    (`docs/benchmarks/phase-23-gate/`) in WSL and on Windows. Actual over
    predicted is reported beside the Phase 23 figures.
- **Plan changes.**
  - A faster dense rate changes `estimated_seconds`, so it can change which
    plans fit a time budget.
  - Ranking stays by work units (item 4 is not in scope), so unbudgeted
    jobs keep their plans.
  - The golden corpus and every `explain()` golden that prints seconds are
    re-checked, and any change is listed.

### 16.6 Owner questions — [OPEN]

- **Q1, circuit families for item 1.** Proposed, by engine:
  - random brickwork (H/T/CX), depth 10 and 20;
  - QFT;
  - a 1D Ising Trotter circuit (low entanglement, the MPS regime);
  - random Clifford (stabilizer);
  - two independent blocks (factorized);
  - a two-block circuit joined by one CX (cut);
  - the planner corpus's expectation jobs (tensor network).

  n is chosen per engine so that one run takes 0.1–10 s.
- **Q2, the factor each time model must meet.** Proposed: set after step 1,
  from the measurement, per engine, as in Phases 15 and 16. The MPS formula
  changes only if its ratio is outside that factor.
- **Q3, `plan_seconds`.** Proposed: include replan planning time in the sum,
  and also print each replan's planning time on its replan line; `preview()`
  reports nothing. Alternative: the initial plan only.
- **Q4, a search cap.** Proposed: only if step 1 shows a remaining slow
  case. It is a node-count cap derived from the measurement, never a time
  cap. A capped search refuses the tensor-network candidate with a named
  reason.
- **Q5, per-region floors keyed by region.** Proposed as in 16.4. The
  alternative, one floor per engine applied to every region, is unsound for
  regions of different size and is not recommended.
- **Q6, cut plans in item 3.** Proposed: factorized only in this phase, and
  cut overruns stay refused. Each cut fragment runs many assignments and the
  recombination is CPU-only, so the gain is smaller.
- **Q7, the threaded calibration size.** Proposed: 20 qubits, measured
  against 18 and 22 qubits in step 1 to show the rate is flat there, as the
  13-qubit choice was justified for one thread.
- **Q8, gate.** Proposed criteria:
  1. every engine's actual over predicted is recorded and within the Q2
     factors;
  2. `plan_seconds` is reported and tested;
  3. the 50-term case plans under a bound set from step 1;
  4. an injected-clock test shows a factorized overrun replanning;
  5. the Phase 23 disk-tier ratio is re-measured in WSL and is within the
     Phase 23 2× bound;
  6. the golden planner corpus is unchanged;
  7. results are bitwise identical;
  8. `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0.

### 16.7 Owner answers (2026-11-05)

The owner answered 16.6 the same day. A review followed, and the owner then
accepted every correction it proposed. The answers below are the corrected
ones and replace the proposals in 16.2–16.6 wherever they differ.

- **Q1, workloads.** Every engine is measured on easy, representative and
  stress workloads:

  | Engine | Workloads |
  |---|---|
  | Dense, CPU and CUDA | random/general circuits, QFT, entangling circuits |
  | Stabilizer | Clifford and graph-state circuits |
  | Pauli propagation | Pauli-heavy expectation workloads |
  | MPS | low- and high-entanglement circuits |
  | Tensor network | low- and high-treewidth circuits |
  | Cut-and-recombine | partitionable and highly connected circuits |
  | Factorized | two independent blocks, and mixed-engine workloads |
  | Density matrix, trajectory | noisy circuits (added in review) |
  | Disk tier | the Phase 23 gate circuits (added in review) |
- **Q2, prediction accuracy.**
  - **Thresholds.** Set after the first complete benchmark.
  - **Reported per engine:**
    - the median and P90 of the symmetric error |log(actual / predicted)|;
    - the actual/predicted ratio itself;
    - systematic underprediction and overprediction, separately.
  - **How predictions are made.** Every prediction comes from a calibrated
    profile in the same process as the run it predicts. That is the only
    setting in which a tight bound can be met, because calibration varies
    about 1.5× between processes. The variation between processes is
    reported separately.
  - **Proposed eventual targets.** Median error ≤ 20% and P90 ≤ 50%,
    subject to the baseline.
- **Q3, timing.**
  - **`plan_seconds()`.** C++ `Result::plan_seconds()` and Python
    `plan_seconds` cover the initial plan only. On a plan-cache hit the
    figure is the cost of the lookup, and the documentation says so.
  - **`replan_seconds()`.** C++ `Result::replan_seconds()` and Python
    `replan_seconds` are the sum of the replans' planning time. Each replan
    line also prints its own figure.
  - **Execution time.** It stays `execute_s` and `sample_s`.
  - **Total wall-clock.** No total-time field is added. The caller's own
    clock measures it, and `plan_s + replan_s + execute_s + sample_s`
    accounts for MENOS's part of it.
  - **`explain()`.** The `actual` line gains `plan_s=` and `replan_s=`, and
    the timing normalisers in the tests learn both fields.
- **Q4, search limit.**
  - A deterministic node-count limit is added only if the post-pruning
    measurement shows a real planning-time problem.
  - There is never a wall-clock limit.
  - When the limit is reached, the candidate is refused with a named reason
    in `explain()`, as a pruned candidate is today.
- **Q5, observations.**
  - **The key.** A per-region observation is keyed by the region's
    identity: its qubit set within the job, and its engine, backend and
    tier.
  - **What it is never applied to.** A region of another qubit set,
    whatever its size.
  - **What the planner uses.** Only the time floor.
  - **What `explain()` reports.** Each replan line reports each region's
    estimated and actual time.
  - **Dropped.** Gate distribution and memory metrics would have no
    consumer (prime directive 2), so they are not recorded.
- **Q6, scope.** Factorized plans only. Circuit cutting stays refused on an
  overrun, with its message unchanged, until its reconstruction cost is
  modelled separately.
- **Q7, calibration.**
  - **The threaded rate.** Measured at 20 qubits as the primary point and
    validated against 18 and 22 qubits.
  - **The single-thread rate.** The 13-qubit rate is kept.
  - **Which rate prices a candidate.** The rate matching the thread count
    the kernel uses at the candidate's size.
  - **`max_threads` below the hardware thread count.** A job that sets
    `RunOptions::max_threads` below the hardware thread count is priced at
    the single-thread rate. That overpredicts, so it is the safe side under
    a time budget.
  - **Disk.** Reads and writes are priced at their own measured bandwidths.
- **Q8, gate.** The mandatory criteria:
  1. **Correctness:**
     - results bitwise identical to the Phase 21 baseline;
     - the golden planner corpus unchanged;
     - `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0.
  2. **Memory safety:** the sanitizer stages of both scripts.
  3. **Deterministic planning:**
     - the same job and the same hardware profile give the same plan;
     - plans that change because item 9 changes `estimated_seconds` are
       listed, not forbidden;
     - an unbudgeted job keeps its plan.
  4. **Replanning terminates,** by the proof in 16.8, pinned by an
     injected-clock test.

  The performance criteria are engine-specific and set from the first
  baseline. Planning and replanning overhead are reported separately.
- **Order.** Before implementation, 16.8 (the termination proof) and 16.9
  (the benchmark method) are written. Step 1 then establishes the complete
  baseline.
  - It adds benchmark code only: the accuracy benchmark, and planning-time
    measurement through the existing planner benchmarks.
  - It changes no library code or behaviour.

### 16.8 Termination of per-region replanning — [DERIVED]

**Setting.** A job J has a time budget B. The planner, `plan_job`, chooses
from a finite set P(J) of plans. P(J) is finite because:
- the strategies are a fixed list;
- the factorization's regions are fixed by the circuit;
- each region has finitely many (engine, backend, tier) choices.

For a plan p, let pred(p, O) be its predicted seconds under the
observations O, and let feasible(p, O) require pred(p, O) ≤ B. Today's
single-region argument (`replan_controller.cpp` lines 92–138) is the case
where every plan has one region.

**Assumptions,** each checked in the implementation and pinned by a test:
- **A1 (monotone).** Every floor only rises, and pred(p, O) is
  non-decreasing in every floor. A factorized plan's prediction is the sum
  of its regions' predictions, each raised to its floor.
- **A2 (plan floor).** At a crossing of plan p after total elapsed T > B,
  the runtime records:
  - a floor F_r = (observed time) for every region r of p that started, the
    running one included, keyed by (qubit set, engine, backend, tier);
  - a plan floor F_p = T, keyed by p's full region assignment.
  The planner raises pred(p, O) to at least F_p.
- **A3 (soundness).** Each F_r is the region's own time, a lower bound for
  that region on that implementation. F_p is the time that exact plan
  already took before being abandoned.
  - Neither figure is applied to any other key.
  - Time spent between regions is not attributed to any region. It is
    covered by F_p alone, which is why A2 needs F_p: Σ F_r alone can be
    below B when that overhead is large.

**Claim.** `run_with_replanning` performs at most |P(J)| − 1 replans before
it either completes or throws `ResourceExceeded`.

**Proof.**
1. Suppose plan p overruns. By A2, afterwards pred(p, O) ≥ F_p = T > B, so p
   is infeasible.
2. By A1, floors never fall, so p stays infeasible for the rest of the job.
3. Hence every replan either throws (no feasible plan remains) or chooses a
   plan not chosen before.
4. P(J) is finite, so there are at most |P(J)| attempts, and at most
   |P(J)| − 1 replans. ∎

**Notes.**
- The per-region floors F_r are not needed for termination; F_p alone
  suffices. They exist so that a *different* plan that reuses a region on
  the same implementation is priced with what was learned.
- A2 makes the existing refusal of multi-region overruns unnecessary for
  factorized plans. Cut plans keep it (Q6).
- For a single-region plan, F_p and F_r coincide with today's floor, so the
  existing proof is the special case.

### 16.9 Benchmark method for step 1 — [MENOS]

- **Builds.** Release presets: `scripts/verify-windows.ps1`'s release build
  natively, and `release` and `cuda12-release` in the WSL clone `~/MENOS`.
  The machine's load is logged before every round, as in section 14.
- **Accuracy (item 1).** A new program,
  `benchmarks/cost_model_accuracy_benchmark.cpp`, built on Q1's workloads.
  - Each engine is forced where forcing exists. Otherwise the planner's
    choice is checked.
  - The profile is calibrated once per process.
  - Each job runs once as a warm-up, then 5 timed repetitions.
  - Actual time is `execute_seconds + sample_seconds`. Predicted time is the
    chosen plan's `estimated_seconds`.
  - Each per-engine table reports the Q2 statistics: median and P90 of
    |log(actual / predicted)|, the ratio, and the signed bias.
  - Three separate processes per platform show the variation between
    processes.
  - Sizes are chosen so that one run takes 0.1–10 s.
- **Planning time (item 2).** The existing planner benchmarks on the 45-job
  corpus, plus the 20-qubit, depth-10, 50-term expectation case from
  `docs/BUILD_PLAN.md`.
  - Reported: the median and spread of `plan_job` time, and the number of
    min-fill search nodes visited.
  - The node count is the figure a Q4 limit would be derived from.
- **Calibration (item 9).**
  - The dense work rate is measured at 13 qubits on 1 thread, and at 18, 20
    and 22 qubits on every thread.
  - The calibration's own cost is measured.
  - The disk write and read bandwidths are recorded separately.
  - The Phase 23 gate circuits are re-measured against both the current
    pricing and the proposed one, computed offline from the recorded
    figures.
- **Replanning (item 3).** No baseline is needed: today a multi-region
  overrun is refused. The test is an injected clock.
- **Output.** Raw runs go in `docs/benchmarks/phase-24/step1/`. Tables are
  copied into a new section 16.10, and the owner sets the performance
  thresholds from them.

### 16.10 Step 1: the baseline — [MEASURED]

**Code.** Benchmark code only; no library behaviour changed.
- `ddec28d` and `8537681` add:
  - `benchmarks/cost_model_accuracy_benchmark.cpp`;
  - `benchmarks/dense_rate_probe.cpp`;
  - the 50-term planning case in `benchmarks/planner_benchmark.cpp`;
  - min-fill counters, under `MENOS_PLANNER_COUNTERS` only;
  - `bench/tools/accuracy_table.py`.
- Raw runs are in `docs/benchmarks/phase-24/step1/{windows,linux}/`.

**Conditions.** The same laptop for both: Intel Core Ultra 7 155H, 22 threads, mains power checked before every round (`load.log`).

| | Windows | Linux |
|---|---|---|
| OS | Windows 11 | WSL Ubuntu 26.04 |
| Compiler | MSVC 14.50 | GCC 15.2 |
| Build | `build/windows-bench` | `build/cuda12-release`, so CUDA rows on the RTX 4060 Laptop GPU |
| Accuracy runs | three separate processes | three separate processes |
| Planner benchmark | 5 repetitions | 5 repetitions |

**A run discarded.** A first Windows run, on 2026-11-06 at about 00:57, was on battery. Its calibrations differed 2.7× between processes. It was discarded, not recorded.

#### 16.10.1 Item 1: actual over predicted, per engine

Statistics are over every workload of Q1 and all three processes. Ratio r = actual / predicted. Error is |ln r|: 0.182 is a factor of 1.2, 0.405 a factor of 1.5.

**Linux (WSL):**

| Engine | n | median abs(ln r) (factor) | P90 abs(ln r) (factor) | median r | min..max r | under / over | bias ln r |
|---|---|---|---|---|---|---|---|
| dense (CPU) | 27 | 0.297 (1.35x) | 0.539 (1.71x) | 1.33 | 0.577..1.75 | 18 / 9 | +0.287 |
| dense (CUDA) | 6 | 0.067 (1.07x) | 0.210 (1.23x) | 0.936 | 0.811..0.977 | 0 / 6 | -0.067 |
| density matrix | 9 | 0.900 (2.46x) | 0.922 (2.51x) | 0.407 | 0.398..1.13 | 3 / 6 | -0.900 |
| factorized | 6 | 0.414 (1.51x) | 0.721 (2.06x) | 1.58 | 1.03..2.06 | 6 / 0 | +0.414 |
| MPS | 18 | 2.518 (12.4x) | 4.028 (56.2x) | 21.7 | 0.796..57.3 | 15 / 3 | +2.518 |
| Pauli | 18 | 2.017 (7.52x) | 4.323 (75.5x) | 0.409 | 0.0126..9.58 | 6 / 12 | -1.077 |
| stabilizer | 12 | 2.156 (8.63x) | 3.361 (28.8x) | 11.0 | 0.852..29.0 | 9 / 3 | +2.156 |
| tensor network | 6 | 1.595 (4.93x) | 1.618 (5.04x) | 4.93 | 4.90..5.04 | 6 / 0 | +1.595 |
| trajectory | 6 | 0.412 (1.51x) | 0.417 (1.52x) | 1.51 | 1.47..1.52 | 6 / 0 | +0.412 |

**Windows:**

| Engine | n | median abs(ln r) (factor) | P90 abs(ln r) (factor) | median r | min..max r | under / over | bias ln r |
|---|---|---|---|---|---|---|---|
| dense (CPU) | 27 | 0.818 (2.27x) | 1.519 (4.57x) | 0.441 | 0.194..1.50 | 1 / 26 | -0.818 |
| density matrix | 9 | 1.099 (3.00x) | 1.372 (3.94x) | 0.333 | 0.254..2.17 | 2 / 7 | -1.099 |
| factorized | 9 | 0.868 (2.38x) | 2.233 (9.33x) | 0.630 | 0.32..9.33 | 3 / 6 | -0.463 |
| MPS | 18 | 2.872 (17.7x) | 5.491 (242x) | 20.1 | 1.35..264 | 18 / 0 | +2.872 |
| Pauli | 18 | 1.961 (7.10x) | 4.829 (125x) | 0.270 | 0.00705..16.5 | 7 / 11 | -1.326 |
| stabilizer | 12 | 2.519 (12.4x) | 3.533 (34.2x) | 12.9 | 0.383..63.1 | 10 / 2 | +2.519 |
| tensor network | 6 | 2.503 (12.2x) | 2.693 (14.8x) | 12.2 | 4.13..14.8 | 6 / 0 | +2.503 |
| trajectory | 6 | 0.616 (1.85x) | 1.190 (3.29x) | 2.05 | 1.12..3.29 | 6 / 0 | +0.616 |

**Not measured:**
- **CUDA on Windows.** It is refused at configure time (Phase 18).
- **Cut-and-recombine, on any workload.** The planner never chose it on any workload tried:
  - two blocks joined by one CX (the golden corpus's `expectation_cut` family);
  - the same blocks scrambled by long-range pairs;
  - a highly connected circuit;
  - each of these under a 2 MiB memory budget.

  MPS, the tensor network or dense was always cheaper. Under the tight budget, the cut fragments needed slightly more memory than MPS did (for example 3.68 MB against 3.52 MB at w = 12). The golden corpus never plans a cut either. A cut plan cannot be built by hand faithfully, so the cut time model has no measurement.
- **Two tensor-network rows.** `random_pairs_d6_ising/n24` stopped at the memory budget.

**What the numbers say:**
- **Variation between processes.**
  - **Linux** is reproducible: the three processes agree within about 5% per workload, and the three calibrations within 1%.
  - **Windows** is not. The three calibrations of the dense rate gave 1.05, 1.35 and 1.56 × 10⁹ work units/s, and per-workload ratios move with them (for example MPS `ising_trotter_s4/n32`: 93, 37 and 242).

  A Windows figure is therefore a factor of 1.5 or more noisier than a Linux one, before any model error.
- **Dense CPU.**
  - **Linux** underpredicts by about 1.3–1.7× at 16–24 qubits on random brickwork and GHZ. QFT is within 0.8–0.97.
  - **Windows** overpredicts by about 2–5× at 20–24 qubits.
  - The two platforms differ because the 13-qubit single-thread calibration is about 3× slower under MSVC (0.78–1.6 × 10⁹) than under GCC (2.2 × 10⁹). The jobs themselves are much closer in speed.
- **Dense CUDA** is the one model within the owner's provisional target: median 1.07×, P90 1.23×.
- **MPS** underpredicts by 20–57× on Linux, worst on the low-entanglement Trotter circuits (40–56×), where every bond is small.
  - Its per-gate cost is therefore dominated by something the χ³ term does not see: a fixed per-gate cost of the SVD and its allocations.
  - Charging each gate at its own bond instead of the peak, as the BUILD_PLAN proposal suggested, would make the estimate smaller still, so it would move the prediction the wrong way. **[DERIVED]**
- **Stabilizer** underpredicts by 18–29× on depth-100 random Clifford circuits (n = 256, 1000), but by only 0.85–4× on graph states. The error grows with the gate count, not with n.
- **Pauli propagation.** The prediction grows about 4× per added non-Clifford rotation, as the worst-case branching bound charges it. The measured time stays flat at 20–35 ms. Its error runs from underprediction by 9.5× (2 rotations) to overprediction by 75× (12 rotations).
- **Tensor network** underpredicts by about 5× (Linux), consistently. Its predictions come from the formula: the candidate was pruned, and the engine's own estimate gives the figure.
- **Density matrix** overpredicts by about 2.5× at 10–12 qubits. **Trajectory** underpredicts by about 1.5×.
- **Factorized** is within 1.03–2.06× on Linux.

#### 16.10.2 Item 2: planning cost

**Planning time** (`plan_job` alone, the hand-built profile, median of 5):
- **The 50-term case** (20 qubits, depth 10, 50 random Pauli terms, a benchmark-only job): 1.7 ms on Linux and 6.1 ms on Windows, with or without a time budget. It took 4.4 s on 2026-11-02, before branch-and-bound.
- **The slowest corpus job** is `expectation_random_pairs_controlled/large`: 106 ms on Linux and 488 ms on Windows. That is the Schmidt-rank analysis behind the MPS estimate, with one min-fill search.
- **Every other job** plans in under 16 ms on Linux.

**Min-fill search** (the `MENOS_PLANNER_COUNTERS` build, Linux):
- At most 7,796 elimination steps in one job, over at most 2 searches.
- Every search in the corpus either completed or was stopped early by branch-and-bound.
- No search runs long enough for a cap to matter. **[DERIVED]**

#### 16.10.3 Item 9: the dense rate by size and threads, and the disk link

`menos_dense_rate_probe`, work units/s, is the calibration's own circuit shape at each size. The threading threshold is 2^16 groups, reached at 17 qubits.

| Qubits | Linux, 1 thread | Linux, 22 threads | Windows, 1 thread | Windows, 22 threads |
|---|---|---|---|---|
| 13 | 2.24e9 | 2.29e9 | 0.78e9 | 1.08e9 |
| 16 | 2.26e9 | 1.84e9 | 0.54e9 | 0.84e9 |
| 17 | 1.95e9 | 1.24e9 | 0.74e9 | 0.75e9 |
| 18 | 2.11e9 | 2.49e9 | 0.70e9 | 2.79e9 |
| 20 | 1.83e9 | **5.34e9** | 0.68e9 | **3.48e9** |
| 22 | 1.28e9 | 4.62e9 | 0.47e9 | 1.43e9 |

- **The threaded rate is not flat.** Q7 asked for the 20-qubit rate to be validated against 18 and 22 qubits. It is not flat there: on Windows it falls 2.4× from 20 to 22 qubits (the state leaves the last-level cache), and on Linux 1.16×.
- **The cost of a 20-qubit threaded measurement** is 0.13 s (Linux) and 0.19 s (Windows). Today's whole calibration costs about 0.1 s.
- **A threaded rate would not fix in-memory dense jobs on Linux.**
  - The dense jobs at 20–24 qubits ran *slower* than the single-thread calibration predicts, even on 22 threads (ratio 1.33–1.7).
  - Pricing them at the threaded rate would multiply that underprediction by about 2–2.4×, to roughly 3–3.5×.
  - On Windows the same change would move the 20–22-qubit ratios from 0.2–0.45 to about 0.45–0.6.
  - The jobs' gate mix (RY/RZ/CX brickwork, QFT's CP) runs more slowly per work unit than the calibration's H/CX at these sizes, so a single dense rate cannot fit both platforms. **[DERIVED]**
- **Disk link.**
  - **Measured bandwidths:**
    - Linux (ext4 under WSL): write 1.42 GB/s, read 5.40 GB/s.
    - Windows: write 3.62 GB/s, read 4.47 GB/s.
  - **Traffic.** In the Phase 23 gate runs, 88–89% of the link bytes are reads. The executor writes the file 1 + passes times and reads it passes + outputs times.
- **The Phase 23 disk-tier runs re-priced offline** from their recorded figures (work time = recorded prediction − link / min(write, read); threaded factor = this probe's 22-qubit threaded rate over its 13-qubit single-thread rate):

  | Run | n | Measured s | Today | Split read/write | Threaded rate | Both |
  |---|---|---|---|---|---|---|
  | Windows a0 | 31 | 915.4 | 1.17 | 1.27 | 1.64 | 1.84 |
  | Windows a | 31 | 479.7 | 0.58 | 0.63 | 0.82 | 0.92 |
  | Windows b | 31 | 765.2 | 0.89 | 0.97 | 1.25 | 1.40 |
  | Windows c | 31 | 439.5 | 0.51 | 0.56 | 0.71 | 0.81 |
  | WSL a | 30 | 155.6 | **0.35** | 0.39 | 0.54 | **0.62** |

  The figures are measured over predicted. Only the two changes together bring every run, the WSL miss included, within the Phase 23 2× bound. **[DERIVED]**

#### 16.10.4 Recommendations for checkpoint A — [OPEN]

These recommendations decide what the remaining items do.
- **Q4, the search cap: none.** The case that motivated it plans in 2–6 ms, and no search in the corpus needs a cap.
- **Item 9: build the threaded rate and the split link for the disk tier only.** Leave the in-memory dense price on today's rate.
  - The measurement supports the change where the state is far beyond the cache (29–31 qubits) and contradicts it at 20–24 qubits on Linux.
  - The size-dependent gap between the calibration circuit and real gate mixes is a cost-model question, recorded for a later item, not guessed at here.
- **MPS: do not build per-gate bonds.** That would make an already 20–57× optimistic estimate more optimistic. A per-gate fixed cost measured from the Trotter rows would address the observed error, but it is a new modelling term needing its own measured derivation, and it is proposed as a separate item.
- **Q2 thresholds.**
  - Only the CUDA dense model meets the provisional target (median 20%, P90 50%).
  - Recommended for this phase's gate: record the table as the baseline, and require that items 2, 3 and 9 make no engine's median or P90 error worse on Linux.
  - The disk tier must meet the Phase 23 2× bound on the Phase 23 circuits on both platforms.
  - Per-engine model fixes (stabilizer, MPS, Pauli, tensor network, density matrix) are a follow-on phase whose targets are set from this table.

### 16.11 Checkpoint A: owner answers (2026-11-06)

The owner took every recommendation of 16.10.4:
- **Item 9.**
  - **The threaded rate prices the disk tier only.** It is measured at 20 qubits on every thread.
  - **Reads and writes** are priced at their own measured bandwidths.
  - **In-memory dense** keeps today's 13-qubit single-thread rate. A follow-on item records the gap between the calibration circuit and real gate mixes at 20–24 qubits.
- **MPS.** The formula is unchanged in this phase. A per-gate fixed-cost term, derived from the measured Trotter rows with its own measurement and ADR, is recorded as a follow-on item. The per-gate-bond refinement is withdrawn: it moves the estimate the wrong way.
- **Accuracy gate.**
  - The 16.10.1 tables are the baseline.
  - Items 2, 3 and 9 must make no engine's median or P90 error worse on Linux.
  - The disk tier must be within 2× on the Phase 23 gate circuits on both platforms.
  - Per-engine model fixes (stabilizer, MPS, Pauli, tensor network, density matrix) are a follow-on phase with targets set from the baseline.
- **Q4.** No search cap. The measurement shows no planning-time problem.

### 16.12 Implementation and final measurement — [MEASURED]

**Commits** on `phase-24-planner-accuracy`, after the step-1 records:

| Commit | Change |
|---|---|
| `97990ca` | Item 2: `Result::plan_seconds()` / `replan_seconds()`, `ReplanEvent::plan_seconds`, Python `plan_seconds` / `replan_seconds` |
| `e173f13` | Item 9: `HardwareProfile::dense_threaded_work_units_per_second`, `TierCost::link_write_bytes`, the disk tier's new price |
| `f8aa3d3` | Item 3: `ExecutionMonitor::at_region_boundary`, per-region and plan floors, the Factorized replan, `Region::estimated_seconds`, `RegionObservation` in `ReplanEvent`, the replan loop's termination check |
| `291de4c` | The cut-differential fixture added to the accuracy benchmark, so cut-and-recombine has measured rows |

**What `explain()` reports.**
- **`plan_s=`** is always on the `actual` line, beside the other wall-clock fields.
- **`replan_s=`, each replan line's `replanned_in=`, and the per-region lines of a replanned Factorized plan** appear only after a replan. So a run that never replanned carries no replan text at all, which is the Phase 11 invariant and is pinned by `ReplanTest.ARunWithNoReplanEmitsNoReplanTextAtAll`. Printing `replan_s=0` on every report would have broken that test, which is why `replan_s=` is conditional.

**Termination.** The 16.8 argument is now checked in `run_with_replanning` on every replan, not bounded by a fixed attempt count. The key of every plan run so far is kept, and a replan that returns one of them is an `InternalError`. The single-region bound of six attempts is subsumed by this check.

**Final measurement** (`docs/benchmarks/phase-24/final/`, on `291de4c` plus documentation edits, mains power checked before every round, same machine and conditions as 16.10):

**Accuracy, Linux,** three processes. Calibrations: 2.10–2.23 × 10⁹ work units/s.

| Engine | n | median abs(ln r) | P90 abs(ln r) | median r | step-1 median / P90 |
|---|---|---|---|---|---|
| dense (CPU) | 27 | 0.280 | 0.525 | 1.16 | 0.297 / 0.539 |
| dense (CUDA) | 6 | 0.089 | 0.220 | 0.915 | 0.067 / 0.210 |
| density matrix | 9 | 0.911 | 0.987 | 0.402 | 0.900 / 0.922 |
| factorized | 6 | 0.376 | 0.718 | 1.50 | 0.414 / 0.721 |
| MPS | 18 | 2.519 | 4.039 | 21.4 | 2.518 / 4.028 |
| Pauli | 18 | 2.007 | 4.352 | 0.399 | 2.017 / 4.323 |
| stabilizer | 12 | 2.068 | 3.363 | 10.4 | 2.156 / 3.361 |
| tensor network | 6 | 1.585 | 1.623 | 4.88 | 1.595 / 1.618 |
| trajectory | 6 | 0.419 | 0.464 | 1.52 | 0.412 / 0.417 |
| cut (new rows) | 6 | 1.301 | 1.634 | 3.77 | not measured |

**Accuracy, Windows,** three processes. Calibrations: 1.57–1.59 × 10⁹. The tables are in the run's `accuracy_run*.csv`, summarised by `bench/tools/accuracy_table.py`. The same pattern as step 1 holds:
- dense and density matrix overpredict about 3×;
- MPS underpredicts 18×, stabilizer about 6×, tensor network about 5×;
- cut underpredicts about 5×;
- trajectory is within 1.13×.

**The disk tier on the Phase 23 gate circuits** (`gate-run/`, `docs/benchmarks/phase-23-gate/scripts/gate.{ps1,sh}`, run `p24`):

| Platform | n (reach) | Predicted s | Measured s | Measured / predicted | Within 2× | Overlap n = 29: counts / expectation | Scratch files left |
|---|---|---|---|---|---|---|---|
| Windows | 31 | 383.7 | 449.1 | **1.17** | yes | bitwise / bitwise | 0 |
| Linux (WSL) | 30 | 202.3 | 163.9 | **0.81** | yes | bitwise / bitwise | 0 |

The WSL reach run measured 0.35 on Phase 23's pricing (`docs/adr/0027` section 10.3). The 16.10.3 offline re-pricing predicted 0.62; the run gives 0.81.

**The accuracy gate against the baseline** (16.11: no engine's median or P90 worse on Linux). Two kinds of engine need to be told apart:
- **Engines whose pricing this phase changed** are covered by the disk-tier rows above. Item 9 prices only the disk tier. Item 3's floors apply only after an overrun, and the benchmark sets a budget that never binds.
- **Every other engine's prediction is computed by unchanged code.** `DiskTierSelectionTest.TheThreadedRatePricesTheDiskTierOnlyOnEveryThread` checks this for in-memory dense, and the golden corpus is unchanged.

Read strictly, the comparison above is not clean:
- **Improved:** dense, factorized, stabilizer and tensor network medians, and the Pauli median.
- **Worse:** the dense (CUDA) median (+0.022) and P90 (+0.010), density matrix (+0.011 / +0.065), MPS (+0.001 / +0.011), the Pauli P90 (+0.029), the tensor-network P90 (+0.005), and trajectory (+0.007 / +0.047).

These moves are the size of the run-to-run variation of unchanged code between two sessions: the GPU's clock state varies about 2× between runs (CLAUDE.md, Phase 16), and process-to-process ratios differ by up to about 5%. They are not an effect of this phase's changes, which do not reach these predictions. **[DERIVED]** This is stated for the owner's ruling, not claimed as met.

**Planning cost:** see 16.10.2. Nothing in items 2, 3 or 9 runs during planning without an observation, except one clock read around the planner call.

**Verification** — recorded in 16.13.

### 16.13 Verification of the accuracy branch — [MEASURED]

Final commit of items 1, 2, 3 and 9: `05056f8` on `phase-24-planner-accuracy`. Two commits after 16.12 changed no behaviour. They add clang-tidy annotations that the first `scripts/verify.sh` run asked for:
- `84be301`: `NOLINT` on `measure_work_units_per_second` in `src/plan/hardware_profile.cpp`;
- `6dfc2cb`/`05056f8`: `NOLINT` blocks and `engine->engine_id()` in `benchmarks/cost_model_accuracy_benchmark.cpp`.

| Script | Where | Result |
|---|---|---|
| `scripts/verify.sh` | WSL clone `~/MENOS`, `UV_PYTHON_PREFERENCE=only-managed` | exit 0, "all stages passed": dev and release 869/869; Python 576 passed; `cuda12-dev` and `cuda12-release` 949/949 each, with the GPU tests executing; the docs stage skipped (no Node in the clone) |
| `scripts/verify-windows.ps1` | native Windows, detached worktree `C:/Users/janak/menos-verify` | exit 0, "verify-windows: all stages passed": `windows-dev` (ASan) 870/870, `windows-release` 870/870, pytest 576 passed, wheel smoke 2 passed |

The first Windows attempt ran at the same time as the WSL run. It failed in the ASan stage with an out-of-memory error from the sanitizer, not a test failure. Run alone, it passed as above. The two scripts are now run one after the other.

## 17. Kickoff: item 7, gate fusion in the dense CPU engine — [MENOS] / [OPEN]

**Started by the owner on 2026-11-09**, in the order the owner set on
2026-11-06 (`docs/BUILD_PLAN.md`, "Order of the remaining items"): item 7
first, then item 8 (SIMD kernels), then the time-model fixes, then item 4.
Branch `phase-24-dense-fusion`, from `phase-24-planner-accuracy` after its
verification record (16.13). The process is the usual one: this section's
questions (17.4) are answered by the owner after step 1 (17.5) has measured,
a plan is then approved, and nothing is built before that. Section 17.5 is
the only part of this section with numbers in it.

### 17.1 What the code does today — [MENOS], read from the source

- **One pass per operation.** `apply_operations`
  (`src/engines/dense/dense_engine.cpp`) walks the region's operations in
  order and calls `apply_matrix` once per `GateOp` or `UnitaryOp`, generating
  a `GateOp`'s matrix with `ir::generate_matrix` on every call. The gap
  between two operations is the Phase 11 safe boundary; `completed` counts
  operations.
- **The kernels** (`src/kernels/cpu/dense_kernels.cpp`):
  - k = 1: `pair_run`, unit-stride over runs of 2^q pairs, with an AVX2
    `target_clones` on GCC (MSVC compiles the plain loop);
  - k = 2: a gather of four amplitudes per group, not vectorised;
  - a diagonal matrix of any k: one streamed multiply per amplitude;
  - k ≥ 3: `apply_matrix_generic`, a per-group gather of 2^k amplitudes and
    indices into per-chunk scratch, the matrix applied, a scatter back.
  - Every call first checks `is_diagonal` over the whole matrix (4^k reads).
  - The team is one thread below 2^16 groups and every thread from there.
- **Diagonal fusion exists, opt-in.** `ir::fuse_diagonal_gates`
  (`src/ir/fuse_diagonal.{h,cpp}`, `RunOptions::fuse_diagonal_gates`,
  default off) groups each run of consecutive diagonal gates into blocks of
  at most `kMaxUnitaryQubits` = 5 qubits and replaces a block of two or more
  gates by one diagonal `UnitaryOp`. It is applied in
  `DenseEngine::execute` only: the CUDA engine and the disk tier apply the
  original operations, `DenseEngine::estimate` charges the unfused circuit,
  and with it on `completed` already counts blocks. The documentation says
  the fused run is exact within rounding and not bitwise the unfused one.
- **Cost model.** A gate on k qubits of an n-qubit region costs 2^(n+k)
  work units (`dense_work_units`); the calibration converts work units to
  seconds from a 13-qubit, 2-layer H/CX circuit run through the engine
  itself (`hardware_profile.cpp`), so whatever the engine does to a circuit
  before sweeping it is inside the measured rate.
- **What is pinned bitwise today, and by what:**
  - every thread count gives the same bytes (Phase 15,
    `tests/engines/dense/threaded_kernels_test.cpp`);
  - the specialised and diagonal kernels equal the generic kernel
    (`specialised_kernels_test.cpp`);
  - CPU and CUDA statevectors and counts are identical
    (`docs/adr/0014` section 10, `tests/api/cpu_cuda_differential_test.cpp`);
  - the disk tier's counts and expectation values are the in-memory
    engine's (Phase 23, `tests/engines/dense_tiered`);
  - the memory-accounting fixtures' outputs equal the Phase 21 baseline
    (`tests/memory/data/baseline_{linux,windows}.json`), with
    `fuse_diagonal_gates` off.
- **Where MENOS stands** (`docs/benchmarks/cross-simulator-2026-11-05`,
  Linux, every simulator at its defaults, best of 3):
  - dense random circuits: at 20 qubits MENOS 0.29 s against Aer 1.58 s;
    at 22 qubits 1.54 s against 1.57 s; at 24 qubits (counts) 9.7 s
    against 12.9 s; **at 26 qubits 49 s against 16 s**;
  - **Grover, 16 qubits, 122k gates: 35 s against Aer's 9 s** (288 µs per
    gate on a 1 MiB state);
  - qsim, which fuses and vectorises, runs every dense case in 0.1–0.4×
    MENOS's time.
  - On Windows (`docs/BUILD_PLAN.md`, "Measured on 2026-11-02") MENOS was
    1.5–2× slower than Aer at 20–24 qubits on every thread and 6× slower on
    one thread.

  Aer's fusion is on by default from 14 qubits (`fusion_threshold`), with a
  5-qubit window and per-size cost factors; qsim fuses to 6 qubits. Neither
  is a MENOS invariant; they are what the comparison is against.

### 17.2 Design proposed — [MENOS]

- **A fusion pass, `ir::fuse_gates`** (`src/ir/fuse_gates.{h,cpp}`, beside
  `fuse_diagonal`): it turns a region's operation list into a list of
  blocks, each of at most `window` qubits, and replaces every block of two
  or more gates by one `UnitaryOp`; a block of one gate is left exactly as
  it was, so a gate that fuses with nothing is applied bitwise as today.
  `MeasureOp` is carried through unchanged; a `ChannelOp` never reaches a
  dense engine.
- **Grouping rule.** Program order with the one reordering that is free of
  any commutation analysis: a gate may join an open block only if every
  block opened after that one is on qubits disjoint from the gate's (so the
  gate commutes back past them), and among those it joins the oldest block
  it fits in under the cost rule below; otherwise it opens a new block.
  Blocks are emitted in the order they opened. The soundness argument is
  three lines and is written in the header; a test applies the fused and
  unfused lists to random states.
- **Cost rule, structural and machine-independent.** A gate joins a block
  only if the fused block is predicted to cost no more than its parts:
  c(k_fused) ≤ c(k_block) + c(k_gate), with c(k) the cost of one k-qubit
  pass relative to a 1-qubit pass, a fixed table derived from step 1 (17.5)
  and recorded here, never calibrated at run time. The rule must not read
  the machine: CLAUDE.md requires the same circuit, options and seed to
  give the same counts on every platform, and the grouping decides the
  rounding of the result, so a grouping that depended on a measured rate
  would make results differ between machines. A diagonal block (every gate
  diagonal) has its own, much lower, c_d(k) from the same measurement, so
  the pass subsumes what `fuse_diagonal_gates` does today.
- **Below a size the engine is unchanged.** Step 1 says from which n a pass
  is bandwidth-bound; below that size (n_f) fusion would trade passes for
  arithmetic on a cache-resident state, and the pass is not applied, so a
  small job's results and report stay byte-identical to today.
- **The fused matrix is computed by the kernels themselves.** A block's
  matrix is the generic kernel's action on each basis vector of the block's
  2^k-dimensional space, applying the block's gates in order
  (`kernels::cpu::apply_matrix_generic` on 2^k amplitudes). That keeps the
  product inside the no-contraction, bitwise-reproducible arithmetic every
  kernel already obeys, so a fused matrix is the same bytes on Linux,
  Windows and both backends, and the fused run's results are bitwise
  identical across thread counts, CPU/CUDA and the disk tier exactly as the
  unfused run's are.
- **Which engines apply the pass.** The three consumers of the dense
  operation stream: `DenseEngine` (CPU), `CudaDenseEngine` and the disk
  tier's chunked executor, each calling the same pass with the same window,
  so the three bitwise identities above survive with fusion on. The
  density-matrix and trajectory engines are not touched (they share the
  kernel, not the operation loop; item 7 is the dense engine).
- **Cost model.** `DenseEngine::estimate` (and the CUDA and disk-tier
  estimates) charge the fused circuit: the pass runs inside the estimate
  (it is O(gates · 4^window) and the blocks' matrices are not built there,
  only their qubit sets), work units are summed over blocks, and the
  per-gate transient in `peak_bytes_of` becomes the largest block's matrix
  plus the generic kernel's scratch. The planner therefore prices the
  passes it will really make. Every dense candidate's `work_units` changes,
  so the golden planner corpus changes where a dense candidate's figure is
  printed; the changed goldens are listed in the ADR and regenerated with
  `MENOS_UPDATE_GOLDEN=1`, and no plan may change its winner except where
  the ADR says why.
- **Monitoring.** The Phase 11 safe boundary becomes the gap between two
  blocks; `completed` counts blocks, as it already does for diagonal
  fusion. `ReplanEvent::boundaries_crossed` keeps its meaning (boundaries,
  not gates).
- **`explain()`.** Each dense region's line gains, only when fusion
  absorbed at least one gate, ` fusion=window:K blocks=B absorbed=G`,
  computed at plan time by the estimate (so `preview()` reports it too). A
  region with nothing fused, and every non-dense region, prints exactly as
  today.
- **API.** `RunOptions::fuse_gates` (C++), `fuse_gates` (Python
  `RunOptions`, `menos.run`), CLI `--fusion on|off`. The window is not a
  public option: it is the constant the owner sets from step 1. The
  existing `fuse_diagonal_gates` is kept with its meaning (diagonal-only
  fusion when `fuse_gates` is off); with `fuse_gates` on it is implied.
  The default is Q2.
- **Tests.** The fused list equals the unfused list within
  `common::kDefaultAbsTol` on random states (`tests/ir/fuse_gates_test.cpp`,
  the oracle of `fuse_diagonal_test.cpp`); the grouping rule's soundness on
  circuits with known dependencies; a fused run bitwise identical on 1, 2,
  3 and every thread; CPU/CUDA and disk-tier differential tests run with
  fusion on; a job below n_f and a job with `fuse_gates=false` give
  byte-identical results and `explain()` to the run before this item (the
  Phase 21 baseline pins the former); the cost estimate equals the sum over
  blocks; `explain()` goldens; the monitor's `completed` equals the block
  count on an injected clock.

### 17.3 Step 1: the measurement before any fusion code — [MENOS]

`benchmarks/dense_fusion_probe.cpp`, on the unchanged kernels through
`engines::dense::apply_matrix`, no library change:

1. **One pass by k and n.** For n = 14, 16, …, 24 and k = 1..5, dense
   (random, every entry non-zero) and diagonal matrices on operands spread
   across the register, on one thread and on every hardware thread: median,
   min and max of 10 timed calls after a warm-up; the bytes the pass streams
   (2 · 16 · 2^n) over the median against the calibration's streaming
   bandwidth. It answers: from which n is a pass bandwidth-bound (the time
   stops depending on k and approaches the bandwidth floor), and what each
   k costs relative to k = 1 where it is not — the c(k) and c_d(k) tables
   and n_f of 17.2.
2. **A circuit, gate by gate and in blocks.** The random brickwork family
   of the cross-simulator comparison (a U on every qubit, then CX on a
   random perfect matching, 10 layers) at n = 16, 20, 22, 24, applied as the
   engine applies it and as blocks fused in the program for windows 2..5 by
   the 17.2 grouping rule with no cost rule (every gate that fits joins):
   the pass count, the median of 3 timed runs, the speed-up, and the
   largest amplitude difference between the two final states. It answers:
   does cutting passes cut time (the BUILD_PLAN's precondition for building
   any fusion), by how much per window, and how large the rounding is.

Both on the gate machine (Core Ultra 7 155H), Linux (WSL, GCC 15.2) in
`build/cuda12-release` and Windows (MSVC 14.50) in `build/windows-bench`,
on mains power, recorded in `docs/benchmarks/phase-24/fusion-step1/`.

### 17.4 Owner questions — [OPEN]

- **Q1, the window.** Proposed: the largest k for which step 1's circuit
  measurement still gains at 22–24 qubits, at most 5 (`kMaxUnitaryQubits`,
  the generic kernel's ceiling). Expected 3–5; the number is 17.5's.
- **Q2, default.** Proposed: **on by default** from n_f qubits,
  `RunOptions::fuse_gates = false` to turn it off. The alternative, opt-in
  like `fuse_diagonal_gates`, leaves the measured gap in place for every
  user who does not read the options; Aer and qsim fuse by default. Under
  "on by default" the Phase 21 bitwise baseline is Q3.
- **Q3, the Phase 21 baseline.** Proposed: keep `baseline_{linux,windows}.json`
  as they are and run the fixtures with `fuse_gates=false` against them
  (so the kernels stay pinned to the bytes of `1f89192`), and add the same
  fixtures with fusion on against a new `baseline_fused_*.json` generated
  once, so the fused path is pinned bitwise too, on both platforms. The
  alternative, regenerating the existing files, loses the pin on the
  unfused path.
- **Q4, which engines.** Proposed: the CPU dense engine, the CUDA dense
  engine and the disk tier, all through the one pass, so CPU/CUDA and
  in-memory/disk stay bitwise identical with fusion on. The alternative,
  CPU only (as `fuse_diagonal_gates` is), breaks both identities whenever
  fusion is on, which under Q2 is by default.
- **Q5, the cost model.** Proposed: charge the fused circuit (17.2), list
  the golden-corpus changes. The alternative, charging the unfused circuit
  and letting calibration absorb the difference, misprices every dense
  candidate against the other engines by the pass ratio.
- **Q6, the gate.** Proposed criteria:
  1. fused results within `common::kDefaultAbsTol` of the unfused run on
     the accuracy benchmark's dense workloads and on the Phase 21 fixtures;
  2. the fused run bitwise identical on every thread count, between CPU and
     CUDA, and between the in-memory engine and the disk tier;
  3. a job with fusion off, or below n_f, byte-identical in results and
     `explain()` to today;
  4. a dense speed-up target per size, set by the owner from 17.5 as
     same-run ratios fused/unfused (as Phase 16 set its targets), on the
     brickwork family and the cross-simulator dense cases at 20–26 qubits;
  5. `explain()` reports the fusion outcome and the cost estimate charges
     the blocks (a test equates them);
  6. the golden corpus's changed plans are listed and none changes its
     winner without a stated reason;
  7. `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0.

### 17.5 Step 1: the measurement — [MEASURED]

`benchmarks/dense_fusion_probe.cpp` at `fcbf947`, run by
`docs/benchmarks/phase-24/fusion-step1/scripts/step1.{sh,ps1}` on
2026-11-09, mains power checked before and after (`load.log`), the machine
otherwise idle (Linux load average 0.87 at the start). Raw CSVs, the
environment lines and the commit are in
`docs/benchmarks/phase-24/fusion-step1/{linux,windows}/`; the tables below
are `bench/tools/fusion_table.py` over them. The same laptop as 16.10: Intel
Core Ultra 7 155H, 22 threads; Linux is WSL Ubuntu 26.04, GCC 15.2,
`build/bench` (Release, CPU only); Windows is MSVC 14.50,
`build/windows-bench`. The "floor" in the pass tables is
2 · 16 · 2^n bytes at the calibration's streaming bandwidth of the same
process (Linux 32.4 GB/s, Windows 35.9 GB/s); a pass below 1× floor streams
the state faster than that single-threaded copy did, which the threaded
rows at 20–24 qubits do, so the floor is a reference, not a bound.

#### 17.5.1 One pass by operand count — Linux

Median µs per pass, 10 repeats; in brackets the ratio to the 1-qubit pass of
the same size and thread count, and the ratio to the floor.

**pass_dense, 1 thread** — median µs per pass (× the k = 1 pass of the same size; × the bandwidth floor)

| n | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| 14 | 6 (1.0×; 0.36× floor) | 52 (8.8×; 3.20× floor) | 172 (29.4×; 10.64× floor) | 257 (43.9×; 15.88× floor) | 451 (76.8×; 27.82× floor) |
| 16 | 23 (1.0×; 0.35× floor) | 219 (9.6×; 3.38× floor) | 690 (30.4×; 10.65× floor) | 1060 (46.6×; 16.35× floor) | 1853 (81.5×; 28.59× floor) |
| 18 | 100 (1.0×; 0.39× floor) | 861 (8.6×; 3.32× floor) | 2715 (27.0×; 10.47× floor) | 4311 (42.9×; 16.64× floor) | 7827 (77.9×; 30.20× floor) |
| 20 | 654 (1.0×; 0.63× floor) | 3409 (5.2×; 3.29× floor) | 11277 (17.2×; 10.88× floor) | 17623 (26.9×; 17.00× floor) | 30574 (46.8×; 29.49× floor) |
| 22 | 3605 (1.0×; 0.87× floor) | 13632 (3.8×; 3.29× floor) | 45209 (12.5×; 10.90× floor) | 69814 (19.4×; 16.84× floor) | 125523 (34.8×; 30.27× floor) |
| 24 | 14945 (1.0×; 0.90× floor) | 53473 (3.6×; 3.22× floor) | 185617 (12.4×; 11.19× floor) | 287888 (19.3×; 17.36× floor) | 497898 (33.3×; 30.02× floor) |

**pass_dense, 22 threads** — median µs per pass (× the k = 1 pass of the same size; × the bandwidth floor)

| n | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| 14 | 6 (1.0×; 0.36× floor) | 52 (8.8×; 3.20× floor) | 163 (27.7×; 10.08× floor) | 257 (43.6×; 15.88× floor) | 520 (88.1×; 32.12× floor) |
| 16 | 24 (1.0×; 0.37× floor) | 207 (8.8×; 3.20× floor) | 668 (28.3×; 10.31× floor) | 1026 (43.4×; 15.84× floor) | 1911 (80.8×; 29.49× floor) |
| 18 | 370 (1.0×; 1.43× floor) | 532 (1.4×; 2.05× floor) | 2765 (7.5×; 10.67× floor) | 4439 (12.0×; 17.13× floor) | 7747 (20.9×; 29.89× floor) |
| 20 | 574 (1.0×; 0.55× floor) | 1045 (1.8×; 1.01× floor) | 2048 (3.6×; 1.98× floor) | 3174 (5.5×; 3.06× floor) | 31638 (55.1×; 30.52× floor) |
| 22 | 1872 (1.0×; 0.45× floor) | 2662 (1.4×; 0.64× floor) | 7444 (4.0×; 1.80× floor) | 12892 (6.9×; 3.11× floor) | 19192 (10.3×; 4.63× floor) |
| 24 | 7824 (1.0×; 0.47× floor) | 9005 (1.2×; 0.54× floor) | 27180 (3.5×; 1.64× floor) | 47299 (6.0×; 2.85× floor) | 77449 (9.9×; 4.67× floor) |

**pass_diagonal, 1 thread** — median µs per pass (× the k = 1 pass of the same size; × the bandwidth floor)

| n | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| 14 | 5 (1.0×; 0.29× floor) | 30 (6.5×; 1.86× floor) | 41 (8.8×; 2.54× floor) | 50 (10.8×; 3.11× floor) | 63 (13.4×; 3.88× floor) |
| 16 | 17 (1.0×; 0.26× floor) | 116 (6.9×; 1.80× floor) | 157 (9.3×; 2.42× floor) | 200 (11.8×; 3.09× floor) | 289 (17.0×; 4.46× floor) |
| 18 | 151 (1.0×; 0.58× floor) | 474 (3.1×; 1.83× floor) | 638 (4.2×; 2.46× floor) | 810 (5.4×; 3.13× floor) | 986 (6.5×; 3.80× floor) |
| 20 | 627 (1.0×; 0.60× floor) | 1962 (3.1×; 1.89× floor) | 2592 (4.1×; 2.50× floor) | 3171 (5.1×; 3.06× floor) | 3838 (6.1×; 3.70× floor) |
| 22 | 4194 (1.0×; 1.01× floor) | 12036 (2.9×; 2.90× floor) | 13549 (3.2×; 3.27× floor) | 14819 (3.5×; 3.57× floor) | 16561 (3.9×; 3.99× floor) |
| 24 | 17219 (1.0×; 1.04× floor) | 47761 (2.8×; 2.88× floor) | 54969 (3.2×; 3.31× floor) | 60099 (3.5×; 3.62× floor) | 67545 (3.9×; 4.07× floor) |

**pass_diagonal, 22 threads** — median µs per pass (× the k = 1 pass of the same size; × the bandwidth floor)

| n | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| 14 | 5 (1.0×; 0.30× floor) | 31 (6.4×; 1.91× floor) | 42 (8.8×; 2.60× floor) | 50 (10.5×; 3.10× floor) | 62 (12.9×; 3.82× floor) |
| 16 | 17 (1.0×; 0.25× floor) | 117 (7.1×; 1.80× floor) | 157 (9.5×; 2.42× floor) | 200 (12.1×; 3.09× floor) | 244 (14.8×; 3.76× floor) |
| 18 | 282 (1.0×; 1.09× floor) | 340 (1.2×; 1.31× floor) | 635 (2.3×; 2.45× floor) | 808 (2.9×; 3.12× floor) | 981 (3.5×; 3.79× floor) |
| 20 | 748 (1.0×; 0.72× floor) | 752 (1.0×; 0.73× floor) | 920 (1.2×; 0.89× floor) | 1031 (1.4×; 0.99× floor) | 4030 (5.4×; 3.89× floor) |
| 22 | 1578 (1.0×; 0.38× floor) | 1898 (1.2×; 0.46× floor) | 2352 (1.5×; 0.57× floor) | 2521 (1.6×; 0.61× floor) | 2804 (1.8×; 0.68× floor) |
| 24 | 8236 (1.0×; 0.50× floor) | 9001 (1.1×; 0.54× floor) | 9523 (1.2×; 0.57× floor) | 8963 (1.1×; 0.54× floor) | 10220 (1.2×; 0.62× floor) |

#### 17.5.2 One pass by operand count — Windows

**pass_dense, 1 thread** — median µs per pass (× the k = 1 pass of the same size; × the bandwidth floor)

| n | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| 14 | 46 (1.0×; 3.17× floor) | 102 (2.2×; 7.01× floor) | 179 (3.9×; 12.29× floor) | 249 (5.4×; 17.08× floor) | 433 (9.3×; 29.63× floor) |
| 16 | 80 (1.0×; 1.37× floor) | 184 (2.3×; 3.15× floor) | 674 (8.4×; 11.54× floor) | 1057 (13.2×; 18.10× floor) | 1734 (21.7×; 29.68× floor) |
| 18 | 360 (1.0×; 1.54× floor) | 766 (2.1×; 3.28× floor) | 2665 (7.4×; 11.41× floor) | 4105 (11.4×; 17.57× floor) | 7233 (20.1×; 30.95× floor) |
| 20 | 1421 (1.0×; 1.52× floor) | 4061 (2.9×; 4.35× floor) | 11485 (8.1×; 12.29× floor) | 17209 (12.1×; 18.41× floor) | 28074 (19.8×; 30.04× floor) |
| 22 | 6074 (1.0×; 1.62× floor) | 19357 (3.2×; 5.18× floor) | 46556 (7.7×; 12.45× floor) | 68856 (11.3×; 18.42× floor) | 115771 (19.1×; 30.97× floor) |
| 24 | 25278 (1.0×; 1.69× floor) | 89245 (3.5×; 5.97× floor) | 189937 (7.5×; 12.70× floor) | 278068 (11.0×; 18.60× floor) | 460843 (18.2×; 30.82× floor) |

**pass_dense, 22 threads** — median µs per pass (× the k = 1 pass of the same size; × the bandwidth floor)

| n | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| 14 | 20 (1.0×; 1.38× floor) | 44 (2.2×; 3.04× floor) | 173 (8.6×; 11.82× floor) | 265 (13.2×; 18.18× floor) | 431 (21.5×; 29.53× floor) |
| 16 | 83 (1.0×; 1.41× floor) | 193 (2.3×; 3.30× floor) | 656 (8.0×; 11.24× floor) | 1008 (12.2×; 17.25× floor) | 1747 (21.2×; 29.91× floor) |
| 18 | 159 (1.0×; 0.68× floor) | 236 (1.5×; 1.01× floor) | 2659 (16.7×; 11.38× floor) | 4140 (26.1×; 17.72× floor) | 7403 (46.6×; 31.68× floor) |
| 20 | 504 (1.0×; 0.54× floor) | 938 (1.9×; 1.00× floor) | 2350 (4.7×; 2.51× floor) | 3116 (6.2×; 3.33× floor) | 28654 (56.8×; 30.66× floor) |
| 22 | 1884 (1.0×; 0.50× floor) | 2852 (1.5×; 0.76× floor) | 8184 (4.3×; 2.19× floor) | 11274 (6.0×; 3.02× floor) | 17673 (9.4×; 4.73× floor) |
| 24 | 8376 (1.0×; 0.56× floor) | 10807 (1.3×; 0.72× floor) | 29454 (3.5×; 1.97× floor) | 44076 (5.3×; 2.95× floor) | 69552 (8.3×; 4.65× floor) |

**pass_diagonal, 1 thread** — median µs per pass (× the k = 1 pass of the same size; × the bandwidth floor)

| n | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| 14 | 11 (1.0×; 0.74× floor) | 30 (2.8×; 2.07× floor) | 41 (3.8×; 2.80× floor) | 53 (4.8×; 3.60× floor) | 66 (6.0×; 4.49× floor) |
| 16 | 43 (1.0×; 0.74× floor) | 122 (2.8×; 2.09× floor) | 162 (3.8×; 2.78× floor) | 208 (4.8×; 3.57× floor) | 263 (6.1×; 4.50× floor) |
| 18 | 188 (1.0×; 0.80× floor) | 529 (2.8×; 2.26× floor) | 713 (3.8×; 3.05× floor) | 848 (4.5×; 3.63× floor) | 1051 (5.6×; 4.50× floor) |
| 20 | 1410 (1.0×; 1.51× floor) | 2906 (2.1×; 3.11× floor) | 2988 (2.1×; 3.20× floor) | 3486 (2.5×; 3.73× floor) | 4347 (3.1×; 4.65× floor) |
| 22 | 5292 (1.0×; 1.42× floor) | 12396 (2.3×; 3.32× floor) | 13240 (2.5×; 3.54× floor) | 15230 (2.9×; 4.07× floor) | 17567 (3.3×; 4.70× floor) |
| 24 | 21218 (1.0×; 1.42× floor) | 51529 (2.4×; 3.45× floor) | 52683 (2.5×; 3.52× floor) | 61896 (2.9×; 4.14× floor) | 75717 (3.6×; 5.06× floor) |

**pass_diagonal, 22 threads** — median µs per pass (× the k = 1 pass of the same size; × the bandwidth floor)

| n | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| 14 | 11 (1.0×; 0.73× floor) | 30 (2.8×; 2.05× floor) | 41 (3.8×; 2.78× floor) | 52 (5.0×; 3.60× floor) | 65 (6.2×; 4.47× floor) |
| 16 | 42 (1.0×; 0.71× floor) | 122 (2.9×; 2.09× floor) | 163 (3.9×; 2.79× floor) | 210 (5.0×; 3.59× floor) | 258 (6.2×; 4.42× floor) |
| 18 | 114 (1.0×; 0.49× floor) | 176 (1.5×; 0.75× floor) | 679 (5.9×; 2.90× floor) | 858 (7.5×; 3.67× floor) | 1087 (9.5×; 4.65× floor) |
| 20 | 422 (1.0×; 0.45× floor) | 663 (1.6×; 0.71× floor) | 631 (1.5×; 0.67× floor) | 731 (1.7×; 0.78× floor) | 4219 (10.0×; 4.51× floor) |
| 22 | 1663 (1.0×; 0.44× floor) | 1872 (1.1×; 0.50× floor) | 2117 (1.3×; 0.57× floor) | 2362 (1.4×; 0.63× floor) | 2672 (1.6×; 0.71× floor) |
| 24 | 8215 (1.0×; 0.55× floor) | 9011 (1.1×; 0.60× floor) | 9762 (1.2×; 0.65× floor) | 10554 (1.3×; 0.71× floor) | 11463 (1.4×; 0.77× floor) |

#### 17.5.3 The brickwork circuit, gate by gate and in fused blocks

10 layers of a U on every qubit and CX on a random perfect matching;
window 1 is the engine's own gate-by-gate sweep, windows 2–5 are the blocks
the 17.2 grouping rule forms with no cost rule (every gate that fits
joins). Median of 3 timed runs after a warm-up; the single-thread rows stop
at 20 qubits.

**Linux**

| n | threads | window | gates | passes | median s | min..max s | speed-up | max abs diff |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 1 | 240 | 240 | 0.021 | 0.020..0.022 | 1.00 | 0.0e+00 |
| 16 | 1 | 2 | 240 | 83 | 0.017 | 0.017..0.017 | 1.26 | 1.8e-17 |
| 16 | 1 | 3 | 240 | 46 | 0.027 | 0.027..0.027 | 0.78 | 1.7e-17 |
| 16 | 1 | 4 | 240 | 40 | 0.042 | 0.041..0.043 | 0.50 | 2.1e-17 |
| 16 | 1 | 5 | 240 | 28 | 0.050 | 0.049..0.050 | 0.43 | 2.3e-17 |
| 16 | 22 | 1 | 240 | 240 | 0.021 | 0.021..0.021 | 1.00 | 0.0e+00 |
| 16 | 22 | 2 | 240 | 83 | 0.017 | 0.017..0.019 | 1.19 | 1.8e-17 |
| 16 | 22 | 3 | 240 | 46 | 0.027 | 0.027..0.028 | 0.75 | 1.7e-17 |
| 16 | 22 | 4 | 240 | 40 | 0.042 | 0.042..0.045 | 0.49 | 2.1e-17 |
| 16 | 22 | 5 | 240 | 28 | 0.047 | 0.047..0.052 | 0.44 | 2.3e-17 |
| 20 | 1 | 1 | 300 | 300 | 0.512 | 0.510..0.524 | 1.00 | 0.0e+00 |
| 20 | 1 | 2 | 300 | 107 | 0.380 | 0.371..0.382 | 1.35 | 5.9e-18 |
| 20 | 1 | 3 | 300 | 60 | 0.602 | 0.600..0.603 | 0.85 | 6.0e-18 |
| 20 | 1 | 4 | 300 | 48 | 0.861 | 0.844..0.863 | 0.59 | 6.1e-18 |
| 20 | 1 | 5 | 300 | 34 | 0.994 | 0.990..1.009 | 0.52 | 6.4e-18 |
| 20 | 22 | 1 | 300 | 300 | 0.159 | 0.156..0.165 | 1.00 | 0.0e+00 |
| 20 | 22 | 2 | 300 | 107 | 0.078 | 0.075..0.083 | 2.03 | 5.9e-18 |
| 20 | 22 | 3 | 300 | 60 | 0.108 | 0.105..0.109 | 1.47 | 6.0e-18 |
| 20 | 22 | 4 | 300 | 48 | 0.149 | 0.146..0.170 | 1.07 | 6.1e-18 |
| 20 | 22 | 5 | 300 | 34 | 0.929 | 0.888..0.929 | 0.17 | 6.4e-18 |
| 22 | 22 | 1 | 330 | 330 | 0.680 | 0.678..0.698 | 1.00 | 0.0e+00 |
| 22 | 22 | 2 | 330 | 117 | 0.326 | 0.312..0.338 | 2.08 | 3.3e-18 |
| 22 | 22 | 3 | 330 | 67 | 0.454 | 0.433..0.463 | 1.50 | 3.3e-18 |
| 22 | 22 | 4 | 330 | 53 | 0.630 | 0.615..0.634 | 1.08 | 3.5e-18 |
| 22 | 22 | 5 | 330 | 37 | 0.731 | 0.711..0.751 | 0.93 | 3.5e-18 |
| 24 | 22 | 1 | 360 | 360 | 3.114 | 3.108..3.128 | 1.00 | 0.0e+00 |
| 24 | 22 | 2 | 360 | 127 | 1.274 | 1.264..1.286 | 2.44 | 1.8e-18 |
| 24 | 22 | 3 | 360 | 75 | 1.775 | 1.770..1.784 | 1.75 | 2.2e-18 |
| 24 | 22 | 4 | 360 | 59 | 2.901 | 2.672..2.903 | 1.07 | 1.8e-18 |
| 24 | 22 | 5 | 360 | 42 | 3.565 | 3.559..3.710 | 0.87 | 2.0e-18 |

**Windows**

| n | threads | window | gates | passes | median s | min..max s | speed-up | max abs diff |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 1 | 240 | 240 | 0.029 | 0.028..0.029 | 1.00 | 0.0e+00 |
| 16 | 1 | 2 | 240 | 88 | 0.016 | 0.016..0.016 | 1.81 | 1.8e-17 |
| 16 | 1 | 3 | 240 | 50 | 0.028 | 0.028..0.029 | 1.03 | 1.9e-17 |
| 16 | 1 | 4 | 240 | 39 | 0.039 | 0.038..0.039 | 0.75 | 2.2e-17 |
| 16 | 1 | 5 | 240 | 29 | 0.049 | 0.048..0.050 | 0.59 | 2.1e-17 |
| 16 | 22 | 1 | 240 | 240 | 0.029 | 0.028..0.029 | 1.00 | 0.0e+00 |
| 16 | 22 | 2 | 240 | 88 | 0.016 | 0.016..0.018 | 1.84 | 1.8e-17 |
| 16 | 22 | 3 | 240 | 50 | 0.028 | 0.028..0.029 | 1.05 | 1.9e-17 |
| 16 | 22 | 4 | 240 | 39 | 0.040 | 0.038..0.041 | 0.73 | 2.2e-17 |
| 16 | 22 | 5 | 240 | 29 | 0.049 | 0.048..0.049 | 0.60 | 2.1e-17 |
| 20 | 1 | 1 | 300 | 300 | 0.646 | 0.641..0.705 | 1.00 | 0.0e+00 |
| 20 | 1 | 2 | 300 | 106 | 0.331 | 0.329..0.334 | 1.95 | 6.4e-18 |
| 20 | 1 | 3 | 300 | 62 | 0.581 | 0.576..0.595 | 1.11 | 5.7e-18 |
| 20 | 1 | 4 | 300 | 49 | 0.813 | 0.809..0.825 | 0.79 | 6.2e-18 |
| 20 | 1 | 5 | 300 | 37 | 0.971 | 0.967..0.986 | 0.67 | 7.2e-18 |
| 20 | 22 | 1 | 300 | 300 | 0.127 | 0.125..0.128 | 1.00 | 0.0e+00 |
| 20 | 22 | 2 | 300 | 106 | 0.072 | 0.070..0.072 | 1.78 | 6.4e-18 |
| 20 | 22 | 3 | 300 | 62 | 0.113 | 0.110..0.119 | 1.13 | 5.7e-18 |
| 20 | 22 | 4 | 300 | 49 | 0.154 | 0.153..0.157 | 0.82 | 6.2e-18 |
| 20 | 22 | 5 | 300 | 37 | 0.912 | 0.910..0.920 | 0.14 | 7.2e-18 |
| 22 | 22 | 1 | 330 | 330 | 0.677 | 0.674..0.715 | 1.00 | 0.0e+00 |
| 22 | 22 | 2 | 330 | 115 | 0.305 | 0.295..0.308 | 2.22 | 3.1e-18 |
| 22 | 22 | 3 | 330 | 64 | 0.476 | 0.470..0.490 | 1.42 | 3.1e-18 |
| 22 | 22 | 4 | 330 | 51 | 0.614 | 0.601..0.628 | 1.10 | 3.3e-18 |
| 22 | 22 | 5 | 330 | 37 | 0.693 | 0.676..0.697 | 0.98 | 3.7e-18 |
| 24 | 22 | 1 | 360 | 360 | 3.167 | 3.161..3.188 | 1.00 | 0.0e+00 |
| 24 | 22 | 2 | 360 | 127 | 1.272 | 1.263..1.304 | 2.49 | 1.8e-18 |
| 24 | 22 | 3 | 360 | 75 | 1.953 | 1.852..2.075 | 1.62 | 1.8e-18 |
| 24 | 22 | 4 | 360 | 59 | 2.816 | 2.785..2.823 | 1.12 | 1.8e-18 |
| 24 | 22 | 5 | 360 | 43 | 3.335 | 3.332..3.423 | 0.95 | 2.0e-18 |

#### 17.5.4 What the numbers say — [DERIVED]

- **Passes are the bound only on every thread at 20 qubits and above, and
  only for 1- and 2-qubit matrices.** There a 2-qubit pass costs 1.2–1.8×
  a 1-qubit pass on Linux and 1.3–1.9× on Windows (both under the
  single-thread copy's floor), so two passes merged into one 2-qubit pass
  is a gain. A 3-qubit pass costs 3.5–4.7×, a 4-qubit pass 5.3–7× and a
  5-qubit pass 8–10× on both platforms (the n = 20, k = 5 rows' 55–57× is
  the same anomaly on both platforms, a 5-qubit gather whose operands
  include qubit 19 and qubit 0, and is recorded as measured).
- **On one thread the arithmetic is the bound at every size.** On Linux a
  2-qubit pass costs 8.6–9.6× a 1-qubit pass at 14–18 qubits and 3.6–5.2×
  at 20–24, against the 2× its arithmetic would predict; a 3-qubit pass
  12–30×; a 5-qubit pass 33–82×. On Windows the ratios are 2.1–3.5×,
  3.9–8.4× and 9–22×, not because its 2-qubit pass is faster (it is within
  10% of Linux's) but because its 1-qubit pass is 2–3.5× slower: the
  1-qubit kernel is a unit-stride loop with the complex multiply written
  out, which GCC vectorises through its AVX2 clone and MSVC compiles
  scalar, while the 2-qubit and generic kernels gather per group and
  multiply through `std::complex::operator*` on both. That gap is item
  8's, not item 7's; it is what sets the window today.
- **Diagonal passes are cheap at every width.** On every thread at 22–24
  qubits a 5-qubit diagonal pass costs 1.2–1.6× a 1-qubit one on both
  platforms; on one thread 2.8–3.9× at 22–24 (Linux) and 3.3–3.6×
  (Windows). The streaming diagonal kernel is bandwidth-bound where the
  state is out of cache, which is what the Phase 24 diagonal fusion relies
  on.
- **Window 2 wins everywhere measured; window 3 wins only on every thread;
  windows 4 and 5 lose or break even.** Window 2: 2.0–2.5× on every thread
  at 20–24 qubits on both platforms, and on one thread 1.19–1.35× (Linux)
  and 1.8–1.95× (Windows) at 16–20. Window 3: 1.4–1.75× on every thread at
  22–24, 0.78–1.11× on one thread. Window 4: 1.07–1.12× on every thread,
  0.5–0.8× on one thread. Window 5: at best 0.98×.
- **The grouping must not depend on the thread count,** because every
  thread count must give the same bytes (Phase 15), and the fused matrices
  decide the rounding. The rule is therefore set by the single-thread
  costs, the conservative side: with them a 2-qubit pass costs 2.1–9.6
  1-qubit passes, so two 1-qubit gates on different qubits must **not** be
  merged into a 2-qubit block (two passes of cost 1 become one of cost 2–10),
  while a 1-qubit gate on a qubit of a 2-qubit block, or two 2-qubit gates on
  the same pair, always should (the block's pass is paid once instead of
  twice). The window-2 rows above include the losing 1+1 merges and still
  gain; the rule proposed in 17.6 keeps only the winning ones.
- **Rounding.** The fused and unfused final states differ by at most
  2.3 × 10⁻¹⁷ in any amplitude on either platform, 5 orders of magnitude
  inside `common::kDefaultAbsTol`.

#### 17.5.5 What step 1 recommends for 17.4 — [MENOS]

- **Q1, the window: 2 for dense blocks, 5 for diagonal blocks**, with the
  containment rule: a gate joins a block only if its qubit set contains the
  block's or is contained in it (so the fused block has the operand count
  of the wider of the two and the pass count falls by one), and a block
  whose gates are all diagonal may instead grow to 5 qubits as
  `fuse_diagonal_gates` does today. No per-k cost table is needed under
  this rule: it is the structural reading of the single-thread measurement
  (c(2) > 2 c(1), c_d(k) ≈ c_d(1) on every thread), and it never reads the
  machine. Item 8, which changes c(2) and c(3), re-measures and may widen
  the window; the probe is kept for that.
- **Q2, the default: on, at every size.** With the containment rule every
  merge removes a pass and never adds arithmetic, so there is no size below
  which it loses; n_f is not needed. The measured gain is 1.2–1.95× on one
  thread and 2.0–2.5× on every thread at 20–24 qubits on this circuit
  family, before item 8.
- **Q3–Q6:** as proposed in 17.4.

### 17.6 Owner answers (2026-11-09)

The owner took every recommendation of 17.4 and 17.5.5 ("let's go with the
recommendation"), with four conditions as the acceptance bar: numerical
correctness, memory accounting, cross-engine consistency and performance
predictability are not compromised. The answers, and one refinement made
while implementing, are:

- **Q1, the window: containment, for dense and diagonal blocks alike.** A
  gate joins the newest open block that shares a qubit with it only when
  one of the two qubit sets contains the other; a dense gate never joins a
  strictly wider diagonal block, and a dense block never takes a strictly
  wider diagonal gate. **Refinement of 17.5.5's "5 for diagonal blocks":**
  the single-thread rows of 17.5.1–17.5.2 price a 2-qubit diagonal pass at
  2.1–7× a 1-qubit one (17.5.4), so merging two diagonal gates on disjoint
  qubits is a loss on one thread, exactly as it is for dense gates, and the
  grouping must not depend on the thread count. The containment rule is
  therefore applied to both classes, and the union of diagonal gates on
  disjoint qubits stays what it was before this item: the opt-in
  `RunOptions::fuse_diagonal_gates`, unchanged, which a QAOA user on many
  threads turns on knowingly. This is the "performance predictability"
  condition applied to the one case where the threaded and single-thread
  measurements disagree.
- **Q2, the default: on, at every size.** `RunOptions::fuse_gates = true`;
  `false` applies every gate on its own, with results and `explain()`
  byte-identical to a run from before this item (pinned by the Phase 21
  fixtures, the golden tests that set it, and `BackendRequestTest`).
- **Q3, the Phase 21 baseline: kept, with fused copies beside it.** The
  fixtures of `tests/memory/memory_fixtures.cpp` run with fusion off against
  their unchanged entries in `baseline_{linux,windows}.json`, so the unfused
  kernels stay pinned to the bytes of `1f89192`; every dense, CUDA, disk-tier
  and whole-job fixture has a `_fused` copy, run with fusion on, whose
  outputs and accounting are pinned by new entries in the same files,
  generated once per platform. The regenerated files keep every existing
  entry byte for byte (checked at generation on both platforms).
- **Q4, which engines: the CPU dense engine, the CUDA dense engine and the
  disk tier,** through the one `ir::plan_fusion` and the one
  `engines::dense::fused_block_matrix`, so CPU/CUDA and in-memory/disk stay
  bitwise identical with fusion on. The density-matrix and trajectory
  engines are untouched.
- **Q5, the cost model: the fused circuit is charged.** Every dense
  estimate counts one pass per block, carries the fusion plan and the
  largest block's transient in its peak bytes, and reports the outcome;
  the golden corpus changes are listed in 17.7.
- **Q6, the gate:** the seven criteria of 17.4, with the speed-up target of
  criterion 4 read from 17.5.3: at 20–24 qubits on every thread the fused
  brickwork sweep must be at least 2.0× the unfused one on both platforms
  (the window-2 rows, which include the 1 + 1 merges the containment rule
  refuses, gave 2.0–2.5×), measured as same-run ratios.

### 17.7 Implementation and gate evidence — [MEASURED]

**Code** (`0a26af3` on `phase-24-dense-fusion`, after the kickoff `fcbf947`
and the step-1 record `6d7dd65`):

| Where | What |
|---|---|
| `src/ir/fuse_gates.{h,cpp}` | `ir::plan_fusion`: the containment rule over a 64-bit qubit mask per open block, with the newest block per qubit tracked in a 64-entry table so the plan is O(gates); the plan is four exactly-sized vectors (`order`, `block_begin`, `block_qubits`, `qubits_begin`) with `heap_bytes()` and `construction_peak_bytes()` for the estimates |
| `src/engines/dense/fused_block.{h,cpp}` | `fused_block_matrix` (the generic kernel on the block's basis vectors, one thread, transposed in place), `block_transient_bytes`, `max_block_transient_bytes`, and the `apply_blocks` template the three dense engines sweep with |
| `src/engines/dense/dense_engine.cpp` | the plan built before the state, the sweep over blocks, `dense_work_units(request, plan)`, the peak as `max(construction, state + plan + max(block transient, outputs))`; with the opt-in `fuse_diagonal_gates` the plan is built over the diagonal-fused list inside `execute()`, which the estimate does not charge, as before this item |
| `src/engines/dense_cuda/cuda_dense_engine.cpp` | the same plan and host-built matrices applied with the CUDA kernel; launches and the host peak count blocks |
| `src/engines/dense_tiered/{pass_plan,chunked_executor,disk_dense_engine}` | the pass plan over the plan's blocks, each pass applying blocks through the window; the layout carries the plan through the run |
| `include/menos/{menos,plan,simulation_engine}.h`, `src/plan/job_spec.h` | `RunOptions::fuse_gates`, `FusionSummary` on `CostEstimate` and `Region`, `fuse_gates` on both engine requests (default on), `JobSpec::fuse_gates` |
| `src/plan/{planner,region_builder,plan_cache,plan_impl}.cpp`, `src/cost/cost_model.{h,cpp}` | the flag threaded to every dense estimate, the summary copied into regions, the plan-cache key, the region line `fusion=containment blocks= absorbed=` printed only when a gate was absorbed |
| `src/api/menos.cpp`, `src/runtime/runtime.cpp`, `src/python/bindings.cpp`, `cli/cli_impl.cpp` | the option carried to the planner and the engines, `RunOptions.fuse_gates`, `--fusion on\|off` |

**Tests added or changed.**
- `tests/ir/fuse_gates_test.cpp`: the rule block by block on hand-built circuits (containment, the diagonal refusals, a diagonal `UnitaryOp`, the barrier, the empty list, the 64-qubit refusal); on random circuits of 1–3-qubit gates, diagonal gates and 2-qubit unitaries, the plan is a permutation into blocks whose operands are their gates' union, operations keep program order within a block, and applying the blocks in plan order equals program order within 1e-12 (the oracle of `fuse_diagonal_test.cpp`).
- `tests/engines/dense/fused_block_test.cpp`: a block's matrix is the product of its gates (oracle, 1e-12); the block sweep equals the gate sweep within 1e-12; the fused sweep is **bitwise identical on 1, 2, 3 and every hardware thread** at 19 qubits; a singleton block is applied with the gate's own matrix and operand order (`cx 3,1`), bitwise the direct call; the monitor sees one boundary per block; the transient bound covers every block.
- `tests/engines/dense/dense_engine_test.cpp`: the fused Bell is one pass (work 4 + 16 + 4) and its peak is the state, the plan and the block's transient; fusion off charges exactly the per-gate formula and reports no fusion.
- `tests/plan/region_builder_test.cpp`: with fusion, three gates on one qubit are one dense pass (work 8) and dense wins over MPS (14); the per-gate case keeps its MPS routing with fusion off.
- `tests/api/execution_monitor_test.cpp`: once per block with fusion, once per operation without.
- `tests/engines/dense_tiered/disk_dense_engine_test.cpp`: the bitwise comparison of the disk tier with the in-memory engine runs with fusion on (the default) at every chunk size, gate shape and thread count, and the structural tests pin the gate-by-gate pass plan with fusion off.
- `tests/memory`: every dense, CUDA, disk-tier and whole-job fixture has a `_fused` copy (12 on a CPU build, 16 with CUDA); the Windows baseline regenerated keeps all 37 existing entries byte for byte and adds the 12; the fused disk-tier fixtures' counts and expectation digests equal the fused in-memory fixtures' (`9babf4d634cd516f`, `3de7c9f686df57fd`), and the fused statevector digest differs from the unfused one, as rounding says it must.
- The CLI and `explain()` goldens that print the Bell's dense candidate now say `work_units=24` (one fused pass) instead of 32; the tests tuned to the per-gate figures (`cost_model_test`, `planner_test`, the 17 000-byte budget cases) set fusion off and say so.
- The planner corpus: 35 of 45 golden plans change on Windows (every dense candidate's `work_units` and `peak_bytes`, the region line's fusion field). **One winner changes:** `counts_factorized.small`, `direct_mps` → `factorized`, because its dense regions, fused, are now cheaper than the MPS candidate; every other job keeps its strategy, regions and refusals.

**Linux.** The planner corpus regenerated in the WSL clone changes the same 35 of 45 plans, with the same one winner change (`counts_factorized.small`, `direct_mps` → `factorized`). The Linux baseline regenerated from the CPU release build keeps all 37 CPU entries byte for byte and adds the 12 fused ones; the cuda12 release build, on the GPU, keeps the 6 CUDA entries and adds the 5 CUDA fused ones. **With fusion on, the CUDA engine's statevector digest equals the CPU engine's (`17358dda1e64619a`) and so do the counts (`9babf4d634cd516f`)**, and the disk tier's fused counts and expectation digests equal the in-memory engine's (`9babf4d634cd516f`, `624344e0d00db372`): the three identities of 17.2 hold on the fused path, as the tests require. The fused statevector differs from the unfused one (`fdd860ae09c0ae37`), the rounding of 17.5.

**Criterion 4, the speed-up** (the probe's `run` rows: the brickwork family through `menos::run`, `execute_seconds` with `fuse_gates` on and off, median of 3 after a warm-up, same run):

**Linux (WSL, GCC 15.2, `build/bench`, `8715d5d`, 2026-11-09 11:19–11:21 UTC, mains power, load average 1.29 at the start)**

| n | threads | gates | fused blocks | absorbed | unfused s (min..max) | fused s (min..max) | speed-up | max abs diff |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 240 | 66 | 158 | 0.023 (0.023..0.023) | 0.016 (0.016..0.017) | 1.43 | 2.2e-17 |
| 16 | 22 | 240 | 66 | 158 | 0.021 (0.020..0.021) | 0.015 (0.015..0.015) | 1.33 | 2.2e-17 |
| 20 | 1 | 300 | 86 | 194 | 0.510 (0.498..0.511) | 0.351 (0.350..0.360) | 1.45 | 5.8e-18 |
| 20 | 22 | 300 | 86 | 194 | 0.159 (0.158..0.165) | 0.074 (0.072..0.082) | 2.14 | 5.8e-18 |
| 22 | 22 | 330 | 93 | 215 | 0.710 (0.692..0.711) | 0.335 (0.316..0.337) | 2.12 | 3.4e-18 |
| 24 | 22 | 360 | 104 | 234 | 3.206 (3.187..3.293) | 1.296 (1.281..1.300) | 2.47 | 1.8e-18 |

**Windows (MSVC 14.50, `build/windows-bench`, `8715d5d`, 10:49–10:54 UTC, mains power)**

| n | threads | gates | fused blocks | absorbed | unfused s (min..max) | fused s (min..max) | speed-up | max abs diff |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 240 | 68 | 156 | 0.127 (0.123..0.135) | 0.062 (0.056..0.068) | 2.06 | 2.0e-17 |
| 16 | 22 | 240 | 68 | 156 | 0.133 (0.128..0.139) | 0.063 (0.044..0.080) | 2.13 | 2.0e-17 |
| 20 | 1 | 300 | 84 | 196 | 2.402 (2.288..2.502) | 1.194 (1.095..1.195) | 2.01 | 7.0e-18 |
| 20 | 22 | 300 | 84 | 196 | 0.357 (0.320..0.367) | 0.159 (0.157..0.164) | 2.25 | 7.0e-18 |
| 22 | 22 | 330 | 96 | 212 | 1.112 (1.070..1.114) | 0.505 (0.504..0.553) | 2.20 | 3.2e-18 |
| 24 | 22 | 360 | 104 | 232 | 4.577 (4.504..4.661) | 2.053 (1.979..2.079) | 2.23 | 1.7e-18 |

The raw records are `docs/benchmarks/phase-24/fusion-gate/{linux,windows}/`, the full probe (the step-1 sections re-run on the same build, then the `run` rows). On every thread at 20–24 qubits the fused engine is **2.12–2.47× (Linux) and 2.20–2.25× (Windows)** the unfused one, above the 2.0× the owner set from 17.5; on one thread it is 1.43–1.45× on Linux (whose 1-qubit kernel is vectorised, so a merged 1-qubit pass saved less) and 2.01–2.06× on Windows. The brickwork circuits fuse to about 29% of their gate count (for example 360 gates into 104 fused blocks and 22 singleton passes at 24 qubits).

**Criterion 1, the rounding:** the largest amplitude difference between the fused and unfused statevectors in those runs is 2.2 × 10⁻¹⁷, inside `common::kDefaultAbsTol` by five orders of magnitude; the accuracy benchmark's dense workloads and the Phase 21 fixtures agree with the unfused run within tolerance by the tests above.

**Criterion 3, fusion off:** `BackendRequestTest.AnUnsetBackendLeavesTheReportByteIdentical` and the golden tests with `fuse_gates=false` print the report of before this item; the Phase 21 fixtures with fusion off keep their bytes.

**Criterion 5:** `explain()` prints the outcome on the region line (tests above) and `DenseEngineTest.FusedBellIsOnePassAndChargesThePlanAndTheBlock` equates the estimate with the plan's sum.

**Criterion 6:** the one changed winner is listed above with its reason.

**Criterion 7:** recorded in 17.8.

### 17.8 Verification of the fusion branch — [MEASURED]

Final commit: `195fc0b` on `phase-24-dense-fusion`. It contains item 7, the section 18 stabilizer fix, and two commits written for this verification. Neither changes engine behaviour:
- **`8c9bdd5`** fixes two clang-tidy findings in `benchmarks/dense_fusion_probe.cpp`: a `NOLINT` for swappable parameters on `brickwork_circuit`, and a checked statevector optional in place of an unchecked dereference. The first `scripts/verify.sh` run, on `08e35f3`, failed at the benchmark clang-tidy stage on them.
- **`195fc0b`** updates `CudaExecutionMonitorTest`. With fusion the CUDA engine reports to the monitor once per fused block, as the CPU engine does, but the GPU-only test still expected one call per operation (38 calls; the engine made 15). The test was renamed `TheCudaEngineCallsTheMonitorOncePerBlockOrOncePerOperation` and checks both modes, mirroring `ExecutionMonitorTest.DenseCallsTheMonitorOncePerBlockOrOncePerOperation`. The second run, on `8c9bdd5`, failed on it in `cuda12-dev`, the only presets that reach the GPU on this machine. Every earlier stage of that run passed.

A third run, on `195fc0b`, stopped at the Python editable install because pypi.org timed out three times after the machine woke from sleep, a network failure. Every C++ stage before it passed. The run below is a fresh, complete run on the same commit.

| Script | Where | Result |
|---|---|---|
| `scripts/verify.sh` | WSL clone `~/MENOS`, `UV_PYTHON_PREFERENCE=only-managed`, 2026-11-10 07:24–09:37 UTC | exit 0, "all stages passed": dev and release 917/917; Python 576 passed (2 deselected); wheel smoke 2 passed; `cuda-dev`, `cuda-release`, `cuda12-dev` and `cuda12-release` 1007/1007 each, the `cuda12` presets with the GPU tests executing; the docs stage skipped (no Node in the clone) |
| `scripts/verify-windows.ps1` | native Windows, detached worktree `C:/Users/janak/menos-verify`, run after the WSL run, never alongside it | exit 0, "verify-windows: all stages passed": `windows-dev` (ASan) 918/918, `windows-release` 918/918, pytest 576 passed (2 deselected), wheel smoke 2 passed |

The `cuda12` stages run the fused CUDA engine against the Phase 21 baseline's `dense_cuda_*_fused` entries. That is the cross-engine check of criterion 1 on the device: the fused CUDA statevector digest `17358dda1e64619a` equals the CPU one.

### 17.9 Gate (2026-11-10)

**The owner passed the item 7 gate on 2026-11-10** ("item 7 gate passed") on the evidence in 17.7 and 17.8 (final commit `195fc0b`, records in `adab80d`). In the same message the owner started item 8 (section 19).

## 18. Stabilizer sampler fix (owner addition, 2026-11-09) — [MEASURED]

**The symptom.** On some Clifford circuits the stabilizer engine's counts
contained outcomes of probability zero: a qubit whose exact marginal is
0.5 came out 1 in every shot. The owner's harness found it on 2026-11-05
at 40–100 qubits (`docs/benchmarks/cross-simulator-2026-11-05/harness/
stabilizer_repro.{py,json}`: a 50-qubit, 54-gate circuit leaving qubits 48
and 49 perfectly anticorrelated, MENOS reporting qubit 48 as 1 always), and
`examples/compare_simulators.ipynb` showed it from 8 qubits up on
2026-11-07. Measuring one qubit alone was right, and so was every
expectation value, so the tableau was right. The owner placed the fix in
this phase on 2026-11-09.

**The cause.** `StabilizerCountsSampler::sample_one`
(`src/engines/stabilizer/stabilizer_engine.cpp`) builds each shot as the
affine subspace's base outcome combined with a random subset of its
generators. The combination was `outcome |= generator`, an OR: a base bit
that is 1 could never be flipped to 0 by a generator, so every outcome in
which such a flip should have happened came out with the bit still set.
The algorithm of `docs/adr/0006` says XOR. `BitString` had only
`operator|=`, written for the factorized-counts combination of disjoint
bit ranges, where OR and XOR coincide.

**Why the tests missed it.** `StabilizerEngineDifferentialTest` enumerated
the reachable outcomes from `Tableau::reduce()` with its own XOR and
compared that set with the dense oracle; it never drew from the engine's
sampler. Small circuits rarely have a base bit set where a generator also
has one, so the Bell and random-Clifford tests saw nothing.

**The fix** (`phase-24-dense-fusion`): `BitString::operator^=`
(`include/menos/bitstring.h`, `src/common/bitstring.cpp`), and the sampler
flipping with it. No other code changes.

**Tests.** `BitStringTest.XorFlipsBitsInBothDirections` (across a word
boundary, an involution) and the mismatched-width refusal;
`StabilizerEngineTest.AnAntiCorrelatedPairSamplesOnlyItsTwoOutcomes`
(X(0), H(1), CX(1,0): base bit 1 on qubit 0, one generator flipping both,
the minimal case); `TheOwnersFiftyQubitReproducerSamplesOnlyAnticorrelated
Outcomes` (the 54 gates of the reproducer, measuring 48 and 49: only `01`
and `10`, both appearing); and the differential test now also draws 256
shots from the engine's sampler per circuit and requires every drawn
outcome to have a nonzero dense probability.

**What changes for users.** Counts from `direct_stabilizer` plans change
wherever the bug fired; nothing else does. The two Phase 21 stabilizer
counts fixtures (`stabilizer_counts`, `stabilizer_region_counts`, 14-qubit
random Clifford circuits) were among them: their baseline entries had
pinned the buggy sampler's outcomes (`1df8588d28fcdb82`,
`6082b83bf9901fa7`, the same on both platforms) and are regenerated
(`5ae6b76eb83ede7f`, `9d44002984b827c1`, again the same on both platforms,
the sampler being integer arithmetic over the same seed; the Windows probe
generated them and the Linux verification of the fusion head confirms
them). The stabilizer expectation fixture and every other baseline entry
are unchanged, and so are the planner, the estimates and `explain()`.

## 19. Kickoff: item 8, SIMD kernels for the dense CPU engine — [MENOS] / [OPEN]

The owner started item 8 on 2026-11-10, after passing item 7's gate. Branch `phase-24-simd-kernels`, from `phase-24-dense-fusion` at `adab80d`. Its scope is the BUILD_PLAN's Phase 24 item 8 text. This section records what the kernels do today, the proposed design, the step-1 method and the owner's questions. Nothing is built before the answers.

### 19.1 What the code does today — [MENOS], read from the source

All CPU gate kernels are in `src/kernels/cpu/dense_kernels.cpp`. The target is compiled for baseline x86-64 (SSE2), with `-ffp-contract=off` on GCC/Clang and `/fp:precise` on MSVC (`src/kernels/CMakeLists.txt`). There are no intrinsics anywhere.

- **One qubit, non-diagonal** (`apply_matrix_1q`): unit-stride runs of pairs (`pair_run`), or whole blocks for operands 0–2 (`pair_blocks<Q>`). Each output is `from_zero(cmul(m_r0, a0)) + cmul(m_r1, a1)`, the generic kernel's expression written out so the compiler can vectorise it.
- **Diagonal, any width** (`apply_diagonal`): one complex multiply per amplitude over unit-stride runs (`scale_run`), or element by element for low operands (`scale_elements`).
- **Two qubits, non-diagonal** (`apply_matrix_2q`): a scalar loop over groups of four. It gathers four amplitudes, forms each output as `sum += matrix[out*4+in] * a[in]` with `std::complex` `operator*`, and scatters them back.
- **Three qubits or more, non-diagonal** (`apply_matrix_generic`): the scalar oracle, which every other kernel must equal bit for bit (`tests/engines/dense/specialised_kernels_test.cpp`).

**The only vectorisation today is GCC's.** The four unit-stride loops (`pair_run`, `pair_blocks`, `scale_run`, `scale_elements`) carry `[[gnu::target_clones("avx2", "default")]]`. The loader picks the AVX2 clone where the CPU has it, and neither clone enables FMA. On MSVC and Clang the macro is empty, so those loops are whatever each compiler's auto-vectoriser makes of baseline SSE2 code. This explains the single-thread gap in 17.7: fusion gained 1.43–1.45× on Linux against 2.01–2.06× on Windows. Linux's 1-qubit passes were already AVX2, so merging them saved less. **The 2-qubit kernel is scalar on every platform.** Since item 7, most fused blocks are 2-qubit (window 2), so this is the kernel that now carries most of a circuit's passes.

**The cost model and calibration.** `plan::calibrated_hardware_profile` times the dense engine itself on an H/CX layer circuit (`measure_work_units_per_second`, `src/plan/hardware_profile.cpp`). A kernel speed-up therefore reaches the calibrated rate with no code change. What it does not reach is the per-k weighting: the cost model charges 2^(n+k) per block for every k. If item 8 speeds up the 1- and 2-qubit kernels but not the generic one, the relative prices of k = 1, 2 and k ≥ 3 blocks move. Unbudgeted CPU-only jobs never calibrate (`hardware_profile_for`), so their fixed rate does not move at all. Step 1 measures both effects. Whether either matters is a question for the model-fixes item, not this one.

**The machines.** The verification machine (Core Ultra 7 155H) has AVX2 and FMA and **no AVX-512**. That holds under WSL (`/proc/cpuinfo`) and natively. No machine in the verification set has been shown to have AVX-512.

### 19.2 Design proposed — [MENOS]

1. **Correctness rule (unchanged from the BUILD_PLAN).** Vectorise across amplitudes (several groups per instruction), never across a sum. Each lane performs the scalar code's multiplies and adds, in the scalar code's order, with no FMA. Every kernel stays bitwise identical to `apply_matrix_generic` on every thread count, and therefore to CUDA (`-fmad=false`), to the disk tier and to the Phase 21 baseline. `cmul`'s products-then-sums form is the operation to reproduce per lane. A complex multiply in a 256-bit register (two complex doubles) computes `ar*br`, `ai*bi`, `ar*bi`, `ai*br` with separate `_mm256_mul_pd` and combines them with `_mm256_addsub_pd`, or an explicit sub and add. That is the same two roundings per component as the scalar code. A `_mm256_fmaddsub_pd` would fuse them and is excluded.
2. **Dispatch, portable to MSVC.** One translation unit per ISA (`dense_kernels_avx2.cpp` beside the baseline code), compiled with `-mavx2` (no `-mfma`) on GCC/Clang and `/arch:AVX2` on MSVC, with contraction off in both. The ISA is chosen once at load time from `cpuid` (plus `xgetbv` for the OS's AVX state), through a function-pointer table. This replaces GCC's `target_clones`, which MSVC and Clang lack, so all three compilers run the same AVX2 code. The baseline path remains for any x86-64 without AVX2, and the wheels keep running on any x86-64. Whether MSVC's `/arch:AVX2` under `/fp:precise` ever emits an FMA is checked by the bitwise tests and by a disassembly grep for `vfmadd` in step 2, not assumed.
3. **Which kernels.** The 2-qubit kernel (the main gain: scalar today, and the bulk of fused passes), the 1-qubit kernel and the diagonal kernels (vectorised on GCC today, not on MSVC). `apply_matrix_generic` stays scalar: it is the oracle, and k ≥ 3 non-diagonal blocks arise only from 3-qubit and wider gates (item 7's window is 2). The expectation kernel (`expectation_of_term_accumulate`) is a sum, so vectorising it would change its summation order. It is out of scope, as the BUILD_PLAN's "never across a sum" requires.
4. **Testing both paths on one machine.** An internal, non-public override (a function in `kernels::cpu`, used only by tests and the probe) forces the baseline table. The bitwise tests then run every kernel on both paths, on every thread count, on the AVX2 machines that do the verification. Without it, the baseline path would go untested on this machine.
5. **Nothing above the kernel seam changes.** Same `apply_matrix` signature, plans, estimates and `explain()` output. No option is added, since there is no accuracy trade to opt into.

### 19.3 Step 1: the measurement before any SIMD code — [MENOS]

This runs on the gate machine, on the unchanged kernels, Linux (WSL, GCC 15.2, `build/bench`) and Windows (MSVC 14.50, `build/windows-bench`), on mains power. It is recorded in `docs/benchmarks/phase-24/simd-step1/` and driven by that directory's `scripts/step1.{sh,ps1}`. A new probe, `benchmarks/dense_simd_probe.cpp`, measures table A. `benchmarks/dense_fusion_probe.cpp` measures table B, through a new `runs` argument that runs its section 3 alone with the same circuits:

- **A. One pass by kernel.** At n = 16, 20, 22, 24, on one thread and on every thread: seconds per pass, and the ratio to the streaming floor from `measure_memory_bandwidth()`. Each operand placement selects a different code path:
  - 1q at operands 0, 2, 3, n/2 and n-1;
  - 2q at {0,1}, {1,0}, {0,n-1}, {n/2-1,n/2} and {n-2,n-1};
  - diagonal 1q at {0} and {n-1};
  - diagonal 2q at {0,1} and {n-2,n-1};
  - generic 3q at {0,n/2,n-1}. This answers which kernels are compute-bound (where SIMD helps) and which are already at memory bandwidth (where it cannot), per size and thread count.
- **B. The fused brickwork sweep through `menos::run`** (17.7's table, unchanged) is the end-to-end figure item 8 must move.
- **C. Linux with the clones off** (table A from `build/bench-noclones`, the same tree with `MENOS_KERNEL_CLONES` defined empty for that build only). This puts a number on what GCC's AVX2 clones give today. That is the best available estimate of the 1q/diagonal gain on MSVC, and of the 2q gain from the same treatment.

Each table A figure is the median, with min..max, of 10 timed passes after a discarded one (the Phase 15 repetition count). Each table B figure is the median of 3 runs after a discarded one, as in 17.7.

### 19.4 Owner questions — [OPEN]

- **Q1, which ISA paths.** Proposed: **baseline and AVX2 only**. AVX-512 is not built. No machine in the verification set has it, and an untested path would break "evidence before claims". It can be added later, with a machine to verify it on.
- **Q2, dispatch.** Proposed: **one AVX2 translation unit plus runtime `cpuid` dispatch on every compiler**, replacing GCC's `target_clones`. The alternative keeps `target_clones` on GCC and adds a separate MSVC path. That leaves two different AVX2 implementations to keep bitwise identical, and Clang still gets none.
- **Q3, intrinsics or auto-vectorisation.** Proposed: **hand-written AVX2 intrinsics for the 2-qubit kernel**, since no compiler vectorises its gather/scatter today, and **intrinsics for the 1-qubit and diagonal loops too**, so that MSVC, GCC and Clang execute the same instruction sequence and the bitwise identity does not depend on each compiler's vectoriser. The alternative, compiling the existing loops with `/arch:AVX2`, is less code but leaves the 2-qubit kernel scalar, and leaves the identity to three auto-vectorisers.
- **Q4, which kernels.** Proposed: 1q, 2q and diagonal. The generic kernel (the oracle) and the expectation sum stay scalar (19.2, item 3).
- **Q5, the fusion window.** 17.6 recorded that item 8 "re-measures and may widen the window". Proposed: **re-run the item 7 probe after item 8 and report it**, but keep the window at 2 unless the measurement shows a gain and the owner decides to change it. That would be a change to item 7's passed design, so it is the owner's decision, not this item's.
- **Q6, the gate.** Proposed criteria, with targets set from step 1 as for item 7:
  1. every vectorised kernel bitwise identical to `apply_matrix_generic` on both dispatch paths, on every thread count, on both platforms; the Phase 21 baselines (unfused and fused) and the golden corpus unchanged;
  2. CPU/CUDA and in-memory/disk identities unchanged (the `cuda12` and disk-tier tests);
  3. a single-thread speed-up of the 2-qubit kernel at 16–20 qubits, and a speed-up of the fused brickwork sweep through `menos::run` on one thread and on every thread at 20–24 qubits, on both platforms, each at a factor the owner sets from step 1's table A and the clones-off measurement;
  4. no measurable slowdown (outside step 1's run-to-run spread) of any kernel on any size or thread count;
  5. no FMA instruction in the AVX2 object on either compiler (disassembly check);
  6. both verification scripts exit 0 on the final commit.
- **Q7, the four conditions.** Proposed: item 7's acceptance bar carries over unchanged. Numerical correctness (criterion 1, bitwise), memory accounting (no new allocation: kernels work in place on the existing state), cross-engine consistency (criterion 2) and performance predictability. For the last, step 1 and the final measurement report the calibrated-rate change and the per-k price shift of 19.1. Any fix goes to the model-fixes item.

### 19.5 Owner answers (2026-11-10)

The owner took every recommendation of 19.4:
- **Q1:** baseline and AVX2 paths only. AVX-512 is deferred until a machine in the verification set can run it.
- **Q2 and Q3:** hand-written AVX2 intrinsics, with no FMA, for the 1-qubit, 2-qubit and diagonal kernels. They go in one translation unit, dispatched by `cpuid` on GCC, Clang and MSVC alike, and replace GCC's `target_clones`.
- **Q4:** as Q3. The generic kernel and the expectation sum stay scalar.
- **Q5:** re-run the item 7 probe after item 8 and report it. The fusion window stays 2.
- **Q6 and Q7:** the gate as proposed, with item 7's four conditions. The speed-up factors are the owner's, set from step 1.
