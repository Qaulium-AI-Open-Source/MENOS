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
