# ADR 0025: Phase 21 — Planner performance

## Status

**Draft (2026-11-03), step 0.** The owner started Phase 21 on 2026-11-03.
Its scope is `plan_job` latency plus the one-time hardware calibration, and
it is placed ahead of the memory roadmap (now Phases 22–24). Section 9 lists
the owner decisions still open. No source, build or test file has changed in
this phase yet.

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

## 6. Equivalence testing (step 5) — [MENOS], proposed

A randomized corpus of jobs is planned by both the frozen pre-phase planner
and the new one, and the results are compared field by field and through
`explain()`. The frozen planner is kept as a test-only copy built from the
`d89dde3` sources, or `explain()` golden files are recorded at `d89dde3`;
which of the two is in §9.

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

## 9. Owner decisions — [OPEN]

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
