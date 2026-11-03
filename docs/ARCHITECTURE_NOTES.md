# Architecture notes: the planning hot path

Phase 21, step 0 (discovery). Written 2026-11-03 against `main` at `d89dde3`.
Nothing below has been measured in this phase yet. Every claim about speed is
marked **[HYPOTHESIS]** until the step-1 baseline (`bench/BASELINE.md`) confirms
or refutes it. Figures from earlier phases are cited with their ADR.

## 1. Summary

MENOS is a quantum circuit simulation runtime (C++20, CMake, Apache-2.0,
Python package `menos` 1.2.0 with no git tags). "Planning" here means choosing
**how** to simulate a circuit. It is not pathfinding or task planning:
`plan::plan_job` (`src/plan/planner.cpp:147`) enumerates a fixed list of
candidate execution strategies (DirectDense, DirectStabilizer, DirectPauli,
DirectMPS, DirectTensorNetwork, DirectDensityMatrix, DirectTrajectory,
Factorized, CutAndRecombine). It asks each simulation engine for a
`CostEstimate` (peak bytes, work units), filters by memory/time budget, picks
the fewest work units (ties to the earlier candidate), and builds a `Plan`.

The planner has no search frontier or open set. It is a fixed enumeration of
candidates. The expensive parts are the *analyses* some candidates' estimates
run:

| Analysis | Algorithm | Where |
|---|---|---|
| Light-cone reduction | one backward pass over operations | `analysis/light_cone.cpp` |
| Interaction graph (components) | union-find over multi-qubit ops | `analysis/interaction_graph.cpp` |
| MPS bond bound | operator-Schmidt rank by **SVD per (multi-qubit gate × cut it spans)** | `analysis/schmidt_rank_bound.cpp:133` |
| Tensor-network contraction width | **greedy min-fill elimination per observable term**, bitset adjacency (≤23 000 nodes) or `std::set` adjacency | `analysis/treewidth.cpp:392/498/598` |
| Wire-cut search | ≤48 single cuts, ≤600 cut pairs, each a union-find fragment scan | `analysis/cut.cpp:288` |
| Hardware calibration (once per process) | deadline-bounded memcpy bandwidth + 13-qubit dense run; CUDA probes | `plan/hardware_profile.cpp:233`, `plan/cuda_calibration.cpp:40` |

Two kinds of reuse already exist:
- **Branch-and-bound pruning** (`planner.cpp:393`, `cost_model.cpp:22`,
  `tensor_network_engine.cpp:155`). It applies only to unbudgeted jobs with no
  CUDA backend.
- An **LRU plan cache** (`plan/plan_cache.cpp:185`, capacity 64). It reuses a
  *Direct* plan when a later call differs only in gate angles.

Earlier measurements: one `plan_job` takes **0.052–17.2 ms mean**, worst
sample 85 ms (`docs/adr/0015` §10). Tensor-network planning overhead is
**0.65–10.6 ms** for expectation jobs up to 20 qubits, depth 8
(`docs/adr/0011`). Calibration is said to cost "well under a second"
(`hardware_profile.cpp:66`), but that has not been measured. In a CPU-only
build, an unbudgeted job does not calibrate at all
(`hardware_profile_for`, `hardware_profile.cpp:253`). A CUDA build always
calibrates.

## 2. Entry points into planning

```
C++    menos::run / preview            src/api/menos.cpp:90 / :101
CLI    menos run|explain               cli/cli_impl.cpp  -> menos::run / preview
Python menos._core (nanobind)          src/python/bindings.cpp -> menos::run
Qiskit MENOSBackend / Estimator / Sampler, PennyLane menos.qubit  -> menos._core
Batch  run_batch_with_options          src/api/batch.cpp:10 -> run per circuit
Replan run_with_replanning             src/api/replan_controller.cpp:63, :160 -> plan_job
```

## 3. Call graph of the hot path

```
validate_and_plan                                        api/menos.cpp:24
 ├─ JobSpec{ .circuit = circuit }                        copy #1 of the circuit
 ├─ hardware_profile_for(budgeted)                       plan/hardware_profile.cpp:253
 │   └─ calibrated_hardware_profile() [static, once]     :266
 │       └─ calibrate_hardware                           :233
 │           ├─ measure_memory_bandwidth                 :84   2×64 MiB alloc+fill, 2 passes ≥15 ms each
 │           ├─ measure_work_units_per_second            :150  13-qubit dense execute, 2 passes ≥15 ms each
 │           └─ measure_cuda_calibration (CUDA builds)    plan/cuda_calibration.cpp:40 (20-qubit probe + context init)
 └─ plan_job_cached                                      plan/plan_cache.cpp:185
     ├─ plan_cache_key  (serialise structure to std::string)  :105
     ├─ hit  → reparameterize_direct_plan                planner.cpp:1110 (re-runs light cone, copies Plan)
     └─ miss → plan_job                                  planner.cpp:147
         ├─ compute_light_cone_support + compute_light_cone   :160–163   reduced circuit = copy #2
         ├─ restrict_observable (expectation only)            :179
         ├─ estimate_direct(dense)                            :360  → DenseEngine::estimate  O(ops)
         ├─ estimate_direct(stabilizer)                       :413  → Clifford scan  O(ops)
         ├─ estimate_direct(pauli)        [expectation]       :441  → O(ops)
         ├─ estimate_direct(mps)                              :465  → MPSEngine::estimate (mps_engine.cpp:95)
         │    └─ compute_schmidt_rank_bound                   schmidt_rank_bound.cpp:133
         │         for each op: generate_matrix (heap alloc, also for 1-qubit ops)   :148
         │           for each cut in [min_q, max_q): build 2^2l×2^2r matrix + LAPACK SVD   :160–178
         ├─ estimate_direct(tensor_network) [expectation]     :498  → TensorNetworkEngine::estimate (:155)
         │    for each observable term:                       tensor_network_engine.cpp:230
         │      build_closed_timelines (re-runs ket_timelines) treewidth.cpp:99
         │      build_line_graph                               treewidth.cpp:171
         │      estimate_treewidth_min_fill                    treewidth.cpp:598
         │        per step: linear scan for min fill  O(V)     :423   → O(V²) overall
         │        fill-in + recompute fill of affected set     :461–495
         ├─ estimate_direct(density_matrix) [noisy]           :526
         ├─ trajectory_count + estimate_direct(trajectory) [noisy, opted in]  :567–586
         ├─ build_interaction_graph                           :603
         ├─ estimate_factorized  [≥2 components]              :607 → cost_model.cpp:95
         │    per component: pick_cheapest_engine (all engines, incl. MPS/TN analyses)   cost_model.cpp:205
         ├─ find_cut_plan + estimate_cut [expectation]        :639–646 → cut.cpp:288, cost_model.cpp:260
         │    ≤648 cut evaluations, each widest_fragment_for O(ops)
         │    per fragment × per term: build_fragment_circuit, pick_cheapest_engine  cost_model.cpp:332
         ├─ CUDA backend choice (estimate_direct again on CUDA engine)    :675–776
         ├─ Factorized per-region CUDA choice → build_factorized_regions  :797  (2nd costing of every component)
         ├─ apply_time_budget, selection                      :874–906
         └─ Plan assembly                                     :908–1105
              plan.candidates = candidates (copy of strings)
              plan.reduced_circuit = circuit                  copy #3
              build_direct_region → .operations = circuit.operations()   copy #4 (region_builder.cpp:95)
              Factorized winner → build_factorized_regions    :1065 (2nd or 3rd costing of every component)
                   per component: pick_cheapest_engine (region_builder.cpp:184)
                   + describe_engine_choice: re-estimates every losing engine, unpruned (region_builder.cpp:84)
```

Data structures in the path:
- `Circuit` is a `std::vector<Operation>`. `Operation` is a `std::variant`.
  `GateOp` holds `std::vector<QubitId> qubits` and `std::vector<double> params`
  (`include/menos/circuit.h:50`), so each copy of a circuit costs about 2
  heap allocations per operation.
- Engines are reached through `SimulationEngine` virtual calls (`estimate()`),
  about 5–7 per candidate set. The call count is small; the work inside the
  calls is what matters.
- Min-fill uses a `BitAdjacency` (one bitset row per vertex, `treewidth.cpp:254ff`)
  with `std::vector<bool>` eliminated/affected flags. Graphs above 23 000 nodes
  fall back to `std::vector<std::set<size_t>>` (`treewidth.cpp:498`).
- Cut search uses a union-find (`DisjointSet`, `cut.cpp:69`).
- The plan cache is a `std::list` plus an `std::unordered_map<std::string, …>`
  under a mutex.

## 4. Hypotheses for why planning is slow (ranked by expected impact)

Each one must be measured in step 1 before any change. The rule for every
candidate fix is that the chosen plan and `explain()` output stay
byte-identical.

1. **[HYPOTHESIS] Tensor-network costing dominates expectation jobs.** The
   full closed network and min-fill search are rebuilt for *every* observable
   term. A term's network depends only on the term's **support** (which qubits
   are non-identity; `build_closed_timelines` adds an observable node iff
   the X or Z bit is set, `treewidth.cpp:137`), not on X/Y/Z. So terms with
   equal support repeat an identical, deterministic search. Possible fixes:
   - memoise the elimination result by support;
   - build `ket_timelines` once per call instead of once per term;
   - in min-fill, use a bucket queue or an indexed heap keyed by (fill,
     vertex) instead of the O(V) scan per step. That keeps the lowest-index
     tie-break, and turns the O(V²) selection into O(V log V).
2. **[HYPOTHESIS] The Schmidt-rank bound does an SVD per (multi-qubit gate ×
   cut it spans).** A 2-qubit gate has the same one-vs-one partition at every
   cut in its span, so its rank is computed `span` times. The matrix is also
   regenerated (heap allocation) for every operation, single-qubit ones
   included, before the single-qubit early-out. This runs for DirectMPS, and
   again for every Factorized component and every cut fragment × term. Fixes
   that keep results bit-exact:
   - compute the rank once per distinct partition per gate;
   - skip matrix generation for 1-qubit ops;
   - memoise by (opcode, params bits, partition).
3. **[HYPOTHESIS] Factorized and cut candidates cost the same sub-problems
   several times.** Every component is costed by `estimate_factorized`, again
   by `build_factorized_regions` when Factorized wins (and a third time in the
   CUDA per-region step), and again by `describe_engine_choice`, which
   re-estimates every losing engine *without* the prune bound. `estimate_cut`
   costs fragment × term × engine. Possible fix: cost each component once and
   carry the `EngineChoice` through to region building.
4. **[HYPOTHESIS] Circuit copies cause allocation churn on large circuits.**
   There are at least 4 full copies of the operation list per plan (JobSpec,
   light-cone result, `plan.reduced_circuit`, Direct region). There are also
   per-component and per-fragment copies, `std::to_string` reason strings,
   and the cache key string. Each copy costs about 2 allocations per
   operation. This matters only if allocation shows in the profile. It
   overlaps the Phase 22 (memory accounting and copy removal) scope, so a
   change to the `Plan`/`Region` layout needs owner sign-off.
5. **[HYPOTHESIS] Cut search (≤648 evaluations × O(ops)) is noticeable only on
   expectation jobs with long circuits.** It is already bounded by design
   (`docs/adr/0013`).
6. **[HYPOTHESIS] Calibration is a fixed ~60–100 ms per process on the CPU.**
   That is 4 deadline passes of ≥15 ms, plus allocating and touching 64 MiB.
   A CUDA build adds context creation and a 20-qubit probe. The deadlines are
   measured design constants (`hardware_profile.cpp:40–80`); shortening them
   trades accuracy and would need an ADR. Lazy or overlapped calibration, or
   skipping a CUDA probe that no plan can use, may be possible without
   changing any figure.
7. **[HYPOTHESIS] Virtual dispatch, `std::function` (`MinFillStopPredicate`)
   and `std::variant` visits are not significant.** Their call counts are
   linear in operations or steps, and the work per call dominates. Listed so
   that the profile can rule them out explicitly.

## 5. How the prompt's suggested techniques map to this code

| Technique | Fit | Why |
|---|---|---|
| Heap/bucket queue for the open set | **Yes, inside min-fill only** | The min-fill vertex choice is a priority selection with decrease-key (H1). Elsewhere there is no open set. |
| Visited-stamp / generation counters | Possible | Per-term scratch in min-fill/timelines (`is_affected`, `eliminated`) is reallocated per term. |
| Struct-of-arrays, flat graphs | Possible | Line graph and timelines are vectors of vectors. Measure first. |
| Memoisation | **Yes** | H1 (by support), H2 (by gate/partition), H3 (by component). |
| A*/IDA*/JPS/navmesh/hierarchical planning | No | There is no pathfinding problem. |
| Arena/pool per plan | Maybe | Only if H4 shows in the profile. `Plan` outlives planning, so only scratch qualifies. |
| Rust kernel / mimalloc | Unlikely | The kernels are small and tightly coupled to the C++ IR. Evaluate only if one self-contained kernel dominates (step 4). |

## 6. Repository map (for reference)

- **Build**: CMake ≥3.25, `CMakePresets.json` (dev, release, cuda-dev,
  cuda-release, cuda12-dev, cuda12-release, windows-dev, windows-release).
  Options: `MENOS_BUILD_PYTHON_BINDINGS`, `MENOS_BUILD_BENCHMARKS` (Google
  Benchmark, fetched), `MENOS_ENABLE_CUDA`. scikit-build-core drives the
  Python wheel (`pyproject.toml`).
- **Source**: `include/menos/` (public headers), `src/{ir,analysis,cost,plan,
  engines/*,kernels,runtime,api,mitigation,qasm2,qasm3,python,common}`, `cli/`,
  `python/menos/` (Qiskit and PennyLane adapters).
- **Dependencies**: Threads, LAPACK (OpenBLAS on Windows, pinned), nanobind
  (Python), Google Benchmark (optional), CUDA ≥12.0 (optional). Python extras:
  qiskit, pennylane==0.45.1.
- **Tests**: `tests/` (C++ ctest, incl. `tests/plan/planner_test.cpp`,
  `tests/analysis/treewidth_test.cpp`, `schmidt_rank_bound_test.cpp`,
  `cut_test.cpp`, `plan_cache_test.cpp`), `python/tests/` (pytest).
- **Benchmarks**: `benchmarks/` (Google Benchmark). There is no dedicated
  `plan_job` benchmark. Planning time is recorded incidentally by the
  tensor-network cost-correlation benchmark.
- **Verification**: `scripts/verify.sh` (Linux: presets, sanitizers,
  clang-tidy, Python) and `scripts/verify-windows.ps1`.
- **CI**: `.github/workflows/build-wheels.yml` (workflow_dispatch only, wheels).
  No per-push build/test CI.
- **Licence**: Apache-2.0. **Version**: 1.2.0 (`pyproject.toml`). **Tags**: none.
- **Local tooling** (this machine): `perf`, `cmake`, `g++` and `clang++` are
  present. `valgrind`, `heaptrack` and `ninja` were not found on `PATH`.
