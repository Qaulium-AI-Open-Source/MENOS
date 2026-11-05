# Phase 24 baseline: planner latency (step 1)

The reference figures for every later Phase 24 claim (`docs/adr/0028` section 5). A step-2
change is reported as a same-machine ratio against this table, measured with the same
command. Every figure was measured on planner, analysis, cost and engine sources identical to
`main` at `d89dde3`. The step-1 commit adds only the benchmark, the corpus, the golden
test, profiling scripts and documentation. After the measurement, `01aa3b6` fixed the
clang-tidy errors PR #12 left on `main`. Its only planner change is an explicit
`if`/`throw` replacing an `ensure()` in `reparameterize_direct_plan` (the plan-cache
hit path), which throws the same exception with the same message. The golden plans
still match after it.

## Machine and build

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 170 (8 cores / 16 threads), L2 512 KiB per core, L3 16 MiB; governor `powersave`, frequency scaling on |
| Memory | 14 GiB |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (CUDA calibration only) |
| OS / compiler | Ubuntu 26.04.1 LTS, GCC 15.2.0 |
| Build | `cmake --preset release -B build/release-syms -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer" -DMENOS_BUILD_BENCHMARKS=ON` (Release `-O3`, plus symbols and frame pointers for profiling) |
| Date | 2026-11-03 |

Google Benchmark warns that CPU scaling and ASLR are on. The spread columns below show
how much that matters: CV is under 2% for every job over 0.1 ms.

## 1. `plan_job` latency and heap traffic — [MEASURED]

Command:

```
build/release-syms/benchmarks/menos_planner_benchmark --benchmark_repetitions=5 \
    --benchmark_min_time=0.2s --benchmark_out=bench/raw/planner_baseline.json \
    --benchmark_out_format=json
```

Raw output: `bench/raw/planner_baseline.json` and `.txt`. Inputs: the 45 jobs of
`tests/plan/planner_corpus.h`, each planned by `plan::plan_job` alone, single-threaded,
against a hand-built hardware profile (no calibration, no plan cache). `allocs`,
`alloc MB` and `peak MB` come from one extra untimed call, through the counting allocator
in `benchmarks/counting_allocator.cpp`. `peak MB` is the peak live heap above what was
live before the call. The `cache_hit/` rows time `plan_job_cached` with the plan
already cached. Their allocation columns are not measured (0).

| Benchmark | median µs | min–max µs (5 reps) | CV | allocs | alloc MB | peak MB |
|---|---:|---:|---:|---:|---:|---:|
| `plan_job/statevector_brickwork/small` | 19.9 | 19.7–29.6 | 20.5% | 795 | 0.04 | 0.03 |
| `plan_job/statevector_brickwork/medium` | 94.5 | 92.9–95.5 | 1.2% | 3,844 | 0.15 | 0.15 |
| `plan_job/statevector_brickwork/large` | 282.9 | 281.2–283.9 | 0.3% | 11,476 | 0.52 | 0.49 |
| `plan_job/counts_brickwork/small` | 65.0 | 64.4–66.0 | 1.2% | 1,135 | 0.07 | 0.03 |
| `plan_job/counts_brickwork/medium` | 278.1 | 274.3–284.0 | 1.3% | 5,349 | 0.31 | 0.15 |
| `plan_job/counts_brickwork/large` | 718.1 | 713.4–774.4 | 3.6% | 15,546 | 0.94 | 0.49 |
| `plan_job/expectation_z_all/small` | 507.3 | 505.3–538.3 | 2.7% | 3,566 | 0.42 | 0.09 |
| `plan_job/expectation_z_all/medium` | 8,329 | 8,294–8,352 | 0.3% | 14,348 | 2.36 | 0.56 |
| `plan_job/expectation_z_all/large` | 79,752 | 79,679–80,224 | 0.3% | 39,828 | 8.95 | 2.86 |
| `plan_job/expectation_ising/small` | 324.1 | 323.5–329.8 | 0.9% | 3,019 | 0.33 | 0.08 |
| `plan_job/expectation_ising/medium` | 8,379 | 8,317–8,713 | 1.9% | 14,412 | 2.36 | 0.56 |
| `plan_job/expectation_ising/large` | 80,147 | 79,529–82,116 | 1.3% | 39,924 | 8.93 | 2.86 |
| `plan_job/expectation_heisenberg/small` | 332.3 | 328.7–336.8 | 0.9% | 3,043 | 0.33 | 0.08 |
| `plan_job/expectation_heisenberg/medium` | 8,289 | 8,213–8,350 | 0.7% | 14,468 | 2.36 | 0.56 |
| `plan_job/expectation_heisenberg/large` | 79,364 | 79,115–79,978 | 0.4% | 40,012 | 8.94 | 2.86 |
| `plan_job/expectation_random_paulis/small` | 325.8 | 325.4–326.5 | 0.1% | 3,039 | 0.33 | 0.08 |
| `plan_job/expectation_random_paulis/medium` | 8,304 | 8,293–8,394 | 0.6% | 14,692 | 2.37 | 0.57 |
| `plan_job/expectation_random_paulis/large` | 80,833 | 79,771–82,529 | 1.3% | 41,340 | 9.00 | 2.90 |
| `plan_job/expectation_ising_budgeted/small` | 6,019 | 5,976–6,064 | 0.6% | 26,677 | 3.70 | 0.08 |
| `plan_job/expectation_ising_budgeted/medium` | 280,407 | 278,462–282,178 | 0.6% | 217,786 | 51.90 | 0.56 |
| `plan_job/expectation_ising_budgeted/large` | 1,894,074 | 1,889,225–1,907,120 | 0.4% | 572,992 | 187.40 | 2.86 |
| `plan_job/counts_clifford/small` | 180.6 | 180.4–181.3 | 0.2% | 2,432 | 0.18 | 0.07 |
| `plan_job/counts_clifford/medium` | 2,880 | 2,820–2,949 | 1.7% | 37,932 | 2.82 | 1.04 |
| `plan_job/counts_clifford/large` | 36,480 | 36,407–36,971 | 0.7% | 390,309 | 32.98 | 11.91 |
| `plan_job/counts_random_pairs/small` | 128.8 | 128.4–130.4 | 0.6% | 1,827 | 0.13 | 0.06 |
| `plan_job/counts_random_pairs/medium` | 759.6 | 755.6–772.0 | 0.9% | 6,621 | 0.61 | 0.12 |
| `plan_job/counts_random_pairs/large` | 3,014 | 2,995–3,021 | 0.4% | 26,246 | 2.43 | 0.48 |
| `plan_job/expectation_random_pairs_controlled/small` | 688.8 | 685.5–711.0 | 1.5% | 3,542 | 0.41 | 0.10 |
| `plan_job/expectation_random_pairs_controlled/medium` | 21,897 | 21,771–22,026 | 0.4% | 19,573 | 3.19 | 1.36 |
| `plan_job/expectation_random_pairs_controlled/large` | 404,532 | 397,685–409,097 | 1.3% | 74,379 | 22.48 | 15.36 |
| `plan_job/counts_factorized/small` | 115.3 | 114.7–117.5 | 1.0% | 1,826 | 0.15 | 0.04 |
| `plan_job/counts_factorized/medium` | 904.6 | 893.6–909.5 | 0.7% | 14,359 | 1.23 | 0.27 |
| `plan_job/counts_factorized/large` | 3,851 | 3,849–4,049 | 2.2% | 58,214 | 5.06 | 1.05 |
| `plan_job/expectation_factorized/small` | 1,455 | 1,453–1,456 | 0.1% | 12,122 | 1.45 | 0.07 |
| `plan_job/expectation_factorized/medium` | 350,452 | 349,671–356,308 | 0.9% | 360,077 | 75.22 | 0.72 |
| `plan_job/expectation_factorized/large` | 13,307,719 | 13,279,876–13,332,315 | 0.1% | 3,635,179 | 1,460.68 | 7.85 |
| `plan_job/expectation_cut/small` | 265.0 | 264.8–265.8 | 0.2% | 2,867 | 0.33 | 0.07 |
| `plan_job/expectation_cut/medium` | 3,479 | 3,474–3,483 | 0.1% | 9,303 | 1.35 | 0.31 |
| `plan_job/expectation_cut/large` | 30,729 | 30,703–30,777 | 0.1% | 25,550 | 4.98 | 1.29 |
| `plan_job/counts_noisy_density_matrix/small` | 14.6 | 14.4–14.8 | 0.9% | 525 | 0.03 | 0.03 |
| `plan_job/counts_noisy_density_matrix/medium` | 58.7 | 58.6–59.2 | 0.4% | 2,085 | 0.11 | 0.11 |
| `plan_job/counts_noisy_density_matrix/large` | 147.9 | 144.5–160.8 | 4.4% | 5,237 | 0.27 | 0.28 |
| `plan_job/counts_noisy_trajectory/small` | 34.1 | 31.9–43.2 | 12.3% | 1,086 | 0.06 | 0.06 |
| `plan_job/counts_noisy_trajectory/medium` | 99.3 | 97.6–111.4 | 5.5% | 3,186 | 0.19 | 0.18 |
| `plan_job/counts_noisy_trajectory/large` | 212.2 | 209.2–212.7 | 0.8% | 7,061 | 0.40 | 0.39 |
| `cache_hit/statevector_brickwork/medium` | 115.3 | 114.5–116.1 | 0.6% | 0 | 0.00 | 0.00 |
| `cache_hit/counts_brickwork/large` | 343.3 | 341.5–347.1 | 0.7% | 0 | 0.00 | 0.00 |
| `cache_hit/expectation_ising/medium` | 113.5 | 113.1–121.0 | 2.9% | 0 | 0.00 | 0.00 |

### Observations

- **The worst case is the factorized expectation job.** `expectation_factorized/large`
  (8 independent 8-qubit blocks, a 127-term Ising observable) takes **13.3 s** to plan,
  with 3.6 M allocations and 1.46 GB requested. The medium size (4×6 qubits) takes
  0.35 s. This is consistent with H3 in `docs/ARCHITECTURE_NOTES.md`, but the profile
  (section 3) decides it.
- **A time budget makes expectation planning 19–34× slower** (`expectation_ising_budgeted`
  against `expectation_ising` at the same size). Branch-and-bound is off in a budgeted
  job, so every candidate's search runs to completion.
- **Plain expectation planning does not scale with the number of terms.** At each size,
  `z_all` (1 term), `ising` (2n−1), `heisenberg` (3(n−1)) and `random_paulis` (20/100/400
  terms) all take the same time: about 0.33 ms, 8.3 ms and 80 ms. The per-term min-fill
  repetition (H1) cannot be the dominant cost of these unbudgeted jobs. The profile must
  show where the 80 ms goes before change (a) is chosen.
- **A plan-cache hit can cost more than planning from scratch.** It takes 115 µs against
  94.5 µs for `statevector_brickwork/medium`, and 343 µs against 718 µs for
  `counts_brickwork/large`: the key serialisation, a second light-cone pass and a full
  `Plan` copy.
- Dense, noisy and Clifford counts jobs plan in under 1 ms, apart from Clifford at 64 and
  256 qubits (2.9 ms and 36 ms).

## 2. Cold hardware calibration — [MEASURED]

Command (20 fresh processes; each process times one cold call, then one warm call):

```
bench/run_calibration.sh build/release-syms/benchmarks/menos_calibration_probe 20 \
    > bench/raw/calibration_cpu.csv
```

CPU-only build:

| Figure | median | min | max | max/min |
|---|---:|---:|---:|---:|
| cold `calibrate_hardware()` | 85.3 ms | 82.3 ms | 96.7 ms | 1.18 |
| warm (second call, same process) | 84.0 ms | 81.9 ms | 95.4 ms | 1.17 |
| `work_units_per_second` | 1.906e9 | 1.228e9 | 1.948e9 | **1.59** |
| `memory_bandwidth_bytes_per_second` | 2.733e10 | 2.463e10 | 2.913e10 | 1.18 |

- Cold and warm cost the same: calibration is a fixed cost of about 85 ms per process, set
  by its four ≥15 ms deadline passes plus allocating and filling 64 MiB. It is not a
  cold-start effect, so caching the *result* (the owner's preferred direction) is the
  only way to remove it without touching probe windows.
- The derived constants already vary by up to 1.59× (throughput) and 1.18× (bandwidth)
  from one process to the next. Any accuracy bound agreed for a later change has to be
  compared with this existing spread.
- A CPU-only build calibrates only for a time-budgeted job (`hardware_profile_for`).

### 2.1 CUDA build

The same probe built from `cmake --preset cuda-release -B build/cuda-release-syms
-DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer" -DMENOS_BUILD_BENCHMARKS=ON` (CUDA 13.4,
RTX 3050 Laptop GPU), 20 fresh processes, `bench/raw/calibration_cuda.csv`:

| Figure | median | min | max | max/min |
|---|---:|---:|---:|---:|
| cold `calibrate_hardware()` | **423 ms** | 416 ms | 629 ms | 1.51 |
| warm (second call) | 289 ms | 286 ms | 298 ms | 1.04 |
| `work_units_per_second` | 1.916e9 | 1.288e9 | 1.960e9 | 1.52 |
| `memory_bandwidth_bytes_per_second` | 2.714e10 | 2.333e10 | 2.820e10 | 1.21 |
| `cuda_host_to_device_bytes_per_second` | 1.047e10 | 8.98e9 | 1.101e10 | 1.23 |
| `cuda_device_to_host_bytes_per_second` | 8.563e9 | 6.75e9 | 8.86e9 | 1.31 |
| `cuda_kernel_launch_seconds` | 5.19e-6 | 5.08e-6 | 8.47e-6 | 1.67 |
| `cuda_device_work_units_per_second` | 6.389e9 | 4.92e9 | 6.40e9 | 1.30 |

- A CUDA build calibrates for **every** job, budgeted or not (`hardware_profile_for`), so
  each process pays about 0.42 s before it plans anything. That is 5× the CPU-only cost,
  and more than every corpus plan except the four slowest. Warm, the CUDA probes add
  about 205 ms to the CPU's 84 ms; cold adds a further ~135 ms (CUDA context creation).
- Every derived constant varies by 1.2–1.7× between processes.

## 3. Profile — [MEASURED]

**Method.** `perf` is unavailable to an unprivileged user here (`kernel.perf_event_paranoid
= 4`), and valgrind/heaptrack are not installed yet. So hotspots were sampled with a
wall-clock stack sampler that needs no root (`bench/tools/pmp.sh`). gdb runs the benchmark
as its own child, a loop sends it `SIGUSR2` every 30–50 ms, and each stop records the
full stack and the innermost MENOS source line. The samples cover a release build with
symbols (`build/release-syms`), one corpus job at a time. Raw stacks:
`bench/raw/profiles/*.stacks.gz`. Aggregation: `bench/tools/agg.py` (inclusive, per
function) and `bench/tools/lines.py` (innermost MENOS line). Each figure below is a
share of samples. With 200–500 samples per job, the uncertainty is about ±3 percentage
points. Cache-miss counters (`perf stat`) need perf, so they are still to come.

### 3.1 Where the time goes

| Job (samples) | Inclusive | Share |
|---|---|---:|
| `expectation_factorized/large` (516) | `TensorNetworkEngine::estimate` | 99.6% |
| | of which the **direct** tensor-network candidate (`estimate_direct`) | 85.0% |
| | of which `describe_engine_choice` re-costing losing engines (`build_factorized_regions`) | 14.6% |
| | `estimate_treewidth_min_fill_dense` | 98.1% |
| | `compute_fill` (incl. `BitAdjacency::common`) | 53.8% |
| `expectation_ising_budgeted/large` (20) | `estimate_cut` → fragment `TensorNetworkEngine::estimate` → min-fill | 95–100% |
| `expectation_z_all/large` (213) | `TensorNetworkEngine::estimate` | 96.7% |
| | split: direct candidate / cut fragments | 48.4% / 49.8% |
| | `compute_fill` | 69.5% |
| `expectation_random_paulis/large` (219) | `TensorNetworkEngine::estimate` (direct 51.1%, cut 44.7%) | 93.6% |
| `expectation_cut/large` (288) | `TensorNetworkEngine::estimate` (direct 47.6%, cut 49.0%) | 93.8% |
| `expectation_random_pairs_controlled/large` (206) | `TensorNetworkEngine::estimate` (direct) | 97.6% |
| `counts_clifford/large` (292) | `MPSEngine::estimate` → `compute_schmidt_rank_bound` | 84.9% |
| | of which LAPACK `zgesdd` (via `common::svd`) | 73.3% |
| | of which `operand_matrix_of` / `generate_matrix` | 5.5% |

### 3.2 Inside the min-fill search (innermost line)

| Line | What it is | `factorized/large` | `z_all/large` | `random_pairs_controlled/large` | `cut/large` | `random_paulis/large` |
|---|---|---:|---:|---:|---:|---:|
| `treewidth.cpp:287–288` | `BitAdjacency::common`: popcount over **every** word of two rows | 43.7% | 63.4% | 59.3% | 54.5% | 59.3% |
| `treewidth.cpp:296–300` | `for_each_neighbor` word walk | 12.3% | ~7% | ~14% | ~8% | ~8% |
| `treewidth.cpp:423–431` | linear scan for the minimum-fill vertex | 38.4% | 21.1% | 22.3% | 19.5% | 18.7% |

### 3.3 What this means for the step-2 order

- **(b) heap for the vertex choice:** confirmed. The linear scan is 19–38% of every large
  expectation plan.
- **A change not on the owner's list is the largest single item.** `BitAdjacency::common`
  ANDs and popcounts all `words_` = V/64 words of two rows for every neighbour, even
  though a row has only `degree` set bits, which the elimination width bounds. It is 44–63% of these
  plans. Restricting it to the non-zero words of `v`'s row (or keeping sparse neighbour
  lists next to the bitset) gives the same integers, so the plans are identical. This is
  proposed as change **(d)** for the owner to confirm.
- **(a) memoise min-fill per qubit set:** not supported as a first change. In the
  unbudgeted jobs, planning time does not grow with the number of terms (section 1),
  because branch-and-bound stops the search early. The 13 s job's 127 Ising terms all act
  on different qubit sets, so memoising by support would save nothing there. It would
  help where supports repeat and pruning is off (a budgeted Heisenberg-like observable,
  up to 3×). Proposed: after (b) and (d), only if the re-measured profile still shows
  repeated identical searches.
- **(c) SVD cache and no 1-qubit matrices:** confirmed for Clifford, MPS and long-circuit
  jobs. LAPACK `zgesdd` is 73% of `counts_clifford/large`. In this corpus it is at most
  2–3% elsewhere.
- **Duplicate costing (H3):** 14.6% of the 13 s job is `describe_engine_choice` re-running
  the losing engines' searches, without the prune bound, just to word `explain()`. Reusing
  the estimates already computed gives identical text. This is a candidate after (b)/(d).
- **Rust:** the hot kernel is a bitset popcount loop over `std::vector<std::uint64_t>`,
  already compiled with a POPCNT clone. Its cost is algorithmic (scanning empty words),
  not language overhead, so the profile does not justify a Rust kernel.
- **Allocation:** the 13 s job requests 1.46 GB in 3.6 M allocations, but `_int_malloc`
  and `memset` are under 1% of its samples. Allocation is not where its time goes, which
  is consistent with leaving copy removal to Phase 22.
- **Calibration:** a fixed 85 ms (CPU) or 423 ms (CUDA, every job) per process.
