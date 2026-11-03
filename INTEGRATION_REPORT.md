# MENOS 0.2.0 + optimization patches 0001–0009: integration report

Date: 2026-11-03. Inputs: `MENOS-main.zip` (MENOS 0.2.0, git history with patches 0001–0004 already committed) and `menos_optimization 2.zip` (nine patches, harness, benchmark results).

## How the patches were applied

| Patch | Subject | How it went in |
|---|---|---|
| 0001–0004 | Planner: bitset min-fill, branch-and-bound, lower bounds, cheaper cut search | Already committed in the received repo. Patch IDs match the patch files exactly. |
| fix | Branch-and-bound guard used `JobSpec::backend` | **Added.** 0002 referred to a field that 0.2.0 does not have, so the received repo did not compile. 0.2.0 picks the CUDA backend automatically, so the bound is now switched off exactly when a calibrated CUDA device is present. |
| 0005 | Streaming diagonal kernels, batched Z-term expectation, vectorised loops | **Ported by hand.** The patch targets MENOS 1.2.0's multithreaded specialised kernels. 0.2.0 has one generic single-threaded kernel, so the fast paths were added in front of it. |
| 0006 | Native `rzz` gate | Applied. Hunks for files that don't exist in 0.2.0 were dropped: the PennyLane converter and its test, and `src/qasm3/`. |
| 0007 | Plan cache | **Ported.** The key drops `max_threads` and `backend`, which 0.2.0 doesn't have. Two known defects are fixed (see below). |
| 0008 | Opt-in diagonal-gate fusion | **Ported.** Field placement was adapted to 0.2.0's structs. |
| 0009 | Skip calibration for unbudgeted CPU jobs | Applied. The executable docs page `docs/reference/python-api.md` was rewritten for 0.2.0. |

Every port is its own commit with an explanation (`git log eb35e36..HEAD`). `eb35e36` ("Base MENOS") is the untouched 0.2.0.

### Defects fixed on top of the original patches

1. **0005:** `target_clones("avx2")` was enabled for any GCC. It is now limited to GCC on x86-64 Linux, because it fails on aarch64. The CPU kernel target is built with `-ffp-contract=off` so that the bitwise-identity guarantee holds.
2. **0007:** a cache hit copied the first job's `CircuitSummary`, so `explain()` reported the first job's `global_phase`. The summary is now recomputed on every hit.
3. **0007:** the `HardwareProfile` was not part of the cache key. It now is.

Each fix in 0007 has a regression test. Both tests were checked to **fail** with the fix removed and to pass with it.

## Verification (this exact tree, Linux x86-64, 2 vCPU, GCC, no GPU)

| Check | Original 0.2.0 (`eb35e36`) | This tree |
|---|---|---|
| `cmake --preset release` build (`-Werror`) | clean | clean |
| `ctest --preset release` | 578 / 578 passed | **607 / 607 passed** (+29 new tests) |
| `pytest python/tests` (wheel smoke deselected) | 207 passed | **212 passed** |
| clang-format 20.1.0 on changed C++ files | — | clean |

Commands:

```bash
cmake --preset release && cmake --build --preset release -j
ctest --preset release
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python "scikit-build-core==1.0.3" "nanobind==3.1.0"
uv pip install --python .venv/bin/python -e ".[test]" --no-build-isolation
.venv/bin/python -m pytest python/tests --deselect python/tests/test_wheel_smoke.py
```

Ubuntu needs `liblapacke-dev liblapack-dev libopenblas-dev` to configure.

## Measured speed-up (QAOA expectation, all-ZZ + Z Hamiltonian, p = 3)

Each run is a fresh process. Patched builds report the median of 20 calls after 2 warm-ups; the original reports the median of 3, because each of its calls takes seconds. Script: `benchmarks/integration/qaoa_bench.py`.

| Qubits | Original 0.2.0 | Patched (default) | Patched (`fuse_diagonal_gates=True`) | Max error vs Qiskit Statevector |
|---|---|---|---|---|
| 10 | 3 231 ms / call | 0.46 ms | 0.38 ms | ≤ 4.4e-16 |
| 15 | 42 081 ms / call | 15.4 ms | 6.5 ms | ≤ 3.1e-15 |

Peak RSS fell from about 165 MB to about 104 MB (patch 0009).

## Not verified

- **CUDA build and GPU tests.** There is no GPU or toolkit on this host. No `.cu` file was changed. The CUDA engine does not fuse, which is why fusion is off by default.
- **`scripts/verify.sh` end to end.** Its clang-tidy stage and wheel smoke test were not run.
- **Windows/MSVC.** The AVX2 clones are GCC-only, so MSVC compiles the plain loops. The supplied `RESULTS.md` measured that as slower on Windows.

## Owner decision still open

- Whether to make diagonal fusion the default. That needs a matching CUDA kernel, or accepting that a fused CPU run is no longer bitwise identical to a CUDA run.
