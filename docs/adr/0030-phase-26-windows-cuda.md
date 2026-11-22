# ADR 0030: Phase 26 — CUDA on native Windows

## Status

**Accepted design; implemented and verified locally; the gate is open.** The owner started Phase 26 on 2026-11-20 and answered section 9 the same day (see "Owner answers"). Branch `phase-26-windows-cuda`, from `main` at `1a2af73`. The gate (section 8) is proposed; the owner confirms it. The verification of 2026-11-21/22 and what remains for the gate are in section 12.

This ADR supersedes two parts of `docs/adr/0022` (Phase 18, frozen and kept as written): its section 7 item "CUDA on Windows (refused at configure time)" and its gate criterion 6, the configure-time refusal. Every other Phase 18 decision stands.

Labels follow `docs/adr/0014`–`0029`: **[MEASURED]**, **[DERIVED]**, **[MENOS]** (a design decision) and **[OPEN]** (an owner decision, section 9).

---

## 1. Context

`menos` 1.3.0 was published on 2026-11-18 with ten wheels: CPython 3.10–3.14 for manylinux x86-64 and for `win_amd64`. The Linux wheels are built with `-DMENOS_ENABLE_CUDA=ON` against CUDA 12.9 and carry `libcudart`; the Windows wheels are CPU-only, because `CMakeLists.txt` stops any MSVC configuration with `MENOS_ENABLE_CUDA=ON` (`docs/adr/0022` section 7). The owner asked for CUDA-enabled Windows wheels on 2026-11-20.

### 1.1 MENOS at `1a2af73` — [MEASURED] from the source

| Item | Where | State |
|---|---|---|
| The refusal | `CMakeLists.txt` | `if(MSVC AND MENOS_ENABLE_CUDA)` is a `FATAL_ERROR`. Nothing else in the tree asserts that Windows is CPU-only: no test, no stage of `scripts/verify-windows.ps1`, no Python code. |
| nvcc host flags | `src/kernels/cuda/CMakeLists.txt`, `benchmarks/CMakeLists.txt` | `-Xcompiler=-Wall,-Wextra,…,-Werror`, GCC spellings. `-fmad=false` is an nvcc flag and is portable. |
| Runtime version check | `CMakeLists.txt` | Matches `libcudart.so.N` against the headers' major version. On Windows the library is `cudart.lib`, which has no version in its name, so the check would pass without checking anything. |
| GCC-only source | `src/kernels/cuda/dense_kernels.cu` | One construct: `__builtin_popcountll` in host code (`expectation_of_term`). Device code uses `__popcll`, a CUDA intrinsic. No POSIX header, no `long` used as a 64-bit type, no other compiler extension in the CUDA kernels, the CUDA engine or their tests. |
| Device detection | `src/kernels/cuda/device_info.cpp` | Any failure of `cudaGetDeviceCount`, `cudaGetDeviceProperties` or `cudaMemGetInfo` gives `present = false` and never throws, so "no driver", "no device" and "driver too old" all leave the CPU engines. Not Linux-specific. |
| Device pool | `src/kernels/cuda/device_pool.cpp` | `pooled = true` needs `cudaDevAttrMemoryPoolsSupported`, `cudaMemPoolCreate` (`cudaMemHandleTypeNone`) and `cudaMemPoolSetAttribute`; any failure falls back to `cudaMalloc` with `pooled=false`. |
| Runtime DLLs | `CMakeLists.txt`, `src/python/CMakeLists.txt` | OpenBLAS is copied beside every executable at configure time and installed beside `_core.pyd`. Nothing does this for the CUDA runtime. |
| Memory baseline | `tests/memory/data/baseline_windows.json` | No `dense_cuda_*` fixture (the Linux baseline has them), so `memory_accounting_test.cpp` fails on a Windows machine with a GPU once CUDA is built. |
| CUDA benchmarks | `benchmarks/cuda_*_benchmark.cpp` | `std::getenv` without the `_CRT_SECURE_NO_WARNINGS` guard the CPU benchmarks carry; an error under `/W4 /WX`. |

## 2. Scope

### 2.1 In scope

1. The existing CUDA dense backend built natively on Windows x64 with MSVC: presets `windows-cuda-dev` and `windows-cuda-release`.
2. The `win_amd64` wheels built with CUDA compiled in, as the Linux wheels are: the "auto" wheel, which offers the GPU only when a device is present at run time.
3. Verification on a Windows GPU, and in the Windows wheel job of the workflow on a runner with none.

### 2.2 Out of scope

Publishing to PyPI (the owner's separate instruction); Windows ARM64; other Windows toolchains; multi-GPU; cuQuantum; any change to a kernel, to a summation order, to the one synchronisation per gate, to the device pool or to the planner.

## 3. Compiling the device code with MSVC — [MENOS]

MENOS's C++ is built with the MSVC 14.50 toolset, pinned because 14.51 miscompiles it (`docs/adr/0022` section 3.1). nvcc 12.9 refuses 14.50 (section 10.1). Under the owner's answer to Q5:

- **The `.cu` files alone are compiled with a Visual Studio 2022 toolset (14.44) as nvcc's host compiler.** There are two such files: `src/kernels/cuda/dense_kernels.cu` and the benchmark-only `benchmarks/cuda_probe_kernels.cu`. Every `.cpp` file, including the CUDA host files (`device_buffer.cpp`, `device_info.cpp`, `device_pool.cpp`, `device_benchmarks.cpp`, `cuda_dense_engine.cpp`), and the link stay on 14.50. `-allow-unsupported-compiler` is not used.
- **How CMake is told.** `CMAKE_CUDA_HOST_COMPILER`, which CMake also reads from the `CUDAHOSTCXX` environment variable, names the 14.44 `cl.exe`. With MSVC and `MENOS_ENABLE_CUDA=ON`, a configuration that names no host compiler is refused before `enable_language(CUDA)`, with a message naming this section, rather than left to fail with nvcc's own error.
- **The host compiler's own standard library.** nvcc runs inside the 14.50 developer environment, whose `INCLUDE` names the 14.50 standard library, which the 14.44 compiler cannot read. The 14.44 toolset's `include` directory, derived from the host compiler's path, is therefore put on the include path of the CUDA translation units, which is searched before `INCLUDE`. The Windows SDK and UCRT headers still come from the environment and are common to both toolsets.
- **Linking two toolsets.** The 14.x toolsets are binary compatible, and Microsoft supports linking objects from an older 14.x toolset with a newer one's linker and runtime, which is the direction used here.
- **Warnings.** The `.cu` files get MSVC's bar, `-Xcompiler=/W4,/WX,/permissive-,/Zc:__cplusplus`, in place of the GCC list. Any warning that has to be excluded is recorded in section 10 with the measurement, as `-Wpedantic` and `-Wold-style-cast` are on Linux.

Source changes are limited to what MSVC needs and change no arithmetic: `__builtin_popcountll` becomes `std::popcount` (what the CPU primitive already uses for the same phase), a missing `<algorithm>` include, a `16UL` literal written as `std::size_t{16} << 20`, and the `_CRT_SECURE_NO_WARNINGS` guard in the five CUDA benchmarks.

## 4. Toolkit and runtime — [MENOS]

- **Toolkit.** CUDA 12.9.1, the release the Linux wheels are built with (their bundled runtime is `libcudart.so.12.9.79`). `scripts/fetch-cuda-windows.ps1` downloads NVIDIA's redistributable archives `cuda_nvcc` 12.9.86 and `cuda_cudart` 12.9.79 for `windows-x86_64`, checks them against the SHA-256 values in NVIDIA's manifest `redistrib_12.9.1.json`, and unpacks them into `build/_deps/cuda-12.9.1`, in the manner of `scripts/fetch-openblas-windows.ps1`. Nothing is installed system-wide and no driver is touched. The same script serves the developer machine and the workflow.
- **Runtime version check.** On Windows the configure step requires `cudart64_<major>.dll` in the toolkit's `bin` directory, where `<major>` is the headers' major version. This is the twin of the Linux `libcudart.so.N` check (`docs/adr/0014` section 8, trap 2).
- **Runtime at run time.** `CUDA::cudart` is the import library of `cudart64_12.dll`, so the DLL is a load-time dependency. It is copied beside every executable at configure time, and for an editable install or a plain `uv build` it is installed beside `_core.pyd`, exactly as `libopenblas.dll` is (`docs/adr/0022` sections 4 and 5). NVIDIA's licence ships as `menos/CUDA_LICENSE`. The published wheels carry it differently (section 5). The GPU driver's `nvcuda.dll` is loaded by the runtime only when a device is asked for, so the wheel imports and runs on the CPU on a machine with no NVIDIA driver.
- **Architectures.** The wheels are built for the Linux wheels' list, `75;80;86;89;90;120` (Turing through Blackwell); the `windows-cuda-*` presets build for 89, the development GPU's.

## 5. Wheels and version — [MENOS]

`[tool.cibuildwheel.windows]` in `pyproject.toml` turns CUDA on (`CMAKE_ARGS`, `CUDACXX`, `CUDAARCHS`); `CUDA_PATH` and `CUDAHOSTCXX` come from the environment that `scripts/verify-windows.ps1` and the workflow set up. The test command stays the Python suite; the backend check (`scripts/check_wheel_backends.py cuda`, with `--gpu` where a device is present) is added by the caller, as the Linux job does.

**Vendored runtime DLLs (changed during the work, section 10.4).** As first designed, the wheel carried `libopenblas.dll` and `cudart64_12.dll` beside `_core.pyd` under their own names, and delvewheel was told to leave them alone, which is how the 1.3.0 wheel carries OpenBLAS (`docs/adr/0022` section 5). That is unsafe: Windows binds an import to any already loaded DLL of the same file name, and `qiskit-aer` ships its own `libopenblas.dll`. When Aer is imported before `menos`, MENOS's LAPACK calls go to Aer's library and the process dies with an access violation. The published wheels therefore let delvewheel vendor OpenBLAS and the CUDA runtime into `menos.libs` under hashed names, as it already does for the MSVC C++ runtime, and rewrite `_core.pyd`'s imports to match; CMake does not install the two DLLs itself in that build (`MENOS_INSTALL_RUNTIME_DLLS=OFF`, default `ON`). The licences are installed either way. This changes the Phase 18 wheel layout for OpenBLAS as well, and fixes a fault that the published 1.3.0 Windows wheel has.

The 1.3.0 `win_amd64` file names are taken on PyPI and cannot be replaced, so the version becomes **1.3.1** on every platform. Nothing is published by this phase.

## 6. Verification — [MENOS]

- `scripts/verify-windows.ps1` gains a CUDA stage: fetch the toolkit, find the 14.44 host compiler, then configure, build and `ctest` `windows-cuda-dev` and `windows-cuda-release`, and build the CUDA benchmarks. The CUDA `ctest` runs are not parallel, because the CUDA tests share one device and several assert on its free memory. With `-Wheels` the wheels are built with CUDA and each must carry `menos/cudart64_12.dll`.
- The workflow's Windows job installs the 14.44 toolset beside 14.50, fetches the toolkit, builds and tests `windows-cuda-release` as well as `windows-release`, and asserts the runtime DLL and licence in every wheel. Its runner has no GPU and no NVIDIA driver, which is the "imports and runs on the CPU" case.
- `tests/memory/data/baseline_windows.json` gains the `dense_cuda_*` fixtures, recorded on the development GPU.

## 7. Risks

- **Two toolsets in one build.** Supported by Microsoft in the direction used, but new to this project. The evidence is the bitwise CPU/CUDA suite on the GPU (gate criteria 2 and 3).
- **AddressSanitizer in `windows-cuda-dev`.** The `.cu` object is compiled by a different toolset from the ASan runtime it links to. Measured in section 10.
- **WDDM.** A Windows display GPU shares its memory with the desktop. The allocation race test asks for far more than the device holds, the free-memory assertions allow a fixed slack, and a kernel that runs past the WDDM timeout is reset by the driver. Measured in section 10; none is changed pre-emptively.
- **The device pool's static destructor** runs while a `.pyd` is being unloaded at Python exit. Measured in section 10.
- **Minimum driver.** CUDA 12.x on Windows needs driver 528.33 or later. A machine with an older driver gets the CPU engines, as on Linux.

## 8. Gate (proposed)

1. `windows-cuda-dev` and `windows-cuda-release` configure and build with nvcc 12.9 and MSVC, warnings as errors.
2. The C++ suite passes on both, with the CUDA tests executing on a Windows GPU; the counts of executed and skipped tests are recorded.
3. CPU and CUDA results are bitwise identical on Windows wherever they are on Linux: the differential test passes, and the CUDA fixtures recorded in `tests/memory/data/baseline_windows.json` have the CPU fixtures' statevector and counts digests. *(Corrected on 2026-11-20 before any gate run, `docs/adr/0030` section 10.3: as first written it asked for every digest, but a CUDA expectation value is summed from per-block partials and its digest differs from the CPU one on Linux too, by design.)*
4. Wheels for CPython 3.10–3.14 `win_amd64` carry the CUDA runtime; `scripts/check_wheel_backends.py cuda --gpu` passes on a Windows GPU; on the GPU-less CI runner the wheels import, run on the CPU and refuse a forced CUDA run.
5. The CPU-only Windows build is unchanged: `windows-dev` and `windows-release` pass and the planner goldens do not move.
6. `scripts/verify-windows.ps1` and `scripts/verify.sh` exit 0 on the final commit.

## 9. Owner decisions

- **Q1 — How the deferral is lifted.** [OPEN] A new phase with its own ADR, or an addendum to ADR 0022.
- **Q2 — CUDA toolkit.** [OPEN] 12.9 as on Linux, or 13.x.
- **Q3 — Release.** [OPEN] Build only; or build and publish 1.3.1; or a build-tagged 1.3.0 Windows wheel.
- **Q4 — Where the GPU check runs.** [OPEN] A toolkit on the development machine, or the workflow only.
- **Q5 — nvcc and MSVC 14.50.** [OPEN] Raised by the section 10.1 measurement: a Visual Studio 2022 toolset for the `.cu` files only; `-allow-unsupported-compiler`; CUDA 13.x; or stop.

### Owner answers (2026-11-20)

- **Q1:** a new phase, Phase 26, with ADR 0030, on branch `phase-26-windows-cuda`. The proposed interoperability roadmap is renumbered from Phases 26–31 to Phases 27–32.
- **Q2:** CUDA 12.9, the same as the Linux wheels. If nvcc 12.9 rejected MSVC 14.50, the work was to stop and report, not switch toolkit or override the check. It did (section 10.1), and Q5 followed.
- **Q3:** version 1.3.1, built and verified only. Nothing is uploaded to PyPI in this phase.
- **Q4:** the toolkit on the development machine (RTX 4060 Laptop GPU, driver 566.14). *As built, the toolkit is unpacked into the build tree from NVIDIA's redistributable archives (section 4) instead of installed system-wide.*
- **Q5:** the Visual Studio 2022 toolset for the `.cu` files only. `-allow-unsupported-compiler` is not used, and was not run even as a measurement.

## 10. Evidence

### 10.1 Step 1: nvcc 12.9 against MSVC 14.50 — [MEASURED] 2026-11-20

`scripts/fetch-cuda-windows.ps1` unpacked nvcc 12.9.86 (`Cuda compilation tools, release 12.9, V12.9.86`, build `cuda_12.9.r12.9/compiler.36037853_0`). In the Visual Studio developer environment with `-vcvars_ver=14.50` (`cl` 14.50.35717), a one-kernel file compiled with `nvcc -std=c++20 -arch=sm_89 -fmad=false` stopped at:

```
include\crt/host_config.h(170): fatal error C1189: #error:  -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
```

The check is `#if _MSC_VER < 1910 || _MSC_VER >= 1950`; MSVC 14.50 is `_MSC_VER` 1950. Under the owner's Q2 answer the work stopped here, and Q5 was put to the owner.

### 10.2 Building with the 14.44 host compiler — [MEASURED] 2026-11-20

The MSVC 14.44 toolset (14.44.35207, component `Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64`) was installed beside 14.50.35717 in the same Visual Studio 2026 Build Tools instance. Working tree on `phase-26-windows-cuda`, uncommitted at the time of these runs.

- **Configure.** `cmake --preset windows-cuda-release` reports `The CUDA compiler identification is NVIDIA 12.9.86 with host compiler MSVC 19.44.35224.0` and `The C compiler identification is MSVC 19.50.35726.0`, then `menos: CUDA enabled -- nvcc 12.9.86 … runtime …/bin/cudart64_12.dll, architectures 89`.
- **`windows-cuda-release` build.** 266 of 266 steps, no warning and no error, with `/W4 /WX` on the C++ and `-Xcompiler=/W4,/WX,/permissive-,/Zc:__cplusplus` on the `.cu` file. No warning had to be excluded for nvcc's generated code.
- **`windows-cuda-dev` build (AddressSanitizer).** The first attempt failed at the first link that pulls in the device object:

  ```
  menos_kernels_cuda.lib(dense_kernels.cu.obj) : error LNK2038: mismatch detected for 'annotate_string': value '0' doesn't match value '1' in memory_probe.cpp.obj
  menos_kernels_cuda.lib(dense_kernels.cu.obj) : error LNK2038: mismatch detected for 'annotate_vector': value '0' doesn't match value '1' in memory_probe.cpp.obj
  ```

  The C++ is compiled with `/fsanitize=address` and the `.cu` host pass was not. CMake's default CUDA Debug flags for an MSVC host are also `-Ob0 -Od /RTC1`, and AddressSanitizer rejects `/RTC1`. The preset now sets `CMAKE_CUDA_FLAGS` to `-D_WINDOWS -Xcompiler=/EHsc,/fsanitize=address` and `CMAKE_CUDA_FLAGS_DEBUG` to `-Xcompiler=/Ob0,/Od`, the C++ flags of `windows-dev`. With that the build completes with no warning, the 14.44-compiled instrumented object linking against the 14.50 AddressSanitizer runtime.

### 10.3 The C++ suite on a Windows GPU — [MEASURED] 2026-11-20

Machine: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, compute capability 8.9), driver 566.14, Windows 11. `ctest --preset windows-cuda-release`, not in parallel, 1655 s.

| | Count |
|---|---|
| Tests | 1068 |
| Passed | 1004 |
| Skipped | 54 |
| Failed | 10 |

- **The 10 failures** were the ten `dense_cuda_*` fixtures of `MemoryFixtureTest`, each with `… is not in …/baseline_windows.json`: the baseline had no CUDA fixture (section 1.1). After the baseline was extended (below) the ten pass.
- **The 54 skips.** 52 are `MemoryRssTest`, which on Windows reports resident memory and does not assert it (the Phase 21 rule; they skip in the CPU-only build too). `CudaBackendRequestTest.ForcedCudaWithoutAUsableDeviceIsUnsupported` runs only on a machine with no device. `CudaExecutionMonitorTest.TenAbandonsOfAnOversubscribingTotalNeverExhaustTheDevice` skipped itself with "this device has 7441743872 bytes free, so ten 512 MiB states would fit without any release -- the test could not detect a leak here".
- **CUDA tests that executed and passed** include the bitwise ones: all five `CudaSpecialisedKernelsTest` cases (specialised kernels bitwise the generic kernel), the eleven `CpuCudaDifferentialTest` cases, the six `CudaDevicePoolTest` cases (`pooled=true` on this WDDM device; the allocation race test, `ADeviceThatRanOutSinceThePlanIsARefusalNamingTheBytes`, passed in 0.59 s), the `CudaBackendSelectionTest` cases including the calibrated crossover, and the `CudaMemoryPoolTest` and `CudaPeakMemoryTest` cases.

**The Windows memory baseline.** `menos_memory_probe --write-baseline`, run from the `windows-cuda-release` build, wrote 60 fixtures: the 49 already in `tests/memory/data/baseline_windows.json`, every one unchanged, and the 11 `dense_cuda_*` fixtures. For each CUDA fixture the statevector digest and the counts digest equal the corresponding CPU fixture's Windows digest (`8a8884b93d3f90f1`, `83183ccbdaba0ea5` fused; counts `9babf4d634cd516f`, `2b3789cb8ffddf17`, `ae6764b6b43a5e5e`). The CUDA expectation digests (`d0d005e315d7cc90`, `34122c5fbe33b49f` fused) differ from the CPU ones, as they do in `baseline_linux.json`: the device sums an expectation value from per-block partials, a different order from the CPU kernel. They also differ from the Linux CUDA digests, because the Windows state itself already differs from the Linux one in the CPU build (the two platforms' CPU statevector digests differ in the existing baselines). Gate criterion 3 as first written asked for every CUDA digest to equal the CPU one; that was wrong on Linux too, and it was corrected to statevector and counts before any gate run.

### 10.4 Wheels: a DLL name clash with qiskit-aer — [MEASURED] 2026-11-20

The first wheel built as designed in section 4 (cp310, runtime DLLs beside `_core.pyd`) passed `scripts/check_wheel_backends.py cuda --gpu` ("OK: CUDA wheel ran the dense engine on the GPU") and then failed four of the Python suite's tests (392 passed, 9 skipped): `test_docs_snippets.py` for `build/noise-models.md#0`, `#1` and `build/readout-mitigation.md#0`, and `test_examples.py` for `noise_density_matrix`. Each runs a script in a fresh interpreter, and each exited with status 3221225477 (0xC0000005, access violation).

Reproduced outside the suite with a 2-qubit circuit and a readout-error noise model:

| Wheel | Import order | Job | `libopenblas` modules in the process | Result |
|---|---|---|---|---|
| 1.3.1 CUDA, DLLs beside `_core.pyd` | `qiskit_aer` first | noisy, no budget | `qiskit_aer\backends\libopenblas.dll` only | access violation |
| same | `menos` first | noisy, no budget | `menos\libopenblas.dll` | correct counts |
| same | `qiskit_aer` first, GPU hidden (`CUDA_VISIBLE_DEVICES=""`) | noisy | Aer's only | access violation |
| **1.3.0 from PyPI (CPU-only)** | `qiskit_aer` first | noisy, no budget | Aer's only | correct counts |
| **1.3.0 from PyPI (CPU-only)** | `qiskit_aer` first | noiseless, `time_budget_seconds=30` | Aer's only | **access violation** |
| 1.3.0 from PyPI | `menos` first | noiseless, `time_budget_seconds=30` | `menos\libopenblas.dll` | correct counts |

So the fault is not in the CUDA code and not on the device. `qiskit-aer` 0.17.2 ships `qiskit_aer/backends/libopenblas.dll`; once it is loaded, Windows resolves `_core.pyd`'s import of `libopenblas.dll` to it, and MENOS's copy is never loaded. The first MENOS job that reaches LAPACK then crashes. In the CPU-only 1.3.0 wheel that needs a job that runs the hardware calibration (a time budget, for example); a CUDA build calibrates on ordinary jobs, which is why the existing tests found it only now. **The published 1.3.0 Windows wheel has this fault.**

With the section 5 change (delvewheel vendors `libopenblas-<hash>.dll` and `cudart64_12-<hash>.dll` into `menos.libs`), the same three `qiskit_aer`-first and `menos`-first jobs all return correct counts, and the process holds both `menos.libs\libopenblas-0486735c359a67419d8832e3c40e2837.dll` and Aer's `libopenblas.dll`.

### 10.5 The wheels — [MEASURED] 2026-11-20

`cibuildwheel` 4.2.1 on the development machine, with `CIBW_TEST_COMMAND_WINDOWS` set to `scripts/check_wheel_backends.py cuda --gpu` followed by the Python suite, built five wheels for architectures `75;80;86;89;90;120`, working tree uncommitted:

| Wheel | Size | Backend check on the RTX 4060 | Python suite |
|---|---|---|---|
| `menos-1.3.1-cp310-cp310-win_amd64.whl` | 13.9 MB | OK: CUDA wheel ran the dense engine on the GPU | 396 passed, 9 skipped |
| `menos-1.3.1-cp311-cp311-win_amd64.whl` | 13.9 MB | OK | 639 passed |
| `menos-1.3.1-cp312-cp312-win_amd64.whl` | 13.9 MB | OK | 639 passed |
| `menos-1.3.1-cp313-cp313-win_amd64.whl` | 13.9 MB | OK | 639 passed |
| `menos-1.3.1-cp314-cp314-win_amd64.whl` | 14.1 MB | OK | 639 passed |

Each wheel carries `menos.libs/cudart64_12-<hash>.dll`, `menos.libs/libopenblas-<hash>.dll`, the two MSVC runtime DLLs, `menos/CUDA_LICENSE` and `menos/OPENBLAS_LICENSE`, and no plainly named runtime DLL. The four tests that failed in section 10.4 pass in every wheel.

### 10.6 `scripts/verify-windows.ps1`: four runs that did not finish — [MEASURED] 2026-11-20/21

Working tree uncommitted on `phase-26-windows-cuda`. None of these runs reached the end; each found something, and two changed the script.

| Run | How | Outcome |
|---|---|---|
| 1 | as written, 22 tests at once | `windows-dev` (AddressSanitizer) failed: the time-model calibration's repeatability checks saw spreads of 17× and 12.8× against their 4× limit, with 22 calibrating tests running together |
| 2 | `NUMBER_OF_PROCESSORS=4` in the environment, the Phase 24 method (`docs/adr/0028` section 22.5) | the same failure: on this Windows build (26300) the variable reads 22 in the process and in its children whatever it is set to, so the override no longer does anything |
| 3 | the new `-Jobs 4` | `windows-dev` 978/978, `windows-release` 978/978, benchmarks built; then `windows-cuda-dev` failed one memory fixture (`dense_statevector`, tracked 4194903 bytes against an estimate of 4194840) |
| 4 | CUDA stages only, from the repository's true path | `windows-cuda-dev` stopped at 833 of 1068 tests with five failures (section 11) |

**`-Jobs`.** The script takes `-Jobs N`, the number of CPU-preset tests `ctest` runs at once, defaulting to every hardware thread as before. The CUDA presets stay sequential.

**Run 3's failure was not a test failure but a wrong build.** The run was started from `C:\Users\janak\menos`; the directory on disk is `C:\Users\janak\MENOS`. CMake compared `CMAKE_CUDA_COMPILER` with the cached value, found the path changed, deleted the cache and configured again without the preset's variables. The result had no CUDA and no AddressSanitizer, so the heap tracker, which a sanitizer build disables, was live in a build whose baseline expects it off. The script now reads `MENOS_ENABLE_CUDA:BOOL=ON` back from each CUDA preset's cache after configuring and fails the stage if it is not there. The verification is to be started from the path as it is on disk.

### 10.7 Not yet run

At the time of writing (2026-11-21): `ctest --preset windows-cuda-dev` (AddressSanitizer) to the end on the GPU; `scripts/verify-windows.ps1` end to end; `scripts/verify.sh` in WSL, including the section 11 change on `cuda12-dev` and `cuda12-release`; the workflow on a GPU-less runner (gate criterion 4's second half); a second Windows GPU or any architecture other than 8.9.

*Since run (section 12): the first three. Still not run: the workflow on a GPU-less runner, and a second Windows GPU or another architecture.*

## 11. The CUDA time model: host gate preparation — [MEASURED] 2026-11-21

This section changes what the CUDA side of the backend comparison is charged. Phase 16 requires an ADR for a change to the CUDA calibration method (`docs/adr/0020`); this is that record. No new quantity is calibrated: an existing measured rate is applied to a count the CUDA engine did not report.

### 11.1 The failure

`ctest --preset windows-cuda-dev` (AddressSanitizer, RTX 4060 Laptop GPU, driver 566.14) failed five tests of the first 833, all for one reason, a tiny circuit's dense candidate planned onto the GPU:

- `CliGoldenTest.ExplainProducesExactDeterministicOutput`, `CliGoldenTest.RunProducesExactOutputAfterNormalizingTimings` and a third `CliGoldenTest` case (a Bell circuit, 100 shots);
- `CudaBackendSelectionTest.TheCalibratedProfileKeepsTinyCircuitsOnTheCpuAndLargeOnesOnTheGpu` (3 qubits, statevector);
- `ExplainTest.MatchesTheDocumentedReportShape` (3 qubits, 10 shots and a statevector).

`windows-cuda-release` does not fail them (section 10.3). The same symptom on Linux was the subject of `docs/adr/0028` section 23, whose fix added the host sampling and the per-execution cost.

### 11.2 Predicted against measured

Predictions, five processes each, in milliseconds:

| Build | Circuit | CPU predicted | CUDA predicted | Chosen |
|---|---|---|---|---|
| AddressSanitizer | Bell, 100 shots | 2.75–2.90 | 1.90–2.08 | CUDA |
| AddressSanitizer | 3 qubits, 10 shots + statevector | 3.94–9.43 | 0.85–2.09 | CUDA |
| release | Bell, 100 shots | 0.053–0.132 | 0.142–0.229 | CPU |
| release | 3 qubits, 10 shots + statevector | 0.060–0.065 | 0.190–0.288 | CPU |

Measured, execution and sampling together, medians in two AddressSanitizer processes:

| Circuit | CPU | CUDA |
|---|---|---|
| Bell and a T gate, 100 shots | 3.57 / 4.20 | 4.39 / 5.80 |
| 3 qubits, 10 shots + statevector | 1.66 / 1.84 | 2.20 / 3.37 |
| 3 qubits, statevector (the selection test's circuit) | 5.23 / 5.84 | 8.07 / 8.98 |

In release the CPU took 0.016–0.047 ms and the device 0.22–0.77 ms. So the CPU is the faster backend in both builds, and the planner's choice in the sanitizer build was wrong: the CUDA prediction there was 4 to 9 times too low.

### 11.3 The missing term

`CudaDenseEngine::execute` prepares every gate pass on the host: with fusion on, `ir::plan_fusion` and `engines::dense::apply_blocks` build each block's matrix with the code the CPU engine runs, and only then is the kernel launched; with fusion off, `ir::generate_matrix` per gate. `cuda_seconds` priced none of it. On the selection test's 3-qubit circuit (19 gates fused into 4 blocks, no output), in two AddressSanitizer processes:

| | ms |
|---|---|
| Host block building alone (the plan and 4 blocks, nothing applied) | 5.31 / 5.00 |
| CPU engine, whole run | 6.04 / 5.02 |
| CUDA engine, whole run | 6.74 / 5.77 |
| CUDA engine, the same register with no gates | 0.30 / 0.25 |
| CUDA prediction (execution + 5 launches) | about 0.43 / 0.46 |
| CPU prediction (5 passes at 1.07–1.13 ms) | about 5.3 / 5.6 |

Block building is 80–90% of the run on either backend under AddressSanitizer. In release it is 0.023–0.027 ms, against a CUDA run of 0.21–0.35 ms and a CPU run of 0.014–0.038 ms, which is why release never showed it. The CPU model already carries this cost: it is in the per-pass rate the time model fits (`docs/adr/0028` section 20), 1.07–1.13 ms per pass in this sanitizer build and 0.0054–0.0058 ms in release.

### 11.4 The fix (owner, 2026-11-21: option 1)

`CudaDenseEngine::estimate` reports, in the CPU dense engine's time component 0, **the pass count `DenseEngine` reports for the same request**: initialisation, one per gate pass and one per output pass. `cuda_seconds` already prices a CUDA estimate's host terms at the CPU dense engine's measured rates, so nothing else changes: no new profile figure, no change to the calibration probes, the plan-cache key or `explain()`'s format.

The CUDA figure is then the CPU's pass term plus the device's own execution, launch, transfer and work terms. The device wins only where its work rate pays for those, which does not depend on two independently timed figures agreeing.

Three options were put to the owner:

1. The CPU's whole pass count (chosen).
2. The gate passes only, which is strictly what the engine does on the host. It predicts about 4.7 ms against the CPU's 5.3 ms on the circuit above, so the planner would still choose CUDA: the CPU model averages its per-pass cost over the initialisation and output passes too, which overprices the CPU by about one pass.
3. A separately calibrated per-pass cost for the CUDA engine. It compares two figures that are really within about 12% of each other in the sanitizer build, where timings swing up to 2× between runs. A variant of it was tried and reverted in Phase 24 (`docs/adr/0028` section 23.2).

**What option 1 over-counts, knowingly.** The device's initialisation is a launch and its statevector output is a copy, neither a host pass, yet each is charged one CPU pass: about 1 ms each in the sanitizer build and about 0.006 ms in release. The device work rate's probe (20 qubits, 2 layers) still has its own host block building inside the measured work time, so that share is counted in the rate and again in the pass term; both errors make the device look slower, never faster.

A profile built by hand has no dense time figures, so every test that sets the CUDA figures itself plans as before.

**Tests.** `CudaBackendSelectionTest.TheCudaEngineReportsTheCpuDenseEnginesPassCount` holds the two engines' pass counts equal over fusion on and off and every output combination, and the work and shot components at zero unless counts are sampled on the host. `CudaBackendSelectionTest.HostGatePreparationKeepsATinyCircuitOnTheCpu` pins the decision on a synthetic profile with the sanitizer build's measured figures, rounded (a pass 1 ms, the execution cost 0.2 ms, a launch 0.04 ms): 3 qubits stay on the CPU, 16 qubits still go to the device. Without the pass term that profile prices the device at about 0.4 ms against the CPU's 6 ms.

### 11.5 After the fix

Working tree uncommitted, both presets rebuilt, RTX 4060 Laptop GPU.

- `ctest -R "CliGoldenTest|CudaBackendSelectionTest|FactorizedRegionBackendTest|ExplainTest" --repeat until-fail:3`, 55 tests, each three times: the five tests of section 11.1 passed three times out of three in `windows-cuda-dev` (AddressSanitizer, 17 798 s) and in `windows-cuda-release` (364 s). So did every other test selected except the new synthetic-profile test, which failed in both builds as first written: its profile made the device's fixed costs almost nothing, so the device's faster work rate honestly won the 3-qubit circuit. The profile was wrong, not the model, and it now carries the measured execution and launch costs.
- The two new tests and `AFasterDeviceWinsTheDenseRegion` then pass in both presets.

Not yet run after the fix, at the time of writing: the rest of `windows-cuda-dev`, the full `scripts/verify-windows.ps1`, and `cuda12-dev`/`cuda12-release` in WSL (section 10.7). All three were then run and passed (section 12).

## 12. Verification — [MEASURED] 2026-11-21/22

### 12.1 What was verified

Both platforms ran on one set of files: the working tree of `phase-26-windows-cuda` on top of `1a2af73`, uncommitted at the time, git tree `51de5ca6d30d3a33d87b4fa475fc9651b2a1ad8c`. The WSL clone received it as a patch applied to a detached `1a2af73`, and `git write-tree` there gave the same hash before `scripts/verify.sh` started.

The owner then committed the 37 modified files as `ecb7796` ("building cuda on widows"). That commit plus the two files it leaves out, this ADR as it stood and `scripts/fetch-cuda-windows.ps1`, is the verified tree: `git diff 51de5ca6 ecb7796` shows only those two files, as deletions. This section and the edits to the status, section 10.7 and section 11.5 were written after the runs; they change no source, test, script or build file.

Machine: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, compute capability 8.9), driver 566.14, Windows 11 build 26300. The two verifications ran one after the other, never together.

### 12.2 `scripts/verify-windows.ps1 -Jobs 4 -Wheels`: two runs

`cl` 14.50.35717 for the C++, 14.44.35207 as nvcc's host compiler, nvcc 12.9.86, started from `C:\Users\janak\MENOS`.

**Run 5** (2026-11-21 22:37 to 2026-11-22 07:31) passed every stage up to and including the last test stage, then failed:

| Stage | Tests | Passed | Skipped | Time |
|---|---|---|---|---|
| `windows-dev` (AddressSanitizer), 4 at once | 978 | 978 | 43 | 5 877 s |
| `windows-release`, 4 at once | 978 | 978 | 41 | 684 s |
| benchmarks build (`windows-release`) | | built | | |
| `windows-cuda-dev` (AddressSanitizer), on the GPU, sequential | 1070 | 1070 | 56 | 23 701 s |
| `windows-cuda-release`, on the GPU, sequential | 1070 | 1070 | 54 | 1 412 s |
| benchmarks configure (`windows-cuda-release`) | | **failed** | | |

CTest counts a skipped test as passed. The failure was `fatal: unable to access 'https://github.com/google/benchmark.git/': Could not resolve host: github.com`, three times, in FetchContent's clone of google_benchmark: a name-resolution outage on the machine, not a fault in MENOS. `build/windows-cuda-release` had no google_benchmark source yet, so this was its first clone; github.com resolved again when checked after the run.

This is the first time `windows-cuda-dev` reached the end. The five tests of section 11.1 pass in it.

**The skips.** In the CUDA presets 52 are `MemoryRssTest` (reported, not asserted, on Windows) and two are the device tests of section 10.3 (`ForcedCudaWithoutAUsableDeviceIsUnsupported`, which needs a machine with no device, and `TenAbandonsOfAnOversubscribingTotalNeverExhaustTheDevice`, which skips itself on a device with this much free memory). The two more in each sanitizer preset are `PlannerGolden.EveryCorpusPlanMatchesTheRecordedPlan` and `WallTimeReportTest.TwentyFourQubitsTwoHundredOperations`, which do not run under AddressSanitizer; both run and pass in the release presets. The CPU presets have 41 `MemoryRssTest` skips because they have no `dense_cuda_*` fixtures.

**Run 6** (2026-11-22, after the network returned) ran the stages run 5 did not reach, from a copy of the script with the two CPU presets, the CPU benchmarks build and the two CUDA `ctest` stages removed and nothing else changed; it still configured and built both CUDA presets, which were up to date.

| Stage | Result |
|---|---|
| benchmarks build (`windows-cuda-release`) | built |
| editable install and `pytest` (CPython 3.14) | 637 passed, 2 deselected |
| wheel build and smoke test in a clean venv | 2 passed |
| `cibuildwheel` 4.2.1, cp310 | backend check OK on the GPU; 396 passed, 9 skipped |
| cp311, cp312, cp313, cp314 | backend check OK on the GPU; 639 passed each |
| wheel contents (`menos.libs/libopenblas-*`, `cudart64_12-*`, `msvcp140-*`) | present in all five |

It ended `verify-windows: all stages passed`, exit 0. These five wheels replace those of section 10.5, which were built before the section 11 change.

So every stage of `scripts/verify-windows.ps1` has passed on the verified tree, across two runs and not in one. Whether that satisfies gate criterion 6 is the owner's ruling (section 12.5).

### 12.3 `scripts/verify.sh` in WSL: exit 0

One run, 2026-11-22, 5 h 58 min, `UV_PYTHON_PREFERENCE=only-managed`, GCC 15.2.0, clang-format and clang-tidy 21.1.8, CUDA 13.3 (`cuda-*`) and 12.4 with g++-13 (`cuda12-*`). It ended `verify.sh: all stages passed`.

| Stage | Tests | Passed | Time |
|---|---|---|---|
| format check; clang-tidy on the C++, the benchmarks, the bindings, the CUDA targets and the CUDA benchmarks | | clean | |
| `dev` (ASan, UBSan, LSan) | 977 | 977 | 1 546 s |
| `release` | 977 | 977 | 128 s |
| `pytest` (editable install) | | 637 passed, 2 deselected | |
| wheel smoke test | | 2 passed | |
| `cuda-dev` | 1069 | 1069 | 5 619 s |
| `cuda-release` | 1069 | 1069 | 409 s |
| `cuda12-dev`, GPU tests executing | 1069 | 1069 (55 skipped) | 8 446 s |
| `cuda12-release`, GPU tests executing | 1069 | 1069 (2 skipped) | 672 s |

The section 11 change is thereby verified on the Linux device too: `CliGoldenTest`, `CudaBackendSelectionTest` and `PlannerGolden` pass in all four CUDA presets. As on every earlier phase, the CUDA 13 runtime cannot reach this driver, so the GPU tests skip in `cuda-dev` and `cuda-release` (134 and 92 skips) and execute in the `cuda12-*` presets.

**The documentation site was not built**: the stage printed `docs: SKIPPED (node/npm not found; the site was NOT built)`. The Python snippets in the documentation were executed by the `pytest` stage on both platforms.

The `cuda-dev` and `cuda12-dev` stages took about four and six times as long as in an earlier full run in this clone during Phase 24 (1 348 s and 1 452 s, `build/verify-fusion-final4.log`). The cause was not investigated.

### 12.4 The gate criteria against the evidence

| Criterion | Evidence | State |
|---|---|---|
| 1. Both CUDA presets configure and build, warnings as errors | sections 10.2, 12.2 | met |
| 2. The C++ suite passes on both, CUDA tests executing on a Windows GPU, counts recorded | section 12.2: 1070 of 1070 on each, 56 and 54 skipped | met |
| 3. CPU and CUDA bitwise identical where they are on Linux | section 10.3 (the baseline digests); `CpuCudaDifferentialTest`, `CudaSpecialisedKernelsTest` and the `dense_cuda_*` fixtures pass in section 12.2 | met |
| 4. Wheels for CPython 3.10–3.14 carry the CUDA runtime; the backend check passes on a Windows GPU; on a GPU-less runner they import, run on the CPU and refuse a forced CUDA run | section 12.2 for the first two parts | **the GPU-less runner part is not run** |
| 5. The CPU-only Windows build is unchanged | section 12.2: `windows-dev` and `windows-release` 978 of 978; `PlannerGolden` passes in `windows-release`; no file under `tests/plan/golden` differs from `1a2af73` | met |
| 6. `scripts/verify-windows.ps1` and `scripts/verify.sh` exit 0 on the final commit | sections 12.2 and 12.3 | **for the owner's ruling** |

### 12.5 Open for the owner

1. **The GPU-less runner** (criterion 4). `.github/workflows/build-wheels.yml` has not run with this branch; it needs the branch pushed.
2. **The split Windows run** (criterion 6). Run 5 did not exit 0, for a network reason; run 6 completed the stages after it. One uninterrupted run takes about ten hours.
3. **The final commit** (criterion 6). The runs were on the working tree, which is `ecb7796` plus `scripts/fetch-cuda-windows.ps1` and this ADR. The script must be committed in any case: `scripts/verify-windows.ps1` and `.github/workflows/build-wheels.yml` call it, so the workflow fails without it.
4. **The documentation site build**, skipped in WSL for want of Node.

Not covered by any run: a second Windows GPU, or a compute capability other than 8.9.

## 13. The wheel workflow: an OpenBLAS kernel-selection difference — [MEASURED] 2026-11-22

### 13.1 The failure

`.github/workflows/build-wheels.yml` was dispatched on `phase-26-windows-cuda` at `8c14f55` (run 45020858503). The Linux job passed. The Windows job failed in `ctest --preset windows-release`, twice (the job was re-run once): 977 of 978 tests passed, and `MemoryFixtureTest.OutputsMatchTheBaselineAndMemoryIsReported/"mps_expectation"` gave expectation digest `174e7f1963d05ebf` both times against the baseline's `03caedeae1ee0dad`. The job stops there, so no Windows wheel was built and gate criterion 4's GPU-less half was not reached.

### 13.2 The cause

| Run | Commit | Runner CPU (`PROCESSOR_IDENTIFIER`) | Runner image | `mps_expectation` |
|---|---|---|---|---|
| 44155536559, `main`, 2026-11-17 | `1a2af73` | AMD64 Family 25 Model 1 | `windows-2025-vs2026` 20261103.298.1 | passed |
| 45020858503, both attempts | `8c14f55` | AMD64 Family 26 Model 2 | `windows-2025-vs2026` 20261116.315.1 | failed, the same digest |
| the development machine | the verified tree | Intel, OpenBLAS core `Haswell` | | passed |

The compiler is MSVC 14.50.35717 in both workflow runs, OpenBLAS is the hash-pinned 0.3.34, Phase 26 changed no MPS source, and the baseline entry is the one at `1a2af73`. The official OpenBLAS build chooses its kernels from the CPU at load time. On the development machine, the same `windows-release` test binary with `OPENBLAS_VERBOSE=2`:

| `OPENBLAS_CORETYPE` | Core OpenBLAS reports | `mps_expectation` digest |
|---|---|---|
| unset | Haswell | `03caedeae1ee0dad` (baseline) |
| `Haswell` | Haswell | `03caedeae1ee0dad` |
| `Zen` | Zen | `03caedeae1ee0dad` |
| `Sandybridge` | Sandybridge | `bc5f43748f1d98b6` |
| `Nehalem` | Nehalem | `ed219c325f799ff1` |

So the MPS expectation's last bits depend on the kernel set the SVD runs on. Which core OpenBLAS picks on the Family 26 runner was not read (the development machine has no AVX-512 and cannot run those kernels); that it is a different one is inferred from the digest. `main` would fail the same test on that runner: the difference is in the runner fleet, not in this branch. It is also not Windows-only in kind, since the Linux builds take LAPACK from a CPU-dispatching library too.

### 13.3 The decision (owner, 2026-11-22: option 1)

Three options were put to the owner under the Phase 18 rule that a Windows last-bit numeric difference is the owner's and never a tolerance change:

1. Pin OpenBLAS's kernel choice for the test runs only (chosen).
2. Pin it inside the shipped library, which would cost AVX-512 machines their faster kernels.
3. Compare this fixture with a tolerance instead of a digest.

**Built:** the four Windows test presets (`windows-dev`, `windows-release`, `windows-cuda-dev`, `windows-cuda-release`) set `OPENBLAS_CORETYPE=Haswell` in the test environment. Haswell is the core the Windows baseline was recorded on. Nothing in the library, the wheels or a user's process is changed, and no baseline digest moves.

**What this does not change, knowingly.** A result that passes through the SVD still differs in its last bits between machines whose OpenBLAS cores differ; the pin only stops the digest tests from depending on the runner. A machine without AVX2 cannot run the Windows test presets as pinned. The Python suite that `cibuildwheel` runs against each wheel is not pinned.

### 13.4 After the pin

On the development machine `ctest --preset windows-release -R MemoryFixtureTest` passes 40 of 40 with the pin (it selects Haswell there with or without it, so this shows only that nothing moved). Not yet run: the workflow on the runner with the pin.

