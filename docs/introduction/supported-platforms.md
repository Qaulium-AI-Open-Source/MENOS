# Supported Platforms

This page states what the project has actually been verified on. "Intended to work" and "verified" are different claims and are kept apart here.

## Verified

| Component | Platform | Status |
|---|---|---|
| CPU engines, planner, runtime | Linux, x86-64 | **Verified.** The full verification script (`scripts/verify.sh`) passes: C++ unit and integration tests in Debug (with sanitizers) and Release, the Python suite, a wheel build, and a clean-environment wheel smoke test. |
| CPU engines, planner, runtime, CLI, Python package | **Windows, x64, MSVC** | **Verified** (Phase 18). `scripts/verify-windows.ps1` passes: C++ tests in Debug (with AddressSanitizer) and Release built with MSVC 14.50 (Visual Studio Build Tools 2026), the Python suite, a wheel build and a clean-environment smoke test, and `cibuildwheel` wheels for CPython 3.10–3.14, each running the Python suite. |
| CUDA dense backend | Linux, x86-64, NVIDIA GPU, CUDA toolkit **12.0 or newer** | **Verified** with CUDA toolkit 13.4 and driver 595.91.07 (RTX 3050 Laptop, compute capability 8.6), and with CUDA toolkit 12.4 and driver 566.14 under WSL (RTX 4060 Laptop, compute capability 8.9, built for 8.6). CUDA 12.0 to 12.3 are not tested. |
| CUDA dense backend | **Windows, x64, MSVC**, NVIDIA GPU, CUDA toolkit 12.9 | **Verified** (Phase 26, from version 1.3.1) with CUDA toolkit 12.9 and driver 566.14 (RTX 4060 Laptop, compute capability 8.9, built for 8.9): the Release C++ suite with the CUDA tests executing on the GPU, and the wheels' GPU check. The device code is compiled with the MSVC 14.44 toolset ([CUDA Installation](/gpu/cuda-installation#on-windows)). |

## Distribution

| Channel | What it covers |
|---|---|
| PyPI (`pip install menos`, `uv add menos`) | Version 1.3.1: prebuilt wheels for CPython 3.10–3.14 on Linux x86-64 and Windows x64, described in the next two rows. No source distribution is published |
| Linux wheels | CPython 3.10–3.14, **Linux x86-64** (manylinux 2.28), with the **CUDA backend compiled in**: built with the CUDA 12.9 toolkit for compute capabilities 7.5 to 12.0, with the CUDA runtime, OpenBLAS and LAPACKE inside. The GPU is used only when a device is present; otherwise the wheel runs on the CPU engines. Built by the wheel workflow (`.github/workflows/build-wheels.yml`), whose runner has no GPU: each wheel is tested there on the CPU path only |
| Windows wheels | CPython 3.10–3.14, `win_amd64`, with OpenBLAS inside. Version 1.3.0 is **CPU-only**. From version 1.3.1 they have the **CUDA backend compiled in**, built with the CUDA 12.9 toolkit for compute capabilities 7.5 to 12.0 and carrying the CUDA runtime (`cudart64_12.dll`); the GPU is used only when a device is present. Built by the wheel workflow or `cibuildwheel --platform windows` |
| Source build | Everything on this page, including the CUDA backend on Linux and on Windows |

## Intended but not verified

| Platform | Notes |
|---|---|
| Linux, aarch64 | The code has no x86-specific dependencies, but it has not been built or tested. **The CUDA backend is not verified on aarch64.** |
| macOS, arm64 | The host-memory probe has a macOS code path and LAPACK is found through the Accelerate framework. **Not built or tested.** There is no CUDA backend on macOS. |

## Not supported

- **Windows on ARM64, and MinGW or clang-cl toolchains.** Only x64 with MSVC is built and tested.
- **AMD (ROCm) and Apple (Metal) GPUs.** Not implemented. Only NVIDIA CUDA is supported.
- **Multiple GPUs.** One CUDA device is used.

## Language and library versions

| | Version | Basis |
|---|---|---|
| Python | **3.10–3.14** | The package requires `>=3.10`. Every version is tested by the Windows wheel suite; the Linux gate (`scripts/verify.sh`) runs on 3.14. |
| Qiskit | **2.5.2** | Pinned exactly by the `qiskit` and `test` extras. Other versions are untested. |
| C++ | C++20 | GCC 12+, Clang 16+ or AppleClang 15+ is the stated minimum; GCC 15.2 is what was used. On Windows, MSVC 14.50 (Visual Studio Build Tools 2026) is the only compiler tested. |
| CMake | 3.25+ | 4.2.3 was used. |

See [Requirements](/get-started/requirements) for the complete list and [Compatibility](/reference/compatibility) for the reference table.
