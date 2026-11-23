# Build Errors

Each entry is a **symptom**, its **cause**, and a **solution**. The exact wording of messages from your compiler and from CMake depends on their versions; the entries describe the situation rather than quote every tool's text.

## Missing LAPACK or LAPACKE

| | |
|---|---|
| **Symptom** | CMake fails while configuring, reporting that it could not find LAPACK, or could not find `lapacke.h` |
| **Cause** | The matrix-product-state engine needs LAPACKE for its singular value decompositions, and the development packages are not installed |
| **Solution** | On Ubuntu: `sudo apt install liblapacke-dev liblapack-dev libblas-dev`. On macOS LAPACK comes from the Accelerate framework (macOS is [unverified](/introduction/supported-platforms)). On Windows, run `scripts\fetch-openblas-windows.ps1`, or set `MENOS_OPENBLAS_ROOT` to an LP64 OpenBLAS release directory |

## OpenBLAS is ILP64 (Windows)

| | |
|---|---|
| **Symptom** | CMake stops with `... is an ILP64 OpenBLAS; use the LP64 (OpenBLAS-<version>-x64.zip) release` |
| **Cause** | MENOS passes 32-bit LAPACK integers; the `x64-64` OpenBLAS release uses 64-bit ones |
| **Solution** | Use the `OpenBLAS-<version>-x64.zip` release, which `scripts\fetch-openblas-windows.ps1` downloads |

## CUDA requested with MSVC

| | |
|---|---|
| **Symptom** | CMake stops with `MENOS_ENABLE_CUDA with MSVC needs CMAKE_CUDA_HOST_COMPILER`, or nvcc stops with `unsupported Microsoft Visual Studio version` |
| **Cause** | The CUDA 12 compiler accepts Visual Studio 2017 to 2022 only, and MENOS's C++ is built with the Visual Studio 2026 toolset (14.50) |
| **Solution** | Install the MSVC 14.44 toolset beside 14.50 and set `CUDAHOSTCXX` to its `cl.exe`; only the device code is compiled with it (`docs/adr/0030` section 3). `scripts\verify-windows.ps1` does this |

## MSVC 14.51 refused

| | |
|---|---|
| **Symptom** | CMake stops with `MSVC 19.51... (toolset 14.51) miscompiles MENOS` |
| **Cause** | The 14.51 toolset's optimizer reads an uninitialised loop bound in MENOS's planner code, so runs crash, hang or compute a wrong result (measured, `docs/adr/0022` section 3.1). The 14.50 toolset compiles it correctly |
| **Solution** | Install the "MSVC v14.50" toolset in the Visual Studio Installer (component `Microsoft.VisualStudio.Component.VC.14.50.18.0.x86.x64`) and enter the developer environment with `-vcvars_ver=14.50`; `scripts\verify-windows.ps1` does this |

## `cl`, `ninja` or the Windows SDK not found

| | |
|---|---|
| **Symptom** | `cmake --preset windows-release` cannot find the C++ compiler, Ninja or system headers |
| **Cause** | The `windows-*` presets need a Visual Studio x64 developer environment |
| **Solution** | Run `scripts\verify-windows.ps1`, which enters one, or open a "x64 Native Tools" prompt first |

## Compiler too old

| | |
|---|---|
| **Symptom** | Compile errors about unsupported C++20 features |
| **Cause** | MENOS needs C++20: GCC 12+, Clang 16+, or AppleClang 15+; on Windows, MSVC from Visual Studio 2026 is what is tested |
| **Solution** | Install a newer compiler and tell CMake to use it (`CXX=g++-13` or `-DCMAKE_CXX_COMPILER=…`) |

## CMake too old

| | |
|---|---|
| **Symptom** | CMake stops immediately, saying a newer version is required |
| **Cause** | The project requires CMake 3.25 or newer |
| **Solution** | Install a newer CMake (for example from your distribution's backports, or `pip install cmake`) |

## Unsupported operating system

| | |
|---|---|
| **Symptom** | `error: "menos supports only Linux, macOS and Windows"` |
| **Cause** | The source refuses to build on other operating systems |
| **Solution** | Build on Linux, macOS or Windows (x64, MSVC) |

## `scikit-build-core` or `nanobind` not found

| | |
|---|---|
| **Symptom** | The Python build fails before compiling, about a missing build backend |
| **Cause** | The build tools are fetched by the installer into an isolated environment; this fails if you disabled build isolation without installing them |
| **Solution** | Install with default isolation, or first install the pinned tools: `scikit-build-core==1.0.3` and `nanobind==3.1.0` |

## Out of memory while compiling

| | |
|---|---|
| **Symptom** | The compiler is killed partway through the build |
| **Cause** | Parallel compilation of large translation units on a machine with little memory |
| **Solution** | Build with fewer parallel jobs, for example `cmake --build --preset release -j2` |

## `clang-format` or `clang-tidy` not found

| | |
|---|---|
| **Symptom** | `scripts/verify.sh` fails at its first stages |
| **Cause** | The full verification script runs formatting and static analysis, which need those tools |
| **Solution** | Install `clang-format` and `clang-tidy`. They are development-only and are not needed to use MENOS |

## CUDA build errors

See [Troubleshooting CUDA](/gpu/troubleshooting-cuda).

## See also

[Installation Problems](/troubleshooting/installation) · [Requirements](/get-started/requirements)
