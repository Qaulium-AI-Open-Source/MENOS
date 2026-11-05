# Requirements

Each requirement is labelled by how much you need it.

| Label | Meaning |
|---|---|
| **Required** | MENOS will not install or run without it |
| **Recommended** | Works without it, but you will want it |
| **Optional** | Only for specific features |
| **GPU-only** | Only for the CUDA backend |
| **Development-only** | Only for working on MENOS itself |

## Operating system and hardware

| | Requirement | Label |
|---|---|---|
| OS | Linux or Windows | **Required** (macOS and aarch64 are [unverified](/introduction/supported-platforms); Windows is CPU-only) |
| CPU | x86-64 (x64) | **Required** for what has been verified |
| Memory | See [Memory](#memory) below | **Required** |

## Software

The **Required to build** rows apply only to a source install. The [PyPI wheels](/get-started/installation#install-from-pypi) (CPython 3.10–3.14, Linux x86-64) and the Windows wheels need none of them.

| | Version | Label | Notes |
|---|---|---|---|
| Python | **3.10–3.14** | **Required** | `requires-python >=3.10` |
| NumPy | any compatible with your Python | **Required** | Installed automatically |
| Qiskit | **2.5.2** | **Required** for `MENOSBackend` and `MENOSEstimator` | Pinned exactly by the `qiskit` extra; other versions are untested |
| C++ compiler | GCC 12+, Clang 16+ or AppleClang 15+ (C++20); on Windows, the MSVC 14.50 toolset from Visual Studio 2026 (Build Tools are enough; 14.51 is refused because it miscompiles MENOS) | **Required** to build | GCC 15.2 and MSVC 14.50 were used |
| CMake | 3.25 or newer | **Required** to build | 4.2.3 was used |
| `git` | any | **Required** to build | CMake fetches test dependencies from GitHub |
| LAPACK, LAPACKE, BLAS (development packages) | any | **Required** to build | On Ubuntu: `liblapacke-dev liblapack-dev libblas-dev`. On Windows, `scripts/fetch-openblas-windows.ps1` downloads the pinned OpenBLAS 0.3.34 release |
| Network access to github.com | | **Required** to build | For GoogleTest and nlohmann/json, pinned |
| `scikit-build-core` 1.0.3, `nanobind` 3.1.0 | exact | **Required** to build | Fetched by the package installer |
| `uv` | any | **Recommended** | `uv add menos`, and used by the project's scripts; `pip` also works |

## GPU (CUDA)

| | Requirement | Label |
|---|---|---|
| NVIDIA GPU | **Compute capability 8.6** is what the build targets by default | **GPU-only** |
| NVIDIA driver | A driver supporting your CUDA toolkit; 595.91.07 was used with toolkit 13.4, 566.14 with toolkit 12.4 | **GPU-only** |
| CUDA toolkit | **12.0 or newer**; 13.4 and 12.4 were verified | **GPU-only** |
| Host compiler | The same C++ compiler as above | **GPU-only** |

GPUs of other compute capabilities can be targeted by overriding the architecture at configure time; only compute capability 8.6 has been tested. Details, including a two-toolkit pitfall, are in [CUDA Requirements](/gpu/cuda-requirements).

## Development

| | Label |
|---|---|
| `clang-format`, `clang-tidy` | **Development-only**: required by `scripts/verify.sh` (Linux) |
| Visual Studio 2026 C++ tools, CMake and Ninja (both ship with Visual Studio) | **Development-only**: used by `scripts/verify-windows.ps1` |
| `pytest` 9.1.1 | **Development-only**: installed by the `test` extra |
| Node.js and npm | **Development-only**: to build this documentation |

## Memory

Memory is the requirement that most often decides what you can run.

**Host memory.** A dense statevector of `n` qubits needs `16 · 2ⁿ` bytes (one complex128 amplitude is 16 bytes).

| Qubits | Statevector size |
|---|---|
| 20 | 16 MiB |
| 25 | 512 MiB |
| 28 | 4 GiB |
| 30 | 16 GiB |
| 32 | 64 GiB |

By default MENOS allows a job to use **50% of the machine's physical memory**. You can change that per job with the memory budget ([Memory Constraints](/planning/memory-constraints)). A job whose plan would need more is refused before any allocation, with an error that states the required and available bytes.

**Scratch disk.** For counts or an expectation value, a dense state larger than the budget can run from a scratch file of `16 · 2ⁿ` bytes in the system's temporary directory (or `RunOptions.scratch_directory`), leaving 5% of the volume or 8 GiB free; every pass over the circuit reads and writes the whole file, so the disk's bandwidth sets the time ([the disk tier](/planning/memory-constraints#the-disk-tier)).

Other engines use far less: a stabilizer simulation of 200 qubits needs about 13 KB, and a matrix-product-state simulation of 100 weakly entangled qubits needed about 50 KB in the [example](/examples/mps).

**GPU memory.** The device budget is the *free* memory the driver reports, which is smaller than the marketing figure: the development GPU is sold as 4096 MiB and reports 3770 MiB total and about 3681 MiB free. A statevector must fit in that, so by arithmetic the ceiling on such a card is 27 qubits (2 GiB); the largest size actually executed in verification was 26. See [GPU Memory](/gpu/gpu-memory).

## Checking what you have

```bash
python3.14 --version
cmake --version
g++ --version
nvidia-smi          # GPU-only
```
