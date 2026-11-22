# Installation

MENOS is published on PyPI as [`menos`](https://pypi.org/project/menos/). The published wheel is prebuilt, so installing it compiles nothing.

## Install from PyPI

With `pip`:

```bash
pip install menos
```

With `uv`, inside a project:

```bash
uv add menos
```

To use `MENOSBackend`, `MENOSEstimator` and `MENOSSampler`, install the `qiskit` extra, which pins Qiskit 2.5.2:

```bash
pip install "menos[qiskit]"
# or
uv add "menos[qiskit]"
```

Without it you can still use the [native API](/build/native-api).

To use the PennyLane device `menos.qubit`, install the `pennylane` extra, which pins PennyLane 0.45.1 and needs CPython 3.11 or later (see [PennyLane Device](/build/pennylane)):

```bash
pip install "menos[pennylane]"
```

::: warning What the PyPI wheels cover
Version 1.3.1 ships wheels for **CPython 3.10–3.14 on Linux x86-64** (manylinux 2.28, so glibc 2.28 or newer) and on **Windows x64** (`win_amd64`). Both have the **CUDA backend compiled in**: with an NVIDIA GPU and driver the planner can use the GPU, and without one they run on the CPU engines (see [Enabling CUDA](#enabling-cuda)). The Windows wheels have it from 1.3.1; the 1.3.0 Windows wheels are **CPU-only**. No source distribution is published, so on any other platform, Python version or architecture `pip` finds no matching distribution; [build from source](#install-from-source) there.
:::

You need **Python 3.10 or newer** (up to 3.14). If your system does not provide it, [`uv`](https://docs.astral.sh/uv/) can install it:

```bash
uv python install 3.14
```

Then [verify the installation](#verify).

## Install from source

Build from source if you are on a platform the wheels do not cover, need a CUDA build for one specific toolkit or GPU, or will modify MENOS. MENOS is a C++ library with a Python extension, so a source install compiles native code; expect it to take a few minutes the first time.

### 1. Install the system prerequisites

You need a C++20 compiler, CMake 3.25 or newer, `git`, and the LAPACK and LAPACKE development libraries (the matrix-product-state engine uses LAPACKE for its singular value decompositions).

On Ubuntu:

```bash
sudo apt install build-essential cmake git liblapacke-dev liblapack-dev libblas-dev
```

On other Linux distributions, install the equivalent packages. On macOS, LAPACK is found through the Accelerate framework; note that macOS is [not verified](/introduction/supported-platforms).

You also need **Python 3.10–3.14** (see above). On Windows, follow [On Windows](#on-windows) instead of this step.

### 2. Get the source

Clone the repository, or unpack a source archive, and change into its root directory (the one containing `pyproject.toml`).

### 3. Create an environment and install

With `uv`:

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python ".[qiskit]"
```

With plain `pip`:

```bash
python3.14 -m venv .venv
.venv/bin/pip install ".[qiskit]"
```

The `qiskit` extra installs Qiskit 2.5.2, which `MENOSBackend` and `MENOSEstimator` require. Without it you can still use the [native API](/build/native-api).

### On Windows

Native Windows builds use MSVC. The steps below build the CPU engines; the CUDA backend needs two more things, described in [CUDA Installation](/gpu/cuda-installation#on-windows). You need:

- Visual Studio 2026 with the **Desktop development with C++** workload (the free Build Tools are enough). Its CMake and Ninja are used.
- `git` and [`uv`](https://docs.astral.sh/uv/).
- OpenBLAS, which provides LAPACKE. A script downloads the pinned OpenBLAS 0.3.34 x64 release, checks its SHA-256 and unpacks it under `build\_deps`, where the build finds it:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch-openblas-windows.ps1
uv venv .venv --python 3.14
uv pip install --python .venv\Scripts\python.exe ".[qiskit]"
```

The OpenBLAS DLL and its licence are installed next to the extension module, so nothing needs to be added to `PATH`. To use an OpenBLAS directory of your own, set the environment variable `MENOS_OPENBLAS_ROOT`; it must be an LP64 build (the `OpenBLAS-<version>-x64.zip` release, not `x64-64`), and the build refuses an ILP64 one.

::: tip What the build needs from the network
Building requires network access. During configuration CMake downloads GoogleTest and nlohmann/json from GitHub at pinned versions. **Both are used only by the test suite**, but they are fetched even for a Python-only install, and the build fails without them. Python build tools (`scikit-build-core` 1.0.3 and `nanobind` 3.1.0) are fetched by the package installer.
:::

## Verify

Whichever way you installed, run this with the environment's Python (`.venv/bin/python` for the source steps above, or `uv run python` in a uv project):

```bash
python -c "import menos; print(menos.__version__)"
```

Then follow [Verify Installation](/get-started/verify-installation), which runs a real circuit.

## Building for development

If you will modify MENOS, use an editable install and the project's presets:

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest python/tests --deselect python/tests/test_wheel_smoke.py
```

The C++ library and its tests build separately with CMake presets:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

`scripts/verify.sh` runs the complete verification (formatting, static analysis, both presets, the Python suite, a wheel build and a clean-environment smoke test). It needs `clang-format` and `clang-tidy` in addition to the prerequisites above.

On Windows, the presets are `windows-dev` (Debug with AddressSanitizer) and `windows-release`, and must run inside a Visual Studio x64 developer environment. `scripts\verify-windows.ps1` enters one, fetches OpenBLAS, builds and tests both presets, runs the Python suite, builds a wheel and smoke-tests it; with `-Wheels` it also builds and tests wheels for CPython 3.10–3.14:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify-windows.ps1 -Wheels
```

## Enabling CUDA

The Linux wheels have the CUDA backend compiled in, and from version 1.3.1 so do the Windows wheels (the 1.3.0 Windows wheels are CPU-only). They are built with the CUDA 12.9 toolkit for compute capabilities 7.5 to 12.0 and carry the CUDA runtime, so they need only an NVIDIA GPU and a driver that supports CUDA 12; no toolkit has to be installed. On a machine with no usable GPU they behave like a CPU-only build.

A default source install has no GPU support. See [CUDA Installation](/gpu/cuda-installation) for how to confirm the GPU is used and how to build with CUDA from source.

## If installation fails

See [Installation Problems](/troubleshooting/installation) and [Build Errors](/troubleshooting/build-errors).
