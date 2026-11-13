# Configuration

MENOS has no configuration file and reads **no environment variables**. Behaviour is controlled by per-call options and by build-time CMake settings.

## Per-call options

### Qiskit

| Option | Where | Default |
|---|---|---|
| `shots` | `MENOSBackend(...)`, `run(...)`, `set_options(...)` | 1024 |
| `seed_simulator` | same | `None` (fresh random) |
| `want_statevector` | same | `False` |
| `default_precision` | `MENOSEstimator(...)` | `0.0` (recorded only) |

See [Qiskit Backend](/reference/qiskit-backend) and [Estimator](/reference/estimator-api).

### Native

| Field of `RunOptions` | Default |
|---|---|
| `shots` | `0` |
| `want_statevector` | `False` |
| `seed` | `0` |
| `memory_budget_bytes` | `None` (50% of physical memory) |
| `expectation` | `None` |
| `time_budget_seconds` | `None` (unmonitored) |
| `tier` | `None` (the disk tier only when the dense state does not fit the budget) |
| `scratch_directory` | `None` (the system's temporary directory) |

See [Python API](/reference/python-api#runoptions).

## Build-time settings

| CMake setting | Default | Purpose |
|---|---|---|
| `MENOS_BUILD_PYTHON_BINDINGS` | `OFF` (the Python package turns it `ON`) | Build the `menos._core` extension |
| `MENOS_ENABLE_CUDA` | `OFF` | Build the CUDA dense backend |
| `MENOS_BUILD_BENCHMARKS` | `OFF` | Build the Google Benchmark programs |
| `CMAKE_CUDA_COMPILER` | unset | Path to `nvcc`. **Set it explicitly** when more than one toolkit is installed |
| `CMAKE_CUDA_ARCHITECTURES` | compiler default | **Set it explicitly** to your GPU's compute capability, for example `86` |
| `CUDAToolkit_ROOT` | unset | Root of the toolkit to link against |

CMake presets: `dev`, `release`, `cuda-dev`, `cuda-release`, `cuda12-dev`, `cuda12-release`. The `cuda-*` presets pin `/usr/local/cuda`, the `cuda12-*` presets pin Ubuntu's CUDA 12 toolkit (`/usr/bin/nvcc`) with GCC 13, and all of them pin compute capability 8.6. These values are which are **specific to the development machine**; override them on the command line for another machine. See [CUDA Installation](/gpu/cuda-installation).

### Passing settings to a Python install

```bash
CMAKE_ARGS="-DMENOS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86" \
  uv pip install --python .venv/bin/python ".[qiskit]"
```

## Budgets

The defaults and semantics of the memory and time budgets are in [Memory Constraints](/planning/memory-constraints) and [Time Budgets](/runtime/time-budgets).

## Calibration

Calibration runs automatically on first use and cannot be configured. See [Cost Model](/planning/cost-model#calibration).
