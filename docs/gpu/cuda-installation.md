# CUDA Installation

The **Linux wheels on PyPI have the CUDA backend compiled in** (version 1.3.0 and later): they are built with the CUDA 12.9 toolkit for compute capabilities 7.5 to 12.0 and carry the CUDA runtime, so `pip install menos` on Linux x86-64 needs only an NVIDIA GPU and a driver that supports CUDA 12. From version 1.3.1 the **Windows wheels (`win_amd64`) have it too**, built with the same toolkit and carrying `cudart64_12.dll`; the 1.3.0 Windows wheels are CPU-only. With such a wheel, skip to [confirming the GPU is used](#confirm-it-is-used). A default source build has **no GPU support**. The rest of this page shows how to build the CUDA backend from source and confirm it is used.

## Do I need CUDA?

No. Every engine has a CPU implementation and the CPU path is complete. CUDA accelerates only the **dense** engine, and only when the circuit is large enough that the GPU's speed outweighs its transfer and launch costs. On the development GPU that crossover was about 12 qubits. If your circuits are smaller, or are planned onto stabilizer, MPS or another engine, a GPU will not be used. See [CPU vs CUDA](/gpu/cpu-vs-cuda).

## Check your GPU

```bash
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
```

You need an NVIDIA GPU, and you need to know its **compute capability** (for example `8.6`) for the build step. The driver version in the same output is what must support your CUDA toolkit; `nvidia-smi` also prints the newest CUDA version the driver advertises.

## Install the CUDA toolkit

Install NVIDIA's CUDA toolkit from NVIDIA's developer site (`developer.nvidia.com/cuda-downloads`). MENOS needs **CUDA 12.0 or newer**; versions **13.4** and **12.4** were verified. Choose a version your driver supports (see [CUDA Requirements](/gpu/cuda-requirements)). You also need a host C++ compiler that your `nvcc` accepts: the 13.4 toolkit accepted GCC 15.2, and the 12.4 toolkit needs GCC 13 or older.

::: warning Two toolkits on one machine
Linux distributions ship their own CUDA toolkit packages, for example Ubuntu's `nvidia-cuda-toolkit`, which can be an older release whose `nvcc` accepts only older GCC versions. If both are installed, a plain CMake run finds whichever `nvcc` comes first on `PATH`, which may not be the one you intend. **Always pass the compiler and toolkit root explicitly**, as below. The development machine had both 12.4 (Ubuntu) and 13.4 (NVIDIA), and the wrong one was picked by default.
:::

A newer toolkit than the driver's advertised CUDA version can still work through CUDA's minor-version compatibility: toolkit 13.4 was verified on a driver advertising 13.2.

## Build the Python package with CUDA

Pass three CMake settings through `CMAKE_ARGS`:

```bash
CMAKE_ARGS="-DMENOS_ENABLE_CUDA=ON \
            -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
            -DCUDAToolkit_ROOT=/usr/local/cuda \
            -DCMAKE_CUDA_ARCHITECTURES=86" \
  uv pip install --python .venv/bin/python ".[qiskit]"
```

Replace `/usr/local/cuda` with the toolkit you installed, and `86` with your GPU's compute capability **without the dot** (`8.6` becomes `86`). The build takes about 4 to 5 minutes.

::: danger `CMAKE_CUDA_ARCHITECTURES` is required
If you leave it out, CMake targets a default architecture that is not your GPU's. The build succeeds and the install succeeds, and then the first GPU job fails at run time with an error such as `cuda: cudaMemcpyToSymbol (matrix) failed: the provided PTX was compiled with an unsupported toolchain`. See [Troubleshooting CUDA](/gpu/troubleshooting-cuda).

The command above was verified with architecture `86` on a compute-capability-8.6 GPU. Other values have not been tested.
:::

## Build the C++ library with CUDA

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release --output-on-failure
```

For a CUDA 12.x toolkit whose `nvcc` rejects your default GCC, the `cuda12-dev` and `cuda12-release` presets build everything with GCC 13 against Ubuntu's `nvidia-cuda-toolkit` (`/usr/bin/nvcc`, with headers and runtime under `/usr`):

```bash
cmake --preset cuda12-release
cmake --build --preset cuda12-release
ctest --preset cuda12-release --output-on-failure
```

The `cuda-dev` and `cuda-release` presets pin `/usr/local/cuda` and compute capability 8.6, and the `cuda12-*` presets pin `/usr/bin/nvcc`, `/usr`, `g++-13` and compute capability 8.6. Those are values for the development machine; on another machine change them. CMake allows command-line definitions to override a preset's cache variables, but overriding the architecture that way has not been tested here.

## On Windows

The native Windows build (MSVC) has the CUDA backend from version 1.3.1 (`docs/adr/0030`). It needs two things beyond the [CPU build](/get-started/installation#on-windows):

- **The CUDA 12.9 compiler and runtime.** `scripts\fetch-cuda-windows.ps1` downloads NVIDIA's redistributable archives, checks their SHA-256 hashes and unpacks them under `build\_deps`; it prints the directory to use as `CUDA_PATH`. Nothing is installed system-wide and the GPU driver is not touched.
- **A second MSVC toolset for the device code.** The CUDA 12 compiler accepts Visual Studio 2017 to 2022 only, and MENOS's C++ is built with the Visual Studio 2026 toolset (14.50). Install the **MSVC v143 (14.44)** toolset beside it (Visual Studio Installer component `Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64`) and set `CUDAHOSTCXX` to its `cl.exe`. Only the `.cu` files are compiled with it.

`scripts\verify-windows.ps1` sets both and then builds and tests the two CUDA presets. By hand, from a Visual Studio x64 developer environment entered with `-vcvars_ver=14.50`:

```powershell
$env:CUDA_PATH = (& .\scripts\fetch-cuda-windows.ps1) -replace "\\", "/"
$env:CUDAHOSTCXX = "<Visual Studio>/VC/Tools/MSVC/14.44.<build>/bin/Hostx64/x64/cl.exe"
cmake --preset windows-cuda-release
cmake --build --preset windows-cuda-release
ctest --preset windows-cuda-release --output-on-failure
```

The `windows-cuda-*` presets build for compute capability 8.9, the development GPU's; on another GPU pass `-DCMAKE_CUDA_ARCHITECTURES=<yours>`. The build places `cudart64_12.dll` beside every executable, and the wheel carries it under a hashed name in `menos.libs`, so a machine that runs MENOS needs only the NVIDIA driver.

## How MENOS detects CUDA

Detection happens **at run time**, not at build time. A CUDA-enabled build on a machine with no usable GPU simply never offers the backend; it does not fail. The CUDA runtime is asked for a device once, and the answer is cached. The planner then considers CUDA only if calibration also succeeded. See [Backend Selection](/planning/backend-selection).

## Confirm it is used

The Python API never names a backend, so look at the plan:

```python
import math
import random

from menos import Circuit, Opcode, RunOptions, run

n = 20
rng = random.Random(3)
circuit = Circuit(n, 0)
for _ in range(3):
    for q in range(n):
        circuit.append(Opcode.RY, [q], [rng.uniform(0, math.pi)])
    for q in range(0, n - 1, 2):
        circuit.append(Opcode.CX, [q, q + 1])
    for q in range(1, n - 1, 2):
        circuit.append(Opcode.CX, [q, q + 1])

options = RunOptions()
options.want_statevector = True
report = run(circuit, options).explain()
print(next(line for line in report.splitlines() if line.startswith("region 0")))
```

On a CUDA-enabled build with a usable GPU this prints `… engine=dense backend=cuda`; on a CPU-only build it prints `backend=cpu`. Both are correct.

## Next

[CUDA Requirements](/gpu/cuda-requirements) · [GPU Memory](/gpu/gpu-memory) · [Troubleshooting CUDA](/gpu/troubleshooting-cuda)
