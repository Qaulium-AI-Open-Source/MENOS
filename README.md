# MENOS

**High-performance quantum simulation runtime.**

An open-source project by [Qaulium AI](https://www.qauliumai.com/). Product page: [qauliumai.com/menos](https://www.qauliumai.com/menos).

MENOS is a quantum circuit simulation **compiler and runtime**. It analyzes a circuit, estimates the cost of several execution strategies, chooses a plan, and runs it on the engine that fits the workload. It is also a Qiskit `BackendV2`, `EstimatorV2` and `SamplerV2`, so existing Qiskit code runs against it unchanged.

Full documentation, from installation to the API reference, is at **[qauliumai.com/menos/docs](https://www.qauliumai.com/menos/docs/)** (see [Documentation](#documentation)).

## What is MENOS?

Instead of pushing every circuit through one fixed algorithm, MENOS:

1. **Analyzes** the circuit: its structure, its entanglement, and what output you asked for;
2. **Estimates** what each candidate strategy would cost, in a cost model shared by all engines;
3. **Plans** by choosing the cheapest feasible strategy;
4. **Executes** the plan, watching it against a time budget and replanning if reality disagrees with the estimate.

## Why MENOS?

- **The choice of engine is a measured cost comparison**, not a rule of thumb. `explain()` shows every candidate, including why each loser lost.
- **Infeasible is a first-class answer.** A candidate that cannot run is reported with the reason.
- **Independent qubit groups are simulated independently** and recombined exactly.
- **Work the output cannot depend on is removed** (light-cone reduction) before planning.
- **Budgets are enforced**, not merely estimated: memory before any allocation, time while the run executes.
- **Exact results, reproducibly.** Every engine reachable from Python is exact, and the same circuit, options and seed give the same counts.

## Key capabilities

| | |
|---|---|
| Output | Counts, exact statevectors, exact Pauli expectation values, exact noisy simulation |
| Engines | Dense, stabilizer, Pauli propagation, matrix product state, tensor network, exact circuit cutting, density matrix |
| Planner | Cost model in work units, automatic factorization, CPU/CUDA selection by calibrated seconds |
| Integration | Qiskit `BackendV2` (`MENOSBackend`), `EstimatorV2` (`MENOSEstimator`) and `SamplerV2` (`MENOSSampler`), with Qiskit Aer noise models; the PennyLane device `menos.qubit`; OpenQASM 3 import (`menos.from_qasm3`, static subset) and OpenQASM 2/3 on the command line |
| Mitigation | Opt-in readout-error mitigation with calibrated, reported uncertainty; batch execution of several circuits (`run_batch`) |
| GPU | NVIDIA CUDA dense backend, bitwise identical to the CPU |
| Runtime | Time-budget monitoring with safe abandonment and replanning |

## Architecture

```text
Qiskit QuantumCircuit → KIR → light-cone reduction → analysis → cost model → planner
        → execution plan → runtime → engines (dense · CUDA · stabilizer · Pauli · MPS · tensor network · cut)
                                   ↘ monitoring → replanning
```

The C++ layers depend on each other in one direction (`api → runtime → engines → kernels`; `api → planner → cost → analysis → ir → common`), enforced by the build. See the documentation's Architecture page, and `docs/adr/` for the recorded design decisions.

## Installation

MENOS is on [PyPI](https://pypi.org/project/menos/) and needs **Python 3.10 or newer**:

```bash
pip install menos
# or, inside a uv project
uv add menos
```

Add the `qiskit` extra (`pip install "menos[qiskit]"` or `uv add "menos[qiskit]"`) to use `MENOSBackend`, `MENOSEstimator` and `MENOSSampler`; it pins Qiskit 2.5.2.

The 1.1.0 wheels on PyPI are prebuilt for CPython 3.10–3.14 on Linux x86-64 (manylinux 2.28) and are **CPU-only**. Windows wheels (CPython 3.10–3.14, `win_amd64`, CPU-only) are built by the wheel workflow but not yet published. No source distribution is published, so on other platforms build from source.

### From source

Build from source for the CUDA backend, an unsupported platform, or development. It needs a C++20 compiler, CMake 3.25+, `git` and LAPACKE. Building downloads test-only dependencies from GitHub at configure time, so it needs network access.

```bash
sudo apt install build-essential cmake git liblapacke-dev liblapack-dev libblas-dev   # Ubuntu
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python ".[qiskit]"
```

On Windows (x64, CPU only), install Visual Studio 2026's C++ tools, then:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch-openblas-windows.ps1   # pinned OpenBLAS, provides LAPACKE
uv venv .venv --python 3.14
uv pip install --python .venv\Scripts\python.exe ".[qiskit]"
```

Run these from the repository root. See the documentation's Installation and Requirements pages for details, including building with CUDA.

## First circuit

```python
from qiskit import QuantumCircuit

from menos.qiskit import MENOSBackend

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

result = MENOSBackend().run(qc, shots=1000, seed_simulator=7).result()
counts = result.get_counts()
print(counts)            # {'00': 507, '11': 493}
assert set(counts) <= {"00", "11"}
```

To see how MENOS planned a job, use the native API, whose results carry a plan report:

```python
from menos import Circuit, Opcode, RunOptions, run

circuit = Circuit(2, 2)
circuit.append(Opcode.H, [0])
circuit.append(Opcode.CX, [0, 1])
circuit.append_measure(0, 0)
circuit.append_measure(1, 1)

options = RunOptions()
options.shots = 100
options.seed = 1
print(run(circuit, options).explain())
```

## PennyLane

With the `pennylane` extra (`pip install "menos[pennylane]"`, CPython 3.11 or later, PennyLane 0.45.1), MENOS is the PennyLane device `menos.qubit`:

```python no-run
import pennylane as qml

dev = qml.device("menos.qubit", wires=2, seed=7)

@qml.qnode(dev)
def circuit(x):
    qml.RX(x, 0)
    qml.CNOT([0, 1])
    return qml.expval(qml.Z(0) @ qml.Z(1)), qml.state()
```

Analytic results are exact and agree with `default.qubit` within 1e-12. Parameter-shift gradients, shots, shot vectors and noise channels are supported. The runnable version is [`examples/pennylane_device.py`](examples/pennylane_device.py); the documentation's PennyLane Device page lists what is refused.

## Supported execution engines

| Engine | Outputs | Best for |
|---|---|---|
| Dense | counts, statevector, expectation | small to medium circuits; the only engine that produces a statevector |
| Stabilizer | counts, expectation | Clifford circuits, hundreds of qubits |
| Pauli propagation | expectation | local observables on circuits of Clifford gates and rotations |
| MPS | counts, expectation | low-entanglement circuits, large qubit counts |
| Tensor network | expectation | circuits whose contraction structure is cheap |
| Cut and recombine | expectation | circuits made of loosely connected clusters |
| Density matrix | counts, expectation | noisy circuits, exactly; about 14 qubits on 16 GiB |
| Trajectory | counts, expectation | larger noisy circuits, by sampling with a reported standard error; only on request |

Which one runs is the planner's decision, made per job from its cost model. See the documentation's Simulation Engines section.

### CPU threads

The dense, density-matrix and trajectory engines apply gates on every hardware thread by default, using MENOS's own thread pool. `RunOptions.max_threads` caps that number:
- `None` (the default) means all hardware threads.
- A value above the hardware count is treated as the hardware count.
- `0` is refused.

The thread count never changes a result: statevectors, counts and expectation values are bit-for-bit identical for every setting. Gates on small states always run on one thread, because waking the pool costs more than it saves. `OMP_NUM_THREADS` has no effect.

```python
from menos import Circuit, Opcode, RunOptions, run

circuit = Circuit(2, 0)
circuit.append(Opcode.H, [0])
circuit.append(Opcode.CX, [0, 1])

options = RunOptions()
options.shots = 0
options.seed = 1
options.want_statevector = True
options.max_threads = 2
print(run(circuit, options).statevector)
```

In a process that also runs NumPy or SciPy work between MENOS calls (Qiskit does), OpenBLAS's own idle workers can compete with MENOS's threads. Setting `OPENBLAS_NUM_THREADS=1` before starting Python measured 1.8× faster at 16–18 qubits on the Phase 15 gate machine (`docs/adr/0019` section 10.9). MENOS does not set it, because the setting is process-wide.

With `time_budget_seconds`, the planner's time estimate uses the calibrated single-threaded rate. That estimate is conservative for large threaded jobs, which can be refused when they would fit, and optimistic for a run capped at fewer threads, which the run-time monitor handles like any other overrun.

## CUDA

Neither the PyPI wheel nor a default source install has GPU support. The CUDA backend accelerates the dense engine when the calibrated model says the GPU is faster end to end, transfers included. It needs CUDA 12.0 or newer, and was verified on Linux x86-64 with CUDA 13.4 (compute capability 8.6) and CUDA 12.4 (compute capability 8.9, built for 8.6).

Building with CUDA requires setting the GPU architecture explicitly:

```bash
CMAKE_ARGS="-DMENOS_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
            -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_ARCHITECTURES=86" \
  uv pip install --python .venv/bin/python ".[qiskit]"
```

Use your own toolkit path and compute capability. See the documentation's GPU section.

## Documentation

The documentation is hosted online at **[qauliumai.com/menos/docs](https://www.qauliumai.com/menos/docs/)**.

It covers installation, a first-circuit tutorial, Qiskit integration, every engine, the planner and cost model, observables, GPU setup, runtime monitoring and replanning, the API reference, examples and troubleshooting.

## Examples

Runnable programs are in [`examples/`](examples/). Each asserts its own results, mostly against Qiskit's reference simulator, and is run by the test suite:

```bash
.venv/bin/python examples/bell_state.py
```

## Development

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

The presets are `dev` (with sanitizers), `release`, `cuda-dev`, `cuda-release`, `cuda12-dev` and `cuda12-release`. The `cuda-*` presets pin `/usr/local/cuda`; the `cuda12-*` presets pin Ubuntu's CUDA 12 toolkit (`/usr/bin/nvcc`) and GCC 13. All of them pin compute capability 8.6. These values are specific to the development machines. Benchmarks are opt-in (`-DMENOS_BUILD_BENCHMARKS=ON`).

### Benchmark dashboard

> **Disclaimer:** every benchmark figure in this repository and its documentation was measured on the developer's own machines (a laptop RTX 3050 / RTX 4060 class GPU, partly under WSL). They show relative behaviour on that hardware, not guaranteed performance; your numbers will differ.

[`tools/dashboard/menos_dashboard.py`](tools/dashboard/menos_dashboard.py) produces a self-contained HTML page. It compares MENOS with Qiskit's reference simulator (`qiskit.quantum_info.Statevector`) and checks that their statevectors agree. It also shows CPU against CUDA, using the crossover benchmark's JSON output:

```bash
cmake --preset cuda12-release -DMENOS_BUILD_BENCHMARKS=ON
cmake --build --preset cuda12-release --target menos_cuda_crossover_benchmark
MENOS_CROSSOVER_JSON=build/dashboard/crossover.json build/cuda12-release/benchmarks/menos_cuda_crossover_benchmark
.venv/bin/python tools/dashboard/menos_dashboard.py run --crossover-json build/dashboard/crossover.json
.venv/bin/python tools/dashboard/menos_dashboard.py render   # writes build/dashboard/dashboard.html
```

The MENOS-against-Qiskit series runs on whichever backend the planner picks. The dashboard does not yet use `RunOptions.backend` (Phase 17), so only the crossover benchmark forces the CPU and CUDA engines onto the same circuit. Without `--crossover-json`, the page marks the CPU-against-CUDA section as not measured.

`run` measures the crossover benchmark's circuit by default. `run --family clifford` (H, S and CX layers) or `--family product` (H and RZ layers, no entangling gate, which the planner splits into independent regions) measures another shape against Qiskit's statevector; `--crossover-json` only goes with the default family. Both ask for a statevector, so they use the dense engine (the stabilizer and MPS engines cannot produce one).

Each point is timed 10 times after a warm-up (`--repeats` changes this). The page shows each series' median with a min–max band, and the table also gives the interquartile range. The speed-up range is the one implied by the two series' ranges; it is not a confidence interval. The results also record the core counts and the power state (AC or battery, battery level, Windows power scheme, read through WSL interop where needed). A value that cannot be read is recorded as unknown. Compare two runs only when both were on AC with the same power scheme.

`menos_dense_gate_application_benchmark` times one generic 1-qubit gate, one generic 2-qubit gate, one Pauli-term expectation pass and one sampling pass at 16–26 qubits, 10 repetitions each. The console shows only the mean, median, standard deviation and coefficient of variation; `--benchmark_out=<file> --benchmark_out_format=json` also keeps every repetition, from which min and max follow.

## Testing

`scripts/verify.sh` is the single definition of "passes". It runs the format check, clang-tidy, the dev and release builds and tests, the benchmark build, an editable Python install, the Python test suite (which includes every example and every Python snippet in the documentation), a wheel build, a clean-environment wheel smoke test, the documentation build when Node is available, and the CUDA stages when a toolkit is present. It needs `clang-format` and `clang-tidy`.

```bash
scripts/verify.sh
```

## Platforms

**Linux on x86-64** and **Windows x64 (MSVC, CPU only)** have been verified; `scripts/verify.sh` and `scripts/verify-windows.ps1` are their verification scripts. macOS arm64 and Linux aarch64 are intended but untested. The CUDA backend is Linux-only (on Windows, use WSL). See the documentation's Supported Platforms and Limitations pages.

## Contributing

Changes must pass `scripts/verify.sh`, come with regression tests, and keep public headers documented. Design decisions are recorded as architecture decision records in [`docs/adr/`](docs/adr/); project conventions are in [`CLAUDE.md`](CLAUDE.md) and the phased roadmap in [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md).

Participation in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
