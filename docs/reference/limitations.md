# Limitations

Every entry here is a real limitation of this release. None is speculative, and none is a planned feature dressed up as one.

## Platforms and installation

- **Only Linux on x86-64 and Windows x64 have been verified.** macOS arm64 and Linux aarch64 are intended but untested.
- **Windows is CPU-only and MSVC-only.** CUDA on Windows is refused at configure time (use WSL); ARM64, MinGW and clang-cl are not supported.
- **Python 3.10–3.14**, with **Qiskit pinned to 2.5.2**.
- **PyPI has Linux wheels only.** Windows wheels are built by the wheel workflow but not yet published. Building from source needs network access to GitHub (for test-only dependencies) and LAPACKE.
- **No CMake install or export rules.** The C++ libraries are not installable packages; the only installed artifact is the Python extension.

## Circuits

- **Measurements must be terminal.** No reset, classical control or mid-circuit measurement.
- **Unitaries are limited to five qubits.**
- **Estimator circuits must have no parameters.**
- **Exact noise is small.** The density matrix needs `16 · 4ⁿ` bytes, about 14 qubits on 16 GiB of RAM with the default budget. [Trajectory sampling](/engines/trajectory) reaches larger noisy circuits, but only when you opt in, and its results are statistical.
- **Only verified noise types** are accepted from an Aer noise model: depolarizing on 1 or 2 qubits, amplitude damping, their one-qubit composition, and single-qubit readout error. Others raise `UnsupportedFeatureError`.
- **Readout errors are single-qubit.** Correlated readout errors are refused.
- **Noise runs on the CPU only**, on the density-matrix or trajectory engine. No CUDA, and noise is not supported by the stabilizer, Pauli, MPS, tensor-network or cut engines; a factorized plan uses them only for noise-free parts.
- **Trajectories:** CPU only, and never a region of a factorized plan. The trajectory count for an expectation is set by a worst-case variance bound, so it is conservative: the planner may prefer the exact density matrix even when fewer trajectories would have met the precision. There is no adaptive stopping. The command-line tool has no flag for it.
- **Noise needs `qiskit-aer==0.17.2`** for the Qiskit path (`pip install "menos[qiskit]" qiskit-aer==0.17.2`). The native API needs nothing extra.

## Python surface

- No `SamplerV2` primitive and no provider class.
- **No way to choose an engine or backend.** Both are always the planner's choice.
- **No accuracy policy**: Python runs are always exact.
- **No structured plan or `replans()`**: only the `explain()` text.
- **Time and memory budgets are available only through the native API**, not through `MENOSBackend`.
- **`Observable` and `PauliString` are importable only from `menos._core`.**
- Native counts are hexadecimal-keyed.
- **An observable whose terms are all the identity raises `InternalError`.**

## PennyLane device

- **Needs CPython 3.11 or later and PennyLane 0.45.1 exactly.** No PennyLane release supports both 3.10 and 3.14.
- **Pauli-sum observables only.** `Hermitian`, `Projector`, `SparseHamiltonian` and `Hadamard` used as an observable are refused.
- **No mid-circuit measurements, `qml.cond` or resets**, as everywhere in MENOS.
- **No device derivatives**: no adjoint and no backprop. Gradients use parameter-shift or another PennyLane method that runs through the device.
- **One MENOS run per analytic `expval`**, and two per `var`. A QNode with many observables runs the circuit many times.
- **Analytic `state` and `probs` need a dense plan.** Analytic `state` and `probs` of a noisy circuit, and `density_matrix`, are refused.
- No Catalyst / `qml.qjit` and no program capture.

## Engines

- **Stabilizer:** Clifford only; no statevector.
- **Pauli propagation:** expectation only; Clifford gates plus `rz rx ry p u`.
- **MPS:** no statevector; its *time* model is badly optimistic (42× to 1,167× measured).
- **Tensor network:** expectation only; safe monitoring boundary is per observable term.
- **Cut and recombine:** expectation only, exact only, at most two cuts, CPU only.

## GPU

- **One device; dense regions only.** A `factorized` plan chooses CPU or CUDA per dense region (Phase 17); a direct plan has one backend; `cut_and_recombine` fragments always run on the CPU. Gates within a region are never split between CPU and GPU.
- **Verified on two GPUs**: compute capability 8.6 (4 GB) with CUDA 13.4, and compute capability 8.9 (8 GB, running code built for 8.6) with CUDA 12.4. CUDA 11.x is not supported, and CUDA 12.0 to 12.3 are not tested.
- **A CUDA build needs an explicit `CMAKE_CUDA_ARCHITECTURES`.** Without it CMake targets a default architecture and the kernels fail at run time.
- The CMake presets pin machine-specific values (`/usr/local/cuda`, compute capability 8.6).
- **CUDA gate kernels are double-precision-bound on consumer GPUs.** One- and two-qubit gates have specialised kernels, but bitwise identity with the CPU rules out fused multiply-add and kernels that skip zero products, so large circuits run at the GPU's double-precision arithmetic rate. Kernels specialised to diagonal or permutation gates are not implemented.
- CUDA on Linux aarch64 and on macOS is not supported.
- The largest state executed in verification on the 4 GB card was 26 qubits; 27 is the arithmetic ceiling and was not run.

## Time budgets and replanning

- **The budget bounds one attempt**, not the total job; a replanned job can take several times the budget.
- **A refusal can discard near-complete work.**
- **Only single-region plans yield an observation.** A factorized or cut plan that overruns is refused.
- **No checkpointing**: restart repeats the work.
- The dense sampler is not monitored per shot.
- Planning time is comparable to execution for millisecond-scale jobs.

## Reporting

- `explain()` labels of 11 characters or more run into their value (`candidate 0strategy=…`).
- **The measured peak (`peak_host_bytes`) counts the buffers that grow with the problem's state.** It leaves out the allocator's own overhead and small fixed-size allocations. It also leaves out two structures that grow with the circuit rather than the state: the tensor network's contraction-order analysis and the cut path's fragment circuits. The figure is therefore a lower bound on the process's own use, and never above the estimate.
- **The memory-accounting checks are verified on libstdc++ and the MSVC standard library.** libc++ is not verified.
- Calibration costs about 0.1 s on the first job in a process and varies about 1.5× between processes.

## Command line

- In OpenQASM 2, measurement must be written bit by bit (`measure q[0] -> c[0];`); OpenQASM 3 also accepts whole registers.
- `--statevector` is limited to 10 qubits; there is no budget option.
- The binary is a CMake target and is not installed by `pip`.
- OpenQASM 3 is imported for its static-circuit subset only: `reset`, control flow, subroutines, timing, calibrations, classical variables and applying a gate to a whole register are refused. `pow` takes integer exponents only, and a controlled gate with no gate-table form is limited to 5 qubits.
- A program must start with an `OPENQASM` header so the CLI can choose the importer.

## Verification scope

- **Determinism across platforms is a design invariant, verified only on Linux x86-64.**
- Overhead of run-time monitoring is bounded by about 2.4% on the cheapest engine and is not resolvable above noise in measurements.
