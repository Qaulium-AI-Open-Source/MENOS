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
- **The circuit builder has no text drawing**, its parameters are affine only, and binding is through `bind` only ([Circuit Builder](/build/circuit-builder)).
- **OpenQASM 3 export refuses explicit unitaries, noise channels and readout errors**, and there is no OpenQASM 2 export.

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
- **MPS:** no statevector; its *time* model is optimistic (actual 0.8× to 58× the prediction, median 21×, measured in Phase 24; 42× to 1,167× in Phase 11 before the bond replay). The stabilizer, Pauli and tensor-network time models are also off by a factor of 5 to 30 on some workloads ([Cost Model](/planning/cost-model#how-accurate-is-it)).
- **Tensor network:** expectation only; safe monitoring boundary is per observable term.
- **Cut and recombine:** expectation only, exact only, at most two cuts, CPU only.
- **Dense, disk tier:** counts and expectation only (a statevector would be the whole file); noiseless circuits only; a job's sole region only (never inside a factorized or cut plan); CPU only.

## Memory tiers

- **Reach only.** The disk tier is chosen only when the in-memory dense state does not fit the memory budget, never because it would be faster; and a forced `tier=disk` never falls back. The host-offloaded CUDA tier was measured and not built (Phase 23, `docs/adr/0027` section 10.1): `Tier.HostOffloaded` is refused with `UnsupportedFeatureError`.
- **The scratch file** is `menos-scratch-<pid>-<serial>.state` in `RunOptions.scratch_directory` (default: the system's temporary directory), `16 · 2ⁿ` bytes, written with unbuffered I/O when the file system allows it. It is unlinked at creation on Linux and opened delete-on-close on Windows, so it is gone when the run ends, when an exception is thrown, and when the process is killed; only a power loss or a crash of the operating system itself can leave one behind, to be deleted by hand.
- **The planner leaves 5% of the volume or 8 GiB free**, whichever is larger, and there is no setting for the reserve.
- **Under WSL, check the directory.** Some WSL distributions mount `/tmp` as a RAM-backed `tmpfs`: the default scratch directory then lives in memory, which defeats the tier's purpose, and the planner's bandwidth measurement reports memory speed. Name a directory on the ext4 volume with `scratch_directory`. Writing a scratch file there grows the WSL virtual disk, which does not shrink by itself (`docs/adr/0027` section 10.1).
- **The bandwidth measurement writes and reads up to 1 GiB** in the scratch directory, once per directory per process, before the first job that could reach the tier. Under a time budget an unmeasured directory is a refusal.
- **The time prediction for the tier is pessimistic on Linux with many threads** (0.35× measured over predicted in one WSL run; 0.51–1.17× in four Windows runs), because the dense work rate is calibrated on a single-threaded 13-qubit circuit (`docs/adr/0027` section 10.3). A threaded calibration is a Phase 24 item.
- **No overlap of transfers with compute**: each pass reads a window, applies its gates and writes it back in turn (deferred, `docs/adr/0027` section 9, Q6).
- **The gate-scale runs were on one machine**: an NVMe drive C: under Windows and the ext4 volume under WSL, at 29–31 qubits.

## GPU

- **One device; dense regions only.** A `factorized` plan chooses CPU or CUDA per dense region (Phase 17); a direct plan has one backend; `cut_and_recombine` fragments always run on the CPU. Gates within a region are never split between CPU and GPU.
- **Verified on two GPUs**: compute capability 8.6 (4 GB) with CUDA 13.4, and compute capability 8.9 (8 GB, running code built for 8.6) with CUDA 12.4. CUDA 11.x is not supported, and CUDA 12.0 to 12.3 are not tested.
- **A CUDA build needs an explicit `CMAKE_CUDA_ARCHITECTURES`.** Without it CMake targets a default architecture and the kernels fail at run time.
- The CMake presets pin machine-specific values (`/usr/local/cuda`, compute capability 8.6).
- **CUDA gate kernels are double-precision-bound on consumer GPUs.** One- and two-qubit gates have specialised kernels, but bitwise identity with the CPU rules out fused multiply-add and kernels that skip zero products, so large circuits run at the GPU's double-precision arithmetic rate. Kernels specialised to diagonal or permutation gates are not implemented.
- **MENOS keeps GPU memory between jobs.** A finished job's device state stays in MENOS's memory pool for the next job of that size (Phase 22). The planner counts it as available and `explain()` shows it as `device_pool_bytes`; other processes see it as used until `release_device_memory()` (C++ and Python) gives it back or the process exits. A device without memory-pool support falls back to a `cudaMalloc` per job (`pooled=false` in `explain()`).
- **The pool has no cap below the device's total memory**, and there is no setting for one; `release_device_memory()` is the only control. If another process takes device memory between `plan_job`'s read of free memory and the allocation, the allocation fails inside CUDA and surfaces as `ResourceExceeded` naming the bytes asked for, the bytes free and the bytes the pool retains (since Phase 23). Under WSL the driver backs device memory with shared system memory and may grant an allocation past the free figure, so there the refusal can come one allocation later than the figure suggests (`docs/adr/0027` section 10.3).
- **Device-to-host copies are pageable.** Page-locked and staged transfers were measured under WSL and were no faster than the pageable copy, so they are not used (`docs/adr/0026` section 10.1). On that machine the copy runs at 3–6 GB/s and is 3–7% of a statevector job.
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
- **`peak_disk_bytes` is the scratch file's size**, charged when the file is created, and `0` for a plan with no disk-tier region.
- **`peak_host_bytes` and `peak_device_bytes` measure execution buffers, not process memory.** They count the buffers that grow with the problem's state during one execution attempt. They leave out the allocator's own overhead, small fixed-size allocations, and everything allocated while planning, including two structures that grow with the circuit: the tensor network's contraction-order analysis and the cut path's fragment circuits. The figure is a lower bound on the process's own use and never above the estimate. `memory_budget_bytes` likewise governs plan selection only; it is not a process cap. See [Memory Constraints](/planning/memory-constraints#what-the-budget-governs-and-what-the-peak-measures).
- **The ledger is per execution thread.** It relies on the kernels' worker threads never allocating, which holds for every engine today; an allocation made on a worker thread would not be counted (`docs/adr/0026` section 11.4).
- **`run_batch` budgets each circuit alone** and holds every result until it returns; a batch whose results together exceed the budget is not refused (`docs/adr/0026` section 11.6).
- **The memory-accounting checks are verified on libstdc++ and the MSVC standard library.** libc++ is not verified.
- Calibration costs about 0.1 s on the first job in a process and varies about 1.5× between processes.

## Command line

- In OpenQASM 2, measurement must be written bit by bit (`measure q[0] -> c[0];`); OpenQASM 3 also accepts whole registers.
- `--statevector` is limited to 10 qubits; there is no budget option.
- `--tier disk` writes the scratch file to `--scratch-dir`, or to the system's temporary directory; the CLI has no option for the reserve.
- The binary is a CMake target and is not installed by `pip`.
- OpenQASM 3 is imported for its static-circuit subset only: `reset`, control flow, subroutines, timing, calibrations, classical variables and applying a gate to a whole register are refused. `pow` takes integer exponents only, and a controlled gate with no gate-table form is limited to 5 qubits.
- A program must start with an `OPENQASM` header so the CLI can choose the importer.

## Verification scope

- **Determinism across platforms is a design invariant, verified only on Linux x86-64.**
- Overhead of run-time monitoring is bounded by about 2.4% on the cheapest engine and is not resolvable above noise in measurements.
