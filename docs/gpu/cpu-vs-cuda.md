# CPU vs CUDA

MENOS uses the GPU only when its own calibrated model says the GPU is **faster end to end**. You do not need to choose. You can force a backend (`RunOptions.backend`, `MENOSBackend(backend=...)`, `--backend`), and a forced backend never falls back; see [Forcing a backend](/planning/backend-selection#forcing-a-backend).

## The decision in one line

```text
algorithm:  cheapest feasible candidate by work units          (by predicted seconds under a time budget)
backend:    for dense only, CUDA iff its estimated seconds < the CPU's    (ties go to the CPU)
```

The CUDA estimate is end to end: kernel launches, host-to-device transfers, the work, and device-to-host transfers. See [Backend Selection](/planning/backend-selection).

## What was measured

::: warning Disclaimer
These benchmarks were run on the developer's own machine, a laptop. They illustrate the planner's behaviour, not performance you should expect; results on your hardware will differ.
:::

On the development machine (RTX 3050 Laptop GPU, 4 GB), through the Python API, on the same 3-layer entangled circuits. The CPU-only and CUDA-enabled builds were compared:

| Qubits | Backend the planner chose (CUDA build) | CPU time | CUDA time |
|---|---|---|---|
| 8 | CPU | 0.0001 s | 0.0001 s |
| 12 | CUDA | 0.0021 s | 0.0017 s |
| 20 | CUDA | 1.18 s | 0.092 s |

At 8 qubits the planner kept the job on the CPU, because each kernel launch costs a fixed amount however little it does. By 12 qubits the GPU was already ahead, and at 20 qubits it was about **13 times faster** end to end. The project's crossover benchmark measured a crossover at about **12 qubits** and roughly **14 to 16 times** at 24 qubits. These numbers belong to this machine; yours will differ, and the planner calibrates on yours.

## Results are identical

The statevector from the CUDA build hashed **identically** to the CPU build's at all three sizes (same SHA-256 of the raw amplitudes). Device kernels are built to match the CPU bit for bit, and counts are sampled on the host from that state, so a given seed gives the same counts on either backend.

## The plan shows it

A real report from the CUDA build, for the 20-qubit statevector:

```text
region 0   qubits=[0,1,…,19] engine=dense backend=cuda
estimated  state_bytes=16777216 work_units=367001600
actual     state_bytes=16777216 execute_s=0.090967 sample_s=0.000000
```

With a time budget, each feasible candidate also shows its estimated seconds; here the model predicted 0.077 s and the run took 0.091 s:

```text
candidate 0strategy=direct_dense feasible=true winner=true peak_bytes=16777216 work_units=367001600 est_seconds=0.0767662
```

## When the CPU is better

- **Small circuits.** Below the crossover, launch overhead dominates.
- **Counts of a large state.** The state must be copied back to the host to be sampled, so the transfer is paid.
- **Anything not dense.** Stabilizer, Pauli, MPS, tensor-network and cut plans never run on the GPU.
- **Factorized and cut plans.** Their regions stay on the CPU.
- **A state that does not fit in device memory.** See [GPU Memory](/gpu/gpu-memory).

The GPU's advantage is largest for **expectation values**, which return one number per term and transfer almost nothing back.

## What is not claimed

MENOS does not claim that a faster kernel makes a faster job. The comparison is end-to-end wall time including transfers, and the GPU is used only where that wins.

## Related

[CUDA](/engines/cuda) · [Backend Selection](/planning/backend-selection) · [CUDA example](/examples/cuda)
