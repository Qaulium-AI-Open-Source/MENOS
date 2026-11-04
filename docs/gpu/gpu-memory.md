# GPU Memory

Device memory is a **separate budget** from host memory, and on most GPUs it is the smaller one.

## The budget

The device budget is the **free** memory the CUDA driver reports, not the marketing capacity. The development GPU is sold as 4096 MiB; the driver reports about **3770 MiB total**, and a little less free once the display and other processes have taken theirs. Budgeting against the nameplate would over-commit the card and then fail at allocation.

## What must fit

For a dense region of `n` qubits, the device holds one allocation of `16 · 2ⁿ` bytes, reused across every gate. Gate matrices are tiny and are uploaded per gate; the initial state is built on the device, so nothing of that size is uploaded.

| Qubits | Device state |
|---|---|
| 20 | 16 MiB |
| 24 | 256 MiB |
| 26 | 1 GiB |
| 27 | 2 GiB |
| 28 | 4 GiB |

On a card with about 3.6 GiB free, **27 qubits is the arithmetic ceiling**. The largest size actually executed in verification was **26 qubits**; 27 is what the arithmetic allows and was not run.

## Why the GPU limit sits below the CPU's

Host memory is typically larger, and by default half of it is available, so the CPU path can go further than a 4 GB card. Above the device ceiling the CUDA candidate is simply infeasible and the CPU wins by default.

## Refusal, then fallback

A region that does not fit the device budget is refused **before any `cudaMalloc`**. The planner then keeps the region on the CPU instead of failing the job, so the plan report simply shows `backend=cpu` for that region. A hardware profile with a smaller device budget is honoured the same way, which is how the project's C++ tests exercise this behaviour.

## Transfers

| Direction | What | When |
|---|---|---|
| Host to device | Gate matrices | Per gate (small) |
| Device to host | The full `16 · 2ⁿ` state | Only if a statevector or counts are requested |
| Device to host | One scalar per term | For an expectation value |

An expectation-only job transfers almost nothing back, which is where the GPU's advantage is largest.

## Other processes

A desktop environment, a browser, or another GPU job all hold device memory. MENOS reads the free figure again before planning each job (since Phase 21), so memory released or claimed between jobs is noticed. A replan within a job uses that job's figure. When a plan puts anything on the device, `explain()` prints the figure it used on the `estimated` line as `device_available_bytes=`, and each CUDA candidate's `device_bytes=`. Because the figure depends on other processes, such a report can differ between two runs of the same job.

## If a device allocation still fails

An allocation that fails for memory even though the budget check passed is a `ResourceExceeded` (since Phase 23) naming the bytes asked for, the bytes free and the bytes MENOS's pool retains: `the CUDA device could not provide N bytes of device memory: F bytes are free now and P are retained in MENOS's pool; another process may have taken device memory since the job was planned`. That is the signal that something else took device memory after the query. Under WSL the driver may serve an allocation past the free figure from shared system memory, so there the refusal can come later than the figure suggests.

## Related

[Memory Constraints](/planning/memory-constraints) · [CPU vs CUDA](/gpu/cpu-vs-cuda) · [Troubleshooting CUDA](/gpu/troubleshooting-cuda)
