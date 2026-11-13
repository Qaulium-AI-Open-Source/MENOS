# Memory Architecture

How MENOS accounts for memory, from the estimate the planner checks to the peak a result reports. Three places hold a job's data: **host memory** (RAM), **device memory** (the GPU's VRAM) and **disk** (a scratch file). One rule ties them together: **every estimate is an upper bound on what the run really holds**, and every result reports what it did hold.

This page shows how the pieces fit. [Memory Constraints](/planning/memory-constraints) covers the budget and the disk tier in detail, and [GPU Memory](/gpu/gpu-memory) covers the device.

## The picture

```text
                 plan_job                                   one execution attempt
  ┌─────────────────────────────────────┐          ┌────────────────────────────────────┐
  │ each candidate's estimate           │          │ MemoryLedger                       │
  │   estimated_peak_bytes  (host)      │  plan →  │   host   ── peak_host_bytes        │
  │   device_bytes          (CUDA)      │          │   device ── peak_device_bytes      │
  │   disk_bytes            (disk tier) │          │   disk   ── peak_disk_bytes        │
  │ checked against:                    │          └────────────────────────────────────┘
  │   memory_budget_bytes               │                     ↓
  │   free device memory + MENOS's pool │          Result + explain()  "actual" line
  │   scratch space − reserve           │
  └─────────────────────────────────────┘

  where the dense state lives (the tier):
    in memory (RAM, or VRAM on CUDA)  ── the default whenever the state fits
    disk      (scratch file)          ── only when it does not fit, counts or expectation only
```

## Estimates are upper bounds

Each engine's estimate counts **everything it holds at once**: the state, scratch and working buffers, the SVD workspace, samplers and the counts histogram. Each container is charged with the allocator's own overhead. There is no slack factor and no guess: the estimate is the worst case the code allows, for example every shot drawing a different outcome (since Phase 21, `docs/adr/0025`).

The project checks this by measurement on a fixture set covering every engine and output kind, on Linux and Windows:
- the heap a run really allocated never exceeds the estimate (asserted on every platform);
- the process's resident memory grows by no more than the estimate plus a fixed baseline (asserted on Linux, reported on Windows).

A GPU estimate is checked the same way, against the device ledger's peak, which it equals.

## The memory ledger

While a job runs, a **memory ledger** adds up every buffer that grows with the problem as it is allocated and freed: states, scratch, SVD workspace, samplers and histograms, device buffers and the runtime's combination buffers. It keeps one running total and one peak per space:

| Space | Result field | `explain()` |
|---|---|---|
| Host | `Result.peak_host_bytes` | `peak_host_bytes=` on the `actual` line |
| Device | `Result.peak_device_bytes` | `peak_device_bytes=` |
| Disk | `Result.peak_disk_bytes` | `peak_disk_bytes=`, printed when it is not 0 |

The ledger covers **one execution attempt**. A job that replans reports the attempt that finished.

What the figures are **not**:
- not the process's resident memory: allocator overhead and small fixed allocations are left out;
- not planning memory: the tensor network's contraction-order search and the cut path's fragment circuits are allocated while planning, outside the ledger;
- not a cap: `memory_budget_bytes` decides which plan is chosen and whether a job is refused. MENOS never stops a running job for memory.

So the estimate is an upper bound on the peak, and the peak is a lower bound on the process's own use. See [What the budget governs](/planning/memory-constraints#what-the-budget-governs-and-what-the-peak-measures).

## Copies that are not made

Phase 21 removed the copies that used to double a job's footprint, with every result bit for bit the same:
- the statevector is **moved** into the result, not copied;
- a CUDA state is copied straight from the device into the result;
- counts samplers keep outcomes as 8-byte indices instead of about 48 bytes per basis state;
- the MPS sampler works on the state itself;
- the tensor network frees each tensor once it is absorbed;
- the SVD owns its workspace.

## Device memory and the pool

The device has its own budget: the free memory the driver reports, **read again at every `plan_job`**, so memory another process takes or releases between jobs is noticed.

Every GPU state is allocated from **one MENOS memory pool** (`cudaMallocAsync`, since Phase 22, `docs/adr/0026`). When a job finishes, its state goes back to the pool instead of to the driver, and the next job, region, replan or batch circuit of that size reuses it. On the development GPU this took the allocation cycle from milliseconds (1–3 ms at 20 qubits, 9–23 ms at 24) down to 1–6 µs. A pooled block is not zero on arrival, so the state is filled once, by the engine's own initialisation.

- **Counted as available.** The planner adds the pool's reusable bytes to the free figure, so no job is refused for memory that is only waiting in the pool. `explain()` prints both as `device_available_bytes=` and `device_pool_bytes=`.
- **Released on request.** `release_device_memory()` (Python `menos.release_device_memory()`) hands the pool's memory back to the driver, for when another process needs the GPU.
- **Fallback.** A device without memory-pool support uses plain `cudaMalloc`, and `explain()` shows `pooled=false`.
- **A race is a refusal.** If another process takes device memory after planning and an allocation then fails, the job raises `ResourceExceeded` naming the bytes asked for, the bytes free and the bytes the pool holds (since Phase 23).

## Tiers: where the dense state lives

An execution engine is three choices: the **engine** (the algorithm), the **backend** (CPU or CUDA) and the **tier** (where the state lives).

| Tier | Where the state is | When |
|---|---|---|
| In memory | RAM (CPU) or VRAM (CUDA) | Always, whenever the state fits |
| Disk | A scratch file of `16 · 2ⁿ` bytes | Only when the in-memory dense state does not fit the budget |
| Host-offloaded | RAM, computed on the GPU window by window | **Measured, not built.** `Tier.HostOffloaded` is refused |

The disk tier (since Phase 23, `docs/adr/0027`) applies gates in **passes**. Each pass reads a window of the file that fits the budget, applies every gate it can to the window's qubits chunk by chunk with the unchanged in-memory kernels, and writes the window back. A gate on a qubit outside the window pairs chunks so the kernel sees the amplitudes it needs. The pass plan uses as few passes as possible. Because the kernels are the in-memory engine's own, counts and expectation values are **bit for bit** the in-memory result, gate fusion included.

The planner only **reaches** for the disk tier: it is priced only when the in-memory state does not fit, the job asks for counts or an expectation value, the circuit is noiseless and the dense engine is the job's only region. It is **never a speed choice**, and a job that fits in memory plans and reports exactly as before. `RunOptions.tier` forces it (`Tier.Disk`, which never falls back) or forbids it (`Tier.InMemory`). The scratch file goes in `RunOptions.scratch_directory` (default: the system's temporary directory), leaves 5% of the volume or 8 GiB free, whichever is larger, and is removed when the run ends.

Under a time budget, the tier is priced at the dense rate measured on every thread, plus the bytes written and read at the directory's own measured write and read bandwidths (since Phase 24). On the development machine a 31-qubit job, a 32 GiB file against a 15.85 GiB budget, ran 100 gates and 1 000 shots in 7–13 minutes.

## Reading it in `explain()`

```text
candidate 0strategy=direct_dense feasible=true winner=true peak_bytes=8590026448 work_units=… tier=disk passes=4 chunk_qubits=25 disk_bytes=34359738368
region 0   qubits=[…] engine=dense backend=cpu tier=disk
estimated  state_bytes=8590026448 work_units=… disk_available_bytes=… scratch_directory="…" disk_write_bandwidth=… disk_read_bandwidth=… (measured)
actual     … peak_host_bytes=8590026448 peak_device_bytes=0 peak_disk_bytes=34359738368
```

- `peak_bytes` on a candidate is its host estimate, and `device_bytes=` is shown on a CUDA candidate.
- `estimated` shows what the plan was checked against: `device_available_bytes=`, `device_pool_bytes=` and `pooled=` when the plan uses the GPU, and the scratch figures when it uses the disk tier.
- `actual` shows the ledger's peaks.

A plan that uses only the CPU and RAM prints none of the device or disk fields.

## Related

[Memory Constraints](/planning/memory-constraints) · [GPU Memory](/gpu/gpu-memory) · [Plan Explanation](/planning/explain) · [Memory Errors](/troubleshooting/memory-errors)
