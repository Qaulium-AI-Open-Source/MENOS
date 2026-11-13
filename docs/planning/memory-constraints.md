# Memory Constraints

Every plan has a memory budget, and the planner refuses any candidate that would exceed it **before** allocating anything.

## The default

Unless you set one, the budget is **50% of the machine's total physical memory**. Half is a deliberate choice: it leaves room for the operating system and for everything else the process holds.

## Setting it

The budget is `memory_budget_bytes` on the native API's `RunOptions`:

```python
from menos import Circuit, Opcode, RunOptions, run
from menos.exceptions import ResourceExceededError

circuit = Circuit(24, 0)
for q in range(24):
    circuit.append(Opcode.H, [q])
circuit.append(Opcode.CX, [0, 1])

options = RunOptions()
options.want_statevector = True
options.memory_budget_bytes = 64 * 1024 * 1024      # 64 MiB; a 24-qubit state needs 256 MiB

try:
    run(circuit, options)
except ResourceExceededError as error:
    print(str(error)[:120])
    assert "budget is 67108864 bytes" in str(error)
```

`MENOSBackend` does not expose it, so Qiskit users get the default.

## What is checked

The budget bounds each candidate's **estimated peak bytes**. For the dense engine that is `16 · 2ⁿ`. A candidate over the budget is infeasible with a reason that states both figures:

```text
direct dense simulation requires 17592186044416 bytes, budget is 7752519680 bytes
```

If every candidate is over the budget, planning throws `ResourceExceeded`, listing each reason, and no allocation has been attempted. One exception, since Phase 23: a dense state over the budget can still run from a scratch file, when the job asks for counts or an expectation value and the disk has the room ([the disk tier](#the-disk-tier) below).

## Other engines are far smaller

The same budget lets very different problems through:

| Engine | Example | Memory |
|---|---|---|
| Dense | 20 qubits | 16 MiB |
| Dense | 30 qubits | 16 GiB |
| Stabilizer | 200 qubits | about 13 KB |
| MPS | 100 weakly entangled qubits | about 50 KB |

## Device memory

GPU memory is a separate budget; see [GPU Memory](/gpu/gpu-memory).

## The disk tier

A dense state larger than the budget can still run with the state in a **scratch file** of `16 · 2ⁿ` bytes (since Phase 23, `docs/adr/0027`). The engine reads a window of the file that fits the budget, applies every gate it can to that window, writes it back, and continues with the next window; a gate on a qubit outside the window pairs the file's chunks so that the kernels see the amplitudes they need. The kernels are the in-memory engine's own, so counts and expectation values are bit for bit the same.

The planner **reaches** for the tier, and only reaches: it prices the disk tier only when the in-memory dense state does not fit the budget, the job wants counts or an expectation value (not a statevector), the circuit is noiseless, and the dense engine would be the job's only region. A job that fits in memory plans and reports exactly as before, and the tier is never chosen for speed.

What it checks, with its reasons in the report when it refuses:

- the window and the outputs fit the budget (`the disk tier's window and outputs need N bytes, budget is B bytes`);
- the file fits the scratch directory with a reserve of **5% of the volume or 8 GiB**, whichever is larger (`the disk tier's scratch file of N bytes plus the reserve of R bytes exceeds the A bytes available in DIR`);
- under a time budget, the tier's predicted seconds fit it. The prediction is the work at the dense rate measured on every hardware thread (the single-thread rate when the job is held to fewer threads), plus the bytes written to the file at the directory's measured write bandwidth and the bytes read at its read bandwidth (since Phase 24; before, every crossing was priced at the slower of the two and the work at the single-thread rate). The bandwidth and the threaded rate are measured once per process (the bandwidth once per directory), with at most 1 GiB written and read, and only before a job that could reach the tier; a directory whose bandwidth is unmeasured is a refusal under a time budget, never a guess.

The scratch directory is `RunOptions.scratch_directory` (default: the system's temporary directory, `std::filesystem::temp_directory_path()`), the file `menos-scratch-<pid>-<serial>.state`. It is removed when the run ends, including after an exception or a kill. `RunOptions.tier` forces the tier (`Tier.Disk`, which never falls back and refuses a statevector, a noisy circuit or `Backend.Cuda`) or forbids it (`Tier.InMemory`). `Result.peak_disk_bytes` is the file's size, and `explain()` names the tier:

```text
candidate 0strategy=direct_dense feasible=true winner=true peak_bytes=8590026448 work_units=… tier=disk passes=4 chunk_qubits=25 disk_bytes=34359738368
region 0   qubits=[…] engine=dense backend=cpu tier=disk
estimated  state_bytes=8590026448 work_units=… disk_available_bytes=… scratch_directory="…" disk_write_bandwidth=… disk_read_bandwidth=… (measured)
actual     … peak_host_bytes=8590026448 peak_device_bytes=0 peak_disk_bytes=34359738368
```

That is the plan for 31 qubits on the development machine: a 32 GiB file against a 15.85 GiB budget, 4 passes with an 8 GiB window, 7–13 minutes for 100 gates (`docs/adr/0027` section 10.3). The practical limits are the disk's space and its bandwidth: every pass reads and writes the whole file.

```python
from menos import Circuit, Opcode, RunOptions, Tier, run

circuit = Circuit(14, 14)
for q in range(14):
    circuit.append(Opcode.RY, [q], [0.1 * (q + 1)])
for q in range(13):
    circuit.append(Opcode.CX, [q, q + 1])
for q in range(14):
    circuit.append_measure(q, q)

options = RunOptions()
options.shots = 100
options.tier = Tier.Disk          # force it; the planner would keep a 14-qubit state in memory
result = run(circuit, options)
assert "tier=disk" in result.explain()
assert result.peak_disk_bytes == 16 * 2**14
```

A host-offloaded CUDA tier (the state in RAM, computed on the GPU window by window) was measured and not built: `Tier.HostOffloaded` is refused with `UnsupportedFeature`. See [Limitations](/reference/limitations#memory-tiers) for the tier's limits, including a note for WSL.

## What the budget governs, and what the peak measures

The budget and the reported peak are both about **execution buffers**. Neither is a measure of the process.

- `memory_budget_bytes` is the figure `plan_job` checks each candidate's `estimated_peak_bytes` against. It decides which plan is chosen and whether a job is refused. It is not a cap on the process: MENOS reads no process-memory counter while running, and never stops or throttles a run for exceeding it.
- `peak_host_bytes`, `peak_device_bytes` and `peak_disk_bytes` are the most the runtime's memory ledger saw charged at once during one execution attempt. The ledger counts the buffers that grow with the problem: states, scratch and SVD workspace, samplers and histograms, device buffers and the runtime's combination buffers. It leaves out the allocator's overhead, small fixed-size allocations, and everything allocated while planning, which includes two structures that grow with the circuit rather than the state: the tensor network's contraction-order analysis and the cut path's fragment circuits. The figure is not resident set size.
- `run_batch` plans and budgets each circuit on its own and holds every result until it returns. The results already produced are not part of any later circuit's budget, so a batch can hold more than the budget even when each circuit fits it.

So the estimate is an upper bound on the peak, and the peak is a lower bound on the process's own use. If you need a strict limit on process memory, enforce it outside MENOS; MENOS has no mechanism for it, and `ResourceExceeded` means the planner refused, not that such a limit was hit. The gaps and the options for closing them are recorded in `docs/adr/0026` section 11.

## Estimates are upper bounds, checked by measurement

Each engine's estimate counts everything it holds at once: the state, scratch and working buffers, the SVD workspace, samplers and the counts histogram. Each container is charged with the allocator's own overhead. The estimate is an upper bound, not a prediction (since Phase 21).

Two checks run on a fixture set covering every engine and output kind, on Linux and Windows:

- the heap memory a run actually allocated never exceeds the estimate;
- the process's resident memory grows by no more than the estimate plus a fixed baseline. This check is asserted on Linux and reported on Windows.

Where an estimate is well above use, it charges the worst case the code permits: for example, every shot drawing a different outcome.

`Result.peak_host_bytes` and `Result.peak_device_bytes` report the most memory the execution really held. A memory ledger adds up every buffer that grows with the problem as it is allocated and freed, so the figure never exceeds the estimate. `explain()` prints both on its `actual` line. `actual_state_bytes` is the older, engine-reported figure, kept for compatibility.

## Related

[Requirements](/get-started/requirements) · [Memory Errors](/troubleshooting/memory-errors) · [ResourceExceeded](/runtime/resource-exceeded)
