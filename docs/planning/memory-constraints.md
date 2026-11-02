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

If every candidate is over the budget, planning throws `ResourceExceeded`, listing each reason, and no allocation has been attempted.

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

## What the budget governs, and what the peak measures

The budget and the reported peak are both about **execution buffers**. Neither is a measure of the process.

- `memory_budget_bytes` is the figure `plan_job` checks each candidate's `estimated_peak_bytes` against. It decides which plan is chosen and whether a job is refused. It is not a cap on the process: MENOS reads no process-memory counter while running, and never stops or throttles a run for exceeding it.
- `peak_host_bytes` and `peak_device_bytes` are the most the runtime's memory ledger saw charged at once during one execution attempt. The ledger counts the buffers that grow with the problem: states, scratch and SVD workspace, samplers and histograms, device buffers and the runtime's combination buffers. It leaves out the allocator's overhead, small fixed-size allocations, and everything allocated while planning, which includes two structures that grow with the circuit rather than the state: the tensor network's contraction-order analysis and the cut path's fragment circuits. The figure is not resident set size.
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
