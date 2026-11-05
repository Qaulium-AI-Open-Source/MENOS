# Memory Errors

Each entry is a **symptom**, its **cause**, and a **solution**.

## `ResourceExceededError`: direct dense simulation requires N bytes

| | |
|---|---|
| **Symptom** | `ResourceExceededError: plan_job: no feasible candidate -- direct dense simulation requires 17592186044416 bytes, budget is 7752519680 bytes; …` |
| **Cause** | You asked for a statevector (or the circuit is only feasible densely), and the state needs more memory than the budget. A dense state of `n` qubits needs `16 · 2ⁿ` bytes |
| **Solution** | Ask for less. Request counts or an expectation value instead (then a state over the budget can run from a scratch file, the [disk tier](/planning/memory-constraints#the-disk-tier)), or use a smaller circuit, or split it. If the machine really has the memory, raise `memory_budget_bytes` ([native API](/build/native-api)) |

## The default budget is lower than my RAM

| | |
|---|---|
| **Symptom** | A job that "should fit" is refused |
| **Cause** | By default only **50%** of physical memory is budgeted |
| **Solution** | Set `RunOptions.memory_budget_bytes` explicitly through the native API. `MENOSBackend` has no such option |

## The message lists several reasons

| | |
|---|---|
| **Symptom** | A long error with several sections separated by semicolons |
| **Cause** | It names every candidate's refusal reason, so you can see what blocked each |
| **Solution** | Read each reason. A statevector request removes the stabilizer and MPS candidates ("only the dense engine can produce a statevector"), leaving only dense, which then fails on memory |

## A large job is refused but I only need an expectation value

| | |
|---|---|
| **Symptom** | A big circuit fails under a statevector request |
| **Cause** | A statevector is `2ⁿ` amplitudes regardless of what you do with it |
| **Solution** | Request an expectation value through `MENOSEstimator`. Pauli propagation, MPS, tensor-network and cut strategies avoid the full state. See [Expectation Values](/build/expectation-values) |

## The disk tier has no room

| | |
|---|---|
| **Symptom** | `… the disk tier's scratch file of N bytes plus the reserve of R bytes exceeds the A bytes available in DIR` |
| **Cause** | The state's scratch file (`16 · 2ⁿ` bytes) plus the reserve (5% of the volume or 8 GiB) does not fit the scratch directory's free space |
| **Solution** | Point `RunOptions.scratch_directory` (CLI `--scratch-dir`) at a volume with the room, or free space on this one. Under WSL, name a directory on the ext4 volume rather than a `tmpfs` `/tmp` |

## The GPU ran out of memory

| | |
|---|---|
| **Symptom** | `ResourceExceededError: the CUDA device could not provide N bytes of device memory: F bytes are free now and P are retained in MENOS's pool; …` |
| **Cause** | Device memory was claimed by another process after MENOS measured it (since Phase 23 this is a refusal, not an `InternalError`) |
| **Solution** | Run the job again, or free the device; see [Troubleshooting CUDA](/gpu/troubleshooting-cuda) |

## The process itself runs out of memory

| | |
|---|---|
| **Symptom** | The operating system kills the process |
| **Cause** | Memory outside what MENOS budgets: for example large counts dictionaries, statevector copies in NumPy, or other processes. MENOS budgets engine state, not your program's other allocations |
| **Solution** | Request less, or lower the budget to leave headroom |

## See also

[Memory Constraints](/planning/memory-constraints) · [ResourceExceeded](/runtime/resource-exceeded) · [Requirements](/get-started/requirements)
