# Dense

The dense engine holds the complete quantum state as `2ⁿ` complex amplitudes and applies each gate to it. It is exact, supports every gate, and is the **only engine that can produce a statevector**.

## Capabilities

| Output | Supported |
|---|---|
| Counts | Yes |
| Statevector | **Yes** (only engine) |
| Expectation value | Yes |

Every gate MENOS supports runs here, including arbitrary unitaries on up to five qubits. Global phase is tracked, which is what makes a statevector meaningful.

## Cost

The state is `16 · 2ⁿ` bytes (one `complex128` is 16 bytes). Work is counted as one unit per complex multiply-add:

```text
work = 2ⁿ                              initialization
     + Σ over gates of 2^(n+k)         k = number of qubits the gate acts on
     + 2ⁿ   if a statevector is wanted
     + 2ⁿ   if counts are wanted
     + 2ⁿ · (number of terms)   if an expectation value is wanted
```

For the two-qubit Bell circuit with counts that is `4 + 8 + 16 + 4 = 32` work units, which is what the plan report shows.

Cost grows **exponentially** with qubit count: one more qubit doubles both memory and time per gate. That is why dense is competitive for small circuits and infeasible beyond what fits in memory.

## How it runs

Gates are applied by a single generic kernel that handles any operand count: it enumerates the `2^(n-k)` groups of amplitudes that a `k`-qubit gate mixes, gathers each group, multiplies by the gate matrix, and scatters the result back. Counts are drawn by one pass over the cumulative outcome distribution. The same algorithm runs on an NVIDIA GPU; see [CUDA](/engines/cuda).

1- and 2-qubit gates and diagonal gates of any width run on specialised kernels that are bit for bit the generic one. Since Phase 24 these use **AVX2** where the processor has it, chosen when the library loads, on Linux and Windows alike. The library still runs on any x86-64 processor without AVX2, through plain loops that give the same bytes. The AVX2 code performs the generic kernel's multiplies and adds one by one, in its order, with no fused multiply-add, so results do not depend on the processor, the compiler or the thread count. Measured on the development machine (`docs/adr/0028` section 19.8), against the same engine before:
- a 2-qubit gate on one thread runs 2.9–4.8× faster at 16–20 qubits;
- a fused brickwork circuit runs 3.3–4.4× faster on one thread;
- on every thread at 22–24 qubits, where the sweep is limited by memory bandwidth, it runs 1.2–1.6× faster.

### Gate fusion

Since Phase 24 the engine sweeps the state once per **fused block** instead of once per gate (`RunOptions.fuse_gates`, on by default; `--fusion off` on the command line). A gate joins the newest block that shares a qubit with it when one of the two qubit sets contains the other, so every merge removes a pass and widens none: a `u` on qubit 0 followed by a `cx` on qubits 0 and 1 is one 2-qubit pass, and so are the next gates on either of those qubits. Gates on disjoint qubits are never merged, and a diagonal pass (`rz`, `cz`, `rzz`, ...) is never turned into a dense one. The block's matrix is the product of its gates, computed by the same kernel that applies it, so a fused run is exact up to rounding (within 1e-12 of the gate-by-gate run) and keeps every bit-for-bit identity the unfused run has: the same bytes on every thread count, on CPU and CUDA, in memory and on the disk tier. The planner charges the blocks, not the gates, and the report's region line says what fusion did: `fusion=containment blocks=12 absorbed=28` means 12 blocks of two or more gates replaced 40 passes with 12. On the measured brickwork circuits the sweep is 2.0–2.5× faster on every thread at 20–24 qubits (`docs/adr/0028` section 17.5). With `fuse_gates=False` results and the report are byte-identical to a run from before fusion existed.

When the state does not fit the memory budget and the job wants counts or an expectation value, the same kernels run over a **scratch file** instead (the disk tier, since Phase 23): a window of the file that fits the budget is read, every gate that can be applied to it is applied, and it is written back, in as few passes as the circuit allows. Results are bit for bit the in-memory ones. See [Memory Constraints](/planning/memory-constraints#the-disk-tier).

## When it is chosen

| Situation | Outcome |
|---|---|
| You request a statevector | Dense, or a factorized plan of dense regions |
| Small circuit with no cheaper engine | Dense |
| Large non-Clifford circuit, low entanglement | Usually MPS instead |
| More qubits than fit in the memory budget, counts or expectation wanted | Dense on the disk tier, if the scratch directory has the room; the plan report says `tier=disk` |
| More qubits than fit in the memory budget, statevector wanted | Dense is refused; the plan report says how many bytes it needed |

```python
from menos import Circuit, Opcode, RunOptions, run

n = 12
circuit = Circuit(n, 0)
for q in range(n):
    circuit.append(Opcode.RY, [q], [0.2 + 0.05 * q])
for q in range(n - 1):
    circuit.append(Opcode.CX, [q, q + 1])

options = RunOptions()
options.want_statevector = True
result = run(circuit, options)
region = next(line for line in result.explain().splitlines() if line.startswith("region"))
print(region)
assert "engine=dense" in region
assert len(result.statevector) == 2**n
```

## Limits

- Memory is the limit. By default a job may use half of physical memory; see [Memory Constraints](/planning/memory-constraints). Beyond it, the disk tier reaches as far as the scratch directory's space and bandwidth allow, for counts and expectation values only.
- The number of qubits must be representable in the cost arithmetic (at most 62), although memory binds long before that.
- A unitary gate is limited to five qubits.

## Related

[CUDA](/engines/cuda) · [Statevectors](/build/statevectors) · [Memory Constraints](/planning/memory-constraints) · [Large statevector example](/examples/large-statevector)
