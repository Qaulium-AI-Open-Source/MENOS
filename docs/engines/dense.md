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
