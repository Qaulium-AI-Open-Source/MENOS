# Supported Gates

Twenty named gates run natively, plus explicit unitaries and measurements. The table below is generated from the same gate table the Qiskit converter and the backend's `Target` read, and an executable check at the bottom of the page fails if the two ever disagree.

## Native gates

| Qiskit name | Native opcode | Qubits | Parameters | Meaning | Clifford | Pauli propagation |
|---|---|---|---|---|---|---|
| `x` | `Opcode.X` | 1 | 0 | Pauli X | yes | yes |
| `y` | `Opcode.Y` | 1 | 0 | Pauli Y | yes | yes |
| `z` | `Opcode.Z` | 1 | 0 | Pauli Z | yes | yes |
| `h` | `Opcode.H` | 1 | 0 | Hadamard | yes | yes |
| `s` | `Opcode.S` | 1 | 0 | S = √Z | yes | yes |
| `sdg` | `Opcode.Sdg` | 1 | 0 | S† | yes | yes |
| `t` | `Opcode.T` | 1 | 0 | T = √S | no | no |
| `tdg` | `Opcode.Tdg` | 1 | 0 | T† | no | no |
| `sx` | `Opcode.SX` | 1 | 0 | √X | yes | yes |
| `sxdg` | `Opcode.SXdg` | 1 | 0 | √X† | yes | yes |
| `rx` | `Opcode.RX` | 1 | 1 | Rotation about X | no | yes |
| `ry` | `Opcode.RY` | 1 | 1 | Rotation about Y | no | yes |
| `rz` | `Opcode.RZ` | 1 | 1 | Rotation about Z (RZ(θ) = diag(e^{-iθ/2}, e^{iθ/2})) | no | yes |
| `p` | `Opcode.P` | 1 | 1 | Phase gate diag(1, e^{iλ}) | no | yes |
| `u` | `Opcode.U` | 1 | 3 | General single-qubit U(θ, φ, λ) | no | yes |
| `cx` | `Opcode.CX` | 2 | 0 | Controlled-X (control, target) | yes | yes |
| `cz` | `Opcode.CZ` | 2 | 0 | Controlled-Z | yes | yes |
| `swap` | `Opcode.SWAP` | 2 | 0 | Swap | yes | yes |
| `cp` | `Opcode.CP` | 2 | 1 | Controlled phase (control, target) | no | no |
| `rzz` | `Opcode.RZZ` | 2 | 1 | Ising ZZ rotation exp(-iθ/2 Z⊗Z) = diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2}); Qiskit `RZZGate`, PennyLane `IsingZZ` | no | no |
| `ccx` | `Opcode.CCX` | 3 | 0 | Toffoli (control, control, target) | no | no |

- **Clifford** gates are the ones the [stabilizer engine](/engines/stabilizer) can run: `x y z h s sdg sx sxdg cx cz swap`.
- **Pauli propagation** accepts the Clifford gates plus `rz rx ry p u`. It does **not** accept `t`, `tdg`, `cp`, `rzz`, `ccx` or unitaries. See [Pauli Propagation](/engines/pauli).
- The **dense**, **MPS**, **tensor-network** and **cut** strategies accept every gate in the table.

## Other operations

| Operation | Support |
|---|---|
| **Unitary** (explicit matrix) | 1 to 5 qubits; must be unitary. Any Qiskit gate with a `to_matrix()` on at most five qubits is lowered to one |
| **Measure** | Terminal only: no operation may follow it on the same qubit |
| `barrier`, `id`, `delay` | Dropped; they cannot change any output |

## Not supported

| Operation | Result |
|---|---|
| `reset` | `UnsupportedFeatureError` |
| Classical control (`if_test`, conditional gates) | `UnsupportedFeatureError` |
| Mid-circuit measurement | `UnsupportedFeatureError` |
| A gate on more than 5 qubits that is not in the table | `UnsupportedFeatureError`; try `transpile(qc, backend)` |
| Unbound parameters | `InvalidCircuitError` |

## Conventions

Parameters and matrices follow Qiskit. Controlled gates list controls first and the target last. See [Conventions](/concepts/conventions).

## Verification

```python
import re
from pathlib import Path

from menos import Opcode
from menos.qiskit._gate_table import GATE_TABLE
from menos.qiskit.target import build_target

page = Path("docs/reference/supported-gates.md").read_text(encoding="utf-8")
documented = set(re.findall(r"^\| `([a-z]+)` \| `Opcode\.(\w+)`", page, flags=re.MULTILINE))
assert documented == {(name, spec.opcode.name) for name, spec in GATE_TABLE.items()}
assert len(GATE_TABLE) == 21
assert {name for name in GATE_TABLE} <= set(build_target().operation_names)
assert sorted(m for m in dir(Opcode) if m[0].isupper()) == sorted(spec.opcode.name for spec in GATE_TABLE.values())
```
