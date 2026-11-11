# Circuit Builder API

Every class and function of the circuit builder (since Phase 25). For a step-by-step guide, see [Writing Circuits](/build/circuit-builder).

All names are importable from the top-level `menos` package. The C++ equivalents are in `include/menos/builder.h` (namespace `menos`), and the exporter is in `include/menos/qasm3.h`. Unless stated otherwise, errors are `menos.InvalidCircuitError`, which is `menos::common::InvalidCircuit` in C++.

```python
from menos import (CircuitBuilder, Parameter, ParameterExpression,
                   QuantumRegister, ClassicalRegister, Circuit, run, from_qasm3)
```

## `CircuitBuilder`

A circuit under construction: registers, operations whose angles may be parameter expressions, and a global phase. Gate methods change the builder and return it, so calls chain. `inverse()` and `repeat()` return new builders.

### Construction

| Signature | Description |
|---|---|
| `CircuitBuilder(num_qubits=0, num_clbits=0)` | Declares register `q[num_qubits]` and `c[num_clbits]`, each only if its size is non-zero |
| `CircuitBuilder.from_circuit(circuit: Circuit) -> CircuitBuilder` | A builder holding `circuit`'s operations and global phase, with registers `q` and `c` and no parameters |

### Registers

| Signature | Description |
|---|---|
| `add_qreg(name: str, size: int) -> QuantumRegister` | Appends a quantum register after the existing ones; returns it |
| `add_creg(name: str, size: int) -> ClassicalRegister` | Appends a classical register after the existing ones; returns it |
| `qreg(name: str) -> QuantumRegister` | The quantum register of that name. Raises if there is none |
| `creg(name: str) -> ClassicalRegister` | The classical register of that name. Raises if there is none |
| `qregs() -> list[QuantumRegister]` | Quantum registers in declaration order |
| `cregs() -> list[ClassicalRegister]` | Classical registers in declaration order |
| `num_qubits() -> int` | Total qubits over all quantum registers |
| `num_clbits() -> int` | Total classical bits over all classical registers |

`add_qreg` and `add_creg` raise if the name does not match `[A-Za-z_][A-Za-z0-9_]*`, if any register (quantum or classical) already has that name, or if `size` is 0. Indices are assigned in declaration order.

### Gates

Each appends exactly the operation `Circuit.append(Opcode.<NAME>, qubits, angles)` would, and returns the builder. An angle is a `float`, a `Parameter` or a `ParameterExpression`. Qubits are `int` indices; a register element `reg[i]` is one.

| Method | Opcode | Matrix |
|---|---|---|
| `x(qubit)` | `X` | Pauli X |
| `y(qubit)` | `Y` | Pauli Y |
| `z(qubit)` | `Z` | Pauli Z |
| `h(qubit)` | `H` | Hadamard |
| `s(qubit)` | `S` | diag(1, i) |
| `sdg(qubit)` | `Sdg` | diag(1, −i) |
| `t(qubit)` | `T` | diag(1, e^{iπ/4}) |
| `tdg(qubit)` | `Tdg` | diag(1, e^{−iπ/4}) |
| `sx(qubit)` | `SX` | √X |
| `sxdg(qubit)` | `SXdg` | √X† |
| `rx(theta, qubit)` | `RX` | exp(−iθX/2) |
| `ry(theta, qubit)` | `RY` | exp(−iθY/2) |
| `rz(phi, qubit)` | `RZ` | diag(e^{−iφ/2}, e^{iφ/2}) |
| `p(theta, qubit)` | `P` | diag(1, e^{iθ}) |
| `u(theta, phi, lam, qubit)` | `U` | Qiskit's U(θ, φ, λ) |
| `cx(control_qubit, target_qubit)` | `CX` | controlled X |
| `cz(control_qubit, target_qubit)` | `CZ` | controlled Z |
| `swap(qubit1, qubit2)` | `SWAP` | swap |
| `cp(theta, control_qubit, target_qubit)` | `CP` | controlled P(θ) |
| `rzz(theta, qubit1, qubit2)` | `RZZ` | exp(−iθ Z⊗Z/2) |
| `ccx(control_qubit1, control_qubit2, target_qubit)` | `CCX` | Toffoli |

A qubit outside the declared registers raises at the call. Everything else is checked by `run`, exactly as for a `Circuit`: repeated operands, angles that are not finite, and operations after a measurement. Matrices and conventions are in [Supported Gates](/reference/supported-gates) and [Conventions](/concepts/conventions).

### Other operations

| Signature | Description |
|---|---|
| `unitary(matrix: Sequence[complex], qubits: Sequence[int])` | An explicit `2ᵏ × 2ᵏ` unitary on `k = len(qubits)` (1 to 5) qubits, flat and row-major; bit `j` of the index is `qubits[j]`. Same as `Circuit.append_unitary(qubits, matrix)` |
| `measure(qubit: int, clbit: int, readout_error: Sequence[float] \| None = None)` | Terminal measurement of `qubit` into `clbit`; `readout_error` is `[P(0\|0), P(1\|0), P(0\|1), P(1\|1)]`. A `clbit` outside the classical registers raises |
| `measure_all()` | Adds a classical register `meas` of `num_qubits()` bits and measures qubit `i` into `meas[i]`. Raises if `meas` already exists or the circuit has no qubits |
| `channel(kraus: Sequence[Sequence[complex]], qubits: Sequence[int])` | A noise channel by its Kraus operators, each flat and row-major; trace preserving within `1e-12` (checked at `run`). Same as `Circuit.append_channel(qubits, kraus)` |
| `global_phase: float` | Read/write, default `0.0`; the state is multiplied by `e^{i·global_phase}` |

All of these except `global_phase` return the builder.

### Composition

| Signature | Description |
|---|---|
| `compose(other: CircuitBuilder, qubits: Sequence[int] = [], clbits: Sequence[int] = []) -> CircuitBuilder` | Appends `other`'s operations, with `other`'s qubit `j` sent to `qubits[j]` and classical bit `j` to `clbits[j]`, and adds `other.global_phase`. An empty map is the identity, which requires `other` to be no wider than this builder. A non-empty map needs one distinct, in-range entry per qubit (classical bit) of `other`. Parameters merge by name. Changes this builder and returns it |
| `inverse() -> CircuitBuilder` | A new builder with the same registers and the exact inverse: operations reversed, each inverted (`s ↔ sdg`, `t ↔ tdg`, `sx ↔ sxdg`, rotation and phase angles negated, `u(θ, φ, λ) → u(−θ, −λ, −φ)`, self-inverse gates unchanged, unitaries conjugate-transposed), and the global phase negated (a zero phase stays `+0.0`). Raises, naming the operation's index, if the circuit has a measurement or a channel |
| `repeat(reps: int) -> CircuitBuilder` | A new builder with the same registers and `reps` copies of the operations; the global phase is added `reps` times. `repeat(0)` is empty |

### Parameters and conversion

| Signature | Description |
|---|---|
| `parameters() -> list[str]` | Every parameter name used, sorted |
| `bind(values: dict[str \| Parameter, float]) -> Circuit` | An ordinary `Circuit` with every angle evaluated, built through `Circuit.append*`. Raises, naming the parameters, if one has no value, a key names no parameter of the circuit, or a value is not finite |
| `to_circuit() -> Circuit` | `bind({})`: raises, naming them, if the circuit has parameters |
| `operations() -> list[tuple]` | The operations in order (see [Operation tuples](#operation-tuples)); a gate's angle is a `float`, or a `ParameterExpression` when it involves a parameter |
| `to_qasm3() -> str` | OpenQASM 3 text (see [OpenQASM 3 export](#openqasm-3-export)) |

## `Parameter`

| Signature | Description |
|---|---|
| `Parameter(name: str)` | A named run-time angle. `name` must match `[A-Za-z_][A-Za-z0-9_]*` |
| `name: str` | Read-only |

Two parameters with the same name are the same parameter. A `Parameter` is accepted wherever an angle is, and supports the same arithmetic as `ParameterExpression`.

## `ParameterExpression`

An affine combination `c1·θ1 + c2·θ2 + … (+ c0)` of parameters, or a constant.

| Signature | Description |
|---|---|
| `ParameterExpression(constant: float)` / `ParameterExpression(parameter: Parameter)` | Usually created by arithmetic instead |
| `+`, `-` (binary and unary), `*` | With floats, `Parameter`s and expressions, on either side. `*` of two operands that both contain parameters raises (not affine). A parameter appearing twice has its coefficients added |
| `terms: list[tuple[str, float]]` | `(name, coefficient)` in first-appearance order |
| `constant: float \| None` | `c0`, or `None` when the expression has none |
| `is_constant() -> bool` | `True` when there are no parameters |
| `evaluate(values: dict[str \| Parameter, float] = {}) -> float` | The value, computed in the order below. Every parameter of the expression needs a value; a missing value is an `InternalError` |

**Evaluation order:** `((c1·v1 + c2·v2) + …) + c0`, with `c0` added only when present. So a bare parameter evaluates to exactly its value and its negation to exactly minus it.

## `QuantumRegister` and `ClassicalRegister`

Returned by `add_qreg`/`add_creg` and looked up with `qreg`/`creg`.

| Member | Description |
|---|---|
| `name: str`, `offset: int`, `size: int` | Read-only; the register covers indices `offset … offset+size−1` |
| `reg[i] -> int` | The circuit-wide index `offset + i`; raises if `i >= size` |
| `len(reg) -> int` | `size` |

## `run`

```text
run(circuit: Circuit | CircuitBuilder, options: RunOptions) -> Result
```

A builder runs as its `to_circuit()`. A builder that still has parameters raises `InvalidCircuitError` naming them; bind first. Everything else is the [`run`](/reference/python-api#run) of a `Circuit`.

## Operation tuples

`CircuitBuilder.operations()` and `Circuit.operations()` return one tuple per operation:

| Tuple | Meaning |
|---|---|
| `("gate", Opcode, qubits: list[int], angles: list)` | A named gate |
| `("unitary", qubits: list[int], matrix: list[complex])` | An explicit unitary, flat row-major |
| `("measure", qubit: int, clbit: int, readout_error: list[float] \| None)` | A measurement |
| `("channel", qubits: list[int], kraus: list[list[complex]])` | A noise channel |

## OpenQASM 3 export

`CircuitBuilder.to_qasm3()` (C++ `menos::qasm3::emit(const CircuitBuilder&)` and `emit(const Circuit&)`) writes:

- `OPENQASM 3.0;` and `include "stdgates.inc";`
- `input float[64] <name>;` for each parameter;
- `qubit[n] <name>;` / `bit[m] <name>;` for each register, in declaration order;
- `gphase(x);` for a non-zero global phase;
- then the operations.

Gates use the [`from_qasm3`](/reference/python-api#from-qasm3) importer's names. Three are written differently:

- `u` is written `U(…)`;
- `sxdg` is written `inv @ sx`;
- `rzz` is written through the definition `gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }`, which the importer reads back as one `rzz` (exactly equal).

Measurements are written `c[i] = measure q[j];`.

`from_qasm3(to_qasm3(b), values)` gives the identical operations, bit for bit, as `b.bind(values)`, except that a global phase of `-0.0` comes back as `0.0`. It raises `UnsupportedFeatureError`, naming the operation's index, for an explicit unitary, a noise channel or a measurement with a readout error. It raises the same error naming a register or parameter whose name is an OpenQASM 3 keyword, built-in constant or function, standard gate name or `rzz`, or is shared by two of them. A non-finite angle or phase raises `InvalidCircuitError`.

## C++

| Python | C++ |
|---|---|
| `CircuitBuilder(n, m)` | `menos::CircuitBuilder(n, m)` |
| `b.h(0).cx(0, 1)` | `b.h(0).cx(0, 1)` (returns `CircuitBuilder&`) |
| `b.global_phase` | `b.global_phase` (public `double`) |
| `Parameter("t")`, `2 * t + 0.5` | `menos::Parameter("t")`, `2.0 * t + 0.5` (`menos::ParameterExpression`, alias `menos::Angle`) |
| `b.bind({"t": 0.3})` | `b.bind({{"t", 0.3}})` (`std::map<std::string, double, std::less<>>`) |
| `b.operations()` | `b.operations()`: `std::vector<menos::BuilderOperation>`, a variant of `BuilderGate`, `UnitaryOp`, `MeasureOp`, `ChannelOp` |
| `run(b, options)` | `menos::run(b, options)` |
| `b.to_qasm3()` | `menos::qasm3::emit(b)` |

## Verification

```python
import menos
from menos import CircuitBuilder, Parameter, ParameterExpression, QuantumRegister, ClassicalRegister

gates = "x y z h s sdg t tdg sx sxdg rx ry rz p u cx cz swap cp rzz ccx".split()
others = ["unitary", "measure", "measure_all", "channel", "compose", "inverse", "repeat",
          "parameters", "operations", "bind", "to_circuit", "to_qasm3", "global_phase",
          "add_qreg", "add_creg", "qreg", "creg", "qregs", "cregs", "num_qubits", "num_clbits",
          "from_circuit"]
assert sorted(n for n in dir(CircuitBuilder) if not n.startswith("_")) == sorted(gates + others)
assert sorted(n for n in dir(ParameterExpression) if not n.startswith("_")) == sorted(
    ["terms", "constant", "is_constant", "evaluate"]
)
assert sorted(n for n in dir(Parameter) if not n.startswith("_")) == ["name"]
for reg in (QuantumRegister, ClassicalRegister):
    assert sorted(n for n in dir(reg) if not n.startswith("_")) == ["name", "offset", "size"]
assert sorted(n.lower() for n in dir(menos.Opcode) if n[0].isupper()) == sorted(gates)
```
