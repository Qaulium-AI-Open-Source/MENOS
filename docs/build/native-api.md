# Native API

Besides the Qiskit classes, the package exposes MENOS's own small API from the top-level `menos` module. It is thinner than the Qiskit layer and reaches two things the Qiskit layer does not: **budgets** and the **plan report**.

```python
from menos import Circuit, Opcode, RunOptions, Result, run
```

## Build a circuit

```python
from menos import Circuit, Opcode

circuit = Circuit(3, 3)                    # qubits, classical bits
circuit.append(Opcode.H, [0])              # opcode, qubits, optional parameters
circuit.append(Opcode.CX, [0, 1])
circuit.append(Opcode.RZ, [2], [0.25])     # parameterized gate: params is a list of floats
circuit.append_measure(0, 0)               # qubit -> classical bit
circuit.append_measure(1, 1)
circuit.append_measure(2, 2)
circuit.global_phase = 0.0

assert circuit.num_qubits() == 3 and circuit.num_clbits() == 3
```

For named gate methods, parameters, registers, composition and OpenQASM 3 export, use the [Circuit Builder](/build/circuit-builder), which produces the same `Circuit`. Explicit matrices use `append_unitary(qubits, matrix)` with a flat, row-major list of complex numbers; the matrix must be unitary and act on at most five qubits. Opcodes are the members of `menos.Opcode`; see [Supported Gates](/reference/supported-gates).

## Run it

```python
from menos import Circuit, Opcode, RunOptions, run

circuit = Circuit(2, 2)
circuit.append(Opcode.H, [0])
circuit.append(Opcode.CX, [0, 1])
circuit.append_measure(0, 0)
circuit.append_measure(1, 1)

options = RunOptions()
options.shots = 200
options.seed = 11

result = run(circuit, options)
print(result.counts, result.exact)
assert sum(result.counts.values()) == 200
assert result.exact
```

`RunOptions` fields: `shots`, `want_statevector`, `seed`, `memory_budget_bytes`, `time_budget_seconds` and `expectation`. All have defaults (`0`, `False`, `0`, `None`, `None`, `None`), so set the ones you need. Note the default `seed` is `0`, unlike `MENOSBackend`, which draws a random seed when you give none.

## Budgets

```python
from menos import Circuit, Opcode, RunOptions, run
from menos.exceptions import ResourceExceededError

circuit = Circuit(10, 0)
for q in range(10):
    circuit.append(Opcode.H, [q])
circuit.append(Opcode.CX, [0, 1])

options = RunOptions()
options.want_statevector = True
options.memory_budget_bytes = 1024          # far too small for 2**10 amplitudes
try:
    run(circuit, options)
except ResourceExceededError as error:
    print("refused:", str(error)[:70])
else:
    raise AssertionError("an impossible memory budget must be refused")
```

Budgets are described in [Memory Constraints](/planning/memory-constraints) and [Time Budgets](/runtime/time-budgets).

## Explain

`result.explain()` returns the plan report as a string, including, when a replan happened, every replan with its reason. See [Plan Explanation](/planning/explain).

## Errors

The exceptions are in `menos.exceptions` and are also exported from `menos`: `MenosError` (base), `InvalidCircuitError`, `UnsupportedFeatureError`, `ResourceExceededError` and `InternalError`. See [Exceptions](/reference/exceptions).

## What is not exposed

| Not available from Python | Where it lives |
|---|---|
| Choosing an engine or backend | Always the planner's decision |
| Accuracy policy (`Controlled(tolerance)` for MPS) | C++ only; Python runs are always exact |
| `Result.replans()` and the structured `Plan` | C++ only; Python sees them through `explain()` |
| `preview()` (plan without running) | C++ and the [command line](/build/command-line) |

## A note on `Observable`

`Observable` and `PauliString`, used for [expectation values](/build/expectation-values), are importable only from `menos._core`. The underscore marks the extension module as an implementation detail, but there is currently no public re-export for these two classes.

```python
from menos._core import Observable, PauliString

assert Observable().terms == []
```
