# Qiskit Integration

MENOS integrates with Qiskit through two classes in `menos.qiskit`:

| Class | Qiskit interface | Purpose |
|---|---|---|
| `MENOSBackend` | `BackendV2` | Run circuits and get counts or a statevector |
| `MENOSEstimator` | `BaseEstimatorV2` | Compute expectation values |
| `MENOSSampler` | `BaseSamplerV2` | Sample measured circuits, one `BitArray` per classical register ([SamplerV2](/build/sampler)) |

`MENOSJob` (a `JobV1`) is the object `run` returns. There is no provider class.

## Running circuits

```python
from qiskit import QuantumCircuit

from menos.qiskit import MENOSBackend

backend = MENOSBackend()
qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure([0, 1, 2], [0, 1, 2])

job = backend.run(qc, shots=500, seed_simulator=3)
print(job.status(), job.result().get_counts())
assert job.result().success
```

Options are `shots` (default 1024), `seed_simulator` (default: a fresh random seed) and `want_statevector` (default `False`). Anything else raises `AttributeError`. See [MENOSBackend](/build/menos-backend).

## Transpiling

`MENOSBackend` advertises a `Target` built from exactly the gates it can execute, so Qiskit's transpiler knows how to lower a circuit for it:

```python
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate

from menos.qiskit import MENOSBackend

backend = MENOSBackend()
qc = QuantumCircuit(3)
qc.append(QFTGate(3), [0, 1, 2])
qc.measure_all()

lowered = transpile(qc, backend, seed_transpiler=0)
print(sorted(lowered.count_ops()))
counts = backend.run(lowered, shots=100, seed_simulator=1).result().get_counts()
assert sum(counts.values()) == 100
```

The backend's target has no fixed qubit count (`num_qubits=None`), matching Qiskit's own `BasicSimulator`, because the real limit is memory, not a device size.

## What the converter accepts

| Circuit content | Behaviour |
|---|---|
| A gate in the [supported table](/reference/supported-gates) | Run natively |
| Any other gate with a matrix on 1 to 5 qubits | Lowered to a dense unitary and run |
| A gate on more than 5 qubits that is not native | `UnsupportedFeatureError` ("try `transpile(qc, backend)` first") |
| `barrier`, `id`, `delay` | Dropped; they do not change the result |
| `measure` | Supported, but must be terminal for its qubit |
| `reset`, classical control (`if_test`), mid-circuit measurement | `UnsupportedFeatureError` |
| Unbound `Parameter`s | `InvalidCircuitError`; call `assign_parameters` first |

Global phase is preserved: a circuit's `global_phase` is part of the statevector MENOS returns.

## Estimator

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from menos.qiskit import MENOSEstimator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

estimator = MENOSEstimator()
pub_result = estimator.run([(qc, [SparsePauliOp("ZZ"), SparsePauliOp("XX")])]).result()[0]
print(pub_result.data.evs)
```

Results are exact, so `stds` is always zero and any requested `precision` is recorded in the metadata but never used to add noise. See [EstimatorV2](/build/estimator).

## What the Qiskit layer does not expose

The Qiskit classes cover the common path and deliberately nothing more. These are reachable only through the [native API](/build/native-api):

- time and memory budgets (`time_budget_seconds`, `memory_budget_bytes`)
- the plan report, `explain()`
- arbitrary observables other than through `MENOSEstimator`

Backend selection (CPU or CUDA) is automatic and is not exposed anywhere in Python.
