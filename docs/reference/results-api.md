# Results

Two result types exist, one per entry point.

## Qiskit: `qiskit.result.Result`

Returned by `MENOSJob.result()`. Standard Qiskit.

| Field | Content |
|---|---|
| `success` | `True` |
| `backend_name`, `backend_version` | `"menos"` and the package version |
| `job_id` | UUID |
| `time_taken` | Execute plus sample seconds |
| `get_counts()` | Binary-string keys, highest classical bit left |
| `get_statevector()` | Present when `want_statevector=True` |
| `results[0].seed_simulator` | The seed used |
| `results[0].shots` | The shot count |
| `results[0].header` | A dict: `name`, `n_qubits`, `qreg_sizes`, `creg_sizes`, `qubit_labels`, `clbit_labels`, `memory_slots`, `global_phase`, `metadata` |

The statevector is a NumPy complex128 array of length `2ⁿ` in little-endian index order.

## Native: `menos.Result`

Returned by `menos.run`. See [Python API](/reference/python-api#result) for the full table.

| Property | Type |
|---|---|
| `counts` | `dict[str, int]`, **hexadecimal** keys |
| `statevector` | `numpy.ndarray` or `None` |
| `expectation_value` | `complex` or `None` |
| `exact` | `bool` |
| `actual_state_bytes` | `int` |
| `peak_host_bytes`, `peak_device_bytes` | `int` |
| `execute_seconds`, `sample_seconds` | `float` |
| `explain()` | `str` |

## Differences

| | Qiskit `Result` | Native `Result` |
|---|---|---|
| Counts keys | Binary, register-aware | Hexadecimal |
| Plan report | Not available | `explain()` |
| Replans | Not available | In `explain()` text only |
| Expectation value | Via `MENOSEstimator` | `expectation_value` |

## Verification

```python
from qiskit import QuantumCircuit

from menos import Circuit, Opcode, RunOptions, run
from menos.qiskit import MENOSBackend

qc = QuantumCircuit(2, 2)
qc.x(0)
qc.measure([0, 1], [0, 1])
qiskit_result = MENOSBackend().run(qc, shots=4, seed_simulator=1).result()
assert qiskit_result.get_counts() == {"01": 4}          # binary, highest bit left

native = Circuit(2, 2)
native.append(Opcode.X, [0])
native.append_measure(0, 0)
native.append_measure(1, 1)
options = RunOptions()
options.shots = 4
options.seed = 1
assert run(native, options).counts == {"0x1": 4}        # hexadecimal
```
