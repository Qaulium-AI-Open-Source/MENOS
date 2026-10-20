# Results

`MENOSBackend.run` returns a `MENOSJob`, and `job.result()` is a standard Qiskit `qiskit.result.Result`. Everything you know about that class applies.

## What is in it

| Field | Value |
|---|---|
| `success` | `True` for a completed job |
| `backend_name` | `"menos"` |
| `backend_version` | The installed package version |
| `job_id` | A UUID string |
| `time_taken` | Seconds spent executing and sampling |
| `get_counts()` | Present when `shots > 0` and the circuit measures |
| `get_statevector()` | Present when `want_statevector=True` |
| `results[0].seed_simulator` | The seed that was used, including a drawn one |
| `results[0].header` | Circuit name, register sizes, global phase, metadata |

```python
from qiskit import QuantumCircuit

from menos.qiskit import MENOSBackend

qc = QuantumCircuit(2, 2, name="bell")
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

result = MENOSBackend().run(qc, shots=100, seed_simulator=4).result()
print(result.backend_name, result.results[0].seed_simulator, result.time_taken >= 0)
assert result.success
assert result.backend_name == "menos"
assert result.results[0].seed_simulator == 4
assert result.results[0].header["name"] == "bell"
```

## Counts and statevectors

See [Counts](/build/counts) and [Statevectors](/build/statevectors).

## The native `Result`

`menos.run` returns a different object, `menos.Result`, used by the [native API](/build/native-api):

| Property | Type | Meaning |
|---|---|---|
| `counts` | `dict[str, int]` | Hexadecimal bitstring keys (`"0x3"`) to counts |
| `statevector` | `numpy.ndarray` or `None` | complex128 amplitudes |
| `expectation_value` | `complex` or `None` | The expectation value, when one was requested |
| `exact` | `bool` | `True` if nothing was approximated |
| `actual_state_bytes` | `int` | The state bytes the engine reports for the run |
| `peak_host_bytes` | `int` | The most host memory the execution held at once, measured (Phase 21) |
| `peak_device_bytes` | `int` | The same for CUDA device memory; `0` without a CUDA region |
| `execute_seconds` | `float` | Time executing the successful attempt |
| `sample_seconds` | `float` | Time drawing samples |
| `explain()` | `str` | The plan report |

It has no `plan` or `replans` attribute in Python. The plan and any replans are reported through `explain()` only.

## Exactness

`exact` is always `True` for anything reachable from Python. See [Exactness](/concepts/exactness).
