# Exceptions

Every MENOS failure is one of five exception types. They are created by the native extension, so a C++ exception surfaces as exactly one of them with no translation step.

```text
MenosError                      base class; catch this to handle any MENOS failure
├── InvalidCircuitError         the request or circuit is malformed
├── UnsupportedFeatureError     valid, but MENOS cannot represent or execute it
├── ResourceExceededError       no plan fits the memory or time budget
└── InternalError               a MENOS invariant was violated
```

```python
from menos.exceptions import (
    InternalError,
    InvalidCircuitError,
    MenosError,
    ResourceExceededError,
    UnsupportedFeatureError,
)
```

They are also exported from `menos`. The classes' own module is `menos._core`; `menos.exceptions` is the stable import path.

## `InvalidCircuitError`

The circuit or output request is wrong.

| Example message |
|---|
| `request asks for neither counts (shots=0) nor a statevector nor an expectation; nothing to produce` |
| `shots > 0 was requested but the circuit has no Measure operation` |
| `operation 0 (rx): unbound parameter(s) [Parameter(t)]; bind all parameters before running (QuantumCircuit.assign_parameters)` |
| `operation 0: qubit index 5 … out of range` |
| `Unitary matrix is not unitary within tolerance` |

## `UnsupportedFeatureError`

A valid circuit using something MENOS does not support.

| Example message |
|---|
| `operation 1: qubit 0 has a non-terminal measurement (an operation follows a measurement of this qubit)` |
| `operation 1 (reset): not a MENOS-supported named gate and has no matrix representation; unsupported. Try transpile(qc, backend) first.` |
| `operation 0 (qft): acts on 6 qubits, exceeding kMaxUnitaryQubits=5, and is not a MENOS-supported named gate. Try transpile(qc, backend) first.` |
| `MENOSEstimator does not currently support parameterized circuits (1 unbound parameter(s)); bind all parameters first (QuantumCircuit.assign_parameters).` |

## `ResourceExceededError`

No plan fits. See [ResourceExceeded](/runtime/resource-exceeded).

| Example message |
|---|
| `plan_job: no feasible candidate -- direct dense simulation requires 17592186044416 bytes, budget is 7752519680 bytes; …` |
| `plan_job: no feasible candidate within the time budget of … s -- estimated runtime … s exceeds the job's time budget of … s; …` |
| `menos::run: observed runtime of … s exceeded the job's time budget of … s after … safe boundaries on engine=… backend=…, and no different plan meets that budget -- …` |

## `InternalError`

MENOS detected that one of its own invariants was violated. Seeing one means a bug, a corrupted plan, or a degenerate case MENOS does not yet handle, not a mistake in your code. One known case: an observable whose every term is the identity ([details](/observables/pauli-observables#an-all-identity-observable-is-not-supported)). CUDA failures other than running out of device memory also surface as `InternalError`, for example a device error message such as `cuda: cudaMemcpyToSymbol (matrix) failed: …`; a device allocation that fails for memory after the plan's check is a `ResourceExceeded` naming the bytes (since Phase 23).

## Other exceptions you may see

The Qiskit layer raises ordinary Python exceptions for misuse of the interface:

| Exception | When |
|---|---|
| `TypeError` | `MENOSBackend.run` was given something other than a `QuantumCircuit` |
| `ValueError` | `MENOSBackend.run` was given more than one circuit |
| `AttributeError` | An unknown option or backend field |

## Verification

```python
from menos import InvalidCircuitError, ResourceExceededError
from menos.exceptions import (
    InternalError,
    MenosError,
    UnsupportedFeatureError,
)

for error in (InvalidCircuitError, UnsupportedFeatureError, ResourceExceededError, InternalError):
    assert issubclass(error, MenosError)
assert issubclass(MenosError, Exception)
assert MenosError.__mro__[1] is Exception
```
