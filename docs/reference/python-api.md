# Python API

The native API, exported from the top-level `menos` package. It is the thin layer over the C++ core; the Qiskit classes are in [Qiskit Backend](/reference/qiskit-backend) and [Estimator](/reference/estimator-api).

```python
from menos import Circuit, NoiseMethod, Opcode, RunOptions, Result, run
```

The top-level module exports exactly: `Circuit`, `NoiseMethod`, `Opcode`, `Result`, `RunOptions`, `run`, `run_batch`, `ReadoutMitigationOptions`, `MitigatedResult`, `readout_calibration_circuits`, `run_readout_mitigated`, `from_qasm3`, `MenosError`, `InvalidCircuitError`, `UnsupportedFeatureError`, `ResourceExceededError`, `InternalError`. `Observable` and `PauliString` are available only from `menos._core`.

## `run`

```text
run(circuit: Circuit, options: RunOptions) -> Result
```

Validates the circuit and the output request, plans the job, executes it, and returns the result. Raises [`InvalidCircuitError`, `UnsupportedFeatureError` or `ResourceExceededError`](/reference/exceptions). When `options.time_budget_seconds` is set, the run is [monitored](/runtime/monitoring) and may be [replanned](/runtime/replanning).

## `run_batch`

```text
run_batch(circuits: Sequence[Circuit], options: RunOptions) -> list[Result]
```

Runs several circuits and returns their results in order. Each circuit is planned and run exactly as `run` does. Circuit `i` runs with the `(i+1)`-th SplitMix64 output of `options.seed` rather than `options.seed` itself, so the circuits draw independent random streams. `menos._core.derive_seeds(seed, count)` returns those seeds. An empty list raises `InvalidCircuitError`.

## `from_qasm3`

```text
from_qasm3(text: str, inputs: dict[str, float] | None = None) -> Circuit
```

Parses an OpenQASM 3 program into a `Circuit` (since Phase 19). `inputs` binds the program's `input` declarations by name; every `input` needs a binding and every binding must name an `input`, otherwise it raises `InvalidCircuitError`. A construct outside the supported static subset (see [Command Line](/build/command-line#openqasm-support)) raises `UnsupportedFeatureError` naming it, its line and its column. The `stdgates.inc` gates have Qiskit's matrices, global phase included.

```python
import math

import menos

circuit = menos.from_qasm3(
    """OPENQASM 3.0;
include "stdgates.inc";
input float theta;
qubit[2] q;
bit[2] c;
h q[0];
ctrl @ rx(theta) q[0], q[1];
c = measure q;
""",
    {"theta": math.pi},
)
options = menos.RunOptions()
options.shots = 100
options.seed = 1
assert sum(menos.run(circuit, options).counts.values()) == 100
```

## Readout mitigation

```text
run_readout_mitigated(target: Circuit, calibration: tuple[Circuit, Circuit] | None,
                      options: RunOptions, mitigation: ReadoutMitigationOptions) -> MitigatedResult
readout_calibration_circuits(target: Circuit) -> tuple[Circuit, Circuit]
ReadoutMitigationOptions()           # calibration_shots: int, read/write
```

Runs `target` together with an all-zero and an all-one calibration circuit, estimates each measured clbit's readout matrix, and inverts it. See [Readout Mitigation](/build/readout-mitigation). `calibration=None` uses `readout_calibration_circuits(target)`, which prepares without gate noise. Pass your own pair to calibrate under a noise model.

| `MitigatedResult` property | Type | Meaning |
|---|---|---|
| `quasi_probabilities` | `dict[str, float]` | Hexadecimal outcome keys (as in `counts`) to the mitigated quasi-probability, over all `2ᵐ` outcomes. May be negative |
| `standard_errors` | `dict[str, float]` | Standard error of each quasi-probability: target shot noise (exact) plus calibration noise (first order) |
| `target_standard_errors` | `dict[str, float]` | The target-shot-noise part alone |
| `gamma` | `float` | `∏ ‖A_c⁻¹‖₁`, the mitigation overhead |
| `calibration` | `list[tuple[int, float, float]]` | `(clbit, P(1|0), P(0|1))` per measured clbit |
| `calibration_shots` | `int` | Shots per calibration circuit |
| `measured_clbits` | `list[int]` | The measured clbits, ascending |
| `raw` | `Result` | The unmitigated target run |
| `calibration_results` | `list[Result]` | The all-zero and all-one calibration runs |
| `exact` | `bool` | Always `False` |

`MitigatedResult.explain()` returns the raw run's report followed by a mitigation section.

## `Circuit`

```text
Circuit(num_qubits: int, num_clbits: int)
Circuit.append(opcode: Opcode, qubits: Sequence[int], params: Sequence[float] = []) -> None
Circuit.append_unitary(qubits: Sequence[int], matrix: Sequence[complex]) -> None
Circuit.append_measure(qubit: int, clbit: int, readout_error: Sequence[float] | None = None) -> None
Circuit.append_channel(qubits: Sequence[int], kraus: Sequence[Sequence[complex]]) -> None
Circuit.append_batch(opcodes, qubits, qubit_offsets, params, param_offsets) -> None
Circuit.num_qubits() -> int
Circuit.num_clbits() -> int
Circuit.global_phase: float          # read/write, default 0.0
```

| Member | Notes |
|---|---|
| `append` | Adds a named gate. `qubits` and `params` must match the opcode's arity and parameter count |
| `append_unitary` | `matrix` is a flat, row-major list of `2ᵏ × 2ᵏ` complex entries; 1 to 5 qubits; must be unitary |
| `append_measure` | Adds a terminal measurement of `qubit` into classical bit `clbit`. `readout_error` is `[P(0|0), P(1|0), P(0|1), P(1|1)]`, the probability of each recorded outcome given the true one |
| `append_channel` | Adds a noise channel: each Kraus operator a flat, row-major `2ᵏ × 2ᵏ` list; 1 to 5 qubits; must be trace preserving within `1e-12`. See [Noise Models](/build/noise-models) |
| `append_batch` | Bulk ingestion of named gates in compressed-sparse-row form (qubit and parameter offsets). Used by the Qiskit converter; equivalent to repeated `append` |
| `global_phase` | Multiplies the statevector by `e^{i·phase}` |

## `Opcode`

The named gates: `X Y Z H S Sdg T Tdg SX SXdg RX RY RZ P U CX CZ SWAP CP RZZ CCX`. See [Supported Gates](/reference/supported-gates).

## `RunOptions`

```text
RunOptions()
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `shots` | `int` | `0` | Samples to draw. `0` requests no counts |
| `want_statevector` | `bool` | `False` | Return the exact statevector |
| `seed` | `int` | `0` | Seed for sampling |
| `memory_budget_bytes` | `int` or `None` | `None` | Memory budget; `None` means 50% of physical memory |
| `expectation` | `Observable` or `None` | `None` | Observable to evaluate |
| `time_budget_seconds` | `float` or `None` | `None` | Wall-clock budget for one execution attempt. Must be finite and > 0, otherwise `run` raises `InvalidCircuitError` |
| `noise_method` | `NoiseMethod` | `NoiseMethod.Exact` | `Auto` lets a noisy counts request be sampled by [trajectories](/build/noise-models#trajectory-simulation) when cheaper; `Exact` keeps the density matrix |
| `expectation_precision` | `float` or `None` | `None` | Target standard error of the expectation value. Finite and `> 0`, and requires `expectation`. Lets a noisy expectation be sampled by trajectories |
| `max_threads` | `int` or `None` | `None` | Most CPU threads for gate application. `None` means every hardware thread; a value above the hardware count is treated as it; `0` raises `InvalidCircuitError`. Results are bit-for-bit identical for every value |
| `fuse_diagonal_gates` | `bool` | `False` | Fuse each run of consecutive diagonal gates (`rz rzz cp cz s t ...`) into one diagonal unitary per block of at most 5 qubits before the CPU dense engine applies them. Exact up to rounding (within `1e-12`), but not bit-for-bit identical to the unfused run, and a CUDA run never fuses; off by default |
| `backend` | `Backend` or `None` | `None` | `None`: the planner chooses. `Backend.Cpu` or `Backend.Cuda` forces one (Phase 17); see [Forcing a backend](/planning/backend-selection#forcing-a-backend). A CUDA request that cannot be honoured raises `UnsupportedFeatureError` or `ResourceExceededError`, never runs on the CPU |

At least one of counts (`shots > 0` with a measured circuit), a statevector, or an expectation must be requested. All fields are read/write.

## `plan_cache_stats` and `plan_cache_clear`

```text
menos._core.plan_cache_stats() -> {"hits": int, "misses": int, "bypasses": int}
menos._core.plan_cache_clear() -> None
```

`run` reuses a plan across calls whose jobs differ only in gate parameter values: a variational loop that rebinds angles plans once. The plan is keyed on the circuit's structure and the options, never on the angles; a job the key cannot cover (a time budget, a noise channel, a two-qubit rotation within `1e-8` of an angle where it degenerates, such as `rzz(0)`) is planned afresh and counted as a bypass, as is a plan whose winner is `factorized` or `cut_and_recombine`. The reused plan is the plan the planner would compute for the new angles; results and `explain()` are unchanged.

## `Backend`

`Backend.Cpu` and `Backend.Cuda`, for `RunOptions.backend` (Phase 17).

## `Result`

| Property | Type | Meaning |
|---|---|---|
| `counts` | `dict[str, int]` | Hexadecimal bitstring keys to counts; empty when none were requested |
| `statevector` | `numpy.ndarray` (complex128) or `None` | The exact statevector |
| `expectation_value` | `complex` or `None` | The expectation value |
| `expectation_standard_error` | `float` or `None` | The measured standard error of a sampled expectation value; `None` for an exact one |
| `exact` | `bool` | Whether nothing was approximated or sampled. `False` for a trajectory result |
| `actual_state_bytes` | `int` | Peak bytes the engine actually used |
| `execute_seconds` | `float` | Seconds executing the attempt that succeeded |
| `sample_seconds` | `float` | Seconds sampling |
| `replans` | `list[dict]` | One dict per abandoned attempt of a job run under `time_budget_seconds`, in order, with `sequence`, `threshold_seconds`, `observed_seconds`, `predicted_seconds`, `boundaries_crossed`, `engine`, `backend` and `reason`. Empty if the job never replanned (Phase 17) |

```text
Result.explain() -> str
```

Returns the [plan report](/planning/explain).

## `Observable` and `PauliString`

From `menos._core`.

| Class | Members |
|---|---|
| `PauliString()` | `num_qubits: int`, `x_words: list[int]`, `z_words: list[int]` |
| `Observable()` | `terms: list[tuple[PauliString, complex]]` |

Assign whole lists: in-place modification of these properties changes a copy and has no effect. See [Pauli Observables](/observables/pauli-observables).

## `kMaxUnitaryQubits`

`menos._core.kMaxUnitaryQubits` is `5`, the largest unitary MENOS accepts.

## Verification

This page is checked against the live package. The following block fails if a documented signature or member stops being true:

```python
import menos
import menos._core as core
from menos import Circuit, Opcode, RunOptions

def sig(obj):
    return obj.__doc__.strip().splitlines()[0]

assert sig(core.run) == "run(circuit: menos._core.Circuit, options: menos._core.RunOptions) -> menos._core.Result"
assert sig(Circuit.__init__) == "__init__(self, num_qubits: int, num_clbits: int) -> None"
assert sig(Circuit.append).startswith("append(self, opcode: menos._core.Opcode, qubits: collections.abc.Sequence[int], params:")
assert sig(Circuit.append_measure) == (
    "append_measure(self, qubit: int, clbit: int, "
    "readout_error: collections.abc.Sequence[float] | None = None) -> None"
)
assert sig(Circuit.append_channel) == (
    "append_channel(self, qubits: collections.abc.Sequence[int], "
    "kraus: collections.abc.Sequence[collections.abc.Sequence[complex]]) -> None"
)
assert sig(core.Result.explain) == "explain(self) -> str"
assert sig(core.from_qasm3) == (
    "from_qasm3(text: str, inputs: collections.abc.Mapping[str, float]) -> menos._core.Circuit"
)

assert sorted(n for n in dir(RunOptions) if not n.startswith("_")) == sorted(
    ["shots", "want_statevector", "seed", "memory_budget_bytes", "expectation", "time_budget_seconds",
     "noise_method", "expectation_precision", "max_threads", "fuse_diagonal_gates", "backend"]
)
assert sorted(n for n in dir(core.Result) if not n.startswith("_")) == sorted(
    ["counts", "statevector", "expectation_value", "expectation_standard_error", "exact",
     "actual_state_bytes", "execute_seconds", "sample_seconds", "replans", "explain"]
)
assert sorted(m for m in dir(Opcode) if m[0].isupper()) == sorted(
    "X Y Z H S Sdg T Tdg SX SXdg RX RY RZ P U CX CZ SWAP CP RZZ CCX".split()
)
assert sorted(menos.__all__) == sorted(
    ["Backend", "Circuit", "NoiseMethod", "Opcode", "Result", "RunOptions", "run", "run_batch", "ReadoutMitigationOptions",
     "MitigatedResult", "readout_calibration_circuits", "run_readout_mitigated", "from_qasm3", "MenosError",
     "InvalidCircuitError", "UnsupportedFeatureError", "ResourceExceededError", "InternalError"]
)
assert sorted(n for n in dir(core.MitigatedResult) if not n.startswith("_")) == sorted(
    ["quasi_probabilities", "standard_errors", "target_standard_errors", "gamma", "calibration",
     "calibration_shots", "measured_clbits", "raw", "calibration_results", "exact", "explain"]
)
assert core.derive_seeds(7, 2) == core.derive_seeds(7, 3)[:2]
assert core.kMaxUnitaryQubits == 5
defaults = RunOptions()
assert (defaults.shots, defaults.want_statevector, defaults.seed) == (0, False, 0)
assert defaults.memory_budget_bytes is None and defaults.time_budget_seconds is None
assert defaults.expectation is None
assert defaults.noise_method == menos.NoiseMethod.Exact and defaults.expectation_precision is None
assert defaults.max_threads is None
assert defaults.fuse_diagonal_gates is False
```
