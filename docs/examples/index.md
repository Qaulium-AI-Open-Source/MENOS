# Examples

Every example here is a real, executable program. Each one lives in the repository's `examples/` directory, asserts its own results (mostly against Qiskit's reference simulator), and is run by the test suite, so what you see is what runs.

Run any of them from the repository root:

```bash
.venv/bin/python examples/bell_state.py
```

## Getting started

| Example | Shows |
|---|---|
| [Bell State](/examples/bell-state) | The smallest useful circuit, through Qiskit |
| [GHZ State](/examples/ghz-state) | Counts and the exact statevector |
| [Counts](/examples/counts) | Seeds, reproducibility, partial measurement |
| [Statevector](/examples/statevector) | Exact amplitudes, checked against Qiskit |
| [Random Circuits](/examples/random-circuits) | Transpiling arbitrary circuits for the backend |
| [Large Statevector](/examples/large-statevector) | A 20-qubit exact state |

## Observables

| Example | Shows |
|---|---|
| [Expectation Value](/examples/expectation-value) | An exact `⟨ZZ + 0.5 XX⟩` |
| [SparsePauliOp](/examples/sparse-pauli-op) | Many terms and arrays of observables |
| [EstimatorV2](/examples/estimator) | Several pubs with bound parameters |

## Engines and planning

| Example | Shows |
|---|---|
| [Planner Explanation](/examples/planner-explain) | Reading a plan report |
| [Stabilizer](/examples/stabilizer) | A 60-qubit Clifford circuit |
| [MPS](/examples/mps) | A 100-qubit circuit no statevector could hold |
| [Pauli Propagation](/engines/pauli) | An expectation with no state |
| [Advanced Workload](/examples/cut-and-recombine) | Exact circuit cutting |
| [Noise](/examples/noise) | A Bell pair under a Qiskit Aer noise model, simulated exactly |

## Other frameworks

| Example | Shows |
|---|---|
| [PennyLane](/examples/pennylane) | A variational energy and its gradient on the `menos.qubit` device |
| `examples/compare_simulators.ipynb` (notebook, in the repository) | The same circuits through MENOS, Qiskit Aer and PennyLane Lightning: agreement and timing |

## Hardware and budgets

| Example | Shows |
|---|---|
| [CUDA](/examples/cuda) | Which backend the planner chose |
| [Time Budget](/examples/time-budget) | Plan-time refusals |
| [Replanning](/examples/replanning) | A run that overran its budget and was restarted |

## Which examples need what

All of them need only the default install plus Qiskit (the `qiskit` extra), except [CUDA](/examples/cuda), which runs on any build and reports what it used, [Noise](/examples/noise), which needs Qiskit Aer (the `noise` extra), and [PennyLane](/examples/pennylane), which needs the `pennylane` extra (CPython 3.11 or later).
