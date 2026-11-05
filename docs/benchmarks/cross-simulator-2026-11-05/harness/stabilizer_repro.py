"""Reproducer: MENOS stabilizer sampler returns a wrong deterministic outcome.

A 50-qubit, 54-gate Clifford circuit (stabilizer_repro.json, found by
delta-debugging a random Clifford circuit) leaves qubits 48 and 49 perfectly
anticorrelated: the exact joint distribution on (48, 49) is {'10': 0.5,
'01': 0.5} (Qiskit StabilizerState; Qiskit Aer's stabilizer method agrees).
MENOS's stabilizer engine samples qubit 49 correctly at random but always
reports qubit 48 as 1, so it returns the impossible outcome 11 half of the
time. Measuring qubit 48 alone is correct (0.5), and <Z_48> through the
expectation path is correct (0), so the tableau is right and the fault is in
the sign of a deterministic outcome once another qubit has been collapsed.
The same symptom appeared on random Clifford circuits of 40, 50, 64 and 100
qubits (depth 50, seed 19) and never at 33 qubits or fewer.

usage: python stabilizer_repro.py   (needs menos, qiskit)
"""

import json
import os

import menos
from qiskit import QuantumCircuit
from qiskit.quantum_info import StabilizerState

HERE = os.path.dirname(os.path.abspath(__file__))
gates = [(a, tuple(b)) for a, b, _ in json.load(open(os.path.join(HERE, "..", "stabilizer_repro.json")))]
OPC = {"h": menos.Opcode.H, "s": menos.Opcode.S, "sdg": menos.Opcode.Sdg, "x": menos.Opcode.X,
       "z": menos.Opcode.Z, "cx": menos.Opcode.CX, "cz": menos.Opcode.CZ}

qc = QuantumCircuit(50)
for name, qubits in gates:
    getattr(qc, name)(*qubits)
print("exact joint P(q48 q49):", StabilizerState(qc).probabilities_dict([48, 49]))

circuit = menos.Circuit(50, 2)
for name, qubits in gates:
    circuit.append(OPC[name], list(qubits), [])
circuit.append_measure(48, 0)
circuit.append_measure(49, 1)
options = menos.RunOptions()
options.shots = 1024
options.seed = 1
result = menos.run(circuit, options)
print("MENOS counts (clbit0 = q48, clbit1 = q49):", {format(int(k, 16), "02b")[::-1]: v for k, v in result.counts.items()})
print([line for line in result.explain().splitlines() if line.startswith("strategy")][0])
