"""QAOA expectation timing: python qaoa_bench.py <qubits> <depth> <fuse 0|1> <calls>.

Prints the median per-call time of menos._core.run, the largest |error| against
Qiskit Statevector over the first calls, and the process peak RSS."""
import sys, time, statistics, resource
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from menos import _core
from menos.qiskit.convert import convert_circuit
from menos.qiskit._observable_convert import convert_observable

n, p, fuse, calls = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4])
rng = np.random.default_rng(42)
J = {(i, j): rng.uniform(-1, 1) for i in range(n) for j in range(i + 1, n)}
h = rng.uniform(-1, 1, n)
terms = []
for (i, j), w in J.items():
    lab = ["I"] * n; lab[n-1-i] = "Z"; lab[n-1-j] = "Z"; terms.append(("".join(lab), w))
for i in range(n):
    lab = ["I"] * n; lab[n-1-i] = "Z"; terms.append(("".join(lab), h[i]))
H = SparsePauliOp.from_list(terms)

def circ(angles):
    qc = QuantumCircuit(n); qc.h(range(n))
    for l in range(p):
        g, b = angles[2*l], angles[2*l+1]
        for (i, j), w in J.items(): qc.rzz(2*g*w, i, j)
        for i in range(n): qc.rz(2*g*h[i], i)
        qc.rx(2*b, range(n))
    return qc

ro = _core.RunOptions(); ro.shots = 0; ro.want_statevector = False
ro.expectation = convert_observable(H, n)
if fuse: ro.fuse_diagonal_gates = True
times, maxerr = [], 0.0
for k in range(calls + 2):
    qc = circ(rng.uniform(0.1, 2.0, 2*p))
    c = convert_circuit(qc)
    t = time.perf_counter(); r = _core.run(c, ro); dt = time.perf_counter() - t
    if k >= 2: times.append(dt)
    if k < 3:
        maxerr = max(maxerr, abs(r.expectation_value.real - Statevector(qc).expectation_value(H).real))
print(f"n={n} p={p} fuse={int(fuse)} median_ms={statistics.median(times)*1e3:.2f} max_abs_err={maxerr:.1e} peak_rss_MB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}")
