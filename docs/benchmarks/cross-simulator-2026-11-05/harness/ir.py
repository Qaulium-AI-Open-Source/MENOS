"""Framework-neutral circuit IR and the benchmark circuit families.

A gate is (name, qubits, params). Names follow Qiskit's standard gate
names (h x y z s sdg t tdg sx sxdg rx ry rz p u cx cz swap cp rzz ccx).
Two extra names mark noise channels: 'dep1' (q,) (p,) and 'dep2' (q1,q2) (p,)
where p is the probability of a uniformly random NON-identity Pauli
(Cirq / PennyLane convention; Aer's lambda = p * d^2 / (d^2 - 1)).

An observable is a list of (coeff, {qubit: 'X'|'Y'|'Z'}) terms.
"""

from __future__ import annotations

import math
import random

NATIVE_BASIS = [
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz",
    "p", "u", "cx", "cz", "swap", "cp", "rzz", "ccx",
]


def qiskit_to_ir(qc):
    """Transpile a Qiskit circuit to the native basis (translation only) and walk it."""
    from qiskit import transpile

    tqc = transpile(qc, basis_gates=NATIVE_BASIS, optimization_level=0)
    gates = []
    for inst in tqc.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        if name == "u3":
            name = "u"
        if name not in NATIVE_BASIS:
            raise ValueError(f"gate {name} not in native basis after transpile")
        qubits = tuple(tqc.find_bit(q).index for q in inst.qubits)
        params = tuple(float(p) for p in inst.operation.params)
        gates.append((name, qubits, params))
    return gates, float(tqc.global_phase)


# --------------------------------------------------------------------------
# Circuit families
# --------------------------------------------------------------------------

def dense_random(n, depth, seed, offset=0):
    """Brickwork: random u3 on every qubit, then CX on alternating pairs."""
    rng = random.Random(seed)
    gates = []
    for layer in range(depth):
        for q in range(n):
            gates.append(("u", (offset + q,), (rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi))))
        start = layer % 2
        for q in range(start, n - 1, 2):
            gates.append(("cx", (offset + q, offset + q + 1), ()))
    return gates


def ghz(n):
    return [("h", (0,), ())] + [("cx", (i, i + 1), ()) for i in range(n - 1)]


def random_clifford(n, depth, seed):
    rng = random.Random(seed)
    one_q = ["h", "s", "sdg", "x", "z", "h", "h"]
    gates = []
    for layer in range(depth):
        for q in range(n):
            gates.append((rng.choice(one_q), (q,), ()))
        start = layer % 2
        for q in range(start, n - 1, 2):
            gates.append((rng.choice(["cx", "cz"]), (q, q + 1), ()))
    return gates


def low_entanglement_1d(n, depth, seed, theta=0.1):
    """Shallow nearest-neighbour brickwork of small rotations: bond dimension <= 2^depth."""
    rng = random.Random(seed)
    gates = []
    for layer in range(depth):
        for q in range(n):
            gates.append(("ry", (q,), (rng.uniform(-0.6, 0.6),)))
        start = layer % 2
        for q in range(start, n - 1, 2):
            gates.append(("rzz", (q, q + 1), (theta,)))
    return gates


def qaoa_maxcut(n, p, seed):
    """Ring plus a random perfect matching (degree 3), p layers of rzz / rx."""
    rng = random.Random(seed)
    edges = [(i, (i + 1) % n) for i in range(n)]
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(0, n - 1, 2):
        a, b = perm[i], perm[i + 1]
        if (b - a) % n not in (1, n - 1):
            edges.append((min(a, b), max(a, b)))
    gates = [("h", (q,), ()) for q in range(n)]
    for _ in range(p):
        gamma, beta = rng.uniform(0, math.pi), rng.uniform(0, math.pi / 2)
        for a, b in edges:
            gates.append(("rzz", (a, b), (gamma,)))
        for q in range(n):
            gates.append(("rx", (q,), (2 * beta,)))
    observable = [(1.0, {a: "Z", b: "Z"}) for a, b in edges]
    return gates, observable


def trotter_tfim(n, steps, seed, dt=0.1, J=1.0, h=1.0):
    gates = [("h", (q,), ()) for q in range(n)]
    for _ in range(steps):
        for q in range(0, n - 1, 2):
            gates.append(("rzz", (q, q + 1), (-2 * J * dt,)))
        for q in range(1, n - 1, 2):
            gates.append(("rzz", (q, q + 1), (-2 * J * dt,)))
        for q in range(n):
            gates.append(("rx", (q,), (-2 * h * dt,)))
    observable = [(1.0, {q: "Z", q + 1: "Z"}) for q in range(n - 1)] + [(1.0, {q: "X"}) for q in range(n)]
    return gates, observable


def heisenberg(n):
    obs = []
    for q in range(n - 1):
        for P in "XYZ":
            obs.append((1.0, {q: P, q + 1: P}))
    return obs


def hea_ansatz(n, reps, seed):
    """Hardware-efficient ansatz: RY+RZ on every qubit, CX ladder, repeated."""
    rng = random.Random(seed)
    gates = []
    for r in range(reps + 1):
        for q in range(n):
            gates.append(("ry", (q,), (rng.uniform(-math.pi, math.pi),)))
            gates.append(("rz", (q,), (rng.uniform(-math.pi, math.pi),)))
        if r < reps:
            for q in range(n - 1):
                gates.append(("cx", (q, q + 1), ()))
    return gates


def qft(n):
    from qiskit.circuit.library import QFT

    gates, phase = qiskit_to_ir(QFT(n).decompose())
    return gates


def grover(n, seed, iterations=None):
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    marked = [rng.randint(0, 1) for _ in range(n)]
    k = iterations if iterations is not None else max(1, int(round(math.pi / 4 * math.sqrt(2 ** n))))
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(k):
        for q, bit in enumerate(marked):
            if bit == 0:
                qc.x(q)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for q, bit in enumerate(marked):
            if bit == 0:
                qc.x(q)
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
    gates, phase = qiskit_to_ir(qc)
    marked_str = "".join(str(b) for b in reversed(marked))  # Qiskit order: qubit 0 rightmost
    return gates, marked_str, k


def qpe(m, phi=0.3125):
    """m precision qubits + 1 target; U = p(2*pi*phi) on the target prepared in |1>.
    U^(2^k) is the single gate p(2^k * 2*pi*phi), so each controlled power is one cp."""
    from qiskit.circuit.library import QFT

    gates = [("x", (m,), ())] + [("h", (k,), ()) for k in range(m)]
    for k in range(m):
        gates.append(("cp", (k, m), ((2 ** k) * 2 * math.pi * phi,)))
    iqft, phase = qiskit_to_ir(QFT(m, inverse=True).decompose())
    gates += iqft
    expected = format(int(round(phi * 2 ** m)) % (2 ** m), f"0{m}b")
    return gates, expected


def with_noise(gates, p1, p2):
    out = []
    for g in gates:
        out.append(g)
        if len(g[1]) == 1:
            out.append(("dep1", g[1], (p1,)))
        elif len(g[1]) == 2:
            out.append(("dep2", g[1], (p2,)))
    return out


def factorized(blocks, size, depth, seed):
    gates = []
    for b in range(blocks):
        gates += dense_random(size, depth, seed + b, offset=b * size)
    return gates
