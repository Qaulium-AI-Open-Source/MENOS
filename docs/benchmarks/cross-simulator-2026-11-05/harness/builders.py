"""IR -> framework circuit builders and runners.

Every runner returns a dict with the raw output in Qiskit's little-endian
convention (bitstring: qubit 0 is the rightmost character; statevector index
bit i = qubit i).
"""

from __future__ import annotations

import math
import os

import numpy as np

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def depolarizing_kraus(p, k):
    if k == 1:
        ops = [PAULI[a] for a in "IXYZ"]
    else:
        ops = [np.kron(PAULI[a], PAULI[b]) for a in "IXYZ" for b in "IXYZ"]
    d2 = len(ops)
    kraus = [math.sqrt(1 - p) * ops[0]] + [math.sqrt(p / (d2 - 1)) * op for op in ops[1:]]
    return kraus


def u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def reverse_axes(state, n):
    return np.asarray(state).reshape([2] * n).transpose(list(range(n - 1, -1, -1))).ravel()


def counts_from_rows(rows, n):
    """rows: (shots, n) 0/1 array with column j = qubit j."""
    rows = np.asarray(rows, dtype=np.int64)
    ints = rows @ (1 << np.arange(n, dtype=np.int64))
    vals, cnt = np.unique(ints, return_counts=True)
    return {format(int(v), f"0{n}b"): int(c) for v, c in zip(vals, cnt)}


def normalize_counts(counts):
    out = {}
    for k, v in counts.items():
        k = k.replace(" ", "")
        out[k] = out.get(k, 0) + int(v)
    return out


# --------------------------------------------------------------------------
# MENOS (native API)
# --------------------------------------------------------------------------

class MenosRunner:
    def __init__(self, variant, case):
        import menos

        self.menos = menos
        self.variant = variant
        self.case = case
        O = menos.Opcode
        self.opc = {
            "h": O.H, "x": O.X, "y": O.Y, "z": O.Z, "s": O.S, "sdg": O.Sdg, "t": O.T, "tdg": O.Tdg,
            "sx": O.SX, "sxdg": O.SXdg, "rx": O.RX, "ry": O.RY, "rz": O.RZ, "p": O.P, "u": O.U,
            "cx": O.CX, "cz": O.CZ, "swap": O.SWAP, "cp": O.CP, "rzz": O.RZZ, "ccx": O.CCX,
        }
        self.kraus_cache = {}

    def build(self, gates):
        case = self.case
        n = case["n"]
        measure = case["task"] in ("counts",)
        c = self.menos.Circuit(n, n if measure else 0)
        append = c.append
        opc = self.opc
        for name, qubits, params in gates:
            if name == "dep1" or name == "dep2":
                key = (name, params[0])
                if key not in self.kraus_cache:
                    self.kraus_cache[key] = [list(k.ravel()) for k in depolarizing_kraus(params[0], len(qubits))]
                c.append_channel(list(qubits), self.kraus_cache[key])
            else:
                append(opc[name], list(qubits), list(params))
        if measure:
            for q in range(n):
                c.append_measure(q, q)
        return c

    def options(self):
        menos = self.menos
        case = self.case
        o = menos.RunOptions()
        o.seed = 12345
        if case["task"] == "statevector":
            o.want_statevector = True
        elif case["task"] == "counts":
            o.shots = case["shots"]
        elif case["task"] in ("expectation", "variational"):
            o.expectation = self.observable(case["observable"], case["n"])
        if case.get("noise") and case.get("menos_noise") == "auto":
            o.noise_method = menos.NoiseMethod.Auto
        if self.variant == "menos-fused":
            o.fuse_diagonal_gates = True
        if self.variant == "menos-1t":
            o.max_threads = 1
        return o

    def observable(self, terms, n):
        from menos._core import Observable, PauliString

        words = (n + 63) // 64
        out = []
        for coeff, paulis in terms:
            x_words, z_words = [0] * words, [0] * words
            for q, P in paulis.items():
                q = int(q)
                if P in "XY":
                    x_words[q // 64] |= 1 << (q % 64)
                if P in "ZY":
                    z_words[q // 64] |= 1 << (q % 64)
            ps = PauliString()
            ps.num_qubits = n
            ps.x_words = x_words
            ps.z_words = z_words
            out.append((ps, complex(coeff)))
        obs = Observable()
        obs.terms = out
        return obs

    def run(self, circuit, options):
        r = self.menos.run(circuit, options)
        return r

    def extract(self, r):
        case = self.case
        out = {"menos_execute_s": r.execute_seconds, "menos_sample_s": r.sample_seconds,
               "menos_peak_host_bytes": r.peak_host_bytes, "exact": r.exact}
        explain = r.explain()
        for line in explain.splitlines():
            if line.startswith("strategy"):
                out["plan"] = line.split(None, 1)[1].strip()
            if line.startswith("region"):
                out.setdefault("regions", []).append(" ".join(t for t in line.split() if t.startswith(("engine=", "backend=", "tier="))))
        out["explain"] = explain
        if case["task"] == "statevector":
            out["statevector"] = np.asarray(r.statevector)
        elif case["task"] == "counts":
            counts = {}
            n = case["n"]
            for k, v in r.counts.items():
                counts[format(int(k, 16), f"0{n}b")] = int(v)
            out["counts"] = counts
        else:
            out["expectation"] = float(np.real(r.expectation_value))
            out["expectation_stderr"] = r.expectation_standard_error
        return out


# --------------------------------------------------------------------------
# Qiskit Aer
# --------------------------------------------------------------------------

class AerRunner:
    def __init__(self, variant, case):
        from qiskit_aer import AerSimulator

        self.case = case
        self.variant = variant
        method = {"aer": "automatic", "aer-sv": "statevector", "aer-mps": "matrix_product_state",
                  "aer-stab": "stabilizer", "aer-dm": "density_matrix", "reference": None}[variant]
        if variant == "reference":
            method = case.get("reference_method", "statevector")
        kwargs = {"method": method, "seed_simulator": 12345}
        if case.get("noise"):
            kwargs["noise_model"] = self.noise_model(case["noise"])
        self.sim = AerSimulator(**kwargs)

    @staticmethod
    def noise_model(noise):
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        p1, p2 = noise
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(p1 * 4 / 3, 1), ["u", "h", "rx", "ry", "rz", "x", "p", "s", "t"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 * 16 / 15, 2), ["cx", "cz", "rzz", "cp", "swap"])
        return nm

    def build(self, gates):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        case = self.case
        n = case["n"]
        qc = QuantumCircuit(n)
        for name, qubits, params in gates:
            if name.startswith("dep"):
                continue  # Aer applies the noise model to the preceding gate
            getattr(qc, name)(*params, *qubits)
        task = case["task"]
        if task == "statevector":
            qc.save_statevector()
        elif task == "counts":
            if self.variant == "reference":
                if case.get("reference_output") == "marginals":
                    for q in range(n):
                        qc.save_expectation_value(SparsePauliOp("Z"), [q], label=f"z{q}")
                else:
                    qc.save_probabilities(label="probs")
            else:
                qc.measure_all()
        else:
            qc.save_expectation_value(self.sparse_op(case["observable"], n), list(range(n)), label="ev")
        return qc

    @staticmethod
    def sparse_op(terms, n):
        from qiskit.quantum_info import SparsePauliOp

        labels, coeffs = [], []
        for coeff, paulis in terms:
            s = ["I"] * n
            for q, P in paulis.items():
                s[n - 1 - int(q)] = P
            labels.append("".join(s))
            coeffs.append(coeff)
        return SparsePauliOp(labels, coeffs)

    def options(self):
        return None

    def run(self, qc, _):
        case = self.case
        if case["task"] == "counts" and self.variant != "reference":
            return self.sim.run(qc, shots=case["shots"]).result()
        return self.sim.run(qc).result()

    def extract(self, res):
        case = self.case
        n = case["n"]
        out = {"aer_method": res.results[0].metadata.get("method"),
               "aer_time_taken": res.results[0].time_taken}
        task = case["task"]
        if task == "statevector":
            out["statevector"] = np.asarray(res.get_statevector().data)
        elif task == "counts":
            if self.variant == "reference":
                data = res.data(0)
                if case.get("reference_output") == "marginals":
                    out["marginals"] = [float((1 - data[f"z{q}"]) / 2) for q in range(n)]
                else:
                    out["probabilities"] = np.asarray(data["probs"])
            else:
                out["counts"] = normalize_counts(res.get_counts())
        else:
            out["expectation"] = float(np.real(res.data(0)["ev"]))
        return out


# --------------------------------------------------------------------------
# PennyLane (lightning.qubit, default.qubit, default.mixed)
# --------------------------------------------------------------------------

class PennyLaneRunner:
    def __init__(self, variant, case):
        import pennylane as qml

        self.qml = qml
        self.case = case
        self.variant = variant
        name = {"lightning": "lightning.qubit", "default.qubit": "default.qubit", "default.mixed": "default.mixed"}[variant]
        shots = case["shots"] if case["task"] == "counts" else None
        self.dev = qml.device(name, wires=case["n"], shots=shots)
        self.H = self.hamiltonian(case["observable"]) if case["task"] in ("expectation", "variational") else None

    def hamiltonian(self, terms):
        qml = self.qml
        ops, coeffs = [], []
        for coeff, paulis in terms:
            factors = [getattr(qml, P)(int(q)) for q, P in paulis.items()]
            ops.append(factors[0] if len(factors) == 1 else qml.prod(*factors))
            coeffs.append(coeff)
        return qml.Hamiltonian(coeffs, ops)

    def build(self, gates):
        qml = self.qml
        case = self.case
        task = case["task"]
        H = self.H
        g = {
            "h": qml.Hadamard, "x": qml.PauliX, "y": qml.PauliY, "z": qml.PauliZ, "s": qml.S, "t": qml.T,
            "sx": qml.SX, "rx": qml.RX, "ry": qml.RY, "rz": qml.RZ, "p": qml.PhaseShift, "u": qml.U3,
            "cx": qml.CNOT, "cz": qml.CZ, "swap": qml.SWAP, "cp": qml.ControlledPhaseShift, "rzz": qml.IsingZZ,
            "ccx": qml.Toffoli,
        }
        kraus_cache = {}

        def apply():
            for name, qubits, params in gates:
                if name == "sdg":
                    qml.adjoint(qml.S)(wires=qubits[0])
                elif name == "tdg":
                    qml.adjoint(qml.T)(wires=qubits[0])
                elif name == "sxdg":
                    qml.adjoint(qml.SX)(wires=qubits[0])
                elif name == "dep1":
                    qml.DepolarizingChannel(params[0], wires=qubits[0])
                elif name == "dep2":
                    key = params[0]
                    if key not in kraus_cache:
                        kraus_cache[key] = depolarizing_kraus(key, 2)
                    qml.QubitChannel(kraus_cache[key], wires=list(qubits))
                else:
                    g[name](*params, wires=list(qubits))

        if task == "statevector":
            def f():
                apply()
                return qml.state()
        elif task == "counts":
            def f():
                apply()
                return qml.counts()
        else:
            def f():
                apply()
                return qml.expval(H)
        return qml.QNode(f, self.dev)

    def options(self):
        return None

    def run(self, qnode, _):
        return qnode()

    def extract(self, res):
        case = self.case
        n = case["n"]
        task = case["task"]
        if task == "statevector":
            return {"statevector": reverse_axes(np.asarray(res), n)}
        if task == "counts":
            return {"counts": {str(k)[::-1]: int(v) for k, v in res.items()}}
        return {"expectation": float(res)}


# --------------------------------------------------------------------------
# Cirq / qsim
# --------------------------------------------------------------------------

class CirqRunner:
    def __init__(self, variant, case):
        import cirq

        self.cirq = cirq
        self.case = case
        self.variant = variant
        self.q = cirq.LineQubit.range(case["n"])
        if variant == "cirq":
            self.sim = cirq.Simulator(dtype=np.complex128)
        elif variant == "cirq-clifford":
            self.sim = cirq.CliffordSimulator()
        elif variant == "cirq-dm":
            self.sim = cirq.DensityMatrixSimulator(dtype=np.complex128)
        elif variant == "qsim":
            import qsimcirq

            self.sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=os.cpu_count()))
        self.psum = self.pauli_sum(case["observable"]) if case["task"] in ("expectation", "variational") else None

    def pauli_sum(self, terms):
        cirq = self.cirq
        total = cirq.PauliSum()
        for coeff, paulis in terms:
            ps = cirq.PauliString({self.q[int(qb)]: getattr(cirq, P) for qb, P in paulis.items()})
            total += coeff * ps
        return total

    def build(self, gates):
        cirq = self.cirq
        q = self.q
        case = self.case
        ops = []
        for name, qubits, params in gates:
            w = [q[i] for i in qubits]
            if name == "h": ops.append(cirq.H(*w))
            elif name == "x": ops.append(cirq.X(*w))
            elif name == "y": ops.append(cirq.Y(*w))
            elif name == "z": ops.append(cirq.Z(*w))
            elif name == "s": ops.append(cirq.S(*w))
            elif name == "sdg": ops.append((cirq.S ** -1)(*w))
            elif name == "t": ops.append(cirq.T(*w))
            elif name == "tdg": ops.append((cirq.T ** -1)(*w))
            elif name == "sx": ops.append((cirq.X ** 0.5)(*w))
            elif name == "sxdg": ops.append((cirq.X ** -0.5)(*w))
            elif name == "rx": ops.append(cirq.rx(params[0])(*w))
            elif name == "ry": ops.append(cirq.ry(params[0])(*w))
            elif name == "rz": ops.append(cirq.rz(params[0])(*w))
            elif name == "p": ops.append(cirq.ZPowGate(exponent=params[0] / math.pi)(*w))
            elif name == "u": ops.append(cirq.MatrixGate(u3_matrix(*params))(*w))
            elif name == "cx": ops.append(cirq.CNOT(*w))
            elif name == "cz": ops.append(cirq.CZ(*w))
            elif name == "swap": ops.append(cirq.SWAP(*w))
            elif name == "cp": ops.append(cirq.CZPowGate(exponent=params[0] / math.pi)(*w))
            elif name == "rzz": ops.append(cirq.ZZPowGate(exponent=params[0] / math.pi, global_shift=-0.5)(*w))
            elif name == "ccx": ops.append(cirq.CCX(*w))
            elif name == "dep1": ops.append(cirq.depolarize(params[0])(*w))
            elif name == "dep2": ops.append(cirq.depolarize(params[0], n_qubits=2)(*w))
            else: raise ValueError(name)
        if case["task"] == "counts":
            ops.append(cirq.measure(*q, key="m"))
        return cirq.Circuit(ops)

    def options(self):
        return None

    def run(self, circuit, _):
        case = self.case
        task = case["task"]
        if task == "counts":
            return self.sim.run(circuit, repetitions=case["shots"])
        if task == "statevector":
            return self.sim.simulate(circuit)
        return self.sim.simulate_expectation_values(circuit, observables=[self.psum])

    def extract(self, res):
        case = self.case
        n = case["n"]
        task = case["task"]
        if task == "statevector":
            return {"statevector": reverse_axes(res.final_state_vector, n)}
        if task == "counts":
            return {"counts": counts_from_rows(res.measurements["m"], n)}
        return {"expectation": float(np.real(res[0]))}


def make_runner(sim, case):
    if sim.startswith("menos"):
        return MenosRunner(sim, case)
    if sim.startswith("aer") or sim == "reference":
        return AerRunner(sim, case)
    if sim in ("lightning", "default.qubit", "default.mixed"):
        return PennyLaneRunner(sim, case)
    if sim in ("cirq", "cirq-clifford", "cirq-dm", "qsim"):
        return CirqRunner(sim, case)
    raise ValueError(sim)
