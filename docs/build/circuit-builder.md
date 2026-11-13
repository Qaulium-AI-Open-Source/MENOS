# Writing Circuits

The circuit builder writes quantum circuits by hand in Python or C++ without an external SDK. This page shows how to write circuits with it, step by step. Every function it offers is listed in the [Circuit Builder API](/reference/circuit-builder-api).

The builder adds no operation and no engine. Every circuit it produces is an ordinary `Circuit`, planned and run exactly like one built with `Circuit.append` ([Native API](/build/native-api)). The Python classes are bindings of the C++ ones in `include/menos/builder.h`, so both languages behave identically.

There are three ways to give MENOS a circuit:

| Use | When |
|---|---|
| `menos.CircuitBuilder` (this page) | Writing circuits by hand, with named gates, parameters, registers and composition |
| `menos.Circuit.append(...)` | Generating circuits programmatically, one opcode at a time |
| Qiskit, PennyLane or OpenQASM | You already have circuits in those formats ([MENOSBackend](/build/menos-backend), [PennyLane](/build/pennylane), [`from_qasm3`](/reference/python-api#from-qasm3)) |

## A first circuit

```python
import menos

bell = menos.CircuitBuilder(2)          # 2 qubits, no classical bits yet
bell.h(0).cx(0, 1).measure_all()        # each method returns the builder, so calls chain

options = menos.RunOptions()
options.shots = 1000
options.seed = 7
result = menos.run(bell, options)
print(result.counts)                    # e.g. {'0x0': 497, '0x3': 503}
assert set(result.counts) <= {"0x0", "0x3"} and sum(result.counts.values()) == 1000
```

`menos.run` accepts the builder directly. It can also run `bell.to_circuit()`, the `Circuit` the builder holds.

## Step 1: create the circuit

```python
import menos

a = menos.CircuitBuilder(3)        # qubits q[0..2], no classical bits
b = menos.CircuitBuilder(3, 2)     # qubits q[0..2], classical bits c[0..1]
c = menos.CircuitBuilder()         # empty: declare registers yourself (step 6)
assert (a.num_qubits(), a.num_clbits()) == (3, 0)
assert [r.name for r in b.qregs()] == ["q"] and [r.name for r in b.cregs()] == ["c"]
assert c.num_qubits() == 0
```

Qubits are numbered from 0. MENOS is little-endian: qubit `j` is bit `j` of a statevector index, and classical bit `c` is bit `c` of a counts key. So `'0x2'` means classical bit 1 is 1 and every other bit is 0. See [Conventions](/concepts/conventions).

## Step 2: add gates

Every named gate has a method. **Angles come before qubits** and **controls come before targets**, as in Qiskit:

| Kind | Methods |
|---|---|
| Fixed 1-qubit | `x(q)` `y(q)` `z(q)` `h(q)` `s(q)` `sdg(q)` `t(q)` `tdg(q)` `sx(q)` `sxdg(q)` |
| 1-qubit rotations | `rx(theta, q)` `ry(theta, q)` `rz(phi, q)` `p(theta, q)` `u(theta, phi, lam, q)` |
| 2-qubit | `cx(control, target)` `cz(control, target)` `swap(q1, q2)` `cp(theta, control, target)` `rzz(theta, q1, q2)` |
| 3-qubit | `ccx(control1, control2, target)` |

The matrices are those of [Supported Gates](/reference/supported-gates): Qiskit's, global phase included. For example, `rz(θ) = diag(e^{−iθ/2}, e^{iθ/2})` is not `p(θ) = diag(1, e^{iθ})`.

```python
import math
import menos

ghz = menos.CircuitBuilder(4)
ghz.h(0)
for q in range(3):
    ghz.cx(q, q + 1)

rotations = menos.CircuitBuilder(2)
rotations.rx(math.pi / 2, 0).ry(0.3, 1).u(0.1, 0.2, 0.3, 0).cp(math.pi / 4, 0, 1)

options = menos.RunOptions()
options.want_statevector = True
state = menos.run(ghz, options).statevector
assert abs(state[0] - 2 ** -0.5) < 1e-12 and abs(state[15] - 2 ** -0.5) < 1e-12
```

Each method appends exactly the operation `Circuit.append` would with the same opcode, qubits and angles. A qubit outside the circuit raises `InvalidCircuitError` at the call. Other mistakes are reported by `run`, as for `Circuit`: the same qubit used twice in one gate, a non-unitary matrix, or an operation after a measurement on the same qubit.

## Step 3: arbitrary unitaries

`unitary(matrix, qubits)` applies an explicit `2ᵏ × 2ᵏ` matrix on 1 to 5 qubits. The matrix is a **flat, row-major** list, and bit `j` of its row and column index is `qubits[j]`.

```python
import numpy as np
import menos

iswap = np.array([[1, 0, 0, 0],
                  [0, 0, 1j, 0],
                  [0, 1j, 0, 0],
                  [0, 0, 0, 1]])
b = menos.CircuitBuilder(2)
b.x(0).unitary(iswap.flatten().tolist(), [0, 1])   # |01> -> i|10>

options = menos.RunOptions()
options.want_statevector = True
state = menos.run(b, options).statevector
assert abs(state[2] - 1j) < 1e-12
```

## Step 4: measure

```python
import menos

b = menos.CircuitBuilder(3, 2)
b.h(0).cx(0, 1)
b.measure(0, 0).measure(1, 1)           # qubit -> classical bit

all_measured = menos.CircuitBuilder(3)
all_measured.h(0).measure_all()         # adds register "meas" (3 bits): qubit i -> meas[i]
assert all_measured.creg("meas").size == 3

options = menos.RunOptions()
options.shots = 200
options.seed = 1
assert set(menos.run(b, options).counts) <= {"0x0", "0x3"}
```

- **Measurements are terminal.** No operation may follow a measurement on the same qubit, because mid-circuit measurement is not supported.
- `measure_all()` adds a classical register named `meas` after the existing ones, as Qiskit's `measure_all()` does. A circuit can call it only once, because a second `meas` register would clash.
- `measure(qubit, clbit, readout_error=[P(0|0), P(1|0), P(0|1), P(1|1)])` records the outcome through a classical readout error ([Noise Models](/build/noise-models)).

## Step 5: noise channels and global phase

```python
import math
import menos

p = 0.1
bit_flip = [
    [math.sqrt(1 - p), 0, 0, math.sqrt(1 - p)],   # sqrt(1-p) * I, flat row-major
    [0, math.sqrt(p), math.sqrt(p), 0],           # sqrt(p) * X
]
noisy = menos.CircuitBuilder(1, 1)
noisy.x(0).channel(bit_flip, [0]).measure(0, 0)

options = menos.RunOptions()
options.shots = 1000
options.seed = 3
counts = menos.run(noisy, options).counts     # about 10% '0x0'
assert set(counts) <= {"0x0", "0x1"}

phased = menos.CircuitBuilder(1)
phased.global_phase = math.pi / 3              # multiplies the state by e^{i·pi/3}
```

`channel(kraus, qubits)` takes a list of Kraus operators, each flat and row-major, and must be trace preserving within `1e-12`. A circuit with a channel runs on the density-matrix engine, or on trajectories on request; see [Noise Models](/build/noise-models).

## Step 6: registers

Name groups of qubits and classical bits, and address bits by register:

```python
import menos

b = menos.CircuitBuilder()
data = b.add_qreg("data", 3)            # qubits 0, 1, 2
anc = b.add_qreg("anc", 1)              # qubit 3
syndrome = b.add_creg("syndrome", 1)    # classical bit 0

# parity of the data qubits into the ancilla
for q in range(len(data)):
    b.cx(data[q], anc[0])
b.measure(anc[0], syndrome[0])

assert (anc[0], b.qreg("data").size, b.num_qubits(), b.num_clbits()) == (3, 3, 4, 1)
```

Registers are numbered in **declaration order**. The first quantum register holds qubits `0 … size−1` and the next continues after it; classical registers likewise. This is the OpenQASM 3 importer's rule and Qiskit's. `reg[i]` returns the circuit-wide index, so register bits and plain indices can be mixed freely. Register names follow `[A-Za-z_][A-Za-z0-9_]*`, must differ from every other register's, quantum or classical, and a register has at least one bit.

## Step 7: parameters

A `Parameter` is a named angle whose value is given later. Build the circuit once and bind values many times:

```python
import menos

theta = menos.Parameter("theta")
phi = menos.Parameter("phi")

ansatz = menos.CircuitBuilder(2)
ansatz.ry(theta, 0).ry(theta, 1)
ansatz.rzz(2 * theta - phi, 0, 1)       # an affine expression of parameters
ansatz.rx(-phi, 1)
assert ansatz.parameters() == ["phi", "theta"]

circuit = ansatz.bind({"theta": 0.3, phi: 1.2})   # keys: names or Parameter objects
assert isinstance(circuit, menos.Circuit)
```

**Angles** can be a float, a `Parameter`, or an affine `ParameterExpression` built with `+`, `-` and `*` from parameters and floats, such as `2 * theta - phi + 0.5`. A product of two parameters (`theta * phi`) raises `InvalidCircuitError`. Division is not supported; write `0.5 * theta`.

**Identity is the name.** Two `Parameter("theta")` objects are the same parameter, also across `compose`. Names follow `[A-Za-z_][A-Za-z0-9_]*`.

**`bind(values)`** returns an ordinary `Circuit` and raises `InvalidCircuitError` naming the parameters when:
- a parameter has no value;
- a value names no parameter of the circuit;
- a value is not finite.

`menos.run(builder, …)` refuses a builder that still has parameters in the same way, so bind first.

**Exactness.** A bound angle is computed as `((c1·v1 + c2·v2) + …) + c0`: each coefficient times its value, left to right in the order the parameters first appeared, then the constant. A bare parameter binds to exactly its value and `-theta` to exactly minus it. A bound circuit therefore runs bit for bit like the same circuit written by hand with the same arithmetic.

### A variational loop

```python
import menos
from menos._core import Observable, PauliString

theta = menos.Parameter("theta")
ansatz = menos.CircuitBuilder(2)
ansatz.ry(theta, 0).cx(0, 1)

zz = PauliString()
zz.num_qubits = 2
zz.x_words = [0b00]
zz.z_words = [0b11]                      # Z on qubits 0 and 1
observable = Observable()
observable.terms = [(zz, 1.0 + 0.0j)]

options = menos.RunOptions()
options.expectation = observable

energies = []
for step in range(5):
    value = 0.4 * step
    result = menos.run(ansatz.bind({"theta": value}), options)
    energies.append(result.expectation_value.real)

assert abs(energies[0] - 1.0) < 1e-12    # <ZZ> = 1 at theta = 0
```

Each `run` is planned on its own. Because only the angles change, the [plan cache](/reference/python-api#plan-cache-stats-and-plan-cache-clear) reuses the first plan for the rest of the loop. Observables are covered in [Expectation Values](/build/expectation-values).

## Step 8: reusable blocks — compose, inverse, repeat

Write sub-circuits as functions that return builders, then combine them:

```python
import math
import menos

def qft(n):
    """The quantum Fourier transform on n qubits (Qiskit's QFT, with the final swaps)."""
    b = menos.CircuitBuilder(n)
    for j in reversed(range(n)):
        b.h(j)
        for k in reversed(range(j)):
            b.cp(math.pi / 2 ** (j - k), k, j)
    for i in range(n // 2):
        b.swap(i, n - 1 - i)
    return b

n = 3
circuit = menos.CircuitBuilder(n)
circuit.x(0).x(2)                        # prepare |101>
circuit.compose(qft(n))                  # QFT
circuit.compose(qft(n).inverse())        # and undo it

options = menos.RunOptions()
options.want_statevector = True
state = menos.run(circuit, options).statevector
assert abs(abs(state[0b101]) - 1.0) < 1e-12
```

- **`compose(other, qubits=[], clbits=[])`** appends `other` with its qubit `j` sent to `qubits[j]`, its classical bit `j` to `clbits[j]`, and adds its global phase. Empty maps mean "the same indices". It changes the builder in place and returns it, unlike Qiskit's `compose`, which returns a new circuit by default.
- **`inverse()`** returns a new builder with the exact inverse: the operations reversed, each inverted by the gate table's rule (`s ↔ sdg`, `t ↔ tdg`, `sx ↔ sxdg`, angles negated, `u(θ, φ, λ) → u(−θ, −λ, −φ)`), unitaries conjugate-transposed, and the global phase negated. Parameter expressions are negated too, so a parametric circuit can be inverted before binding. A circuit containing a measurement or a channel cannot be inverted: `InvalidCircuitError` names the operation's index.
- **`repeat(k)`** returns a new builder with `k` copies of the operations; the global phase is added `k` times.

Placing a block on other qubits, and repeating a Trotter step:

```python
import menos

def ising_step(n, dt, j=1.0, h=0.5):
    """One first-order Trotter step of H = J·sum ZZ + h·sum X, time step dt (a Parameter or float)."""
    b = menos.CircuitBuilder(n)
    for q in range(n - 1):
        b.rzz(2 * j * dt, q, q + 1)
    for q in range(n):
        b.rx(2 * h * dt, q)
    return b

dt = menos.Parameter("dt")
evolution = menos.CircuitBuilder(5)
evolution.compose(ising_step(3, dt).repeat(4), [1, 2, 3])   # 4 steps on qubits 1, 2, 3
assert evolution.parameters() == ["dt"]
assert len(evolution.operations()) == 4 * (2 + 3)
circuit = evolution.bind({"dt": 0.05})
```

## Step 9: inspect

```python
import menos

b = menos.CircuitBuilder(2, 1)
b.h(0).rx(menos.Parameter("a"), 1).measure(1, 0)

for op in b.operations():
    print(op)
# ('gate', Opcode.H, [0], [])
# ('gate', Opcode.RX, [1], [ParameterExpression(1.0*a)])
# ('measure', 1, 0, None)

assert b.operations()[0] == ("gate", menos.Opcode.H, [0], [])
assert b.parameters() == ["a"]
plain = b.bind({"a": 0.5})
assert plain.operations()[1] == ("gate", menos.Opcode.RX, [1], [0.5])
assert menos.CircuitBuilder.from_circuit(plain).parameters() == []
```

An operation is a tuple, one of:
- `("gate", Opcode, qubits, angles)`
- `("unitary", qubits, matrix)`
- `("measure", qubit, clbit, readout_error or None)`
- `("channel", qubits, kraus)`

A builder's angles are floats, or `ParameterExpression`s where a parameter is involved. `Circuit.operations()` gives the same tuples, with floats only. `CircuitBuilder.from_circuit(circuit)` turns any `Circuit`, from Qiskit conversion or `from_qasm3` for example, into a builder that can be extended, composed or inverted.

## Step 10: run and explain

```python
import menos

b = menos.CircuitBuilder(2)
b.h(0).cx(0, 1).measure_all()

options = menos.RunOptions()
options.shots = 100
options.seed = 1
result = menos.run(b, options)
print(result.explain())                  # the plan MENOS chose and why
```

Everything in [Running Jobs](/build/running-jobs) applies: shots, statevectors, expectation values, seeds, budgets, `explain()`. The builder changes only how the circuit is written.

## Step 11: OpenQASM 3

```python
import menos

theta = menos.Parameter("theta")
b = menos.CircuitBuilder(2, 2)
b.h(0).rzz(theta, 0, 1).sxdg(1).measure(0, 0).measure(1, 1)

text = b.to_qasm3()
print(text)
again = menos.from_qasm3(text, {"theta": 0.3})   # identical to b.bind({"theta": 0.3})
assert again.operations() == b.bind({"theta": 0.3}).operations()
```

`to_qasm3()` writes:

- `OPENQASM 3.0;` and `include "stdgates.inc";`
- an `input float[64]` for each parameter;
- the registers in declaration order;
- `gphase(x);` for a non-zero global phase;
- each operation under the importer's name: `U` for `u`, `inv @ sx` for `sxdg`, and `rzz` through the definition `gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }`, which [`from_qasm3`](/reference/python-api#from-qasm3) reads back as one `rzz`.

Numbers are written in their shortest round-trip form and expressions in their evaluation order, so export then import gives the identical operations, bit for bit. Two limits apply: a global phase of `-0.0` comes back as `0.0`, and an angle too small for the importer to read (a subnormal) is refused on import.

Export raises `UnsupportedFeatureError`:

- naming the operation's index, for an explicit unitary, a noise channel or a measurement with a readout error, because OpenQASM 3 has no form for them;
- naming the register or parameter, when its name is an OpenQASM 3 keyword or standard gate name (`cx`, `pi`, `input`, …) or two of them share it.

Only OpenQASM 3 is written.

## When something is wrong

| Mistake | Error | When |
|---|---|---|
| Qubit or classical bit outside the circuit | `InvalidCircuitError` naming it | at the method call |
| Register name taken, invalid, or size 0; unknown register; `reg[i]` out of range | `InvalidCircuitError` | at the call |
| Invalid parameter name; product of two parameters | `InvalidCircuitError` | at the call |
| Missing, unknown or non-finite binding | `InvalidCircuitError` naming the parameters | `bind`, `to_circuit`, `run` |
| `compose` map of the wrong length, repeated or out of range; other circuit wider than this one | `InvalidCircuitError` | `compose` |
| `inverse()` of a measurement or channel | `InvalidCircuitError` naming the operation | `inverse` |
| Same qubit twice in a gate, non-unitary matrix, operation after a measurement, … | `InvalidCircuitError` | `run` |
| Unitary, channel or readout error in export; reserved or shared name | `UnsupportedFeatureError` | `to_qasm3` |

## C++

The same API in C++ (`include/menos/builder.h`, `include/menos/menos.h`, `include/menos/qasm3.h`). This example is compiled and run as a test (`tests/api/builder_run_test.cpp`, `BuilderRunTest.DocsExample`):

```cpp
#include "menos/menos.h"
#include "menos/qasm3.h"

menos::Parameter theta("theta");
menos::CircuitBuilder b(2, 2);
b.h(0).rzz(2.0 * theta, 0, 1).measure(0, 0).measure(1, 1);
const menos::Circuit circuit = b.bind({{"theta", 0.25}});
const menos::Result result = menos::run(circuit, menos::RunOptions{.shots = 100, .seed = 1});
const std::string qasm = menos::qasm3::emit(b);
```

Every Python method has the same name in C++. A few differ in form:

- `global_phase` is a public member;
- registers are `menos::QuantumRegister` / `menos::ClassicalRegister`;
- `bind` takes a `std::map<std::string, double, std::less<>>`;
- `inverse()` and `repeat()` return new builders.

## Not provided

There is no text drawing of a circuit. Parameters are affine only. Binding is done with `bind`, not by an argument to `run`. There is no OpenQASM 2 export, and no mid-circuit measurement, `reset` or classical control ([Limitations](/reference/limitations)).
