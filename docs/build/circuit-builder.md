# Circuit Builder

The circuit builder writes a circuit by hand in Python or C++ without an external SDK. It provides named gate methods, run-time parameters, composition, inverse, repetition, named registers and OpenQASM 3 export. It adds no operation and no engine. Every circuit it produces is an ordinary `Circuit`, planned and run exactly like one built with `append` ([Native API](/build/native-api)). The Python classes are bindings of the C++ ones in `include/menos/builder.h`, so both languages behave identically.

## Build and run

```python
import menos

bell = menos.CircuitBuilder(2)          # registers q[2]; no classical bits yet
bell.h(0).cx(0, 1).measure_all()        # every method returns the builder

options = menos.RunOptions()
options.shots = 1000
options.seed = 7
counts = menos.run(bell, options).counts
assert set(counts) <= {"0x0", "0x3"} and sum(counts.values()) == 1000
```

The gate methods are `x y z h s sdg t tdg sx sxdg rx ry rz p u cx cz swap cp rzz ccx`. Angles come before qubits, as in Qiskit: `rx(theta, qubit)`, `u(theta, phi, lam, qubit)`, `cp(theta, control, target)`. Each method appends exactly the operation `Circuit.append` would with the same opcode, qubits and angles. The other methods are:

- `unitary(matrix, qubits)`: a flat, row-major list, as `append_unitary`.
- `measure(qubit, clbit, readout_error=None)`.
- `channel(kraus, qubits)`.
- `measure_all()`: adds a classical register `meas` with one bit per qubit and measures qubit `i` into `meas[i]`, as Qiskit's `measure_all()` does.

A qubit or classical bit outside the declared registers raises `InvalidCircuitError` at the call. Every other check (operand counts, matrix shapes, unitarity) happens at `run`, as for `Circuit`.

`to_circuit()` returns the `Circuit`. `CircuitBuilder.from_circuit(circuit)` goes the other way.

## Registers

```python
import menos

b = menos.CircuitBuilder()
data = b.add_qreg("data", 3)            # qubits 0, 1, 2
anc = b.add_qreg("anc", 1)              # qubit 3
out = b.add_creg("out", 1)              # classical bit 0
b.h(data[0]).cx(data[0], anc[0]).measure(anc[0], out[0])
assert (anc[0], b.qreg("data").size, b.num_qubits()) == (3, 3, 4)
```

Registers are numbered in declaration order. The first register holds indices `0 … size−1` and the next continues after it. This is the OpenQASM 3 importer's rule and Qiskit's. `CircuitBuilder(n, m)` declares `q[n]` and `c[m]`. Register names follow `[A-Za-z_][A-Za-z0-9_]*`, are unique across quantum and classical registers, and a register has at least one bit.

## Parameters

```python
import menos

theta = menos.Parameter("theta")
phi = menos.Parameter("phi")
ansatz = menos.CircuitBuilder(2)
ansatz.ry(theta, 0).ry(theta, 1).rzz(2 * theta - phi, 0, 1).rx(-phi, 1)
assert ansatz.parameters() == ["phi", "theta"]

options = menos.RunOptions()
options.want_statevector = True
for step in range(3):
    circuit = ansatz.bind({"theta": 0.1 * step, phi: 0.5})   # keys: names or Parameters
    state = menos.run(circuit, options).statevector
```

A `Parameter`'s identity is its name. Two `Parameter("theta")` objects are the same parameter, also across `compose`. An angle is a float, a `Parameter`, or an **affine** `ParameterExpression` built with `+ - *` from parameters and floats: `2 * theta - phi + 0.5`. A product of two parameters raises `InvalidCircuitError`.

`bind(values)` returns an ordinary `Circuit` with every angle evaluated. It raises `InvalidCircuitError` naming the parameters when a parameter has no value, a value names no parameter of the circuit, or a value is not finite. Running a builder that still has parameters raises the same error, naming them. Bind first and run the `Circuit`.

**Evaluation order.** A bound angle is computed as `((c1·v1 + c2·v2) + …) + c0`: each coefficient times its value, summed left to right in the order the parameters first appeared, then the constant. A bare parameter binds to exactly its value and `-theta` to exactly minus it. A bound circuit therefore runs bit for bit like the same circuit built by hand with the same arithmetic.

A loop that rebinds the same builder reuses its plan through the [plan cache](/reference/python-api#plan-cache-stats-and-plan-cache-clear), because only the angles change.

## Compose, inverse, repeat

```python
import menos

layer = menos.CircuitBuilder(2)
layer.h(0).cx(0, 1).rz(menos.Parameter("t"), 1)

big = menos.CircuitBuilder(3)
big.compose(layer, [2, 0])               # layer's qubit 0 -> 2, qubit 1 -> 0
big.compose(layer.repeat(2).inverse())   # identity map: qubits 0, 1
assert len(big.operations()) == 3 + 6
```

- `compose(other, qubits=[], clbits=[])` appends `other` with its qubit `j` sent to `qubits[j]` (and likewise for classical bits) and adds its global phase. An empty map is the identity. It changes the builder in place and returns it, unlike Qiskit's `compose`, which returns a new circuit by default.
- `inverse()` returns the exact inverse: the operations reversed, each inverted by the gate table's own rule (`s ↔ sdg`, `sx ↔ sxdg`, angles negated, `u(θ, φ, λ) → u(−θ, −λ, −φ)`), a unitary by its conjugate transpose, and the global phase negated. A circuit containing a measurement or a channel raises `InvalidCircuitError` naming the operation's index.
- `repeat(k)` returns `k` copies of the operations; the global phase is added `k` times.

## OpenQASM 3 export

```python
import menos

theta = menos.Parameter("theta")
b = menos.CircuitBuilder(2, 2)
b.h(0).rzz(theta, 0, 1).sxdg(1).measure(0, 0).measure(1, 1)
text = b.to_qasm3()
assert "input float[64] theta;" in text
again = menos.from_qasm3(text, {"theta": 0.3})   # the same operations as b.bind({"theta": 0.3})
```

`to_qasm3()` writes:

- `OPENQASM 3.0;` and `include "stdgates.inc";`
- an `input float[64]` for each parameter;
- the registers in declaration order;
- `gphase(x);` for a non-zero global phase;
- each operation under the importer's own name: `U` for `u`, `inv @ sx` for `sxdg`, and `rzz` through the definition `gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }`.

[`from_qasm3`](/reference/python-api#from-qasm3) reads that exact definition back as one `rzz`, which equals it exactly; any other `rzz` definition is expanded. Numbers are written in their shortest round-trip form and expressions in their evaluation order, so export then import gives the identical operations, bit for bit. Two limits apply: a global phase of `-0.0` comes back as `0.0`, and an angle too small for the importer to read (a subnormal) is refused on import.

Export raises `UnsupportedFeatureError` naming the operation's index for an explicit unitary, a noise channel or a measurement with a readout error, because OpenQASM 3 has no form for them. It raises the same error naming a register or parameter whose name is an OpenQASM 3 keyword or standard gate name (for example `cx` or `pi`), or is used twice. Only OpenQASM 3 is written; there is no OpenQASM 2 export.

In C++ the exporter is `menos::qasm3::emit(builder)` in `include/menos/qasm3.h`.

## C++

```cpp
#include "menos/menos.h"
#include "menos/qasm3.h"

menos::Parameter theta("theta");
menos::CircuitBuilder b(2, 2);
b.h(0).rzz(2.0 * theta, 0, 1).measure(0, 0).measure(1, 1);
menos::Circuit c = b.bind({{"theta", 0.25}});
menos::Result r = menos::run(c, menos::RunOptions{.shots = 100, .seed = 1});
std::string qasm = menos::qasm3::emit(b);
```

## Not provided

There is no text drawing of a circuit. Parameters are affine only. Binding is through `bind`, not an argument to `run`.
