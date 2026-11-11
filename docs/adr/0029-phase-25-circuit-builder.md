# ADR 0029: Phase 25 — Circuit builder (C++ and Python)

## Status

**Accepted design; implemented 2026-11-11; verification pending (section 10).** The owner answered section 9 on 2026-11-11, taking every recommended option (see "Owner answers"), and approved the implementation plan the same day. The owner started Phase 25 on 2026-11-11, while Phase 24's verification and gate are still open (`docs/adr/0028`). Branch `phase-25-circuit-builder`, from `phase-24-simd-kernels` at `5d074e8`, so it merges after Phase 24. No code is written until the owner has approved the implementation plan.

Labels follow `docs/adr/0014`–`0028`: **[MEASURED]** (here: read from the source at `5d074e8`), **[DERIVED]**, **[MENOS]** (a proposed design decision) and **[OPEN]** (an owner decision, section 9).

---

## 1. Context

`docs/BUILD_PLAN.md`, Phase 25: let a user write a circuit by hand in C++ or Python without an external SDK. This means named gate methods, run-time parameters, composition, named registers and OpenQASM export. The builder adds no operation kind and no engine. Every circuit it produces is an ordinary KIR `Circuit`, planned by `plan_job` exactly as one built by `append`.

### 1.1 MENOS today (`5d074e8`) — [MEASURED] from the source

| Item | Where | State |
|---|---|---|
| KIR circuit | `include/menos/circuit.h` | `Circuit(num_qubits, num_clbits)`; `append(Opcode, qubits, params)`, `append_unitary`, `append_measure` (optional `ReadoutError`), `append_channel`, `append_batch`; `global_phase`. The counts are fixed at construction. Appending never throws; `ir::validate` runs once in `run`. |
| Opcodes | same | X Y Z H S Sdg T Tdg SX SXdg RX RY RZ P U CX CZ SWAP CP RZZ CCX, plus `Unitary` and `Measure`. Parameters are plain `double`s. |
| Python | `src/python/bindings.cpp` | `menos.Circuit` binds the methods above. Operations **cannot be read back** from Python (no `operations()` binding). |
| Exact inverse | `src/qasm3/modifiers.cpp` `inverse_of` | Already exact for every named gate (self-inverse; S↔Sdg, T↔Tdg, SX↔SXdg; negated angle for RX RY RZ P CP RZZ; `U(θ,φ,λ)† = U(−θ,−λ,−φ)`) and for `UnitaryOp` (conjugate transpose). It is internal to the OpenQASM 3 importer. |
| Plan cache | `src/plan/plan_cache.h` | Keyed by structure, not parameter values; a Direct in-memory plan is reparameterised from the new circuit. A rebinding loop therefore already reuses its plan, except near a special angle (`kSpecialAngleMargin`). |
| OpenQASM 3 import | `include/menos/qasm3.h` | `input` parameters bound by name; registers flattened in declaration order; `gphase`; `inv @` uses `inverse_of`. Built-in names map **directly** to opcodes for `p x y z h s sdg t tdg sx rx ry rz cx cz cp swap ccx` and `U`/`u3` → `U`. **`sxdg` and `rzz` have no direct name** (they are not in `stdgates.inc`); `inv @ sx` gives `SXdg`. A `UnitaryOp`, a `ChannelOp` and a readout error have **no OpenQASM 3 form**. Numbers are parsed with `std::stod`. |
| OpenQASM 2 import | `include/menos/qasm2.h` | `qelib1.inc` subset; no `input`, no global phase. |

### 1.2 What the gate needs that does not exist

- A way to read operations from Python, so that "each method appends exactly what `append` would" can be tested in both languages.
- An OpenQASM 3 spelling of `RZZ` that the importer reads back as `RZZ` (Q5).

## 2. Scope

### 2.1 In scope (subject to section 9)

1. Named gate methods for every named opcode, plus `unitary`, `measure`, `measure_all`, `channel` and `global_phase`, each appending exactly the KIR operation `append` would, returning the builder for chaining.
2. `Parameter` and parameter expressions (Q2); `bind` to an ordinary `Circuit`; an unbound run refused, naming the parameters.
3. `compose` with qubit and classical-bit maps, `inverse()`, `repeat(k)`.
4. Named quantum and classical registers.
5. OpenQASM 3 export, read back by the Phase 19 importer into identical operations.
6. A C++ implementation with a Python binding of it.

### 2.2 Refused by name

- `inverse()` of a circuit containing a measurement or a channel: `InvalidCircuit`, naming the operation and its index.
- Export of a `UnitaryOp`, a `ChannelOp` or a readout error (Q5): `UnsupportedFeature`, naming the operation and its index.
- Running or exporting with a missing binding (Q3): `InvalidCircuit`, naming every unbound parameter. A binding that names no parameter of the circuit is also refused, like the importer's unused `input`.
- A non-finite bound value: `InvalidCircuit`.

## 3. Design — [MENOS]

### 3.1 One new type; `Circuit` unchanged (Q1)

`menos::CircuitBuilder` (header `include/menos/builder.h`; Python `menos.CircuitBuilder`) holds registers, a list of builder operations and the global phase. A builder operation is a KIR operation whose angle slots may hold a `ParameterExpression` instead of a `double`. `bind(values)` returns a `Circuit` built with the existing `append*` calls. A builder with no parameters converts with `to_circuit()`. `Circuit`, its methods, `ir::validate` and every frontend are untouched. This keeps the KIR operation set unchanged (BUILD_PLAN invariant), and parameters never reach the planner.

Validation follows the KIR rule: a gate's operand count, duplicate qubits, matrix shapes and Kraus trace preservation are checked by `ir::validate` at `run`. The builder checks only what it alone knows, at the call: a register or bit name that does not exist, and a qubit or clbit index outside the declared registers (`InvalidCircuit`, naming it). A named method fixes its opcode's arity, so a wrong operand count cannot be written through it.

### 3.2 Parameters (Q2, Q8)

`Parameter{name}`: identity is the name, which must be an OpenQASM 3 identifier (`[A-Za-z_][A-Za-z0-9_]*`, not a keyword), so every parameter can be exported as an `input`. Under Q2 (a), a `ParameterExpression` is affine, `c0 + Σ cᵢ·θᵢ`, with terms kept in first-appearance order. It is built by `+ − *` with doubles and other expressions; a product of two parameters is refused. When bound, it is evaluated as `((c1·v1 + c2·v2) + …) + c0` in `double` with contraction off (the repository's flags), with c0 added only when the expression has a constant. *(Corrected at implementation, section 10.1: the order first proposed, `c0` first, would turn `θ` bound to −0.0 into +0.0.)* So `θ` alone binds to exactly `v`, −0.0 included, and `-θ` to exactly `-v`. That gives the gate's "bitwise the same circuit built by hand" for the expressions a user writes by hand with the same order.

### 3.3 Binding and running (Q3)

`bind(map<string,double>) -> Circuit`. `menos::run` and `menos.run` gain an overload taking a `CircuitBuilder`. It converts with `to_circuit()`, which refuses any free parameter by name, and then runs exactly as `run(Circuit)`. Under Q3 (a) there is no `values` argument to `run`.

### 3.4 Composition, inverse, repeat

- `compose(other, qubits = identity, clbits = identity)`: `other`'s operations are appended with operands remapped, and global phases added. Parameters merge by name (Q8). The maps must be injective and in range.
- `inverse()`: the operations in reverse order, each by the rule of `inverse_of`. That rule moves from `src/qasm3/modifiers.cpp` to `src/ir/inverse.{h,cpp}` and is used by both the importer and the builder; this is its second caller. For a parametric gate the expression is negated, which is exact. The global phase is negated, computed as `0.0 − phase` so that a zero phase stays +0.0 (section 10.1). A measurement or channel is refused (section 2.2).
- `repeat(k)`, k ≥ 0: k copies; `repeat(0)` is empty with the same registers.

The gate's "circuit then inverse is the identity within 1e-12, global phase included" is measured on the dense statevector.

### 3.5 Registers (Q7)

`add_qreg(name, size)` / `add_creg(name, size)`. Indices are assigned in declaration order, so the first register is qubits `0..n−1`, the next continues after it. This is the importer's flattening rule and Qiskit's. `CircuitBuilder(n, m)` declares registers `q[n]` and `c[m]`. Operands are an index, or a `(register, i)` bit reference (Python: `qr[i]`, where `qr` is the value `add_qreg` returns). `measure_all()` adds a classical register `meas` of `num_qubits` bits and measures qubit i into `meas[i]`, as Qiskit's `measure_all()` does. So fixtures compared against Qiskit through `MENOSBackend` give identical count keys.

### 3.6 OpenQASM 3 export (Q4, Q5)

`menos::qasm3::emit(const CircuitBuilder&) -> std::string` (Python `CircuitBuilder.to_qasm3()`). It writes `OPENQASM 3.0; include "stdgates.inc";`, an `input float[64] θ;` per free parameter, the registers as `qubit[n] name; bit[m] name;`, `gphase(x);` for a non-zero global phase, and each operation by the importer's direct name. `SXdg` is written `inv @ sx`, and `RZZ` as decided in Q5. Doubles are written with `std::to_chars` shortest round-trip form, which `std::stod` reads back exactly. Affine expressions are written in the same term order, which the importer evaluates as written. Export of a bound circuit (`Circuit`) is the same function applied to `CircuitBuilder::from_circuit(c)`.

### 3.7 Python

`menos.CircuitBuilder`, `menos.Parameter` and the register handles are nanobind bindings of the C++ types; no Python logic beyond argument conversion. `menos.Circuit.operations()` is added (read-only; tuples per operation) for the gate's per-method tests and for users. Method names are in Q7.

### 3.8 Unchanged

`Circuit` and its methods; the KIR operation set and the gate table; `plan_job`, `explain()` and results for any circuit built through `append`; the Qiskit, OpenQASM 2/3 (except the Q5 rule, if chosen) and PennyLane frontends; the Phase 15/16/21/22/23/24 invariants.

## 4. Testing — [MENOS]

- C++ (`tests/builder/`) and Python (`python/tests/test_builder.py`): one test per method, checking that the operation equals the `append` call's; registers and `measure_all`; every refusal of section 2.2 by name.
- Binding: results of `bind` are bitwise those of the hand-built circuit (counts and expectation, dense and stabilizer plans). A rebinding loop reports plan-cache hits (`plan_cache_stats`).
- Inverse: for every named gate at random angles, for random `UnitaryOp`s of 1–5 qubits and for random mixed circuits, `c + c.inverse()` acting on random input states is the identity within 1e-12, global phase included.
- Export: export then import gives identical operations and global phase on a fixture set, which covers every opcode, parameters, several registers and `gphase`.
- Qiskit agreement: fixtures built with the builder and with Qiskit 2.5.2, both run through `MENOSBackend`/`menos.run`, give identical counts under a fixed seed and expectation values within 1e-12.

## 5. Order of work

1. Owner answers to section 9; plan mode; owner approval.
2. Move `inverse_of` to `src/ir/inverse.{h,cpp}` (importer unchanged in behaviour; its tests pass unmodified).
3. C++ `CircuitBuilder`: registers, named methods, `to_circuit`, tests.
4. Parameters, `bind`, `run(CircuitBuilder)`, tests.
5. `compose`, `inverse`, `repeat`, tests.
6. OpenQASM 3 export (and the Q5 importer rule), round-trip tests.
7. Python bindings, `Circuit.operations()`, Python tests, Qiskit agreement fixtures.
8. Documentation (`docs/reference/python-api.md`, a builder reference page, `supported-gates.md`, changelog); `scripts/verify.sh` (WSL) and `scripts/verify-windows.ps1`, run one at a time, on the final commit.

## 6. Out of scope

A text drawing (unless Q6 says otherwise); non-affine parameter expressions; control flow, `reset` and mid-circuit classical logic (Phase 29, dynamic circuits); new opcodes or a gate-definition mechanism; OpenQASM 2 export (unless Q4 says otherwise); export of unitaries, channels and readout errors; changes to `MENOSBackend` or the PennyLane device.

## 7. Risks

- `std::stod` is locale-sensitive. Export with `to_chars` is locale-independent. If the import side is ever run under a comma-decimal locale, the round trip would break. The round-trip test runs under the default "C" locale, and the risk is recorded as an existing importer property, not changed here.
- Affine expressions add a small algebra. Its evaluation order is fixed and documented so bitwise results are reproducible. Q2 (b) removes the algebra if the owner prefers.
- The Q5 importer rule changes the importer's output for one exact definition. It is mathematically exact (section 9, Q5) but is a behaviour change to a frozen phase's component and so needs the owner's decision.

## 8. Derivation for Q5 — [DERIVED]

With `rz(θ) = diag(e^{−iθ/2}, e^{iθ/2})` (the gate table), `cx a,b; rz(θ) b; cx a,b;` maps |ab⟩ to phase `e^{∓iθ/2}` by the parity a⊕b: `diag(e^{−iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{−iθ/2})`. That is the gate table's `RZZ(θ) = exp(−iθ Z⊗Z/2)` exactly, with no global phase. So reading that exact definition as one `RZZ` is exact, not an approximation.

## 9. Owner decisions

- **Q1 — Where the methods live.**
  - (a, recommended) A new `CircuitBuilder` type (C++ and Python) with every builder feature. `bind`/`to_circuit` produce an ordinary `Circuit`, and `Circuit` is unchanged apart from Python's read-only `operations()`.
  - (b) Named methods added to `Circuit` itself for numeric angles, plus a separate parametric type. Two types then carry the same methods.
- **Q2 — Parameter expressions.**
  - (a, recommended) Affine: `c0 + Σ cᵢ·θᵢ`, with a fixed, documented evaluation order (section 3.2). This covers `2*θ`, `θ + π/2`, `−θ` (needed by `inverse()`), and shared parameters.
  - (b) Bare names, with only negation (enough for `inverse()`).
  - (c) General expressions (functions, products of parameters): not recommended; deferred.
- **Q3 — Binding.**
  - (a, recommended) `bind(values)` only, plus `run` accepting a builder that has no free parameters (refusing by name otherwise).
  - (b) Also a `values` argument to `run`/`run_batch`.
- **Q4 — Export format.**
  - (a, recommended) OpenQASM 3 only. OpenQASM 2 has no `input` parameters and no global phase, so a parametric or phased circuit could not round-trip.
  - (b) OpenQASM 3 and 2 (2 refusing parameters and a non-zero global phase by name).
- **Q5 — Exporting `RZZ`, `UnitaryOp`, channels and readout errors.**
  - (a, recommended) `RZZ` is written with the portable definition `gate rzz(θ) a, b { cx a, b; rz(θ) b; cx a, b; }`. The importer reads a definition that matches it token for token as `RZZ` (exact, section 8); any other `rzz` definition is expanded as today. `UnitaryOp`, `ChannelOp` and readout errors are refused by name.
  - (b) As (a), but without the importer rule: `RZZ` round-trips to `cx·rz·cx`. The round-trip criterion then compares that expansion for `RZZ`.
  - (c) Export of `RZZ` refused by name.
- **Q6 — Text drawing.**
  - (a, recommended) Out of scope; recorded as a deferral.
  - (b) In scope: a plain-text drawing, one row per qubit, tested against goldens.
- **Q7 — Names.**
  - (a, recommended) Qiskit's where one exists: gate methods named as the lowercase opcodes (`x` … `ccx`, `sxdg`, `rzz`, `u`, `p`); `measure`, `measure_all` (adds register `meas`, as Qiskit), `compose`, `inverse`, `repeat`, `add_register` split as `add_qreg`/`add_creg`; binding named `bind` (Qiskit's is `assign_parameters`). In C++, `CircuitBuilder::to_circuit` and `bind`.
  - (b) Qiskit's `assign_parameters` instead of `bind`, and a single `add_register`.
- **Q8 — Parameter identity.**
  - (a, recommended) By name: two `Parameter("θ")` are the same parameter, also across `compose`.
  - (b) By object identity, as Qiskit (a name clash in `compose` refused).
- **Q9 — The gate.** As in `docs/BUILD_PLAN.md`, Phase 25, refined as:
  1. Each method produces exactly the operation of the equivalent `append` call: one test per method, in C++ and Python.
  2. A bound circuit's results are bitwise those of the same circuit built by hand; a rebinding loop reuses its plan (plan-cache hits).
  3. Circuit then `inverse()` is the identity within 1e-12 on the dense statevector, global phase included, for every named gate, random `UnitaryOp`s and random mixed circuits.
  4. Export then import gives identical operations and global phase on the fixture set (with `RZZ` as Q5 decides).
  5. Every refusal of section 2.2, by name.
  6. Builder fixtures agree with the same circuits built in Qiskit and run through `MENOSBackend`: identical seeded counts, expectations within 1e-12.
  7. Existing suites unchanged; every new public symbol documented, with a production caller and tests; `scripts/verify.sh` and `scripts/verify-windows.ps1` exit 0 on the final commit.

### Owner answers (2026-11-11)

The owner took the recommended option, (a), for every question:

- **Q1:** a new `CircuitBuilder` type; `Circuit` is unchanged apart from Python's read-only `operations()`.
- **Q2:** affine parameter expressions with the fixed evaluation order of section 3.2.
- **Q3:** `bind(values)` only, plus `run` accepting a builder with no free parameters.
- **Q4:** OpenQASM 3 export only.
- **Q5:** `RZZ` exported with the portable definition, which the importer reads back as `RZZ` when it matches token for token; `UnitaryOp`, `ChannelOp` and readout errors refused by name.
- **Q6:** no text drawing; recorded as a deferral.
- **Q7:** Qiskit's names where they exist, `bind` for binding, `add_qreg`/`add_creg`.
- **Q8:** parameter identity by name.
- **Q9:** the gate as written in Q9.

## 10. Implementation record

### 10.1 As built, differing from section 3 (2026-11-11)

- **Evaluation order (section 3.2).** The constant is added last, `((c1·v1 + c2·v2) + …) + c0`, and only when the expression has one. The order first written, `c0` first, would bind a bare parameter set to −0.0 to +0.0, so `θ` would not bind to exactly its value.
- **`inverse()` keeps a zero global phase +0.0** (`0.0 − phase`, which equals `−phase` for every other value). A plain negation gave −0.0, which OpenQASM 3 cannot carry: the importer starts from 0.0 and adds each `gphase`, so −0.0 came back as +0.0 and broke the export round trip of an inverse.
- **Export limits, documented:** a global phase of −0.0 is written as none and so comes back as +0.0; an angle the importer's `std::stod` cannot read (a subnormal) is refused on import with the importer's own "out of range" error. Register and parameter names that are OpenQASM 3 keywords, built-in constants or functions, standard gate names or `rzz`, or that two registers or parameters share, are refused at export with `UnsupportedFeature` naming them. The builder itself accepts them, because they are valid builder names.
- **Bound expressions are written in evaluation order** as `term + term + (c0)`, each coefficient other than 1 as `(c)*name`. The importer evaluates `+` left to right, so it computes the same sequence of operations.
- **The canonical `rzz` rule** also requires `stdgates.inc` to be included when the definition is read. Its body calls the `stdgates.inc` `cx` and `rz`, and the include is what stops a user definition of either.
- **`compose` changes the builder in place and returns it**, unlike Qiskit's, which returns a new circuit by default; `inverse()` and `repeat()` return new builders, as Qiskit's do. Qiskit's argument names are used for the Python gate methods (for example `rz(phi, qubit)`, `cx(control_qubit, target_qubit)`).
- **The Qiskit agreement criterion (Q9 item 6)** is tested on statevectors, within 1e-12 of both `MENOSBackend` and Qiskit's own `Statevector`, global phase included. That bounds every expectation value. Seeded counts after `measure_all()` are identical.

### 10.2 Commits

- `eeb44f5`: the exact inverse moved to `src/ir/inverse.{h,cpp}`; the importer's `inv` modifier calls it.
- `f2e7635`: `CircuitBuilder`, `Parameter`, `ParameterExpression` (`include/menos/builder.h`, `src/ir/{builder,parameter}.cpp`, `src/ir/identifier.h`) and their unit tests.
- `65bb5cb`: `run(CircuitBuilder)` and `tests/api/builder_run_test.cpp` (bitwise bound runs, plan reuse, inverse identity).
- `4ab6910`: `qasm3::emit` (`src/qasm3/emitter.cpp`), the importer's canonical `rzz` rule and `tests/qasm3/emitter_test.cpp`.
- `932f68e`: the Python bindings, `Circuit.operations()`, `python/tests/test_builder.py`, `python/tests/test_builder_qiskit.py`, and the documentation (`docs/build/circuit-builder.md`, `python-api.md`, `native-api.md`, `limitations.md`, `changelog.md`).
- The following commit: clang-format, this record and the BUILD_PLAN status.

### 10.3 Gate evidence

Recorded when `scripts/verify.sh` and `scripts/verify-windows.ps1` have run on the final commit.
