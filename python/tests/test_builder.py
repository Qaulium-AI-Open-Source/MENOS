"""The circuit builder from Python (docs/adr/0029, Phase 25).

Gate criteria 1, 2, 4 and 5 in Python: each method appends exactly what
Circuit.append would; a bound builder runs bitwise as the hand-built circuit;
export then import gives identical operations; refusals name what they refuse.
"""

import math

import numpy as np
import pytest

import menos
from menos import _core

# (method call on a builder, equivalent append call on a Circuit)
NAMED = [
    ("x", lambda b: b.x(1), lambda c: c.append(menos.Opcode.X, [1])),
    ("y", lambda b: b.y(1), lambda c: c.append(menos.Opcode.Y, [1])),
    ("z", lambda b: b.z(1), lambda c: c.append(menos.Opcode.Z, [1])),
    ("h", lambda b: b.h(1), lambda c: c.append(menos.Opcode.H, [1])),
    ("s", lambda b: b.s(1), lambda c: c.append(menos.Opcode.S, [1])),
    ("sdg", lambda b: b.sdg(1), lambda c: c.append(menos.Opcode.Sdg, [1])),
    ("t", lambda b: b.t(1), lambda c: c.append(menos.Opcode.T, [1])),
    ("tdg", lambda b: b.tdg(1), lambda c: c.append(menos.Opcode.Tdg, [1])),
    ("sx", lambda b: b.sx(1), lambda c: c.append(menos.Opcode.SX, [1])),
    ("sxdg", lambda b: b.sxdg(1), lambda c: c.append(menos.Opcode.SXdg, [1])),
    ("rx", lambda b: b.rx(0.3, 1), lambda c: c.append(menos.Opcode.RX, [1], [0.3])),
    ("ry", lambda b: b.ry(-0.4, 1), lambda c: c.append(menos.Opcode.RY, [1], [-0.4])),
    ("rz", lambda b: b.rz(1.7, 1), lambda c: c.append(menos.Opcode.RZ, [1], [1.7])),
    ("p", lambda b: b.p(0.9, 1), lambda c: c.append(menos.Opcode.P, [1], [0.9])),
    (
        "u",
        lambda b: b.u(0.1, 0.2, 0.3, 1),
        lambda c: c.append(menos.Opcode.U, [1], [0.1, 0.2, 0.3]),
    ),
    ("cx", lambda b: b.cx(2, 0), lambda c: c.append(menos.Opcode.CX, [2, 0])),
    ("cz", lambda b: b.cz(0, 2), lambda c: c.append(menos.Opcode.CZ, [0, 2])),
    ("swap", lambda b: b.swap(1, 2), lambda c: c.append(menos.Opcode.SWAP, [1, 2])),
    ("cp", lambda b: b.cp(0.5, 0, 1), lambda c: c.append(menos.Opcode.CP, [0, 1], [0.5])),
    ("rzz", lambda b: b.rzz(-1.25, 2, 1), lambda c: c.append(menos.Opcode.RZZ, [2, 1], [-1.25])),
    ("ccx", lambda b: b.ccx(0, 1, 2), lambda c: c.append(menos.Opcode.CCX, [0, 1, 2])),
    (
        "unitary",
        lambda b: b.unitary([0, 1, 1, 0], [2]),
        lambda c: c.append_unitary([2], [0, 1, 1, 0]),
    ),
    ("measure", lambda b: b.measure(2, 1), lambda c: c.append_measure(2, 1)),
    (
        "measure_readout",
        lambda b: b.measure(0, 0, [0.9, 0.1, 0.2, 0.8]),
        lambda c: c.append_measure(0, 0, [0.9, 0.1, 0.2, 0.8]),
    ),
    (
        "channel",
        lambda b: b.channel([[math.sqrt(0.5), 0, 0, math.sqrt(0.5)], [math.sqrt(0.5), 0, 0, -math.sqrt(0.5)]], [1]),
        lambda c: c.append_channel([1], [[math.sqrt(0.5), 0, 0, math.sqrt(0.5)], [math.sqrt(0.5), 0, 0, -math.sqrt(0.5)]]),
    ),
]


def _options(**fields):
    options = menos.RunOptions()
    for name, value in fields.items():
        setattr(options, name, value)
    return options


def _bits(values):
    """Floats (or complexes) as their IEEE-754 bit patterns, for bitwise equality."""
    return np.asarray(values, dtype=np.complex128).view(np.uint64).tolist()


def _same_operations(a, b):
    ops_a, ops_b = a.operations(), b.operations()
    assert len(ops_a) == len(ops_b)
    for x, y in zip(ops_a, ops_b):
        assert x[0] == y[0]
        if x[0] == "gate":
            assert (x[1], list(x[2])) == (y[1], list(y[2]))
            assert _bits(x[3]) == _bits(y[3])
        elif x[0] in ("unitary",):
            assert list(x[1]) == list(y[1]) and _bits(x[2]) == _bits(y[2])
        elif x[0] == "channel":
            assert list(x[1]) == list(y[1])
            assert [_bits(k) for k in x[2]] == [_bits(k) for k in y[2]]
        else:
            assert x == y
    assert _bits([a.global_phase]) == _bits([b.global_phase])
    assert (a.num_qubits(), a.num_clbits()) == (b.num_qubits(), b.num_clbits())


@pytest.mark.parametrize("name,build,append", NAMED, ids=[n[0] for n in NAMED])
def test_each_method_appends_exactly_the_append_operation(name, build, append):
    builder = menos.CircuitBuilder(3, 2)
    returned = build(builder)
    assert returned is builder  # chaining returns the builder itself
    circuit = menos.Circuit(3, 2)
    append(circuit)
    _same_operations(builder.to_circuit(), circuit)


def test_circuit_operations_reads_back_every_kind():
    c = menos.Circuit(2, 1)
    c.append(menos.Opcode.RX, [0], [0.5])
    c.append_unitary([1], [0, 1, 1, 0])
    c.append_measure(1, 0)
    c.append_channel([0], [[1, 0, 0, 1]])
    kinds = [op[0] for op in c.operations()]
    assert kinds == ["gate", "unitary", "measure", "channel"]
    assert c.operations()[0][1] == menos.Opcode.RX
    assert c.operations()[2] == ("measure", 1, 0, None)


def test_registers_and_measure_all():
    b = menos.CircuitBuilder()
    a = b.add_qreg("a", 2)
    m = b.add_creg("m", 1)
    r = b.add_qreg("r", 3)
    assert (a[1], r[0], r[2], m[0]) == (1, 2, 4, 0)
    assert len(r) == 3 and b.qreg("r").offset == 2
    b.cx(a[1], r[2]).measure_all()
    assert b.num_clbits() == 1 + 5
    assert b.creg("meas").offset == 1
    expected = menos.Circuit(5, 6)
    expected.append(menos.Opcode.CX, [1, 4])
    for q in range(5):
        expected.append_measure(q, 1 + q)
    _same_operations(b.to_circuit(), expected)
    with pytest.raises(menos.InvalidCircuitError, match="out of range"):
        r[3]
    with pytest.raises(menos.InvalidCircuitError, match="already declared"):
        b.add_qreg("m", 1)


def test_parameter_arithmetic_and_bind():
    theta = menos.Parameter("theta")
    phi = menos.Parameter("phi")
    b = menos.CircuitBuilder(2)
    b.rx(theta, 0).ry(-theta, 1).rz(2 * theta - 0.5 * phi + 0.25, 0).cp(theta * 3, 0, 1)
    assert b.parameters() == ["phi", "theta"]
    expr = b.operations()[2][3][0]
    assert isinstance(expr, menos.ParameterExpression)
    assert expr.terms == [("theta", 2.0), ("phi", -0.5)]
    assert expr.constant == 0.25
    t, f = 0.123, -2.5
    expected = menos.Circuit(2, 0)
    expected.append(menos.Opcode.RX, [0], [t])
    expected.append(menos.Opcode.RY, [1], [-t])
    expected.append(menos.Opcode.RZ, [0], [(2.0 * t + -0.5 * f) + 0.25])
    expected.append(menos.Opcode.CP, [0, 1], [3.0 * t])
    _same_operations(b.bind({"theta": t, phi: f}), expected)
    with pytest.raises(menos.InvalidCircuitError, match="not affine"):
        theta * phi
    with pytest.raises(menos.InvalidCircuitError, match="not an identifier"):
        menos.Parameter("1x")


def test_bind_and_run_refusals_name_the_parameters():
    b = menos.CircuitBuilder(1)
    b.rx(menos.Parameter("alpha"), 0).ry(menos.Parameter("beta"), 0)
    with pytest.raises(menos.InvalidCircuitError, match="unbound parameters: 'alpha', 'beta'"):
        b.to_circuit()
    with pytest.raises(menos.InvalidCircuitError, match="'beta'"):
        menos.run(b, _options(shots=0, want_statevector=True))
    with pytest.raises(menos.InvalidCircuitError, match="no parameter of the circuit: 'gamma'"):
        b.bind({"alpha": 1.0, "beta": 1.0, "gamma": 2.0})
    with pytest.raises(menos.InvalidCircuitError, match="non-finite value: 'alpha'"):
        b.bind({"alpha": math.inf, "beta": 1.0})


def _ansatz(n):
    theta = menos.Parameter("theta")
    phi = menos.Parameter("phi")
    b = menos.CircuitBuilder(n, n)
    for q in range(n):
        b.h(q).ry(theta, q)
    for q in range(n - 1):
        b.rzz(2.0 * theta - phi, q, q + 1)
    for q in range(n):
        b.u(phi, -theta, 0.5 + phi, q)
    return b


def _ansatz_by_hand(n, t, f):
    c = menos.Circuit(n, n)
    for q in range(n):
        c.append(menos.Opcode.H, [q])
        c.append(menos.Opcode.RY, [q], [t])
    for q in range(n - 1):
        c.append(menos.Opcode.RZZ, [q, q + 1], [2.0 * t + -1.0 * f])
    for q in range(n):
        c.append(menos.Opcode.U, [q], [f, -t, 1.0 * f + 0.5])
    return c


def test_bound_builder_runs_bitwise_as_the_hand_built_circuit():
    n, t, f = 5, 0.4321, -1.234
    bound = _ansatz(n).bind({"theta": t, "phi": f})
    hand = _ansatz_by_hand(n, t, f)
    opts = _options(shots=0, want_statevector=True, seed=7)
    a = menos.run(bound, opts).statevector
    h = menos.run(hand, opts).statevector
    assert a.view(np.uint64).tolist() == h.view(np.uint64).tolist()


def test_parameter_free_builder_runs_directly():
    b = menos.CircuitBuilder(2)
    b.h(0).cx(0, 1).measure_all()
    counts = menos.run(b, _options(shots=1000, seed=5)).counts
    assert set(counts) <= {"0x0", "0x3"}
    assert sum(counts.values()) == 1000


def test_rebinding_reuses_the_plan():
    n = 6
    builder = _ansatz(n)
    _core.plan_cache_clear()
    before = _core.plan_cache_stats()
    opts = _options(shots=0, want_statevector=True, seed=7)
    for step in range(4):
        menos.run(builder.bind({"theta": 0.1 + 0.2 * step, "phi": 0.3 - 0.05 * step}), opts)
    after = _core.plan_cache_stats()
    assert after["misses"] - before["misses"] == 1
    assert after["hits"] - before["hits"] == 3


def test_compose_inverse_repeat():
    inner = menos.CircuitBuilder(2)
    inner.cx(0, 1).rz(menos.Parameter("t"), 1)
    outer = menos.CircuitBuilder(3)
    outer.h(0).compose(inner, [2, 0])
    expected = menos.Circuit(3, 0)
    expected.append(menos.Opcode.H, [0])
    expected.append(menos.Opcode.CX, [2, 0])
    expected.append(menos.Opcode.RZ, [0], [0.7])
    _same_operations(outer.bind({"t": 0.7}), expected)

    inv = outer.inverse().bind({"t": 0.7})
    assert [op[1] for op in inv.operations()] == [menos.Opcode.RZ, menos.Opcode.CX, menos.Opcode.H]
    assert inv.operations()[0][3] == [-0.7]
    assert len(outer.repeat(3).operations()) == 9

    measured = menos.CircuitBuilder(1, 1)
    measured.h(0).measure(0, 0)
    with pytest.raises(menos.InvalidCircuitError, match="operation 1 is a measurement"):
        measured.inverse()
    noisy = menos.CircuitBuilder(1)
    noisy.channel([[1, 0, 0, 1]], [0])
    with pytest.raises(menos.InvalidCircuitError, match="operation 0 is a channel"):
        noisy.inverse()


def test_qasm3_export_round_trips():
    theta = menos.Parameter("theta")
    b = menos.CircuitBuilder()
    q = b.add_qreg("q", 3)
    c = b.add_creg("c", 2)
    b.global_phase = 0.25
    b.h(q[0]).sxdg(q[1]).rzz(theta, q[0], q[2]).u(theta, 0.5 - theta, 1.0, q[1])
    b.measure(q[2], c[1])
    text = b.to_qasm3()
    assert "input float[64] theta;" in text
    for value in (0.3, -0.0, 2.0):
        _same_operations(menos.from_qasm3(text, {"theta": value}), b.bind({"theta": value}))


def test_qasm3_export_refusals():
    b = menos.CircuitBuilder(1)
    b.unitary([0, 1, 1, 0], [0])
    with pytest.raises(menos.UnsupportedFeatureError, match="operation 0 is an explicit unitary"):
        b.to_qasm3()
    r = menos.CircuitBuilder(1, 1)
    r.measure(0, 0, [0.9, 0.1, 0.1, 0.9])
    with pytest.raises(menos.UnsupportedFeatureError, match="readout error"):
        r.to_qasm3()
