#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include "common/error.h"
#include "menos/builder.h"
#include "same_circuit.h"

namespace menos {
namespace {

using Complex = std::complex<double>;
using test::same_circuit;
using test::same_double;

template <class F>
std::string message_of(F&& f) {
    try {
        f();
    } catch (const common::InvalidCircuit& e) {
        return e.what();
    }
    return "<no InvalidCircuit>";
}

// ---- named methods: one per method, against the equivalent append --------

struct NamedCase {
    const char* name;
    void (*build)(CircuitBuilder&);
    void (*append)(Circuit&);
};

const NamedCase kNamedCases[] = {
    {"x", [](CircuitBuilder& b) { b.x(1); }, [](Circuit& c) { c.append(Opcode::X, {1}); }},
    {"y", [](CircuitBuilder& b) { b.y(1); }, [](Circuit& c) { c.append(Opcode::Y, {1}); }},
    {"z", [](CircuitBuilder& b) { b.z(1); }, [](Circuit& c) { c.append(Opcode::Z, {1}); }},
    {"h", [](CircuitBuilder& b) { b.h(1); }, [](Circuit& c) { c.append(Opcode::H, {1}); }},
    {"s", [](CircuitBuilder& b) { b.s(1); }, [](Circuit& c) { c.append(Opcode::S, {1}); }},
    {"sdg", [](CircuitBuilder& b) { b.sdg(1); }, [](Circuit& c) { c.append(Opcode::Sdg, {1}); }},
    {"t", [](CircuitBuilder& b) { b.t(1); }, [](Circuit& c) { c.append(Opcode::T, {1}); }},
    {"tdg", [](CircuitBuilder& b) { b.tdg(1); }, [](Circuit& c) { c.append(Opcode::Tdg, {1}); }},
    {"sx", [](CircuitBuilder& b) { b.sx(1); }, [](Circuit& c) { c.append(Opcode::SX, {1}); }},
    {"sxdg",
     [](CircuitBuilder& b) { b.sxdg(1); },
     [](Circuit& c) { c.append(Opcode::SXdg, {1}); }},
    {"rx",
     [](CircuitBuilder& b) { b.rx(0.3, 1); },
     [](Circuit& c) { c.append(Opcode::RX, {1}, {0.3}); }},
    {"ry",
     [](CircuitBuilder& b) { b.ry(-0.4, 1); },
     [](Circuit& c) { c.append(Opcode::RY, {1}, {-0.4}); }},
    {"rz",
     [](CircuitBuilder& b) { b.rz(1.7, 1); },
     [](Circuit& c) { c.append(Opcode::RZ, {1}, {1.7}); }},
    {"p",
     [](CircuitBuilder& b) { b.p(0.9, 1); },
     [](Circuit& c) { c.append(Opcode::P, {1}, {0.9}); }},
    {"u",
     [](CircuitBuilder& b) { b.u(0.1, 0.2, 0.3, 1); },
     [](Circuit& c) { c.append(Opcode::U, {1}, {0.1, 0.2, 0.3}); }},
    {"cx",
     [](CircuitBuilder& b) { b.cx(2, 0); },
     [](Circuit& c) { c.append(Opcode::CX, {2, 0}); }},
    {"cz",
     [](CircuitBuilder& b) { b.cz(0, 2); },
     [](Circuit& c) { c.append(Opcode::CZ, {0, 2}); }},
    {"swap",
     [](CircuitBuilder& b) { b.swap(1, 2); },
     [](Circuit& c) { c.append(Opcode::SWAP, {1, 2}); }},
    {"cp",
     [](CircuitBuilder& b) { b.cp(0.5, 0, 1); },
     [](Circuit& c) { c.append(Opcode::CP, {0, 1}, {0.5}); }},
    {"rzz",
     [](CircuitBuilder& b) { b.rzz(-1.25, 2, 1); },
     [](Circuit& c) { c.append(Opcode::RZZ, {2, 1}, {-1.25}); }},
    {"ccx",
     [](CircuitBuilder& b) { b.ccx(0, 1, 2); },
     [](Circuit& c) { c.append(Opcode::CCX, {0, 1, 2}); }},
    {"unitary",
     [](CircuitBuilder& b) { b.unitary({0.0, 1.0, 1.0, 0.0}, {2}); },
     [](Circuit& c) { c.append_unitary({2}, {0.0, 1.0, 1.0, 0.0}); }},
    {"measure",
     [](CircuitBuilder& b) { b.measure(2, 1); },
     [](Circuit& c) { c.append_measure(2, 1); }},
    {"measure_readout",
     [](CircuitBuilder& b) { b.measure(0, 0, ReadoutError{{0.9, 0.1, 0.2, 0.8}}); },
     [](Circuit& c) { c.append_measure(0, 0, ReadoutError{{0.9, 0.1, 0.2, 0.8}}); }},
    {"channel",
     [](CircuitBuilder& b) {
         b.channel({{std::sqrt(0.5), 0.0, 0.0, std::sqrt(0.5)},
                    {std::sqrt(0.5), 0.0, 0.0, -std::sqrt(0.5)}},
                   {1});
     },
     [](Circuit& c) {
         c.append_channel({1},
                          {{std::sqrt(0.5), 0.0, 0.0, std::sqrt(0.5)},
                           {std::sqrt(0.5), 0.0, 0.0, -std::sqrt(0.5)}});
     }},
};

TEST(CircuitBuilderTest, EachNamedMethodAppendsExactlyTheAppendOperation) {
    for (const NamedCase& c : kNamedCases) {
        SCOPED_TRACE(c.name);
        CircuitBuilder builder(3, 2);
        c.build(builder);
        Circuit expected(3, 2);
        c.append(expected);
        EXPECT_TRUE(same_circuit(builder.to_circuit(), expected));
    }
}

TEST(CircuitBuilderTest, MethodsChainAndKeepOrder) {
    CircuitBuilder builder(2, 2);
    builder.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
    Circuit expected(2, 2);
    expected.append(Opcode::H, {0});
    expected.append(Opcode::CX, {0, 1});
    expected.append_measure(0, 0);
    expected.append_measure(1, 1);
    EXPECT_TRUE(same_circuit(builder.to_circuit(), expected));
}

TEST(CircuitBuilderTest, GlobalPhaseCarriesOver) {
    CircuitBuilder builder(1);
    builder.global_phase = 0.75;
    EXPECT_TRUE(same_double(builder.to_circuit().global_phase, 0.75));
}

TEST(CircuitBuilderTest, OutOfRangeQubitOrClbitIsRefusedAtTheCall) {
    CircuitBuilder builder(2, 1);
    EXPECT_NE(message_of([&] { builder.x(2); }).find("qubit 2"), std::string::npos);
    EXPECT_NE(message_of([&] { builder.measure(0, 1); }).find("classical bit 1"),
              std::string::npos);
    EXPECT_TRUE(builder.operations().empty());
}

TEST(CircuitBuilderTest, StructuralChecksAreLeftToValidate) {
    // A duplicate operand is ir::validate's to refuse, at run, as for Circuit.
    CircuitBuilder builder(2);
    EXPECT_NO_THROW(builder.cx(1, 1));
}

// ---- registers --------------------------------------------------------------

TEST(CircuitBuilderTest, RegistersAreNumberedInDeclarationOrder) {
    CircuitBuilder builder;
    const QuantumRegister a = builder.add_qreg("a", 2);
    const ClassicalRegister m = builder.add_creg("m", 1);
    const QuantumRegister b = builder.add_qreg("b", 3);
    EXPECT_EQ(a[0], 0U);
    EXPECT_EQ(a[1], 1U);
    EXPECT_EQ(b[0], 2U);
    EXPECT_EQ(b[2], 4U);
    EXPECT_EQ(m[0], 0U);
    EXPECT_EQ(builder.num_qubits(), 5U);
    EXPECT_EQ(builder.num_clbits(), 1U);
    EXPECT_EQ(builder.qreg("b").offset, 2U);
    builder.cx(builder.qreg("a")[1], builder.qreg("b")[2]).measure(b[2], m[0]);
    Circuit expected(5, 1);
    expected.append(Opcode::CX, {1, 4});
    expected.append_measure(4, 0);
    EXPECT_TRUE(same_circuit(builder.to_circuit(), expected));
}

TEST(CircuitBuilderTest, DefaultRegistersAreQAndC) {
    const CircuitBuilder builder(3, 2);
    ASSERT_EQ(builder.qregs().size(), 1U);
    EXPECT_EQ(builder.qregs()[0].name, "q");
    ASSERT_EQ(builder.cregs().size(), 1U);
    EXPECT_EQ(builder.cregs()[0].name, "c");
    EXPECT_TRUE(CircuitBuilder(0, 0).qregs().empty());
}

TEST(CircuitBuilderTest, RegisterRefusals) {
    CircuitBuilder builder(2, 1);
    EXPECT_NE(message_of([&] { builder.add_qreg("c", 1); }).find("already declared"),
              std::string::npos);
    EXPECT_NE(message_of([&] { builder.add_qreg("1x", 1); }).find("not an identifier"),
              std::string::npos);
    EXPECT_NE(message_of([&] { builder.add_creg("z", 0); }).find("at least one"),
              std::string::npos);
    EXPECT_NE(message_of([&] { (void)builder.qreg("nope"); }).find("'nope'"), std::string::npos);
    EXPECT_NE(message_of([&] { (void)builder.qreg("q")[2]; }).find("out of range"),
              std::string::npos);
}

TEST(CircuitBuilderTest, MeasureAllAddsAMeasRegisterAsQiskit) {
    CircuitBuilder builder(3, 1);
    builder.h(0).measure_all();
    EXPECT_EQ(builder.num_clbits(), 4U);
    EXPECT_EQ(builder.creg("meas").offset, 1U);
    Circuit expected(3, 4);
    expected.append(Opcode::H, {0});
    expected.append_measure(0, 1);
    expected.append_measure(1, 2);
    expected.append_measure(2, 3);
    EXPECT_TRUE(same_circuit(builder.to_circuit(), expected));
    EXPECT_NE(message_of([&] { builder.measure_all(); }).find("'meas'"), std::string::npos);
}

// ---- parameters and bind ---------------------------------------------------

TEST(CircuitBuilderTest, BindEvaluatesEveryAngle) {
    const Parameter a("a");
    const Parameter b("b");
    CircuitBuilder builder(2);
    builder.rx(a, 0).rzz(2.0 * a + b, 0, 1).u(b, -a, 0.5, 1).p(0.25, 0);
    EXPECT_EQ(builder.parameters(), (std::vector<std::string>{"a", "b"}));
    const double va = 0.3;
    const double vb = -1.1;
    Circuit expected(2, 0);
    expected.append(Opcode::RX, {0}, {va});
    expected.append(Opcode::RZZ, {0, 1}, {2.0 * va + 1.0 * vb});
    expected.append(Opcode::U, {1}, {vb, -va, 0.5});
    expected.append(Opcode::P, {0}, {0.25});
    EXPECT_TRUE(same_circuit(builder.bind({{"a", va}, {"b", vb}}), expected));
}

TEST(CircuitBuilderTest, BindRefusalsNameTheParameters) {
    CircuitBuilder builder(1);
    builder.rx(Parameter("alpha"), 0).ry(Parameter("beta"), 0);
    const std::string unbound = message_of([&] { (void)builder.to_circuit(); });
    EXPECT_NE(unbound.find("unbound parameters: 'alpha', 'beta'"), std::string::npos) << unbound;
    const std::string partly = message_of([&] { (void)builder.bind({{"alpha", 1.0}}); });
    EXPECT_NE(partly.find("'beta'"), std::string::npos);
    EXPECT_EQ(partly.find("'alpha'"), std::string::npos);
    const std::string unknown = message_of(
        [&] { (void)builder.bind({{"alpha", 1.0}, {"beta", 1.0}, {"gamma", 1.0}}); });
    EXPECT_NE(unknown.find("no parameter of the circuit: 'gamma'"), std::string::npos);
    const std::string inf = message_of([&] {
        (void)builder.bind({{"alpha", std::numeric_limits<double>::infinity()}, {"beta", 1.0}});
    });
    EXPECT_NE(inf.find("non-finite value: 'alpha'"), std::string::npos);
}

TEST(CircuitBuilderTest, FromCircuitRoundTrips) {
    Circuit circuit(2, 1);
    circuit.global_phase = -0.5;
    circuit.append(Opcode::U, {1}, {0.1, -0.0, 0.3});
    circuit.append_unitary({0}, {0.0, 1.0, 1.0, 0.0});
    circuit.append_measure(1, 0);
    EXPECT_TRUE(same_circuit(CircuitBuilder::from_circuit(circuit).to_circuit(), circuit));
}

// ---- compose, inverse, repeat ----------------------------------------------

TEST(CircuitBuilderTest, ComposeRemapsAndAddsPhase) {
    CircuitBuilder inner(2, 1);
    inner.global_phase = 0.25;
    inner.cx(0, 1).rz(Parameter("t"), 1).measure(1, 0);
    CircuitBuilder outer(3, 2);
    outer.global_phase = 0.5;
    outer.h(0).compose(inner, {2, 0}, {1}).rx(Parameter("t"), 1);
    Circuit expected(3, 2);
    expected.global_phase = 0.5 + 0.25;
    expected.append(Opcode::H, {0});
    expected.append(Opcode::CX, {2, 0});
    expected.append(Opcode::RZ, {0}, {0.7});
    expected.append_measure(0, 1);
    expected.append(Opcode::RX, {1}, {0.7});
    EXPECT_EQ(outer.parameters(), (std::vector<std::string>{"t"}));
    EXPECT_TRUE(same_circuit(outer.bind({{"t", 0.7}}), expected));
}

TEST(CircuitBuilderTest, ComposeIdentityAndRefusals) {
    CircuitBuilder small(1);
    small.x(0);
    CircuitBuilder big(2);
    big.compose(small);
    Circuit expected(2, 0);
    expected.append(Opcode::X, {0});
    EXPECT_TRUE(same_circuit(big.to_circuit(), expected));
    CircuitBuilder wide(3);
    EXPECT_NE(message_of([&] { small.compose(wide); }).find("more than"), std::string::npos);
    EXPECT_NE(message_of([&] { big.compose(big, {0, 0}); }).find("twice"), std::string::npos);
    EXPECT_NE(message_of([&] { big.compose(big, {0}); }).find("entries"), std::string::npos);
    EXPECT_NE(message_of([&] { big.compose(big, {0, 5}); }).find("outside"), std::string::npos);
}

TEST(CircuitBuilderTest, InverseUsesTheExactRule) {
    const Parameter a("a");
    CircuitBuilder builder(2);
    builder.global_phase = 0.3;
    builder.s(0).sx(1).rx(a, 0).u(0.1, 0.2, a + 0.3, 1).cp(-0.4, 0, 1).unitary(
        {Complex{0.0, 1.0}, 0.0, 0.0, Complex{0.0, -1.0}}, {1});
    const CircuitBuilder inv = builder.inverse();
    Circuit expected(2, 0);
    expected.global_phase = -0.3;
    // The conjugate transpose: std::conj also flips the sign of a zero
    // imaginary part, which the bitwise comparison sees.
    expected.append_unitary(
        {1}, {std::conj(Complex{0.0, 1.0}), std::conj(Complex{0.0}), std::conj(Complex{0.0}),
              std::conj(Complex{0.0, -1.0})});
    expected.append(Opcode::CP, {0, 1}, {0.4});
    expected.append(Opcode::U, {1}, {-0.1, -(0.5 + 0.3), -0.2});
    expected.append(Opcode::RX, {0}, {-0.5});
    expected.append(Opcode::SXdg, {1});
    expected.append(Opcode::Sdg, {0});
    EXPECT_TRUE(same_circuit(inv.bind({{"a", 0.5}}), expected));
}

TEST(CircuitBuilderTest, InverseRefusesMeasurementAndChannelByIndex) {
    CircuitBuilder measured(1, 1);
    measured.h(0).measure(0, 0);
    const std::string m = message_of([&] { (void)measured.inverse(); });
    EXPECT_NE(m.find("operation 1 is a measurement"), std::string::npos) << m;
    CircuitBuilder noisy(1);
    noisy.channel({{1.0, 0.0, 0.0, 1.0}}, {0}).x(0);
    const std::string c = message_of([&] { (void)noisy.inverse(); });
    EXPECT_NE(c.find("operation 0 is a channel"), std::string::npos) << c;
}

TEST(CircuitBuilderTest, RepeatCopiesOperationsAndPhase) {
    CircuitBuilder builder(1);
    builder.global_phase = 0.1;
    builder.h(0).t(0);
    const CircuitBuilder three = builder.repeat(3);
    Circuit expected(1, 0);
    expected.global_phase = (0.1 + 0.1) + 0.1;
    for (int i = 0; i < 3; ++i) {
        expected.append(Opcode::H, {0});
        expected.append(Opcode::T, {0});
    }
    EXPECT_TRUE(same_circuit(three.to_circuit(), expected));
    EXPECT_TRUE(builder.repeat(0).operations().empty());
    EXPECT_EQ(builder.repeat(0).global_phase, 0.0);
}

}  // namespace
}  // namespace menos
