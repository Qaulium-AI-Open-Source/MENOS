// OpenQASM 3 export (docs/adr/0029 section 3.6, Phase 25 gate criteria 4 and
// 5): export then import gives the identical operations and global phase,
// and what OpenQASM 3 cannot express is refused by name.

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "../ir/same_circuit.h"
#include "common/error.h"
#include "menos/builder.h"
#include "menos/qasm3.h"

namespace menos {
namespace {

using test::same_circuit;

// Every named gate, several registers, a global phase and measurements.
CircuitBuilder every_gate() {
    CircuitBuilder b;
    const QuantumRegister a = b.add_qreg("a", 2);
    const QuantumRegister r = b.add_qreg("r", 2);
    const ClassicalRegister m = b.add_creg("m", 3);
    const ClassicalRegister k = b.add_creg("k", 1);
    b.global_phase = -0.3125;
    b.x(a[0]).y(a[1]).z(r[0]).h(r[1]).s(a[0]).sdg(a[1]).t(r[0]).tdg(r[1]);
    b.sx(a[0]).sxdg(a[1]).rx(0.1, r[0]).ry(-2.5e-7, r[1]).rz(1e21, a[0]).p(-0.0, a[1]);
    b.u(1.0 / 3.0, -std::sqrt(2.0), 3.0, r[0]).cx(a[0], r[1]).cz(r[1], a[1]);
    b.swap(a[0], r[0]).cp(0.75, r[0], a[1]).rzz(-1.125, a[1], r[1]).ccx(r[1], a[0], r[0]);
    b.measure(a[0], m[2]).measure(r[1], k[0]).measure(a[1], m[0]);
    return b;
}

TEST(EmitterTest, EveryGateRoundTripsBitwise) {
    const CircuitBuilder b = every_gate();
    const std::string text = qasm3::emit(b);
    EXPECT_TRUE(same_circuit(qasm3::parse(text), b.to_circuit())) << text;
}

TEST(EmitterTest, ParametersRoundTripAsInputs) {
    const Parameter theta("theta");
    const Parameter phi("phi");
    CircuitBuilder b(3, 1);
    b.rx(theta, 0)
        .ry(-theta, 1)
        .rz(2.0 * theta - 0.5 * phi + 0.25, 2)
        .rzz(phi, 0, 2)
        .u(theta, phi + theta, -0.0 - phi, 1)
        .cp(3.0 * phi, 2, 0)
        .measure(1, 0);
    const std::string text = qasm3::emit(b);
    EXPECT_NE(text.find("input float[64] phi;"), std::string::npos) << text;
    EXPECT_NE(text.find("input float[64] theta;"), std::string::npos) << text;
    for (const auto& values : std::vector<std::map<std::string, double, std::less<>>>{
             {{"theta", 0.1}, {"phi", -0.7}},
             {{"theta", -0.0}, {"phi", 1e-300}},
             {{"theta", 3.0}, {"phi", 0.0}}}) {
        qasm3::InputBindings inputs;
        for (const auto& [name, value] : values) {
            inputs.emplace(name, value);
        }
        EXPECT_TRUE(same_circuit(qasm3::parse(text, inputs), b.bind(values))) << text;
    }
}

TEST(EmitterTest, CircuitOverloadUsesRegistersQAndC) {
    Circuit c(2, 2);
    c.append(Opcode::H, {0});
    c.append(Opcode::CX, {0, 1});
    c.append_measure(0, 0);
    c.append_measure(1, 1);
    const std::string text = qasm3::emit(c);
    EXPECT_EQ(text,
              "OPENQASM 3.0;\n"
              "include \"stdgates.inc\";\n"
              "qubit[2] q;\n"
              "bit[2] c;\n"
              "h q[0];\n"
              "cx q[0], q[1];\n"
              "c[0] = measure q[0];\n"
              "c[1] = measure q[1];\n");
    EXPECT_TRUE(same_circuit(qasm3::parse(text), c));
}

TEST(EmitterTest, InverseOfABuilderRoundTrips) {
    CircuitBuilder b(2);
    b.sx(0).rzz(0.5, 0, 1).u(0.1, 0.2, 0.3, 1).t(0);
    const CircuitBuilder inv = b.inverse();
    EXPECT_TRUE(same_circuit(qasm3::parse(qasm3::emit(inv)), inv.to_circuit()));
}

template <class E, class F>
std::string message_of(F&& f) {
    try {
        f();
    } catch (const E& e) {
        return e.what();
    }
    return "<not thrown>";
}

TEST(EmitterTest, InexpressibleOperationsAreRefusedByIndex) {
    CircuitBuilder u(1);
    u.h(0).unitary({0.0, 1.0, 1.0, 0.0}, {0});
    EXPECT_NE(message_of<common::UnsupportedFeature>([&] { (void)qasm3::emit(u); })
                  .find("operation 1 is an explicit unitary"),
              std::string::npos);
    CircuitBuilder ch(1);
    ch.channel({{1.0, 0.0, 0.0, 1.0}}, {0});
    EXPECT_NE(message_of<common::UnsupportedFeature>([&] { (void)qasm3::emit(ch); })
                  .find("operation 0 is a noise channel"),
              std::string::npos);
    CircuitBuilder ro(1, 1);
    ro.measure(0, 0, ReadoutError{{0.9, 0.1, 0.1, 0.9}});
    EXPECT_NE(message_of<common::UnsupportedFeature>([&] { (void)qasm3::emit(ro); })
                  .find("operation 0 is a measurement with a readout error"),
              std::string::npos);
}

TEST(EmitterTest, ReservedOrSharedNamesAreRefused) {
    CircuitBuilder reg;
    reg.add_qreg("cx", 1);
    EXPECT_NE(message_of<common::UnsupportedFeature>([&] { (void)qasm3::emit(reg); })
                  .find("register name 'cx'"),
              std::string::npos);
    CircuitBuilder par(1);
    par.rx(Parameter("pi"), 0);
    EXPECT_NE(message_of<common::UnsupportedFeature>([&] { (void)qasm3::emit(par); })
                  .find("parameter name 'pi'"),
              std::string::npos);
    CircuitBuilder clash(1);
    clash.rx(Parameter("q"), 0);
    EXPECT_NE(message_of<common::UnsupportedFeature>([&] { (void)qasm3::emit(clash); })
                  .find("'q' is used by more than one"),
              std::string::npos);
}

TEST(EmitterTest, NonFiniteAngleIsRefused) {
    CircuitBuilder b(1);
    b.rx(std::nan(""), 0);
    EXPECT_NE(message_of<common::InvalidCircuit>([&] { (void)qasm3::emit(b); })
                  .find("operation 0 is not finite"),
              std::string::npos);
}

// ---- the importer's canonical rzz rule (docs/adr/0029 section 8) -----------

std::string with_rzz_body(const std::string& body) {
    return "OPENQASM 3.0;\ninclude \"stdgates.inc\";\ngate rzz(t) a, b { " + body +
           " }\nqubit[2] q;\nrzz(0.5) q[0], q[1];\n";
}

TEST(EmitterTest, CanonicalRzzDefinitionImportsAsOneRzz) {
    const Circuit c = qasm3::parse(with_rzz_body("cx a, b; rz(t) b; cx a, b;"));
    Circuit expected(2, 0);
    expected.append(Opcode::RZZ, {0, 1}, {0.5});
    EXPECT_TRUE(same_circuit(c, expected));
}

TEST(EmitterTest, InverseOfCanonicalRzzIsRzzOfTheNegatedAngle) {
    const Circuit c = qasm3::parse(
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n"
        "gate rzz(t) a, b { cx a, b; rz(t) b; cx a, b; }\n"
        "qubit[2] q;\ninv @ rzz(0.5) q[1], q[0];\n");
    Circuit expected(2, 0);
    expected.append(Opcode::RZZ, {1, 0}, {-0.5});
    EXPECT_TRUE(same_circuit(c, expected));
}

TEST(EmitterTest, AnyOtherRzzBodyIsExpandedAsBefore) {
    // Same operations in another form: not the canonical text, so expanded.
    const Circuit c = qasm3::parse(with_rzz_body("cx a, b; rz(t) b; cx a, b; id a;"));
    EXPECT_EQ(c.operations().size(), 3U);
    const Circuit swapped = qasm3::parse(with_rzz_body("cx b, a; rz(t) a; cx b, a;"));
    ASSERT_EQ(swapped.operations().size(), 3U);
    EXPECT_EQ(std::get<GateOp>(swapped.operations()[0]).opcode, Opcode::CX);
}

}  // namespace
}  // namespace menos
