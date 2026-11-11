// The circuit builder end to end (docs/adr/0029, Phase 25 gate criteria 2 and
// 3): a bound builder runs bit for bit as the same circuit built by hand, a
// rebinding loop reuses its plan, and a circuit followed by its inverse is
// the identity on the dense statevector, global phase included.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "common/error.h"
#include "menos/menos.h"

namespace menos {
namespace {

using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;

bool same_bits(const Complex& a, const Complex& b) {
    return std::bit_cast<std::uint64_t>(a.real()) == std::bit_cast<std::uint64_t>(b.real()) &&
           std::bit_cast<std::uint64_t>(a.imag()) == std::bit_cast<std::uint64_t>(b.imag());
}

// A Haar-random unitary of dimension dim: Gram-Schmidt on complex Gaussian
// columns (row-major result).
Matrix random_unitary(std::size_t dim, std::mt19937_64& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Matrix> cols(dim, Matrix(dim));
    for (std::size_t c = 0; c < dim; ++c) {
        for (std::size_t r = 0; r < dim; ++r) {
            cols[c][r] = Complex{normal(rng), normal(rng)};
        }
        for (std::size_t k = 0; k < c; ++k) {
            Complex dot{0.0, 0.0};
            for (std::size_t r = 0; r < dim; ++r) {
                dot += std::conj(cols[k][r]) * cols[c][r];
            }
            for (std::size_t r = 0; r < dim; ++r) {
                cols[c][r] -= dot * cols[k][r];
            }
        }
        double norm = 0.0;
        for (std::size_t r = 0; r < dim; ++r) {
            norm += std::norm(cols[c][r]);
        }
        norm = std::sqrt(norm);
        for (std::size_t r = 0; r < dim; ++r) {
            cols[c][r] /= norm;
        }
    }
    Matrix out(dim * dim);
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            out[r * dim + c] = cols[c][r];
        }
    }
    return out;
}

std::vector<QubitId> all_qubits(std::uint32_t n) {
    std::vector<QubitId> qubits(n);
    for (QubitId q = 0; q < n; ++q) {
        qubits[q] = q;
    }
    return qubits;
}

std::vector<Complex> statevector_of(const Circuit& circuit) {
    const Result result = run(circuit, RunOptions{.shots = 0, .want_statevector = true, .seed = 1});
    return *result.statevector();
}

// prep, then body, then body.inverse() leaves exactly prep's state (within
// 1e-12 per amplitude, global phase included).
void expect_inverse_is_identity(const CircuitBuilder& body, std::mt19937_64& rng) {
    const std::uint32_t n = body.num_qubits();
    CircuitBuilder prep(n);
    prep.unitary(random_unitary(std::size_t{1} << n, rng), all_qubits(n));
    CircuitBuilder full = prep;
    full.compose(body).compose(body.inverse());
    const std::vector<Complex> expected = statevector_of(prep.to_circuit());
    const std::vector<Complex> actual = statevector_of(full.to_circuit());
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_LE(std::abs(actual[i] - expected[i]), 1e-12) << "amplitude " << i;
    }
}

double angle(std::mt19937_64& rng) {
    return std::uniform_real_distribution<double>(-std::numbers::pi, std::numbers::pi)(rng);
}

// Every named gate on n = 3 qubits, at random angles, with a non-zero global
// phase: one body per gate.
std::vector<CircuitBuilder> one_gate_bodies(std::mt19937_64& rng) {
    std::vector<CircuitBuilder> bodies;
    const auto add = [&](auto&& f) {
        CircuitBuilder b(3);
        b.global_phase = angle(rng);
        f(b);
        bodies.push_back(b);
    };
    add([](CircuitBuilder& b) { b.x(0); });
    add([](CircuitBuilder& b) { b.y(1); });
    add([](CircuitBuilder& b) { b.z(2); });
    add([](CircuitBuilder& b) { b.h(0); });
    add([](CircuitBuilder& b) { b.s(1); });
    add([](CircuitBuilder& b) { b.sdg(2); });
    add([](CircuitBuilder& b) { b.t(0); });
    add([](CircuitBuilder& b) { b.tdg(1); });
    add([](CircuitBuilder& b) { b.sx(2); });
    add([](CircuitBuilder& b) { b.sxdg(0); });
    add([&](CircuitBuilder& b) { b.rx(angle(rng), 1); });
    add([&](CircuitBuilder& b) { b.ry(angle(rng), 2); });
    add([&](CircuitBuilder& b) { b.rz(angle(rng), 0); });
    add([&](CircuitBuilder& b) { b.p(angle(rng), 1); });
    add([&](CircuitBuilder& b) { b.u(angle(rng), angle(rng), angle(rng), 2); });
    add([](CircuitBuilder& b) { b.cx(2, 0); });
    add([](CircuitBuilder& b) { b.cz(0, 1); });
    add([](CircuitBuilder& b) { b.swap(1, 2); });
    add([&](CircuitBuilder& b) { b.cp(angle(rng), 0, 2); });
    add([&](CircuitBuilder& b) { b.rzz(angle(rng), 2, 1); });
    add([](CircuitBuilder& b) { b.ccx(1, 2, 0); });
    return bodies;
}

TEST(BuilderRunTest, InverseIsIdentityForEveryNamedGate) {
    std::mt19937_64 rng(25);
    for (const CircuitBuilder& body : one_gate_bodies(rng)) {
        expect_inverse_is_identity(body, rng);
    }
}

TEST(BuilderRunTest, InverseIsIdentityForRandomUnitaries) {
    std::mt19937_64 rng(2501);
    for (std::uint32_t k = 1; k <= kMaxUnitaryQubits; ++k) {
        SCOPED_TRACE(k);
        CircuitBuilder body(5);
        body.global_phase = angle(rng);
        std::vector<QubitId> qubits = all_qubits(5);
        std::shuffle(qubits.begin(), qubits.end(), rng);
        qubits.resize(k);
        body.unitary(random_unitary(std::size_t{1} << k, rng), qubits);
        expect_inverse_is_identity(body, rng);
    }
}

TEST(BuilderRunTest, InverseIsIdentityForRandomMixedCircuits) {
    std::mt19937_64 rng(2502);
    for (int trial = 0; trial < 10; ++trial) {
        CircuitBuilder body(4);
        body.global_phase = angle(rng);
        std::uniform_int_distribution<int> pick(0, 6);
        std::uniform_int_distribution<QubitId> qubit(0, 3);
        for (int g = 0; g < 30; ++g) {
            const QubitId a = qubit(rng);
            const QubitId b = (a + 1 + qubit(rng) % 3) % 4;
            switch (pick(rng)) {
                case 0:
                    body.h(a).t(b);
                    break;
                case 1:
                    body.u(angle(rng), angle(rng), angle(rng), a);
                    break;
                case 2:
                    body.cx(a, b).sxdg(b);
                    break;
                case 3:
                    body.rzz(angle(rng), a, b);
                    break;
                case 4:
                    body.cp(angle(rng), a, b).s(a);
                    break;
                case 5:
                    body.unitary(random_unitary(4, rng), {a, b});
                    break;
                default:
                    body.rx(angle(rng), a).ry(angle(rng), b);
                    break;
            }
        }
        expect_inverse_is_identity(body, rng);
    }
}

// The same ansatz built with parameters and bound, and built by hand with
// append and the same arithmetic.
CircuitBuilder ansatz(std::uint32_t n) {
    CircuitBuilder b(n, n);
    const Parameter theta("theta");
    const Parameter phi("phi");
    for (QubitId q = 0; q < n; ++q) {
        b.h(q).ry(theta, q);
    }
    for (QubitId q = 0; q + 1 < n; ++q) {
        b.rzz(2.0 * theta - phi, q, q + 1);
    }
    for (QubitId q = 0; q < n; ++q) {
        b.u(phi, -theta, 0.5 + phi, q);
    }
    return b;
}

Circuit ansatz_by_hand(std::uint32_t n, double theta, double phi) {
    Circuit c(n, n);
    for (QubitId q = 0; q < n; ++q) {
        c.append(Opcode::H, {q});
        c.append(Opcode::RY, {q}, {theta});
    }
    for (QubitId q = 0; q + 1 < n; ++q) {
        c.append(Opcode::RZZ, {q, q + 1}, {2.0 * theta + -1.0 * phi});
    }
    for (QubitId q = 0; q < n; ++q) {
        c.append(Opcode::U, {q}, {phi, -theta, 1.0 * phi + 0.5});
    }
    return c;
}

Observable z_on(std::uint32_t n, QubitId q) {
    PauliString p;
    p.num_qubits = n;
    p.x_words = {0};
    p.z_words = {std::uint64_t{1} << q};
    Observable obs;
    obs.terms.emplace_back(p, Complex{1.0, 0.0});
    return obs;
}

TEST(BuilderRunTest, BoundCircuitRunsBitwiseAsTheHandBuiltOne) {
    const std::uint32_t n = 5;
    const CircuitBuilder builder = ansatz(n);
    const double theta = 0.4321;
    const double phi = -1.234;
    const Circuit bound = builder.bind({{"theta", theta}, {"phi", phi}});
    const Circuit hand = ansatz_by_hand(n, theta, phi);

    const RunOptions sv{.shots = 0, .want_statevector = true, .seed = 7};
    const std::vector<Complex> a = *run(bound, sv).statevector();
    const std::vector<Complex> b = *run(hand, sv).statevector();
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_TRUE(same_bits(a[i], b[i])) << "amplitude " << i;
    }

    RunOptions ev{.shots = 0, .seed = 7};
    ev.expectation = z_on(n, 2);
    EXPECT_TRUE(same_bits(*run(bound, ev).expectation_value(), *run(hand, ev).expectation_value()));

    CircuitBuilder measured = builder;
    for (QubitId q = 0; q < n; ++q) {
        measured.measure(q, q);
    }
    Circuit hand_measured = hand;
    for (QubitId q = 0; q < n; ++q) {
        hand_measured.append_measure(q, q);
    }
    const RunOptions counts{.shots = 2000, .seed = 99};
    EXPECT_EQ(run(measured.bind({{"theta", theta}, {"phi", phi}}), counts).counts(),
              run(hand_measured, counts).counts());
}

TEST(BuilderRunTest, CliffordBuilderRunsOnTheStabilizerEngineAsByHand) {
    // A parameter-free Clifford circuit runs through run(CircuitBuilder), and
    // on the stabilizer plan gives the counts of the same circuit by hand.
    const std::uint32_t n = 40;
    CircuitBuilder b(n);
    Circuit hand(n, 0);
    for (QubitId q = 0; q < n; ++q) {
        b.h(q);
        hand.append(Opcode::H, {q});
    }
    // Long-range CX layers, so the state is far from a low-bond-dimension
    // chain and the stabilizer engine is the plan.
    for (QubitId layer = 1; layer <= 4; ++layer) {
        for (QubitId q = 0; q < n; ++q) {
            const QubitId t = (q + layer * 7) % n;
            b.cx(q, t).s(t);
            hand.append(Opcode::CX, {q, t});
            hand.append(Opcode::S, {t});
        }
    }
    b.measure_all();
    const CircuitBuilder hand_builder = [&] {
        CircuitBuilder hb = CircuitBuilder::from_circuit(hand);
        hb.measure_all();
        return hb;
    }();
    const RunOptions counts{.shots = 500, .seed = 3};
    const Result from_builder = run(b, counts);
    EXPECT_EQ(from_builder.plan().strategy, Strategy::DirectStabilizer);
    EXPECT_EQ(from_builder.counts(), run(hand_builder.to_circuit(), counts).counts());
}

TEST(BuilderRunTest, RunOfABuilderWithParametersIsRefusedByName) {
    CircuitBuilder b(1);
    b.rx(Parameter("gamma"), 0);
    try {
        (void)run(b, RunOptions{.shots = 0, .want_statevector = true});
        FAIL() << "expected InvalidCircuit";
    } catch (const common::InvalidCircuit& e) {
        EXPECT_NE(std::string(e.what()).find("'gamma'"), std::string::npos) << e.what();
    }
}

TEST(BuilderRunTest, RebindingLoopReusesThePlan) {
    const std::uint32_t n = 6;
    const CircuitBuilder builder = ansatz(n);
    RunOptions ev{.shots = 0, .seed = 7};
    ev.expectation = z_on(n, 0);
    plan_cache_clear();
    const PlanCacheStats before = plan_cache_stats();
    for (int step = 0; step < 5; ++step) {
        const double theta = 0.1 + 0.2 * step;
        (void)run(builder.bind({{"theta", theta}, {"phi", 0.3 - 0.05 * step}}), ev);
    }
    const PlanCacheStats after = plan_cache_stats();
    EXPECT_EQ(after.misses - before.misses, 1U);
    EXPECT_EQ(after.hits - before.hits, 4U);
}

}  // namespace
}  // namespace menos
