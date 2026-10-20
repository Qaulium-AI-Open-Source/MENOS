#include "menos/menos.h"

#include <gtest/gtest.h>

#include <complex>
#include <random>
#include <vector>

namespace menos {
namespace {

// A QAOA-shaped circuit: H layer, then p layers of RZZ on every pair, RZ on
// every qubit and RX on every qubit.
Circuit qaoa(std::uint32_t n, int layers, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    Circuit circuit(n, 0);
    for (QubitId q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {q});
    }
    for (int layer = 0; layer < layers; ++layer) {
        for (QubitId i = 0; i < n; ++i) {
            for (QubitId j = i + 1; j < n; ++j) {
                circuit.append(Opcode::RZZ, {i, j}, {angle(rng)});
            }
        }
        for (QubitId q = 0; q < n; ++q) {
            circuit.append(Opcode::RZ, {q}, {angle(rng)});
        }
        for (QubitId q = 0; q < n; ++q) {
            circuit.append(Opcode::RX, {q}, {angle(rng)});
        }
    }
    return circuit;
}

Observable all_zz(std::uint32_t n) {
    Observable obs;
    for (QubitId i = 0; i < n; ++i) {
        for (QubitId j = i + 1; j < n; ++j) {
            PauliString p;
            p.num_qubits = n;
            p.x_words = {0};
            p.z_words = {(std::uint64_t{1} << i) | (std::uint64_t{1} << j)};
            obs.terms.emplace_back(p, std::complex<double>{0.25 * (i + 1), 0.0});
        }
    }
    return obs;
}

// RunOptions::fuse_diagonal_gates changes how the dense engine applies a
// run of diagonal gates, not what it computes: the statevector and the
// expectation agree with the unfused run within the expectation tolerance.
TEST(FuseDiagonalRunTest, FusedAndUnfusedRunsAgree) {
    std::mt19937_64 rng(2024);
    for (const std::uint32_t n : {3u, 6u, 9u}) {
        const Circuit circuit = qaoa(n, 2, rng);
        RunOptions off{.shots = 0, .want_statevector = true, .seed = 1};
        RunOptions on = off;
        on.fuse_diagonal_gates = true;
        const Result a = run(circuit, off);
        const Result b = run(circuit, on);
        ASSERT_EQ(a.plan().strategy, Strategy::DirectDense);
        ASSERT_EQ(b.plan().strategy, Strategy::DirectDense);
        const std::vector<std::complex<double>>& sa = *a.statevector();
        const std::vector<std::complex<double>>& sb = *b.statevector();
        ASSERT_EQ(sa.size(), sb.size());
        for (std::size_t i = 0; i < sa.size(); ++i) {
            EXPECT_NEAR(std::abs(sa[i] - sb[i]), 0.0, 1e-12) << "n=" << n << " i=" << i;
        }

        RunOptions eoff{.shots = 0, .want_statevector = false, .seed = 1, .expectation = all_zz(n)};
        RunOptions eon = eoff;
        eon.fuse_diagonal_gates = true;
        const std::complex<double> ea = *run(circuit, eoff).expectation_value();
        const std::complex<double> eb = *run(circuit, eon).expectation_value();
        EXPECT_NEAR(std::abs(ea - eb), 0.0, 1e-12) << "n=" << n;
    }
}

// Off by default: an unchanged job is byte-identical to before the option
// existed (the specialised kernels are bitwise identical to the generic one,
// so nothing else in this patch can move a bit).
TEST(FuseDiagonalRunTest, OffByDefaultAndIdenticalWhenRepeated) {
    std::mt19937_64 rng(7);
    const Circuit circuit = qaoa(5, 1, rng);
    const RunOptions options{.shots = 0, .want_statevector = true, .seed = 1};
    EXPECT_FALSE(options.fuse_diagonal_gates);
    const auto a = *run(circuit, options).statevector();
    const auto b = *run(circuit, options).statevector();
    EXPECT_EQ(a, b);
}

}  // namespace
}  // namespace menos
