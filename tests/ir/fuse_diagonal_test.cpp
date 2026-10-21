#include <algorithm>
#include "ir/fuse_diagonal.h"

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <random>
#include <variant>
#include <vector>

#include "common/error.h"
#include "ir/gate_table.h"

namespace menos::ir {
namespace {

using Complex = std::complex<double>;

// Applies op to a dense state (operand-order bit convention) with a plain
// gather/matrix/scatter -- the oracle, independent of the kernels.
void apply(std::vector<Complex>& state, const Operation& op) {
    const auto [qubits, matrix] = std::visit(
        [](const auto& o) -> std::pair<std::vector<QubitId>, std::vector<Complex>> {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                return {o.qubits, generate_matrix(o.opcode, o.params)};
            } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                return {o.qubits, o.matrix};
            } else {
                throw common::InternalError("apply: not a gate");
            }
        },
        op);
    const std::size_t k = qubits.size();
    const std::size_t dim = std::size_t{1} << k;
    std::vector<Complex> next = state;
    for (std::size_t m = 0; m < state.size(); ++m) {
        std::size_t local = 0;
        for (std::size_t j = 0; j < k; ++j) {
            local |= ((m >> qubits[j]) & std::size_t{1}) << j;
        }
        Complex sum{0.0, 0.0};
        for (std::size_t in = 0; in < dim; ++in) {
            std::size_t src = m;
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t bit = std::size_t{1} << qubits[j];
                src = ((in >> j) & 1u) ? (src | bit) : (src & ~bit);
            }
            sum += matrix[local * dim + in] * state[src];
        }
        next[m] = sum;
    }
    state = next;
}

std::vector<Complex> random_state(std::uint32_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> s(std::size_t{1} << n);
    for (Complex& c : s) {
        c = Complex{dist(rng), dist(rng)};
    }
    return s;
}

void expect_same_action(const std::vector<Operation>& original,
                        const std::vector<Operation>& fused,
                        std::uint32_t n,
                        std::mt19937_64& rng) {
    std::vector<Complex> a = random_state(n, rng);
    std::vector<Complex> b = a;
    for (const Operation& op : original) {
        apply(a, op);
    }
    for (const Operation& op : fused) {
        apply(b, op);
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(std::abs(a[i] - b[i]), 0.0, 1e-12) << "amplitude " << i;
    }
}

std::size_t gate_count(const std::vector<Operation>& ops) {
    std::size_t c = 0;
    for (const Operation& op : ops) {
        c += std::holds_alternative<GateOp>(op) || std::holds_alternative<UnitaryOp>(op);
    }
    return c;
}

TEST(FuseDiagonalTest, AQaoaCostLayerFusesIntoFewBlocksWithTheSameAction) {
    std::mt19937_64 rng(31);
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    const std::uint32_t n = 7;
    Circuit circuit(n, 0);
    for (QubitId q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {q});
    }
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
    const std::vector<Operation> fused = fuse_diagonal_gates(circuit.operations(), 5);
    // 7 H + (21 RZZ + 7 RZ fused) + 7 RX. The 28 diagonal gates on 7 qubits
    // need at least ceil(21 / C(5,2)) = 3 blocks; greedy gives a handful.
    EXPECT_LT(gate_count(fused), 7 + 10 + 7);
    EXPECT_GE(gate_count(fused), 7 + 3 + 7);
    // Every block is a diagonal unitary on at most 5 qubits.
    for (const Operation& op : fused) {
        if (const auto* u = std::get_if<UnitaryOp>(&op)) {
            EXPECT_LE(u->qubits.size(), 5u);
            EXPECT_TRUE(is_diagonal_operation(op));
        }
    }
    expect_same_action(circuit.operations(), fused, n, rng);
}

TEST(FuseDiagonalTest, NonDiagonalGatesAndMeasurementsEndARun) {
    Circuit circuit(3, 3);
    circuit.append(Opcode::RZ, {0}, {0.3});
    circuit.append(Opcode::CP, {0, 1}, {0.4});
    circuit.append(Opcode::H, {1});  // breaks the run
    circuit.append(Opcode::RZZ, {1, 2}, {0.5});
    circuit.append(Opcode::T, {2});
    circuit.append_measure(2, 2);  // breaks the run
    circuit.append(Opcode::S, {0});
    circuit.append(Opcode::Z, {1});
    const std::vector<Operation> fused = fuse_diagonal_gates(circuit.operations(), 5);
    // {RZ, CP} -> 1 block; H; {RZZ, T} -> 1 block; measure; {S, Z} -> 1 block.
    ASSERT_EQ(fused.size(), 5u);
    EXPECT_TRUE(std::holds_alternative<UnitaryOp>(fused[0]));
    EXPECT_TRUE(std::holds_alternative<GateOp>(fused[1]));
    EXPECT_TRUE(std::holds_alternative<UnitaryOp>(fused[2]));
    EXPECT_TRUE(std::holds_alternative<MeasureOp>(fused[3]));
    EXPECT_TRUE(std::holds_alternative<UnitaryOp>(fused[4]));
    EXPECT_EQ(std::get<UnitaryOp>(fused[4]).qubits.size(), 2u);
    std::mt19937_64 rng(32);
    // Same action on the gate part (measurements are not applied by the oracle).
    std::vector<Operation> original_gates;
    std::vector<Operation> fused_gates;
    for (const Operation& op : circuit.operations()) {
        if (!std::holds_alternative<MeasureOp>(op)) {
            original_gates.push_back(op);
        }
    }
    for (const Operation& op : fused) {
        if (!std::holds_alternative<MeasureOp>(op)) {
            fused_gates.push_back(op);
        }
    }
    expect_same_action(original_gates, fused_gates, 3, rng);
}

TEST(FuseDiagonalTest, ASingleDiagonalGateIsLeftAlone) {
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::RZ, {0}, {0.3});
    circuit.append(Opcode::H, {0});
    const std::vector<Operation> fused = fuse_diagonal_gates(circuit.operations(), 5);
    ASSERT_EQ(fused.size(), 3u);
    EXPECT_TRUE(std::holds_alternative<GateOp>(fused[1]));
}

TEST(FuseDiagonalTest, RespectsTheQubitCapAndKeepsWiderGates) {
    Circuit circuit(6, 0);
    for (QubitId i = 0; i < 6; ++i) {
        for (QubitId j = i + 1; j < 6; ++j) {
            circuit.append(Opcode::CZ, {i, j});
        }
    }
    const std::vector<Operation> fused = fuse_diagonal_gates(circuit.operations(), 3);
    for (const Operation& op : fused) {
        if (const auto* u = std::get_if<UnitaryOp>(&op)) {
            EXPECT_LE(u->qubits.size(), 3u);
        }
    }
    std::mt19937_64 rng(33);
    expect_same_action(circuit.operations(), fused, 6, rng);
    // A 4-qubit diagonal unitary under a 3-qubit cap stays as it is.
    std::vector<Complex> d(256, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < 16; ++i) {
        d[i * 16 + i] = Complex{0.0, (i % 2) ? 1.0 : -1.0};
    }
    Circuit wide(4, 0);
    wide.append(Opcode::RZ, {0}, {0.2});
    wide.append_unitary({0, 1, 2, 3}, d);
    const std::vector<Operation> kept = fuse_diagonal_gates(wide.operations(), 3);
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_TRUE(std::holds_alternative<GateOp>(kept[0]));
    EXPECT_TRUE(std::holds_alternative<UnitaryOp>(kept[1]));
    EXPECT_THROW((void)fuse_diagonal_gates(wide.operations(), 1), common::InternalError);
}

TEST(FuseDiagonalTest, RandomDiagonalRunsOnShuffledOperandsMatchTheOracle) {
    std::mt19937_64 rng(34);
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    const Opcode diagonal[] = {Opcode::Z,
                               Opcode::S,
                               Opcode::Sdg,
                               Opcode::T,
                               Opcode::Tdg,
                               Opcode::RZ,
                               Opcode::P,
                               Opcode::CZ,
                               Opcode::CP,
                               Opcode::RZZ};
    for (int trial = 0; trial < 40; ++trial) {
        const std::uint32_t n = 3 + static_cast<std::uint32_t>(rng() % 4);
        Circuit circuit(n, 0);
        for (int g = 0; g < 25; ++g) {
            const Opcode op = diagonal[rng() % 10];
            const GateInfo& info = gate_info(op);
            std::vector<QubitId> qubits;
            while (qubits.size() < static_cast<std::size_t>(info.arity)) {
                const QubitId q = static_cast<QubitId>(rng() % n);
                if (std::find(qubits.begin(), qubits.end(), q) == qubits.end()) {
                    qubits.push_back(q);
                }
            }
            std::vector<double> params(static_cast<std::size_t>(info.num_params));
            for (double& v : params) {
                v = angle(rng);
            }
            circuit.append(op, qubits, params);
            if (rng() % 7 == 0) {
                circuit.append(Opcode::RX, {static_cast<QubitId>(rng() % n)}, {angle(rng)});
            }
        }
        const std::uint32_t cap = 2 + static_cast<std::uint32_t>(rng() % 4);
        expect_same_action(
            circuit.operations(), fuse_diagonal_gates(circuit.operations(), cap), n, rng);
    }
}

}  // namespace
}  // namespace menos::ir
