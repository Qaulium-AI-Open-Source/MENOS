// Gate fusion by containment (Phase 24 item 7, docs/adr/0028 section 17):
// the grouping rule, block by block on hand-built circuits, and its
// soundness on random circuits through an oracle that applies the blocks'
// gates in the plan's order and compares with program order.

#include "ir/fuse_gates.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <numeric>
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
                return {{}, {}};  // a MeasureOp applies nothing
            }
        },
        op);
    if (qubits.empty()) {
        return;
    }
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
                src = ((in >> j) & 1U) != 0 ? (src | bit) : (src & ~bit);
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

std::vector<std::vector<std::uint32_t>> blocks_of(const FusionPlan& plan) {
    std::vector<std::vector<std::uint32_t>> out;
    for (std::size_t b = 0; b < plan.block_count(); ++b) {
        const auto members = plan.block_operations(b);
        out.emplace_back(members.begin(), members.end());
    }
    return out;
}

std::vector<std::vector<QubitId>> operands_of(const FusionPlan& plan) {
    std::vector<std::vector<QubitId>> out;
    for (std::size_t b = 0; b < plan.block_count(); ++b) {
        const auto q = plan.block_operands(b);
        out.emplace_back(q.begin(), q.end());
    }
    return out;
}

TEST(FuseGatesTest, AGateJoinsTheNewestBlockSharingAQubitWhenOneSetContainsTheOther) {
    // U(0), U(1), CX(0,1): CX shares a qubit with both, the newest is U(1)'s
    // block, {1} is contained in {0,1}, so CX joins it; U(0)'s block stays
    // apart (it is older than CX's dependency) and runs first.
    const std::vector<Operation> ops{
        GateOp{Opcode::U, {0}, {0.1, 0.2, 0.3}},
        GateOp{Opcode::U, {1}, {0.4, 0.5, 0.6}},
        GateOp{Opcode::CX, {0, 1}, {}},
    };
    const FusionPlan plan = plan_fusion(ops);
    EXPECT_EQ(blocks_of(plan), (std::vector<std::vector<std::uint32_t>>{{0}, {1, 2}}));
    EXPECT_EQ(operands_of(plan), (std::vector<std::vector<QubitId>>{{0}, {0, 1}}));
    EXPECT_EQ(plan.fused_blocks, 1U);
    EXPECT_EQ(plan.gates_absorbed, 1U);
}

TEST(FuseGatesTest, ALaterGateOnAQubitOfTheBlockJoinsIt) {
    // CX(0,1) then U(0) and U(1): both are contained in the block.
    const std::vector<Operation> ops{
        GateOp{Opcode::CX, {0, 1}, {}},
        GateOp{Opcode::U, {0}, {0.1, 0.2, 0.3}},
        GateOp{Opcode::U, {1}, {0.4, 0.5, 0.6}},
        GateOp{Opcode::H, {2}, {}},
    };
    const FusionPlan plan = plan_fusion(ops);
    EXPECT_EQ(blocks_of(plan), (std::vector<std::vector<std::uint32_t>>{{0, 1, 2}, {3}}));
    EXPECT_EQ(plan.fused_blocks, 1U);
    EXPECT_EQ(plan.gates_absorbed, 2U);
}

TEST(FuseGatesTest, TwoGatesOnDifferentQubitsNeverMerge) {
    // Neither {0} nor {1} contains the other: two 1-qubit passes stay two.
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::H, {1}, {}},
        GateOp{Opcode::CX, {2, 3}, {}},
        GateOp{Opcode::CX, {3, 4}, {}},  // {3,4} and {2,3} overlap without containment
    };
    const FusionPlan plan = plan_fusion(ops);
    EXPECT_EQ(plan.block_count(), 4U);
    EXPECT_EQ(plan.gates_absorbed, 0U);
}

TEST(FuseGatesTest, TwoGatesOnTheSameQubitsMerge) {
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::T, {0}, {}},
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::CX, {1, 2}, {}},
        GateOp{Opcode::CX, {2, 1}, {}},  // the same set in the other order
    };
    const FusionPlan plan = plan_fusion(ops);
    EXPECT_EQ(blocks_of(plan), (std::vector<std::vector<std::uint32_t>>{{0, 1, 2}, {3, 4}}));
    EXPECT_EQ(operands_of(plan), (std::vector<std::vector<QubitId>>{{0}, {1, 2}}));
}

TEST(FuseGatesTest, ADenseGateNeverWidensADiagonalPassAndViceVersa) {
    // CZ(0,1) is diagonal and wider than U(0): apart. U(2) then CZ(2,3):
    // the wider gate is diagonal, the narrower block dense: apart. RZ(4)
    // then CX(4,5): the wider gate is dense: together. RZ(6) then CZ(6,7):
    // both diagonal: together, and the block stays diagonal so T(6) joins.
    const std::vector<Operation> ops{
        GateOp{Opcode::CZ, {0, 1}, {}},
        GateOp{Opcode::U, {0}, {0.1, 0.2, 0.3}},
        GateOp{Opcode::U, {2}, {0.1, 0.2, 0.3}},
        GateOp{Opcode::CZ, {2, 3}, {}},
        GateOp{Opcode::RZ, {4}, {0.7}},
        GateOp{Opcode::CX, {4, 5}, {}},
        GateOp{Opcode::RZ, {6}, {0.7}},
        GateOp{Opcode::CZ, {6, 7}, {}},
        GateOp{Opcode::T, {6}, {}},
    };
    const FusionPlan plan = plan_fusion(ops);
    EXPECT_EQ(blocks_of(plan),
              (std::vector<std::vector<std::uint32_t>>{{0}, {1}, {2}, {3}, {4, 5}, {6, 7, 8}}));
}

TEST(FuseGatesTest, ADiagonalUnitaryCountsAsDiagonal) {
    // A diagonal UnitaryOp on {0,1} is wider than a dense H(0): apart; a
    // dense UnitaryOp on {0,1} takes it.
    const std::vector<Complex> diagonal = {{1, 0},
                                           {0, 0},
                                           {0, 0},
                                           {0, 0},
                                           {0, 0},
                                           {0, 1},
                                           {0, 0},
                                           {0, 0},
                                           {0, 0},
                                           {0, 0},
                                           {-1, 0},
                                           {0, 0},
                                           {0, 0},
                                           {0, 0},
                                           {0, 0},
                                           {0, -1}};
    std::vector<Complex> dense = diagonal;
    dense[1] = Complex{0.5, 0.0};
    const std::vector<Operation> a{UnitaryOp{{0, 1}, diagonal}, GateOp{Opcode::H, {0}, {}}};
    const std::vector<Operation> b{UnitaryOp{{0, 1}, dense}, GateOp{Opcode::H, {0}, {}}};
    EXPECT_EQ(plan_fusion(a).block_count(), 2U);
    EXPECT_EQ(plan_fusion(b).block_count(), 1U);
}

TEST(FuseGatesTest, AMeasurementIsABarrier) {
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        MeasureOp{0, 0},
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::T, {0}, {}},
    };
    const FusionPlan plan = plan_fusion(ops);
    EXPECT_EQ(blocks_of(plan), (std::vector<std::vector<std::uint32_t>>{{0}, {1}, {2, 3}}));
    EXPECT_TRUE(plan.block_operands(1).empty());
}

TEST(FuseGatesTest, AnEmptyListHasNoBlocks) {
    const FusionPlan plan = plan_fusion({});
    EXPECT_EQ(plan.block_count(), 0U);
    EXPECT_EQ(plan.fused_blocks, 0U);
    EXPECT_TRUE(plan.order.empty());
}

TEST(FuseGatesTest, AnOperandBeyond63QubitsIsRefused) {
    const std::vector<Operation> ops{GateOp{Opcode::H, {64}, {}}};
    EXPECT_THROW(static_cast<void>(plan_fusion(ops)), common::InternalError);
}

// A random circuit over n qubits: named gates of 1-3 qubits (CCX included),
// diagonal gates, and explicit 1- and 2-qubit unitaries, with a measurement
// in the middle of some.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): n qubits, gates operations.
std::vector<Operation> random_circuit(std::uint32_t n, std::size_t gates, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    std::uniform_int_distribution<std::uint32_t> qubit(0, n - 1);
    std::uniform_int_distribution<int> kind(0, 8);
    std::vector<Operation> ops;
    const auto distinct = [&](std::size_t k) {
        std::vector<QubitId> q;
        while (q.size() < k) {
            const QubitId c = qubit(rng);
            if (std::find(q.begin(), q.end(), c) == q.end()) {
                q.push_back(c);
            }
        }
        return q;
    };
    for (std::size_t i = 0; i < gates; ++i) {
        switch (kind(rng)) {
            case 0:
                ops.emplace_back(
                    GateOp{Opcode::U, distinct(1), {angle(rng), angle(rng), angle(rng)}});
                break;
            case 1:
                ops.emplace_back(GateOp{Opcode::RZ, distinct(1), {angle(rng)}});
                break;
            case 2:
                ops.emplace_back(GateOp{Opcode::CX, distinct(2), {}});
                break;
            case 3:
                ops.emplace_back(GateOp{Opcode::CZ, distinct(2), {}});
                break;
            case 4:
                ops.emplace_back(GateOp{Opcode::RZZ, distinct(2), {angle(rng)}});
                break;
            case 5:
                if (n >= 3) {
                    ops.emplace_back(GateOp{Opcode::CCX, distinct(3), {}});
                }
                break;
            case 6: {
                // A 2-qubit unitary, exactly unitary up to rounding: the
                // Kronecker product of two U gates, so amplitudes keep their
                // size and the tolerance below stays meaningful.
                const std::vector<Complex> a = generate_matrix(
                    Opcode::U, std::vector<double>{angle(rng), angle(rng), angle(rng)});
                const std::vector<Complex> b = generate_matrix(
                    Opcode::U, std::vector<double>{angle(rng), angle(rng), angle(rng)});
                std::vector<Complex> m(16);
                for (std::size_t r = 0; r < 4; ++r) {
                    for (std::size_t c = 0; c < 4; ++c) {
                        m[r * 4 + c] = a[(r & 1U) * 2 + (c & 1U)] * b[(r >> 1U) * 2 + (c >> 1U)];
                    }
                }
                ops.emplace_back(UnitaryOp{distinct(2), std::move(m)});
                break;
            }
            case 7:
                ops.emplace_back(GateOp{Opcode::H, distinct(1), {}});
                break;
            default:
                if (i == gates / 2) {
                    ops.emplace_back(MeasureOp{qubit(rng), 0});
                } else {
                    ops.emplace_back(GateOp{Opcode::T, distinct(1), {}});
                }
                break;
        }
    }
    return ops;
}

TEST(FuseGatesTest, ThePlanIsAPermutationIntoBlocksWhoseOperandsAreTheirGatesUnion) {
    std::mt19937_64 rng(1701);
    for (int trial = 0; trial < 50; ++trial) {
        const std::uint32_t n = 2 + static_cast<std::uint32_t>(trial % 7);
        const std::vector<Operation> ops = random_circuit(n, 40, rng);
        const FusionPlan plan = plan_fusion(ops);
        std::vector<std::uint32_t> sorted = plan.order;
        std::sort(sorted.begin(), sorted.end());
        std::vector<std::uint32_t> identity(ops.size());
        std::iota(identity.begin(), identity.end(), 0U);
        EXPECT_EQ(sorted, identity);
        ASSERT_EQ(plan.block_begin.size(), plan.block_count() + 1);
        EXPECT_EQ(plan.block_begin.front(), 0U);
        EXPECT_EQ(plan.block_begin.back(), ops.size());
        std::uint64_t fused = 0;
        std::uint64_t absorbed = 0;
        for (std::size_t b = 0; b < plan.block_count(); ++b) {
            std::vector<QubitId> expected;
            for (const std::uint32_t index : plan.block_operations(b)) {
                std::visit(
                    [&](const auto& o) {
                        using T = std::decay_t<decltype(o)>;
                        if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                            for (const QubitId q : o.qubits) {
                                if (std::find(expected.begin(), expected.end(), q) ==
                                    expected.end()) {
                                    expected.push_back(q);
                                }
                            }
                        }
                    },
                    ops[index]);
            }
            std::sort(expected.begin(), expected.end());
            const auto operands = plan.block_operands(b);
            EXPECT_EQ(std::vector<QubitId>(operands.begin(), operands.end()), expected);
            // Operations within a block keep program order.
            const auto members = plan.block_operations(b);
            EXPECT_TRUE(std::is_sorted(members.begin(), members.end()));
            if (members.size() >= 2) {
                ++fused;
                absorbed += members.size() - 1;
            }
        }
        EXPECT_EQ(plan.fused_blocks, fused);
        EXPECT_EQ(plan.gates_absorbed, absorbed);
        EXPECT_LE(plan.heap_bytes(),
                  FusionPlan::construction_peak_bytes(
                      ops.size(), plan.block_count(), plan.block_qubits.size()));
    }
}

TEST(FuseGatesTest, ApplyingTheBlocksInPlanOrderEqualsProgramOrder) {
    // The reordering the plan performs (a gate joining an older block past
    // blocks on disjoint qubits) must leave the circuit's action unchanged:
    // the oracle applies the gates one by one in both orders.
    std::mt19937_64 rng(1702);
    for (int trial = 0; trial < 60; ++trial) {
        const std::uint32_t n = 2 + static_cast<std::uint32_t>(trial % 6);
        const std::vector<Operation> ops = random_circuit(n, 30, rng);
        const FusionPlan plan = plan_fusion(ops);
        std::vector<Complex> a = random_state(n, rng);
        std::vector<Complex> b = a;
        for (const Operation& op : ops) {
            apply(a, op);
        }
        for (const std::uint32_t index : plan.order) {
            apply(b, ops[index]);
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            ASSERT_NEAR(std::abs(a[i] - b[i]), 0.0, 1e-12) << "trial " << trial << " i=" << i;
        }
    }
}

}  // namespace
}  // namespace menos::ir
