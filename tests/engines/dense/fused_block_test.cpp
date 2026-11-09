// The fused block matrix and the block sweep (Phase 24 item 7, docs/adr/0028
// section 17): a block's matrix equals the product of its gates (an
// independent oracle, within tolerance), the sweep over blocks equals the
// gate-by-gate sweep within tolerance, and the fused sweep is BITWISE
// identical for every thread count, as the unfused one is (Phase 15).

#include "engines/dense/fused_block.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <random>
#include <variant>
#include <vector>

#include "common/thread_count.h"
#include "engines/dense/apply_matrix.h"
#include "ir/fuse_gates.h"
#include "ir/gate_table.h"
#include "kernels/dense_kernels.h"

namespace menos::engines::dense {
namespace {

using Complex = std::complex<double>;

// The oracle of tests/ir/fuse_gates_test.cpp: a plain gather/matrix/scatter.
void apply_oracle(std::vector<Complex>& state,
                  std::span<const Complex> matrix,
                  std::span<const QubitId> qubits) {
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

void apply_oracle(std::vector<Complex>& state, const Operation& op) {
    std::visit(
        [&](const auto& o) {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                apply_oracle(state, ir::generate_matrix(o.opcode, o.params), o.qubits);
            } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                apply_oracle(state, o.matrix, o.qubits);
            }
        },
        op);
}

std::vector<Complex> random_state(std::uint32_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> s(std::size_t{1} << n);
    for (Complex& c : s) {
        c = Complex{dist(rng), dist(rng)};
    }
    return s;
}

// Brickwork with the gate mix fusion acts on: U on every qubit, CX on
// neighbouring pairs, a CZ and an RZ here and there.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): n qubits, layers deep.
std::vector<Operation> brickwork(std::uint32_t n, int layers, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    std::vector<Operation> ops;
    for (int layer = 0; layer < layers; ++layer) {
        for (QubitId q = 0; q < n; ++q) {
            ops.emplace_back(GateOp{Opcode::U, {q}, {angle(rng), angle(rng), angle(rng)}});
        }
        for (QubitId q = static_cast<QubitId>(layer % 2); q + 1 < n; q += 2) {
            ops.emplace_back(GateOp{Opcode::CX, {q, static_cast<QubitId>(q + 1)}, {}});
        }
        ops.emplace_back(GateOp{Opcode::RZ, {0}, {angle(rng)}});
        if (n >= 2) {
            ops.emplace_back(GateOp{Opcode::CZ, {0, 1}, {}});
        }
    }
    return ops;
}

TEST(FusedBlockTest, ABlocksMatrixIsTheProductOfItsGates) {
    std::mt19937_64 rng(1801);
    for (int trial = 0; trial < 20; ++trial) {
        const std::uint32_t n = 3 + static_cast<std::uint32_t>(trial % 3);
        const std::vector<Operation> ops = brickwork(n, 2, rng);
        const ir::FusionPlan plan = ir::plan_fusion(ops);
        for (std::size_t b = 0; b < plan.block_count(); ++b) {
            if (plan.block_operations(b).size() < 2) {
                continue;
            }
            const std::vector<Complex> matrix = fused_block_matrix(plan, b, ops);
            const auto operands = plan.block_operands(b);
            // Oracle: the block's gates applied to a random state, against the
            // fused matrix applied once.
            std::vector<Complex> a = random_state(n, rng);
            std::vector<Complex> fused = a;
            for (const std::uint32_t index : plan.block_operations(b)) {
                apply_oracle(a, ops[index]);
            }
            apply_oracle(fused, matrix, operands);
            for (std::size_t i = 0; i < a.size(); ++i) {
                ASSERT_NEAR(std::abs(a[i] - fused[i]), 0.0, 1e-12) << "block " << b << " i=" << i;
            }
        }
    }
}

TEST(FusedBlockTest, TheBlockSweepEqualsTheGateSweepWithinTolerance) {
    std::mt19937_64 rng(1802);
    for (int trial = 0; trial < 10; ++trial) {
        const std::uint32_t n = 4 + static_cast<std::uint32_t>(trial % 4);
        const std::vector<Operation> ops = brickwork(n, 3, rng);
        const ir::FusionPlan plan = ir::plan_fusion(ops);
        ASSERT_GT(plan.gates_absorbed, 0U);
        std::vector<Complex> unfused = random_state(n, rng);
        std::vector<Complex> fused = unfused;
        for (const Operation& op : ops) {
            apply_oracle(unfused, op);
        }
        apply_blocks(
            plan, ops, nullptr, [&](std::span<const Complex> m, std::span<const QubitId> q) {
                apply_matrix(fused, m, q, /*num_threads=*/1);
            });
        for (std::size_t i = 0; i < unfused.size(); ++i) {
            ASSERT_NEAR(std::abs(unfused[i] - fused[i]), 0.0, 1e-12) << "i=" << i;
        }
    }
}

TEST(FusedBlockTest, TheFusedSweepIsBitwiseIdenticalForEveryThreadCount) {
    // At a size above the threading cut-off for 1- and 2-qubit blocks.
    std::mt19937_64 rng(1803);
    const std::uint32_t n = 19;
    const std::vector<Operation> ops = brickwork(n, 2, rng);
    const ir::FusionPlan plan = ir::plan_fusion(ops);
    ASSERT_GT(plan.gates_absorbed, 0U);
    const std::vector<Complex> initial = random_state(n, rng);
    std::vector<std::vector<Complex>> results;
    for (const std::uint32_t threads : {1U, 2U, 3U, common::default_thread_count()}) {
        std::vector<Complex> state = initial;
        apply_blocks(
            plan, ops, nullptr, [&](std::span<const Complex> m, std::span<const QubitId> q) {
                apply_matrix(state, m, q, threads);
            });
        results.push_back(std::move(state));
    }
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_EQ(
            std::memcmp(results[0].data(), results[i].data(), results[0].size() * sizeof(Complex)),
            0)
            << "thread count index " << i;
    }
}

TEST(FusedBlockTest, ASingletonBlockIsAppliedWithTheGatesOwnMatrixAndOperandOrder) {
    // One gate: the sweep over the plan applies exactly what the unfused
    // engine applies, so the bytes match the direct call -- including a gate
    // whose operands are not ascending (cx 3,1 is not cx 1,3).
    std::mt19937_64 rng(1804);
    const std::vector<Operation> ops{GateOp{Opcode::U, {2}, {0.3, 0.4, 0.5}},
                                     GateOp{Opcode::CX, {3, 1}, {}}};
    const ir::FusionPlan plan = ir::plan_fusion(ops);
    ASSERT_EQ(plan.block_count(), 2U);
    const std::vector<Complex> initial = random_state(5, rng);
    std::vector<Complex> direct = initial;
    std::vector<Complex> swept = initial;
    apply_matrix(direct,
                 ir::generate_matrix(Opcode::U, std::vector<double>{0.3, 0.4, 0.5}),
                 std::vector<QubitId>{2},
                 1);
    apply_matrix(direct, ir::generate_matrix(Opcode::CX, {}), std::vector<QubitId>{3, 1}, 1);
    apply_blocks(plan, ops, nullptr, [&](std::span<const Complex> m, std::span<const QubitId> q) {
        apply_matrix(swept, m, q, 1);
    });
    EXPECT_EQ(std::memcmp(direct.data(), swept.data(), direct.size() * sizeof(Complex)), 0);
}

TEST(FusedBlockTest, TheMonitorSeesOneBoundaryPerBlock) {
    struct Counting final : ExecutionMonitor {
        mutable std::uint64_t last = 0;
        mutable std::uint64_t calls = 0;
        void at_safe_boundary(std::uint64_t boundaries_crossed) const override {
            last = boundaries_crossed;
            ++calls;
        }
    };
    std::mt19937_64 rng(1805);
    const std::vector<Operation> ops = brickwork(4, 2, rng);
    const ir::FusionPlan plan = ir::plan_fusion(ops);
    std::vector<Complex> state = random_state(4, rng);
    const Counting monitor;
    apply_blocks(plan, ops, &monitor, [&](std::span<const Complex> m, std::span<const QubitId> q) {
        apply_matrix(state, m, q, 1);
    });
    EXPECT_EQ(monitor.calls, plan.block_count());
    EXPECT_EQ(monitor.last, plan.block_count());
}

TEST(FusedBlockTest, TransientBytesCoverTheMatrixAndTheScratch) {
    std::mt19937_64 rng(1806);
    const std::vector<Operation> ops = brickwork(4, 2, rng);
    const ir::FusionPlan plan = ir::plan_fusion(ops);
    const std::uint64_t worst = max_block_transient_bytes(plan, ops, 1);
    EXPECT_GT(worst, 0U);
    for (std::size_t b = 0; b < plan.block_count(); ++b) {
        EXPECT_LE(block_transient_bytes(plan, b, ops, 1), worst);
    }
}

}  // namespace
}  // namespace menos::engines::dense
