// Phase 23 (docs/adr/0027 section 3.2): the chunked executor's pass plan as
// a pure function of a circuit, the chunk size L and the window size L + w.

#include "engines/dense_tiered/pass_plan.h"

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <vector>

#include "crossover_circuit.h"
#include "menos/circuit.h"

namespace menos::engines::dense_tiered {
namespace {

Circuit ladder(std::uint32_t n) {
    Circuit circuit(n, 0);
    for (int layer = 0; layer < benchmarks::kCrossoverLayers; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        for (std::uint32_t q = 0; q + 1 < n; ++q) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    return circuit;
}

std::span<const Operation> ops(const Circuit& circuit) {
    return circuit.operations();
}

TEST(PassPlanTest, ACircuitWithNoGateHasNoPass) {
    Circuit circuit(4, 4);
    circuit.append_measure(0, 0);
    const PassPlan plan = build_pass_plan(ops(circuit), 4, 2, 3);
    EXPECT_EQ(plan.pass_count(), 0U);
    EXPECT_EQ(plan.global_gates, 0U);
    EXPECT_FALSE(plan.window_too_small);
}

TEST(PassPlanTest, LocalGatesOnlyAreOnePassWithAnEmptyGroup) {
    Circuit circuit(6, 0);
    for (QubitId q = 0; q < 3; ++q) {
        circuit.append(Opcode::H, {q});
    }
    circuit.append(Opcode::CX, {0, 2});
    const PassPlan plan = build_pass_plan(ops(circuit), 6, 3, 3);  // w = 0
    ASSERT_EQ(plan.pass_count(), 1U);
    EXPECT_EQ(plan.passes[0].global_mask, 0U);
    EXPECT_EQ(plan.passes[0].first_operation, 0U);
    EXPECT_EQ(plan.passes[0].end_operation, 4U);
    EXPECT_EQ(plan.global_gates, 0U);
    EXPECT_EQ(plan.max_global_operands, 0U);
}

TEST(PassPlanTest, TheGreedyRuleExtendsAGroupWhileItFitsAndClosesOtherwise) {
    // n = 6, L = 3: globals are qubits 3, 4, 5 (mask bits 0, 1, 2).
    Circuit circuit(6, 6);
    circuit.append(Opcode::CX, {0, 3});  // G = {3}
    circuit.append(Opcode::H, {1});      // local, same pass
    circuit.append(Opcode::CX, {4, 1});  // G = {3, 4}, fits w = 2
    circuit.append_measure(0, 0);        // same pass
    circuit.append(Opcode::CX, {5, 0});  // {3, 4, 5} does not fit: new pass G = {5}
    circuit.append(Opcode::CX, {3, 5});  // G = {3, 5}
    circuit.append_measure(1, 1);
    const PassPlan plan = build_pass_plan(ops(circuit), 6, 3, 5);  // w = 2
    ASSERT_EQ(plan.pass_count(), 2U);
    EXPECT_EQ(plan.passes[0].global_mask, 0b011U);
    EXPECT_EQ(plan.passes[0].first_operation, 0U);
    EXPECT_EQ(plan.passes[0].end_operation, 4U);
    EXPECT_EQ(plan.passes[1].global_mask, 0b101U);
    EXPECT_EQ(plan.passes[1].first_operation, 4U);
    EXPECT_EQ(plan.passes[1].end_operation, 7U);
    EXPECT_EQ(plan.global_gates, 4U);
    EXPECT_EQ(plan.max_global_operands, 2U);
    EXPECT_FALSE(plan.window_too_small);

    // With w = 3 everything fits in one pass.
    const PassPlan wide = build_pass_plan(ops(circuit), 6, 3, 6);
    ASSERT_EQ(wide.pass_count(), 1U);
    EXPECT_EQ(wide.passes[0].global_mask, 0b111U);
}

TEST(PassPlanTest, EveryGateGlobalGivesOnePassPerGateWhenTheWindowHoldsOne) {
    Circuit circuit(4, 0);
    circuit.append(Opcode::H, {2});
    circuit.append(Opcode::H, {3});
    circuit.append(Opcode::H, {2});
    const PassPlan plan = build_pass_plan(ops(circuit), 4, 2, 3);  // w = 1
    ASSERT_EQ(plan.pass_count(), 3U);
    EXPECT_EQ(plan.passes[0].global_mask, 0b01U);
    EXPECT_EQ(plan.passes[1].global_mask, 0b10U);
    EXPECT_EQ(plan.passes[2].global_mask, 0b01U);
    EXPECT_EQ(plan.global_gates, 3U);
}

TEST(PassPlanTest, AGateWithMoreGlobalOperandsThanTheWindowHoldsIsReported) {
    // A 5-qubit unitary with all five operands global.
    Circuit circuit(8, 0);
    const std::vector<std::complex<double>> identity = [] {
        std::vector<std::complex<double>> m(32 * 32, {0.0, 0.0});
        for (std::size_t i = 0; i < 32; ++i) {
            m[i * 32 + i] = {1.0, 0.0};
        }
        return m;
    }();
    circuit.append_unitary({3, 4, 5, 6, 7}, identity);
    const PassPlan small = build_pass_plan(ops(circuit), 8, 3, 7);  // w = 4
    EXPECT_TRUE(small.window_too_small);
    EXPECT_EQ(small.max_global_operands, 5U);
    const PassPlan enough = build_pass_plan(ops(circuit), 8, 3, 8);  // w = 5
    EXPECT_FALSE(enough.window_too_small);
    ASSERT_EQ(enough.pass_count(), 1U);
    EXPECT_EQ(enough.passes[0].global_mask, 0b11111U);
}

TEST(PassPlanTest, PassesPartitionTheOperationsInOrder) {
    const Circuit circuit = benchmarks::make_crossover_circuit(12);
    for (std::uint32_t chunk = 6; chunk <= 12; ++chunk) {
        for (std::uint32_t window = chunk; window <= 12; ++window) {
            const PassPlan plan = build_pass_plan(ops(circuit), 12, chunk, window);
            if (plan.window_too_small) {
                continue;
            }
            ASSERT_FALSE(plan.passes.empty());
            EXPECT_EQ(plan.passes.front().first_operation, 0U);
            EXPECT_EQ(plan.passes.back().end_operation, circuit.operations().size());
            for (std::size_t i = 1; i < plan.passes.size(); ++i) {
                EXPECT_EQ(plan.passes[i].first_operation, plan.passes[i - 1].end_operation);
                EXPECT_LT(plan.passes[i].first_operation, plan.passes[i].end_operation);
            }
            EXPECT_EQ(plan.passes.capacity(), plan.passes.size());
        }
    }
}

// The figures of docs/adr/0027 section 10.1, analysis (iv), produced in step 1
// by the same greedy rule in benchmarks/tiered_pass_count.h on commit 88294be
// (docs/benchmarks/phase-23-baseline/analysis/passes-gate-sizes.txt).
TEST(PassPlanTest, StepOnePassCountsOfTheCrossoverFamilyArePinned) {
    const Circuit at31 = benchmarks::make_crossover_circuit(31);
    EXPECT_EQ(at31.operations().size(), 282U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 27, 29).pass_count(), 24U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 27, 30).pass_count(), 13U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 27, 31).pass_count(), 1U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 28, 30).pass_count(), 15U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 25, 27).pass_count(), 36U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 25, 29).pass_count(), 18U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 24, 28).pass_count(), 21U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 27, 29).global_gates, 36U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 27, 29).max_global_operands, 2U);
    const Circuit at33 = benchmarks::make_crossover_circuit(33);
    EXPECT_EQ(build_pass_plan(ops(at33), 33, 28, 30).pass_count(), 30U);
    EXPECT_EQ(build_pass_plan(ops(at33), 33, 27, 29).pass_count(), 36U);
    EXPECT_EQ(build_pass_plan(ops(at33), 33, 24, 28).pass_count(), 27U);
}

TEST(PassPlanTest, StepOnePassCountsOfTheLadderFamilyArePinned) {
    const Circuit at31 = ladder(31);
    EXPECT_EQ(at31.operations().size(), 366U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 27, 29).pass_count(), 30U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 28, 30).pass_count(), 24U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 24, 28).pass_count(), 24U);
    EXPECT_EQ(build_pass_plan(ops(at31), 31, 24, 26).pass_count(), 60U);
}

// Step 1's correction to the plan: in a 2^29 window the crossover circuit at
// n = 31 takes 18 passes with L = 25 against 24 with L = 27, so the executor
// picks the chunk size by pass count, not the largest chunk that fits.
TEST(PassPlanTest, ChoosesTheChunkSizeWithTheFewestPassesLargerOnATie) {
    const Circuit at31 = benchmarks::make_crossover_circuit(31);
    const std::optional<PassPlan> chosen = choose_pass_plan(ops(at31), 31, 29, false);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(chosen->chunk_qubits, 25U);
    EXPECT_EQ(chosen->window_qubits, 29U);
    EXPECT_EQ(chosen->pass_count(), 18U);

    // A window holding the whole state is one pass at the largest chunk.
    const Circuit small = benchmarks::make_crossover_circuit(12);
    const std::optional<PassPlan> whole = choose_pass_plan(ops(small), 12, 12, false);
    ASSERT_TRUE(whole.has_value());
    EXPECT_EQ(whole->chunk_qubits, 12U);
    EXPECT_EQ(whole->pass_count(), 1U);

    // Below the preferred minimum every chunk size competes: the window of
    // 2^8 on a 12-qubit crossover circuit.
    const std::optional<PassPlan> tiny = choose_pass_plan(ops(small), 12, 8, false);
    ASSERT_TRUE(tiny.has_value());
    std::uint64_t fewest = ~std::uint64_t{0};
    std::uint32_t at = 0;
    for (std::uint32_t chunk = 8;; --chunk) {
        const PassPlan plan = build_pass_plan(ops(small), 12, chunk, 8);
        if (!plan.window_too_small && plan.pass_count() < fewest) {
            fewest = plan.pass_count();
            at = chunk;
        }
        if (chunk == 0) {
            break;
        }
    }
    EXPECT_EQ(tiny->pass_count(), fewest);
    EXPECT_EQ(tiny->chunk_qubits, at);
}

TEST(PassPlanTest, AChunkPairRequirementRulesOutTheWholeWindowAsOneChunk) {
    Circuit circuit(6, 0);
    circuit.append(Opcode::H, {0});
    // Without the requirement the whole window is one chunk, one pass.
    const std::optional<PassPlan> free = choose_pass_plan(ops(circuit), 6, 6, false);
    ASSERT_TRUE(free.has_value());
    EXPECT_EQ(free->chunk_qubits, 6U);
    // With it, the chunk is at most half the window.
    const std::optional<PassPlan> paired = choose_pass_plan(ops(circuit), 6, 6, true);
    ASSERT_TRUE(paired.has_value());
    EXPECT_EQ(paired->chunk_qubits, 5U);
    EXPECT_EQ(paired->pass_count(), 1U);
}

TEST(PassPlanTest, NoChunkSizeFitsAGateWiderThanTheWindowsGlobals) {
    Circuit circuit(8, 0);
    const std::vector<std::complex<double>> identity = [] {
        std::vector<std::complex<double>> m(32 * 32, {0.0, 0.0});
        for (std::size_t i = 0; i < 32; ++i) {
            m[i * 32 + i] = {1.0, 0.0};
        }
        return m;
    }();
    circuit.append_unitary({3, 4, 5, 6, 7}, identity);
    // A window of 2^7 holds at most 2^4 chunks of 2^3, or 2^7 chunks of 1
    // amplitude: with L = 0 every operand is global, w = 7 >= 5, so it fits.
    EXPECT_TRUE(choose_pass_plan(ops(circuit), 8, 7, false).has_value());
    // A window of 2^4 cannot hold the 2^5 amplitudes the gate couples.
    EXPECT_FALSE(choose_pass_plan(ops(circuit), 8, 4, false).has_value());
}

TEST(PassPlanTest, ChunksBelowThePreferredMinimumAreTriedOnlyWhenNothingAboveFits) {
    // A 6-qubit unitary on the top six qubits of n = 30 in a 2^27 window: at
    // L >= 24 the window has at most 3 global qubits, so the plan falls back
    // to L = 21, where w = 6 holds the gate.
    Circuit circuit(30, 0);
    const std::vector<std::complex<double>> identity = [] {
        std::vector<std::complex<double>> m(64 * 64, {0.0, 0.0});
        for (std::size_t i = 0; i < 64; ++i) {
            m[i * 64 + i] = {1.0, 0.0};
        }
        return m;
    }();
    circuit.append_unitary({24, 25, 26, 27, 28, 29}, identity);
    const std::optional<PassPlan> plan = choose_pass_plan(ops(circuit), 30, 27, false);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->chunk_qubits, 21U);
    EXPECT_EQ(plan->pass_count(), 1U);
    // A gate the preferred sizes do fit keeps them.
    Circuit easy(30, 0);
    easy.append(Opcode::CX, {28, 29});
    const std::optional<PassPlan> preferred = choose_pass_plan(ops(easy), 30, 27, false);
    ASSERT_TRUE(preferred.has_value());
    EXPECT_EQ(preferred->chunk_qubits, 25U);
}

}  // namespace
}  // namespace menos::engines::dense_tiered
