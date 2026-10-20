#include "runtime/runtime.h"

#include <gtest/gtest.h>

#include <algorithm>

#include "analysis/interaction_graph.h"
#include "plan/available_engines.h"
#include "plan/planner.h"
#include "plan/region_builder.h"

namespace menos::runtime {
namespace {

// Test-only convenience, mirroring tests/engines/dense/sample_test.cpp's
// helper: builds a BitString of the given width from a small unsigned
// integer literal.
// width (a clbit count, typically small) and value (the bits to set) are
// not realistically confusable at any call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
BitString make_bitstring(std::uint32_t width, std::uint64_t value) {
    BitString bs(width);
    for (std::uint32_t bit = 0; bit < width; ++bit) {
        if ((value >> bit) & std::uint64_t{1}) {
            bs.set(bit, true);
        }
    }
    return bs;
}

TEST(RuntimeTest, ExecutesBellCircuitAndPopulatesCounts) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);

    menos::plan::JobSpec job_spec{
        .circuit = circuit,
        .shots = 1000,
        .want_statevector = true,
        .seed = 42,
        .memory_budget_bytes = std::nullopt,
        .expectation = std::nullopt,
    };
    const menos::plan::HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = menos::plan::plan_job(job_spec, hw);

    const Runtime rt;
    const Result result = rt.execute(plan, job_spec);

    EXPECT_TRUE(result.exact());
    ASSERT_TRUE(result.statevector().has_value());
    // ASSERT_TRUE above already guarantees this (it returns out of the
    // test on failure); clang-tidy can't see through the gtest macro.
    EXPECT_EQ(result.statevector()->size(), 4U);  // NOLINT(bugprone-unchecked-optional-access)

    std::uint64_t total = 0;
    const BitString zero = make_bitstring(2, 0);
    const BitString three = make_bitstring(2, 3);
    for (const auto& [key, count] : result.counts()) {
        EXPECT_TRUE(key == zero || key == three) << "Bell state should only measure |00> or |11>";
        total += count;
    }
    EXPECT_EQ(total, 1000U);
    EXPECT_GE(result.execute_seconds(), 0.0);
    EXPECT_GE(result.sample_seconds(), 0.0);
    EXPECT_EQ(result.actual_state_bytes(), 4U * sizeof(std::complex<double>));

    EXPECT_EQ(result.plan().strategy, Strategy::DirectDense);
    EXPECT_TRUE(result.plan().exact);
    ASSERT_EQ(result.plan().regions.size(), 1U);
    EXPECT_EQ(result.plan().regions[0].local_to_global, (std::vector<QubitId>{0, 1}));
    EXPECT_EQ(result.plan().regions[0].engine, Engine::Dense);
    EXPECT_EQ(result.plan().regions[0].backend, Backend::Cpu);
}

TEST(RuntimeTest, NoShotsProducesEmptyCountsAndZeroSampleTime) {
    Circuit circuit(1, 0);
    circuit.append(Opcode::H, {0});

    menos::plan::JobSpec job_spec{
        .circuit = circuit,
        .shots = 0,
        .want_statevector = true,
        .seed = 1,
        .memory_budget_bytes = std::nullopt,
        .expectation = std::nullopt,
    };
    const menos::plan::HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = menos::plan::plan_job(job_spec, hw);

    const Runtime rt;
    const Result result = rt.execute(plan, job_spec);

    EXPECT_TRUE(result.counts().empty());
    EXPECT_EQ(result.sample_seconds(), 0.0);
}

// Idle-qubit behavior end-to-end (docs/BUILD_PLAN.md Phase 3, 3.2 /
// 3.11 item 8): qubits {0,1} form a real Bell pair; qubit 2 is fully
// idle and unmeasured; qubit 3 is idle but measured -- its outcome must
// be deterministic |0> without allocating a full one-qubit state, and
// the statevector must still show qubit 2/3 as |0> at their global
// positions (never removed from the output shape).
TEST(RuntimeTest, IdleQubitsAreDeterministicZeroWithoutAllocation) {
    Circuit circuit(4, 3);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    circuit.append_measure(3, 2);  // idle qubit 3 -> clbit 2

    menos::plan::JobSpec job_spec{
        .circuit = circuit,
        .shots = 500,
        .want_statevector = true,
        .seed = 5,
        .memory_budget_bytes = std::nullopt,
        .expectation = std::nullopt,
    };
    const menos::plan::HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = menos::plan::plan_job(job_spec, hw);
    ASSERT_EQ(plan.strategy, Strategy::Factorized);  // {0,1} vs {2} vs {3}: 3 components

    const Runtime rt;
    const Result result = rt.execute(plan, job_spec);

    // clbit 2 (from idle qubit 3) is always 0.
    for (const auto& [key, count] : result.counts()) {
        EXPECT_FALSE(key.test(2)) << "idle qubit 3's measurement must always read 0";
    }

    // Statevector: nonzero only where q2=0 and q3=0 (both idle qubits
    // stay |0>), and among those, only where q0==q1 (the Bell pair).
    ASSERT_TRUE(result.statevector().has_value());
    const auto& sv = *result.statevector();  // NOLINT(bugprone-unchecked-optional-access)
    ASSERT_EQ(sv.size(), 16U);
    for (std::size_t a = 0; a < 16; ++a) {
        const bool q0 = (a >> 0) & 1;
        const bool q1 = (a >> 1) & 1;
        const bool q2 = (a >> 2) & 1;
        const bool q3 = (a >> 3) & 1;
        if (q2 || q3 || q0 != q1) {
            EXPECT_NEAR(std::abs(sv[a]), 0.0, 1e-12) << "index " << a;
        }
    }
}

// Peak-memory / region-release proof (docs/BUILD_PLAN.md Phase 3, 3.11
// item 14): 3 independent 5-qubit components (16*2^5=512 bytes each, sum
// 1536) -- if the runtime kept every region's state alive simultaneously,
// the genuinely-measured actual_state_bytes (added in this phase
// specifically so this is a real measurement, not a reuse of the
// planner's estimate) would reflect the sum; because each region's
// StateVector is a loop-scoped local, released before the next
// iteration, the true peak is only the single largest region (counts-
// only) or that region plus the global buffer (statevector).
Circuit make_three_independent_five_qubit_components() {
    Circuit circuit(15, 15);
    for (QubitId block = 0; block < 3; ++block) {
        const QubitId base = block * 5;
        for (QubitId i = 0; i < 4; ++i) {
            circuit.append(Opcode::CX,
                           {static_cast<QubitId>(base + i), static_cast<QubitId>(base + i + 1)});
        }
        for (QubitId i = 0; i < 5; ++i) {
            circuit.append_measure(static_cast<QubitId>(base + i), static_cast<QubitId>(base + i));
        }
    }
    return circuit;
}

// Pinned to the dense engine on one component with a non-Clifford T gate
// (docs/BUILD_PLAN.md Phase 4: golden/fixed-value tests over an
// all-Clifford circuit must be pinned, since the stabilizer engine now
// wins CX-only components on cost). Only one component needs pinning:
// the peak is a max over regions, and the dense-pinned 5-qubit region's
// 512 bytes still exceeds any of the other (now cheaper, stabilizer)
// regions' footprint, so the expected peak is unchanged. The T must
// come before block 0's own measurements (terminal-measurement-only).
Circuit make_three_independent_five_qubit_components_pinned_to_dense() {
    Circuit circuit(15, 15);
    for (QubitId block = 0; block < 3; ++block) {
        const QubitId base = block * 5;
        for (QubitId i = 0; i < 4; ++i) {
            circuit.append(Opcode::CX,
                           {static_cast<QubitId>(base + i), static_cast<QubitId>(base + i + 1)});
        }
        if (block == 0) {
            circuit.append(Opcode::T, {base});
        }
        for (QubitId i = 0; i < 5; ++i) {
            circuit.append_measure(static_cast<QubitId>(base + i), static_cast<QubitId>(base + i));
        }
    }
    return circuit;
}

TEST(RuntimeTest, CountsOnlyPeakBytesIsSingleRegionNotSumOfAllRegions) {
    const Circuit circuit = make_three_independent_five_qubit_components_pinned_to_dense();
    menos::plan::JobSpec job_spec{
        .circuit = circuit,
        .shots = 10,
        .want_statevector = false,
        .seed = 1,
        .memory_budget_bytes = 1'000'000,
        .expectation = std::nullopt,
    };

    // Forces Factorized directly (bypassing plan_job's own Direct-vs-
    // Factorized cost comparison), the same "hand-build a Plan" pattern
    // benchmarks/factorized_vs_direct_benchmark.cpp already uses to
    // test/measure execute_factorized in isolation: as of Phase 6, a
    // whole-circuit DirectMPS candidate can legitimately win over
    // Factorized for a product of several low-entanglement components
    // like this one (each inter-component cut has Schmidt rank exactly
    // 1, so MPS's own bond-based cost stays cheap for the whole 15-qubit
    // circuit) -- a correct planner outcome, not a bug, but this test's
    // own purpose is specifically execute_factorized's peak-bytes
    // accounting, independent of which strategy the planner would
    // otherwise choose.
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    Plan plan;
    plan.strategy = Strategy::Factorized;
    plan.exact = true;
    plan.regions = menos::plan::build_factorized_regions(circuit,
                                                         graph,
                                                         menos::plan::available_engines(),
                                                         /*want_statevector=*/false,
                                                         /*want_counts=*/true,
                                                         /*want_expectation=*/false,
                                                         /*observable=*/nullptr,
                                                         /*budget_bytes=*/1'000'000'000ULL,
                                                         {},
                                                         job_spec.shots);
    plan.reduced_circuit = circuit;
    ASSERT_EQ(plan.regions.size(), 3U);

    std::uint64_t largest_region = 0;
    std::uint64_t all_regions = 0;
    for (const Region& region : plan.regions) {
        largest_region = std::max(largest_region, region.estimated_state_bytes);
        all_regions += region.estimated_state_bytes;
    }

    const Runtime rt;
    const Result result = rt.execute(plan, job_spec);

    // actual_state_bytes() reports the largest single region's figure, not
    // the sum across all 3. (The memory the job really holds, with every
    // region's sampler kept until sampling, is Result::peak_host_bytes(),
    // docs/adr/0025.) Each region's figure is its engine's estimate, 16 * 2^5
    // for the state plus its buffers and sampler (docs/adr/0025 section 3.4).
    EXPECT_EQ(result.actual_state_bytes(), largest_region)
        << "peak must be one region's bytes, not the sum across all 3 regions (" << all_regions
        << ")";
    EXPECT_GE(largest_region, 16ULL * 32U);
}

TEST(RuntimeTest, StatevectorPeakBytesIsOneRegionPlusGlobalNotSumOfAllRegions) {
    const Circuit circuit = make_three_independent_five_qubit_components();
    menos::plan::JobSpec job_spec{
        .circuit = circuit,
        .shots = 0,
        .want_statevector = true,
        .seed = 1,
        .memory_budget_bytes = 10'000'000,
        .expectation = std::nullopt,
    };
    const menos::plan::HardwareProfile hw{.total_memory_bytes = 10'000'000};
    const Plan plan = menos::plan::plan_job(job_spec, hw);
    ASSERT_EQ(plan.strategy, Strategy::Factorized);

    std::uint64_t largest_region = 0;
    for (const Region& region : plan.regions) {
        largest_region = std::max(largest_region, region.estimated_state_bytes);
    }

    const Runtime rt;
    const Result result = rt.execute(plan, job_spec);

    constexpr std::uint64_t kGlobalBytes = 16ULL * (1ULL << 15);  // 16 * 2^15
    EXPECT_EQ(result.actual_state_bytes(), largest_region + kGlobalBytes)
        << "peak must be one region plus the global buffer, not the sum of all 3 regions plus "
           "the global buffer";
    EXPECT_GE(largest_region, 16ULL * 32U);  // 16 * 2^5, plus the region's buffers
}

// BitString migration (docs/BUILD_PLAN.md Phase 3, 3.8/3.11 item 10):
// 100 classical bits end-to-end through the full run() pipeline --
// rejected outright before this phase (kMaxClbits=64), must succeed now.
TEST(RuntimeTest, OneHundredClassicalBitsSucceedEndToEnd) {
    Circuit circuit(1, 100);
    circuit.append(Opcode::X, {0});
    circuit.append_measure(0, 99);  // the highest clbit

    menos::plan::JobSpec job_spec{
        .circuit = circuit,
        .shots = 5,
        .want_statevector = false,
        .seed = 1,
        .memory_budget_bytes = std::nullopt,
        .expectation = std::nullopt,
    };
    const menos::plan::HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = menos::plan::plan_job(job_spec, hw);
    const Runtime rt;
    const Result result = rt.execute(plan, job_spec);

    ASSERT_EQ(result.counts().size(), 1U);
    const BitString& key = result.counts().begin()->first;
    EXPECT_EQ(key.num_bits(), 100U);
    EXPECT_TRUE(key.test(99));
    for (std::uint32_t bit = 0; bit < 99; ++bit) {
        EXPECT_FALSE(key.test(bit)) << "bit " << bit;
    }
    EXPECT_EQ(result.counts().begin()->second, 5U);
}

}  // namespace
}  // namespace menos::runtime
