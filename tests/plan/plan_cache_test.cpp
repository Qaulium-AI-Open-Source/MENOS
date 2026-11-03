#include "plan/plan_cache.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <variant>
#include <vector>

#include "common/error.h"
#include "plan/planner.h"

namespace menos::plan {
namespace {

// A small QAOA-shaped circuit: H on every qubit, then p layers of RZZ on
// every pair, RZ on every qubit and RX on every qubit, with the given angles.
Circuit qaoa_circuit(std::uint32_t n, int layers, const std::vector<double>& angles) {
    Circuit circuit(n, 0);
    for (QubitId q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {q});
    }
    std::size_t a = 0;
    const auto next = [&]() { return angles[a++ % angles.size()]; };
    for (int layer = 0; layer < layers; ++layer) {
        for (QubitId i = 0; i < n; ++i) {
            for (QubitId j = i + 1; j < n; ++j) {
                circuit.append(Opcode::RZZ, {i, j}, {next()});
            }
        }
        for (QubitId q = 0; q < n; ++q) {
            circuit.append(Opcode::RZ, {q}, {next()});
        }
        for (QubitId q = 0; q < n; ++q) {
            circuit.append(Opcode::RX, {q}, {next()});
        }
    }
    return circuit;
}

Observable zz_chain(std::uint32_t n) {
    Observable obs;
    for (QubitId i = 0; i + 1 < n; ++i) {
        PauliString p;
        p.num_qubits = n;
        p.x_words = {0};
        p.z_words = {(std::uint64_t{1} << i) | (std::uint64_t{1} << (i + 1))};
        obs.terms.emplace_back(p, std::complex<double>{0.5 + 0.1 * i, 0.0});
    }
    return obs;
}

JobSpec expectation_job(Circuit circuit, std::uint32_t n) {
    return JobSpec{
        .circuit = std::move(circuit),
        .shots = 0,
        .want_statevector = false,
        .seed = 7,
        .memory_budget_bytes = std::uint64_t{1} << 30,
        .expectation = zz_chain(n),
    };
}

const HardwareProfile kHw{.total_memory_bytes = std::uint64_t{1} << 32};

std::vector<double> gate_params(const Plan& plan) {
    std::vector<double> out;
    for (const Operation& op : plan.regions.at(0).operations) {
        if (const auto* g = std::get_if<GateOp>(&op)) {
            out.insert(out.end(), g->params.begin(), g->params.end());
        }
    }
    return out;
}

class PlanCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        plan_cache_clear();
    }
    void TearDown() override {
        plan_cache_clear();
    }
};

TEST_F(PlanCacheTest, SameStructureDifferentAnglesIsAHitWithTheNewAngles) {
    const JobSpec first = expectation_job(qaoa_circuit(4, 2, {0.3, 1.1, 2.0}), 4);
    const JobSpec second = expectation_job(qaoa_circuit(4, 2, {0.9, 0.4, 2.7, 1.6}), 4);

    const Plan a = plan_job_cached(first, kHw);
    const Plan b = plan_job_cached(second, kHw);
    const PlanCacheStats stats = plan_cache_stats();
    EXPECT_EQ(stats.misses, 1u);
    EXPECT_EQ(stats.hits, 1u);
    EXPECT_EQ(stats.bypasses, 0u);

    // The reused plan carries the SECOND job's angles ...
    EXPECT_EQ(gate_params(b), gate_params(plan_job(second, kHw)));
    EXPECT_NE(gate_params(b), gate_params(a));
    // ... and is otherwise the plan plan_job computes for it: same strategy,
    // same candidates with the same figures, same explain() text.
    const Plan fresh = plan_job(second, kHw);
    EXPECT_EQ(b.strategy, fresh.strategy);
    EXPECT_EQ(b.estimated_work_units, fresh.estimated_work_units);
    ASSERT_EQ(b.candidates.size(), fresh.candidates.size());
    for (std::size_t i = 0; i < b.candidates.size(); ++i) {
        EXPECT_EQ(b.candidates[i].feasible, fresh.candidates[i].feasible) << i;
        EXPECT_EQ(b.candidates[i].estimated_work_units, fresh.candidates[i].estimated_work_units)
            << i;
        EXPECT_EQ(b.candidates[i].is_winner, fresh.candidates[i].is_winner) << i;
    }
    EXPECT_EQ(b.explain(), fresh.explain());
    ASSERT_TRUE(b.reduced_circuit.has_value());
    EXPECT_EQ(b.reduced_circuit->operations().size(), fresh.reduced_circuit->operations().size());
}

TEST_F(PlanCacheTest, DifferentStructureOrOptionsIsAMiss) {
    const JobSpec base = expectation_job(qaoa_circuit(4, 1, {0.3}), 4);
    (void)plan_job_cached(base, kHw);

    JobSpec deeper = expectation_job(qaoa_circuit(4, 2, {0.3}), 4);
    (void)plan_job_cached(deeper, kHw);
    EXPECT_EQ(plan_cache_stats().misses, 2u);

    JobSpec other_coefficients = expectation_job(qaoa_circuit(4, 1, {0.3}), 4);
    other_coefficients.expectation->terms[0].second = {9.0, 0.0};
    (void)plan_job_cached(other_coefficients, kHw);
    EXPECT_EQ(plan_cache_stats().misses, 3u);

    JobSpec other_threads = expectation_job(qaoa_circuit(4, 1, {0.3}), 4);
    other_threads.max_threads = base.max_threads + 1;
    (void)plan_job_cached(other_threads, kHw);
    EXPECT_EQ(plan_cache_stats().misses, 4u);

    JobSpec with_counts = expectation_job(qaoa_circuit(4, 1, {0.3}), 4);
    with_counts.shots = 100;
    (void)plan_job_cached(with_counts, kHw);
    EXPECT_EQ(plan_cache_stats().misses, 5u);
    EXPECT_EQ(plan_cache_stats().hits, 0u);

    // The seed is not part of the key: planning never reads it.
    JobSpec other_seed = expectation_job(qaoa_circuit(4, 1, {0.3}), 4);
    other_seed.seed = 12345;
    (void)plan_job_cached(other_seed, kHw);
    EXPECT_EQ(plan_cache_stats().hits, 1u);
}

TEST_F(PlanCacheTest, SpecialAnglesAndBudgetedJobsBypassTheCache) {
    // RZZ(0) is the identity (operator Schmidt rank 1, not 2): never cached.
    const JobSpec identity_rzz = expectation_job(qaoa_circuit(4, 1, {0.0, 0.5}), 4);
    (void)plan_job_cached(identity_rzz, kHw);
    (void)plan_job_cached(identity_rzz, kHw);
    EXPECT_EQ(plan_cache_stats().bypasses, 2u);
    EXPECT_EQ(plan_cache_stats().hits, 0u);
    EXPECT_FALSE(plan_cache_key(identity_rzz).has_value());

    // RZZ(pi): -i Z.Z, rank 1 as well. RZZ(2 pi) = -I. Both bypass.
    EXPECT_FALSE(
        plan_cache_key(expectation_job(qaoa_circuit(4, 1, {std::numbers::pi}), 4)).has_value());
    EXPECT_FALSE(
        plan_cache_key(expectation_job(qaoa_circuit(4, 1, {2 * std::numbers::pi}), 4)).has_value());
    // Just off the special angle: cacheable.
    EXPECT_TRUE(plan_cache_key(expectation_job(qaoa_circuit(4, 1, {1e-6}), 4)).has_value());

    JobSpec budgeted = expectation_job(qaoa_circuit(4, 1, {0.3}), 4);
    budgeted.time_budget_seconds = 10.0;
    EXPECT_FALSE(plan_cache_key(budgeted).has_value());

    // CP(0) is the identity too; CP(0.7) is not.
    Circuit cp0(2, 0);
    cp0.append(Opcode::H, {0});
    cp0.append(Opcode::CP, {0, 1}, {0.0});
    EXPECT_FALSE(plan_cache_key(expectation_job(cp0, 2)).has_value());
    Circuit cp(2, 0);
    cp.append(Opcode::H, {0});
    cp.append(Opcode::CP, {0, 1}, {0.7});
    EXPECT_TRUE(plan_cache_key(expectation_job(cp, 2)).has_value());
}

TEST_F(PlanCacheTest, AnExplicitUnitaryIsPartOfTheStructure) {
    const std::vector<std::complex<double>> a = {{1, 0}, {0, 0}, {0, 0}, {0, 1}};
    const std::vector<std::complex<double>> b = {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};
    // CX makes the two qubits one interaction component, so the winner is a
    // Direct plan (a Factorized winner is never stored).
    Circuit ca(2, 0);
    ca.append(Opcode::H, {0});
    ca.append(Opcode::CX, {0, 1});
    ca.append_unitary({1}, a);
    Circuit cb(2, 0);
    cb.append(Opcode::H, {0});
    cb.append(Opcode::CX, {0, 1});
    cb.append_unitary({1}, b);
    (void)plan_job_cached(expectation_job(ca, 2), kHw);
    (void)plan_job_cached(expectation_job(cb, 2), kHw);
    (void)plan_job_cached(expectation_job(ca, 2), kHw);
    EXPECT_EQ(plan_cache_stats().misses, 2u);
    EXPECT_EQ(plan_cache_stats().hits, 1u);
}

TEST_F(PlanCacheTest, EvictsLeastRecentlyUsedBeyondCapacity) {
    for (std::uint32_t i = 0; i < kPlanCacheCapacity + 1; ++i) {
        // A different structure each time: i + 1 layers of a 2-qubit circuit.
        (void)plan_job_cached(expectation_job(qaoa_circuit(2, static_cast<int>(i) + 1, {0.3}), 2),
                              kHw);
    }
    EXPECT_EQ(plan_cache_stats().misses, kPlanCacheCapacity + 1);
    // The first structure was evicted; the last one is still there.
    (void)plan_job_cached(expectation_job(qaoa_circuit(2, 1, {0.4}), 2), kHw);
    EXPECT_EQ(plan_cache_stats().misses, kPlanCacheCapacity + 2);
    (void)plan_job_cached(
        expectation_job(qaoa_circuit(2, static_cast<int>(kPlanCacheCapacity) + 1, {0.4}), 2), kHw);
    EXPECT_EQ(plan_cache_stats().hits, 1u);
}

TEST_F(PlanCacheTest, RandomJobsAlwaysMatchPlanJob) {
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> angle(0.05, 3.0);
    for (int trial = 0; trial < 30; ++trial) {
        const std::uint32_t n = 2 + static_cast<std::uint32_t>(rng() % 4);
        const int layers = 1 + static_cast<int>(rng() % 2);
        std::vector<double> first_angles(5);
        std::vector<double> second_angles(5);
        for (double& v : first_angles) {
            v = angle(rng);
        }
        for (double& v : second_angles) {
            v = angle(rng);
        }
        const JobSpec first = expectation_job(qaoa_circuit(n, layers, first_angles), n);
        const JobSpec second = expectation_job(qaoa_circuit(n, layers, second_angles), n);
        (void)plan_job_cached(first, kHw);
        const Plan cached = plan_job_cached(second, kHw);
        const Plan fresh = plan_job(second, kHw);
        ASSERT_EQ(cached.explain(), fresh.explain()) << "n=" << n << " layers=" << layers;
        ASSERT_EQ(gate_params(cached), gate_params(fresh));
    }
}

}  // namespace
}  // namespace menos::plan
