// The calibrated time model and the ranking rule (Phase 24 model fixes and
// item 4, docs/adr/0028 sections 20 and 21).

#include "plan/time_model.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "menos/circuit.h"
#include "menos/observable.h"
#include "menos/plan.h"
#include "menos/simulation_engine.h"
#include "plan/available_engines.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "plan/planner.h"

namespace menos::plan {
namespace {

using Counts = std::array<double, kTimeComponents>;

double predict(const Counts& counts, const Counts& rates) {
    double s = 0.0;
    for (std::size_t c = 0; c < kTimeComponents; ++c) {
        s += counts[c] * rates[c];
    }
    return s;
}

TEST(TimeModelFitTest, RecoversTheRatesThatProducedTheTimes) {
    const Counts rates{1.0e-6, 2.0e-9, 0.0, 5.0e-8};
    const std::vector<Counts> rows = {
        {1000.0, 1.0e6, 0.0, 10.0},
        {10.0, 1.0e9, 0.0, 100.0},
        {5000.0, 1.0e5, 0.0, 1.0e5},
        {200.0, 5.0e7, 0.0, 1.0e4},
    };
    std::vector<double> seconds;
    for (const Counts& r : rows) {
        seconds.push_back(predict(r, rates));
    }
    const Counts fitted = fit_seconds_per_count(rows, seconds);
    for (std::size_t c = 0; c < kTimeComponents; ++c) {
        EXPECT_NEAR(fitted[c], rates[c], 1e-9 * rates[0]) << "component " << c;
    }
    EXPECT_EQ(fitted[2], 0.0) << "a component zero in every row gets no rate";
}

TEST(TimeModelFitTest, NeverReturnsANegativeRate) {
    // Times that an unconstrained fit would explain with a negative second
    // component: more of component 1 makes a run faster.
    const std::vector<Counts> rows = {
        {1.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {1.0, 2.0, 0.0, 0.0}};
    const std::vector<double> seconds = {3.0, 2.0, 1.0};
    const Counts fitted = fit_seconds_per_count(rows, seconds);
    for (const double v : fitted) {
        EXPECT_GE(v, 0.0);
    }
    EXPECT_GT(fitted[0], 0.0);
}

PauliString z0(std::uint32_t n) {
    const std::size_t words = (static_cast<std::size_t>(n) + 63) / 64;
    PauliString p{n, std::vector<std::uint64_t>(words, 0), std::vector<std::uint64_t>(words, 0)};
    p.z_words[0] = 1;
    return p;
}

// A job each CPU engine can run, and the engine's own time terms on it.
struct EngineCase {
    Engine engine;
    Circuit circuit;
    bool counts;
    std::uint64_t trajectories;
};

std::vector<EngineCase> engine_cases() {
    Circuit general(3, 3);
    general.append(Opcode::H, {0});
    general.append(Opcode::CX, {0, 1});
    general.append(Opcode::RY, {2}, {0.4});
    general.append(Opcode::CX, {1, 2});
    Circuit clifford(3, 3);
    clifford.append(Opcode::H, {0});
    clifford.append(Opcode::CX, {0, 1});
    clifford.append(Opcode::S, {2});
    Circuit noisy = general;
    const std::complex<double> a{std::sqrt(0.99), 0.0};
    const std::complex<double> b{std::sqrt(0.01), 0.0};
    noisy.append_channel({1}, {{a, {0.0, 0.0}, {0.0, 0.0}, a}, {{0.0, 0.0}, b, b, {0.0, 0.0}}});
    return {
        {Engine::Dense, general, true, 0},
        {Engine::Stabilizer, clifford, true, 0},
        {Engine::Pauli, general, false, 0},
        {Engine::MPS, general, true, 0},
        {Engine::TensorNetwork, general, false, 0},
        {Engine::DensityMatrix, noisy, true, 0},
        {Engine::Trajectory, noisy, false, 4},
    };
}

const SimulationEngine* cpu_engine(Engine id) {
    if (id == Engine::Trajectory) {
        return &trajectory_engine();
    }
    for (const SimulationEngine* e : noisy_job_engines()) {
        if (e->engine_id() == id && e->backend() == Backend::Cpu && e->tier() == Tier::InMemory) {
            return e;
        }
    }
    return nullptr;
}

TEST(TimeTermsTest, EveryCpuEngineReportsItsOwnTermsAndNoOtherEngines) {
    for (const EngineCase& c : engine_cases()) {
        const SimulationEngine* engine = cpu_engine(c.engine);
        ASSERT_NE(engine, nullptr);
        const Observable observable{{{z0(c.circuit.num_qubits()), {1.0, 0.0}}}};
        const CostEstimate estimate = engine->estimate(EngineEstimateRequest{
            .num_qubits = c.circuit.num_qubits(),
            .operations = &c.circuit.operations(),
            .want_statevector = false,
            .want_counts = c.counts,
            .want_expectation = !c.counts,
            .observable = c.counts ? nullptr : &observable,
            .trajectories = c.trajectories,
            .shots = c.counts ? 100 : 0,
            .bitstring_width = c.counts ? c.circuit.num_qubits() : 0,
        });
        ASSERT_TRUE(estimate.feasible) << static_cast<int>(c.engine) << ": "
                                       << estimate.infeasible_reason;
        for (std::size_t e = 0; e < kTimeEngineCount; ++e) {
            double sum = 0.0;
            for (const double v : estimate.time.counts[e]) {
                EXPECT_GE(v, 0.0);
                sum += v;
            }
            if (e == static_cast<std::size_t>(c.engine)) {
                EXPECT_GT(sum, 0.0) << "engine " << e << " reported no time terms";
            } else {
                EXPECT_EQ(sum, 0.0) << "engine " << static_cast<int>(c.engine)
                                    << " reported terms for engine " << e;
            }
        }
    }
}

// Two disjoint Bell pairs, measured: a Factorized plan of two regions.
JobSpec two_block_counts_job() {
    Circuit circuit(4, 4);
    circuit.append(Opcode::RY, {0}, {0.3});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::RY, {2}, {0.7});
    circuit.append(Opcode::CX, {2, 3});
    for (std::uint32_t q = 0; q < 4; ++q) {
        circuit.append_measure(q, q);
    }
    return JobSpec{.circuit = circuit,
                   .shots = 64,
                   .want_statevector = false,
                   .seed = 0,
                   .memory_budget_bytes = std::uint64_t{1} << 30,
                   .expectation = std::nullopt};
}

TEST(TimeTermsTest, AFactorizedCandidateCarriesTheSumOfItsRegionsTerms) {
    const HardwareProfile hw{.total_memory_bytes = std::uint64_t{1} << 30};
    const Plan plan = plan_job(two_block_counts_job(), hw);
    ASSERT_EQ(plan.strategy, Strategy::Factorized) << plan.explain();
    const CandidateSummary* factorized = nullptr;
    for (const CandidateSummary& c : plan.candidates) {
        if (c.strategy == Strategy::Factorized) {
            factorized = &c;
        }
    }
    ASSERT_NE(factorized, nullptr);
    TimeTerms sum;
    for (const Region& region : plan.regions) {
        for (std::size_t e = 0; e < kTimeEngineCount; ++e) {
            for (std::size_t c = 0; c < kTimeComponents; ++c) {
                sum.counts[e][c] += region.time.counts[e][c];
            }
        }
    }
    for (std::size_t e = 0; e < kTimeEngineCount; ++e) {
        for (std::size_t c = 0; c < kTimeComponents; ++c) {
            EXPECT_DOUBLE_EQ(factorized->time.counts[e][c], sum.counts[e][c])
                << "engine " << e << " component " << c;
        }
    }
}

// A Clifford counts job: the stabilizer candidate has the fewest work units.
JobSpec clifford_counts_job() {
    Circuit circuit(4, 4);
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        for (std::uint32_t q = 0; q < 4; ++q) {
            circuit.append(Opcode::H, {q});
        }
        circuit.append(Opcode::CX, {0, 1});
        circuit.append(Opcode::CX, {2, 3});
    }
    for (std::uint32_t q = 0; q < 4; ++q) {
        circuit.append_measure(q, q);
    }
    return JobSpec{.circuit = circuit,
                   .shots = 16,
                   .want_statevector = false,
                   .seed = 0,
                   .memory_budget_bytes = std::uint64_t{1} << 30,
                   .expectation = std::nullopt};
}

// A profile whose time model makes the stabilizer slow and dense fast. The
// other engines have no figures, so they are priced at work units / 1.0,
// which is slower than either.
HardwareProfile slow_stabilizer_profile() {
    HardwareProfile hw{.total_memory_bytes = std::uint64_t{1} << 30,
                       .memory_bandwidth_bytes_per_second = 1.0e300,
                       .work_units_per_second = 1.0};
    hw.seconds_per_time_count[static_cast<std::size_t>(Engine::Dense)] = {1e-12, 1e-12, 1e-12, 0};
    hw.seconds_per_time_count[static_cast<std::size_t>(Engine::Stabilizer)] = {1.0, 1.0, 1.0, 0};
    return hw;
}

TEST(RankingRuleTest, AJobWithNoTimeBudgetRanksByWorkUnits) {
    const Plan plan = plan_job(clifford_counts_job(), slow_stabilizer_profile());
    EXPECT_EQ(plan.strategy, Strategy::DirectStabilizer) << plan.explain();
}

TEST(RankingRuleTest, ABudgetedJobRanksByPredictedSeconds) {
    JobSpec job = clifford_counts_job();
    job.time_budget_seconds = 1.0e9;
    const Plan plan = plan_job(job, slow_stabilizer_profile());
    EXPECT_EQ(plan.strategy, Strategy::DirectDense) << plan.explain();
    const CandidateSummary* winner = nullptr;
    for (const CandidateSummary& c : plan.candidates) {
        if (c.is_winner) {
            winner = &c;
        }
    }
    ASSERT_NE(winner, nullptr);
    for (const CandidateSummary& c : plan.candidates) {
        if (c.feasible) {
            EXPECT_LE(winner->estimated_seconds, c.estimated_seconds);
        }
    }
}

TEST(TimeModelCalibrationTest, MeasuresEveryCpuEngineWithNonNegativeFigures) {
    const SecondsPerTimeCount model = calibrate_time_model();
    for (std::size_t e = 0; e < kTimeEngineCount; ++e) {
        double sum = 0.0;
        for (const double v : model[e]) {
            EXPECT_GE(v, 0.0) << "engine " << e;
            EXPECT_TRUE(std::isfinite(v)) << "engine " << e;
            sum += v;
        }
        EXPECT_GT(sum, 0.0) << "engine " << e << " was not calibrated";
    }
}

}  // namespace
}  // namespace menos::plan
