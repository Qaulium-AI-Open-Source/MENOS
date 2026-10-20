// End to end: run() plans a noisy job onto the density-matrix engine and runs
// it exactly (Phase 12, docs/adr/0016). Every pure-state engine stays listed
// as infeasible with its noise reason, a noiseless job never sees the
// density-matrix engine, and nothing is silently dropped.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/error.h"
#include "common/tolerance.h"
#include "menos/menos.h"

namespace menos {
namespace {

using Complex = std::complex<double>;
using Kraus = std::vector<std::vector<Complex>>;

Kraus depolarizing_1q(double p) {
    const double a = std::sqrt(1.0 - 3.0 * p / 4.0);
    const double b = std::sqrt(p / 4.0);
    return {{a, 0.0, 0.0, a},
            {0.0, b, b, 0.0},
            {0.0, Complex{0.0, -b}, Complex{0.0, b}, 0.0},
            {b, 0.0, 0.0, -b}};
}

Observable single_term(std::uint32_t n, std::uint64_t x, std::uint64_t z) {
    Observable observable;
    observable.terms.emplace_back(PauliString{n, {x}, {z}}, Complex{1.0, 0.0});
    return observable;
}

// width (a clbit count) and value (the bits to set) are not realistically
// confusable at any call site, as in run_test.cpp's make_bitstring.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
BitString bits(std::uint32_t width, std::uint64_t value) {
    BitString b(width);
    for (std::uint32_t i = 0; i < width; ++i) {
        b.set(i, ((value >> i) & 1U) != 0);
    }
    return b;
}

TEST(NoiseRunTest, ACountsJobWithAChannelRunsOnTheDensityMatrixEngine) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_channel({1}, depolarizing_1q(0.2));
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);

    const Result result = run(circuit, RunOptions{.shots = 4000, .seed = 1});
    EXPECT_EQ(result.plan().strategy, Strategy::DirectDensityMatrix);
    EXPECT_TRUE(result.exact());
    std::uint64_t total = 0;
    for (const auto& [outcome, count] : result.counts()) {
        total += count;
    }
    EXPECT_EQ(total, 4000U);
    // Depolarizing the second qubit of a Bell pair makes the odd-parity outcomes
    // possible: P(01) = P(10) = p/4 = 0.05 each, so about 400 of 4000.
    const std::uint64_t odd =
        (result.counts().contains(bits(2, 0b01)) ? result.counts().at(bits(2, 0b01)) : 0) +
        (result.counts().contains(bits(2, 0b10)) ? result.counts().at(bits(2, 0b10)) : 0);
    EXPECT_GT(odd, 300U);
    EXPECT_LT(odd, 500U);
}

// Every pure-state Direct candidate is still enumerated and infeasible with the
// noise reason, so explain() shows why each lost; the noise line reports it.
TEST(NoiseRunTest, ExplainShowsTheNoiseAndWhyEachPureEngineLost) {
    Circuit circuit(1, 1);
    circuit.append(Opcode::H, {0});
    circuit.append_channel({0}, depolarizing_1q(0.1));
    circuit.append_measure(0, 0, ReadoutError{{0.9, 0.1, 0.2, 0.8}});

    const Result result = run(circuit, RunOptions{.shots = 10, .seed = 1});
    const std::string report = result.explain();
    EXPECT_NE(report.find("noise      channels=1 readout_errors=1"), std::string::npos) << report;
    EXPECT_NE(report.find("direct_density_matrix"), std::string::npos) << report;
    EXPECT_NE(report.find("is a noise channel; this engine simulates pure states only"),
              std::string::npos)
        << report;
}

// A measurement with a certain bit flip and no gates. Before Phase 12's region
// fix such a qubit was idle and sampled as a constant 0; the flip must now be
// recorded on every shot.
TEST(NoiseRunTest, AReadoutErrorOnAnOtherwiseIdleQubitIsApplied) {
    Circuit circuit(1, 1);
    circuit.append_measure(0, 0, ReadoutError{{0.0, 1.0, 1.0, 0.0}});

    const Result result = run(circuit, RunOptions{.shots = 100, .seed = 1});
    ASSERT_EQ(result.counts().size(), 1U);
    EXPECT_EQ(result.counts().at(bits(1, 1)), 100U);
}

// A gate on qubit 0 and a readout error on qubit 1's separate component: the
// Factorized plan runs the pure region on a pure-state engine and the noisy
// region on the density-matrix engine, and the flip is recorded.
TEST(NoiseRunTest, AFactorizedPlanRunsOnlyTheNoisyRegionOnTheDensityMatrixEngine) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::X, {0});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1, ReadoutError{{0.0, 1.0, 1.0, 0.0}});

    const Result result = run(circuit, RunOptions{.shots = 200, .seed = 3});
    ASSERT_EQ(result.plan().strategy, Strategy::Factorized) << result.explain();
    bool saw_density_matrix = false;
    bool saw_pure = false;
    for (const Region& region : result.plan().regions) {
        if (region.engine == Engine::DensityMatrix) {
            saw_density_matrix = true;
        } else {
            saw_pure = true;
        }
    }
    EXPECT_TRUE(saw_density_matrix);
    EXPECT_TRUE(saw_pure);
    ASSERT_EQ(result.counts().size(), 1U);
    EXPECT_EQ(result.counts().at(bits(2, 0b11)), 200U);
}

// rho -> (1 - p) rho + p I/2 on |+> gives <X> = 1 - p, computed exactly.
TEST(NoiseRunTest, AnExpectationJobIsExact) {
    Circuit circuit(1, 0);
    circuit.append(Opcode::H, {0});
    circuit.append_channel({0}, depolarizing_1q(0.3));

    const Result result =
        run(circuit, RunOptions{.shots = 0, .seed = 1, .expectation = single_term(1, 1, 0)});
    ASSERT_TRUE(result.expectation_value().has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_NEAR(result.expectation_value()->real(), 0.7, common::kDefaultAbsTol);
    EXPECT_TRUE(result.exact());
}

// Two independent noisy components: the Factorized plan runs each on the
// density-matrix engine and multiplies the regions' exact values (the
// mixed-product rule holds because the joint state is rho_0 (x) rho_1). Amplitude
// damping 0.3 on |1> gives <Z0> = 2 * 0.3 - 1 = -0.4; depolarizing 0.2 on |+>
// gives <X1> = 0.8; so <Z0 X1> = -0.32 exactly. The cut search finds no plan for
// a circuit with a channel, so CutAndRecombine is never a candidate here.
TEST(NoiseRunTest, AFactorizedNoisyExpectationMultipliesExactRegionValues) {
    const double gamma = 0.3;
    Circuit circuit(2, 0);
    circuit.append(Opcode::X, {0});
    circuit.append_channel(
        {0}, {{1.0, 0.0, 0.0, std::sqrt(1.0 - gamma)}, {0.0, std::sqrt(gamma), 0.0, 0.0}});
    circuit.append(Opcode::H, {1});
    circuit.append_channel({1}, depolarizing_1q(0.2));

    const Result result =
        run(circuit, RunOptions{.shots = 0, .seed = 1, .expectation = single_term(2, 0b10, 0b01)});
    ASSERT_EQ(result.plan().strategy, Strategy::Factorized) << result.explain();
    for (const Region& region : result.plan().regions) {
        EXPECT_EQ(region.engine, Engine::DensityMatrix);
    }
    for (const CandidateSummary& candidate : result.plan().candidates) {
        EXPECT_NE(candidate.strategy, Strategy::CutAndRecombine);
    }
    ASSERT_TRUE(result.expectation_value().has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_NEAR(result.expectation_value()->real(), -0.32, common::kDefaultAbsTol);
    EXPECT_TRUE(result.exact());
}

// Noise acting only outside the light cone of the requested output cannot
// change it, so the job is planned as the noiseless job it is equivalent to --
// but the report still says the job carried noise.
TEST(NoiseRunTest, NoiseOutsideTheLightConeIsPlannedAsNoiseless) {
    Circuit circuit(2, 1);
    circuit.append(Opcode::X, {0});
    circuit.append_channel({1}, depolarizing_1q(0.5));
    circuit.append_measure(0, 0);

    const Result result = run(circuit, RunOptions{.shots = 50, .seed = 1});
    EXPECT_NE(result.plan().strategy, Strategy::DirectDensityMatrix);
    EXPECT_EQ(result.counts().at(bits(1, 1)), 50U);
    EXPECT_NE(result.explain().find("channels=1 readout_errors=0"), std::string::npos);
}

TEST(NoiseRunTest, TheSameSeedGivesTheSameNoisyCounts) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_channel({0}, depolarizing_1q(0.25));
    circuit.append_measure(0, 0, ReadoutError{{0.95, 0.05, 0.1, 0.9}});
    circuit.append_measure(1, 1);

    const RunOptions options{.shots = 1000, .seed = 77};
    EXPECT_EQ(run(circuit, options).counts(), run(circuit, options).counts());
}

// The density matrix needs 16 * 4^n bytes, checked against the memory budget
// like every other candidate: one byte under the estimate is refused, the
// estimate itself fits.
TEST(NoiseRunTest, TheMemoryBudgetIsEnforcedAtTheDensityMatrixSize) {
    Circuit circuit(3, 3);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::CX, {1, 2});
    circuit.append_channel({2}, depolarizing_1q(0.1));
    for (QubitId q = 0; q < 3; ++q) {
        circuit.append_measure(q, q);
    }
    // The plan's own figure (docs/adr/0025 section 3.4): at least rho
    // 16 * 4^3 = 1024 and the one-qubit superoperator 256, plus the gates'
    // buffers, the sampler and the runtime's bookkeeping, whose exact sizes
    // depend on the standard library.
    RunOptions options{.shots = 10, .seed = 1};
    const std::uint64_t peak = preview(circuit, options).estimated_state_bytes;
    ASSERT_GE(peak, 1024U + 256U);

    options.memory_budget_bytes = peak - 1;
    EXPECT_THROW(static_cast<void>(run(circuit, options)), common::ResourceExceeded);
    options.memory_budget_bytes = peak;
    EXPECT_EQ(run(circuit, options).plan().strategy, Strategy::DirectDensityMatrix);
}

}  // namespace
}  // namespace menos
