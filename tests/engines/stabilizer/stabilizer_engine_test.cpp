#include "engines/stabilizer/stabilizer_engine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <map>
#include <random>
#include <set>

#include "engines/dense/dense_engine.h"
#include "engines/stabilizer/tableau.h"

namespace menos::engines::stabilizer {
namespace {

using engines::dense::DenseEngine;
using MeasureTarget = std::pair<QubitId, ClbitId>;

CostEstimate stabilizer_estimate(const StabilizerEngine& engine,
                                 std::uint32_t num_qubits,
                                 const std::vector<Operation>& ops,
                                 bool want_statevector,
                                 bool want_counts) {
    const EngineEstimateRequest request{
        .num_qubits = num_qubits,
        .operations = &ops,
        .want_statevector = want_statevector,
        .want_counts = want_counts,
        .want_expectation = false,
        .observable = nullptr,
    };
    return engine.estimate(request);
}

TEST(StabilizerEngineTest, NameAndCapabilities) {
    const StabilizerEngine engine;
    EXPECT_EQ(engine.name(), "stabilizer");
    EXPECT_EQ(engine.engine_id(), Engine::Stabilizer);
    const EngineCapabilities caps = engine.capabilities();
    EXPECT_TRUE(caps.counts);
    EXPECT_FALSE(caps.statevector);
    EXPECT_TRUE(caps.expectation);
}

TEST(StabilizerEngineTest, StatevectorRequestIsInfeasibleNamingDense) {
    const StabilizerEngine engine;
    const CostEstimate estimate =
        stabilizer_estimate(engine, 2, {}, /*want_statevector=*/true, false);
    EXPECT_FALSE(estimate.feasible);
    EXPECT_NE(estimate.infeasible_reason.find("dense"), std::string::npos);
}

TEST(StabilizerEngineTest, NonCliffordOperationIsInfeasibleNamingIndexAndGate) {
    const StabilizerEngine engine;
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::T, {0}, {}},
    };
    const CostEstimate estimate = stabilizer_estimate(engine, 1, ops, false, true);
    EXPECT_FALSE(estimate.feasible);
    EXPECT_NE(estimate.infeasible_reason.find("operation 1"), std::string::npos);
    EXPECT_NE(estimate.infeasible_reason.find('t'), std::string::npos);
}

TEST(StabilizerEngineTest, FiveHundredQubitCliffordCircuitIsFeasible) {
    // The whole point of the stabilizer engine: a scale dense cannot
    // even represent (16 * 2^500 bytes) is trivially cheap here, because
    // the cost formula is polynomial in num_qubits, never checked_pow2.
    Circuit circuit(500, 500);
    for (QubitId q = 0; q < 500; ++q) {
        circuit.append(Opcode::H, {q});
    }
    for (QubitId q = 0; q + 1 < 500; q += 2) {
        circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    for (QubitId q = 0; q < 500; ++q) {
        circuit.append_measure(q, q);
    }
    const StabilizerEngine engine;
    const CostEstimate estimate = stabilizer_estimate(
        engine, 500, circuit.operations(), /*want_statevector=*/false, /*want_counts=*/true);
    ASSERT_TRUE(estimate.feasible);
    EXPECT_LT(estimate.estimated_peak_bytes, 1'000'000U)
        << "the tableau for 500 qubits must be kilobytes, not 16*2^500 bytes";
}

TEST(StabilizerEngineTest, SameSeedGivesIdenticalCountsAcrossRuns) {
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::CX, {0, 1}, {}},
    };
    const std::vector<MeasureTarget> targets{{0, 0}, {1, 1}};
    const StabilizerEngine engine;
    const EngineExecutionRequest request{
        .num_qubits = 2,
        .operations = ops,
        .want_statevector = false,
        .want_counts = true,
        .want_expectation = false,
        .shots = 2000,
        .measure_targets = targets,
        .bitstring_width = 2,
        .seed = 2024,
        .global_phase = 0.0,
        .budget_bytes = 1'000'000,
        .is_sole_region = true,
        .observable = nullptr,
    };
    const EngineExecutionResult a = engine.execute(request);
    const EngineExecutionResult b = engine.execute(request);
    ASSERT_TRUE(a.counts.has_value());
    ASSERT_TRUE(b.counts.has_value());
    // ASSERT_TRUE above already guarantees this; clang-tidy can't see
    // through the gtest macro.
    EXPECT_EQ(*a.counts, *b.counts);  // NOLINT(bugprone-unchecked-optional-access)
}

TEST(StabilizerEngineTest, BellCircuitCountsAreOnlyCorrelatedOutcomes) {
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::CX, {0, 1}, {}},
    };
    const std::vector<MeasureTarget> targets{{0, 0}, {1, 1}};
    const StabilizerEngine engine;
    const EngineExecutionRequest request{
        .num_qubits = 2,
        .operations = ops,
        .want_statevector = false,
        .want_counts = true,
        .want_expectation = false,
        .shots = 1000,
        .measure_targets = targets,
        .bitstring_width = 2,
        .seed = 42,
        .global_phase = 0.0,
        .budget_bytes = 1'000'000,
        .is_sole_region = true,
        .observable = nullptr,
    };
    const EngineExecutionResult result = engine.execute(request);
    ASSERT_TRUE(result.counts.has_value());
    std::uint64_t total = 0;
    // ASSERT_TRUE above already guarantees this; clang-tidy can't see
    // through the gtest macro.
    for (const auto& [key, count] : *result.counts) {  // NOLINT(bugprone-unchecked-optional-access)
        EXPECT_EQ(key.test(0), key.test(1));
        total += count;
    }
    EXPECT_EQ(total, 1000U);
}

// --- Random Clifford differential test against the dense oracle
// (docs/BUILD_PLAN.md Phase 4 acceptance criterion 1) ---

BitString project_outcome(const std::vector<bool>& full,
                          const std::vector<MeasureTarget>& targets,
                          std::uint32_t width) {
    BitString bs(width);
    for (const auto& [qubit, clbit] : targets) {
        if (full[qubit]) {
            bs.set(clbit, true);
        }
    }
    return bs;
}

// All 2^generators.size() reachable full-width outcomes, projected onto
// targets and deduplicated -- feasible since generators.size() <= 12
// (at most 4096 subsets) for every circuit this test generates.
std::set<BitString> reachable_projected_outcomes(const std::vector<bool>& base,
                                                 const std::vector<std::vector<bool>>& generators,
                                                 const std::vector<MeasureTarget>& targets,
                                                 std::uint32_t width) {
    std::set<BitString> outcomes;
    const std::size_t r = generators.size();
    const std::uint64_t subsets = std::uint64_t{1} << r;
    for (std::uint64_t mask = 0; mask < subsets; ++mask) {
        std::vector<bool> full = base;
        for (std::size_t g = 0; g < r; ++g) {
            if (((mask >> g) & 1U) != 0) {
                for (std::size_t q = 0; q < full.size(); ++q) {
                    full[q] = full[q] != generators[g][q];
                }
            }
        }
        outcomes.insert(project_outcome(full, targets, width));
    }
    return outcomes;
}

// Exact Born-rule probability of each projected outcome, from the dense
// statevector (the independent oracle every other engine is validated
// against, docs/BUILD_PLAN.md Phase 4 acceptance criterion 1: "compare
// against the dense oracle").
std::map<BitString, double> dense_probabilities(const Circuit& circuit,
                                                const std::vector<MeasureTarget>& targets,
                                                std::uint32_t width) {
    const DenseEngine dense;
    const EngineExecutionRequest request{
        .num_qubits = circuit.num_qubits(),
        .operations = circuit.operations(),
        .want_statevector = true,
        .want_counts = false,
        .want_expectation = false,
        .shots = 0,
        .measure_targets = targets,
        .bitstring_width = width,
        .seed = 0,
        .global_phase = circuit.global_phase,
        .budget_bytes = 1ULL << 30,
        .is_sole_region = true,
        .observable = nullptr,
    };
    const EngineExecutionResult result = dense.execute(request);
    // want_statevector is fixed true above, and DenseEngine::execute()
    // always populates it in that case; clang-tidy can't see that.
    const std::vector<std::complex<double>>& amplitudes =
        result.statevector.value();  // NOLINT(bugprone-unchecked-optional-access)
    std::map<BitString, double> probabilities;
    for (std::size_t basis = 0; basis < amplitudes.size(); ++basis) {
        const double p = std::norm(amplitudes[basis]);
        if (p == 0.0) {
            continue;
        }
        BitString outcome(width);
        for (const auto& [qubit, clbit] : targets) {
            if ((basis >> qubit) & std::size_t{1}) {
                outcome.set(clbit, true);
            }
        }
        probabilities[outcome] += p;
    }
    return probabilities;
}

std::map<ClbitId, QubitId> last_writes(const std::vector<Operation>& operations) {
    std::map<ClbitId, QubitId> writes;
    for (const Operation& op : operations) {
        if (const auto* measure = std::get_if<MeasureOp>(&op)) {
            writes[measure->clbit] = measure->qubit;
        }
    }
    return writes;
}

std::uint32_t random_below(std::mt19937_64& rng, std::uint32_t bound) {
    return static_cast<std::uint32_t>(rng() % bound);
}

// The counts sampler flips a generator INTO the base outcome: XOR, never OR.
// X(0), H(1), CX(1, 0) leaves (q0, q1) in {(1, 0), (0, 1)}: the base outcome
// has q0 = 1 and the one generator flips both qubits, so an OR could never
// clear q0 and would report the impossible (1, 1) half the time (the Phase
// 24 stabilizer sampler bug, docs/adr/0028 section 18).
TEST(StabilizerEngineTest, AnAntiCorrelatedPairSamplesOnlyItsTwoOutcomes) {
    const std::vector<Operation> ops{
        GateOp{Opcode::X, {0}, {}},
        GateOp{Opcode::H, {1}, {}},
        GateOp{Opcode::CX, {1, 0}, {}},
    };
    const std::vector<MeasureTarget> targets{{0, 0}, {1, 1}};
    const StabilizerEngine engine;
    const EngineExecutionRequest request{
        .num_qubits = 2,
        .operations = ops,
        .want_statevector = false,
        .want_counts = true,
        .want_expectation = false,
        .shots = 1000,
        .measure_targets = targets,
        .bitstring_width = 2,
        .seed = 42,
        .global_phase = 0.0,
        .budget_bytes = 1'000'000,
        .is_sole_region = true,
        .observable = nullptr,
    };
    const EngineExecutionResult result = engine.execute(request);
    ASSERT_TRUE(result.counts.has_value());
    std::uint64_t total = 0;
    std::uint64_t q0_set = 0;
    for (const auto& [key, count] : *result.counts) {  // NOLINT(bugprone-unchecked-optional-access)
        EXPECT_NE(key.test(0), key.test(1)) << key.to_hex_string();
        total += count;
        if (key.test(0)) {
            q0_set += count;
        }
    }
    EXPECT_EQ(total, 1000U);
    // Both outcomes appear: q0 is 1 in about half the shots, not all of them.
    EXPECT_GT(q0_set, 400U);
    EXPECT_LT(q0_set, 600U);
}

// The owner's 50-qubit reproducer (docs/benchmarks/cross-simulator-2026-11-05/
// harness/stabilizer_repro.{py,json}, 2026-11-05): after these 54 gates
// qubits 48 and 49 are perfectly anticorrelated, and MENOS reported qubit 48
// as 1 in every shot.
TEST(StabilizerEngineTest, TheOwnersFiftyQubitReproducerSamplesOnlyAnticorrelatedOutcomes) {
    struct Gate {
        Opcode opcode;
        std::vector<QubitId> qubits;
    };
    const std::vector<Gate> gates{
        {Opcode::CX, {35, 36}}, {Opcode::CZ, {37, 38}}, {Opcode::CZ, {39, 40}},
        {Opcode::CZ, {41, 42}}, {Opcode::CZ, {43, 44}}, {Opcode::CZ, {45, 46}},
        {Opcode::CZ, {47, 48}}, {Opcode::H, {48}},      {Opcode::H, {49}},
        {Opcode::CZ, {0, 1}},   {Opcode::CX, {2, 3}},   {Opcode::CX, {4, 5}},
        {Opcode::CX, {6, 7}},   {Opcode::CX, {8, 9}},   {Opcode::CZ, {10, 11}},
        {Opcode::CZ, {12, 13}}, {Opcode::CZ, {14, 15}}, {Opcode::CX, {16, 17}},
        {Opcode::CZ, {18, 19}}, {Opcode::CX, {20, 21}}, {Opcode::CX, {22, 23}},
        {Opcode::CZ, {24, 25}}, {Opcode::CZ, {26, 27}}, {Opcode::CX, {28, 29}},
        {Opcode::CX, {30, 31}}, {Opcode::CX, {32, 33}}, {Opcode::CX, {34, 35}},
        {Opcode::CX, {36, 37}}, {Opcode::CX, {38, 39}}, {Opcode::CX, {40, 41}},
        {Opcode::CZ, {42, 43}}, {Opcode::CX, {44, 45}}, {Opcode::CX, {46, 47}},
        {Opcode::CZ, {48, 49}}, {Opcode::H, {48}},      {Opcode::X, {49}},
        {Opcode::CX, {1, 2}},   {Opcode::CZ, {3, 4}},   {Opcode::CZ, {5, 6}},
        {Opcode::CX, {7, 8}},   {Opcode::CX, {9, 10}},  {Opcode::CX, {11, 12}},
        {Opcode::CX, {13, 14}}, {Opcode::CZ, {15, 16}}, {Opcode::CX, {17, 18}},
        {Opcode::CZ, {19, 20}}, {Opcode::CX, {21, 22}}, {Opcode::CZ, {23, 24}},
        {Opcode::CX, {25, 26}}, {Opcode::CZ, {27, 28}}, {Opcode::CZ, {29, 30}},
        {Opcode::CX, {31, 32}}, {Opcode::CX, {33, 34}}, {Opcode::CX, {37, 38}},
    };
    std::vector<Operation> ops;
    ops.reserve(gates.size());
    for (const Gate& gate : gates) {
        ops.emplace_back(GateOp{gate.opcode, gate.qubits, {}});
    }
    const std::vector<MeasureTarget> targets{{48, 0}, {49, 1}};
    const StabilizerEngine engine;
    const EngineExecutionRequest request{
        .num_qubits = 50,
        .operations = ops,
        .want_statevector = false,
        .want_counts = true,
        .want_expectation = false,
        .shots = 1024,
        .measure_targets = targets,
        .bitstring_width = 2,
        .seed = 1,
        .global_phase = 0.0,
        .budget_bytes = 1'000'000,
        .is_sole_region = true,
        .observable = nullptr,
    };
    const EngineExecutionResult result = engine.execute(request);
    ASSERT_TRUE(result.counts.has_value());
    std::uint64_t total = 0;
    std::uint64_t q48_set = 0;
    for (const auto& [key, count] : *result.counts) {  // NOLINT(bugprone-unchecked-optional-access)
        EXPECT_NE(key.test(0), key.test(1)) << key.to_hex_string();
        total += count;
        if (key.test(0)) {
            q48_set += count;
        }
    }
    EXPECT_EQ(total, 1024U);
    EXPECT_GT(q48_set, 400U);
    EXPECT_LT(q48_set, 624U);
}

// A random Clifford circuit over the full Clifford set (docs/BUILD_PLAN.md
// Phase 4: "the full Clifford set"), with a subset of active qubits
// (leaving the rest genuinely idle) and a subset of measured qubits
// (some clbits reused so a later measurement overwrites an earlier
// one -- "repeated writes to one classical bit").
// seed (an arbitrary 64-bit RNG seed) and n (a qubit count, 1-12 in this
// test) are not realistically confusable at any call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Circuit make_random_clifford_circuit(std::uint64_t seed, std::uint32_t n) {
    std::mt19937_64 rng(seed);
    Circuit circuit(n, n);

    const std::uint32_t num_active = std::max<std::uint32_t>(1, n - random_below(rng, n / 2 + 1));
    std::vector<QubitId> active(num_active);
    for (std::uint32_t i = 0; i < num_active; ++i) {
        active[i] = i;
    }

    const std::array<Opcode, 6> one_qubit{
        Opcode::X, Opcode::Y, Opcode::Z, Opcode::H, Opcode::S, Opcode::Sdg};
    const std::array<Opcode, 5> one_qubit_more{
        Opcode::SX, Opcode::SXdg, Opcode::X, Opcode::Y, Opcode::Z};
    const std::array<Opcode, 3> two_qubit{Opcode::CX, Opcode::CZ, Opcode::SWAP};

    const std::uint32_t num_gates = 5 + random_below(rng, 20);
    for (std::uint32_t i = 0; i < num_gates; ++i) {
        if (num_active >= 2 && random_below(rng, 2) == 0) {
            const QubitId a = active[random_below(rng, num_active)];
            QubitId b = active[random_below(rng, num_active)];
            while (b == a) {
                b = active[random_below(rng, num_active)];
            }
            circuit.append(
                two_qubit[random_below(rng, static_cast<std::uint32_t>(two_qubit.size()))], {a, b});
        } else {
            const QubitId a = active[random_below(rng, num_active)];
            const Opcode op =
                (random_below(rng, 2) == 0)
                    ? one_qubit[random_below(rng, static_cast<std::uint32_t>(one_qubit.size()))]
                    : one_qubit_more[random_below(
                          rng, static_cast<std::uint32_t>(one_qubit_more.size()))];
            circuit.append(op, {a});
        }
    }

    // Measure every qubit (active or idle) to clbit == qubit index, then
    // sometimes overwrite an earlier clbit from a later qubit
    // ("repeated writes"), exactly mirroring CLAUDE.md's last-write-wins
    // invariant.
    for (QubitId q = 0; q < n; ++q) {
        circuit.append_measure(q, q);
    }
    if (n >= 2 && random_below(rng, 3) == 0) {
        const ClbitId reused_clbit = random_below(rng, n);
        const QubitId overwriting_qubit = random_below(rng, n);
        circuit.append_measure(overwriting_qubit, reused_clbit);
    }

    return circuit;
}

TEST(StabilizerEngineDifferentialTest, MatchesDenseOracleAcrossRandomCliffordCircuits) {
    constexpr std::uint64_t kSeed = 20260922;
    std::mt19937_64 seed_source(kSeed);
    const StabilizerEngine stabilizer;

    for (std::uint32_t n = 1; n <= 12; ++n) {
        for (int trial = 0; trial < 3; ++trial) {
            const std::uint64_t circuit_seed = seed_source();
            const Circuit circuit = make_random_clifford_circuit(circuit_seed, n);

            const std::map<ClbitId, QubitId> writes = last_writes(circuit.operations());
            std::vector<MeasureTarget> targets;
            targets.reserve(writes.size());
            for (const auto& [clbit, qubit] : writes) {
                targets.emplace_back(qubit, clbit);
            }

            Tableau tableau(n);
            for (const Operation& op : circuit.operations()) {
                if (const auto* gate = std::get_if<GateOp>(&op)) {
                    switch (gate->opcode) {
                        case Opcode::X:
                            tableau.apply_x(gate->qubits[0]);
                            break;
                        case Opcode::Y:
                            tableau.apply_y(gate->qubits[0]);
                            break;
                        case Opcode::Z:
                            tableau.apply_z(gate->qubits[0]);
                            break;
                        case Opcode::H:
                            tableau.apply_h(gate->qubits[0]);
                            break;
                        case Opcode::S:
                            tableau.apply_s(gate->qubits[0]);
                            break;
                        case Opcode::Sdg:
                            tableau.apply_sdg(gate->qubits[0]);
                            break;
                        case Opcode::SX:
                            tableau.apply_sx(gate->qubits[0]);
                            break;
                        case Opcode::SXdg:
                            tableau.apply_sxdg(gate->qubits[0]);
                            break;
                        case Opcode::CX:
                            tableau.apply_cx(gate->qubits[0], gate->qubits[1]);
                            break;
                        case Opcode::CZ:
                            tableau.apply_cz(gate->qubits[0], gate->qubits[1]);
                            break;
                        case Opcode::SWAP:
                            tableau.apply_swap(gate->qubits[0], gate->qubits[1]);
                            break;
                        default:
                            FAIL() << "generator produced a non-Clifford opcode; seed="
                                   << circuit_seed;
                    }
                }
            }

            const Tableau::AffineSubspace subspace = tableau.reduce();
            const std::set<BitString> reachable =
                reachable_projected_outcomes(subspace.base, subspace.generators, targets, n);
            const auto r_observed =
                static_cast<int>(std::round(std::log2(static_cast<double>(reachable.size()))));
            ASSERT_EQ(std::uint64_t{1} << r_observed, reachable.size())
                << "reachable-outcome count must be an exact power of two; seed=" << circuit_seed;

            const std::map<BitString, double> dense_probs =
                dense_probabilities(circuit, targets, n);

            std::set<BitString> dense_nonzero;
            for (const auto& [key, p] : dense_probs) {
                if (p > 1e-9) {
                    dense_nonzero.insert(key);
                }
            }

            EXPECT_EQ(dense_nonzero, reachable)
                << "seed=" << circuit_seed << " n=" << n
                << ": stabilizer's reachable outcome set must equal dense's nonzero outcome set";

            // The engine's own sampler, not only reduce(): every sampled outcome
            // must be one dense gives a nonzero probability (the sampler's OR
            // bug produced outcomes outside the set, docs/adr/0028 section 18).
            const EngineExecutionRequest request{
                .num_qubits = n,
                .operations = circuit.operations(),
                .want_statevector = false,
                .want_counts = true,
                .want_expectation = false,
                .shots = 256,
                .measure_targets = targets,
                .bitstring_width = n,
                .seed = circuit_seed,
                .global_phase = 0.0,
                .budget_bytes = 1'000'000,
                .is_sole_region = true,
                .observable = nullptr,
            };
            const EngineExecutionResult sampled = stabilizer.execute(request);
            ASSERT_TRUE(sampled.counts.has_value());
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            for (const auto& [outcome, count] : *sampled.counts) {
                static_cast<void>(count);
                EXPECT_TRUE(dense_nonzero.contains(outcome))
                    << "seed=" << circuit_seed << " n=" << n << ": the sampler drew "
                    << outcome.to_hex_string() << ", which dense gives probability 0";
            }

            const double expected_p = std::pow(2.0, -r_observed);
            for (const BitString& outcome : reachable) {
                const auto it = dense_probs.find(outcome);
                ASSERT_NE(it, dense_probs.end()) << "seed=" << circuit_seed;
                EXPECT_NEAR(it->second, expected_p, 1e-9) << "seed=" << circuit_seed << " n=" << n
                                                          << ": every reachable outcome must have "
                                                             "probability exactly 2^-r";
            }
        }
    }
}

}  // namespace
}  // namespace menos::engines::stabilizer
