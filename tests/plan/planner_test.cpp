#include "plan/planner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/error.h"
#include "menos/simulation_engine.h"

namespace menos::plan {
namespace {

JobSpec make_job(std::uint32_t num_qubits, std::optional<std::uint64_t> budget = std::nullopt) {
    return JobSpec{
        .circuit = Circuit(num_qubits, 0),
        .shots = 0,
        .want_statevector = true,
        .seed = 0,
        .memory_budget_bytes = budget,
        .expectation = std::nullopt,
    };
}

TEST(PlanJobTest, DirectDenseCoversAllQubitsWithinBudget) {
    const JobSpec job = make_job(3, /*budget=*/1'000'000);
    const HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, hw);

    EXPECT_EQ(plan.strategy, Strategy::DirectDense);
    EXPECT_TRUE(plan.exact);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].local_to_global, (std::vector<QubitId>{0, 1, 2}));
    EXPECT_EQ(plan.regions[0].engine, Engine::Dense);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cpu);
    EXPECT_EQ(plan.estimated_state_bytes, 8U * sizeof(std::complex<double>));
}

TEST(PlanJobTest, UsesExplicitBudgetOverDefault) {
    const JobSpec job =
        make_job(20, /*budget=*/16);  // 2^20 amplitudes needs far more than 16 bytes
    const HardwareProfile hw{.total_memory_bytes = 1ULL << 40};  // plenty, if the default were used
    EXPECT_THROW((void)plan_job(job, hw), common::ResourceExceeded);
}

TEST(PlanJobTest, UsesDefaultFractionOfHardwareProfileWhenBudgetUnset) {
    const JobSpec job = make_job(3);  // no explicit budget
    // 8 amplitudes * 16 bytes = 128 bytes required; 50% of 100 = 50 bytes, too small.
    const HardwareProfile hw{.total_memory_bytes = 100};
    EXPECT_THROW((void)plan_job(job, hw), common::ResourceExceeded);

    const HardwareProfile generous_hw{.total_memory_bytes = 1'000'000};
    EXPECT_NO_THROW((void)plan_job(job, generous_hw));
}

TEST(PlanJobTest, BudgetTooSmallThrowsResourceExceededBeforeAllocation) {
    const JobSpec job = make_job(30, /*budget=*/1024);
    const HardwareProfile hw{.total_memory_bytes = 1ULL << 40};
    EXPECT_THROW((void)plan_job(job, hw), common::ResourceExceeded);
}

TEST(PlanJobTest, SixtyThreeQubitsThrowsResourceExceeded) {
    const JobSpec job = make_job(63, /*budget=*/(1ULL << 62));
    const HardwareProfile hw{.total_memory_bytes = 1ULL << 62};
    EXPECT_THROW((void)plan_job(job, hw), common::ResourceExceeded);
}

TEST(PlanJobTest, SixtyFourQubitsThrowsResourceExceeded) {
    const JobSpec job = make_job(64, /*budget=*/(1ULL << 62));
    const HardwareProfile hw{.total_memory_bytes = 1ULL << 62};
    EXPECT_THROW((void)plan_job(job, hw), common::ResourceExceeded);
}

// The planner cases below all request want_statevector=true, which makes
// DirectStabilizer (candidates[1], always enumerated as of Phase 4) and
// DirectMPS (candidates[2], always enumerated as of Phase 6) both
// infeasible unconditionally (neither capabilities().statevector is
// true) -- so these cases' DirectDense-vs-Factorized comparisons
// (docs/BUILD_PLAN.md Phase 3, 3.6 and 3.11 item 13) are unaffected in
// substance, only candidates[]'s indices shift: [0]=DirectDense,
// [1]=DirectStabilizer (infeasible here), [2]=DirectMPS (infeasible
// here), [3]=Factorized (when enumerated). Each expected winner is
// verified by hand against the exact work-unit formula (also
// independently validated in tests/cost/cost_model_test.cpp), not
// assumed.

TEST(PlanJobTwoCandidateTest, SingleComponentAlwaysDirect) {
    // A fully-connected 2-qubit Bell circuit is one component: Factorized
    // is never even enumerated.
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = true,
                      .seed = 0,
                      .memory_budget_bytes = 1'000'000,
                      .expectation = std::nullopt};
    const HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, hw);
    EXPECT_EQ(plan.strategy, Strategy::DirectDense);
    ASSERT_EQ(plan.candidates.size(), 3U);  // Factorized not enumerated
    EXPECT_EQ(plan.candidates[0].strategy, Strategy::DirectDense);
    EXPECT_TRUE(plan.candidates[0].is_winner);
    EXPECT_EQ(plan.candidates[1].strategy, Strategy::DirectStabilizer);
    EXPECT_FALSE(plan.candidates[1].feasible)
        << "statevector was requested; the stabilizer engine cannot produce one";
    EXPECT_EQ(plan.candidates[2].strategy, Strategy::DirectMPS);
    EXPECT_FALSE(plan.candidates[2].feasible)
        << "statevector was requested; the MPS engine cannot produce one";
}

TEST(PlanJobTwoCandidateTest, ManyIdleComponentsMakeFactorizationPricierThanDirect) {
    // 10 qubits, one CX on {0,1}, 8 idle singletons -- recombination cost
    // (2^n per component, 9 components) dominates and Direct wins.
    // Hand-computed: Direct work = 2^10 + 2^12 + 2^10 = 6144.
    //                Factorized work = (2^2+2^4+2^2) [component {0,1}] + 0 [8 idle]
    //                                  + 2^10 * 9 [recombination] = 24 + 9216 = 9240.
    Circuit circuit(10, 0);
    circuit.append(Opcode::CX, {0, 1});
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = true,
                      .seed = 0,
                      .memory_budget_bytes = 10'000'000,
                      .expectation = std::nullopt};
    const HardwareProfile hw{.total_memory_bytes = 10'000'000};
    const Plan plan = plan_job(job, hw);
    EXPECT_EQ(plan.strategy, Strategy::DirectDense);
    ASSERT_EQ(plan.candidates.size(), 4U);
    EXPECT_EQ(plan.candidates[0].estimated_work_units, 6144U);
    EXPECT_FALSE(plan.candidates[1].feasible);
    EXPECT_FALSE(plan.candidates[2].feasible);
    EXPECT_EQ(plan.candidates[3].estimated_work_units, 9240U);
    EXPECT_TRUE(plan.candidates[0].is_winner);
    EXPECT_FALSE(plan.candidates[3].is_winner);
}

TEST(PlanJobTwoCandidateTest, ManySmallComponentsMakeFactorizationCheaperThanDirect) {
    // 10 qubits as 5 independent 2-qubit Bell pairs: avoiding the 2^10
    // exponential blowup for 5 separate 2^2-scale regions wins decisively.
    // Hand-computed: Direct work = 2^10 + 5*2^12 + 2^10 = 1024+20480+1024=22528.
    //                Factorized work = 5*(2^2+2^4+2^2) + 2^10*5 = 5*24+5120=5240.
    // (want_statevector=true keeps every region on dense, same as the
    // top-level DirectStabilizer candidate, so these values are the same
    // Phase 3 established, unaffected by the stabilizer engine's
    // existence.)
    Circuit circuit(10, 0);
    for (QubitId q = 0; q < 10; q += 2) {
        circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = true,
                      .seed = 0,
                      .memory_budget_bytes = 10'000'000,
                      .expectation = std::nullopt};
    const HardwareProfile hw{.total_memory_bytes = 10'000'000};
    const Plan plan = plan_job(job, hw);
    EXPECT_EQ(plan.strategy, Strategy::Factorized);
    ASSERT_EQ(plan.candidates.size(), 4U);
    EXPECT_EQ(plan.candidates[0].estimated_work_units, 22528U);
    EXPECT_FALSE(plan.candidates[1].feasible);
    EXPECT_FALSE(plan.candidates[2].feasible);
    EXPECT_EQ(plan.candidates[3].estimated_work_units, 5240U);
    EXPECT_FALSE(plan.candidates[0].is_winner);
    EXPECT_TRUE(plan.candidates[3].is_winner);
    EXPECT_EQ(plan.regions.size(), 5U);
}

TEST(PlanJobTwoCandidateTest, ExactTieGoesToDirect) {
    // 2 qubits, H on qubit 1 only (qubit 0 fully idle): DirectDense and
    // Factorized compute to exactly 16 work units each (brute-force-
    // verified this session, not assumed):
    //   Direct: 2^2 (init) + 2^(2+1) (H) + 2^2 (statevector) = 4+8+4=16.
    //   Factorized: component{1} = 2^1+2^(1+1)+2^1=2+4+2=8; component{0}
    //     idle, free; + recombination 2^2*2=8; total=8+8=16.
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {1});
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = true,
                      .seed = 0,
                      .memory_budget_bytes = 1'000'000,
                      .expectation = std::nullopt};
    const HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, hw);
    ASSERT_EQ(plan.candidates.size(), 4U);
    EXPECT_EQ(plan.candidates[0].estimated_work_units, plan.candidates[3].estimated_work_units);
    EXPECT_EQ(plan.strategy, Strategy::DirectDense)
        << "a cost tie must go to the earlier-enumerated "
           "(direct) candidate, never factorized";
}

TEST(PlanJobTwoCandidateTest, FactorizedMemoryInfeasibleWhileDirectFeasibleStillPicksDirect) {
    // Same 10-qubit/1-real-component/8-idle circuit as the "pricier" test
    // above, but with a budget between Direct's peak (16384 bytes) and
    // Factorized's peak (64 + 16384 = 16448 bytes, since Factorized needs
    // both a region buffer AND the global recombination buffer
    // simultaneously) -- Factorized becomes infeasible purely on memory,
    // regardless of work units.
    Circuit circuit(10, 0);
    circuit.append(Opcode::CX, {0, 1});
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = true,
                      .seed = 0,
                      .memory_budget_bytes = 16400,
                      .expectation = std::nullopt};
    const HardwareProfile hw{.total_memory_bytes = 16400};
    const Plan plan = plan_job(job, hw);
    ASSERT_EQ(plan.candidates.size(), 4U);
    EXPECT_TRUE(plan.candidates[0].feasible);
    EXPECT_FALSE(plan.candidates[1].feasible);
    EXPECT_FALSE(plan.candidates[2].feasible);
    EXPECT_FALSE(plan.candidates[3].feasible);
    EXPECT_EQ(plan.strategy, Strategy::DirectDense);
}

TEST(PlanJobTwoCandidateTest, BothCandidatesInfeasibleThrowsWithBothReasons) {
    // 30 idle qubits (no gates at all -> 30 singleton components, so
    // Factorized is enumerated), budget far too small for either.
    const JobSpec job = make_job(30, /*budget=*/1024);
    const HardwareProfile hw{.total_memory_bytes = 1ULL << 40};
    try {
        (void)plan_job(job, hw);
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("direct"), std::string::npos) << message;
        EXPECT_NE(message.find("factorized"), std::string::npos) << message;
    }
}

// --- Phase 5: expectation-request routing (docs/BUILD_PLAN.md's planner
// test requirement: "a Clifford region routes to DirectStabilizer not
// DirectPauli or dense; a non-Clifford region within the term-count
// bound routes to DirectPauli when it is cheaper than dense"). ---

PauliString one_qubit_z() {
    PauliString p;
    p.num_qubits = 1;
    p.x_words = {0};
    p.z_words = {1};
    return p;
}

TEST(PlanJobExpectationTest, AllCliffordCircuitRoutesToStabilizerNotPauliOrDense) {
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    PauliString zz;
    zz.num_qubits = 2;
    zz.x_words = {0};
    zz.z_words = {0b11};
    const Observable obs{{{zz, {1.0, 0.0}}}};
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = false,
                      .seed = 0,
                      .memory_budget_bytes = 1'000'000,
                      .expectation = obs};
    const HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, hw);

    // DirectDense, DirectStabilizer, DirectPauli, DirectMPS,
    // DirectTensorNetwork, CutAndRecombine -- this circuit (H, CX) is one
    // connected component (docs/BUILD_PLAN.md Phase 3's own rule), so
    // Factorized is correctly not enumerated at all, but a wire cut does exist
    // and is therefore offered and costed (Phase 9).
    ASSERT_EQ(plan.candidates.size(), 6U);
    EXPECT_EQ(plan.candidates[0].strategy, Strategy::DirectDense);
    EXPECT_EQ(plan.candidates[1].strategy, Strategy::DirectStabilizer);
    EXPECT_EQ(plan.candidates[2].strategy, Strategy::DirectPauli);
    EXPECT_EQ(plan.candidates[3].strategy, Strategy::DirectMPS);
    EXPECT_EQ(plan.candidates[4].strategy, Strategy::DirectTensorNetwork);
    EXPECT_EQ(plan.strategy, Strategy::DirectStabilizer)
        << "stabilizer's own polynomial cost still strictly undercuts MPS's here, even though "
           "nothing hard-excludes MPS from a Clifford region the way DirectPauli is excluded";
}

// Hand-computable (docs/adr/0007's cost formulas): 1 qubit, one RZ gate,
// one-term Z observable. Dense work = 2^1 (init) + 2^(1+1) (RZ) +
// 1*2^1 (expectation pass) = 2+4+2 = 8. Stabilizer is infeasible (RZ is
// not Clifford). Pauli: branching = 2^1 (one RZ) = 2, max_terms = 2,
// work = max_terms * num_gates = 2*1 = 2 -- strictly cheaper than dense.
TEST(PlanJobExpectationTest, NonCliffordWithinBoundRoutesToPauliWhenCheaper) {
    Circuit circuit(1, 0);
    circuit.append(Opcode::RZ, {0}, {0.5});
    const Observable obs{{{one_qubit_z(), {1.0, 0.0}}}};
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = false,
                      .seed = 0,
                      .memory_budget_bytes = 1'000'000,
                      .expectation = obs};
    const HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, hw);

    // DirectDense, DirectStabilizer, DirectPauli, DirectMPS,
    // DirectTensorNetwork -- a single qubit is trivially one connected
    // component, so Factorized is correctly not enumerated at all.
    ASSERT_EQ(plan.candidates.size(), 5U);
    EXPECT_EQ(plan.candidates[0].estimated_work_units, 8U);
    EXPECT_FALSE(plan.candidates[1].feasible);
    EXPECT_EQ(plan.candidates[2].estimated_work_units, 2U);
    EXPECT_TRUE(plan.candidates[2].is_winner);
    ASSERT_TRUE(plan.candidates[3].feasible);
    EXPECT_GT(plan.candidates[3].estimated_work_units, plan.candidates[2].estimated_work_units)
        << "MPS is feasible here too, but strictly more expensive than Pauli's O(term count) "
           "propagation for this single-qubit case";
    EXPECT_EQ(plan.strategy, Strategy::DirectPauli);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].engine, Engine::Pauli);
}

// A counts-only (no expectation) request must never enumerate
// DirectPauli -- its capabilities() can never satisfy that request.
TEST(PlanJobExpectationTest, CountsOnlyRequestNeverEnumeratesDirectPauli) {
    Circuit circuit(1, 1);
    circuit.append(Opcode::H, {0});
    circuit.append_measure(0, 0);
    const JobSpec job{.circuit = circuit,
                      .shots = 100,
                      .want_statevector = false,
                      .seed = 0,
                      .memory_budget_bytes = 1'000'000,
                      .expectation = std::nullopt};
    const HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, hw);

    ASSERT_EQ(plan.candidates.size(), 3U);  // DirectDense, DirectStabilizer, DirectMPS
    for (const CandidateSummary& c : plan.candidates) {
        EXPECT_NE(c.strategy, Strategy::DirectPauli);
    }
}

// docs/BUILD_PLAN.md Phase 6 and 8: "DirectMPS slots into the cumulative
// enumeration order from section 4, after DirectPauli and before
// Factorized" and "DirectTensorNetwork slots in ... after DirectMPS and
// before Factorized." A single scenario exercising every enumeration
// condition at once (an expectation request, so DirectPauli and
// DirectTensorNetwork are both enumerated; 2 disjoint components, so
// Factorized is enumerated; a wire cut exists, so CutAndRecombine is
// enumerated) is the only way to see the complete, 7-candidate cumulative
// order in one plan.
TEST(PlanJobExpectationTest, FullCumulativeEnumerationOrderWithAllSevenCandidates) {
    Circuit circuit(4, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::H, {2});
    circuit.append(Opcode::CX, {2, 3});

    // Z0 Z2 spans both disjoint components, so light-cone reduction's
    // backward walk from {0, 2} keeps both -- an observable entirely
    // within one component (e.g. Z0 Z1) would let light-cone drop the
    // other component's qubits outright, leaving only one component
    // and never enumerating Factorized at all.
    PauliString z0z2;
    z0z2.num_qubits = 4;
    z0z2.x_words = {0};
    z0z2.z_words = {0b0101};
    const Observable obs{{{z0z2, {1.0, 0.0}}}};
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = false,
                      .seed = 0,
                      .memory_budget_bytes = 1'000'000,
                      .expectation = obs};
    const HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, hw);

    ASSERT_EQ(plan.candidates.size(), 7U);
    EXPECT_EQ(plan.candidates[0].strategy, Strategy::DirectDense);
    EXPECT_EQ(plan.candidates[1].strategy, Strategy::DirectStabilizer);
    EXPECT_EQ(plan.candidates[2].strategy, Strategy::DirectPauli);
    EXPECT_EQ(plan.candidates[3].strategy, Strategy::DirectMPS);
    EXPECT_EQ(plan.candidates[4].strategy, Strategy::DirectTensorNetwork);
    EXPECT_EQ(plan.candidates[5].strategy, Strategy::Factorized);
    EXPECT_EQ(plan.candidates[6].strategy, Strategy::CutAndRecombine);
}

// A counts-only request must never enumerate DirectTensorNetwork either --
// its capabilities() can no more satisfy counts than DirectPauli's can
// (docs/BUILD_PLAN.md Phase 8's own planner test requirement).
TEST(PlanJobExpectationTest, CountsOnlyRequestNeverEnumeratesDirectTensorNetwork) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    const JobSpec job{.circuit = circuit,
                      .shots = 100,
                      .want_statevector = false,
                      .seed = 0,
                      .memory_budget_bytes = 1'000'000,
                      .expectation = std::nullopt};
    const HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, hw);

    for (const CandidateSummary& c : plan.candidates) {
        EXPECT_NE(c.strategy, Strategy::DirectTensorNetwork);
    }
}

// The fast regression guard for acceptance criterion 4. The scale-driven
// claim lives in tests/api/tensor_network_scale_test.cpp, at a realistic
// budget on a tree topology; this is the cheap version that keeps the same
// routing behaviour honest on every run.
//
// A 20-qubit single-layer brickwork circuit plus one T gate, with a
// full-width observable so light-cone reduction is a no-op. Measured
// directly rather than guessed: dense needs 16 * 2^20 = 16,777,216 bytes,
// MPS needs 2,432, and the contraction needs 512 (its closed network's line
// graph stays narrow for a single nearest-neighbour layer). A 1,000-byte
// budget therefore admits the contraction and nothing else.
//
// Note this budget is deliberately artificial, and is disclosed as such: a
// constrained budget is a legitimate operating condition for MENOS, but on
// its own it would be weak evidence for the criterion, which is why the
// realistic-budget test exists alongside it.
TEST(PlanJobExpectationTest, TightBudgetLeavesDirectTensorNetworkTheOnlyFeasibleCandidate) {
    constexpr std::uint32_t n = 20;
    Circuit circuit(n, 0);
    for (QubitId q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {q});
    }
    for (QubitId q = 0; q + 1 < n; q += 2) {
        circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    for (QubitId q = 1; q + 1 < n; q += 2) {
        circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    circuit.append(Opcode::T, {0});
    PauliString z_all;
    z_all.num_qubits = n;
    z_all.x_words = {0};
    z_all.z_words = {(std::uint64_t{1} << n) - 1};  // Z on every qubit
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = false,
                      .seed = 0,
                      .memory_budget_bytes = 1000,
                      .expectation = Observable{{{z_all, {1.0, 0.0}}}}};
    const HardwareProfile hw{.total_memory_bytes = 1ULL << 40};
    const Plan plan = plan_job(job, hw);

    for (const CandidateSummary& c : plan.candidates) {
        if (c.strategy == Strategy::DirectTensorNetwork) {
            EXPECT_TRUE(c.feasible);
            EXPECT_TRUE(c.is_winner);
        } else {
            EXPECT_FALSE(c.feasible) << "strategy " << static_cast<int>(c.strategy)
                                     << " unexpectedly feasible under a 1,000-byte budget";
        }
    }
    ASSERT_EQ(plan.strategy, Strategy::DirectTensorNetwork);
    EXPECT_TRUE(plan.exact);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].engine, Engine::TensorNetwork);
}

// The architectural guarantee, asserted rather than assumed: the
// tensor-network candidate is selected on cost, never because it exists.
// When a cheaper candidate is feasible, it wins -- there is no last-resort
// gate, and no qubit-count rule that would exclude or privilege this engine.
TEST(PlanJobExpectationTest, TensorNetworkLosesToACheaperCandidateOnCost) {
    // One RZ on one qubit: Pauli propagation costs 2 work units, the
    // contraction costs more, and both are feasible. Cost alone decides. With
    // branch and bound the contraction's costing stops as soon as it passes
    // the cheaper candidate, and explain() says so.
    Circuit circuit(1, 0);
    circuit.append(Opcode::RZ, {0}, {0.5});
    const Observable obs{{{one_qubit_z(), {1.0, 0.0}}}};
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = false,
                      .seed = 0,
                      .memory_budget_bytes = 1'000'000,
                      .expectation = obs};
    const HardwareProfile hw{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, hw);

    const CandidateSummary* tn = nullptr;
    const CandidateSummary* pauli = nullptr;
    for (const CandidateSummary& c : plan.candidates) {
        if (c.strategy == Strategy::DirectTensorNetwork) {
            tn = &c;
        }
        if (c.strategy == Strategy::DirectPauli) {
            pauli = &c;
        }
    }
    ASSERT_NE(tn, nullptr);
    ASSERT_NE(pauli, nullptr);
    ASSERT_TRUE(pauli->feasible);
    EXPECT_FALSE(tn->feasible);
    EXPECT_EQ(tn->infeasible_reason.rfind(kPrunedReasonPrefix, 0), 0U) << tn->infeasible_reason;
    EXPECT_EQ(plan.strategy, Strategy::DirectPauli);
    EXPECT_FALSE(tn->is_winner);
}

TEST(PlanJobExpectationTest, WithATimeBudgetTheTensorNetworkIsCostedInFull) {
    // The same job with a (generous) time budget: branch and bound is off,
    // because a candidate cheaper in work units could still be rejected on
    // seconds, so the contraction keeps its full figure, as before.
    Circuit circuit(1, 0);
    circuit.append(Opcode::RZ, {0}, {0.5});
    const Observable obs{{{one_qubit_z(), {1.0, 0.0}}}};
    const JobSpec job{.circuit = circuit,
                      .shots = 0,
                      .want_statevector = false,
                      .seed = 0,
                      .memory_budget_bytes = 1'000'000,
                      .expectation = obs,
                      .time_budget_seconds = 1e12};
    const HardwareProfile hw{.total_memory_bytes = 1'000'000,
                             .memory_bandwidth_bytes_per_second = 1.0,
                             .work_units_per_second = 1.0};
    const Plan plan = plan_job(job, hw);

    const CandidateSummary* tn = nullptr;
    const CandidateSummary* pauli = nullptr;
    for (const CandidateSummary& c : plan.candidates) {
        if (c.strategy == Strategy::DirectTensorNetwork) {
            tn = &c;
        }
        if (c.strategy == Strategy::DirectPauli) {
            pauli = &c;
        }
    }
    ASSERT_NE(tn, nullptr);
    ASSERT_NE(pauli, nullptr);
    ASSERT_TRUE(tn->feasible) << "the contraction is feasible here; it simply is not cheapest";
    ASSERT_TRUE(pauli->feasible);
    EXPECT_LT(pauli->estimated_work_units, tn->estimated_work_units);
    EXPECT_EQ(plan.strategy, Strategy::DirectPauli);
    EXPECT_FALSE(tn->is_winner);
}

// Branch and bound must never change the plan: on random expectation jobs the
// winner and its figures are the same with pruning on (no time budget) and
// off (a time budget so large it rejects nothing).
TEST(PlanJobExpectationTest, BranchAndBoundNeverChangesTheWinner) {
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    const auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    const Opcode one_qubit[] = {Opcode::H, Opcode::RX, Opcode::RZ, Opcode::T, Opcode::S};
    const Opcode two_qubit[] = {Opcode::CX, Opcode::CZ, Opcode::CP};
    std::size_t pruned_seen = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const auto n = static_cast<std::uint32_t>(2 + next() % 6);
        Circuit circuit(n, 0);
        const std::size_t num_ops = 4 + next() % 30;
        for (std::size_t i = 0; i < num_ops; ++i) {
            if (next() % 3 == 0) {
                const auto a = static_cast<QubitId>(next() % n);
                auto b = static_cast<QubitId>(next() % n);
                if (a == b) {
                    b = static_cast<QubitId>((a + 1) % n);
                }
                const Opcode op = two_qubit[next() % 3];
                if (op == Opcode::CP) {
                    circuit.append(op, {a, b}, {0.3});
                } else {
                    circuit.append(op, {a, b});
                }
            } else {
                const Opcode op = one_qubit[next() % 5];
                const auto q = static_cast<QubitId>(next() % n);
                if (op == Opcode::RX || op == Opcode::RZ) {
                    circuit.append(op, {q}, {0.7});
                } else {
                    circuit.append(op, {q});
                }
            }
        }
        Observable obs;
        const std::size_t num_terms = 1 + next() % 6;
        for (std::size_t t = 0; t < num_terms; ++t) {
            PauliString p;
            p.num_qubits = n;
            p.x_words = {next() & ((std::uint64_t{1} << n) - 1)};
            p.z_words = {next() & ((std::uint64_t{1} << n) - 1)};
            obs.terms.emplace_back(p, std::complex<double>{1.0, 0.0});
        }
        JobSpec job{.circuit = circuit,
                    .shots = 0,
                    .want_statevector = false,
                    .seed = 0,
                    .memory_budget_bytes = 1ULL << 30,
                    .expectation = obs};
        const HardwareProfile hw{.total_memory_bytes = 1ULL << 30,
                                 .memory_bandwidth_bytes_per_second = 1e30,
                                 .work_units_per_second = 1e30};
        const Plan pruned = plan_job(job, hw);
        job.time_budget_seconds = 1e30;
        const Plan full = plan_job(job, hw);

        EXPECT_EQ(pruned.strategy, full.strategy) << "trial " << trial;
        EXPECT_EQ(pruned.estimated_work_units, full.estimated_work_units) << "trial " << trial;
        ASSERT_EQ(pruned.candidates.size(), full.candidates.size());
        for (std::size_t i = 0; i < full.candidates.size(); ++i) {
            const CandidateSummary& p = pruned.candidates[i];
            const CandidateSummary& f = full.candidates[i];
            EXPECT_EQ(p.is_winner, f.is_winner) << "trial " << trial << " candidate " << i;
            if (p.infeasible_reason.rfind(kPrunedReasonPrefix, 0) == 0) {
                ++pruned_seen;
                // A pruned candidate was either infeasible anyway or strictly
                // dearer than the winner.
                EXPECT_TRUE(!f.feasible || f.estimated_work_units > full.estimated_work_units)
                    << "trial " << trial << " candidate " << i;
            } else {
                EXPECT_EQ(p.feasible, f.feasible) << "trial " << trial << " candidate " << i;
                EXPECT_EQ(p.estimated_work_units, f.estimated_work_units)
                    << "trial " << trial << " candidate " << i;
            }
        }
    }
    EXPECT_GT(pruned_seen, 0U) << "the random jobs never exercised pruning";
}

// ---------------------------------------------------------------------------
// Phase 10: time_budget_seconds
//
// These use a deliberately synthetic HardwareProfile rather than a calibrated
// one. A real calibration varies between machines, so a test built on it would
// assert on this laptop's speed; with a fixed profile the arithmetic the
// planner performs is pinned exactly and the test means the same thing
// everywhere.
// ---------------------------------------------------------------------------

// One work unit per second and one byte per second: estimated_seconds then
// equals max(work_units, peak_bytes) numerically, which makes the expected
// value computable by hand from the candidate's own estimate.
constexpr HardwareProfile kUnitRateProfile{
    .total_memory_bytes = 1ULL << 40,
    .memory_bandwidth_bytes_per_second = 1.0,
    .work_units_per_second = 1.0,
};

TEST(TimeBudgetTest, AGenerousBudgetLeavesTheWinnerUnchanged) {
    JobSpec job = make_job(3, /*budget=*/1'000'000);
    const Plan without = plan_job(job, kUnitRateProfile);

    job.time_budget_seconds = 1.0e9;  // far above anything a 3-qubit job needs
    const Plan with = plan_job(job, kUnitRateProfile);

    EXPECT_EQ(with.strategy, without.strategy);
    EXPECT_EQ(with.strategy, Strategy::DirectDense);
}

// The figure is populated whenever the profile is calibrated, budget or not,
// so explain() can report it.
TEST(TimeBudgetTest, EstimatedSecondsIsPopulatedFromTheCalibratedRates) {
    const JobSpec job = make_job(3, /*budget=*/1'000'000);
    const Plan plan = plan_job(job, kUnitRateProfile);

    for (const CandidateSummary& c : plan.candidates) {
        if (!c.feasible) {
            continue;
        }
        const double work_seconds = static_cast<double>(c.estimated_work_units);
        const double traffic_seconds = static_cast<double>(c.estimated_peak_bytes);
        EXPECT_DOUBLE_EQ(c.estimated_seconds, std::max(work_seconds, traffic_seconds))
            << "estimated_seconds must be max(work term, memory-traffic floor)";
    }
}

// The memory-traffic floor is the planner's only consumer of the calibrated
// bandwidth. Here the work term is made negligible by a very fast work rate,
// so the floor is what the reported figure must come from.
TEST(TimeBudgetTest, TheMemoryTrafficFloorDominatesWhenWorkIsCheap) {
    const JobSpec job = make_job(3, /*budget=*/1'000'000);
    const HardwareProfile hw{
        .total_memory_bytes = 1ULL << 40,
        .memory_bandwidth_bytes_per_second = 1.0,  // 1 byte per second
        .work_units_per_second = 1.0e18,           // work is effectively free
    };
    const Plan plan = plan_job(job, hw);

    const CandidateSummary* winner = nullptr;
    for (const CandidateSummary& c : plan.candidates) {
        if (c.is_winner) {
            winner = &c;
        }
    }
    ASSERT_NE(winner, nullptr);
    EXPECT_DOUBLE_EQ(winner->estimated_seconds, static_cast<double>(winner->estimated_peak_bytes));
}

// The infeasible case: an unsatisfiable budget is refused, not quietly
// downgraded to a best-effort run of the cheapest candidate.
TEST(TimeBudgetTest, AnImpossibleBudgetThrowsResourceExceeded) {
    JobSpec job = make_job(3, /*budget=*/1'000'000);
    job.time_budget_seconds = 1.0e-12;  // below any candidate's estimate
    EXPECT_THROW((void)plan_job(job, kUnitRateProfile), common::ResourceExceeded);
}

// A budget that excludes some candidates but not all must still plan, and the
// surviving winner must genuinely respect the budget.
TEST(TimeBudgetTest, TheChosenCandidateAlwaysRespectsTheBudget) {
    JobSpec job = make_job(3, /*budget=*/1'000'000);

    const Plan unbudgeted = plan_job(job, kUnitRateProfile);
    double cheapest = 0.0;
    for (const CandidateSummary& c : unbudgeted.candidates) {
        if (c.is_winner) {
            cheapest = c.estimated_seconds;
        }
    }
    ASSERT_GT(cheapest, 0.0);

    job.time_budget_seconds = cheapest;  // exactly affordable, nothing to spare
    const Plan plan = plan_job(job, kUnitRateProfile);
    for (const CandidateSummary& c : plan.candidates) {
        if (c.is_winner) {
            EXPECT_LE(c.estimated_seconds, *job.time_budget_seconds);
        }
    }
}

// Candidates excluded by the budget are recorded as infeasible with a reason,
// so explain() can say why rather than silently omitting them.
TEST(TimeBudgetTest, ExcludedCandidatesCarryAReasonMentioningTheBudget) {
    JobSpec job = make_job(3, /*budget=*/1'000'000);
    job.time_budget_seconds = 1.0e-12;
    try {
        (void)plan_job(job, kUnitRateProfile);
        FAIL() << "an unsatisfiable time budget must throw";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find("time budget"), std::string::npos)
            << "the message must name the budget that failed: " << e.what();
    }
}

// An uncalibrated profile cannot convert work units to seconds. Refusing is
// the only honest answer: dividing by zero would make every candidate look
// instantaneous and admit every budget.
TEST(TimeBudgetTest, AnUncalibratedProfileRefusesATimeBudget) {
    JobSpec job = make_job(3, /*budget=*/1'000'000);
    job.time_budget_seconds = 1.0;
    const HardwareProfile uncalibrated{.total_memory_bytes = 1'000'000};
    EXPECT_THROW((void)plan_job(job, uncalibrated), common::InternalError);
}

// ...but an uncalibrated profile is still perfectly usable without a budget,
// which is what keeps every pre-Phase-10 caller working unchanged.
TEST(TimeBudgetTest, AnUncalibratedProfileIsFineWithoutATimeBudget) {
    const JobSpec job = make_job(3, /*budget=*/1'000'000);
    const HardwareProfile uncalibrated{.total_memory_bytes = 1'000'000};
    const Plan plan = plan_job(job, uncalibrated);
    EXPECT_EQ(plan.strategy, Strategy::DirectDense);
    for (const CandidateSummary& c : plan.candidates) {
        EXPECT_LT(c.estimated_seconds, 0.0) << "no seconds figure exists without calibration";
    }
}

// ---------------------------------------------------------------------------
// Phase 11: runtime observations (docs/adr/0015)
//
// Same synthetic-profile discipline as the Phase 10 block above: the rates are
// pinned so the planner's arithmetic, not this laptop's speed, is what is
// being asserted.

// Clifford, with counts rather than a statevector, so BOTH the dense and the
// stabilizer candidates are feasible and an observation can move the winner
// from one to the other.
JobSpec make_clifford_job(std::uint32_t num_qubits) {
    Circuit circuit(num_qubits, num_qubits);
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        for (std::uint32_t q = 0; q < num_qubits; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        for (std::uint32_t q = layer % 2; q + 1 < num_qubits; q += 2) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    for (std::uint32_t q = 0; q < num_qubits; ++q) {
        circuit.append_measure(static_cast<QubitId>(q), static_cast<ClbitId>(q));
    }
    return JobSpec{
        .circuit = circuit,
        .shots = 16,
        .want_statevector = false,
        .seed = 0,
        .memory_budget_bytes = std::uint64_t{1} << 30,
        .expectation = std::nullopt,
    };
}

const CandidateSummary& winner_of(const Plan& plan) {
    for (const CandidateSummary& c : plan.candidates) {
        if (c.is_winner) {
            return c;
        }
    }
    throw std::logic_error("no winner");
}

// The core of the phase: the same JobSpec and the same HardwareProfile, with
// ONLY the observations differing, must produce a different plan. This is what
// "the new plan uses observed runtime information" means.
TEST(RuntimeObservationTest, AnObservedFloorAboveTheBudgetMovesTheWinnerToAnotherEngine) {
    JobSpec job = make_clifford_job(4);
    job.time_budget_seconds = 1.0e9;  // generous: every candidate fits the model

    const Plan before = plan_job(job, kUnitRateProfile);
    ASSERT_EQ(before.strategy, Strategy::DirectStabilizer)
        << "a Clifford counts job is expected to route to the stabilizer engine";

    RuntimeObservations observations;
    observations.observe(Engine::Stabilizer, Backend::Cpu, 2.0e9);  // over the budget

    const Plan after = plan_job(job, kUnitRateProfile, observations);
    EXPECT_NE(after.strategy, Strategy::DirectStabilizer)
        << "the observed engine must not be re-selected";
    // Which candidate takes over is the EXISTING selection rule's business --
    // cheapest work units among the survivors -- not something this phase
    // overrides. Here that is DirectMPS, which undercuts DirectDense on work
    // units; the assertion is that the rule still ran, not that dense wins.
    EXPECT_EQ(after.strategy, Strategy::DirectMPS);
    EXPECT_LE(winner_of(after).estimated_seconds, *job.time_budget_seconds)
        << "the replacement must itself respect the budget";
    for (const CandidateSummary& c : after.candidates) {
        if (c.strategy == Strategy::DirectStabilizer) {
            EXPECT_FALSE(c.feasible);
        }
    }
}

// An observation raises ONE engine's figure. Every other candidate must be
// costed exactly as it was before, or the observation would be a global
// recalibration rather than a fact about one engine.
TEST(RuntimeObservationTest, AnObservedFloorAppliesToThatEngineAlone) {
    const JobSpec job = make_clifford_job(4);
    const Plan before = plan_job(job, kUnitRateProfile);

    RuntimeObservations observations;
    observations.observe(Engine::Stabilizer, Backend::Cpu, 5.0e6);
    const Plan after = plan_job(job, kUnitRateProfile, observations);

    ASSERT_EQ(before.candidates.size(), after.candidates.size());
    for (std::size_t i = 0; i < before.candidates.size(); ++i) {
        if (before.candidates[i].strategy == Strategy::DirectStabilizer) {
            EXPECT_DOUBLE_EQ(after.candidates[i].estimated_seconds, 5.0e6);
        } else {
            EXPECT_DOUBLE_EQ(after.candidates[i].estimated_seconds,
                             before.candidates[i].estimated_seconds)
                << "an observation about one engine must not move another's estimate";
        }
    }
}

// max(), never replacement. An observation can only ever make a candidate look
// MORE expensive, so a stale or understated observation cannot make the
// planner optimistic.
TEST(RuntimeObservationTest, AnObservedFloorBelowTheModelEstimateChangesNothing) {
    const JobSpec job = make_clifford_job(4);
    const Plan before = plan_job(job, kUnitRateProfile);
    const double modelled = winner_of(before).estimated_seconds;
    ASSERT_GT(modelled, 0.0);

    RuntimeObservations observations;
    observations.observe(winner_of(before).strategy == Strategy::DirectStabilizer
                             ? Engine::Stabilizer
                             : Engine::Dense,
                         Backend::Cpu,
                         modelled / 2.0);

    const Plan after = plan_job(job, kUnitRateProfile, observations);
    EXPECT_EQ(after.strategy, before.strategy);
    EXPECT_DOUBLE_EQ(winner_of(after).estimated_seconds, modelled);
}

// A floor is attributed to an (engine, BACKEND) pair. An observation of the CPU
// engine says nothing about the same algorithm on a GPU.
TEST(RuntimeObservationTest, AnObservationDoesNotCrossBackends) {
    const JobSpec job = make_clifford_job(4);
    const Plan before = plan_job(job, kUnitRateProfile);

    RuntimeObservations observations;
    observations.observe(Engine::Stabilizer, Backend::Cuda, 9.0e9);
    const Plan after = plan_job(job, kUnitRateProfile, observations);

    EXPECT_EQ(after.strategy, before.strategy)
        << "a CUDA observation must not constrain the CPU candidate";
    EXPECT_DOUBLE_EQ(winner_of(after).estimated_seconds, winner_of(before).estimated_seconds);
}

// The documented limitation, asserted so it is visible in the suite: a
// multi-engine candidate takes no floor, because elapsed time is not attributed
// per region in this phase.
TEST(RuntimeObservationTest, AMultiEngineCandidateTakesNoObservedFloor) {
    // Two disjoint Clifford blocks -> the interaction graph has two components,
    // so Factorized is enumerated.
    Circuit circuit(4, 4);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::H, {2});
    circuit.append(Opcode::CX, {2, 3});
    for (std::uint32_t q = 0; q < 4; ++q) {
        circuit.append_measure(static_cast<QubitId>(q), static_cast<ClbitId>(q));
    }
    const JobSpec job{
        .circuit = circuit,
        .shots = 16,
        .want_statevector = false,
        .seed = 0,
        .memory_budget_bytes = std::uint64_t{1} << 30,
        .expectation = std::nullopt,
    };

    const Plan before = plan_job(job, kUnitRateProfile);
    RuntimeObservations observations;
    observations.observe(Engine::Stabilizer, Backend::Cpu, 7.0e6);
    observations.observe(Engine::Dense, Backend::Cpu, 7.0e6);
    const Plan after = plan_job(job, kUnitRateProfile, observations);

    for (std::size_t i = 0; i < after.candidates.size(); ++i) {
        if (after.candidates[i].strategy != Strategy::Factorized) {
            continue;
        }
        EXPECT_DOUBLE_EQ(after.candidates[i].estimated_seconds,
                         before.candidates[i].estimated_seconds)
            << "Factorized spans several engines, so no single floor applies to it";
        return;
    }
    FAIL() << "the Factorized candidate was not enumerated, so nothing was asserted";
}

// A reader has to be able to tell a modelled refusal from a measured one: they
// call for completely different responses.
TEST(RuntimeObservationTest, TheInfeasibilityReasonSaysWhenTheRuntimeWasObserved) {
    JobSpec job = make_clifford_job(4);
    job.time_budget_seconds = 1.0e9;
    RuntimeObservations observations;
    observations.observe(Engine::Stabilizer, Backend::Cpu, 2.0e9);

    const Plan plan = plan_job(job, kUnitRateProfile, observations);
    for (const CandidateSummary& c : plan.candidates) {
        if (c.strategy != Strategy::DirectStabilizer) {
            continue;
        }
        ASSERT_FALSE(c.feasible);
        EXPECT_NE(c.infeasible_reason.find("observed runtime of at least"), std::string::npos)
            << c.infeasible_reason;
        return;
    }
    FAIL() << "the stabilizer candidate was not enumerated";
}

// When observations rule out everything, planning refuses. This is the strict
// reading of the time budget carried into the replan path: the run is not
// quietly downgraded to a best-effort attempt at whatever is left.
TEST(RuntimeObservationTest, ObservationsThatExcludeEveryCandidateThrow) {
    JobSpec job = make_clifford_job(4);
    job.time_budget_seconds = 1.0e9;

    RuntimeObservations observations;
    for (const Engine engine :
         {Engine::Dense, Engine::Stabilizer, Engine::Pauli, Engine::MPS, Engine::TensorNetwork}) {
        observations.observe(engine, Backend::Cpu, 2.0e9);
    }
    EXPECT_THROW((void)plan_job(job, kUnitRateProfile, observations), common::ResourceExceeded);
}

// Without a time budget an observation changes no decision, because selection
// is by work units and only the budget consumes seconds. Pinned so the scope of
// the mechanism is explicit rather than assumed.
TEST(RuntimeObservationTest, WithoutATimeBudgetAnObservationChangesNoWinner) {
    const JobSpec job = make_clifford_job(4);  // no time_budget_seconds
    const Plan before = plan_job(job, kUnitRateProfile);

    RuntimeObservations observations;
    observations.observe(Engine::Stabilizer, Backend::Cpu, 1.0e12);
    const Plan after = plan_job(job, kUnitRateProfile, observations);

    EXPECT_EQ(after.strategy, before.strategy);
}

// The premise the replan loop's termination proof rests on (src/api/
// replan_controller.cpp, "WHY A REPLAN NEVER REPRODUCES..." step 5): for every
// single-engine candidate, an observed floor above the budget on that
// candidate's own (engine, backend) makes THAT candidate infeasible. Checked per
// CANDIDATE, not only per winner, because the winner set does not cover every
// engine -- DirectTensorNetwork never wins on these jobs but is still enumerated
// and must still be excluded. Coverage of all six engines is asserted so the
// test cannot pass by exercising fewer of them than it claims.
std::optional<Engine> engine_of_direct(Strategy strategy) {
    switch (strategy) {
        case Strategy::DirectDense:
            return Engine::Dense;
        case Strategy::DirectStabilizer:
            return Engine::Stabilizer;
        case Strategy::DirectPauli:
            return Engine::Pauli;
        case Strategy::DirectMPS:
            return Engine::MPS;
        case Strategy::DirectTensorNetwork:
            return Engine::TensorNetwork;
        case Strategy::DirectDensityMatrix:
            return Engine::DensityMatrix;
        case Strategy::DirectTrajectory:
            return Engine::Trajectory;
        case Strategy::Factorized:
        case Strategy::CutAndRecombine:
            return std::nullopt;
    }
    return std::nullopt;
}

Observable single_z_observable() {
    PauliString pauli;
    pauli.num_qubits = 1;
    pauli.x_words = {0};
    pauli.z_words = {1};
    Observable observable;
    observable.terms.emplace_back(pauli, std::complex<double>{1.0, 0.0});
    return observable;
}

// One aggregate rather than two adjacent std::uint32_t parameters, so a qubit
// count cannot be silently transposed with a layer count at a call site.
struct LadderShape {
    std::uint32_t num_qubits = 0;
    std::uint32_t layers = 0;
};

Circuit rotation_ladder(LadderShape shape, Opcode rotation) {
    const std::uint32_t num_qubits = shape.num_qubits;
    const std::uint32_t layers = shape.layers;
    Circuit circuit(num_qubits, 0);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        for (std::uint32_t q = 0; q < num_qubits; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        if (rotation == Opcode::RZ) {
            circuit.append(Opcode::RZ, {0}, {0.37});
        } else {
            circuit.append(rotation, {0});
        }
        for (std::uint32_t q = layer % 2; q + 1 < num_qubits; q += 2) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    return circuit;
}

TEST(RuntimeObservationTest, EveryDirectCandidateIsExcludedByItsOwnObservedFloor) {
    std::vector<JobSpec> jobs;
    jobs.push_back(make_clifford_job(4));  // stabilizer, dense, mps
    {
        JobSpec job = make_job(6, std::uint64_t{1} << 30);  // statevector: dense only
        job.circuit = rotation_ladder({.num_qubits = 6, .layers = 3}, Opcode::T);
        jobs.push_back(job);
    }
    for (const Opcode rotation : {Opcode::RZ, Opcode::T}) {
        // Expectation requests enumerate pauli and tensor-network as well.
        JobSpec job = make_job(6, std::uint64_t{1} << 30);
        job.circuit = rotation_ladder({.num_qubits = 6, .layers = 2}, rotation);
        job.want_statevector = false;
        job.expectation = single_z_observable();
        jobs.push_back(job);
    }
    {
        // A noisy job (Phase 12) enumerates the density-matrix candidate, the
        // only one that can run it, so observing it must leave nothing.
        JobSpec job = make_clifford_job(3);
        Circuit noisy(3, 3);
        for (const Operation& op : job.circuit.operations()) {
            if (const auto* gate = std::get_if<GateOp>(&op)) {
                noisy.append(gate->opcode, gate->qubits, gate->params);
            }
        }
        noisy.append_channel({1}, {{1.0, 0.0, 0.0, 1.0}});
        for (QubitId q = 0; q < 3; ++q) {
            noisy.append_measure(q, q);
        }
        job.circuit = noisy;
        jobs.push_back(job);
    }

    constexpr double kBudget = 1.0e9;
    constexpr double kOverrun = 2.0e9;  // strictly above the budget, as a crossing is
    std::set<Engine> engines_covered;
    std::size_t refusals = 0;

    for (JobSpec job : jobs) {
        job.time_budget_seconds = kBudget;
        const Plan initial = plan_job(job, kUnitRateProfile);

        for (const CandidateSummary& candidate : initial.candidates) {
            const std::optional<Engine> engine = engine_of_direct(candidate.strategy);
            if (!candidate.feasible || !engine.has_value()) {
                continue;
            }
            engines_covered.insert(*engine);

            RuntimeObservations observations;
            observations.observe(*engine, candidate.backend, kOverrun);

            // Either nothing is left (a refusal) or something else is chosen.
            // What must never happen is the observed candidate surviving.
            try {
                const Plan replanned = plan_job(job, kUnitRateProfile, observations);
                for (const CandidateSummary& after : replanned.candidates) {
                    if (after.strategy == candidate.strategy) {
                        EXPECT_FALSE(after.feasible)
                            << "a candidate with an observed floor above the budget stayed "
                               "feasible";
                        EXPECT_FALSE(after.is_winner);
                    }
                }
                EXPECT_NE(replanned.strategy, candidate.strategy)
                    << "the replan reproduced the strategy that was just observed to overrun";
            } catch (const common::ResourceExceeded&) {
                // Nothing else fits: the strict refusal, which is a valid outcome.
                ++refusals;
            }
        }
    }

    // The refusal path is genuinely exercised: the statevector job can only run
    // on the dense engine, so observing dense leaves nothing.
    EXPECT_GE(refusals, 1U) << "no case reached the strict refusal, so that path went untested";

    EXPECT_EQ(engines_covered.size(), 6U)
        << "this test claims to cover every Direct* engine; if a job shape stopped enumerating "
           "one, it must be replaced, not silently dropped";
}

// observe() keeps the larger figure, so accumulating across successive
// abandoned attempts can never walk a floor backwards -- which is what makes
// the replan loop monotone, and therefore terminating.
TEST(RuntimeObservationTest, ObserveKeepsTheLargerFloor) {
    RuntimeObservations observations;
    observations.observe(Engine::Dense, Backend::Cpu, 5.0);
    observations.observe(Engine::Dense, Backend::Cpu, 2.0);
    EXPECT_EQ(observations.floor_for(Engine::Dense, Backend::Cpu), std::optional<double>{5.0});
    EXPECT_EQ(observations.floor_for(Engine::MPS, Backend::Cpu), std::nullopt);
}

}  // namespace
}  // namespace menos::plan
