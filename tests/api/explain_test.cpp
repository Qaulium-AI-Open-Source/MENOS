#include "menos/menos.h"

#include <gtest/gtest.h>

#include <cmath>
#include <regex>
#include <string>

namespace menos {
namespace {

Circuit make_circuit() {
    Circuit circuit(3, 3);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::CX, {1, 2});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    circuit.append_measure(2, 2);
    return circuit;
}

// The two wall-clock fields (execute_s, sample_s) are, by construction,
// not deterministic; blank them before comparing (acceptance criterion 6:
// "explain() output for a fixed circuit is byte-identical across runs").
std::string normalize_timings(const std::string& text) {
    static const std::regex kTimingPattern(R"((execute_s|sample_s|plan_s|replan_s)=[^ ]+)");
    return std::regex_replace(text, kTimingPattern, "$1=<t>");
}

TEST(ExplainTest, ByteIdenticalAcrossRunsAfterNormalizingTimings) {
    const Circuit circuit = make_circuit();
    const RunOptions options{.shots = 100, .want_statevector = true, .seed = 7};

    const std::string first = normalize_timings(run(circuit, options).explain());
    const std::string second = normalize_timings(run(circuit, options).explain());

    EXPECT_EQ(first, second);
}

// Pinned to the dense engine with a non-Clifford T gate
// (docs/BUILD_PLAN.md Phase 4: "Golden-count tests that use a fully
// Clifford circuit would now be routed to the stabilizer engine...
// pin each such test to the dense engine by adding a non-Clifford gate,
// a T"). make_circuit() alone (H, CX, CX) is all-Clifford and would now
// be routed to the (cheaper) stabilizer engine, which cannot produce
// the exact "engine=dense" report shape this test asserts. As of Phase
// 6, the T gate alone is no longer enough to pin a 3-qubit circuit to
// dense for a counts-only request: MPS's own bond dimension for any
// 3-qubit chain is capped at 2 (2^floor(3/2)) regardless of circuit
// structure, so its cost is always competitive with or cheaper than
// dense's exponential-in-qubit-count formula at this tiny scale (a
// genuine, correct outcome, not a bug -- verified in
// tests/plan/planner_test.cpp and tests/plan/region_builder_test.cpp).
// want_statevector=true below excludes both stabilizer and MPS
// unconditionally (neither capabilities().statevector is true),
// matching the sibling test above's own technique.
Circuit make_circuit_pinned_to_dense() {
    Circuit circuit(3, 3);
    circuit.append(Opcode::T, {0});
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::CX, {1, 2});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    circuit.append_measure(2, 2);
    return circuit;
}

// Phase 24 (docs/adr/0028 section 16.7 Q3): menos::run reports how long its
// planner call took, as a measured figure on the Result and on the actual
// line, and a job that never replanned reports no replanning time.
TEST(ExplainTest, RunReportsItsPlanningTimeAndNoReplanningTime) {
    const Result result =
        run(make_circuit(), RunOptions{.shots = 10, .want_statevector = false, .seed = 3});
    EXPECT_GT(result.plan_seconds(), 0.0);
    EXPECT_TRUE(std::isfinite(result.plan_seconds()));
    EXPECT_EQ(result.replan_seconds(), 0.0);
    const std::string text = result.explain();
    const std::size_t actual = text.find("actual     state_bytes=");
    ASSERT_NE(actual, std::string::npos) << text;
    const std::string line = text.substr(actual, text.find('\n', actual) - actual);
    // With the other wall-clock fields, before the measured peaks; and no
    // replanning field, because nothing was replanned.
    EXPECT_LT(line.find("sample_s="), line.find(" plan_s="));
    EXPECT_LT(line.find(" plan_s="), line.find(" peak_host_bytes="));
    EXPECT_EQ(text.find("replan"), std::string::npos) << text;
}

// A plan-cache hit still reports a planning time: the cost of the lookup and
// the re-parameterisation, which is all the planner ran.
TEST(ExplainTest, APlanCacheHitReportsTheTimeOfTheLookup) {
    plan_cache_clear();
    const auto circuit_with = [](double angle) {
        Circuit circuit(3, 3);
        circuit.append(Opcode::RY, {0}, {angle});
        circuit.append(Opcode::CX, {0, 1});
        circuit.append(Opcode::CX, {1, 2});
        circuit.append_measure(0, 0);
        circuit.append_measure(1, 1);
        circuit.append_measure(2, 2);
        return circuit;
    };
    const RunOptions options{.shots = 10, .want_statevector = true, .seed = 3};
    static_cast<void>(run(circuit_with(0.25), options));
    const PlanCacheStats before = plan_cache_stats();
    const Result hit = run(circuit_with(0.5), options);
    ASSERT_EQ(plan_cache_stats().hits, before.hits + 1) << "the second run must hit the cache";
    EXPECT_GE(hit.plan_seconds(), 0.0);
    EXPECT_TRUE(std::isfinite(hit.plan_seconds()));
    EXPECT_EQ(hit.replan_seconds(), 0.0);
}

TEST(ExplainTest, MatchesTheDocumentedReportShape) {
    const Circuit circuit = make_circuit_pinned_to_dense();
    const std::string text =
        run(circuit, RunOptions{.shots = 10, .want_statevector = true, .seed = 1}).explain();

    EXPECT_NE(text.find("MENOS EXECUTION REPORT\n"), std::string::npos);
    EXPECT_NE(text.find("circuit    qubits=3 clbits=3 operations=7 depth=5 global_phase=0\n"),
              std::string::npos);
    EXPECT_NE(text.find("strategy   direct_dense (exact)\n"), std::string::npos);
    EXPECT_NE(text.find("region 0   qubits=[0,1,2] engine=dense backend=cpu\n"), std::string::npos);
    EXPECT_NE(text.find("estimated  state_bytes="), std::string::npos);
    EXPECT_NE(text.find("actual     state_bytes="), std::string::npos);
}

TEST(ExplainTest, InfeasibleCandidateShowsItsRejectionReason) {
    // Same shape as tests/plan/planner_test.cpp's
    // FactorizedMemoryInfeasibleWhileDirectFeasibleStillPicksDirect:
    // Factorized is enumerated (2 components) but infeasible on memory,
    // while Direct fits. Acceptance criterion 9 (docs/BUILD_PLAN.md
    // Phase 3, 3.12): candidate explanations must contain feasibility,
    // bytes, work units, winner, and rejection reasons.
    Circuit circuit(10, 0);
    circuit.append(Opcode::CX, {0, 1});
    const RunOptions options{
        .shots = 0, .want_statevector = true, .seed = 0, .memory_budget_bytes = 17000};
    const std::string text = run(circuit, options).explain();

    EXPECT_NE(text.find("candidate 0"), std::string::npos);
    EXPECT_NE(text.find("candidate 1"), std::string::npos);
    EXPECT_NE(text.find("feasible=true"), std::string::npos);
    EXPECT_NE(text.find("feasible=false"), std::string::npos);
    EXPECT_NE(text.find("winner=true"), std::string::npos);
    EXPECT_NE(text.find("winner=false"), std::string::npos);
    EXPECT_NE(text.find("reason=\""), std::string::npos);
    EXPECT_NE(text.find("bytes"), std::string::npos);  // present in the rejection reason text
    EXPECT_NE(text.find("strategy   direct_dense (exact)\n"), std::string::npos);
}

// docs/BUILD_PLAN.md Phase 5 acceptance criterion 6: explain() must show
// the light-cone reduction step and, for an expectation request, the
// candidate table extended with the Pauli-propagation candidate.
TEST(ExplainTest, ShowsLightConeReductionAndExpectationValue) {
    // Qubit 2's X gate is unrelated to the observable's support (qubit
    // 0 only) and must be dropped by light-cone reduction before
    // planning; qubit 1 is never touched at all.
    Circuit circuit(3, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::X, {2});

    PauliString z0;
    z0.num_qubits = 3;
    z0.x_words = {0};
    z0.z_words = {0b001};
    const Observable obs{{{z0, {1.0, 0.0}}}};

    const RunOptions options{.shots = 0, .want_statevector = false, .seed = 0, .expectation = obs};
    const Result result = run(circuit, options);
    const std::string text = result.explain();

    EXPECT_NE(text.find("light_cone qubits 3->1 operations 2->1\n"), std::string::npos) << text;
    EXPECT_NE(text.find("expectation"), std::string::npos) << text;

    // H|0> is the equal superposition; <Z> is exactly 0 (both amplitude
    // magnitudes are computed identically, so the dense/stabilizer/Pauli
    // formula's subtraction is an exact floating-point cancellation, not
    // merely within tolerance).
    ASSERT_TRUE(result.expectation_value().has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(result.expectation_value()->real(), 0.0);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(result.expectation_value()->imag(), 0.0);
}

// docs/BUILD_PLAN.md Phase 7 acceptance criterion 3: "Plan::explain()
// and Result::explain()'s estimated section agree byte-for-byte across
// at least five circuits, one per engine choice plus one factorized
// example." preview() and run() each plan the circuit independently
// (planning has no randomness -- same circuit and options always
// produce the same Plan), so Result::explain() must begin with exactly
// the text preview(circuit, options).explain() produces.
void expect_plan_explain_is_result_explain_prefix(const Circuit& circuit,
                                                  const RunOptions& options) {
    const std::string preview_text = preview(circuit, options).explain();
    const std::string result_text = run(circuit, options).explain();
    ASSERT_GE(result_text.size(), preview_text.size());
    EXPECT_EQ(result_text.substr(0, preview_text.size()), preview_text)
        << "preview:\n"
        << preview_text << "\nresult:\n"
        << result_text;
}

// 100-qubit nearest-neighbor shallow circuit (same shape as
// tests/api/mps_scale_test.cpp): shallow enough that MPS's bond stays
// tiny while dense is refused outright, so DirectMPS wins outright.
Circuit make_mps_winning_circuit() {
    constexpr std::uint32_t n = 100;
    Circuit circuit(n, n);
    for (QubitId q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {q});
    }
    for (QubitId q = 0; q + 1 < n; q += 2) {
        circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    for (QubitId q = 1; q + 1 < n; q += 2) {
        circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    for (QubitId q = 0; q < n; ++q) {
        circuit.append_measure(q, q);
    }
    return circuit;
}

TEST(ExplainTest, PlanExplainMatchesResultExplainEstimatedSectionPerEngine) {
    // DirectDense.
    {
        SCOPED_TRACE("dense");
        expect_plan_explain_is_result_explain_prefix(
            make_circuit_pinned_to_dense(),
            RunOptions{.shots = 10, .want_statevector = true, .seed = 1});
    }
    // DirectStabilizer (tests/plan/planner_test.cpp's
    // AllCliffordCircuitRoutesToStabilizerNotPauliOrDense circuit: all-
    // Clifford, an expectation request, stabilizer strictly undercuts
    // both dense and MPS here).
    {
        SCOPED_TRACE("stabilizer");
        Circuit circuit(2, 0);
        circuit.append(Opcode::H, {0});
        circuit.append(Opcode::CX, {0, 1});
        PauliString zz;
        zz.num_qubits = 2;
        zz.x_words = {0};
        zz.z_words = {0b11};
        const Observable obs{{{zz, {1.0, 0.0}}}};
        const RunOptions options{.shots = 0,
                                 .want_statevector = false,
                                 .seed = 0,
                                 .memory_budget_bytes = 1'000'000,
                                 .expectation = obs};
        ASSERT_EQ(run(circuit, options).plan().strategy, Strategy::DirectStabilizer);
        expect_plan_explain_is_result_explain_prefix(circuit, options);
    }
    // DirectPauli (tests/plan/planner_test.cpp's
    // NonCliffordWithinBoundRoutesToPauliWhenCheaper circuit: a single
    // RZ, Pauli propagation's O(term count) cost undercuts dense).
    {
        SCOPED_TRACE("pauli");
        Circuit circuit(1, 0);
        circuit.append(Opcode::RZ, {0}, {0.5});
        PauliString z;
        z.num_qubits = 1;
        z.x_words = {0};
        z.z_words = {0b1};
        const Observable obs{{{z, {1.0, 0.0}}}};
        const RunOptions options{.shots = 0,
                                 .want_statevector = false,
                                 .seed = 0,
                                 .memory_budget_bytes = 1'000'000,
                                 .expectation = obs};
        ASSERT_EQ(run(circuit, options).plan().strategy, Strategy::DirectPauli);
        expect_plan_explain_is_result_explain_prefix(circuit, options);
    }
    // DirectMPS.
    {
        SCOPED_TRACE("mps");
        const Circuit circuit = make_mps_winning_circuit();
        const RunOptions options{.shots = 20, .want_statevector = false, .seed = 1};
        ASSERT_EQ(run(circuit, options).plan().strategy, Strategy::DirectMPS);
        expect_plan_explain_is_result_explain_prefix(circuit, options);
    }
    // Factorized.
    {
        SCOPED_TRACE("factorized");
        Circuit circuit(4, 0);
        circuit.append(Opcode::CX, {0, 1});
        circuit.append(Opcode::CX, {2, 3});
        const RunOptions options{.shots = 0, .want_statevector = true, .seed = 1};
        ASSERT_EQ(run(circuit, options).plan().strategy, Strategy::Factorized);
        expect_plan_explain_is_result_explain_prefix(circuit, options);
    }
}

TEST(ExplainTest, ShowsBothCandidatesAndMultipleRegionsWhenFactorized) {
    // 4 qubits as two independent Bell pairs {0,1} and {2,3}: Factorized
    // strictly cheaper than Direct (same shape as
    // tests/plan/planner_test.cpp's ManySmallComponentsMakeFactorizationCheaperThanDirect).
    Circuit circuit(4, 0);
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::CX, {2, 3});
    const std::string text =
        run(circuit, RunOptions{.shots = 0, .want_statevector = true, .seed = 1}).explain();

    EXPECT_NE(text.find("candidate 0"), std::string::npos);
    EXPECT_NE(text.find("candidate 1"), std::string::npos);
    EXPECT_NE(text.find("strategy   factorized (exact)\n"), std::string::npos);
    EXPECT_NE(text.find("region 0"), std::string::npos);
    EXPECT_NE(text.find("region 1"), std::string::npos);
}

// docs/BUILD_PLAN.md Phase 8 acceptance criterion 6: explain() reports the
// estimated and the actual peak tensor size for a DirectTensorNetwork
// region, so the two can be compared.
//
// Showing both numbers is the visible half of the criterion; the half that
// matters is that they are comparable and that the estimate really bounds
// the measurement. An earlier version of this test asserted only that both
// strings appeared and that the actual figure was above zero, which would
// have passed just as happily while the estimator understated the true peak
// by a factor of two -- as it in fact did. This asserts the relationship.
TEST(ExplainTest, ShowsEstimatedAndActualPeakTensorSizeForTensorNetworkRegion) {
    // A 31-qubit balanced binary tree: dense and MPS are both over budget at
    // this size, a wire cut is offered and loses on cost, and the contraction
    // wins -- so this exercises the tensor-network region's explain() output.
    // (A smaller tree would now route to CutAndRecombine, which is the planner
    // being cost-driven rather than a regression.)
    constexpr std::uint32_t n = 31;
    Circuit circuit(n, 0);
    for (std::uint32_t layer = 0; layer < 4; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(
                Opcode::RY,
                {static_cast<QubitId>(q)},
                {0.37 + 0.11 * static_cast<double>(layer) + 0.017 * static_cast<double>(q)});
        }
        for (std::uint32_t parent = layer % 2; parent < n; parent += 2) {
            for (std::uint32_t k = 1; k <= 2; ++k) {
                const std::uint32_t child = 2 * parent + k;
                if (child < n) {
                    circuit.append(Opcode::CZ,
                                   {static_cast<QubitId>(parent), static_cast<QubitId>(child)});
                }
            }
        }
    }
    circuit.append(Opcode::T, {0});

    PauliString z_all;
    z_all.num_qubits = n;
    z_all.x_words.assign(1, 0);
    z_all.z_words.assign(1, 0);
    for (std::uint32_t q = 0; q < n; ++q) {
        z_all.z_words[0] |= (std::uint64_t{1} << q);
    }

    const Result result = run(circuit,
                              RunOptions{.shots = 0,
                                         .want_statevector = false,
                                         .seed = 0,
                                         .memory_budget_bytes = 7'752'519'680ULL,
                                         .expectation = Observable{{{z_all, {1.0, 0.0}}}}});

    ASSERT_EQ(result.plan().strategy, Strategy::DirectTensorNetwork);
    ASSERT_EQ(result.plan().regions.size(), 1U);
    ASSERT_TRUE(result.plan().regions[0].tensor_network_actual_peak_bytes.has_value());

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const std::uint64_t actual = *result.plan().regions[0].tensor_network_actual_peak_bytes;
    std::uint64_t estimated = 0;
    for (const CandidateSummary& c : result.plan().candidates) {
        if (c.strategy == Strategy::DirectTensorNetwork) {
            estimated = c.estimated_peak_bytes;
        }
    }
    EXPECT_GT(actual, 0U);
    EXPECT_LE(actual, estimated)
        << "the candidate table's estimated peak must bound the region's measured peak, or the "
           "two numbers explain() puts side by side are not comparable";

    const std::string text = result.explain();
    EXPECT_NE(text.find("strategy   direct_tensor_network"), std::string::npos) << text;
    EXPECT_NE(text.find("peak_bytes="), std::string::npos) << text;
    EXPECT_NE(text.find("actual_peak_bytes="), std::string::npos) << text;
}

}  // namespace
}  // namespace menos
