// docs/BUILD_PLAN.md Phase 4 acceptance criteria 2, 3, 4 -- exercised
// through the full C++ API (run()), not just the unit-level
// StabilizerEngine/region_builder tests.
//
// Two of these criteria (2 and 3) are re-verified below against Phase
// 4's own engine set (dense, stabilizer, pauli -- no MPS), via
// plan_job's own lower-level building blocks (cost::estimate_direct,
// plan::build_direct_region, plan::build_factorized_regions), rather
// than run() itself: run() always uses the full, current
// available_engines() (Phase 6: + MPS), and MPS now legitimately wins
// both of those circuits on cost (its bond stays small -- 2 throughout
// the 500-qubit chain, 1 for the isolated single-qubit T-only
// component -- so its poly(bond) cost undercuts stabilizer's/dense's
// poly(qubit-count) cost). That is not a regression of Phase 4's own
// gate, only evidence that available_engines() has grown since Phase
// 4 -- each historical test is immediately followed by a Phase 6
// counterpart showing MPS winning the same shape of case through the
// normal, full run() path.

#include "menos/menos.h"

#include <gtest/gtest.h>

#include <algorithm>

#include "analysis/interaction_graph.h"
#include "common/error.h"
#include "cost/cost_model.h"
#include "engines/dense/dense_engine.h"
#include "engines/pauli/pauli_propagation_engine.h"
#include "engines/stabilizer/stabilizer_engine.h"
#include "ir/depth.h"
#include "plan/job_spec.h"
#include "plan/region_builder.h"
#include "runtime/runtime.h"

namespace menos {
namespace {

using engines::dense::DenseEngine;
using engines::pauli::PauliPropagationEngine;
using engines::stabilizer::StabilizerEngine;

// Phase 4's own available_engines() (dense, stabilizer, pauli) -- see
// tests/plan/region_builder_test.cpp's identical helper and its own
// comment for why this exists.
std::vector<const SimulationEngine*> phase_4_engines() {
    static const DenseEngine dense;
    static const StabilizerEngine stabilizer;
    static const PauliPropagationEngine pauli;
    return {&dense, &stabilizer, &pauli};
}

Circuit make_500_qubit_clifford_circuit() {
    // A brickwork H+CX circuit, entirely Clifford, entangling every
    // qubit into one connected component (so DirectStabilizer, not
    // Factorized, is the only route -- the interaction graph here has
    // exactly one component).
    constexpr std::uint32_t n = 500;
    Circuit circuit(n, n);
    for (QubitId q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {q});
    }
    for (QubitId q = 0; q + 1 < n; ++q) {
        circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    for (QubitId q = 0; q < n; ++q) {
        circuit.append_measure(q, q);
    }
    return circuit;
}

// Acceptance criterion 2 (part 1): a 500-qubit Clifford circuit with
// counts output runs through Phase 4's own engine set with correct
// count keys, choosing the stabilizer engine (dense is infeasible at
// this scale -- checked_pow2 overflows building 2^500 bytes -- so this
// is not a live cost comparison against dense, only confirmation that
// stabilizer is feasible and dense is not, exactly as Phase 4 gated).
TEST(StabilizerScaleTest, FiveHundredQubitCliffordCircuitProducesCorrectCountKeys) {
    const Circuit circuit = make_500_qubit_clifford_circuit();
    const DenseEngine dense_engine;
    const StabilizerEngine stabilizer_engine;
    constexpr std::uint64_t kBudget = 1'000'000'000;

    const CostEstimate dense_estimate = cost::estimate_direct(circuit,
                                                              dense_engine,
                                                              false,
                                                              true,
                                                              false,
                                                              nullptr,
                                                              kBudget,
                                                              {},
                                                              /*trajectories=*/0,
                                                              /*shots=*/0);
    const CostEstimate stabilizer_estimate = cost::estimate_direct(circuit,
                                                                   stabilizer_engine,
                                                                   false,
                                                                   true,
                                                                   false,
                                                                   nullptr,
                                                                   kBudget,
                                                                   {},
                                                                   /*trajectories=*/0,
                                                                   /*shots=*/0);
    ASSERT_FALSE(dense_estimate.feasible);
    ASSERT_TRUE(stabilizer_estimate.feasible);

    Plan plan{};  // value-initialized: light_cone's numeric fields must
                  // not be left indeterminate (Plan is a plain
                  // aggregate; `Plan plan;` alone would default-, not
                  // value-, initialize them).
    plan.strategy = Strategy::DirectStabilizer;
    plan.exact = true;
    plan.regions.push_back(plan::build_direct_region(
        circuit, Engine::Stabilizer, Backend::Cpu, stabilizer_estimate, nullptr));
    plan.reduced_circuit = circuit;
    plan.estimated_state_bytes = stabilizer_estimate.estimated_peak_bytes;
    plan.estimated_work_units = stabilizer_estimate.estimated_work_units;
    plan.circuit_summary = CircuitSummary{
        .num_qubits = circuit.num_qubits(),
        .num_clbits = circuit.num_clbits(),
        .num_operations = circuit.operations().size(),
        .depth = ir::depth(circuit),
        .global_phase = circuit.global_phase,
    };
    // Every qubit is measured (want_counts covers all 500), so
    // light-cone reduction is a genuine no-op here -- these fields
    // reflect that truthfully, not a placeholder.
    plan.light_cone = LightConeSummary{
        .original_num_qubits = circuit.num_qubits(),
        .original_num_operations = circuit.operations().size(),
        .reduced_num_qubits = circuit.num_qubits(),
        .reduced_num_operations = circuit.operations().size(),
    };

    const plan::JobSpec job_spec{.circuit = circuit,
                                 .shots = 5,
                                 .want_statevector = false,
                                 .seed = 7,
                                 .memory_budget_bytes = kBudget,
                                 .expectation = std::nullopt};
    const runtime::Runtime rt;
    const Result result = rt.execute(plan, job_spec);

    EXPECT_EQ(result.plan().regions[0].engine, Engine::Stabilizer);
    std::uint64_t total = 0;
    for (const auto& [key, count] : result.counts()) {
        EXPECT_EQ(key.num_bits(), 500U);
        total += count;
    }
    EXPECT_EQ(total, 5U);
}

// Phase 6: the same 500-qubit circuit through the normal, full run()
// path (every engine, including MPS, available) legitimately picks
// DirectMPS instead -- its bond stays exactly 2 across every one of
// the 499 cuts in this brickwork chain, so its cost is far below
// stabilizer's own poly(qubit-count) formula at this scale. This is
// not a regression of the criterion above (still verified unchanged
// under Phase 4's own engine set), only evidence of the planner
// correctly using a newer, cheaper, still-exact route.
TEST(StabilizerScaleTest, FiveHundredQubitCliffordCircuitRoutesToMPSWhenAvailable) {
    const Circuit circuit = make_500_qubit_clifford_circuit();
    const Result result =
        run(circuit, RunOptions{.shots = 5, .want_statevector = false, .seed = 7});

    EXPECT_EQ(result.plan().strategy, Strategy::DirectMPS);
    ASSERT_EQ(result.plan().regions.size(), 1U);
    EXPECT_EQ(result.plan().regions[0].engine, Engine::MPS);
    EXPECT_TRUE(result.exact());

    std::uint64_t total = 0;
    for (const auto& [key, count] : result.counts()) {
        EXPECT_EQ(key.num_bits(), 500U);
        total += count;
    }
    EXPECT_EQ(total, 5U);
}

// Acceptance criterion 2 (part 2): the same circuit, with a statevector
// request, throws ResourceExceeded from the planner before any
// allocation, naming that only the dense engine can produce one.
TEST(StabilizerScaleTest, FiveHundredQubitStatevectorRequestThrowsNamingDense) {
    const Circuit circuit = make_500_qubit_clifford_circuit();
    try {
        static_cast<void>(
            run(circuit, RunOptions{.shots = 0, .want_statevector = true, .seed = 7}));
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("only the dense engine can produce a statevector"),
                  std::string::npos)
            << message;
    }
}

// Acceptance criterion 3: a circuit with one Clifford component and one
// non-Clifford component runs each on the right engine, and explain()
// shows the engine per region and why the other was rejected. Verified
// against Phase 4's own engine set (see the file-level comment above).
TEST(StabilizerScaleTest, MixedComponentsRunOnTheRightEngineAndExplainShowsWhy) {
    // Component {0,1}: H+CX, all-Clifford. Component {2}: T only (not
    // Clifford), disjoint from {0,1}.
    Circuit circuit(3, 3);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::T, {2});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    circuit.append_measure(2, 2);

    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    Plan plan{};  // value-initialized -- see the sibling test above's
                  // own comment on why `Plan plan;` alone is unsafe.
    plan.strategy = Strategy::Factorized;
    plan.exact = true;
    plan.regions = plan::build_factorized_regions(circuit,
                                                  graph,
                                                  phase_4_engines(),
                                                  /*want_statevector=*/false,
                                                  /*want_counts=*/true,
                                                  /*want_expectation=*/false,
                                                  /*observable=*/nullptr,
                                                  /*budget_bytes=*/1'000'000'000ULL,
                                                  {},
                                                  /*shots=*/10);
    plan.reduced_circuit = circuit;
    ASSERT_EQ(plan.regions.size(), 2U);
    for (const Region& region : plan.regions) {
        plan.estimated_state_bytes =
            std::max(plan.estimated_state_bytes, region.estimated_state_bytes);
        plan.estimated_work_units += region.estimated_work_units;
    }
    plan.circuit_summary = CircuitSummary{
        .num_qubits = circuit.num_qubits(),
        .num_clbits = circuit.num_clbits(),
        .num_operations = circuit.operations().size(),
        .depth = ir::depth(circuit),
        .global_phase = circuit.global_phase,
    };
    plan.light_cone = LightConeSummary{
        .original_num_qubits = circuit.num_qubits(),
        .original_num_operations = circuit.operations().size(),
        .reduced_num_qubits = circuit.num_qubits(),
        .reduced_num_operations = circuit.operations().size(),
    };

    const plan::JobSpec job_spec{.circuit = circuit,
                                 .shots = 10,
                                 .want_statevector = false,
                                 .seed = 1,
                                 .memory_budget_bytes = 1'000'000,
                                 .expectation = std::nullopt};
    const runtime::Runtime rt;
    const Result result = rt.execute(plan, job_spec);

    const Region* clifford_region = nullptr;
    const Region* non_clifford_region = nullptr;
    for (const Region& region : result.plan().regions) {
        if (region.local_to_global == std::vector<QubitId>{2}) {
            non_clifford_region = &region;
        } else {
            clifford_region = &region;
        }
    }
    ASSERT_NE(clifford_region, nullptr);
    ASSERT_NE(non_clifford_region, nullptr);
    EXPECT_EQ(clifford_region->engine, Engine::Stabilizer);
    EXPECT_EQ(non_clifford_region->engine, Engine::Dense);

    const std::string text = result.explain();
    EXPECT_NE(text.find("engine=stabilizer"), std::string::npos) << text;
    EXPECT_NE(text.find("engine=dense"), std::string::npos) << text;
    // The dense region's reason must say why the stabilizer engine was
    // rejected for it (the T gate).
    EXPECT_NE(non_clifford_region->engine_selection_reason.find('t'), std::string::npos)
        << non_clifford_region->engine_selection_reason;
}

// Phase 6: the same mixed-component circuit through the normal, full
// run() path picks MPS for the isolated single-qubit T-only component
// instead of dense -- its bond is trivially 1 (no cuts exist for a
// 1-qubit region), so its cost undercuts dense's own exponential-in-
// qubit-count formula even at this tiny scale. Not a regression of the
// criterion above (still verified unchanged under Phase 4's own engine
// set), only evidence of a newer, cheaper, still-exact route.
TEST(StabilizerScaleTest, MixedComponentsNonCliffordSingleQubitRoutesToMPSWhenAvailable) {
    Circuit circuit(3, 3);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::T, {2});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    circuit.append_measure(2, 2);

    const Result result = run(circuit, RunOptions{.shots = 10, .seed = 1});
    ASSERT_EQ(result.plan().strategy, Strategy::Factorized);
    ASSERT_EQ(result.plan().regions.size(), 2U);

    const Region* clifford_region = nullptr;
    const Region* non_clifford_region = nullptr;
    for (const Region& region : result.plan().regions) {
        if (region.local_to_global == std::vector<QubitId>{2}) {
            non_clifford_region = &region;
        } else {
            clifford_region = &region;
        }
    }
    ASSERT_NE(clifford_region, nullptr);
    ASSERT_NE(non_clifford_region, nullptr);
    EXPECT_EQ(clifford_region->engine, Engine::Stabilizer);
    EXPECT_EQ(non_clifford_region->engine, Engine::MPS);
    EXPECT_TRUE(result.exact());

    const std::string text = result.explain();
    EXPECT_NE(text.find("engine=stabilizer"), std::string::npos) << text;
    EXPECT_NE(text.find("engine=mps"), std::string::npos) << text;
}

// Acceptance criterion 4: a circuit that is mostly Clifford with one T
// in a single component keeps that component on dense, and explain()
// names the T and its operation index.
TEST(StabilizerScaleTest, MostlyCliffordComponentWithOneTStaysDenseAndExplainNamesIt) {
    Circuit circuit(4, 4);
    // One connected component {0,1,2,3}: a long run of Clifford gates
    // plus a single T, so the component is "99% Clifford" in spirit.
    circuit.append(Opcode::H, {0});
    for (int i = 0; i < 20; ++i) {
        circuit.append(Opcode::CX, {0, 1});
        circuit.append(Opcode::CX, {1, 2});
        circuit.append(Opcode::CX, {2, 3});
        circuit.append(Opcode::CX, {2, 3});
        circuit.append(Opcode::CX, {1, 2});
        circuit.append(Opcode::CX, {0, 1});
    }
    circuit.append(Opcode::T, {3});  // the one non-Clifford operation
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    circuit.append_measure(2, 2);
    circuit.append_measure(3, 3);

    const Result result = run(circuit, RunOptions{.shots = 5, .seed = 1});
    // The circuit is one connected component, so the winning strategy is
    // DirectDense (not Factorized) -- DirectStabilizer is still always
    // enumerated as candidate 1, and its own infeasibility reason is
    // exactly where explain() names the T and its index for this shape.
    ASSERT_EQ(result.plan().strategy, Strategy::DirectDense);
    ASSERT_EQ(result.plan().regions.size(), 1U);
    EXPECT_EQ(result.plan().regions[0].engine, Engine::Dense);
    // DirectMPS is also always enumerated (Phase 6), as candidate 2:
    // its own cost for this 4-qubit, ~121-operation circuit is not
    // cheaper than dense here (many local updates over the circuit's
    // full length outweigh its small, bond-capped-at-4 per-update
    // cost), so dense still wins -- only the candidate count changes.
    ASSERT_EQ(result.plan().candidates.size(), 3U);
    EXPECT_EQ(result.plan().candidates[1].strategy, Strategy::DirectStabilizer);
    const std::string& reason = result.plan().candidates[1].infeasible_reason;
    // Operation 0 is the H; operations 1..120 are the 120 CX gates (20
    // iterations * 6 each); the T is operation 121.
    EXPECT_NE(reason.find("operation 121"), std::string::npos) << reason;
    EXPECT_NE(reason.find('t'), std::string::npos) << reason;

    const std::string text = result.explain();
    EXPECT_NE(text.find(reason), std::string::npos) << text;
}

}  // namespace
}  // namespace menos
