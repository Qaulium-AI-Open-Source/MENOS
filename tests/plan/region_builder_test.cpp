#include "plan/region_builder.h"

#include <gtest/gtest.h>

#include <algorithm>

#include "analysis/interaction_graph.h"
#include "engines/dense/dense_engine.h"
#include "engines/pauli/pauli_propagation_engine.h"
#include "engines/stabilizer/stabilizer_engine.h"
#include "plan/available_engines.h"

namespace menos::plan {
namespace {

using engines::dense::DenseEngine;
using engines::pauli::PauliPropagationEngine;
using engines::stabilizer::StabilizerEngine;

// The engine set available_engines() returned as of Phase 4's own gate
// (dense, stabilizer, pauli): used only by the test below that verifies
// Phase 4's already-confirmed acceptance criterion 4 still holds
// exactly as gated, under the engine set that criterion was evaluated
// against. available_engines() itself has since grown to include MPS
// (Phase 6) -- a separate test right after this one verifies MPS can
// legitimately win the same kind of case when it is available and
// genuinely cheaper, which is not a regression of Phase 4's own gate,
// only of a later phase's literal engine list.
std::vector<const SimulationEngine*> phase_4_engines() {
    static const DenseEngine dense;
    static const StabilizerEngine stabilizer;
    static const PauliPropagationEngine pauli;
    return {&dense, &stabilizer, &pauli};
}

TEST(BuildDirectRegionTest, LocalEqualsGlobalIdentityMap) {
    Circuit circuit(3, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    const DenseEngine dense;
    const EngineEstimateRequest est_request{
        .num_qubits = circuit.num_qubits(),
        .operations = &circuit.operations(),
        .want_statevector = false,
        .want_counts = false,
        .want_expectation = false,
        .observable = nullptr,
    };
    const CostEstimate estimate = dense.estimate(est_request);
    const Region region =
        build_direct_region(circuit, Engine::Dense, Backend::Cpu, estimate, nullptr);
    EXPECT_EQ(region.local_to_global, (std::vector<QubitId>{0, 1, 2}));
    EXPECT_EQ(region.global_to_local, (std::vector<QubitId>{0, 1, 2}));
    EXPECT_EQ(region.operations.size(), circuit.operations().size());
    EXPECT_EQ(region.engine, Engine::Dense);
    EXPECT_GT(region.estimated_state_bytes, 0U);
}

TEST(BuildFactorizedRegionsTest, ContiguousComponentsRemapCorrectly) {
    // {0,1} connected, {2} idle.
    Circuit circuit(3, 0);
    circuit.append(Opcode::CX, {0, 1});
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                 graph,
                                                                 available_engines(),
                                                                 false,
                                                                 false,
                                                                 false,
                                                                 nullptr,
                                                                 1'000'000'000,
                                                                 {},
                                                                 /*shots=*/0);
    ASSERT_EQ(regions.size(), 2U);

    EXPECT_EQ(regions[0].local_to_global, (std::vector<QubitId>{0, 1}));
    ASSERT_EQ(regions[0].operations.size(), 1U);
    const auto& gate = std::get<GateOp>(regions[0].operations[0]);
    EXPECT_EQ(gate.opcode, Opcode::CX);
    EXPECT_EQ(gate.qubits,
              (std::vector<QubitId>{0, 1}));  // already local, unchanged since contiguous

    EXPECT_EQ(regions[1].local_to_global, (std::vector<QubitId>{2}));
    EXPECT_TRUE(regions[1].operations.empty());
    EXPECT_EQ(regions[1].estimated_state_bytes, 0U);  // idle
}

TEST(BuildFactorizedRegionsTest, InterleavedComponentsRemapToLocalZeroOne) {
    // {0,2} connected, {1,3} connected -- the spec's own named interleaved
    // example (docs/BUILD_PLAN.md Phase 3, 3.2/3.3).
    Circuit circuit(4, 0);
    circuit.append(Opcode::CX, {0, 2});
    circuit.append(Opcode::CX, {1, 3});
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                 graph,
                                                                 available_engines(),
                                                                 false,
                                                                 false,
                                                                 false,
                                                                 nullptr,
                                                                 1'000'000'000,
                                                                 {},
                                                                 /*shots=*/0);
    ASSERT_EQ(regions.size(), 2U);

    EXPECT_EQ(regions[0].local_to_global, (std::vector<QubitId>{0, 2}));
    ASSERT_EQ(regions[0].operations.size(), 1U);
    // Global CX(0,2) -> region 0's local ids: global 0 -> local 0, global
    // 2 -> local 1.
    EXPECT_EQ(std::get<GateOp>(regions[0].operations[0]).qubits, (std::vector<QubitId>{0, 1}));

    EXPECT_EQ(regions[1].local_to_global, (std::vector<QubitId>{1, 3}));
    ASSERT_EQ(regions[1].operations.size(), 1U);
    // Global CX(1,3) -> region 1's local ids: global 1 -> local 0, global
    // 3 -> local 1.
    EXPECT_EQ(std::get<GateOp>(regions[1].operations[0]).qubits, (std::vector<QubitId>{0, 1}));
}

TEST(BuildFactorizedRegionsTest, MeasureOpKeepsGlobalClbitButRemapsLocalQubit) {
    Circuit circuit(4, 4);
    circuit.append(Opcode::CX, {0, 2});
    circuit.append_measure(2, 3);  // global qubit 2 -> region 0's local qubit 1; clbit 3 untouched
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                 graph,
                                                                 available_engines(),
                                                                 false,
                                                                 true,
                                                                 false,
                                                                 nullptr,
                                                                 1'000'000'000,
                                                                 {},
                                                                 /*shots=*/0);
    ASSERT_EQ(regions[0].local_to_global, (std::vector<QubitId>{0, 2}));

    bool found = false;
    for (const Operation& op : regions[0].operations) {
        if (const auto* measure = std::get_if<MeasureOp>(&op)) {
            EXPECT_EQ(measure->qubit, 1U);  // global 2 -> local 1
            EXPECT_EQ(measure->clbit, 3U);  // untouched, global namespace
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST(BuildFactorizedRegionsTest, IdleMeasuredQubitStillZeroBytesWithMeasureOpPresent) {
    // Idle qubit 1, measured to clbit 0: still 0 bytes (docs/BUILD_PLAN.md
    // Phase 3, 3.2 -- outcome is statically |0>), but the MeasureOp is
    // still present in the region's own operations for shape uniformity.
    Circuit circuit(2, 1);
    circuit.append(Opcode::X, {0});  // qubit 0 its own singleton component too (1-qubit gate)
    circuit.append_measure(1, 0);
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    ASSERT_EQ(graph.components.size(), 2U);
    const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                 graph,
                                                                 available_engines(),
                                                                 false,
                                                                 true,
                                                                 false,
                                                                 nullptr,
                                                                 1'000'000'000,
                                                                 {},
                                                                 /*shots=*/0);

    // Region for qubit 1 (idle, only measured).
    const auto it = std::find_if(regions.begin(), regions.end(), [](const Region& r) {
        return r.local_to_global == std::vector<QubitId>{1};
    });
    ASSERT_NE(it, regions.end());
    EXPECT_EQ(it->estimated_state_bytes, 0U);
    ASSERT_EQ(it->operations.size(), 1U);
    EXPECT_TRUE(std::holds_alternative<MeasureOp>(it->operations[0]));
}

TEST(BuildFactorizedRegionsTest, AllCliffordComponentPicksStabilizerAndRecordsReason) {
    // {0,1}: H+CX, all-Clifford -- stabilizer is cheaper than dense here
    // (docs/BUILD_PLAN.md Phase 4), so it must win, and the reason string
    // must name the winner.
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    ASSERT_EQ(graph.components.size(), 1U);
    const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                 graph,
                                                                 available_engines(),
                                                                 false,
                                                                 true,
                                                                 false,
                                                                 nullptr,
                                                                 1'000'000'000,
                                                                 {},
                                                                 /*shots=*/0);
    ASSERT_EQ(regions.size(), 1U);
    EXPECT_EQ(regions[0].engine, Engine::Stabilizer);
    EXPECT_FALSE(regions[0].engine_selection_reason.empty());
}

TEST(BuildFactorizedRegionsTest, OneNonCliffordGateKeepsComponentOnDenseAndNamesIt) {
    // 99%-Clifford-plus-one-T (docs/BUILD_PLAN.md Phase 4 acceptance
    // criterion 4, already confirmed and gated): the component must
    // stay on dense, and the reason must name the T gate and its
    // (region-local) operation index. Evaluated against Phase 4's own
    // engine set (phase_4_engines(), not available_engines()) -- this
    // criterion was gated before MPS (Phase 6) existed, and MPS's own
    // legitimate ability to win a similarly-shaped case is verified
    // separately below, not folded into this historical assertion.
    Circuit circuit(1, 1);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::T, {0});
    circuit.append_measure(0, 0);
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                 graph,
                                                                 phase_4_engines(),
                                                                 false,
                                                                 true,
                                                                 false,
                                                                 nullptr,
                                                                 1'000'000'000,
                                                                 {},
                                                                 /*shots=*/0);
    ASSERT_EQ(regions.size(), 1U);
    EXPECT_EQ(regions[0].engine, Engine::Dense);
    EXPECT_NE(regions[0].engine_selection_reason.find("operation 2"), std::string::npos)
        << regions[0].engine_selection_reason;
    EXPECT_NE(regions[0].engine_selection_reason.find('t'), std::string::npos)
        << regions[0].engine_selection_reason;
}

// Phase 6: with MPS in the available engine set, the SAME shape of
// case (a non-Clifford single-qubit component) can legitimately route
// to MPS instead of dense, when MPS's own cost estimate is genuinely
// cheaper -- verified by hand: MPS work = 1 (init) + 3 * (bond=1 * 4)
// (three single-qubit gates, no cuts exist for a 1-qubit region) + 1
// (counts setup, num_qubits * bond^3) = 14; dense work = 2^1 (init) +
// 3 * 2^(1+1) (each single-qubit gate) + 2^1 (counts) = 16. This is
// not a regression of Phase 4's own gate (still verified unchanged,
// under Phase 4's own engine set, by the test above) -- only evidence
// that available_engines() has grown since then and the planner's own
// cost-based selection correctly reflects that growth.
// With gate fusion on (the default since Phase 24 item 7) the same three
// gates on one qubit are ONE dense pass: dense work = 2 (init) + 2^(1+1)
// (the block) + 2 (counts) = 8, under MPS's 14, so dense wins. A changed
// winner by the fused cost model, recorded in docs/adr/0028 section 17.7.
TEST(BuildFactorizedRegionsTest, WithFusionThreeGatesOnOneQubitAreOnePassAndDenseWins) {
    Circuit circuit(1, 1);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::T, {0});
    circuit.append_measure(0, 0);
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                 graph,
                                                                 available_engines(),
                                                                 false,
                                                                 true,
                                                                 false,
                                                                 nullptr,
                                                                 1'000'000'000,
                                                                 {},
                                                                 /*shots=*/0,
                                                                 /*fuse_gates=*/true);
    ASSERT_EQ(regions.size(), 1U);
    EXPECT_EQ(regions[0].engine, Engine::Dense);
    EXPECT_EQ(regions[0].estimated_work_units, 8U);
    EXPECT_EQ(regions[0].fusion.fused_blocks, 1U);
    EXPECT_EQ(regions[0].fusion.gates_absorbed, 2U);
}

TEST(BuildFactorizedRegionsTest, OneNonCliffordGateCanRouteToMPSWhenAvailableAndCheaper) {
    Circuit circuit(1, 1);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::T, {0});
    circuit.append_measure(0, 0);
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                 graph,
                                                                 available_engines(),
                                                                 false,
                                                                 true,
                                                                 false,
                                                                 nullptr,
                                                                 1'000'000'000,
                                                                 {},
                                                                 /*shots=*/0,
                                                                 /*fuse_gates=*/false);
    ASSERT_EQ(regions.size(), 1U);
    EXPECT_EQ(regions[0].engine, Engine::MPS);
    EXPECT_NE(regions[0].engine_selection_reason.find("operation 2"), std::string::npos)
        << regions[0].engine_selection_reason;
    EXPECT_NE(regions[0].engine_selection_reason.find('t'), std::string::npos)
        << regions[0].engine_selection_reason;
}

TEST(BuildFactorizedRegionsTest, StatevectorRequestKeepsAllCliffordComponentOnDense) {
    // want_statevector=true must force dense even for an all-Clifford
    // component (stabilizer's capabilities().statevector is false).
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                 graph,
                                                                 available_engines(),
                                                                 /*want_statevector=*/true,
                                                                 false,
                                                                 false,
                                                                 nullptr,
                                                                 1'000'000'000,
                                                                 {},
                                                                 /*shots=*/0);
    ASSERT_EQ(regions.size(), 1U);
    EXPECT_EQ(regions[0].engine, Engine::Dense);
}

}  // namespace
}  // namespace menos::plan
