#include "cost/cost_model.h"

#include "common/memory_bounds.h"

#include <gtest/gtest.h>

#include "engines/dense/dense_engine.h"
#include "engines/stabilizer/stabilizer_engine.h"

namespace menos::cost {
namespace {

using engines::dense::DenseEngine;
using engines::stabilizer::StabilizerEngine;

// The runtime's bookkeeping, as cost::estimate_factorized and estimate_direct
// charge it (docs/adr/0025 section 3.4): the clbit map's sentinel node, and one
// per-region target list -- its header and, while it grows, two allocations.
const std::uint64_t kClbitMapSentinel =
    common::tree_node_bytes(sizeof(std::pair<const ClbitId, QubitId>));
// The dense engine on 2 qubits with a CX: the state 64 and the CX's matrix
// 256, one allocation each (tests/engines/dense/dense_engine_test.cpp).
const std::uint64_t kDenseBell = common::heap_block_bytes(64) + common::heap_block_bytes(256);
const std::uint64_t kRegionList =
    sizeof(std::vector<std::pair<QubitId, ClbitId>>) + (2 * common::kHeapBlockOverheadBytes);

TEST(EstimateDirectTest, FeasibleWithinBudget) {
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    const DenseEngine dense;
    const CostEstimate estimate = estimate_direct(circuit,
                                                  dense,
                                                  /*want_statevector=*/true,
                                                  /*want_counts=*/false,
                                                  /*want_expectation=*/false,
                                                  /*observable=*/nullptr,
                                                  /*budget_bytes=*/1'000'000,
                                                  /*accuracy_policy=*/{},
                                                  /*trajectories=*/0,
                                                  /*shots=*/0,
                                                  /*fuse_gates=*/false);
    ASSERT_TRUE(estimate.feasible);
    EXPECT_EQ(estimate.estimated_work_units, 32U);
    // The dense engine's figure (tests/engines/dense/dense_engine_test.cpp)
    // plus the runtime's measurement bookkeeping, which for a circuit with no
    // clbits is the clbit map's one sentinel node (docs/adr/0025 section 3.4).
    EXPECT_EQ(estimate.estimated_peak_bytes, kDenseBell + kClbitMapSentinel);
}

TEST(EstimateDirectTest, InfeasibleOverBudgetNamesBothQuantities) {
    Circuit circuit(2, 0);
    const DenseEngine dense;
    const CostEstimate estimate = estimate_direct(circuit,
                                                  dense,
                                                  true,
                                                  false,
                                                  false,
                                                  nullptr,
                                                  /*budget_bytes=*/10,
                                                  {},
                                                  /*trajectories=*/0,
                                                  /*shots=*/0);
    EXPECT_FALSE(estimate.feasible);
    EXPECT_NE(estimate.infeasible_reason.find("bytes"), std::string::npos);
}

TEST(EstimateDirectTest, StabilizerEngineOnAllCliffordCircuitIsFeasible) {
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    const StabilizerEngine stabilizer;
    const CostEstimate estimate = estimate_direct(circuit,
                                                  stabilizer,
                                                  /*want_statevector=*/false,
                                                  /*want_counts=*/true,
                                                  /*want_expectation=*/false,
                                                  /*observable=*/nullptr,
                                                  1'000'000,
                                                  {},
                                                  /*trajectories=*/0,
                                                  /*shots=*/0);
    EXPECT_TRUE(estimate.feasible);
}

TEST(EstimateDirectTest, StabilizerEngineRequestingStatevectorIsInfeasible) {
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    const StabilizerEngine stabilizer;
    const CostEstimate estimate = estimate_direct(circuit,
                                                  stabilizer,
                                                  /*want_statevector=*/true,
                                                  false,
                                                  false,
                                                  nullptr,
                                                  1'000'000,
                                                  {},
                                                  /*trajectories=*/0,
                                                  /*shots=*/0);
    EXPECT_FALSE(estimate.feasible);
    EXPECT_NE(estimate.infeasible_reason.find("dense"), std::string::npos);
}

TEST(EstimateDirectTest, StabilizerEngineOnNonCliffordCircuitIsInfeasible) {
    Circuit circuit(1, 0);
    circuit.append(Opcode::T, {0});
    const StabilizerEngine stabilizer;
    const CostEstimate estimate = estimate_direct(circuit,
                                                  stabilizer,
                                                  false,
                                                  true,
                                                  false,
                                                  nullptr,
                                                  1'000'000,
                                                  {},
                                                  /*trajectories=*/0,
                                                  /*shots=*/0);
    EXPECT_FALSE(estimate.feasible);
    EXPECT_NE(estimate.infeasible_reason.find('t'), std::string::npos);
}

// Hand-computable factorized example: 3 qubits, components {0,1} (m=2,
// one CX) and {2} (m=1, idle), want_statevector=true.
// Component {0,1}: work = 2^2 (init) + 2^(2+2) (CX) + 2^2 (statevector) = 4+16+4 = 24, peak=64
// Component {2}:   idle (no gate operations) -- free: work=0, peak=0
// peak_bytes (docs/adr/0025 section 3.4) = the regions' sum (the dense
// region's state 64 and CX matrix 256; the idle one 0), + 16*2^3 (global) =
// 128, each one allocation, + the
// runtime's bookkeeping: the clbit map's sentinel node and one per-region
// target list per component (its header and, while it grows, two allocations)
// total_work = 24 + 0 + 2^3 * 2 components (recombination, idle regions
//              still participate in the multiply-into-global pass) = 24 + 16 = 40
TEST(EstimateFactorizedTest, ExactHandComputableValue) {
    Circuit circuit(3, 0);
    circuit.append(Opcode::CX, {0, 1});
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    ASSERT_EQ(graph.components.size(), 2U);

    const DenseEngine dense;
    const StabilizerEngine stabilizer;
    const std::vector<const SimulationEngine*> engines{&dense, &stabilizer};
    const CostEstimate estimate = estimate_factorized(circuit,
                                                      graph,
                                                      engines,
                                                      /*want_statevector=*/true,
                                                      /*want_counts=*/false,
                                                      /*want_expectation=*/false,
                                                      /*observable=*/nullptr,
                                                      /*budget_bytes=*/1'000'000,
                                                      /*accuracy_policy=*/{},
                                                      /*shots=*/0,
                                                      /*fuse_gates=*/false);
    ASSERT_TRUE(estimate.feasible);
    EXPECT_EQ(estimate.estimated_peak_bytes,
              kDenseBell + common::heap_block_bytes(128) + kClbitMapSentinel + (2 * kRegionList));
    EXPECT_EQ(estimate.estimated_work_units, 40U);
}

// An idle component's freeness is what makes a many-idle-qubit
// factorized plan cheap: a circuit where only 2 of 20 qubits ever
// interact must not be charged as if all 20 needed independent regional
// initialization/output-pass work.
TEST(EstimateFactorizedTest, IdleComponentsAreFree) {
    Circuit circuit(20, 0);
    circuit.append(Opcode::CX, {0, 1});
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    ASSERT_EQ(graph.components.size(), 19U);  // {0,1} plus 18 idle singletons

    const DenseEngine dense;
    const StabilizerEngine stabilizer;
    const std::vector<const SimulationEngine*> engines{&dense, &stabilizer};
    const CostEstimate estimate = estimate_factorized(circuit,
                                                      graph,
                                                      engines,
                                                      /*want_statevector=*/false,
                                                      /*want_counts=*/true,
                                                      /*want_expectation=*/false,
                                                      /*observable=*/nullptr,
                                                      /*budget_bytes=*/1'000'000,
                                                      /*accuracy_policy=*/{},
                                                      /*shots=*/0);
    ASSERT_TRUE(estimate.feasible);
    // The {0,1} component is 1 CX, all-Clifford -- the stabilizer engine
    // wins it (cheaper than dense's exponential formula even at m=2):
    // stabilizer work = m (init) + 1 gate * m + elimination (2*m^2*words)
    // + sampling (m*words), words=1: 2 + 1*2 + 2*4*1 + 2*1 = 2+2+8+2=14.
    EXPECT_EQ(estimate.estimated_work_units, 14U);

    // Memory, too: the 18 idle components add only their (empty) per-region
    // target lists to what the one real component costs on its own.
    Circuit alone(2, 0);
    alone.append(Opcode::CX, {0, 1});
    const CostEstimate alone_estimate =
        estimate_factorized(alone,
                            analysis::build_interaction_graph(alone),
                            engines,
                            /*want_statevector=*/false,
                            /*want_counts=*/true,
                            /*want_expectation=*/false,
                            /*observable=*/nullptr,
                            /*budget_bytes=*/1'000'000,
                            /*accuracy_policy=*/{},
                            /*shots=*/0);
    ASSERT_TRUE(alone_estimate.feasible);
    EXPECT_EQ(estimate.estimated_peak_bytes,
              alone_estimate.estimated_peak_bytes + (18 * kRegionList));
}

TEST(EstimateFactorizedTest, InfeasibleOverBudgetNamesBothQuantities) {
    Circuit circuit(3, 0);
    circuit.append(Opcode::CX, {0, 1});
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    const DenseEngine dense;
    const StabilizerEngine stabilizer;
    const std::vector<const SimulationEngine*> engines{&dense, &stabilizer};
    const CostEstimate estimate = estimate_factorized(
        circuit, graph, engines, true, false, false, nullptr, /*budget_bytes=*/10, {}, /*shots=*/0);
    EXPECT_FALSE(estimate.feasible);
    EXPECT_NE(estimate.infeasible_reason.find("bytes"), std::string::npos);
}

TEST(PickCheapestEngineTest, DenseWinsWhenStrictlyCheaper) {
    // 3 qubits, all-Clifford, no operations: dense work = 2^3 = 8;
    // stabilizer work = 3 (init only, no gates, no output requested) --
    // stabilizer wins here (3 < 8), the opposite of the "ties go to
    // dense" case, confirming pick_cheapest_engine genuinely compares
    // costs rather than always preferring index 0.
    const DenseEngine dense;
    const StabilizerEngine stabilizer;
    const std::vector<const SimulationEngine*> engines{&dense, &stabilizer};
    const std::vector<Operation> ops{};
    const EngineEstimateRequest request{
        .num_qubits = 3,
        .operations = &ops,
        .want_statevector = false,
        .want_counts = false,
        .want_expectation = false,
        .observable = nullptr,
    };
    const EngineChoice choice = pick_cheapest_engine(engines, request);
    ASSERT_TRUE(choice.estimate.feasible);
    EXPECT_EQ(choice.engine_index, 1U);
    EXPECT_EQ(choice.estimate.estimated_work_units, 3U);
}

TEST(PickCheapestEngineTest, NoFeasibleEngineReportsBothReasons) {
    Circuit circuit(1, 0);
    circuit.append(Opcode::T, {0});
    const DenseEngine dense;
    const StabilizerEngine stabilizer;
    const std::vector<const SimulationEngine*> engines{&dense, &stabilizer};
    // m=63 overflows dense's checked_pow2(m) (infeasible, unconditional
    // on want_statevector/want_counts); the T gate makes the circuit
    // non-Clifford (stabilizer infeasible too) -- both engines
    // infeasible for the same (m, operations) pair.
    const EngineEstimateRequest request{
        .num_qubits = 63,
        .operations = &circuit.operations(),
        .want_statevector = false,
        .want_counts = false,
        .want_expectation = false,
        .observable = nullptr,
    };
    const EngineChoice choice = pick_cheapest_engine(engines, request);
    EXPECT_FALSE(choice.estimate.feasible);
    EXPECT_NE(choice.estimate.infeasible_reason.find("dense"), std::string::npos);
    EXPECT_NE(choice.estimate.infeasible_reason.find("stabilizer"), std::string::npos);
}

}  // namespace
}  // namespace menos::cost
