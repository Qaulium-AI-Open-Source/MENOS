#include "engines/dense/dense_engine.h"

#include <gtest/gtest.h>

#include "common/memory_bounds.h"

namespace menos::engines::dense {
namespace {

CostEstimate dense_estimate(const DenseEngine& engine,
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

TEST(DenseEngineTest, NameAndCapabilities) {
    const DenseEngine engine;
    EXPECT_EQ(engine.name(), "dense");
    EXPECT_EQ(engine.engine_id(), Engine::Dense);
    const EngineCapabilities caps = engine.capabilities();
    EXPECT_TRUE(caps.counts);
    EXPECT_TRUE(caps.statevector);
    EXPECT_TRUE(caps.expectation);
}

// Hand-computable: m=2, ops = [H(1 operand), CX(2 operands)],
// want_statevector=true, want_counts=false.
// work = 2^2 (init) + 2^(2+1) (H) + 2^(2+2) (CX) + 2^2 (statevector pass)
//      = 4 + 8 + 16 + 4 = 32
// peak_bytes = the state 16 * 2^2 = 64, plus the largest gate's transient:
// CX's generated 4 x 4 matrix, 16 * 16 = 256 (H's is 64), each one allocation
// (common::heap_block_bytes). The statevector is moved into the result, so it
// costs nothing more (docs/adr/0025 section 3.4; until Phase 21 the figure was
// the state alone, 64).
// The work-unit formula is relocated verbatim from Phase 3's cost::region_cost
// (docs/BUILD_PLAN.md Phase 4: "refactor the dense engine behind the
// interface without changing behavior").
TEST(DenseEngineTest, ExactHandComputableValueWithStatevector) {
    const DenseEngine engine;
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::CX, {0, 1}, {}},
    };
    const CostEstimate estimate =
        dense_estimate(engine, 2, ops, /*want_statevector=*/true, /*want_counts=*/false);
    ASSERT_TRUE(estimate.feasible);
    EXPECT_EQ(estimate.estimated_work_units, 32U);
    EXPECT_EQ(estimate.estimated_peak_bytes,
              common::heap_block_bytes(64) + common::heap_block_bytes(256));
}

TEST(DenseEngineTest, ExactHandComputableValueWithCounts) {
    const DenseEngine engine;
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::CX, {0, 1}, {}},
    };
    const CostEstimate estimate =
        dense_estimate(engine, 2, ops, /*want_statevector=*/false, /*want_counts=*/true);
    ASSERT_TRUE(estimate.feasible);
    EXPECT_EQ(estimate.estimated_work_units, 32U);
}

TEST(DenseEngineTest, BothOutputsRequestedAddsBothPassTerms) {
    const DenseEngine engine;
    const std::vector<Operation> ops{GateOp{Opcode::H, {0}, {}}};
    // work = 2^2 (init) + 2^(2+1) (H) + 2^2 (statevector) + 2^2 (counts) = 4+8+4+4 = 20
    const CostEstimate estimate = dense_estimate(engine, 2, ops, true, true);
    ASSERT_TRUE(estimate.feasible);
    EXPECT_EQ(estimate.estimated_work_units, 20U);
}

TEST(DenseEngineTest, MeasureOpContributesNoPerOperationWork) {
    const DenseEngine engine;
    const std::vector<Operation> with_measure{
        GateOp{Opcode::H, {0}, {}},
        MeasureOp{0, 0},
    };
    const std::vector<Operation> without_measure{GateOp{Opcode::H, {0}, {}}};
    const CostEstimate with = dense_estimate(engine, 2, with_measure, false, false);
    const CostEstimate without = dense_estimate(engine, 2, without_measure, false, false);
    ASSERT_TRUE(with.feasible);
    ASSERT_TRUE(without.feasible);
    EXPECT_EQ(with.estimated_work_units, without.estimated_work_units);
}

TEST(DenseEngineTest, EmptyRegionIsJustInitialization) {
    const DenseEngine engine;
    const CostEstimate estimate = dense_estimate(engine, 3, {}, false, false);
    ASSERT_TRUE(estimate.feasible);
    EXPECT_EQ(estimate.estimated_work_units, 8U);  // 2^3
    // 16 * 2^3, one allocation (docs/adr/0025 section 3.4).
    EXPECT_EQ(estimate.estimated_peak_bytes, common::heap_block_bytes(128));
}

TEST(DenseEngineTest, OverflowingOperandTermIsInfeasibleNotWraparound) {
    // m=60, a 2-qubit op: 2^(60+2) = 2^62 fits, but 2^62 + init (2^60) +
    // other terms could still fit in uint64_t; push m higher so the
    // per-operation checked_pow2 call itself overflows (m + k > 62).
    const DenseEngine engine;
    const std::vector<Operation> ops{GateOp{Opcode::CX, {0, 1}, {}}};
    const CostEstimate estimate = dense_estimate(engine, 62, ops, false, false);
    EXPECT_FALSE(estimate.feasible);
    EXPECT_FALSE(estimate.infeasible_reason.empty());
    EXPECT_EQ(estimate.estimated_work_units, 0U);
    EXPECT_EQ(estimate.estimated_peak_bytes, 0U);
}

TEST(DenseEngineTest, OverflowingInitializationIsInfeasible) {
    const DenseEngine engine;
    const CostEstimate estimate = dense_estimate(engine, 63, {}, false, false);
    EXPECT_FALSE(estimate.feasible);
    EXPECT_FALSE(estimate.infeasible_reason.empty());
}

TEST(DenseEngineTest, ExecuteBellCircuitProducesCorrelatedCountsAndStatevector) {
    const DenseEngine engine;
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::CX, {0, 1}, {}},
    };
    const std::vector<std::pair<QubitId, ClbitId>> targets{{0, 0}, {1, 1}};
    const EngineExecutionRequest request{
        .num_qubits = 2,
        .operations = ops,
        .want_statevector = true,
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
    ASSERT_TRUE(result.statevector.has_value());
    // ASSERT_TRUE above already guarantees this; clang-tidy can't see
    // through the gtest macro.
    EXPECT_EQ(result.statevector->size(), 4U);  // NOLINT(bugprone-unchecked-optional-access)
    ASSERT_TRUE(result.counts.has_value());
    std::uint64_t total = 0;
    for (const auto& [key, count] : *result.counts) {  // NOLINT(bugprone-unchecked-optional-access)
        EXPECT_TRUE(key.test(0) == key.test(1)) << "Bell state should only measure |00> or |11>";
        total += count;
    }
    EXPECT_EQ(total, 1000U);
    EXPECT_EQ(result.actual_bytes, 4U * sizeof(std::complex<double>));
}

TEST(DenseEngineTest, MultiRegionRequestReturnsCountsSamplerNotHistogram) {
    const DenseEngine engine;
    const std::vector<Operation> ops{GateOp{Opcode::H, {0}, {}}};
    const std::vector<std::pair<QubitId, ClbitId>> targets{{0, 0}};
    const EngineExecutionRequest request{
        .num_qubits = 1,
        .operations = ops,
        .want_statevector = false,
        .want_counts = true,
        .want_expectation = false,
        .shots = 10,
        .measure_targets = targets,
        .bitstring_width = 1,
        .seed = 1,
        .global_phase = 0.0,
        .budget_bytes = 1'000'000,
        .is_sole_region = false,
        .observable = nullptr,
    };
    const EngineExecutionResult result = engine.execute(request);
    EXPECT_FALSE(result.counts.has_value());
    ASSERT_NE(result.counts_sampler, nullptr);
}

// --- Y-convention expectation cases, values verified against
// qiskit.quantum_info.Statevector.expectation_value this session
// (docs/BUILD_PLAN.md Phase 5's required "explicit cases for Y on 1, 2
// and 3 qubits confirm the sign convention"; Qiskit 2.5.2,
// tools/.venv). Qiskit's SparsePauliOp label convention is big-endian
// (leftmost character = highest qubit index); MENOS's PauliString bit j
// is qubit j, so a label is translated per-qubit, not read left-to-right. ---

std::complex<double> dense_expectation(const std::vector<Operation>& ops,
                                       std::uint32_t n,
                                       const PauliString& pauli) {
    const DenseEngine engine;
    const Observable obs{{{pauli, {1.0, 0.0}}}};
    const EngineExecutionRequest request{
        .num_qubits = n,
        .operations = ops,
        .want_statevector = false,
        .want_counts = false,
        .want_expectation = true,
        .shots = 0,
        .measure_targets = {},
        .bitstring_width = 0,
        .seed = 0,
        .global_phase = 0.0,
        .budget_bytes = 1'000'000,
        .is_sole_region = true,
        .observable = &obs,
    };
    const EngineExecutionResult result = engine.execute(request);
    return result.expectation_terms.value()[0];  // NOLINT(bugprone-unchecked-optional-access)
}

TEST(DenseEngineYConventionTest, OneQubitSHZeroMatchesQiskit) {
    // S*H|0>: Qiskit gives <Y>=1, <X>=0, <Z>=0.
    const std::vector<Operation> ops{GateOp{Opcode::H, {0}, {}}, GateOp{Opcode::S, {0}, {}}};
    PauliString y{1, {0b1}, {0b1}};
    PauliString x{1, {0b1}, {0b0}};
    PauliString z{1, {0b0}, {0b1}};
    EXPECT_NEAR(dense_expectation(ops, 1, y).real(), 1.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 1, x).real(), 0.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 1, z).real(), 0.0, 1e-12);
}

TEST(DenseEngineYConventionTest, TwoQubitHCxSMatchesQiskit) {
    // H(0), CX(0,1), S(0) on |00>: Qiskit gives <YX>=1, <XY>=1, <ZZ>=1,
    // <YY>=0, <XX>=0 (Qiskit label order: qubit1 then qubit0).
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}}, GateOp{Opcode::CX, {0, 1}, {}}, GateOp{Opcode::S, {0}, {}}};
    // Qiskit "YX": qubit0=X, qubit1=Y.
    PauliString yx{2, {0b11}, {0b10}};
    // Qiskit "XY": qubit0=Y, qubit1=X.
    PauliString xy{2, {0b11}, {0b01}};
    PauliString zz{2, {0b00}, {0b11}};
    PauliString yy{2, {0b11}, {0b11}};
    PauliString xx{2, {0b11}, {0b00}};
    EXPECT_NEAR(dense_expectation(ops, 2, yx).real(), 1.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 2, xy).real(), 1.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 2, zz).real(), 1.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 2, yy).real(), 0.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 2, xx).real(), 0.0, 1e-12);
}

TEST(DenseEngineYConventionTest, ThreeQubitProductStateMatchesQiskit) {
    // Product state (S*H|0>) x (H|0>) x (S*H|0>) on qubits 0,1,2: Qiskit
    // gives <YII>=1, <IYI>=0, <IIY>=1, <YIY>=1, <YYI>=0, <XII>=0, <ZII>=0.
    const std::vector<Operation> ops{
        GateOp{Opcode::H, {0}, {}},
        GateOp{Opcode::S, {0}, {}},
        GateOp{Opcode::H, {1}, {}},
        GateOp{Opcode::H, {2}, {}},
        GateOp{Opcode::S, {2}, {}},
    };
    PauliString y_on_0{3, {0b001}, {0b001}};
    PauliString y_on_1{3, {0b010}, {0b010}};
    PauliString y_on_2{3, {0b100}, {0b100}};
    PauliString y_on_0_and_2{3, {0b101}, {0b101}};
    PauliString y_on_0_and_1{3, {0b011}, {0b011}};
    PauliString x_on_0{3, {0b001}, {0b000}};
    PauliString z_on_0{3, {0b000}, {0b001}};
    EXPECT_NEAR(dense_expectation(ops, 3, y_on_0).real(), 1.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 3, y_on_1).real(), 0.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 3, y_on_2).real(), 1.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 3, y_on_0_and_2).real(), 1.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 3, y_on_0_and_1).real(), 0.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 3, x_on_0).real(), 0.0, 1e-12);
    EXPECT_NEAR(dense_expectation(ops, 3, z_on_0).real(), 0.0, 1e-12);
}

}  // namespace
}  // namespace menos::engines::dense
