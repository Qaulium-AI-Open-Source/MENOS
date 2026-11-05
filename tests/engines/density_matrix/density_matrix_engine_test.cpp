#include "engines/density_matrix/density_matrix_engine.h"

#include <gtest/gtest.h>

#include "common/memory_bounds.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "common/error.h"
#include "common/rng.h"
#include "common/tolerance.h"
#include "engines/dense/dense_engine.h"
#include "ir/gate_table.h"

namespace menos::engines::density_matrix {
namespace {

using Complex = std::complex<double>;
using Kraus = std::vector<std::vector<Complex>>;

constexpr std::uint64_t kBudget = std::uint64_t{1} << 30;

// Every Pauli string on n qubits (4^n of them), unit coefficients, so each
// term's raw expectation is compared on its own.
Observable all_paulis(std::uint32_t n) {
    Observable observable;
    const std::uint64_t dim = std::uint64_t{1} << n;
    for (std::uint64_t x = 0; x < dim; ++x) {
        for (std::uint64_t z = 0; z < dim; ++z) {
            observable.terms.emplace_back(PauliString{n, {x}, {z}}, Complex{1.0, 0.0});
        }
    }
    return observable;
}

EngineExecutionRequest expectation_request(const Circuit& circuit, const Observable& observable) {
    return EngineExecutionRequest{.num_qubits = circuit.num_qubits(),
                                  .operations = circuit.operations(),
                                  .want_statevector = false,
                                  .want_counts = false,
                                  .want_expectation = true,
                                  .shots = 0,
                                  .measure_targets = {},
                                  .bitstring_width = circuit.num_clbits(),
                                  .seed = 0,
                                  .global_phase = 0.0,
                                  .budget_bytes = kBudget,
                                  .is_sole_region = true,
                                  .observable = &observable};
}

std::vector<Complex> density_matrix_terms(const Circuit& circuit, const Observable& observable) {
    const DensityMatrixEngine engine;
    EngineExecutionResult result = engine.execute(expectation_request(circuit, observable));
    return result.expectation_terms.value_or(std::vector<Complex>{});
}

std::vector<Complex> dense_terms(const Circuit& circuit, const Observable& observable) {
    const dense::DenseEngine engine;
    EngineExecutionResult result = engine.execute(expectation_request(circuit, observable));
    return result.expectation_terms.value_or(std::vector<Complex>{});
}

void expect_all_close(const std::vector<Complex>& actual, const std::vector<Complex>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_LE(std::abs(actual[i] - expected[i]), common::kDefaultAbsTol)
            << "term " << i << ": " << actual[i] << " vs " << expected[i];
    }
}

Kraus depolarizing_1q(double p) {
    const double a = std::sqrt(1.0 - 3.0 * p / 4.0);
    const double b = std::sqrt(p / 4.0);
    return {{a, 0.0, 0.0, a},
            {0.0, b, b, 0.0},
            {0.0, Complex{0.0, -b}, Complex{0.0, b}, 0.0},
            {b, 0.0, 0.0, -b}};
}

Kraus amplitude_damping(double gamma) {
    return {{1.0, 0.0, 0.0, std::sqrt(1.0 - gamma)}, {0.0, std::sqrt(gamma), 0.0, 0.0}};
}

Complex term(const std::vector<Complex>& terms, std::uint32_t n, std::uint64_t x, std::uint64_t z) {
    return terms[x * (std::uint64_t{1} << n) + z];
}

// --- Noiseless circuits: rho = |psi><psi| must reproduce the dense engine. ---

TEST(DensityMatrixEngineTest, NoiselessCircuitsMatchTheDenseEngineOnEveryPauli) {
    const std::vector<Complex> iswap = {1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        Complex{0.0, 1.0},
                                        0.0,
                                        0.0,
                                        Complex{0.0, 1.0},
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0};
    Circuit circuit(3, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append(Opcode::T, {1});
    circuit.append(Opcode::RX, {2}, {0.7});
    circuit.append(Opcode::CCX, {0, 1, 2});
    circuit.append(Opcode::U, {0}, {0.1, 0.2, 0.3});
    circuit.append(Opcode::CP, {1, 2}, {0.4});
    circuit.append(Opcode::SWAP, {0, 2});
    circuit.append(Opcode::SX, {1});
    circuit.append(Opcode::RY, {0}, {1.3});
    circuit.append_unitary({2, 0}, iswap);
    circuit.append(Opcode::RZ, {2}, {-0.9});

    const Observable observable = all_paulis(3);
    expect_all_close(density_matrix_terms(circuit, observable), dense_terms(circuit, observable));
}

TEST(DensityMatrixEngineTest, AnEmptyCircuitIsTheAllZeroState) {
    const Circuit circuit(2, 0);
    const Observable observable = all_paulis(2);
    expect_all_close(density_matrix_terms(circuit, observable), dense_terms(circuit, observable));
}

// --- Channels, checked against closed forms. ---

// rho -> (1 - p) rho + p I/2 leaves <Z> = 1 - p on |0>, and <X> = 1 - p on |+>.
TEST(DensityMatrixEngineTest, DepolarizingShrinksTheBlochVectorByOneMinusP) {
    const Observable observable = all_paulis(1);
    Circuit zero(1, 0);
    zero.append_channel({0}, depolarizing_1q(0.3));
    const std::vector<Complex> zero_terms = density_matrix_terms(zero, observable);
    EXPECT_NEAR(term(zero_terms, 1, 0, 1).real(), 0.7, common::kDefaultAbsTol);     // Z
    EXPECT_NEAR(std::abs(term(zero_terms, 1, 1, 0)), 0.0, common::kDefaultAbsTol);  // X
    EXPECT_NEAR(term(zero_terms, 1, 0, 0).real(), 1.0, common::kDefaultAbsTol);     // trace

    Circuit plus(1, 0);
    plus.append(Opcode::H, {0});
    plus.append_channel({0}, depolarizing_1q(0.3));
    const std::vector<Complex> plus_terms = density_matrix_terms(plus, observable);
    EXPECT_NEAR(term(plus_terms, 1, 1, 0).real(), 0.7, common::kDefaultAbsTol);  // X
}

// Amplitude damping with gamma moves population gamma from |1> to |0>:
// <Z> = gamma - (1 - gamma) = 2 gamma - 1 on |1>.
TEST(DensityMatrixEngineTest, AmplitudeDampingMovesPopulationFromOneToZero) {
    Circuit circuit(1, 0);
    circuit.append(Opcode::X, {0});
    circuit.append_channel({0}, amplitude_damping(0.25));
    const std::vector<Complex> terms = density_matrix_terms(circuit, all_paulis(1));
    EXPECT_NEAR(term(terms, 1, 0, 1).real(), -0.5, common::kDefaultAbsTol);
}

// A one-operator channel whose Kraus operator is a unitary is that unitary.
// CX's matrix in operand order, applied as a channel on (control, target),
// must equal the CX gate: this pins the superoperator's operand ordering.
TEST(DensityMatrixEngineTest, AUnitaryKrausChannelEqualsTheGate) {
    const Observable observable = all_paulis(2);
    Circuit as_gate(2, 0);
    as_gate.append(Opcode::H, {0});
    as_gate.append(Opcode::RY, {1}, {0.4});
    as_gate.append(Opcode::CX, {0, 1});
    Circuit as_channel(2, 0);
    as_channel.append(Opcode::H, {0});
    as_channel.append(Opcode::RY, {1}, {0.4});
    as_channel.append_channel({0, 1}, {ir::generate_matrix(Opcode::CX, {})});
    expect_all_close(density_matrix_terms(as_channel, observable),
                     density_matrix_terms(as_gate, observable));
}

// Amplitude damping on qubit 1, written as a two-qubit channel K (x) I with
// operands (1, 0) -- K on operand 0 (qubit 1), identity on operand 1 (qubit 0)
// -- must equal the one-qubit channel on qubit 1.
TEST(DensityMatrixEngineTest, ATwoQubitChannelRespectsItsOperandOrder) {
    const Kraus one_qubit = amplitude_damping(0.4);
    Kraus two_qubit;
    for (const std::vector<Complex>& k : one_qubit) {
        std::vector<Complex> m(16, 0.0);
        for (std::size_t r = 0; r < 4; ++r) {
            for (std::size_t c = 0; c < 4; ++c) {
                if ((r >> 1) == (c >> 1)) {
                    m[r * 4 + c] = k[(r & 1) * 2 + (c & 1)];
                }
            }
        }
        two_qubit.push_back(m);
    }
    const Observable observable = all_paulis(2);
    Circuit reference(2, 0);
    reference.append(Opcode::X, {1});
    reference.append(Opcode::H, {0});
    reference.append_channel({1}, one_qubit);
    Circuit tested(2, 0);
    tested.append(Opcode::X, {1});
    tested.append(Opcode::H, {0});
    tested.append_channel({1, 0}, two_qubit);
    expect_all_close(density_matrix_terms(tested, observable),
                     density_matrix_terms(reference, observable));
}

// --- Counts and readout errors. ---

EngineExecutionRequest counts_request(const Circuit& circuit,
                                      std::span<const std::pair<QubitId, ClbitId>> targets,
                                      std::uint64_t shots,
                                      std::uint64_t seed,
                                      bool sole_region = true) {
    return EngineExecutionRequest{.num_qubits = circuit.num_qubits(),
                                  .operations = circuit.operations(),
                                  .want_statevector = false,
                                  .want_counts = true,
                                  .want_expectation = false,
                                  .shots = shots,
                                  .measure_targets = targets,
                                  .bitstring_width = circuit.num_clbits(),
                                  .seed = seed,
                                  .global_phase = 0.0,
                                  .budget_bytes = kBudget,
                                  .is_sole_region = sole_region,
                                  .observable = nullptr};
}

std::map<BitString, std::uint64_t> counts_of(const Circuit& circuit,
                                             std::span<const std::pair<QubitId, ClbitId>> targets,
                                             std::uint64_t shots,
                                             std::uint64_t seed) {
    const DensityMatrixEngine engine;
    EngineExecutionResult result = engine.execute(counts_request(circuit, targets, shots, seed));
    return result.counts.value_or(std::map<BitString, std::uint64_t>{});
}

// width (a clbit count) and value (the bits to set) are not realistically
// confusable at any call site, as in tests/api/run_test.cpp's make_bitstring.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
BitString bits(std::uint32_t width, std::uint64_t value) {
    BitString b(width);
    for (std::uint32_t i = 0; i < width; ++i) {
        b.set(i, ((value >> i) & 1U) != 0);
    }
    return b;
}

TEST(DensityMatrixEngineTest, CountsWithoutNoiseFollowTheDiagonal) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::X, {1});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    const std::vector<std::pair<QubitId, ClbitId>> targets = {{0, 0}, {1, 1}};
    const std::map<BitString, std::uint64_t> counts = counts_of(circuit, targets, 500, 3);
    ASSERT_EQ(counts.size(), 1U);
    EXPECT_EQ(counts.at(bits(2, 0b10)), 500U);
}

// P(0|1) = 1: a qubit in |1> is always recorded as 0.
TEST(DensityMatrixEngineTest, ACertainReadoutFlipAlwaysFlips) {
    Circuit circuit(1, 1);
    circuit.append(Opcode::X, {0});
    circuit.append_measure(0, 0, ReadoutError{{0.0, 1.0, 1.0, 0.0}});
    const std::vector<std::pair<QubitId, ClbitId>> targets = {{0, 0}};
    const std::map<BitString, std::uint64_t> counts = counts_of(circuit, targets, 1000, 5);
    ASSERT_EQ(counts.size(), 1U);
    EXPECT_EQ(counts.at(bits(1, 0)), 1000U);
}

// P(1|0) = 0.25 on |0>: over 20000 shots, 5000 ones expected, standard
// deviation sqrt(20000 * 0.25 * 0.75) ~ 61; +-400 is more than 6 sigma.
TEST(DensityMatrixEngineTest, AReadoutErrorFlipsWithItsProbability) {
    Circuit circuit(1, 1);
    circuit.append_measure(0, 0, ReadoutError{{0.75, 0.25, 0.0, 1.0}});
    const std::vector<std::pair<QubitId, ClbitId>> targets = {{0, 0}};
    const std::map<BitString, std::uint64_t> counts = counts_of(circuit, targets, 20000, 11);
    const std::uint64_t ones = counts.contains(bits(1, 1)) ? counts.at(bits(1, 1)) : 0;
    EXPECT_GT(ones, 4600U);
    EXPECT_LT(ones, 5400U);
}

TEST(DensityMatrixEngineTest, TheSameSeedGivesTheSameCountsWithReadoutErrors) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_channel({1}, depolarizing_1q(0.2));
    circuit.append_measure(0, 0, ReadoutError{{0.9, 0.1, 0.15, 0.85}});
    circuit.append_measure(1, 1);
    const std::vector<std::pair<QubitId, ClbitId>> targets = {{0, 0}, {1, 1}};
    EXPECT_EQ(counts_of(circuit, targets, 2000, 42), counts_of(circuit, targets, 2000, 42));
    EXPECT_NE(counts_of(circuit, targets, 2000, 42), counts_of(circuit, targets, 2000, 43));
}

// The readout error of a clbit written twice is that of its last write.
TEST(DensityMatrixEngineTest, TheSurvivingMeasurementsReadoutErrorApplies) {
    Circuit circuit(2, 1);
    circuit.append_measure(0, 0, ReadoutError{{0.0, 1.0, 1.0, 0.0}});  // overwritten
    circuit.append_measure(1, 0);                                      // survives, no error
    const std::vector<std::pair<QubitId, ClbitId>> targets = {{1, 0}};
    const std::map<BitString, std::uint64_t> counts = counts_of(circuit, targets, 300, 2);
    ASSERT_EQ(counts.size(), 1U);
    EXPECT_EQ(counts.at(bits(1, 0)), 300U);
}

// Readout error changes recorded counts only, never an expectation value.
TEST(DensityMatrixEngineTest, AReadoutErrorDoesNotChangeAnExpectationValue) {
    Circuit circuit(1, 1);
    circuit.append(Opcode::X, {0});
    circuit.append_measure(0, 0, ReadoutError{{0.0, 1.0, 1.0, 0.0}});
    const std::vector<Complex> terms = density_matrix_terms(circuit, all_paulis(1));
    EXPECT_NEAR(term(terms, 1, 0, 1).real(), -1.0, common::kDefaultAbsTol);
}

// The Factorized path draws through the region sampler, readout included.
TEST(DensityMatrixEngineTest, TheRegionSamplerAppliesTheReadoutError) {
    Circuit circuit(1, 1);
    circuit.append(Opcode::X, {0});
    circuit.append_measure(0, 0, ReadoutError{{0.0, 1.0, 1.0, 0.0}});
    const std::vector<std::pair<QubitId, ClbitId>> targets = {{0, 0}};
    const DensityMatrixEngine engine;
    EngineExecutionResult result =
        engine.execute(counts_request(circuit, targets, 0, 0, /*sole_region=*/false));
    ASSERT_NE(result.counts_sampler, nullptr);
    EXPECT_FALSE(result.counts.has_value());
    common::Rng rng(9);
    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(result.counts_sampler->sample_one(rng), bits(1, 0));
    }
}

// --- Resources, estimate and the engine contract. ---

TEST(DensityMatrixEngineTest, TheBudgetIsCheckedAtSixteenTimesFourToTheN) {
    Circuit circuit(3, 0);
    circuit.append(Opcode::H, {0});
    const Observable observable = all_paulis(1);
    EngineExecutionRequest request = expectation_request(circuit, observable);
    request.observable = nullptr;
    request.want_expectation = false;
    const DensityMatrixEngine engine;

    request.budget_bytes = std::uint64_t{16} * 64 - 1;
    EXPECT_THROW(static_cast<void>(engine.execute(request)), common::ResourceExceeded);
    request.budget_bytes = std::uint64_t{16} * 64;
    EXPECT_EQ(engine.execute(request).actual_bytes, 16U * 64U);
}

// n = 2: init 4^2 = 16; H (k=1) 2 * 2^(4+1) = 64; CX (k=2) 2 * 2^(4+2) = 128;
// a one-qubit channel 2^(4+2) = 64; counts 2^2 = 4. Total 276 work units.
// Peak (docs/adr/0025 section 3.4), each buffer one allocation
// (common::heap_block_bytes): rho 16 * 16 = 256, plus the largest transient,
// which here is the CX's: its generated 4 x 4 matrix and the conjugated copy,
// 2 * 256, and its column-qubit list, 2 * 4. The channel's transient (its 4 x 4
// superoperator 256, the operand lists) and the counts outputs are both
// smaller. Until Phase 21 this was 544: rho, the superoperator and the
// diagonal, without the gate's buffers.
TEST(DensityMatrixEngineTest, EstimateFollowsTheDocumentedFormulas) {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_channel({1}, depolarizing_1q(0.1));
    circuit.append_measure(0, 0);
    const DensityMatrixEngine engine;
    const CostEstimate estimate =
        engine.estimate(EngineEstimateRequest{.num_qubits = 2,
                                              .operations = &circuit.operations(),
                                              .want_statevector = false,
                                              .want_counts = true,
                                              .want_expectation = false,
                                              .observable = nullptr});
    ASSERT_TRUE(estimate.feasible) << estimate.infeasible_reason;
    EXPECT_EQ(estimate.estimated_work_units, 276U);
    EXPECT_EQ(estimate.estimated_peak_bytes,
              common::heap_block_bytes(256) + (2 * common::heap_block_bytes(256)) +
                  common::heap_block_bytes(8));
}

TEST(DensityMatrixEngineTest, AStatevectorIsRefused) {
    const Circuit circuit(1, 0);
    const DensityMatrixEngine engine;
    const CostEstimate estimate =
        engine.estimate(EngineEstimateRequest{.num_qubits = 1,
                                              .operations = &circuit.operations(),
                                              .want_statevector = true,
                                              .want_counts = false,
                                              .want_expectation = false,
                                              .observable = nullptr});
    EXPECT_FALSE(estimate.feasible);
    EXPECT_FALSE(engine.capabilities().statevector);
}

TEST(DensityMatrixEngineTest, AnUnrepresentableSizeIsInfeasibleNotAnOverflow) {
    const Circuit circuit(40, 0);
    const DensityMatrixEngine engine;
    const CostEstimate estimate =
        engine.estimate(EngineEstimateRequest{.num_qubits = 40,
                                              .operations = &circuit.operations(),
                                              .want_statevector = false,
                                              .want_counts = true,
                                              .want_expectation = false,
                                              .observable = nullptr});
    EXPECT_FALSE(estimate.feasible);
}

class CountingMonitor : public ExecutionMonitor {
public:
    void at_safe_boundary(std::uint64_t boundaries_crossed) const override {
        last = boundaries_crossed;
    }
    mutable std::uint64_t last = 0;
};

TEST(DensityMatrixEngineTest, EveryOperationIsAMonitoredBoundary) {
    Circuit circuit(1, 1);
    circuit.append(Opcode::H, {0});
    circuit.append_channel({0}, depolarizing_1q(0.1));
    circuit.append_measure(0, 0);
    const std::vector<std::pair<QubitId, ClbitId>> targets = {{0, 0}};
    const CountingMonitor monitor;
    EngineExecutionRequest request = counts_request(circuit, targets, 10, 1);
    request.monitor = &monitor;
    const DensityMatrixEngine engine;
    static_cast<void>(engine.execute(request));
    EXPECT_EQ(monitor.last, 3U);
}

TEST(DensityMatrixEngineTest, IdentifiesItself) {
    const DensityMatrixEngine engine;
    EXPECT_EQ(engine.name(), "density_matrix");
    EXPECT_EQ(engine.engine_id(), Engine::DensityMatrix);
    EXPECT_EQ(engine.backend(), Backend::Cpu);
}

}  // namespace
}  // namespace menos::engines::density_matrix
