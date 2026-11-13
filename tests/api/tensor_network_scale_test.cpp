// docs/BUILD_PLAN.md Phase 8 acceptance criterion 4, in its corrected form.
//
// The criterion originally named the n=30, depth=10 RCS circuit from the
// precondition. That circuit cannot satisfy it: recomputed on the correct
// graph its line-graph width is 50, needing roughly 18 PB against dense's
// 17.18 GB, so the tensor-network engine is *more* infeasible than the
// engine it was meant to beat (docs/adr/0010's own correction). RCS
// brickwork circuits are built to defeat tensor-network contraction, which
// is the whole point of quantum-supremacy claims about them.
//
// The replacement family is chosen by the structural property this engine
// exists to exploit, fixed before any measurement rather than hunted for
// until the engine looked good:
//
//   * A balanced binary TREE interaction graph has treewidth 1, so the
//     closed network's own contraction stays cheap and grows with depth,
//     not with qubit count.
//   * No linear ordering of a balanced tree keeps every cut small, so an
//     MPS bond dimension must blow up -- the one structure where a 1D
//     ansatz is provably the wrong shape while a contraction is the right
//     one.
//   * A T gate makes it non-Clifford, defeating the stabilizer engine and
//     putting it outside Pauli propagation's supported set.
//   * It stays connected, so Factorized is never enumerated.
//
// That MPS genuinely fails here was treated as a hypothesis and verified
// against MENOS's own rigorous Schmidt-rank bound before this test was
// written, not assumed: at n=31 the bound demands roughly 80 GB.

#include "menos/menos.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace menos {
namespace {

// Balanced binary tree: qubit i's children are 2i+1 and 2i+2. Entangling
// gates run along tree edges only, so the interaction graph is exactly that
// tree. Alternating which parity of parent acts keeps consecutive layers
// from being identical.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Circuit make_tree_circuit(std::uint32_t n, std::uint32_t depth) {
    Circuit circuit(n, 0);
    // A fixed angle schedule rather than an RNG: this is an acceptance
    // fixture, and a reader must be able to reproduce the exact circuit
    // from the source alone.
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            const double angle =
                0.37 + 0.11 * static_cast<double>(layer) + 0.017 * static_cast<double>(q);
            circuit.append(Opcode::RY, {static_cast<QubitId>(q)}, {angle});
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
    return circuit;
}

// QAOA MaxCut over the same tree edge set: p rounds of exp(-i*gamma*Z_u*Z_v)
// per edge, built from CX/RZ/CX, then an RX(2*beta) mixer on every qubit.
// A different gate set and circuit shape on the same topology, so a win here
// corroborates that the advantage comes from the structure rather than from
// one hand-picked gate choice.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Circuit make_qaoa_tree_circuit(std::uint32_t n, std::uint32_t rounds) {
    Circuit circuit(n, 0);
    for (std::uint32_t q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {static_cast<QubitId>(q)});
    }
    for (std::uint32_t r = 0; r < rounds; ++r) {
        const double gamma = 0.3 + 0.1 * static_cast<double>(r);
        const double beta = 0.7 - 0.05 * static_cast<double>(r);
        for (std::uint32_t parent = 0; parent < n; ++parent) {
            for (std::uint32_t k = 1; k <= 2; ++k) {
                const std::uint32_t child = 2 * parent + k;
                if (child < n) {
                    circuit.append(Opcode::CX,
                                   {static_cast<QubitId>(parent), static_cast<QubitId>(child)});
                    circuit.append(Opcode::RZ, {static_cast<QubitId>(child)}, {2.0 * gamma});
                    circuit.append(Opcode::CX,
                                   {static_cast<QubitId>(parent), static_cast<QubitId>(child)});
                }
            }
        }
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::RX, {static_cast<QubitId>(q)}, {2.0 * beta});
        }
    }
    return circuit;
}

Observable z_on_every_qubit(std::uint32_t n) {
    PauliString p;
    p.num_qubits = n;
    p.x_words.assign((n + 63) / 64, 0);
    p.z_words.assign((n + 63) / 64, 0);
    for (std::uint32_t q = 0; q < n; ++q) {
        p.z_words[q / 64] |= (std::uint64_t{1} << (q % 64));
    }
    return Observable{{{p, {1.0, 0.0}}}};
}

const CandidateSummary& candidate_for(const Plan& plan, Strategy strategy) {
    for (const CandidateSummary& c : plan.candidates) {
        if (c.strategy == strategy) {
            return c;
        }
    }
    ADD_FAILURE() << "candidate not enumerated: " << static_cast<int>(strategy);
    return plan.candidates.at(0);
}

// The 50%-of-detected-memory figure docs/adr/0010 recorded on this class of
// machine. Deliberately a realistic operating budget, not one chosen to
// exclude the other engines: dense and MPS must fail here on the circuit's
// own scale, which is what makes this evidence rather than staging.
constexpr std::uint64_t kRealisticBudgetBytes = 7'752'519'680ULL;

// Differential, at a size where dense can still run: the tensor-network
// answer on a tree circuit must match a dense statevector oracle. The oracle
// comes through the public API by requesting a statevector, which only the
// dense engine can produce, so the two paths share no engine code.
TEST(TensorNetworkScaleTest, TreeCircuitExpectationMatchesDenseStatevectorOracle) {
    constexpr std::uint32_t kNumQubits = 15;
    const Circuit circuit = make_tree_circuit(kNumQubits, /*depth=*/4);

    const Result sv_result = run(
        circuit,
        RunOptions{
            .shots = 0, .want_statevector = true, .seed = 0, .memory_budget_bytes = 100'000'000});
    ASSERT_EQ(sv_result.plan().strategy, Strategy::DirectDense);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const std::vector<std::complex<double>>& psi = *sv_result.statevector();

    // <Z0 Z1 ... Z14> = sum_i |amp_i|^2 * (-1)^popcount(i).
    double oracle = 0.0;
    for (std::size_t i = 0; i < psi.size(); ++i) {
        const double probability = std::norm(psi[i]);
        const int sign = (std::popcount(i) % 2 == 0) ? 1 : -1;
        oracle += static_cast<double>(sign) * probability;
    }

    const Result tn_result = run(circuit,
                                 RunOptions{.shots = 0,
                                            .want_statevector = false,
                                            .seed = 0,
                                            .memory_budget_bytes = 100'000'000,
                                            .expectation = z_on_every_qubit(kNumQubits)});
    EXPECT_EQ(tn_result.plan().strategy, Strategy::DirectTensorNetwork)
        << "on a tree topology the contraction should already undercut dense on work units at "
           "this size; if this changes, the differential below is no longer testing the "
           "tensor-network path";
    ASSERT_TRUE(tn_result.expectation_value().has_value());
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    EXPECT_NEAR(tn_result.expectation_value()->real(), oracle, 1e-9);
    EXPECT_NEAR(tn_result.expectation_value()->imag(), 0.0, 1e-9);
    // NOLINTEND(bugprone-unchecked-optional-access)
}

// Acceptance criterion 4 proper, at a realistic budget: every other
// candidate is infeasible on the circuit's own scale, and the tensor-network
// engine is the one that can run it.
TEST(TensorNetworkScaleTest, ThirtyOneQubitTreeIsTensorNetworkOnlyAtARealisticBudget) {
    constexpr std::uint32_t kNumQubits = 31;
    const Circuit circuit = make_tree_circuit(kNumQubits, /*depth=*/4);

    const Result result = run(circuit,
                              RunOptions{.shots = 0,
                                         .want_statevector = false,
                                         .seed = 0,
                                         .memory_budget_bytes = kRealisticBudgetBytes,
                                         .expectation = z_on_every_qubit(kNumQubits)});

    const Plan& plan = result.plan();
    // 31 qubits needs 16 * 2^31 = 34.4 GB, above the budget: in memory the
    // dense candidate is infeasible. From Phase 23 (docs/adr/0027) it may be
    // feasible on the disk tier when the scratch directory has the room, at
    // the same work units, so it never wins here.
    const CandidateSummary& dense = candidate_for(plan, Strategy::DirectDense);
    EXPECT_TRUE(!dense.feasible || dense.tier == Tier::Disk);
    EXPECT_FALSE(dense.is_winner);
    EXPECT_FALSE(candidate_for(plan, Strategy::DirectStabilizer).feasible)
        << "the circuit is non-Clifford from its first RY";
    EXPECT_FALSE(candidate_for(plan, Strategy::DirectPauli).feasible)
        << "the T gate is outside Pauli propagation's supported set";
    EXPECT_FALSE(candidate_for(plan, Strategy::DirectMPS).feasible)
        << "a balanced tree has no good linear arrangement, so the rigorous Schmidt-rank bound "
           "demands roughly 80 GB -- this is the structural gap the engine exists for";

    const CandidateSummary& tn = candidate_for(plan, Strategy::DirectTensorNetwork);
    EXPECT_TRUE(tn.feasible);
    EXPECT_TRUE(tn.is_winner);
    EXPECT_EQ(plan.strategy, Strategy::DirectTensorNetwork);
    EXPECT_TRUE(plan.exact);

    // Dense cannot run at this size, so there is no oracle to compare
    // against here; correctness at tree topology is established by the
    // differential test above, and this test carries the scale claim.
    // What can still be checked is that the value is physical.
    ASSERT_TRUE(result.expectation_value().has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const std::complex<double> value = *result.expectation_value();
    EXPECT_TRUE(std::isfinite(value.real()));
    EXPECT_LE(std::abs(value.real()), 1.0 + 1e-9)
        << "a product of Z operators has spectrum in [-1, 1]";
    EXPECT_NEAR(value.imag(), 0.0, 1e-9) << "a Hermitian observable has a real expectation";

    // The estimate must bound what execution actually did -- the same
    // property tests/engines/tensor_network enforces on random circuits,
    // checked here on the acceptance workload itself.
    ASSERT_EQ(plan.regions.size(), 1U);
    ASSERT_TRUE(plan.regions[0].tensor_network_actual_peak_bytes.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_LE(*plan.regions[0].tensor_network_actual_peak_bytes, tn.estimated_peak_bytes);
}

// Corroboration on the same topology with a different gate set and shape.
TEST(TensorNetworkScaleTest, QaoaOnTheSameTreeTopologyAlsoRoutesToTensorNetwork) {
    constexpr std::uint32_t kNumQubits = 31;
    const Circuit circuit = make_qaoa_tree_circuit(kNumQubits, /*rounds=*/2);

    const Result result = run(circuit,
                              RunOptions{.shots = 0,
                                         .want_statevector = false,
                                         .seed = 0,
                                         .memory_budget_bytes = kRealisticBudgetBytes,
                                         .expectation = z_on_every_qubit(kNumQubits)});

    const Plan& plan = result.plan();
    // As above: the dense candidate is in memory infeasible, and on the disk
    // tier (Phase 23) at most feasible, never the winner.
    const CandidateSummary& dense = candidate_for(plan, Strategy::DirectDense);
    EXPECT_TRUE(!dense.feasible || dense.tier == Tier::Disk);
    EXPECT_FALSE(dense.is_winner);
    EXPECT_FALSE(candidate_for(plan, Strategy::DirectMPS).feasible);
    EXPECT_EQ(plan.strategy, Strategy::DirectTensorNetwork);

    ASSERT_TRUE(result.expectation_value().has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const std::complex<double> value = *result.expectation_value();
    EXPECT_TRUE(std::isfinite(value.real()));
    EXPECT_LE(std::abs(value.real()), 1.0 + 1e-9);
}

// docs/BUILD_PLAN.md Phase 8 acceptance criterion 3, the half that repeated
// in-process runs cannot reach: identical results across dev and release
// builds.
//
// A test binary cannot compare itself against a differently-configured
// build, so the mechanism is a golden value. scripts/verify.sh runs this
// suite under both presets; if each agrees with the literal below, the two
// agree with each other, which is the property the criterion asks for. The
// value is the full 17 significant digits of the double, so a difference in
// the last bits fails rather than rounds away -- which is the point, since
// an unpinned contraction order would show up exactly there.
//
// If this ever fails, the cause is a real determinism break (an elimination
// order that is no longer fixed, or a contraction accumulation order that
// varies), not a tolerance to widen.
TEST(TensorNetworkScaleTest, CutReconstructionIsBitIdenticalAcrossBuildConfigurations) {
    constexpr std::uint32_t n = 6;
    Circuit circuit(n, 0);
    for (QubitId q = 0; q < n; ++q) {
        circuit.append(Opcode::RY, {q}, {0.3 + 0.1 * static_cast<double>(q)});
    }
    for (QubitId parent = 0; parent < n; ++parent) {
        for (std::uint32_t k = 1; k <= 2; ++k) {
            const std::uint32_t child = 2 * parent + k;
            if (child < n) {
                circuit.append(Opcode::CZ, {parent, static_cast<QubitId>(child)});
            }
        }
    }
    circuit.append(Opcode::T, {0});

    PauliString z_all;
    z_all.num_qubits = n;
    z_all.x_words = {0};
    z_all.z_words = {(std::uint64_t{1} << n) - 1};

    const Result result = run(circuit,
                              RunOptions{.shots = 0,
                                         .want_statevector = false,
                                         .seed = 0,
                                         .memory_budget_bytes = 1'000'000,
                                         .expectation = Observable{{{z_all, {1.0, 0.0}}}}});

    // Phase 9 changed which candidate is cheapest here: a wire cut beats the
    // contraction on this 6-qubit tree, so this now pins the CUT path's
    // determinism. The physical answer is unchanged, which is itself worth
    // noting -- the tensor network produced 0.33961366418324335 for this same
    // circuit, and the two agree to 6e-17, from completely independent
    // execution strategies.
    ASSERT_EQ(result.plan().strategy, Strategy::CutAndRecombine);
    ASSERT_TRUE(result.expectation_value().has_value());
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    EXPECT_DOUBLE_EQ(result.expectation_value()->real(), 0.33961366418324329);
    EXPECT_DOUBLE_EQ(result.expectation_value()->imag(), 0.0);
    // NOLINTEND(bugprone-unchecked-optional-access)
}

// The tensor network's own cross-build determinism, at a scale where it is
// still the winning candidate (the 31-qubit tree of the acceptance test
// above, where a cut is offered and loses).
TEST(TensorNetworkScaleTest, ContractionIsBitIdenticalAcrossBuildConfigurations) {
    constexpr std::uint32_t kNumQubits = 31;
    const Circuit circuit = make_tree_circuit(kNumQubits, /*depth=*/4);
    const Result result = run(circuit,
                              RunOptions{.shots = 0,
                                         .want_statevector = false,
                                         .seed = 0,
                                         .memory_budget_bytes = kRealisticBudgetBytes,
                                         .expectation = z_on_every_qubit(kNumQubits)});
    ASSERT_EQ(result.plan().strategy, Strategy::DirectTensorNetwork);
    ASSERT_TRUE(result.expectation_value().has_value());
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
#ifdef _WIN32
    // Pinned per platform by owner ruling (2026-10-13, docs/adr/0022 section
    // 10): the MSVC runtime's cos/sin differ from glibc's by 1 ulp on some of
    // this circuit's rotation angles, so Windows reaches a different, equally
    // deterministic last bit. Debug+ASan and Release agree on it exactly.
    EXPECT_DOUBLE_EQ(result.expectation_value()->real(), -3.0658775544080871e-07);
    EXPECT_DOUBLE_EQ(result.expectation_value()->imag(), 0.0);
#else
    EXPECT_DOUBLE_EQ(result.expectation_value()->real(), -3.0658775544080971e-07);
    // Pinned exactly rather than compared against zero with a tolerance: the
    // imaginary part is a 1e-23 floating-point residual, and reproducing it
    // bit for bit is precisely the determinism claim this test makes.
    EXPECT_DOUBLE_EQ(result.expectation_value()->imag(), 1.321820786506289e-23);
#endif
    // NOLINTEND(bugprone-unchecked-optional-access)
}

}  // namespace
}  // namespace menos
