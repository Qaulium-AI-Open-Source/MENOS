// docs/BUILD_PLAN.md Phase 6 acceptance criterion 5: "A globally
// entangling circuit stays on dense under Exact, and explain() shows
// the rejected MPS candidate and its bound."

#include "menos/menos.h"

#include <gtest/gtest.h>

namespace menos {
namespace {

// 10 qubits, 15 brickwork layers (deep enough to saturate the
// Schmidt-rank bound at the physical cap, 2^5 = 32, at the middle
// cut -- unlike the shallow circuits elsewhere in this phase's tests)
// plus a T gate each layer to keep it off the stabilizer engine too.
// Verified empirically (a throwaway probe program, not guessed): dense
// costs 16384 bytes / 606208 work units here, MPS costs 43648 bytes --
// dense is both cheaper AND fits a budget MPS cannot.
Circuit make_globally_entangling_circuit() {
    constexpr std::uint32_t n = 10;
    Circuit circuit(n, n);
    for (QubitId q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {q});
    }
    for (int layer = 0; layer < 15; ++layer) {
        for (QubitId q = 0; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
        }
        for (QubitId q = 1; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
        }
        circuit.append(Opcode::T, {0});
    }
    for (QubitId q = 0; q < n; ++q) {
        circuit.append_measure(q, q);
    }
    return circuit;
}

TEST(MPSRejectionTest, GloballyEntangledCircuitStaysOnDenseAndExplainShowsMPSRejectionAndBound) {
    const Circuit circuit = make_globally_entangling_circuit();
    // Between dense's figure and MPS's at this scale: dense fits, MPS does
    // not. Since Phase 21 (docs/adr/0025 section 3.4) those are what each
    // engine really allocates: about 21 KB for dense (the 16384-byte state and
    // its buffers) and about 209 KB for MPS (the bonds it builds, with every
    // SVD's workspace).
    constexpr std::uint64_t kBudgetBytes = 30000;
    const Result result = run(circuit,
                              RunOptions{.shots = 10,
                                         .want_statevector = false,
                                         .seed = 1,
                                         .memory_budget_bytes = kBudgetBytes});

    ASSERT_EQ(result.plan().strategy, Strategy::DirectDense);
    EXPECT_TRUE(result.plan().exact);
    EXPECT_TRUE(result.exact());

    const CandidateSummary* mps_candidate = nullptr;
    for (const CandidateSummary& candidate : result.plan().candidates) {
        if (candidate.strategy == Strategy::DirectMPS) {
            mps_candidate = &candidate;
        }
    }
    ASSERT_NE(mps_candidate, nullptr) << "DirectMPS must always be enumerated";
    EXPECT_FALSE(mps_candidate->feasible);
    EXPECT_FALSE(mps_candidate->is_winner);
    // The rejection reason must name the actual bound (bytes required)
    // that exceeded the budget.
    EXPECT_NE(mps_candidate->infeasible_reason.find("bytes"), std::string::npos)
        << mps_candidate->infeasible_reason;

    const std::string text = result.explain();
    EXPECT_NE(text.find("strategy=direct_mps"), std::string::npos) << text;
    EXPECT_NE(text.find("feasible=false"), std::string::npos) << text;
    EXPECT_NE(text.find(mps_candidate->infeasible_reason), std::string::npos) << text;
}

}  // namespace
}  // namespace menos
