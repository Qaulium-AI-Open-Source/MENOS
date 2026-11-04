// The "same plan up to pruning" comparison (prune_equivalence.h) that the
// golden and differential tests rely on for section D (docs/adr/0025
// section 13). It must accept a newly pruned losing candidate or region
// engine, and reject every other difference.

#include "prune_equivalence.h"

#include <gtest/gtest.h>

#include <string>

namespace menos::planner_corpus {
namespace {

const std::string kReference =
    "MENOS EXECUTION REPORT\n"
    "candidate 0strategy=direct_dense feasible=false winner=false reason=\"too big\"\n"
    "candidate 1strategy=direct_tensor_network feasible=true winner=false peak_bytes=65536 "
    "work_units=3580198848\n"
    "candidate 2strategy=factorized feasible=true winner=true peak_bytes=4096 work_units=2105344\n"
    "candidate 3strategy=direct_mps feasible=false winner=false reason=\"needs memory\"\n"
    "strategy   factorized (exact)\n"
    "region 0   qubits=[0,1] engine=dense backend=cpu why=\"dense (stabilizer rejected: operation "
    "0 (ry) is not a Clifford gate)\"\n"
    "estimated  state_bytes=4096 work_units=2105344\n";

std::string replaced(std::string text, const std::string& from, const std::string& to) {
    const std::size_t at = text.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    return text.replace(at, from.size(), to);
}

const std::string kPrunedTn =
    "candidate 1strategy=direct_tensor_network feasible=false winner=false reason=\"pruned by "
    "branch-and-bound: costing stopped at 2200000 work units\"";

TEST(PruneEquivalence, IdenticalOutcomesAreEquivalent) {
    EXPECT_FALSE(differs_beyond_pruning(kReference, kReference).has_value());
    EXPECT_FALSE(
        differs_beyond_pruning("THROWS ResourceExceeded: x\n", "THROWS ResourceExceeded: x\n")
            .has_value());
}

TEST(PruneEquivalence, ALosingCostlierCandidateMayBecomePruned) {
    const std::string live = replaced(kReference,
                                      "candidate 1strategy=direct_tensor_network feasible=true "
                                      "winner=false peak_bytes=65536 work_units=3580198848",
                                      kPrunedTn);
    EXPECT_FALSE(differs_beyond_pruning(kReference, live).has_value());
}

TEST(PruneEquivalence, AnInfeasibleCandidateMayBecomePruned) {
    const std::string live = replaced(
        kReference,
        "candidate 3strategy=direct_mps feasible=false winner=false reason=\"needs memory\"",
        "candidate 3strategy=direct_mps feasible=false winner=false reason=\"pruned by "
        "branch-and-bound: stopped\"");
    EXPECT_FALSE(differs_beyond_pruning(kReference, live).has_value());
}

TEST(PruneEquivalence, ACandidateNoCostlierThanTheWinnerCannotBePruned) {
    const std::string reference =
        replaced(kReference, "work_units=3580198848", "work_units=2105344");
    const std::string live = replaced(reference,
                                      "candidate 1strategy=direct_tensor_network feasible=true "
                                      "winner=false peak_bytes=65536 work_units=2105344",
                                      kPrunedTn);
    EXPECT_TRUE(differs_beyond_pruning(reference, live).has_value());
}

TEST(PruneEquivalence, TheWinnerCannotBePruned) {
    const std::string live =
        replaced(kReference,
                 "candidate 2strategy=factorized feasible=true winner=true peak_bytes=4096 "
                 "work_units=2105344",
                 "candidate 2strategy=factorized feasible=false winner=false reason=\"pruned by "
                 "branch-and-bound: x\"");
    EXPECT_TRUE(differs_beyond_pruning(kReference, live).has_value());
}

TEST(PruneEquivalence, AnyOtherCandidateChangeIsRejected) {
    EXPECT_TRUE(
        differs_beyond_pruning(
            kReference, replaced(kReference, "work_units=3580198848", "work_units=3580198849"))
            .has_value());
    EXPECT_TRUE(differs_beyond_pruning(kReference, replaced(kReference, "too big", "too large"))
                    .has_value());
}

TEST(PruneEquivalence, TheStrategyAndEstimateMustMatch) {
    EXPECT_TRUE(differs_beyond_pruning(kReference,
                                       replaced(kReference,
                                                "state_bytes=4096 work_units=2105344\n",
                                                "state_bytes=4096 work_units=2105345\n"))
                    .has_value());
    EXPECT_TRUE(differs_beyond_pruning(kReference, replaced(kReference, "(exact)", "(approx)"))
                    .has_value());
    EXPECT_TRUE(differs_beyond_pruning(kReference, kReference + "extra\n").has_value());
}

TEST(PruneEquivalence, ARegionMayListANewlyPrunedEngine) {
    const std::string live = replaced(kReference,
                                      "is not a Clifford gate)\"",
                                      "is not a Clifford gate) (tensor_network rejected: pruned by "
                                      "branch-and-bound: costing stopped (at 5 units))\"");
    EXPECT_FALSE(differs_beyond_pruning(kReference, live).has_value());
}

TEST(PruneEquivalence, ARegionRejectionMayBecomePruned) {
    const std::string live =
        replaced(kReference,
                 "(stabilizer rejected: operation 0 (ry) is not a Clifford gate)",
                 "(stabilizer rejected: pruned by branch-and-bound: x)");
    EXPECT_FALSE(differs_beyond_pruning(kReference, live).has_value());
}

TEST(PruneEquivalence, OtherRegionChangesAreRejected) {
    EXPECT_TRUE(
        differs_beyond_pruning(kReference, replaced(kReference, "engine=dense", "engine=mps"))
            .has_value());
    EXPECT_TRUE(
        differs_beyond_pruning(kReference, replaced(kReference, "why=\"dense (", "why=\"mps ("))
            .has_value());
    // A rejection dropped, or changed to another non-pruned reason.
    EXPECT_TRUE(
        differs_beyond_pruning(
            kReference,
            replaced(
                kReference, " (stabilizer rejected: operation 0 (ry) is not a Clifford gate)", ""))
            .has_value());
    EXPECT_TRUE(
        differs_beyond_pruning(kReference, replaced(kReference, "(ry)", "(rx)")).has_value());
    // A new non-pruned rejection.
    EXPECT_TRUE(differs_beyond_pruning(kReference,
                                       replaced(kReference,
                                                "is not a Clifford gate)\"",
                                                "is not a Clifford gate) (mps rejected: big)\""))
                    .has_value());
}

TEST(PruneEquivalence, ARefusalMustMatchExactly) {
    EXPECT_TRUE(differs_beyond_pruning("THROWS ResourceExceeded: x\n", kReference).has_value());
    EXPECT_TRUE(
        differs_beyond_pruning("THROWS ResourceExceeded: x\n", "THROWS ResourceExceeded: y\n")
            .has_value());
}

}  // namespace
}  // namespace menos::planner_corpus
