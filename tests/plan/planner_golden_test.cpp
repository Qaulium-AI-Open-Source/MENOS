// Golden plans for the fixed Phase 21 planner corpus (docs/adr/0025 sections
// 3 and 6): every corpus job's plan, as Plan::explain() plus the plan fields
// explain() does not print, must match the file recorded from the pre-phase
// planner at d89dde3. A planner speed-up that changes any of it fails here.
//
// To re-record (only with the owner's approval of the changed plans):
//   MENOS_UPDATE_GOLDEN=1 ./menos_plan_tests --gtest_filter='PlannerGolden*'

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>

#include "common/error.h"
#include "plan/planner.h"
#include "planner_corpus.h"

namespace menos {
namespace {

// explain() plus what it leaves out: each region's operation count, observable
// term count and estimates, and the reduced circuit's size.
std::string plan_fingerprint(const Plan& plan) {
    std::string out = plan.explain();
    out += "--- plan fields not printed by explain() ---\n";
    for (std::size_t i = 0; i < plan.regions.size(); ++i) {
        const Region& region = plan.regions[i];
        out += std::format(
            "region {}: operations={} observable_terms={} state_bytes={} "
            "work_units={} global_to_local_size={}\n",
            i,
            region.operations.size(),
            region.observable.has_value() ? region.observable->terms.size() : 0,
            region.estimated_state_bytes,
            region.estimated_work_units,
            region.global_to_local.size());
    }
    out += std::format("reduced_circuit: {}\n",
                       plan.reduced_circuit.has_value()
                           ? std::format("qubits={} operations={}",
                                         plan.reduced_circuit->num_qubits(),
                                         plan.reduced_circuit->operations().size())
                           : std::string("none"));
    return out;
}

std::string outcome_of(const planner_corpus::CorpusJob& job) {
    try {
        return plan_fingerprint(plan::plan_job(job.spec, job.hardware));
    } catch (const common::ResourceExceeded& e) {
        return std::string("THROWS ResourceExceeded: ") + e.what() + "\n";
    } catch (const common::UnsupportedFeature& e) {
        return std::string("THROWS UnsupportedFeature: ") + e.what() + "\n";
    }
}

// The jobs whose Release plan_job takes over 0.25 s (bench/BASELINE.md). A
// Debug ASan+UBSan build is one to two orders of magnitude slower, which would
// make the dev ctest run take minutes for these alone, so they are checked
// only in optimised builds (the release and cuda-release ctest stages of
// scripts/verify.sh). Every other job is checked in every build.
bool checked_in_this_build(const std::string& name) {
#ifdef NDEBUG
    static_cast<void>(name);
    return true;
#else
    return name != "expectation_factorized/large" && name != "expectation_factorized/medium" &&
           name != "expectation_ising_budgeted/large" &&
           name != "expectation_ising_budgeted/medium" &&
           name != "expectation_random_pairs_controlled/large";
#endif
}

std::filesystem::path golden_path(const std::string& name) {
    std::string file = name;
    for (char& c : file) {
        if (c == '/') {
            c = '.';
        }
    }
    return std::filesystem::path(MENOS_PLANNER_GOLDEN_DIR) / (file + ".txt");
}

TEST(PlannerGolden, EveryCorpusPlanMatchesTheRecordedPlan) {
    const bool update = std::getenv("MENOS_UPDATE_GOLDEN") != nullptr;
    for (const planner_corpus::CorpusJob& job : planner_corpus::planner_corpus()) {
        SCOPED_TRACE(job.name);
        if (!update && !checked_in_this_build(job.name)) {
            continue;
        }
        const std::string actual = outcome_of(job);
        const std::filesystem::path path = golden_path(job.name);
        if (update) {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream(path, std::ios::binary) << actual;
            continue;
        }
        std::ifstream in(path, std::ios::binary);
        ASSERT_TRUE(in.good()) << "missing golden file " << path;
        std::stringstream expected;
        expected << in.rdbuf();
        EXPECT_EQ(actual, expected.str());
    }
}

// The corpus itself is fixed: its size and order are part of the record, so
// an accidental edit to planner_corpus.h is caught here, not only as a diff
// in many golden files.
TEST(PlannerGolden, CorpusIsTheRecordedSet) {
    const auto jobs = planner_corpus::planner_corpus();
    EXPECT_EQ(jobs.size(), 45U);
    EXPECT_EQ(jobs.front().name, "statevector_brickwork/small");
    EXPECT_EQ(jobs.back().name, "counts_noisy_trajectory/large");
}

}  // namespace
}  // namespace menos
