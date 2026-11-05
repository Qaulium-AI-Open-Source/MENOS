// Golden plans for the fixed Phase 24 planner corpus (docs/adr/0028 sections
// 3 and 6): every corpus job's plan, as Plan::explain() plus the plan fields
// explain() does not print, must match the file recorded from the pre-phase
// planner at d89dde3. A planner speed-up that changes any of it fails here.
//
// To re-record (only with the owner's approval of the changed plans):
//   MENOS_UPDATE_GOLDEN=1 ./menos_plan_tests --gtest_filter='PlannerGolden*'

#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
// std::getenv reads MENOS_UPDATE_GOLDEN; MSVC warns about it as it does for
// every C standard call with a _s variant (benchmarks/tiered_gate_benchmark.cpp
// takes the same route).
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "plan_fingerprint.h"
#include "planner_corpus.h"

namespace menos {
namespace {

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
#if defined(_MSC_VER) && !defined(NDEBUG)
    // The MSVC debug STL (iterator debugging) makes std::vector 32 bytes
    // instead of 24, and the memory estimates count container sizes (Phase
    // 21, docs/adr/0025 section 3.4): a counts plan's peak_bytes is 8 bytes
    // larger per histogram node here than in every optimised build and on
    // libstdc++. The recorded plans are the optimised builds'; windows-dev
    // still runs the memory fixtures, which hold each build's own estimate
    // to its heap tracker.
    GTEST_SKIP() << "golden plans are checked in optimised builds on MSVC (debug STL sizes)";
#else
    const bool update = std::getenv("MENOS_UPDATE_GOLDEN") != nullptr;
    for (const planner_corpus::CorpusJob& job : planner_corpus::planner_corpus()) {
        SCOPED_TRACE(job.name);
        if (!update && !checked_in_this_build(job.name)) {
            continue;
        }
        const std::string actual = planner_corpus::outcome_of(job);
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
        // The files are recorded with "\n"; a checkout with core.autocrlf=true
        // (Windows) delivers them with "\r\n", which is not a plan difference.
        std::string recorded = expected.str();
        std::erase(recorded, '\r');
        EXPECT_EQ(actual, recorded);
    }
#endif
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
