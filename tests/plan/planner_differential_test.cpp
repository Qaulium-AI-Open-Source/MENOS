// Differential test: the live planner against the frozen d89dde3 planner
// (Phase 21, docs/adr/0025 sections 3 and 6; owner decision 2026-11-03).
//
// A Phase 21 speed-up must give exactly the plan the pre-phase planner gave.
// The golden files (planner_golden_test.cpp) pin that down for the 45 fixed
// corpus jobs; this test extends it to randomised jobs, planned by both
// planners in one process and compared through planner_corpus::outcome_of
// (explain() plus the fields explain() does not print, or the refusal).
//
// Seeds: MENOS_DIFFERENTIAL_SEEDS jobs (default below) starting at
// MENOS_DIFFERENTIAL_FIRST_SEED (default 0). A failure names its seed; rerun
// it alone with MENOS_DIFFERENTIAL_FIRST_SEED=<seed> MENOS_DIFFERENTIAL_SEEDS=1.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "plan_fingerprint.h"
#include "planner_corpus.h"
#include "planner_differential.h"
#include "random_planner_jobs.h"

namespace menos {
namespace {

std::uint64_t env_or(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe): read before any thread
    return value == nullptr ? fallback : std::strtoull(value, nullptr, 10);
}

// The default seed count keeps each ctest stage under about a minute: a Debug
// ASan+UBSan build plans one to two orders of magnitude slower than Release.
#ifdef NDEBUG
constexpr std::uint64_t kDefaultSeeds = 2000;
#else
constexpr std::uint64_t kDefaultSeeds = 60;
#endif

// The same exclusions, for the same reason, as planner_golden_test.cpp.
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

std::string golden_file(const std::string& name) {
    std::string file = name;
    for (char& c : file) {
        if (c == '/') {
            c = '.';
        }
    }
    std::ifstream in(std::filesystem::path(MENOS_PLANNER_GOLDEN_DIR) / (file + ".txt"),
                     std::ios::binary);
    std::stringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

// What a plan's outcome is, for the coverage summary: the winning strategy, or
// the refusal.
std::string outcome_kind(const std::string& outcome) {
    if (outcome.starts_with("THROWS ")) {
        return outcome.substr(0, outcome.find(':'));
    }
    const std::string marker = "\nstrategy   ";
    const std::size_t at = outcome.find(marker);
    if (at == std::string::npos) {
        return "unrecognised";
    }
    const std::size_t begin = at + marker.size();
    return outcome.substr(begin, outcome.find(' ', begin) - begin);
}

// The oracle itself: the frozen copy must reproduce the files recorded from
// the unchanged planner, or it is not the d89dde3 planner.
TEST(PlannerDifferential, FrozenPlannerReproducesTheGoldenFiles) {
    const auto corpus = planner_corpus::planner_corpus();
    ASSERT_EQ(planner_differential::frozen_corpus_size(), corpus.size());
    for (std::size_t i = 0; i < corpus.size(); ++i) {
        SCOPED_TRACE(corpus[i].name);
        ASSERT_EQ(planner_differential::frozen_corpus_name(i), corpus[i].name);
        if (!checked_in_this_build(corpus[i].name)) {
            continue;
        }
        EXPECT_EQ(planner_differential::frozen_corpus_outcome(i), golden_file(corpus[i].name));
    }
}

TEST(PlannerDifferential, LivePlannerMatchesTheFrozenPlannerOnRandomJobs) {
    const std::uint64_t first = env_or("MENOS_DIFFERENTIAL_FIRST_SEED", 0);
    const std::uint64_t count = env_or("MENOS_DIFFERENTIAL_SEEDS", kDefaultSeeds);
    std::map<std::string, std::uint64_t> kinds;
    std::uint64_t pruned = 0;
    std::uint64_t mismatches = 0;
    for (std::uint64_t seed = first; seed < first + count; ++seed) {
        const planner_corpus::CorpusJob job = planner_corpus::random_job(seed);
        const std::string live = planner_corpus::outcome_of(job);
        const std::string frozen = planner_differential::frozen_random_outcome(seed);
        ++kinds[outcome_kind(frozen)];
        pruned += frozen.find("pruned by branch-and-bound") != std::string::npos ? 1U : 0U;
        if (live != frozen) {
            ++mismatches;
            // The first few in full; a systematic change would otherwise flood
            // the log.
            if (mismatches <= 3) {
                ADD_FAILURE() << job.name << " (seed " << seed << ") differs from the frozen "
                              << planner_differential::kFrozenCommit << " planner.\n--- frozen\n"
                              << frozen << "--- live\n"
                              << live;
            }
        }
    }
    EXPECT_EQ(mismatches, 0U) << "of " << count << " random jobs";

    std::cout << "[ differential ] " << count << " random jobs from seed " << first << ", "
              << pruned << " with a branch-and-bound prune\n";
    for (const auto& [kind, n] : kinds) {
        std::cout << "[ differential ]   " << kind << ": " << n << "\n";
    }
}

}  // namespace
}  // namespace menos
