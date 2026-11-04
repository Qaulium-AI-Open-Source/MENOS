// The frozen side of the planner differential test (docs/adr/0025 section 6).
//
// Compiled into menos_frozen_d89dde3 together with the d89dde3 source tree,
// with the token `menos` defined as `menos_frozen_d89dde3`. Every planner
// type and function used here is therefore the d89dde3 one, while the corpus
// and fingerprint headers are the same files the live side compiles.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../plan_fingerprint.h"
#include "../planner_corpus.h"
#include "../planner_differential.h"
#include "../random_planner_jobs.h"

namespace planner_differential {

namespace {

const std::vector<menos::planner_corpus::CorpusJob>& corpus() {
    static const std::vector<menos::planner_corpus::CorpusJob> kJobs =
        menos::planner_corpus::planner_corpus();
    return kJobs;
}

}  // namespace

std::size_t frozen_corpus_size() {
    return corpus().size();
}

std::string frozen_corpus_name(std::size_t index) {
    return corpus().at(index).name;
}

std::string frozen_corpus_outcome(std::size_t index) {
    return menos::planner_corpus::outcome_of(corpus().at(index));
}

std::string frozen_random_outcome(std::uint64_t seed) {
    return menos::planner_corpus::outcome_of(menos::planner_corpus::random_job(seed));
}

}  // namespace planner_differential
