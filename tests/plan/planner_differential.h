// The boundary between the live planner and the frozen d89dde3 planner
// (tests/plan/frozen_d89dde3, docs/adr/0025 section 6).
//
// The frozen planner is the d89dde3 source tree compiled with the namespace
// `menos` renamed, so its types cannot meet the live ones. These functions
// therefore take and return only plain values: a job is named by its corpus
// index or its random seed, and each side builds it from the same headers.
//
// This namespace must never contain the token "menos": the frozen side is
// compiled with that token defined as a macro.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace planner_differential {

// The commit the frozen planner was taken from.
inline constexpr const char* kFrozenCommit = "d89dde3c7e66f84ae1c9ff99d92550ad439928e8";

// The frozen planner's outcome (planner_corpus::outcome_of) for fixed-corpus
// job `index` and for random_job(seed).
std::size_t frozen_corpus_size();
std::string frozen_corpus_name(std::size_t index);
std::string frozen_corpus_outcome(std::size_t index);
std::string frozen_random_outcome(std::uint64_t seed);

}  // namespace planner_differential
