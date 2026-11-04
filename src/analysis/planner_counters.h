// Optional planner counters for benchmark reports (docs/adr/0025 section
// 12.3, C.3). Compiled in only when CMake's MENOS_PLANNER_COUNTERS option is
// ON, which no shipped or timed build sets: menos_planner_benchmark built
// with it reports, per corpus job, how the Schmidt-rank cache was used. Not
// public API.

#pragma once

#ifdef MENOS_PLANNER_COUNTERS

#include <algorithm>
#include <cstdint>

#include "analysis/schmidt_rank_internal.h"

namespace menos::analysis::detail {

struct PlannerCounters {
    std::uint64_t rank_hits = 0;
    std::uint64_t rank_misses = 0;
    std::uint64_t rank_evictions = 0;
    std::uint64_t rank_uncached = 0;
    std::uint64_t rank_peak_bytes = 0;  // the largest single cache's peak

    void add(const SchmidtRankCacheStats& s) {
        rank_hits += s.hits;
        rank_misses += s.misses;
        rank_evictions += s.evictions;
        rank_uncached += s.uncached;
        rank_peak_bytes = std::max(rank_peak_bytes, s.peak_bytes);
    }
};

// Per thread, so concurrent planning never races on it.
inline thread_local PlannerCounters planner_counters;

}  // namespace menos::analysis::detail

#endif
