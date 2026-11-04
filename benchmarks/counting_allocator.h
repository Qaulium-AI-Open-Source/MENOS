// Counters exposed by benchmarks/counting_allocator.cpp (Phase 21 step 1).
// Only meaningful in a binary that links that file.

#pragma once

#include <cstdint>

namespace menos_benchmark {

struct AllocationSnapshot {
    std::uint64_t allocations;      // operator new calls so far
    std::uint64_t bytes_requested;  // sum of requested sizes so far
    std::int64_t live_bytes;        // currently allocated (malloc_usable_size)
    std::int64_t peak_live_bytes;   // max live_bytes since the last reset
};

AllocationSnapshot allocation_snapshot();

// Start a new peak window at the current live size.
void reset_peak_to_current();

}  // namespace menos_benchmark
