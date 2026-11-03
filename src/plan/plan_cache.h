#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "menos/plan.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"

namespace menos::plan {

// Plan reuse across calls that differ only in gate parameter values.
//
// A variational workload (VQE, QAOA, any optimiser loop) runs one circuit
// structure hundreds of times with new angles each time. Everything plan_job
// decides is a function of the job's STRUCTURE -- opcodes, operands, qubit
// count, measurement targets, the observable, the run options -- with one
// exception: the MPS candidate's Schmidt-rank bound reads each multi-qubit
// gate's matrix (analysis::compute_schmidt_rank_bound), whose operator
// Schmidt rank can drop at isolated special angles (RZZ(0) is the identity).
// So the cache key is the structure, and a job whose multi-qubit
// parametrised gate sits within kSpecialAngleMargin of such an angle is not
// cached at all (planned afresh, exactly as before). Away from those angles
// every planner input is identical, so the cached plan IS the plan plan_job
// would compute; only the operations' parameter values differ, and those are
// stitched in from the new circuit (reparameterize_direct_plan).
//
// Scope, deliberately narrow:
//   - noiseless jobs without a time budget (a budgeted job's plan depends on
//     observations from abandoned attempts, docs/adr/0015);
//   - plans whose winner is a Direct* strategy (one region whose operations
//     are the light-cone-reduced circuit's, so re-parameterisation is the
//     light-cone pass again -- linear -- plus a copy). Factorized and
//     CutAndRecombine winners are computed but never stored.
// Everything else goes straight to plan_job, unchanged.
//
// The cache is process-global, mutex-protected, LRU, kPlanCacheCapacity
// entries. plan_cache_clear() empties it (tests, or after a hardware profile
// change); plan_cache_stats() reports hits / misses / bypasses.

inline constexpr std::size_t kPlanCacheCapacity = 64;
// Distance (in the gate's own angle) from a rank-dropping angle inside which
// a job is not cached. operator_schmidt_rank's cutoff is sigma_max * 4 * eps
// ~ 1e-15 relative; the smallest singular value of RZZ/CP's realigned matrix
// is linear in that distance, so 1e-8 leaves seven orders of magnitude of
// margin on the "rank is at its generic value" side.
inline constexpr double kSpecialAngleMargin = 1e-8;

struct PlanCacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t bypasses = 0;  // uncacheable job, or a non-Direct winner
};

// The structural key of a job, or nullopt when the job must not be cached.
[[nodiscard]] std::optional<std::string> plan_cache_key(const JobSpec& job_spec);

// plan_job with the cache in front of it. Identical results to plan_job for
// every job (the cache is an optimisation, never a change of plan).
[[nodiscard]] Plan plan_job_cached(const JobSpec& job_spec,
                                   const HardwareProfile& hardware_profile);

[[nodiscard]] PlanCacheStats plan_cache_stats();
void plan_cache_clear();

}  // namespace menos::plan
