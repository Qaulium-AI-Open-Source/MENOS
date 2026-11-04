#pragma once

#include "menos/plan.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "plan/runtime_observations.h"

namespace menos::plan {

// Computes the dense memory requirement and throws ResourceExceeded
// (naming the required and available bytes) if it does not fit. With one
// candidate this is budget enforcement, not a search.
//
// `observations` is what a previously abandoned attempt at THIS job measured
// (Phase 11, docs/adr/0015). It defaults to empty, which is every first
// planning of every job and every caller that never replans, so those are
// unaffected. This is the only planner in MENOS: a replan is this same
// function called again with a non-empty third argument, never a second
// planner living in the runtime.
[[nodiscard]] Plan plan_job(const JobSpec& job_spec,
                            const HardwareProfile& hardware_profile,
                            const RuntimeObservations& observations = {});

// Whether plan_job could reach the disk tier for this job (Phase 23,
// docs/adr/0027 section 9, Q3): the tier is forced, or it is not ruled out
// (no statevector, no noise, no forced in-memory tier or CUDA backend) and the
// in-memory dense estimate of the whole circuit does not fit the budget. The
// api layer measures the scratch directory's bandwidth only when this is
// true, so the one-off measurement is never paid by a job that cannot use it.
// Judged on the whole circuit, an upper bound of the light-cone-reduced one
// plan_job prices, so it can say yes to a job the planner then keeps in
// memory, never no to one it moves.
[[nodiscard]] bool may_reach_disk_tier(const JobSpec& job_spec,
                                       const HardwareProfile& hardware_profile);

}  // namespace menos::plan
