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

// The plan `cached` (a Direct* plan plan_job computed for a job of the same
// structure, see plan/plan_cache.h) with its operations' parameter values
// replaced by job_spec's. Runs the light-cone pass on job_spec's circuit --
// the one step between the job and a Direct region's operations -- and
// requires the result to match the cached reduced circuit operation for
// operation in kind, opcode and operands (InternalError otherwise: the
// cache key was wrong). Everything else in the plan is structural and is
// copied as is.
[[nodiscard]] Plan reparameterize_direct_plan(const Plan& cached, const JobSpec& job_spec);

}  // namespace menos::plan
