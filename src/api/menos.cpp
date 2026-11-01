#include "menos/menos.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "api/replan_controller.h"
#include "common/error.h"
#include "common/thread_count.h"
#include "ir/validate.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "plan/planner.h"

#ifdef MENOS_ENABLE_CUDA
#include "kernels/cuda/device_pool.h"
#endif

namespace menos {
namespace {

// What validate_and_plan() produces: the plan, the job it was planned for,
// and the hardware profile it was planned against.
struct PlannedJob {
    Plan plan;
    plan::JobSpec job_spec;
    plan::HardwareProfile hardware_profile;
};

// Shared by run() and preview(): validate, build the JobSpec, and
// produce a Plan. run() additionally executes it; preview()
// (docs/BUILD_PLAN.md Phase 7) returns it as-is, for a preview that
// never pays execution cost.
PlannedJob validate_and_plan(const Circuit& circuit, const RunOptions& options) {
    ir::validate(circuit);
    ir::validate_output_request(
        circuit, options.shots, options.want_statevector, options.expectation.has_value());
    // A precision is a target standard error for the expectation value
    // (docs/adr/0017 section 3), so it must be a positive finite number and
    // needs an expectation to apply to. Refused rather than ignored: a silently
    // dropped precision would let a caller believe a bound was in force.
    if (options.expectation_precision.has_value()) {
        const double precision = *options.expectation_precision;
        if (!std::isfinite(precision) || precision <= 0.0) {
            throw common::InvalidCircuit("expectation_precision must be finite and > 0");
        }
        if (!options.expectation.has_value()) {
            throw common::InvalidCircuit(
                "expectation_precision was set but no expectation was requested");
        }
    }
    // A time budget must be finite and > 0 (docs/adr/0021 section 3). Before
    // Phase 17 nothing checked it: zero or a negative value was refused only
    // indirectly (no candidate fits), and NaN was silently ignored, since no
    // comparison with NaN is true, so neither the planner nor the run-time
    // monitor ever applied it. Refused here, as expectation_precision is.
    if (options.time_budget_seconds.has_value()) {
        const double budget = *options.time_budget_seconds;
        if (!std::isfinite(budget) || budget <= 0.0) {
            throw common::InvalidCircuit("time_budget_seconds must be finite and > 0");
        }
    }
    // max_threads is a cap (docs/adr/0019 section 3.3): unset means every
    // hardware thread, above the hardware count means the hardware count, and 0
    // is refused rather than reinterpreted.
    const std::uint32_t hardware_threads = common::default_thread_count();
    if (options.max_threads.has_value() && *options.max_threads == 0) {
        throw common::InvalidCircuit("max_threads must be at least 1");
    }
    const std::uint32_t max_threads =
        std::min(options.max_threads.value_or(hardware_threads), hardware_threads);

    plan::JobSpec job_spec{
        .circuit = circuit,
        .shots = options.shots,
        .want_statevector = options.want_statevector,
        .seed = options.seed,
        .memory_budget_bytes = options.memory_budget_bytes,
        .expectation = options.expectation,
        .accuracy_policy = options.accuracy_policy,
        .time_budget_seconds = options.time_budget_seconds,
        .noise_method = options.noise_method,
        .expectation_precision = options.expectation_precision,
        .max_threads = max_threads,
        .backend = options.backend,
    };
    // Calibrated, not merely detected: the time budget is evaluated against
    // measured throughput. Cached after the first call, so only the first job
    // in a process pays for the microbenchmarks. The device's free memory is
    // read again for every job (docs/adr/0025, Q3).
    const plan::HardwareProfile hardware_profile = plan::current_hardware_profile();
    Plan job_plan = plan::plan_job(job_spec, hardware_profile);
    return {std::move(job_plan), std::move(job_spec), hardware_profile};
}

}  // namespace

Result run(const Circuit& circuit, const RunOptions& options) {
    PlannedJob job = validate_and_plan(circuit, options);
    // A fresh steady_clock per execution attempt (Phase 11, docs/adr/0015 --
    // see BudgetMonitor for why the budget bounds an attempt, not the job).
    return api::run_with_replanning(
        job.job_spec, job.hardware_profile, std::move(job.plan), &api::steady_elapsed_source);
}

Plan preview(const Circuit& circuit, const RunOptions& options) {
    return validate_and_plan(circuit, options).plan;
}

void release_device_memory() {
#ifdef MENOS_ENABLE_CUDA
    kernels::cuda::device_pool().trim();
#endif
}

}  // namespace menos
