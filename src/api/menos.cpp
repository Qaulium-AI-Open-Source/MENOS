#include "menos/menos.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <utility>

#include "api/replan_controller.h"
#include "common/error.h"
#include "common/thread_count.h"
#include "ir/validate.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "plan/plan_cache.h"
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
    // Wall-clock seconds of the planner call alone (Result::plan_seconds()).
    double plan_seconds = 0.0;
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

    // The disk tier's scratch directory (Phase 23, docs/adr/0027 section 3.3):
    // the caller's, or the system's temporary directory.
    const std::filesystem::path scratch_directory = options.scratch_directory.has_value()
                                                        ? *options.scratch_directory
                                                        : std::filesystem::temp_directory_path();
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
        .fuse_diagonal_gates = options.fuse_diagonal_gates,
        .fuse_gates = options.fuse_gates,
        .backend = options.backend,
        .tier = options.tier,
        .scratch_directory = scratch_directory,
    };
    // The profile the job is planned against. The planner's own reach test,
    // on the whole circuit, says whether this job could reach the disk tier
    // (docs/adr/0027 section 9, Q3); it reads memory facts only, so the
    // detected profile answers it. A job with a time budget, or one that could
    // reach the disk tier (whose crossings the planner prices at the
    // calibrated rates, docs/adr/0027 section 3.5), is planned against the
    // calibrated profile: measured throughput, cached after the first call, so
    // only the first such job in a process pays for the microbenchmarks. Any
    // other CPU-only job never runs them (plan::hardware_profile_for, Phase
    // 24, docs/adr/0028). On top of that base the device's free memory is read
    // again for every job (docs/adr/0025, Q3), and so is the scratch
    // directory's; its bandwidth is measured, once per directory per process,
    // only when this job could reach the tier.
    const bool measure_scratch =
        plan::may_reach_disk_tier(job_spec, plan::hardware_profile_for(/*time_budgeted=*/false));
    const plan::HardwareProfile hardware_profile = plan::current_hardware_profile(
        plan::hardware_profile_for(options.time_budget_seconds.has_value() || measure_scratch),
        scratch_directory,
        measure_scratch);
    // Timed around the planner call only: validation, the JobSpec and the
    // hardware profile (whose calibration is cached per process) are not
    // planning (docs/adr/0028 section 16.7 Q3).
    const auto planning_start = std::chrono::steady_clock::now();
    Plan job_plan = plan::plan_job_cached(job_spec, hardware_profile);
    const double plan_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - planning_start).count();
    return {std::move(job_plan), std::move(job_spec), hardware_profile, plan_seconds};
}

}  // namespace

Result run(const Circuit& circuit, const RunOptions& options) {
    PlannedJob job = validate_and_plan(circuit, options);
    // A fresh steady_clock per execution attempt (Phase 11, docs/adr/0015 --
    // see BudgetMonitor for why the budget bounds an attempt, not the job).
    return api::run_with_replanning(
               job.job_spec, job.hardware_profile, std::move(job.plan), &api::steady_elapsed_source)
        .with_plan_seconds(job.plan_seconds);
}

Plan preview(const Circuit& circuit, const RunOptions& options) {
    return validate_and_plan(circuit, options).plan;
}

void release_device_memory() {
#ifdef MENOS_ENABLE_CUDA
    kernels::cuda::device_pool().trim();
#endif
}

PlanCacheStats plan_cache_stats() {
    const plan::PlanCacheStats s = plan::plan_cache_stats();
    return PlanCacheStats{s.hits, s.misses, s.bypasses};
}

void plan_cache_clear() {
    plan::plan_cache_clear();
}

}  // namespace menos
