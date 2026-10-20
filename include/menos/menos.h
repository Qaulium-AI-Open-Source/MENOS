#pragma once

#include <cstdint>
#include <optional>

#include "menos/accuracy_policy.h"
#include "menos/circuit.h"
#include "menos/noise_method.h"
#include "menos/observable.h"
#include "menos/result.h"

namespace menos {

// The output request and resource budget for a run. shots and seed are
// mandatory: CLAUDE.md's determinism invariant requires an explicit seed,
// and no shot-count default is authorized by the spec, so neither is
// silently invented here. shots == 0 and want_statevector == false and
// expectation == nullopt together mean nothing was requested, and
// run() throws InvalidCircuit (docs/BUILD_PLAN.md Phase 1's output-
// request rules, extended in Phase 5 for expectation). accuracy_policy
// defaults to Exact (Phase 6) -- every plan before Phase 6 is exact, and
// stays exact unless the caller opts into Controlled(tolerance).
struct RunOptions {
    std::uint64_t shots;
    bool want_statevector = false;
    std::uint64_t seed;
    std::optional<std::uint64_t> memory_budget_bytes = std::nullopt;
    std::optional<Observable> expectation = std::nullopt;
    AccuracyPolicy accuracy_policy = {};
    // Wall-clock seconds one EXECUTION ATTEMPT of the job may take
    // (docs/BUILD_PLAN.md Phase 10, docs/adr/0014; monitored at run time since
    // Phase 11, docs/adr/0015). A constraint, not a hint.
    //
    // At plan time: a candidate whose estimated runtime exceeds it is
    // infeasible, and run() throws ResourceExceeded if none fits.
    //
    // At run time: a run whose observed elapsed time exceeds it is abandoned at
    // a safe boundary (between gates, observable terms or sampled shots) and the
    // planner is re-run with what that run actually cost. If a different plan
    // meets the budget the job restarts under it; otherwise run() throws
    // ResourceExceeded. Nothing is silently allowed to overrun.
    //
    // WHAT IT BOUNDS -- read this before relying on it. The budget bounds each
    // individual attempt, NOT the total wall time of a job that replans. A job
    // that is abandoned and restarted has spent up to one full budget on the
    // abandoned attempt AND then runs the replacement, so its total can approach
    // two budgets or more, up to one per candidate plan. The abandoned attempt's
    // seconds are reported by Result::replans() and explain(), never hidden, and
    // Result::execute_seconds() times only the attempt that succeeded. Planning
    // and replanning themselves are not charged against the budget. If you need a
    // hard bound on total wall time, this option does not provide it.
    //
    // Unset (the default) installs no monitor: execution and explain() are
    // unchanged from a run that never heard of this option.
    std::optional<double> time_budget_seconds = std::nullopt;
    // Noise simulation opt-ins (Phase 13, docs/adr/0017). Both defaults keep a
    // noisy job on the exact density-matrix engine, exactly as in Phase 12, and
    // neither has any effect on a noiseless job.
    //
    // noise_method == Auto lets the planner run a noisy COUNTS request by
    // trajectory sampling when that is cheaper or the only plan that fits.
    //
    // expectation_precision is the target absolute standard error eps of the
    // expectation value (the meaning EstimatorV2's precision has), and is the
    // opt-in for running a noisy EXPECTATION by trajectories. The trajectory
    // count is chosen so the true standard error is at most eps, and the result
    // reports its measured standard error (Result::expectation_standard_error).
    // It must be finite and > 0, and requires `expectation`. It opts in only the
    // expectation: a job that also requests counts still needs Auto.
    NoiseMethod noise_method = NoiseMethod::Exact;
    std::optional<double> expectation_precision = std::nullopt;
    // The most CPU threads gate application may use (Phase 15, docs/adr/0019
    // section 3.3). A cap, not a request:
    //
    // - Unset (the default) means every hardware thread.
    // - A value above the hardware thread count uses the hardware count.
    // - 0 is refused: run() throws InvalidCircuit.
    // - OMP_NUM_THREADS is ignored; only this option sets MENOS's threads.
    //
    // Results never depend on it: every thread count gives bitwise-identical
    // statevectors, counts and expectation values. Only speed changes, and
    // small circuits run on one thread whatever the cap.
    //
    // With time_budget_seconds: the planner's time estimate uses a rate
    // calibrated with the default thread count, so a lower cap makes that
    // estimate optimistic. A capped run that then overruns its budget is
    // caught by the run-time monitor like any other overrun (replanned, or
    // ResourceExceeded); no estimate is scaled by the cap.
    std::optional<std::uint32_t> max_threads = std::nullopt;
    // Fuse runs of consecutive diagonal gates (RZ, RZZ, CP, CZ, S, T, ...)
    // into one diagonal unitary per block of at most kMaxUnitaryQubits qubits
    // before the CPU dense engine applies them (src/ir/fuse_diagonal.h). A
    // QAOA cost layer of n + n(n-1)/2 gates becomes a handful of passes. The
    // result is exact up to rounding but NOT bitwise identical to the unfused
    // run, and the CUDA engine does not fuse, so with this on a CPU run is no
    // longer bitwise identical to a CUDA run of the same job. Off by default.
    bool fuse_diagonal_gates = false;
    // The machine to run on (Phase 17, docs/adr/0021 section 2). A request, not
    // a preference: it is honoured or the job is refused, never quietly
    // replaced.
    //
    // - Unset (the default): the planner chooses, exactly as before.
    // - Backend::Cpu: nothing runs on the GPU.
    // - Backend::Cuda: the planner still chooses the algorithm, but only among
    //   plans that can run on CUDA: direct dense simulation of a noiseless
    //   circuit, or a Factorized plan whose regions are all dense and each fit
    //   the device (all of them then run on CUDA). run() throws UnsupportedFeature if this build
    //   has no CUDA support, no device is available, or the job is noisy, and ResourceExceeded if
    //   no CUDA plan fits the device, the memory budget or the time budget.
    //
    // A forced backend stays forced when a time-budget overrun replans the job:
    // an overrunning CUDA run is refused rather than moved to the CPU.
    //
    // What forcing can change: when both runs use dense simulation, CPU and
    // CUDA statevectors and counts are bitwise identical, and expectation
    // values agree within 1e-12 (the device reduction sums in a different,
    // fixed order). But forcing CUDA can also change the ALGORITHM, from the
    // stabilizer, MPS or another engine the planner would have chosen to dense
    // simulation. The answer is still exact, but counts are then a different
    // sample of the same distribution for the same seed, because a different
    // engine draws them.
    std::optional<Backend> backend = std::nullopt;
};

// Validates circuit and the output request, plans, executes, and returns
// the Result. Throws InvalidCircuit, UnsupportedFeature or
// ResourceExceeded (menos::common) on invalid input, an unsupported
// feature, or an exceeded memory budget, respectively.
[[nodiscard]] Result run(const Circuit& circuit, const RunOptions& options);

// Validates circuit and the output request, plans, and returns the Plan
// without executing it -- the same validation and planning run() performs,
// stopping before Runtime::execute() (docs/BUILD_PLAN.md Phase 7, for
// `menos explain`'s plan-only preview: "previews a plan without paying
// to execute it"). Throws the same exceptions as run() for the same
// reasons. Named preview(), not plan(), because menos::plan is already
// the planner module's own namespace (menos::plan::JobSpec etc.) --
// declaring a function of that exact name in the same enclosing
// namespace does not compile (a namespace and any other entity cannot
// share a name visible in the same scope).
[[nodiscard]] Plan preview(const Circuit& circuit, const RunOptions& options);

// run() and preview() reuse a plan across calls whose jobs differ only in
// gate parameter values (a variational loop re-runs one circuit structure
// with new angles): the plan is keyed on the job's structure and the
// options, never on the angles, and a job the key cannot cover is planned
// afresh (src/plan/plan_cache.h states the exact scope). The reused plan is
// the plan plan_job would compute for the new angles, so results and
// explain() are unchanged; only the planning time is saved.
struct PlanCacheStats {
    std::uint64_t hits;      // run()/preview() calls served from the cache
    std::uint64_t misses;    // planned and stored
    std::uint64_t bypasses;  // planned and not stored (uncacheable job or plan)
};
[[nodiscard]] PlanCacheStats plan_cache_stats();
// Empties the cache and zeroes the statistics.
void plan_cache_clear();

}  // namespace menos
