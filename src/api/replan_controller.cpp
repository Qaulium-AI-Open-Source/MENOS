#include "api/replan_controller.h"

#include <chrono>
#include <format>
#include <set>
#include <string>
#include <utility>

#include "common/error.h"
#include "plan/enum_names.h"
#include "plan/planner.h"
#include "runtime/runtime.h"

namespace menos::api {
namespace {

// The seconds this plan predicted for itself, from its own winning candidate.
// Negative when the planning machine was never calibrated, which cannot happen
// on a path that reached here (plan_job refuses a time budget against an
// uncalibrated profile) but is reported rather than asserted.
double predicted_seconds_of(const Plan& plan) {
    for (const CandidateSummary& candidate : plan.candidates) {
        if (candidate.is_winner) {
            return candidate.estimated_seconds;
        }
    }
    return -1.0;
}

// The common prefix of every refusal: what was observed, against what budget,
// and by which engine. Callers append the specific reason.
std::string refusal_message(const ReplanRequested& request, double budget, const Plan& plan) {
    return std::format(
        "menos::run: observed runtime of {:.6g} s exceeded the job's time budget of {:.6g} s "
        "after {} safe boundaries on engine={} backend={}, and no different plan meets that "
        "budget",
        request.elapsed_seconds(),
        budget,
        request.boundaries_crossed(),
        plan::engine_name(plan.regions.empty() ? Engine::Dense : plan.regions[0].engine),
        plan::backend_name(plan.regions.empty() ? Backend::Cpu : plan.regions[0].backend));
}

// One attempt at `plan`. Runtime::execute moves the plan into the Result only
// on success; an abandoned attempt leaves it as it was, which the replan loop
// below relies on to describe the plan it replaces (docs/adr/0025).
Result run_attempt(const runtime::Runtime& rt,
                   Plan& plan,
                   const plan::JobSpec& job_spec,
                   const BudgetMonitor& monitor) {
    return rt.execute(std::move(plan), job_spec, &monitor);
}

}  // namespace

ElapsedSource steady_elapsed_source() {
    const auto origin = std::chrono::steady_clock::now();
    return [origin]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - origin).count();
    };
}

BudgetMonitor::BudgetMonitor(double budget_seconds, ElapsedSource elapsed)
    : budget_seconds_(budget_seconds), elapsed_(std::move(elapsed)) {}

void BudgetMonitor::at_safe_boundary(std::uint64_t boundaries_crossed) const {
    const double elapsed = elapsed_();
    if (elapsed > budget_seconds_) {
        throw ReplanRequested(
            {.elapsed_seconds = elapsed, .boundaries_crossed = boundaries_crossed});
    }
}

void BudgetMonitor::at_region_boundary(std::size_t region, bool starting) const {
    const double elapsed = elapsed_();
    if (starting) {
        region_timings_.push_back(
            RegionTiming{.region = region, .start_seconds = elapsed, .end_seconds = std::nullopt});
    } else if (!region_timings_.empty() && region_timings_.back().region == region) {
        region_timings_.back().end_seconds = elapsed;
    }
}

Result run_with_replanning(const plan::JobSpec& job_spec,
                           const plan::HardwareProfile& hardware_profile,
                           Plan initial_plan,
                           const ElapsedSourceFactory& make_elapsed) {
    const runtime::Runtime rt;

    // No budget means no monitor, which means no engine observes anything and
    // this is exactly the pre-Phase-11 path. Kept as an early return rather
    // than a null monitor threaded through the loop, so the unmonitored case is
    // visibly untouched.
    if (!job_spec.time_budget_seconds.has_value()) {
        return rt.execute(std::move(initial_plan), job_spec);
    }

    const double budget = *job_spec.time_budget_seconds;
    plan::RuntimeObservations observations;
    std::vector<ReplanEvent> replans;
    Plan current = std::move(initial_plan);

    // WHY THE LOOP TERMINATES (docs/adr/0028 section 16.8, extending the
    // Phase 11 argument of docs/adr/0015).
    //
    // A job has a finite set of plans: a fixed list of strategies, and for each
    // a finite choice of (engine, backend, tier) per region, the regions being
    // fixed by the circuit. An overrun of plan p after elapsed T > budget
    // records, before replanning:
    //   - a single-region plan: the floor of its one (engine, backend, tier),
    //     at T (the Phase 11 floor);
    //   - a Factorized plan: a floor for each region that started, at the
    //     time that region took (whole, or up to the crossing), keyed by the
    //     region's qubits and implementation; and a plan floor at T, keyed by
    //     the plan's strategy and every region's key.
    // Floors only rise (RuntimeObservations), and the planner raises a
    // candidate's seconds to the floor that matches it exactly (planner.cpp,
    // apply_time_budget). So after the overrun p's predicted seconds are at
    // least T > budget for the rest of the job, p is infeasible, and every
    // replan either throws (no feasible plan is left) or returns a plan not run
    // before. With finitely many plans, the loop ends.
    //
    // The region floors are not needed for termination -- the plan floor
    // alone gives it, and it covers time spent between regions, which no
    // region floor does. They exist so that another plan reusing a region on
    // the same implementation is priced with what was learned.
    //
    // A CutAndRecombine plan's overrun is not attributed per region and is
    // refused, as before Phase 24.
    //
    // The premise "a replan never returns a plan already run" is checked on
    // every iteration rather than assumed: `tried` holds the key of every plan
    // run so far, and a repeat is a loud internal error, never a silent loop.
    std::set<plan::PlanKey> tried{plan::plan_key(current)};
    for (;;) {
        // One clock per attempt. See BudgetMonitor's own comment for why the
        // budget bounds an attempt rather than the whole job.
        const BudgetMonitor monitor(budget, make_elapsed());
        try {
            return run_attempt(rt, current, job_spec, monitor).with_replans(std::move(replans));
        } catch (const ReplanRequested& request) {
            const bool factorized = current.strategy == Strategy::Factorized;
            // A cut plan's fragments are not timed per fragment, and claiming
            // the whole of the elapsed time for any one engine would be an
            // over-claim; the floor's soundness is the property everything
            // else rests on. With nothing sound to learn, it is refused.
            if (current.regions.size() != 1 && !factorized) {
                throw common::ResourceExceeded(
                    refusal_message(request, budget, current) +
                    " -- this plan runs several engines in separate regions, and elapsed time is "
                    "not attributed per region, so the overrun yields no information a replan "
                    "could use");
            }

            std::vector<RegionObservation> region_observations;
            // The engine the report names: the plan's one region, or the
            // Factorized region that was running at the crossing (the last
            // one started; the first region if the crossing came after every
            // region had finished, during sampling).
            std::size_t named_region = 0;
            if (factorized) {
                for (const BudgetMonitor::RegionTiming& timing : monitor.region_timings()) {
                    const Region& region = current.regions.at(timing.region);
                    const double observed = timing.end_seconds.value_or(request.elapsed_seconds()) -
                                            timing.start_seconds;
                    observations.observe_region(plan::region_key(region), observed);
                    region_observations.push_back(RegionObservation{
                        .qubits = plan::region_key(region).qubits,
                        .engine = region.engine,
                        .backend = region.backend,
                        .tier = region.tier,
                        .estimated_seconds = region.estimated_seconds,
                        .observed_seconds = observed,
                        .completed = timing.end_seconds.has_value(),
                    });
                    if (!timing.end_seconds.has_value()) {
                        named_region = timing.region;
                    }
                }
                observations.observe_plan(plan::plan_key(current), request.elapsed_seconds());
            } else {
                observations.observe(current.regions[0].engine,
                                     current.regions[0].backend,
                                     current.regions[0].tier,
                                     request.elapsed_seconds());
            }
            const Region& named = current.regions[named_region];

            const double predicted = predicted_seconds_of(current);
            Plan replanned;
            std::string failure;
            // Real wall-clock, not the injected ElapsedSource: that one times
            // execution attempts against the budget, and planning is neither
            // (docs/adr/0028 section 16.7 Q3).
            const auto planning_start = std::chrono::steady_clock::now();
            try {
                replanned = plan::plan_job(job_spec, hardware_profile, observations);
            } catch (const common::ResourceExceeded& exceeded) {
                failure = exceeded.what();
            }
            const double plan_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - planning_start)
                    .count();

            if (!failure.empty()) {
                // The strict reading of time_budget_seconds, carried from plan
                // time into run time: the budget was exceeded and nothing meets
                // it, so the job is refused rather than quietly allowed to
                // overrun. The planner's own reason is carried through.
                throw common::ResourceExceeded(refusal_message(request, budget, current) + " -- " +
                                               failure);
            }
            if (!tried.insert(plan::plan_key(replanned)).second) {
                throw common::InternalError(
                    "menos::run: a replan returned a plan that already ran and overran, which "
                    "means an observed floor failed to exclude it -- internal invariant violated "
                    "(docs/adr/0028 section 16.8)");
            }
            std::string reason =
                factorized
                    ? std::format(
                          "observed runtime exceeded the time budget; this factorized plan "
                          "is excluded by an observed floor of {:.6g} s, and each "
                          "region's own time is recorded below",
                          request.elapsed_seconds())
                    // Names the tier only when it is not the in-memory one, so a
                    // pre-Phase-23 report reads exactly as it did.
                    : std::format(
                          "observed runtime exceeded the time budget; {} on {}{} is excluded by "
                          "an observed floor of {:.6g} s",
                          plan::engine_name(named.engine),
                          plan::backend_name(named.backend),
                          named.tier == Tier::InMemory
                              ? std::string()
                              : std::format(" (tier {})", plan::tier_name(named.tier)),
                          request.elapsed_seconds());
            const Engine named_engine = named.engine;
            const Backend named_backend = named.backend;
            replans.push_back(ReplanEvent{
                .sequence = replans.size(),
                .threshold_seconds = budget,
                .observed_seconds = request.elapsed_seconds(),
                .predicted_seconds = predicted,
                .boundaries_crossed = request.boundaries_crossed(),
                .engine = named_engine,
                .backend = named_backend,
                .reason = std::move(reason),
                .superseded_plan = std::move(current),
                .plan_seconds = plan_seconds,
                .regions = std::move(region_observations),
            });
            current = std::move(replanned);
        }
    }
}

}  // namespace menos::api
