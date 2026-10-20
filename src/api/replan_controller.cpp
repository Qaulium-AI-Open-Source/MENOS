#include "api/replan_controller.h"

#include <chrono>
#include <format>
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

    // WHY A REPLAN NEVER REPRODUCES THE PLAN THAT JUST OVERRAN, and therefore
    // why there is no same-plan branch below and why this loop terminates.
    //
    // Claim: whenever plan_job returns a plan after a crossing, that plan
    // differs from `current` in strategy, engine or backend. Proof, from code:
    //
    //   1. A crossing is a ReplanRequested thrown by BudgetMonitor only when
    //      elapsed > budget (replan_controller.cpp, at_safe_boundary), and it
    //      carries that same elapsed value.
    //   2. `current.regions.size() == 1` (the only case that reaches the
    //      observation below; anything else is refused above) implies `current`
    //      is a single-engine Direct* plan. A Factorized plan has one region per
    //      interaction-graph component and is only enumerated with >= 2 of them
    //      (planner.cpp, `graph.components.size() >= 2`; region_builder.cpp,
    //      one Region per component); a CutAndRecombine plan has one region per
    //      fragment and a cut plan needs >= 2 fragments (analysis/cut.cpp).
    //   3. So its one region is (E, B) with E = single_engine_of(strategy), and
    //      B is that candidate's backend (Cuda only for dense).
    //   4. observations.observe(E, B, elapsed) records floor >= elapsed > budget
    //      (RuntimeObservations::observe keeps the larger value).
    //   5. plan_job is called with the same budget, so `calibrated` holds inside
    //      it (it throws otherwise, and that throw is not a plan). For every
    //      candidate whose strategy has a single engine, apply_time_budget sets
    //      estimated_seconds = max(model, floor for (engine, candidate.backend))
    //      and marks the candidate infeasible when that exceeds the budget.
    //      Selection skips infeasible candidates.
    //   6. A returned plan equal to `current` would need a winning candidate
    //      with strategy S, engine E and backend B. That candidate's seconds are
    //      >= floor(E, B) > budget, so it is infeasible and cannot have won.
    //      Contradiction.
    //
    // The premises are pinned by tests rather than left as prose:
    // RuntimeObservationTest.EveryDirectCandidateIsExcludedByItsOwnObservedFloor
    // checks step 5 for every Direct* engine, and
    // ReplanTest.EveryReplanInAChainOfCrossingsRunsADifferentEngine checks the
    // conclusion through this function.
    //
    // Consequence: each replan permanently removes at least one (engine,
    // backend) pair from a fixed set, so the loop ends. The set is the five
    // single-engine Direct* strategies, with dense counted twice because the
    // planner can place it on either backend (and, since Phase 11, steers it
    // away from a backend that carries an over-budget floor): six pairs. Only a
    // single-region plan yields a floor, and every single-region plan is one of
    // these. The bound below is the loud failure if that argument were ever
    // broken by a later change -- a bounded loop that throws, rather than one
    // that could spin -- not a policy and not a tuned number. It is the single
    // guard; a separate same-plan check was removed as redundant with it.
    constexpr std::size_t kDistinctSingleRegionPlans = 6;
    for (std::size_t attempt = 0; attempt <= kDistinctSingleRegionPlans; ++attempt) {
        // One clock per attempt. See BudgetMonitor's own comment for why the
        // budget bounds an attempt rather than the whole job.
        const BudgetMonitor monitor(budget, make_elapsed());
        try {
            return run_attempt(rt, current, job_spec, monitor).with_replans(std::move(replans));
        } catch (const ReplanRequested& request) {
            // A floor is attributed to one (engine, backend) pair, so only a
            // single-region plan yields one. A multi-region plan divided its
            // elapsed time between several engines, and this phase does not
            // attribute it per region; claiming the whole of it for any one of
            // them would be an over-claim, and the floor's soundness is the
            // property everything else rests on. With nothing to learn, a
            // replan could only reproduce the plan that just overran, so the
            // job is refused here rather than after a pointless planning pass.
            if (current.regions.size() != 1) {
                throw common::ResourceExceeded(
                    refusal_message(request, budget, current) +
                    " -- this plan runs several engines in separate regions, and elapsed time is "
                    "not attributed per region, so the overrun yields no information a replan "
                    "could use");
            }

            observations.observe(
                current.regions[0].engine, current.regions[0].backend, request.elapsed_seconds());

            const double predicted = predicted_seconds_of(current);
            Plan replanned;
            std::string failure;
            try {
                replanned = plan::plan_job(job_spec, hardware_profile, observations);
            } catch (const common::ResourceExceeded& exceeded) {
                failure = exceeded.what();
            }

            if (!failure.empty()) {
                // The strict reading of time_budget_seconds, carried from plan
                // time into run time: the budget was exceeded and nothing meets
                // it, so the job is refused rather than quietly allowed to
                // overrun. The planner's own reason is carried through.
                throw common::ResourceExceeded(refusal_message(request, budget, current) + " -- " +
                                               failure);
            }
            replans.push_back(ReplanEvent{
                .sequence = replans.size(),
                .threshold_seconds = budget,
                .observed_seconds = request.elapsed_seconds(),
                .predicted_seconds = predicted,
                .boundaries_crossed = request.boundaries_crossed(),
                .engine = current.regions[0].engine,
                .backend = current.regions[0].backend,
                .reason = std::format(
                    "observed runtime exceeded the time budget; {} on {} is excluded by an "
                    "observed floor of {:.6g} s",
                    plan::engine_name(current.regions[0].engine),
                    plan::backend_name(current.regions[0].backend),
                    request.elapsed_seconds()),
                .superseded_plan = std::move(current),
            });
            current = std::move(replanned);
        }
    }

    // Unreachable by the argument above: each iteration removes a candidate from
    // a fixed enumeration, so the loop cannot run out of attempts. Loud rather
    // than silent if that argument is ever broken by a later change.
    throw common::InternalError(
        "menos::run: the replan loop exceeded one attempt per strategy, which means an observed "
        "floor failed to exclude the engine it was recorded for -- internal invariant violated");
}

}  // namespace menos::api
