#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "menos/plan.h"
#include "menos/result.h"
#include "menos/simulation_engine.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "plan/runtime_observations.h"

namespace menos::api {

// Runtime monitoring and replanning (Phase 11, docs/adr/0015).
//
// This lives in src/api rather than src/runtime on purpose. The decision to
// abandon a run needs the planner, and the api layer is the one that already
// owns both the planner and the runtime. Putting it here is what keeps a second
// planner out of the runtime: runtime.cpp passes an ExecutionMonitor down to
// the engines and never looks at it.

// Thrown by BudgetMonitor to abandon a run at a safe boundary. Caught by
// run_with_replanning() below and never seen outside this file's translation
// unit pair.
//
// It derives from std::exception directly and NOT from menos::common::Error.
// That is load-bearing: src/ has nine pre-existing catch sites and every one of
// them catches common::ResourceExceeded specifically, while the CLI catches
// common::Error. Deriving from either would let this signal be silently
// swallowed somewhere between the engine and here. There is no catch(...)
// anywhere in src/ or include/, so a type outside that hierarchy reaches the
// replan loop untouched.
class ReplanRequested : public std::exception {
public:
    // One aggregate rather than two scalar parameters, so the throw site names
    // each field with a designated initializer and the two cannot be
    // transposed silently.
    struct Observation {
        double elapsed_seconds = 0.0;
        std::uint64_t boundaries_crossed = 0;
    };

    explicit ReplanRequested(Observation observation) : observation_(observation) {}

    [[nodiscard]] const char* what() const noexcept override {
        return "menos::api::ReplanRequested: a monitored run exceeded its time budget and is "
               "being abandoned at a safe execution boundary";
    }

    [[nodiscard]] double elapsed_seconds() const {
        return observation_.elapsed_seconds;
    }
    [[nodiscard]] std::uint64_t boundaries_crossed() const {
        return observation_.boundaries_crossed;
    }

private:
    Observation observation_;
};

// Seconds elapsed since the source was created. Injectable so a test can script
// the sequence exactly: the physical signal is wall-clock and therefore not
// reproducible, so pinning it is the only way a threshold test can mean the
// same thing on every machine. The same precedent as Phase 10's synthetic
// HardwareProfile tests.
using ElapsedSource = std::function<double()>;

// Makes a fresh ElapsedSource, one per execution attempt.
using ElapsedSourceFactory = std::function<ElapsedSource()>;

// An ElapsedSource reading a steady_clock started now. The production factory.
[[nodiscard]] ElapsedSource steady_elapsed_source();

// Abandons a run once elapsed time exceeds the budget.
//
// The threshold is the budget itself -- no factor, no margin, no tuning
// constant. Two properties matter and both were chosen on measurement:
//
//   Sound. It reports only what has already happened. The obvious alternative,
//   projecting a finish time from partial progress, was measured to
//   under-project by 4x-6x on the MPS engine (whose per-operation cost climbs
//   with its bond dimension) and would therefore fail to fire on the one
//   workload whose prediction is worst.
//
//   Bounded per ATTEMPT, not per job. This is the one point where the reading
//   of time_budget_seconds had to be made precise, and it is stated here rather
//   than left implicit: the budget bounds ONE execution attempt, and the total
//   wall time of a job that replanned is therefore larger than the budget --
//   the abandoned attempt's seconds are real and are reported, not hidden.
//
//   The alternative, one clock for the whole job, was tried and does not work:
//   crossing the threshold means the budget is ALREADY spent, so every
//   replanned attempt would cross again at its first boundary and no replan
//   could ever complete. The feature would collapse into "detect and refuse",
//   which cannot demonstrate a correct restart at all.
//
//   Termination does not depend on the clock. It comes from the observed
//   floors, which accumulate across attempts in RuntimeObservations: an
//   abandoned attempt records a floor strictly above the budget for the plan
//   it was running (its one engine's floor, or, for a Factorized plan, the
//   plan floor of docs/adr/0028 section 16.8), which permanently excludes that
//   plan from the job's finite set of plans. Resetting the clock cannot
//   un-record a floor, and a floor stays true regardless -- the engine really
//   did spend those seconds.
class BudgetMonitor final : public ExecutionMonitor {
public:
    BudgetMonitor(double budget_seconds, ElapsedSource elapsed);

    void at_safe_boundary(std::uint64_t boundaries_crossed) const override;

    // Records the elapsed time at a Factorized plan's region boundaries, from
    // the same ElapsedSource the budget is checked against, so an injected
    // clock scripts them too (Phase 24, docs/adr/0028 section 16.4). Never
    // throws and never decides anything.
    void at_region_boundary(std::size_t region, bool starting) const override;

    // A region's start and, once it finished, its end, in seconds of this
    // attempt. A region still running at the crossing has no end.
    struct RegionTiming {
        std::size_t region;
        double start_seconds;
        std::optional<double> end_seconds;
    };
    [[nodiscard]] const std::vector<RegionTiming>& region_timings() const {
        return region_timings_;
    }

private:
    double budget_seconds_;
    ElapsedSource elapsed_;
    // Written through the const ExecutionMonitor interface the runtime holds;
    // read by the replan loop after the attempt, on the same thread.
    mutable std::vector<RegionTiming> region_timings_;
};

// plan, execute, and replan on a threshold crossing until a run completes.
//
// `make_elapsed` is called once per execution attempt, so each attempt is timed
// from its own start. Planning is not charged against the budget: it is bounded
// and was measured at 0.05 ms to 85 ms, against budgets that are meaningful
// only when far larger than that.
//
// Throws common::ResourceExceeded when the budget is crossed and no DIFFERENT
// plan meets it -- the strict reading of time_budget_seconds carried from plan
// time into run time. Every other exception, including every pre-Phase-11 one,
// propagates unchanged.
[[nodiscard]] Result run_with_replanning(const plan::JobSpec& job_spec,
                                         const plan::HardwareProfile& hardware_profile,
                                         Plan initial_plan,
                                         const ElapsedSourceFactory& make_elapsed);

}  // namespace menos::api
