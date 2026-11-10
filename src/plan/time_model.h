#pragma once

// The calibrated time model (Phase 24 model fixes, docs/adr/0028 section 20).
//
// Every CPU engine's estimate() reports TimeTerms: up to kTimeComponents
// structural counts of what its runtime is made of (gates, gate updates,
// amplitude passes, eliminations, shots ...). calibrate_time_model() measures
// what one count of each costs on this machine, engine by engine, by running
// the engine on a few small fixed circuits whose counts differ in proportion,
// and solving for non-negative seconds per count that reproduce the measured
// times (least squares on relative error). The planner prices a candidate as
// the sum of its counts times these figures (planner.cpp, cpu_seconds).
//
// Why per engine and measured: before this, every CPU engine's work units were
// converted at the dense engine's calibrated rate, so a faster dense kernel
// (Phase 24 item 8, a 2-3x rate change) made every other engine look 2-3x
// faster than it is (section 20.1). Each engine timed in its own unit cannot
// drift that way, and no constant relating one engine's work to another's is
// invented anywhere.

#include <array>
#include <vector>

#include "menos/plan.h"

namespace menos::plan {

using SecondsPerTimeCount = std::array<std::array<double, kTimeComponents>, kTimeEngineCount>;

// Measures every CPU engine's seconds per time-component count on this
// machine. An engine whose calibration cannot run (an infeasible or failing
// calibration circuit) is left all zero, which the planner reads as
// "uncalibrated" for that engine alone. Costs a fraction of a second; called
// once per process by calibrate_hardware().
[[nodiscard]] SecondsPerTimeCount calibrate_time_model();

// The non-negative least-squares fit the calibration uses, exposed for its
// unit test: rows[i] are the time-component counts of measurement i (one
// engine's three), seconds[i] its measured time. Minimises
// sum_i ((rows[i] . x) / seconds[i] - 1)^2 over x >= 0, i.e. the relative
// error, so a short calibration run weighs as much as a long one. Components
// that are zero in every row get 0.
[[nodiscard]] std::array<double, kTimeComponents> fit_seconds_per_count(
    const std::vector<std::array<double, kTimeComponents>>& rows,
    const std::vector<double>& seconds);

}  // namespace menos::plan
