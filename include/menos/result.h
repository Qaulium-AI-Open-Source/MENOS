#pragma once

#include <complex>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "menos/bitstring.h"
#include "menos/plan.h"

namespace menos {

// One abandoned attempt at a job, and why it was abandoned (Phase 11,
// docs/adr/0015).
//
// The threshold is the job's own time_budget_seconds. No factor, margin or
// tuning constant is involved: Phase 10 made the budget a strict constraint
// against a plan-time ESTIMATE, and this phase enforces it against what
// actually happened.
//
// `observed_seconds` is elapsed time within the ABANDONED ATTEMPT, and
// `predicted_seconds` is what that attempt's own plan predicted, so the two side
// by side are the prediction-versus-observation the report needs. The budget
// bounds one attempt, so a job that replanned took longer in total than its
// budget; that is why the abandoned attempt's seconds are reported here instead
// of being folded into Result::execute_seconds(), which times only the attempt
// that succeeded.
//
// boundaries_crossed counts the safe boundaries the abandoned engine invocation
// reached. A boundary is a gate application, an observable term, or a sampled
// shot, depending on the engine and on which phase of its work was running (see
// ExecutionMonitor). No total accompanies it, deliberately: those kinds are not
// commensurable, so any single "of N" figure would be a half-truth. It is also
// not turned into a progress fraction or used to project a finish time, because
// measurement showed a rate taken from partial progress mispredicts the total by
// 0.16x-0.40x on the MPS engine, whose cost per operation climbs as its bond
// dimension grows (docs/adr/0015).
//
// superseded_plan is the complete abandoned Plan, not a summary, so a caller
// can call explain() on it. Result::explain() prints only a one-line summary of
// it, to keep the report readable.
struct ReplanEvent {
    std::size_t sequence;
    double threshold_seconds;
    double observed_seconds;
    double predicted_seconds;
    std::uint64_t boundaries_crossed;
    Engine engine;
    Backend backend;
    std::string reason;
    Plan superseded_plan;
};

// counts (ordered by key -- std::map -- so output is deterministic),
// optional statevector, exact, the Plan, actual state bytes as tracked by
// the engine (not process RSS), wall-clock seconds for execute and
// sample, and the peak host and device memory the execution held.
class Result {
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    Result(std::map<BitString, std::uint64_t> counts,
           std::optional<std::vector<std::complex<double>>> statevector,
           std::optional<std::complex<double>> expectation_value,
           bool exact,
           Plan plan,
           std::uint64_t actual_state_bytes,
           double execute_seconds,
           double sample_seconds,
           std::vector<ReplanEvent> replans = {},
           std::optional<double> expectation_standard_error = std::nullopt,
           std::uint64_t peak_host_bytes = 0,
           std::uint64_t peak_device_bytes = 0);
    // NOLINTEND(bugprone-easily-swappable-parameters)

    [[nodiscard]] const std::map<BitString, std::uint64_t>& counts() const;
    [[nodiscard]] const std::optional<std::vector<std::complex<double>>>& statevector() const;
    // The job's expectation value, populated iff RunOptions::expectation
    // was set (docs/BUILD_PLAN.md Phase 5): sum over terms of
    // coefficient * <psi|P|psi>, combined across regions for a
    // Factorized plan via src/runtime/combine_expectation.h.
    [[nodiscard]] const std::optional<std::complex<double>>& expectation_value() const;
    // The reported standard error of expectation_value(), present only when
    // that value is a statistical estimate (a trajectory plan, Phase 13,
    // docs/adr/0017 section 3): s / sqrt(N), measured from the run's own
    // per-trajectory values. It is an estimate of the error, not a bound, and is
    // usually below the requested RunOptions::expectation_precision. nullopt for
    // every exact result -- never 0.0, because "exact" and "a standard error of
    // zero" are different claims.
    [[nodiscard]] const std::optional<double>& expectation_standard_error() const;
    [[nodiscard]] bool exact() const;
    [[nodiscard]] const Plan& plan() const;
    [[nodiscard]] std::uint64_t actual_state_bytes() const;
    // The most memory the successful attempt's execution held at once, in
    // bytes, on the host and on the CUDA device (Phase 21, docs/adr/0025
    // section 3.2). Measured, not estimated: the runtime's memory ledger
    // adds up every buffer whose size grows with the problem -- states,
    // workspace, samplers, histograms, tensors -- as it is allocated and
    // freed. Allocations that do not grow with the problem, and the
    // allocator's own overhead, are left out, so the figure is at most what
    // the process allocated and at most the plan's estimate. Planning is not
    // included. peak_device_bytes() is 0 for a plan with no CUDA region.
    [[nodiscard]] std::uint64_t peak_host_bytes() const;
    [[nodiscard]] std::uint64_t peak_device_bytes() const;
    [[nodiscard]] double execute_seconds() const;
    [[nodiscard]] double sample_seconds() const;

    // Every plan that was abandoned before this Result was produced, oldest
    // first. Empty for a run that never crossed its time budget, which is
    // every run that sets no budget (Phase 11, docs/adr/0015).
    //
    // execute_seconds() above times the SUCCESSFUL attempt only. The total a
    // caller actually waited is that plus the observed_seconds of the last
    // event here; they are kept separate rather than summed so neither figure
    // has to be explained away.
    [[nodiscard]] const std::vector<ReplanEvent>& replans() const;

    // Attaches a replan history to a Result the runtime has just produced.
    //
    // The runtime cannot know this history: it executes one plan and has no
    // idea another was abandoned first. So the api layer's replan loop supplies
    // it here, exactly once, before the Result reaches the caller. Consumes
    // *this (rvalue-qualified) rather than copying, because a Result can hold a
    // statevector of 16 * 2^n bytes -- 67 MB at 22 qubits -- and rebuilding one
    // just to add a few events would be a real cost for a feature that is
    // supposed to be nearly free.
    [[nodiscard]] Result with_replans(std::vector<ReplanEvent> replans) &&;

    // A deterministic text report. Byte-identical across runs of the same
    // circuit, except for the two wall-clock fields (execute_s, sample_s),
    // which are, by construction, not deterministic.
    [[nodiscard]] std::string explain() const;

private:
    std::map<BitString, std::uint64_t> counts_;
    std::optional<std::vector<std::complex<double>>> statevector_;
    std::optional<std::complex<double>> expectation_value_;
    bool exact_;
    Plan plan_;
    std::uint64_t actual_state_bytes_;
    double execute_seconds_;
    double sample_seconds_;
    std::vector<ReplanEvent> replans_;
    std::optional<double> expectation_standard_error_;
    std::uint64_t peak_host_bytes_;
    std::uint64_t peak_device_bytes_;
};

}  // namespace menos
