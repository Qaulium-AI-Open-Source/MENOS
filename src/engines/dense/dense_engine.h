#pragma once

#include "ir/fuse_gates.h"
#include "menos/simulation_engine.h"

namespace menos::engines::dense {

// Wraps the existing dense-statevector primitives (StateVector,
// apply_matrix, sample, CumulativeDistribution -- all unchanged, see
// their own headers) behind the SimulationEngine interface introduced in
// Phase 4. The only engine that can produce a statevector.
class DenseEngine : public SimulationEngine {
public:
    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] Engine engine_id() const override;
    [[nodiscard]] Backend backend() const override;
    [[nodiscard]] EngineCapabilities capabilities() const override;
    [[nodiscard]] CostEstimate estimate(const EngineEstimateRequest& request) const override;
    [[nodiscard]] EngineExecutionResult execute(
        const EngineExecutionRequest& request) const override;
};

// The work-unit cost of the dense algorithm for request, the figure
// DenseEngine::estimate reports (docs/BUILD_PLAN.md Phase 3, 3.5 / Phase 5's
// expectation extension): 2^m (init) + sum over gates of 2^(m + operand count)
// + [2^m if want_statevector] + [2^m if want_counts] + [num_terms * 2^m if
// want_expectation]. Shared with the disk-tier engine (Phase 23,
// docs/adr/0027), which does the same arithmetic on the same amplitudes and
// differs only in where they live; the tier's own cost is its TierCost.
// Throws common::ResourceExceeded when the figure overflows.
[[nodiscard]] std::uint64_t dense_work_units(const EngineEstimateRequest& request);

// The same figure for the fused sweep (Phase 24 item 7, docs/adr/0028 section
// 17): one pass of 2^(m + operand count) per block of `plan`, the plan of
// request's operations, instead of one per gate. What every dense engine
// charges with EngineEstimateRequest::fuse_gates set.
[[nodiscard]] std::uint64_t dense_work_units(const EngineEstimateRequest& request,
                                             const ir::FusionPlan& plan);

}  // namespace menos::engines::dense
