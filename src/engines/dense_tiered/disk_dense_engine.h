#pragma once

#include "menos/simulation_engine.h"

namespace menos::engines::dense_tiered {

// The dense algorithm on the disk tier (Phase 23, docs/adr/0027): the state in
// a scratch file, the gates applied chunk by chunk through a window in host
// memory by the chunked executor. (Dense, Cpu, Disk): the same algorithm and
// hardware as DenseEngine, which is (Dense, Cpu, InMemory), differing only in
// where the state lives, so counts and expectations are bitwise identical. It
// returns no statevector (that would need the whole state in memory), runs a
// job's sole region only, and refuses noise as DenseEngine does. The planner
// reaches for it only when the in-memory state does not fit (reach-only,
// section 9 Q5), or when the caller forces the tier.
class DiskDenseEngine : public SimulationEngine {
public:
    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] Engine engine_id() const override;
    [[nodiscard]] Backend backend() const override;
    [[nodiscard]] Tier tier() const override;
    [[nodiscard]] EngineCapabilities capabilities() const override;
    // Chooses the window from request.budget_bytes (the largest that leaves
    // room for the outputs and one gate's transient) and the chunk from the
    // pass count (choose_pass_plan), unless the request's tier seams fix them;
    // reports the host peak, the dense work units, and the TierCost the
    // executor will incur: passes, chunk and window sizes, link bytes and the
    // file's size. Infeasible, with the reason, for a statevector request, a
    // noisy circuit, a region that is not the job's sole one, a gate wider
    // than any window the budget allows, or a budget below one amplitude's
    // window.
    [[nodiscard]] CostEstimate estimate(const EngineEstimateRequest& request) const override;
    // Creates the scratch file in request.scratch_directory (the system's
    // temporary directory when empty) and runs the plan estimate() reported;
    // actual_bytes is the window's size.
    [[nodiscard]] EngineExecutionResult execute(
        const EngineExecutionRequest& request) const override;
};

}  // namespace menos::engines::dense_tiered
