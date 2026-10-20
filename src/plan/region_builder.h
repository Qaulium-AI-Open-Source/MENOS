#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "analysis/interaction_graph.h"
#include "menos/plan.h"
#include "menos/simulation_engine.h"

namespace menos::plan {

// One region covering every qubit, local_to_global == [0, 1, ...,
// num_qubits-1] (identity map). `engine` is the winning DirectDense/
// DirectStabilizer candidate's own engine; `estimate` is that same
// candidate's already-computed CostEstimate (from cost::estimate_direct),
// reused directly rather than recomputed with a second, possibly
// engine-mismatched formula -- this is what makes estimated_state_bytes
// correct for DirectStabilizer (a small, polynomial-in-n figure, not the
// dense engine's 16*2^n, which would make budget_bytes unusable for the
// stabilizer engine's own allocation at execution time).
// observable, if not null, is stored on the region unrestricted (there
// is only one region, so nothing to restrict against).
// backend is the hardware the region will execute on. It is passed in rather
// than assumed, because from Phase 10 the dense algorithm has two backends and
// the planner decides between them before building the region (docs/adr/0014).
[[nodiscard]] Region build_direct_region(const Circuit& circuit,
                                         Engine engine,
                                         Backend backend,
                                         const CostEstimate& estimate,
                                         const Observable* observable);

// One region per interaction-graph component, in component order.
// Program order is preserved within each region (a single forward pass
// over circuit.operations()). Each region independently picks its
// cheapest eligible engine from `engines` via cost::pick_cheapest_engine
// (ties go to dense, docs/BUILD_PLAN.md Phase 4), recording why in
// engine_selection_reason for explain(). A region with no GateOp/
// UnitaryOp gets estimated_state_bytes == 0 and estimated_work_units ==
// 0 (an idle component, docs/BUILD_PLAN.md Phase 3, 3.2 -- the runtime
// never allocates or samples it, so no engine is chosen for it either).
// want_statevector/want_counts/want_expectation/observable must be the
// same values the winning Factorized candidate's cost::estimate_factorized
// call used -- the engine each component is assigned here must reproduce
// exactly the per-component choice that estimate implicitly assumed (in
// particular, a component cannot be assigned the stabilizer engine when
// want_statevector is true, even if it is fully Clifford, since
// estimate_factorized would have costed it on dense in that case too).
// When want_expectation, each region's own Region::observable is set to
// its restriction of `observable` (common::restrict_observable).
// budget_bytes is the job's own memory budget, forwarded into each
// component's own EngineEstimateRequest so TensorNetworkEngine's estimate()
// (the only engine that reads it) bounds its own elimination search exactly
// as cost::estimate_factorized's own earlier call already did for this same
// winning candidate -- passing anything else here could make this second
// pick_cheapest_engine call reach a different, or a slower-to-reach, answer
// than the one estimate_factorized already committed to.
[[nodiscard]] std::vector<Region> build_factorized_regions(
    const Circuit& circuit,
    const analysis::InteractionGraph& graph,
    std::span<const SimulationEngine* const> engines,
    bool want_statevector,
    bool want_counts,
    bool want_expectation,
    const Observable* observable,
    std::uint64_t budget_bytes,
    const AccuracyPolicy& accuracy_policy,
    std::uint64_t shots);

}  // namespace menos::plan
