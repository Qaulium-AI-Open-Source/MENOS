#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "analysis/cut.h"
#include "analysis/interaction_graph.h"
#include "menos/circuit.h"
#include "menos/plan.h"
#include "menos/simulation_engine.h"

namespace menos::cost {

// CostEstimate now lives in include/menos/plan.h (Phase 4): it is the
// same shape SimulationEngine::estimate() returns for a single region,
// and a public header must not depend on src/cost. Every existing
// menos::cost::CostEstimate call site keeps compiling unchanged.
using menos::CostEstimate;

// The runtime's own measurement bookkeeping for one job, held beside the
// engines (src/runtime/runtime.cpp, Phase 21, docs/adr/0025 section 3.4): the
// surviving write of each clbit (ir::compute_final_writes, one map node per
// clbit) and the measure targets built from it. estimate_direct and
// estimate_factorized add it to every candidate's peak; the planner's disk-tier
// block adds it the same way (Phase 23).
[[nodiscard]] std::uint64_t measurement_bookkeeping_bytes(const Circuit& circuit);

// From `engines` (fixed enumeration order), the cheapest engine whose
// estimate() is feasible for this region's request; ties go to the
// earliest index (docs/BUILD_PLAN.md Phase 4: "each factorized region
// independently takes its cheapest eligible engine", "ties go to dense"
// -- dense is engines[0] by available_engines()'s fixed order). If no
// engine is feasible, the returned CostEstimate is infeasible and
// infeasible_reason concatenates every engine's own reason.
struct EngineChoice {
    std::size_t engine_index;
    CostEstimate estimate;
};
[[nodiscard]] EngineChoice pick_cheapest_engine(std::span<const SimulationEngine* const> engines,
                                                const EngineEstimateRequest& request);

// The DirectDense/DirectStabilizer/DirectPauli/DirectMPS candidate:
// `engine` runs the whole circuit as one region. want_expectation/
// observable describe the whole job's observable (never region-
// restricted here -- there is only one region). accuracy_policy is
// forwarded to engine.estimate() unchanged (Phase 6); every engine
// before DirectMPS ignores it. trajectories is the count N the
// DirectTrajectory candidate costs (Phase 13, cost::trajectory_count);
// every other engine ignores it (0). shots is the job's shot count, which
// sizes the counts histogram and sampling buffers the peak-byte figure
// includes (Phase 21, docs/adr/0025 section 3.4); 0 without counts.
[[nodiscard]] CostEstimate estimate_direct(const Circuit& circuit,
                                           const SimulationEngine& engine,
                                           bool want_statevector,
                                           bool want_counts,
                                           bool want_expectation,
                                           const Observable* observable,
                                           std::uint64_t budget_bytes,
                                           const AccuracyPolicy& accuracy_policy,
                                           std::uint64_t trajectories,
                                           std::uint64_t shots);

// The Factorized candidate: one region per interaction-graph component,
// executed sequentially, each independently costed via
// pick_cheapest_engine on its own restricted observable
// (common::restrict_observable, docs/BUILD_PLAN.md Phase 5 decision 4 --
// a term outside a component's qubits costs that component nothing
// extra, since its restriction is the identity). Peak bytes is the sum
// of the regions' peaks (each region's outputs are held until all have
// run), plus the global 16 * 2^n output allocation when a statevector is
// requested, plus the runtime's bookkeeping and combined counts histogram
// (Phase 21, docs/adr/0025 section 3.4). Work is the sum of each region's own cost, plus 2^n
// work units per region for the incremental statevector-recombination
// multiply pass when a statevector is requested (one full pass over the
// global amplitude space per region, see
// src/runtime/recombine_statevector.h). No separate count-combination
// cost term: the number of distinct sampled outcomes is data-dependent
// and unknowable at plan time, and is already upper-bounded by each
// region's own sampling term.
// The CutAndRecombine candidate (docs/adr/0013 section 4): each fragment is
// costed on its own cheapest eligible engine, multiplied by that fragment's
// distinct-evaluation count 4^m * 6^p (Eq. 4.1), plus 8^K scalar
// multiply-adds for the reconstruction (Eq. 4.2). Peak bytes is the MAX over
// fragments, not their sum (Eq. 4.3): fragments run sequentially and each
// releases its state, the same model Phase 3's factorized runtime uses.
//
// Feasible only when every fragment's chosen engine is exact. A cut sums 8^K
// signed terms, and no consulted source bounds how a per-fragment error
// composes across them (docs/adr/0013 section 5.2), so an approximate fragment
// is refused rather than handed an invented error bound.
//
// It also chooses, for every fragment evaluation, the engine the runtime runs
// it on (Phase 21, docs/adr/0025): pick_cheapest_engine over exactly the
// request execute_cut builds -- the fragment circuit for that assignment and
// its whole observable -- with the job's budget. The runtime then does no
// planning of its own. Peak bytes is what execute_cut holds at once: the
// cut-plan search it repeats (on `circuit`, the reduced circuit it runs), or,
// with that plan, its assignment lists and the values computed so far held,
// one evaluation's fragment circuit, observable and engine peak, or the
// reconstruction's per-term lists.
struct CutEstimate {
    CostEstimate estimate;
    std::vector<std::vector<CutEvaluation>> evaluations;
};
[[nodiscard]] CutEstimate estimate_cut(const Circuit& circuit,
                                       const analysis::CutPlan& cut_plan,
                                       const Observable& observable,
                                       std::span<const SimulationEngine* const> engines,
                                       std::uint64_t budget_bytes,
                                       const AccuracyPolicy& accuracy_policy);

[[nodiscard]] CostEstimate estimate_factorized(const Circuit& circuit,
                                               const analysis::InteractionGraph& graph,
                                               std::span<const SimulationEngine* const> engines,
                                               bool want_statevector,
                                               bool want_counts,
                                               bool want_expectation,
                                               const Observable* observable,
                                               std::uint64_t budget_bytes,
                                               const AccuracyPolicy& accuracy_policy,
                                               std::uint64_t shots);

}  // namespace menos::cost
