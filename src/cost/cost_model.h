#pragma once

#include <cstdint>
#include <limits>
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
// every other engine ignores it.
[[nodiscard]] CostEstimate estimate_direct(const Circuit& circuit,
                                           const SimulationEngine& engine,
                                           bool want_statevector,
                                           bool want_counts,
                                           bool want_expectation,
                                           const Observable* observable,
                                           std::uint64_t budget_bytes,
                                           const AccuracyPolicy& accuracy_policy,
                                           std::uint64_t trajectories = 0,
                                           std::uint64_t prune_above_work_units =
                                               std::numeric_limits<std::uint64_t>::max());

// The Factorized candidate: one region per interaction-graph component,
// executed sequentially, each independently costed via
// pick_cheapest_engine on its own restricted observable
// (common::restrict_observable, docs/BUILD_PLAN.md Phase 5 decision 4 --
// a term outside a component's qubits costs that component nothing
// extra, since its restriction is the identity). Peak bytes is the
// maximum single region's peak (regions are not all simultaneously
// live), plus the global 16 * 2^n output allocation when a statevector
// is requested. Work is the sum of each region's own cost, plus 2^n
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
[[nodiscard]] CostEstimate estimate_cut(const analysis::CutPlan& cut_plan,
                                        const Observable& observable,
                                        std::span<const SimulationEngine* const> engines,
                                        std::uint64_t budget_bytes,
                                        const AccuracyPolicy& accuracy_policy,
                                        std::uint64_t prune_above_work_units =
                                            std::numeric_limits<std::uint64_t>::max());

[[nodiscard]] CostEstimate estimate_factorized(const Circuit& circuit,
                                               const analysis::InteractionGraph& graph,
                                               std::span<const SimulationEngine* const> engines,
                                               bool want_statevector,
                                               bool want_counts,
                                               bool want_expectation,
                                               const Observable* observable,
                                               std::uint64_t budget_bytes,
                                               const AccuracyPolicy& accuracy_policy);

}  // namespace menos::cost
