#include "plan/region_builder.h"

#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/observable_restrict.h"
#include "cost/cost_model.h"

namespace menos::plan {
namespace {

// True iff a region needs an engine: it holds an operation that changes the
// quantum state (a gate, unitary or noise channel) or a measurement whose
// recorded outcome passes through a readout error. A region of plain
// MeasureOps only is idle and is sampled as a constant. Exhaustive over
// Operation, like remap and the routing in build_factorized_regions: a new
// alternative fails the static_assert until it is given its own rule, so an
// operation can never be skipped by being classed as idle (docs/adr/0016
// section 7).
bool has_gate_operation(const std::vector<Operation>& operations) {
    for (const Operation& op : operations) {
        const bool needs_engine = std::visit(
            [](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp> ||
                              std::is_same_v<T, ChannelOp>) {
                    return true;
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                    return o.readout_error.has_value();
                }
            },
            op);
        if (needs_engine) {
            return true;
        }
    }
    return false;
}

// Remaps a single operation's qubit operands from global to local ids via
// the owning region's global_to_local map. MeasureOp::clbit is left
// untouched (a global, unpartitioned namespace). The operation is copied and
// only its qubits are rewritten, so every other field is carried unchanged.
Operation remap(const Operation& op, const std::vector<QubitId>& global_to_local) {
    return std::visit(
        [&](const auto& o) -> Operation {
            using T = std::decay_t<decltype(o)>;
            T local = o;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp> ||
                          std::is_same_v<T, ChannelOp>) {
                for (QubitId& q : local.qubits) {
                    q = global_to_local[q];
                }
            } else {
                static_assert(std::is_same_v<T, MeasureOp>);
                local.qubit = global_to_local[o.qubit];
            }
            return local;
        },
        op);
}

// "<winner> (<rejected engine>: <reason>)..." for every OTHER engine
// that is infeasible for this region -- explain()'s per-region "why"
// (docs/BUILD_PLAN.md Phase 4: mixed-engine and 99%-Clifford-plus-T
// cases). Empty when there is only one engine to choose from.
//
// `estimates` is what pick_cheapest_engine costed for each engine. The
// reasons reported are the unbounded request's, as they always were: an
// engine's estimate there is the unbounded one unless the engine was stopped
// by the prune bound, and only such an engine is costed again here, without
// the bound (pick_cheapest_engine's contract, cost/cost_model.h).
std::string describe_engine_choice(std::span<const SimulationEngine* const> engines,
                                   std::size_t winner_index,
                                   const EngineEstimateRequest& request,
                                   const std::vector<CostEstimate>& estimates) {
    if (engines.size() <= 1) {
        return "";
    }
    common::ensure(estimates.size() == engines.size(),
                   "describe_engine_choice: one estimate per engine expected");
    std::string reason(engines[winner_index]->name());
    for (std::size_t i = 0; i < engines.size(); ++i) {
        if (i == winner_index) {
            continue;
        }
        const bool pruned = !estimates[i].feasible &&
                            estimates[i].infeasible_reason.starts_with(kPrunedReasonPrefix);
        const CostEstimate estimate = pruned ? engines[i]->estimate(request) : estimates[i];
        if (!estimate.feasible) {
            reason += " (" + std::string(engines[i]->name()) +
                      " rejected: " + estimate.infeasible_reason + ")";
        }
    }
    return reason;
}

}  // namespace

Region build_direct_region(const Circuit& circuit,
                           Engine engine,
                           Backend backend,
                           const CostEstimate& estimate,
                           const Observable* observable) {
    const std::uint32_t n = circuit.num_qubits();
    std::vector<QubitId> identity(n);
    std::iota(identity.begin(), identity.end(), QubitId{0});

    return Region{
        .local_to_global = identity,
        .global_to_local = identity,
        .operations = circuit.operations(),
        .engine = engine,
        .backend = backend,
        .engine_selection_reason = "",
        .observable = observable == nullptr ? std::nullopt : std::optional(*observable),
        .estimated_state_bytes = estimate.estimated_peak_bytes,
        .estimated_work_units = estimate.estimated_work_units,
    };
}

std::vector<Region> build_factorized_regions(const Circuit& circuit,
                                             const analysis::InteractionGraph& graph,
                                             std::span<const SimulationEngine* const> engines,
                                             bool want_statevector,
                                             bool want_counts,
                                             bool want_expectation,
                                             const Observable* observable,
                                             std::uint64_t budget_bytes,
                                             const AccuracyPolicy& accuracy_policy) {
    const std::uint32_t n = circuit.num_qubits();
    const std::size_t num_components = graph.components.size();

    std::vector<Region> regions(num_components);
    for (std::size_t c = 0; c < num_components; ++c) {
        regions[c].local_to_global = graph.components[c].qubits;
        regions[c].global_to_local.assign(n, kNoLocalQubit);
        for (std::size_t local = 0; local < regions[c].local_to_global.size(); ++local) {
            regions[c].global_to_local[regions[c].local_to_global[local]] =
                static_cast<QubitId>(local);
        }
        regions[c].engine = Engine::Dense;  // overwritten below unless idle
        regions[c].backend = Backend::Cpu;
        if (want_expectation) {
            regions[c].observable = common::restrict_observable(
                *observable, regions[c].local_to_global, regions[c].global_to_local);
        }
    }

    for (const Operation& op : circuit.operations()) {
        const QubitId first_qubit = std::visit(
            [](const auto& o) -> QubitId {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp> ||
                              std::is_same_v<T, ChannelOp>) {
                    return o.qubits[0];
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                    return o.qubit;
                }
            },
            op);
        const std::uint32_t component = graph.qubit_component[first_qubit];
        regions[component].operations.push_back(remap(op, regions[component].global_to_local));
    }

    for (Region& region : regions) {
        const std::uint32_t m = static_cast<std::uint32_t>(region.local_to_global.size());
        if (!has_gate_operation(region.operations)) {
            // Idle: no allocation, no work, no engine needed
            // (docs/BUILD_PLAN.md Phase 3, 3.2).
            region.estimated_state_bytes = 0;
            region.estimated_work_units = 0;
            continue;
        }
        // region.observable is set for every region in the earlier loop
        // whenever want_expectation is true; not provable to
        // clang-tidy's local static analysis.
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        const Observable* region_observable = want_expectation ? &*region.observable : nullptr;
        const EngineEstimateRequest request{m,
                                            &region.operations,
                                            want_statevector,
                                            want_counts,
                                            want_expectation,
                                            region_observable,
                                            accuracy_policy,
                                            budget_bytes};
        std::vector<CostEstimate> estimates;
        const cost::EngineChoice choice = cost::pick_cheapest_engine(engines, request, &estimates);
        common::ensure(choice.estimate.feasible,
                       "build_factorized_regions: no engine is feasible for a component the "
                       "winning Factorized candidate already costed as feasible -- internal "
                       "invariant violated");
        region.engine = engines[choice.engine_index]->engine_id();
        region.engine_selection_reason =
            describe_engine_choice(engines, choice.engine_index, request, estimates);
        region.estimated_state_bytes = choice.estimate.estimated_peak_bytes;
        region.estimated_work_units = choice.estimate.estimated_work_units;
    }

    return regions;
}

}  // namespace menos::plan
