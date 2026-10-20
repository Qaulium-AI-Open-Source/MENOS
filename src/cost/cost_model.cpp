#include "cost/cost_model.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/observable_restrict.h"

namespace menos::cost {

using common::checked_add;
using common::checked_mul;
using common::checked_pow2;
using common::ResourceExceeded;

EngineChoice pick_cheapest_engine(std::span<const SimulationEngine* const> engines,
                                  const EngineEstimateRequest& request) {
    std::optional<std::size_t> best;
    CostEstimate best_estimate{/*feasible=*/false, 0, 0, "", {}};
    std::string reasons;
    // Branch and bound: once one engine is feasible, every later engine is
    // only asked whether it can beat it. A pruned engine's running total is
    // already strictly above the incumbent, so it could never have replaced it
    // (replacement needs strictly fewer work units), and a pruned reason is
    // never reported because a feasible engine exists whenever one is pruned.
    // The choice is therefore identical to costing every engine in full.
    EngineEstimateRequest bounded = request;
    for (std::size_t i = 0; i < engines.size(); ++i) {
        bounded.prune_above_work_units =
            best.has_value()
                ? std::min(request.prune_above_work_units, best_estimate.estimated_work_units)
                : request.prune_above_work_units;
        const CostEstimate estimate = engines[i]->estimate(bounded);
        if (!estimate.feasible) {
            if (!reasons.empty()) {
                reasons += "; ";
            }
            reasons += std::string(engines[i]->name()) + ": " + estimate.infeasible_reason;
            continue;
        }
        if (!best.has_value() ||
            estimate.estimated_work_units < best_estimate.estimated_work_units) {
            best = i;
            best_estimate = estimate;
        }
    }
    if (!best.has_value()) {
        return EngineChoice{0, CostEstimate{/*feasible=*/false, 0, 0, reasons, {}}};
    }
    return EngineChoice{*best, best_estimate};
}

CostEstimate estimate_direct(const Circuit& circuit,
                             const SimulationEngine& engine,
                             bool want_statevector,
                             bool want_counts,
                             bool want_expectation,
                             const Observable* observable,
                             std::uint64_t budget_bytes,
                             const AccuracyPolicy& accuracy_policy,
                             std::uint64_t trajectories,
                             std::uint64_t prune_above_work_units) {
    const EngineEstimateRequest request{circuit.num_qubits(),
                                        &circuit.operations(),
                                        want_statevector,
                                        want_counts,
                                        want_expectation,
                                        observable,
                                        accuracy_policy,
                                        budget_bytes,
                                        trajectories,
                                        prune_above_work_units};
    CostEstimate estimate = engine.estimate(request);
    if (!estimate.feasible) {
        return estimate;
    }
    if (estimate.estimated_peak_bytes > budget_bytes) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "direct " + std::string(engine.name()) + " simulation requires " +
                                std::to_string(estimate.estimated_peak_bytes) +
                                " bytes, budget is " + std::to_string(budget_bytes) + " bytes",
                            {}};
    }
    return estimate;
}

CostEstimate estimate_factorized(const Circuit& circuit,
                                 const analysis::InteractionGraph& graph,
                                 std::span<const SimulationEngine* const> engines,
                                 bool want_statevector,
                                 bool want_counts,
                                 bool want_expectation,
                                 const Observable* observable,
                                 std::uint64_t budget_bytes,
                                 const AccuracyPolicy& accuracy_policy) {
    const std::size_t num_components = graph.components.size();
    const std::uint32_t n = circuit.num_qubits();

    // Per-component global-to-local qubit maps, the same convention
    // src/plan/region_builder.cpp's own remap() uses for the identical
    // reason: an engine's estimate() (and, from Phase 6, MPSEngine's
    // Schmidt-rank-bound pass in particular) is handed num_qubits = this
    // component's own LOCAL qubit count, so every operand it reads must
    // already be in [0, num_qubits) -- a GLOBAL qubit id from a
    // non-first component would otherwise index past that range.
    std::vector<std::vector<QubitId>> component_global_to_local(num_components);
    for (std::size_t c = 0; c < num_components; ++c) {
        component_global_to_local[c].assign(n, kNoLocalQubit);
        const std::vector<QubitId>& component_qubits = graph.components[c].qubits;
        for (std::size_t local = 0; local < component_qubits.size(); ++local) {
            component_global_to_local[c][component_qubits[local]] = static_cast<QubitId>(local);
        }
    }

    // Route each operation into its owning component's operation list,
    // remapped to that component's own local qubit ids, in one pass:
    // valid for every operand by the interaction graph's own
    // construction (every multi-operand op's operands share one
    // component, so operands[0]'s component is authoritative).
    // Exhaustive over Operation: a new alternative fails the static_assert
    // until it is given its own rule (docs/adr/0016 section 7).
    std::vector<std::vector<Operation>> component_ops(num_components);
    const auto local_qubits_of = [&](const std::vector<QubitId>& qubits, std::size_t c) {
        const std::vector<QubitId>& local_map = component_global_to_local[c];
        std::vector<QubitId> local_qubits(qubits.size());
        for (std::size_t i = 0; i < qubits.size(); ++i) {
            local_qubits[i] = local_map[qubits[i]];
        }
        return local_qubits;
    };
    for (const Operation& op : circuit.operations()) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    const std::size_t c = graph.qubit_component[o.qubits[0]];
                    component_ops[c].push_back(
                        GateOp{o.opcode, local_qubits_of(o.qubits, c), o.params});
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    const std::size_t c = graph.qubit_component[o.qubits[0]];
                    component_ops[c].push_back(UnitaryOp{local_qubits_of(o.qubits, c), o.matrix});
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    // Routed like a gate, so each region's engine sees its own
                    // channels and either runs them or refuses with a reason.
                    const std::size_t c = graph.qubit_component[o.qubits[0]];
                    component_ops[c].push_back(ChannelOp{local_qubits_of(o.qubits, c), o.kraus});
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                    // No per-region work term. A measurement with a readout error
                    // is still routed, so the region's engine sees it and a
                    // pure-state engine refuses it instead of the error being
                    // dropped (docs/adr/0016 section 7). Plain measurements stay
                    // unrouted, so a noiseless job's estimates are unchanged.
                    if (o.readout_error.has_value()) {
                        const std::size_t c = graph.qubit_component[o.qubit];
                        MeasureOp local = o;
                        local.qubit = component_global_to_local[c][o.qubit];
                        component_ops[c].push_back(local);
                    }
                }
            },
            op);
    }

    try {
        std::uint64_t peak_bytes = 0;
        std::uint64_t total_work = 0;
        for (std::size_t c = 0; c < num_components; ++c) {
            // An idle component (no GateOp/UnitaryOp ever touches it) is
            // free: the runtime never allocates its state or samples it,
            // using a static |0> constant instead (docs/BUILD_PLAN.md
            // Phase 3, 3.2). Charging any engine's estimate() here would
            // overstate factorized's true cost and could bias the
            // planner away from a genuinely cheaper factorized plan.
            if (component_ops[c].empty()) {
                continue;
            }
            const std::vector<QubitId>& component_qubits = graph.components[c].qubits;
            const auto m = static_cast<std::uint32_t>(component_qubits.size());

            std::optional<Observable> restricted;
            const Observable* component_observable = nullptr;
            if (want_expectation) {
                restricted = common::restrict_observable(
                    *observable, component_qubits, component_global_to_local[c]);
                component_observable = &*restricted;
            }

            const EngineEstimateRequest request{m,
                                                &component_ops[c],
                                                want_statevector,
                                                want_counts,
                                                want_expectation,
                                                component_observable,
                                                accuracy_policy,
                                                budget_bytes};
            const EngineChoice choice = pick_cheapest_engine(engines, request);
            if (!choice.estimate.feasible) {
                return choice.estimate;
            }
            peak_bytes = std::max(peak_bytes, choice.estimate.estimated_peak_bytes);
            total_work = checked_add(total_work, choice.estimate.estimated_work_units);
        }

        if (want_statevector) {
            const std::uint64_t global_bytes =
                checked_mul(checked_pow2(n), sizeof(std::complex<double>));
            peak_bytes = checked_add(peak_bytes, global_bytes);
            // One full 2^n-amplitude multiply pass per region (the
            // incremental recombination routine, src/runtime/recombine_statevector.h).
            const std::uint64_t recombination_work =
                checked_mul(checked_pow2(n), static_cast<std::uint64_t>(num_components));
            total_work = checked_add(total_work, recombination_work);
        }

        if (peak_bytes > budget_bytes) {
            return CostEstimate{/*feasible=*/false,
                                0,
                                0,
                                "factorized simulation requires " + std::to_string(peak_bytes) +
                                    " bytes, budget is " + std::to_string(budget_bytes) + " bytes",
                                {}};
        }
        return CostEstimate{
            /*feasible=*/true, peak_bytes, total_work, /*infeasible_reason=*/"", {}};
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

namespace {

// The estimate returned when estimate_cut's branch-and-bound bound is passed
// while costing fragment `fragment_index` (0-based).
CostEstimate pruned_cut_estimate(std::size_t fragment_index,
                                 std::size_t num_fragments,
                                 std::uint64_t reached_work_units,
                                 std::uint64_t bound) {
    return CostEstimate{
        /*feasible=*/false,
        0,
        0,
        std::string(kPrunedReasonPrefix) + ": costing stopped in fragment " +
            std::to_string(fragment_index + 1) + " of " + std::to_string(num_fragments) + ", at " +
            std::to_string(reached_work_units) + " work units, already more than the " +
            std::to_string(bound) + " of a cheaper feasible alternative",
        {}};
}

}  // namespace

CostEstimate estimate_cut(const analysis::CutPlan& cut_plan,
                          const Observable& observable,
                          std::span<const SimulationEngine* const> engines,
                          std::uint64_t budget_bytes,
                          const AccuracyPolicy& accuracy_policy,
                          std::uint64_t prune_above_work_units) {
    if (cut_plan.fragments.size() < 2) {
        return CostEstimate{
            /*feasible=*/false, 0, 0, "a cut plan needs at least two fragments", {}};
    }
    // An approximate fragment would make the 8^K signed sum's error
    // uncharacterized (docs/adr/0013 section 5.2). Refuse rather than guess.
    if (accuracy_policy.kind != AccuracyPolicyKind::Exact) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "cut reconstruction requires exact fragments: no reviewed source "
                            "bounds how a truncated fragment's error composes across the 8^K "
                            "signed terms of the reconstruction",
                            {}};
    }

    try {
        std::uint64_t peak_bytes = 0;
        std::uint64_t total_work = 0;
        std::size_t costed_fragments = 0;
        // Under a branch-and-bound bound, fragments are costed in descending
        // order of their evaluation count, so a dear plan crosses the bound
        // after as few engine estimates as possible. The total (a sum) and the
        // peak (a max) do not depend on the order; without a bound the
        // original order is kept.
        std::vector<std::size_t> fragment_order(cut_plan.fragments.size());
        std::iota(fragment_order.begin(), fragment_order.end(), std::size_t{0});
        if (prune_above_work_units != std::numeric_limits<std::uint64_t>::max()) {
            std::vector<std::uint64_t> evaluation_counts(cut_plan.fragments.size());
            for (std::size_t f = 0; f < cut_plan.fragments.size(); ++f) {
                evaluation_counts[f] = analysis::distinct_evaluation_count(cut_plan.fragments[f]);
            }
            std::stable_sort(
                fragment_order.begin(), fragment_order.end(), [&](std::size_t a, std::size_t b) {
                    return evaluation_counts[a] > evaluation_counts[b];
                });
        }
        for (const std::size_t fragment_index : fragment_order) {
            const analysis::CutFragment& fragment = cut_plan.fragments[fragment_index];
            // Cost this fragment on a representative assignment. Every
            // assignment has the same circuit shape -- only which Clifford
            // prepares each cut wire and which Pauli is measured differ, and
            // neither changes an engine's cost model.
            const analysis::FragmentAssignment probe{
                std::vector<ir::CutObservable>(fragment.measured.size(), ir::CutObservable::Z),
                std::vector<ir::CutPrepState>(fragment.prepared.size(), ir::CutPrepState::MinusI)};
            const Circuit fragment_circuit = analysis::build_fragment_circuit(fragment, probe);

            // Eq. (4.1): this fragment runs once per DISTINCT assignment, not
            // once per branch.
            const std::uint64_t evaluations = analysis::distinct_evaluation_count(fragment);
            std::uint64_t fragment_work = 0;
            std::uint64_t fragment_peak = 0;
            for (const auto& [term, coefficient] : observable.terms) {
                static_cast<void>(coefficient);
                const PauliString fragment_term =
                    analysis::build_fragment_term(fragment, term, probe);
                const Observable fragment_observable{{{fragment_term, {1.0, 0.0}}}};
                const EngineEstimateRequest request{fragment_circuit.num_qubits(),
                                                    &fragment_circuit.operations(),
                                                    /*want_statevector=*/false,
                                                    /*want_counts=*/false,
                                                    /*want_expectation=*/true,
                                                    &fragment_observable,
                                                    accuracy_policy,
                                                    budget_bytes};
                const EngineChoice choice = pick_cheapest_engine(engines, request);
                if (!choice.estimate.feasible) {
                    return CostEstimate{
                        /*feasible=*/false,
                        0,
                        0,
                        "no engine can run one cut fragment: " + choice.estimate.infeasible_reason,
                        {}};
                }
                fragment_work = checked_add(fragment_work, choice.estimate.estimated_work_units);
                fragment_peak = std::max(fragment_peak, choice.estimate.estimated_peak_bytes);
                // Branch and bound, term by term: the fragment's work so far,
                // times its evaluation count, is already part of the total.
                if (prune_above_work_units != std::numeric_limits<std::uint64_t>::max()) {
                    const std::uint64_t running =
                        checked_add(total_work, checked_mul(evaluations, fragment_work));
                    if (running > prune_above_work_units) {
                        return pruned_cut_estimate(costed_fragments,
                                                   cut_plan.fragments.size(),
                                                   running,
                                                   prune_above_work_units);
                    }
                }
            }

            total_work = checked_add(total_work, checked_mul(evaluations, fragment_work));
            // Eq. (4.3): max, not sum.
            peak_bytes = std::max(peak_bytes, fragment_peak);
            ++costed_fragments;
            // Branch and bound: every remaining fragment and the
            // recombination term below only add work, so a running total
            // strictly above the bound can no longer win.
            if (total_work > prune_above_work_units) {
                return pruned_cut_estimate(costed_fragments - 1,
                                           cut_plan.fragments.size(),
                                           total_work,
                                           prune_above_work_units);
            }
        }

        // Eq. (4.2): 8^K signed scalar multiply-adds per observable term.
        std::uint64_t branches = 1;
        for (std::size_t i = 0; i < cut_plan.cuts.size(); ++i) {
            branches = checked_mul(branches, ir::kWireCutTermCount);
        }
        total_work = checked_add(total_work, checked_mul(branches, observable.terms.size()));

        if (peak_bytes > budget_bytes) {
            return CostEstimate{/*feasible=*/false,
                                0,
                                0,
                                "cut fragments require " + std::to_string(peak_bytes) +
                                    " bytes, budget is " + std::to_string(budget_bytes) + " bytes",
                                {}};
        }
        return CostEstimate{
            /*feasible=*/true, peak_bytes, total_work, /*infeasible_reason=*/"", {}};
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

}  // namespace menos::cost
