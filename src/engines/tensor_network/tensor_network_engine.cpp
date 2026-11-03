#include "engines/tensor_network/tensor_network_engine.h"

#include <algorithm>
#include <complex>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "analysis/treewidth.h"
#include "common/checked_math.h"
#include "common/error.h"
#include "common/pauli_row.h"
#include "engines/tensor_network/contraction.h"
#include "ir/noise.h"

namespace menos::engines::tensor_network {
namespace {

using common::checked_add;
using common::checked_mul;
using common::checked_pow2;
using common::ResourceExceeded;
using Complex = std::complex<double>;

// 16 * 2^n (a complex128 tensor with n legs), throwing ResourceExceeded
// for n >= 63 exactly like every other checked_pow2 use in this codebase
// (CLAUDE.md: "Compute 2^n only through the checked helpers. Never shift
// by n >= 63"). n arrives here as a std::size_t (a leg/degree count);
// this guards the narrowing to checked_pow2's own unsigned parameter
// explicitly rather than casting and letting an enormous size_t wrap.
std::uint64_t bytes_for_legs(std::size_t n) {
    if (n >= 63) {
        throw ResourceExceeded("bytes_for_legs: " + std::to_string(n) +
                               " legs would overflow a checked 2^n computation");
    }
    return checked_mul(std::uint64_t{16}, checked_pow2(static_cast<unsigned>(n)));
}

// Eliminating a wire of line-graph degree d builds an intermediate
// spanning d+1 indices -- its d neighbouring wires, plus the wire being
// summed over, which must exist in the intermediate before it can be
// summed away. So the object is 2^(d+1) complex128 entries, and each of
// its entries costs a multiply-add to fill, giving at most (d+1) * 2^(d+1)
// scalar multiply-adds for the merges at that step. Summed over every
// step this bounds the term's own contraction cost, in the same shared
// cost unit every other engine's work_units uses (docs/adr/0006:
// analytic, checked-arithmetic, structural, never a wall-clock figure).
//
// The d+1, rather than d, is the correction this phase made: the earlier
// formula used d, which silently understated both this figure and the
// peak-byte figure below by a factor of two. It was not caught by the
// treewidth-exactness suite, because that suite validates the estimator's
// ARITHMETIC (min-fill reproduces known treewidths) and not whether the
// quantity it computes bounds what the contraction really does.
// TensorNetworkEngineTest.EstimatedPeakBytesBoundTheActualPeakAcrossRandomCircuits
// is what actually holds this honest, by comparing against execute()'s own
// independently-measured peak.
std::uint64_t work_units_for_order(const analysis::EliminationOrder& order) {
    std::uint64_t work = 0;
    for (const std::size_t degree : order.per_step_degree) {
        if (degree >= 62) {
            throw ResourceExceeded("work_units_for_order: degree " + std::to_string(degree) +
                                   " would overflow a checked 2^(d+1) computation");
        }
        const std::uint64_t legs = static_cast<std::uint64_t>(degree) + 1;
        const std::uint64_t step_cost =
            checked_mul(legs, checked_pow2(static_cast<unsigned>(degree + 1)));
        work = checked_add(work, step_cost);
    }
    return work;
}

constexpr std::uint64_t kNoPruneBound = std::numeric_limits<std::uint64_t>::max();

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return a > kNoPruneBound - b ? kNoPruneBound : a + b;
}

// work_units_for_order's per-step cost, (d+1) * 2^(d+1), saturating instead
// of throwing: used only to compare a running total with a prune bound, where
// "too large to represent" simply means "above the bound".
std::uint64_t saturating_step_cost(std::size_t degree) {
    if (degree >= 57) {
        return kNoPruneBound;
    }
    return (static_cast<std::uint64_t>(degree) + 1) * (std::uint64_t{1} << (degree + 1));
}

// A lower bound on work_units_for_order for ANY complete elimination of a
// term's line graph, from its size alone: the degrees at elimination sum to
// at least num_edges (analysis::line_graph_size), and (d+1) * 2^(d+1) is
// convex and increasing in d, so the sum is smallest when those degrees are
// spread as evenly as possible over the num_nodes steps.
std::uint64_t work_lower_bound(const analysis::LineGraphSize& size) {
    if (size.num_nodes == 0) {
        return 0;
    }
    const std::size_t q = size.num_edges / size.num_nodes;
    const std::size_t r = size.num_edges % size.num_nodes;
    const std::uint64_t low = saturating_step_cost(q);
    const std::uint64_t high = saturating_step_cost(q + 1);
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < size.num_nodes; ++i) {
        total = saturating_add(total, i < r ? high : low);
    }
    return total;
}

// The estimate returned when the caller's branch-and-bound bound is passed.
CostEstimate pruned_estimate(const EngineEstimateRequest& request,
                             std::size_t costed_terms,
                             std::uint64_t reached_work_units,
                             bool from_floor) {
    const std::string reached = reached_work_units == kNoPruneBound
                                    ? std::string("more than 2^64")
                                    : std::to_string(reached_work_units);
    const std::string where =
        from_floor
            ? "before any elimination search: a lower bound of " + reached +
                  " work units over its " + std::to_string(request.observable->terms.size()) +
                  " observable terms (from each term's network size) is"
            : "in observable term " + std::to_string(costed_terms + 1) + " of " +
                  std::to_string(request.observable->terms.size()) + ", at " + reached +
                  " work units (including a lower bound for the remaining terms),";
    return CostEstimate{/*feasible=*/false,
                        0,
                        0,
                        std::string(kPrunedReasonPrefix) + ": costing stopped " + where +
                            " already more than the " +
                            std::to_string(request.prune_above_work_units) +
                            " of a cheaper feasible alternative",
                        {}};
}

}  // namespace

std::string_view TensorNetworkEngine::name() const {
    return "tensor_network";
}

Engine TensorNetworkEngine::engine_id() const {
    return Engine::TensorNetwork;
}

Backend TensorNetworkEngine::backend() const {
    return Backend::Cpu;
}

EngineCapabilities TensorNetworkEngine::capabilities() const {
    return EngineCapabilities{/*counts=*/false, /*statevector=*/false, /*expectation=*/true};
}

CostEstimate TensorNetworkEngine::estimate(const EngineEstimateRequest& request) const {
    if (request.want_statevector) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the tensor-network engine never produces a statevector -- "
                            "materializing one at this engine's own target scale defeats the "
                            "reason to use it",
                            {}};
    }
    if (request.want_counts) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the tensor-network engine produces an expectation value only, "
                            "never counts",
                            {}};
    }
    if (!request.want_expectation) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the tensor-network engine has nothing to compute without an "
                            "expectation request",
                            {}};
    }
    // Before the elimination search, whose graph has no node for a channel.
    if (const std::optional<std::string> reason = ir::first_noise_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }

    try {
        std::uint64_t peak_bytes = 0;
        std::uint64_t work = 0;
        std::size_t costed_terms = 0;
        const bool bounded = request.prune_above_work_units != kNoPruneBound;
        // remaining_floor[t]: a lower bound on the work of terms t+1 .. end,
        // from each term's line-graph size (work_lower_bound). The running
        // total plus this floor is what is compared with the bound, so a
        // candidate whose terms cannot all fit is pruned before any min-fill
        // search runs, and otherwise as early within the search as possible.
        const std::size_t num_terms = request.observable->terms.size();
        std::vector<std::uint64_t> remaining_floor(num_terms + 1, 0);
        if (bounded) {
            // Every term's network is the identity term's network plus one
            // observable node on each qubit the term acts on. That node splits
            // one wire in two (one more node) and forms a 2-clique (one more
            // edge); it can also break a pair of parallel wires (more edges
            // still). So a term acting on k qubits has exactly k more nodes and
            // at least k more edges than the identity term, measured once.
            const std::size_t words = (request.num_qubits + 63) / 64;
            const PauliString identity{request.num_qubits,
                                       std::vector<std::uint64_t>(words, 0),
                                       std::vector<std::uint64_t>(words, 0)};
            const analysis::LineGraphSize base =
                analysis::line_graph_size(analysis::build_closed_timelines(
                    request.num_qubits, *request.operations, identity));
            for (std::size_t t = num_terms; t-- > 0;) {
                const PauliString& term = request.observable->terms[t].first;
                std::size_t acted_on = 0;
                for (std::uint32_t q = 0; q < request.num_qubits; ++q) {
                    if (common::get_bit(term.x_words, q) || common::get_bit(term.z_words, q)) {
                        ++acted_on;
                    }
                }
                remaining_floor[t] =
                    saturating_add(remaining_floor[t + 1],
                                   work_lower_bound(analysis::LineGraphSize{
                                       base.num_nodes + acted_on, base.num_edges + acted_on}));
            }
            if (remaining_floor[0] > request.prune_above_work_units) {
                return pruned_estimate(request, 0, remaining_floor[0], /*from_floor=*/true);
            }
        }
        for (const auto& [pauli, coefficient] : request.observable->terms) {
            static_cast<void>(coefficient);  // Applied by the caller (docs/adr/0007).
            const analysis::ClosedTimelines timelines =
                analysis::build_closed_timelines(request.num_qubits, *request.operations, pauli);
            const analysis::LineGraph line_graph = analysis::build_line_graph(timelines);
            // Branch and bound inside the search: each elimination step's
            // cost is known the moment it is taken, so the running total can
            // be compared with the caller's bound step by step instead of only
            // after the whole (expensive) min-fill search for this term.
            const std::uint64_t later_floor = bounded ? remaining_floor[costed_terms + 1] : 0;
            std::uint64_t running = work;
            const analysis::MinFillStopPredicate stop =
                bounded ? analysis::MinFillStopPredicate([&](std::size_t degree) {
                    running = saturating_add(running, saturating_step_cost(degree));
                    return saturating_add(running, later_floor) > request.prune_above_work_units;
                })
                        : analysis::MinFillStopPredicate{};
            const analysis::EliminationOrder order =
                analysis::estimate_treewidth_min_fill(line_graph.graph, request.budget_bytes, stop);
            if (order.stopped_by_caller) {
                return pruned_estimate(
                    request, costed_terms, saturating_add(running, later_floor), false);
            }
            if (order.exceeded_budget) {
                // The search stopped before finishing, so order.width is
                // only a lower bound on what a completed search would
                // report -- worded as "did not fit this budget", never
                // as a proof that no valid plan exists at any budget
                // (analysis/treewidth.h's own EliminationOrder::
                // exceeded_budget doc comment).
                return CostEstimate{
                    /*feasible=*/false,
                    0,
                    0,
                    "the elimination search for one observable term did not finish within "
                    "the job's own memory budget (" +
                        std::to_string(request.budget_bytes) +
                        " bytes) -- this means the search was stopped early, not that no "
                        "valid contraction plan exists for this circuit",
                    {}};
            }

            // width + 1, not width: order.width is the largest degree any
            // wire had at its own elimination, and eliminating a degree-d
            // wire materializes an intermediate over d+1 indices (see
            // work_units_for_order above). Using width alone under-bounds
            // the real peak by a factor of two, which would make this
            // engine claim to fit a budget it does not fit.
            peak_bytes = std::max(peak_bytes, bytes_for_legs(order.width + 1));
            work = checked_add(work, work_units_for_order(order));
            // Every later term only adds work, so once the running total is
            // strictly above the caller's bound this candidate cannot win.
            if (bounded && saturating_add(work, remaining_floor[costed_terms + 1]) >
                               request.prune_above_work_units) {
                return pruned_estimate(request,
                                       costed_terms,
                                       saturating_add(work, remaining_floor[costed_terms + 1]),
                                       false);
            }
            ++costed_terms;
        }
        return CostEstimate{/*feasible=*/true, peak_bytes, work, /*infeasible_reason=*/"", {}};
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

EngineExecutionResult TensorNetworkEngine::execute(const EngineExecutionRequest& request) const {
    std::vector<Complex> terms;
    terms.reserve(request.observable->terms.size());
    std::uint64_t peak_bytes = 0;

    // This engine's ONLY safe boundary is between observable terms (Phase 11,
    // docs/adr/0015). Unlike every other engine it does not sweep operations:
    // each term builds and contracts its own closed network, and there is no
    // point inside a contraction where a partial result means anything. The
    // consequence is stated rather than hidden: a single-term observable is
    // not monitorable mid-run here, so its only boundary is completion.
    std::uint64_t completed = 0;
    for (const auto& [pauli, coefficient] : request.observable->terms) {
        static_cast<void>(coefficient);  // Applied by the caller (docs/adr/0007).
        const analysis::ClosedTimelines timelines =
            analysis::build_closed_timelines(request.num_qubits, request.operations, pauli);
        const analysis::LineGraph line_graph = analysis::build_line_graph(timelines);
        const analysis::EliminationOrder order =
            analysis::estimate_treewidth_min_fill(line_graph.graph, request.budget_bytes);
        common::ensure(!order.exceeded_budget,
                       "TensorNetworkEngine::execute: the elimination search exceeded its own "
                       "budget for a term estimate() already reported feasible -- internal "
                       "invariant violated");
        ContractionNetwork network =
            build_contraction_network(request.num_qubits, request.operations, pauli);
        const ContractionResult contracted = contract(std::move(network), line_graph, order.order);

        terms.push_back(contracted.scalar);
        peak_bytes = std::max(peak_bytes, bytes_for_legs(contracted.actual_peak_legs));
        ++completed;
        if (request.monitor != nullptr) {
            request.monitor->at_safe_boundary(completed);
        }
    }

    EngineExecutionResult result;
    result.expectation_terms = std::move(terms);
    result.actual_bytes = peak_bytes;
    return result;
}

}  // namespace menos::engines::tensor_network
