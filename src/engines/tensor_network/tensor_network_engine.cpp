#include "engines/tensor_network/tensor_network_engine.h"

#include <algorithm>
#include <complex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "analysis/treewidth.h"
#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
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

std::uint64_t term_peak_bytes(const EngineEstimateRequest& request,
                              const PauliString& pauli,
                              const analysis::ClosedTimelines& timelines,
                              const analysis::LineGraph& line_graph,
                              const analysis::EliminationOrder& order) {
    const std::uint64_t timelines_bytes = analysis::closed_timelines_bytes(timelines);
    const std::uint64_t line_graph_bytes = analysis::line_graph_bytes(line_graph);
    const std::uint64_t analysis_bytes = checked_add(checked_add(timelines_bytes, line_graph_bytes),
                                                     analysis::elimination_order_bytes(order));
    const NetworkShape shape = network_shape(request.num_qubits, *request.operations, pauli);
    return std::max(
        {analysis::closed_timelines_peak_bytes(timelines, request.operations->size()),
         checked_add(timelines_bytes, analysis::line_graph_peak_bytes(timelines, line_graph)),
         checked_add(checked_add(timelines_bytes, line_graph_bytes), order.search_peak_bytes),
         checked_add(analysis_bytes, shape.build_peak_bytes),
         checked_add(analysis_bytes, contraction_peak_bytes(shape, line_graph, order.order))});
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
        for (const auto& [pauli, coefficient] : request.observable->terms) {
            static_cast<void>(coefficient);  // Applied by the caller (docs/adr/0007).
            const analysis::ClosedTimelines timelines =
                analysis::build_closed_timelines(request.num_qubits, *request.operations, pauli);
            const analysis::LineGraph line_graph = analysis::build_line_graph(timelines);
            const analysis::EliminationOrder order =
                analysis::estimate_treewidth_min_fill(line_graph.graph, request.budget_bytes);
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

            // Everything execute() holds for this term, at its largest
            // (Phase 21, docs/adr/0025 section 3.4), in execute()'s order:
            // the timelines while they are built; the line graph built
            // beside them; the elimination search beside both; then, with
            // all three held, the network build and the contraction, each
            // replayed on leg lists so every intermediate's size is exact.
            // Until Phase 21 this was one intermediate, 16 * 2^(width + 1).
            peak_bytes =
                std::max(peak_bytes, term_peak_bytes(request, pauli, timelines, line_graph, order));
            work = checked_add(work, work_units_for_order(order));
        }
        // One complex per term, held across every term.
        peak_bytes = checked_add(
            peak_bytes,
            common::heap_block_bytes(checked_mul(
                static_cast<std::uint64_t>(request.observable->terms.size()), sizeof(Complex))));
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
