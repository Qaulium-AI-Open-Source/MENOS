#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "menos/circuit.h"
#include "menos/observable.h"

namespace menos::analysis {

// The tensor-network graph a real contraction would need to eliminate
// (docs/BUILD_PLAN.md Phase 8, productionized from the "Phase 8
// precondition, step 2" analysis tool, docs/adr/0010): one node per
// gate operation (GateOp or UnitaryOp; MeasureOp is a classical
// readout, not a tensor, and is excluded), one undirected edge per
// wire connecting two temporally consecutive operations on the same
// qubit. Distinct from analysis::build_interaction_graph (Phase 3),
// which has one node per *qubit* and answers a different question (can
// the circuit be split into independent regions).
struct TensorNetworkGraph {
    std::size_t num_nodes;
    std::vector<std::pair<std::size_t, std::size_t>> edges;  // deduplicated, a < b
};

// The one-sided (ket-only) graph: matches EngineEstimateRequest's own
// shape (num_qubits, operations), the same convention
// compute_schmidt_rank_bound already uses, since callers here only ever
// have an EngineEstimateRequest/EngineExecutionRequest, never a full
// Circuit. Kept as the building block build_closed_contraction_graph
// reuses, and for the treewidth-exactness test suite's own abstract-
// graph cases.
[[nodiscard]] TensorNetworkGraph build_tensor_network_graph(std::uint32_t num_qubits,
                                                            std::span<const Operation> operations);

// The network an actual expectation-value contraction needs to close to
// a scalar: <0|U^dagger P U|0>. Per qubit: a |0> boundary node, ket
// nodes (one-sided graph above), one Pauli-operator node per qubit
// `term` acts non-trivially on (an identity insertion is a mathematical
// no-op, so a qubit the term is identity on connects its ket and bra
// ends directly, with no node added there), mirrored bra nodes
// (U^dagger, structurally a reversed-order mirror of the ket side --
// topologically identical, gate-for-gate), and a <0| boundary node. The
// two boundary nodes are what actually close the network (without them
// the first ket gate's and last bra gate's own legs would dangle); they
// are degree-1 pendants, which provably never change a min-fill width,
// so they do not affect any width this module has ever reported. This
// is NOT the same graph precondition step 2 used
// for its own general "is this circuit hard" estimate: verified this
// session, on the actual n=30/depth=10 boundary circuit, that the
// closed network's own min-fill width is materially larger than the
// one-sided estimate (roughly +6, not +0), so a real cost estimate for
// this engine must use this construction, not the one-sided one alone.
[[nodiscard]] TensorNetworkGraph build_closed_contraction_graph(
    std::uint32_t num_qubits, std::span<const Operation> operations, const PauliString& term);

// A single entry of one qubit's own combined boundary+ket+observable+bra
// timeline in the closed contraction network (see
// build_closed_contraction_graph above): identifies one network node,
// and -- for a ket/bra gate node specifically -- which 0-based operand
// position of that gate this qubit is (needed by the tensor-network
// engine's own tensor construction, src/engines/tensor_network/
// contraction.cpp, to route this leg to the correct bit of that gate's
// matrix; a boundary or observable node has exactly one operand, always
// at index 0).
struct ClosedTimelineEntry {
    enum class Kind { BoundaryKet, KetGate, Observable, BraGate, BoundaryBra };
    std::size_t node_id;
    std::size_t operand_index;
    Kind kind;
};

// Every qubit's own combined timeline (a |0> boundary, ket gates in
// circuit order, an observable node if `term` is non-trivial here, the
// same gates mirrored in reverse circuit order, a <0| boundary), plus the
// total node count -- the single source of truth for the closed
// network's node-numbering scheme (ket nodes: [0, num_gate_nodes); bra
// nodes: [num_gate_nodes, 2*num_gate_nodes); boundary/observable nodes
// allocated on demand after that, per qubit in ascending order, in the
// order a qubit's own combined timeline is built below). Shared (not
// file-local) because build_closed_contraction_graph below and the
// tensor-network engine's own closed-network tensor construction both
// need the *exact* same node ids for the *exact* same network nodes --
// computing this twice, independently, would risk the two silently
// drifting apart, corrupting an elimination order computed against one
// numbering into nonsense when applied to tensors built under the other.
struct ClosedTimelines {
    std::vector<std::vector<ClosedTimelineEntry>> per_qubit;
    std::size_t num_nodes;
};

[[nodiscard]] ClosedTimelines build_closed_timelines(std::uint32_t num_qubits,
                                                     std::span<const Operation> operations,
                                                     const PauliString& term);

// One wire (one qubit's own edge between two temporally adjacent entries
// in its own combined timeline): (lower node id, higher node id, owning
// qubit). The owning qubit is necessary, not merely convenient: two
// consecutive 2-qubit gates on the same pair of qubits (e.g. CX(0,1)
// immediately followed by another CX(0,1)) connect the *same pair of
// network nodes* by *two distinct wires*, one per qubit -- a node-id
// pair alone is not a unique wire identity.
using WireId = std::tuple<std::size_t, std::size_t, std::uint32_t>;

// The closed contraction network's own LINE GRAPH: one node per WIRE
// (not per tensor -- see TensorNetworkGraph's own doc comment above,
// which this supersedes for contraction-cost purposes), one edge between
// two wires iff they are both incident to the same tensor node (so a
// k-legged tensor contributes a size-k clique among its own wires).
// verified via two independent published sources this session (not
// assumed): contraction complexity is bounded by the treewidth of THIS
// graph, not of the tensor-node graph directly (Markov & Shi, SIAM J.
// Computing 2008, "optimal contraction sequences correspond to optimal
// tree decompositions of a tensor network's line graph"; the equivalent
// classical algorithm is bucket elimination over indices/wires, one at a
// time, e.g. Dechter 1999) -- an earlier attempt at this phase applied
// min-fill directly to the tensor-node graph and found the resulting
// contraction did not terminate in reasonable time on a realistic
// circuit, confirming empirically that graph is the wrong one.
// estimate_treewidth_min_fill itself is unchanged and reused unmodified
// here; only which graph it is given is different.
//
// wire_of[i] is line-graph node i's own original WireId -- needed by the
// contraction engine to find which tensors currently hold that wire's
// own two endpoints (analysis never includes engines, so this module
// cannot look that up itself; it only hands back the identity).
struct LineGraph {
    TensorNetworkGraph graph;
    std::vector<WireId> wire_of;
};

[[nodiscard]] LineGraph build_line_graph(const ClosedTimelines& timelines);

struct EliminationOrder {
    // The validated elimination sequence itself (node ids, in the order
    // they were eliminated) -- needed by the engine's execute() to
    // actually perform the pairwise contraction in this exact order,
    // not only by estimate() for the resulting width.
    std::vector<std::size_t> order;
    // The heuristic's own width: the maximum degree any vertex had at
    // the moment of its own elimination. This is a heuristic UPPER
    // BOUND on the graph's true treewidth, never an exact value or a
    // lower bound (CLAUDE.md's own domain invariant) -- a large result
    // is suggestive, not conclusive; a better ordering may exist that
    // this greedy heuristic did not find.
    std::size_t width;
    // per_step_degree[i] is order[i]'s own degree at the moment of its
    // elimination (so width == *std::max_element(per_step_degree)) --
    // exposed because the tensor-network engine's own estimate()
    // (src/engines/tensor_network/tensor_network_engine.cpp) needs each
    // step's own cost, not only the overall width: a standard tree-
    // decomposition contraction bounds the cost of eliminating a
    // degree-d node by O(d * 2^d) scalar multiply-adds (at most d
    // pairwise merges, each over at most d shared/free legs -- Markov &
    // Shi 2008's own complexity argument, already cited above), and
    // work_units sums that bound over every step. Recomputing this from
    // `order` and `graph` alone would mean re-running this same
    // bookkeeping loop a second time; exposing it here is free (already
    // computed, just not returned before this field existed).
    std::vector<std::size_t> per_step_degree;
    // False (the ordinary case) iff every node was eliminated: order.size()
    // == graph.num_nodes. True iff the search stopped early because
    // estimate_treewidth_min_fill's own max_peak_bytes argument was given
    // and the running width already implied a peak tensor larger than it
    // -- order/per_step_degree are then a PREFIX, not a full ordering, and
    // width is a lower bound on what completing the search would have
    // found, not the heuristic's own final answer. This is not a
    // strengthened conclusion over the un-bounded case (a completed
    // search's own width was already only a heuristic upper bound, never
    // a proof, per this struct's own width doc comment above) -- it only
    // means the search was stopped before even finishing that heuristic,
    // so a caller must report this as "did not fit within the given
    // budget", never as "proven infeasible regardless of budget".
    bool exceeded_budget = false;
};


// Greedy minimum-fill-in elimination ordering (a standard treewidth
// heuristic, e.g. Bodlaender & Koster's treewidth-heuristics survey;
// also the classic choice in the quantum-circuit-treewidth literature,
// Markov & Shi 2008): at each step, eliminates the remaining vertex
// whose removal adds the fewest new fill-in edges among its neighbors
// (ties broken by lowest vertex index, a fixed rule -- never iteration
// or hash order, CLAUDE.md's determinism domain invariant, since
// floating-point addition is not associative and an unpinned
// contraction order could change a result's last few bits between
// otherwise-identical runs), connecting its remaining neighbors to each
// other before removing it.
//
// max_peak_bytes bounds the search itself, not only its answer: the
// search stops as soon as the running width already implies a peak
// tensor (16 * 2^width bytes) larger than max_peak_bytes, returning
// EliminationOrder::exceeded_budget = true with whatever prefix it had
// completed. Left at its default (unbounded), the search always runs to
// completion, exactly as before this parameter existed. Bounding it is
// not a numerical-tolerance guess (CLAUDE.md's "never invent... numerical
// tolerances" prime directive): a graph whose running width already
// implies more bytes than max_peak_bytes can only be reported as
// infeasible against that specific budget regardless of how the search
// finishes, since a candidate's own peak-byte estimate can only grow
// (never shrink) as elimination proceeds -- so stopping early changes
// nothing about the conclusion, only how long it takes to reach it. This
// exists because the naive complete search's own cost is a function of
// the graph's structure, not bounded by the caller's own budget the way
// every other engine's estimate() already is (docs/BUILD_PLAN.md Phase 8:
// an ordinary, low-treewidth-unfriendly circuit's closed network made an
// unbounded search take minutes, confirmed by direct measurement, not
// merely suspected).
[[nodiscard]] EliminationOrder estimate_treewidth_min_fill(
    const TensorNetworkGraph& graph,
    std::uint64_t max_peak_bytes = std::numeric_limits<std::uint64_t>::max());

// The two implementations estimate_treewidth_min_fill dispatches between
// (dense bitset adjacency for graphs up to a size bound, the original
// std::set adjacency above it). They return identical results; exposed so
// tests can check that equivalence directly.
[[nodiscard]] EliminationOrder estimate_treewidth_min_fill_dense(
    const TensorNetworkGraph& graph,
    std::uint64_t max_peak_bytes = std::numeric_limits<std::uint64_t>::max());
[[nodiscard]] EliminationOrder estimate_treewidth_min_fill_sparse(
    const TensorNetworkGraph& graph,
    std::uint64_t max_peak_bytes = std::numeric_limits<std::uint64_t>::max());

}  // namespace menos::analysis
