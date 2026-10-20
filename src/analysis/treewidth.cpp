#include "analysis/treewidth.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <set>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/pauli_row.h"

namespace menos::analysis {
namespace {

// The operand qubits of a GateOp/UnitaryOp, or nullptr for a
// MeasureOp (excluded -- a classical readout, not a tensor). Exhaustive over
// Operation: a new alternative fails the static_assert until it is given its
// own rule (docs/adr/0016 section 7).
const std::vector<QubitId>* operand_qubits(const Operation& op) {
    return std::visit(
        [](const auto& o) -> const std::vector<QubitId>* {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                return &o.qubits;
            } else if constexpr (std::is_same_v<T, ChannelOp>) {
                // A pure-state tensor network has no node for a channel. The
                // only caller, the tensor-network engine, refuses a noisy job in
                // estimate() before building a graph.
                throw common::InternalError(
                    "treewidth: a noise channel reached a pure-state tensor network");
            } else {
                static_assert(std::is_same_v<T, MeasureOp>);
                return nullptr;
            }
        },
        op);
}

// Per qubit, the ordered sequence of (ket-side node id, 0-based operand
// position within that gate) pairs touching it (node_index[i] is
// operations[i]'s own 0-based node id, assigned only to GateOp/UnitaryOp
// entries, consecutively, skipping MeasureOp). The operand position is
// only consumed by build_closed_timelines below; build_tensor_network_graph
// ignores it.
std::vector<std::vector<std::pair<std::size_t, std::size_t>>> ket_timelines(
    std::uint32_t num_qubits,
    std::span<const Operation> operations,
    std::vector<std::size_t>& node_index,
    std::size_t& num_gate_nodes) {
    node_index.assign(operations.size(), 0);
    num_gate_nodes = 0;
    for (std::size_t i = 0; i < operations.size(); ++i) {
        if (operand_qubits(operations[i]) != nullptr) {
            node_index[i] = num_gate_nodes++;
        }
    }

    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> timelines(num_qubits);
    for (std::size_t i = 0; i < operations.size(); ++i) {
        const std::vector<QubitId>* qubits = operand_qubits(operations[i]);
        if (qubits == nullptr) {
            continue;
        }
        for (std::size_t j = 0; j < qubits->size(); ++j) {
            timelines.at(qubits->at(j)).emplace_back(node_index[i], j);
        }
    }
    return timelines;
}

}  // namespace

TensorNetworkGraph build_tensor_network_graph(std::uint32_t num_qubits,
                                              std::span<const Operation> operations) {
    std::vector<std::size_t> node_index;
    std::size_t num_nodes = 0;
    const std::vector<std::vector<std::pair<std::size_t, std::size_t>>> timelines =
        ket_timelines(num_qubits, operations, node_index, num_nodes);

    std::set<std::pair<std::size_t, std::size_t>> unique_edges;
    for (const auto& timeline : timelines) {
        for (std::size_t i = 0; i + 1 < timeline.size(); ++i) {
            const std::size_t a = std::min(timeline[i].first, timeline[i + 1].first);
            const std::size_t b = std::max(timeline[i].first, timeline[i + 1].first);
            if (a != b) {
                unique_edges.emplace(a, b);
            }
        }
    }

    return TensorNetworkGraph{
        num_nodes,
        std::vector<std::pair<std::size_t, std::size_t>>(unique_edges.begin(), unique_edges.end())};
}

ClosedTimelines build_closed_timelines(std::uint32_t num_qubits,
                                       std::span<const Operation> operations,
                                       const PauliString& term) {
    std::vector<std::size_t> node_index;
    std::size_t num_gate_nodes = 0;
    const std::vector<std::vector<std::pair<std::size_t, std::size_t>>> ket =
        ket_timelines(num_qubits, operations, node_index, num_gate_nodes);

    // ket nodes: [0, num_gate_nodes). bra nodes (mirror, same gates):
    // [num_gate_nodes, 2*num_gate_nodes). Boundary and Pauli-operator
    // nodes are allocated on demand, starting right after the bra
    // range.
    const std::size_t bra_offset = num_gate_nodes;
    std::size_t next_free_node = 2 * num_gate_nodes;

    using Kind = ClosedTimelineEntry::Kind;
    ClosedTimelines result;
    result.per_qubit.resize(num_qubits);

    for (QubitId q = 0; q < num_qubits; ++q) {
        // Combined per-qubit timeline: a |0> boundary, ket nodes in
        // circuit order, then (if the term is non-identity here) a
        // Pauli-operator node, then the mirrored bra nodes in REVERSE
        // circuit order (U^dagger runs the same gates backward), then a
        // <0| boundary -- consecutive entries in this combined sequence
        // share a wire. Without the two boundary nodes the network
        // would not actually close to a scalar (the first ket gate's
        // and last bra gate's otherwise-dangling legs) -- they are
        // degree-1 pendants, which provably never change a min-fill
        // width (verified by hand for this exact construction in
        // tests/analysis/treewidth_test.cpp), so omitting them would
        // not have changed any width this ADR already reported, but a
        // real contraction still needs them.
        std::vector<ClosedTimelineEntry>& combined = result.per_qubit[q];
        combined.push_back({next_free_node++, 0, Kind::BoundaryKet});
        for (const auto& [node_id, operand_index] : ket[q]) {
            combined.push_back({node_id, operand_index, Kind::KetGate});
        }
        if (common::get_bit(term.x_words, q) || common::get_bit(term.z_words, q)) {
            combined.push_back({next_free_node++, 0, Kind::Observable});
        }
        for (auto it = ket[q].rbegin(); it != ket[q].rend(); ++it) {
            combined.push_back({it->first + bra_offset, it->second, Kind::BraGate});
        }
        combined.push_back({next_free_node++, 0, Kind::BoundaryBra});
    }

    result.num_nodes = next_free_node;
    return result;
}

TensorNetworkGraph build_closed_contraction_graph(std::uint32_t num_qubits,
                                                  std::span<const Operation> operations,
                                                  const PauliString& term) {
    const ClosedTimelines timelines = build_closed_timelines(num_qubits, operations, term);

    std::set<std::pair<std::size_t, std::size_t>> unique_edges;
    for (const std::vector<ClosedTimelineEntry>& combined : timelines.per_qubit) {
        for (std::size_t i = 0; i + 1 < combined.size(); ++i) {
            const std::size_t a = std::min(combined[i].node_id, combined[i + 1].node_id);
            const std::size_t b = std::max(combined[i].node_id, combined[i + 1].node_id);
            if (a != b) {
                unique_edges.emplace(a, b);
            }
        }
    }

    return TensorNetworkGraph{
        timelines.num_nodes,
        std::vector<std::pair<std::size_t, std::size_t>>(unique_edges.begin(), unique_edges.end())};
}

LineGraph build_line_graph(const ClosedTimelines& timelines) {
    // Every wire, assigned a line-graph node id the first time it is
    // encountered (each (node_id_a, node_id_b, qubit) triple is produced
    // by exactly one qubit's own timeline walk, so no id is ever
    // assigned twice), plus which original tensor node ids incident to
    // it share this wire -- needed below to form each tensor's own
    // clique among its incident wires.
    std::vector<WireId> wire_of;
    // Indexed by tensor node id (every id is < timelines.num_nodes). A vector
    // rather than the original std::map: same contents, visited in the same
    // ascending node order, without a tree allocation per node.
    std::vector<std::vector<std::size_t>> wires_of_tensor_node(timelines.num_nodes);

    for (std::size_t q = 0; q < timelines.per_qubit.size(); ++q) {
        const std::vector<ClosedTimelineEntry>& combined = timelines.per_qubit[q];
        for (std::size_t i = 0; i + 1 < combined.size(); ++i) {
            const std::size_t a = combined[i].node_id;
            const std::size_t b = combined[i + 1].node_id;
            if (a == b) {
                continue;
            }
            const std::size_t wire_id = wire_of.size();
            wire_of.emplace_back(std::min(a, b), std::max(a, b), static_cast<std::uint32_t>(q));
            wires_of_tensor_node.at(a).push_back(wire_id);
            wires_of_tensor_node.at(b).push_back(wire_id);
        }
    }

    // Every tensor node's own incident wires become a clique in the line
    // graph (a k-legged tensor mixes all k of its own indices together).
    // Collected, then sorted and deduplicated: the same ascending, unique
    // edge list the original std::set produced.
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    for (const std::vector<std::size_t>& wires : wires_of_tensor_node) {
        for (std::size_t i = 0; i < wires.size(); ++i) {
            for (std::size_t j = i + 1; j < wires.size(); ++j) {
                edges.emplace_back(std::min(wires[i], wires[j]), std::max(wires[i], wires[j]));
            }
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    return LineGraph{TensorNetworkGraph{wire_of.size(), std::move(edges)}, std::move(wire_of)};
}

namespace {

// Dense bitset adjacency for the min-fill search. The graph the search
// runs on (the closed network's line graph) is small -- hundreds to a few
// thousand nodes -- but DENSE once fill-in edges accumulate, which made the
// previous std::set<std::size_t>-per-vertex representation the planner's
// dominant cost (measured: ~96% of menos.run on a 10-qubit, 55-term QAOA
// estimate, almost all of it red-black-tree traversal). A row of 64-bit words
// per vertex turns "how many edges are there among v's neighbours" into
// popcounts of word-wise ANDs, and every neighbour set is still walked in
// ascending vertex order, so the elimination order, its tie-breaking (lowest
// index) and every reported width are unchanged -- only the data structure is.
class BitAdjacency {
public:
    explicit BitAdjacency(std::size_t n) : n_(n), words_((n + 63) / 64), bits_(n * words_, 0) {}

    void set(std::size_t a, std::size_t b) {
        row(a)[b / 64] |= std::uint64_t{1} << (b % 64);
    }
    void reset(std::size_t a, std::size_t b) {
        row(a)[b / 64] &= ~(std::uint64_t{1} << (b % 64));
    }
    void clear_row(std::size_t a) { std::fill_n(row(a), words_, std::uint64_t{0}); }

    [[nodiscard]] std::size_t degree(std::size_t a) const {
        std::size_t d = 0;
        const std::uint64_t* r = row(a);
        for (std::size_t w = 0; w < words_; ++w) {
            d += static_cast<std::size_t>(std::popcount(r[w]));
        }
        return d;
    }
    [[nodiscard]] std::size_t common(std::size_t a, std::size_t b) const {
        std::size_t c = 0;
        const std::uint64_t* ra = row(a);
        const std::uint64_t* rb = row(b);
        for (std::size_t w = 0; w < words_; ++w) {
            c += static_cast<std::size_t>(std::popcount(ra[w] & rb[w]));
        }
        return c;
    }
    // Calls f(u) for every neighbour u of a, in ascending order of u.
    template <typename F>
    void for_each_neighbor(std::size_t a, F&& f) const {
        const std::uint64_t* r = row(a);
        for (std::size_t w = 0; w < words_; ++w) {
            std::uint64_t word = r[w];
            while (word != 0) {
                const auto bit = static_cast<std::size_t>(std::countr_zero(word));
                f(w * 64 + bit);
                word &= word - 1;
            }
        }
    }
    [[nodiscard]] std::size_t num_nodes() const { return n_; }

private:
    std::uint64_t* row(std::size_t a) { return bits_.data() + a * words_; }
    [[nodiscard]] const std::uint64_t* row(std::size_t a) const { return bits_.data() + a * words_; }

    std::size_t n_;
    std::size_t words_;
    std::vector<std::uint64_t> bits_;
};

// fill(v) = C(deg(v), 2) - |{edges already present among v's neighbors}|
// -- the number of NEW fill-in edges eliminating v right now would add.
// Each edge {a, b} among v's neighbours is counted once from a and once from
// b by sum_{a in N(v)} |N(a) & N(v)|, hence the division by two. Recomputed
// only for vertices whose neighbourhood may have changed (see "affected"
// below), never for every remaining vertex at every step.
//
// On x86-64 Linux the function is compiled twice, for the baseline ISA and
// for CPUs with the POPCNT instruction (every x86-64 CPU since about 2008),
// and the loader picks the right one at startup: the baseline build of the
// popcount above is a software routine several times slower. The result is
// the same integer either way.
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
[[gnu::target_clones("popcnt", "default")]]
#endif
std::size_t compute_fill(const BitAdjacency& adjacency, std::size_t v) {
    std::size_t k = 0;
    std::size_t twice_edges = 0;
    adjacency.for_each_neighbor(v, [&](std::size_t a) {
        ++k;
        twice_edges += adjacency.common(a, v);
    });
    return k * (k - 1) / 2 - twice_edges / 2;
}

// The original std::set-based fill count, kept as the fallback for graphs too
// large for a dense bitset (see estimate_treewidth_min_fill) and as the
// oracle the dense search is tested against.
std::size_t compute_fill_sparse(const std::vector<std::set<std::size_t>>& adjacency, std::size_t v) {
    const std::set<std::size_t>& neighbors = adjacency[v];
    std::size_t edges_among_neighbors = 0;
    for (const std::size_t a : neighbors) {
        for (const std::size_t b : neighbors) {
            if (a < b && adjacency[a].contains(b)) {
                ++edges_among_neighbors;
            }
        }
    }
    const std::size_t k = neighbors.size();
    return k * (k - 1) / 2 - edges_among_neighbors;
}

// True iff a degree-d node's own tensor (16 * 2^d bytes) is already
// larger than max_peak_bytes, including the case where 16 * 2^d would
// not even fit in a std::uint64_t -- an overflow there means the true
// byte count is larger still, so it exceeds max_peak_bytes regardless of
// what finite value that budget happens to be.
// degree + 1, matching the peak-byte figure the tensor-network engine's own
// estimate() reports: eliminating a degree-d wire materializes an
// intermediate over d+1 indices (the d neighbours plus the wire being summed
// away), so the object is 16 * 2^(d+1) bytes. Bounding the search on
// 16 * 2^d instead would let the search keep going past a degree whose real
// intermediate already exceeds the budget -- the two must agree, or the
// early exit stops at a different place than the cost model would.
bool degree_exceeds_budget(std::size_t degree, std::uint64_t max_peak_bytes) {
    if (degree >= 62) {
        return true;
    }
    try {
        return common::checked_mul(std::uint64_t{16},
                                   common::checked_pow2(static_cast<unsigned>(degree + 1))) >
               max_peak_bytes;
    } catch (const common::ResourceExceeded&) {
        return true;
    }
}

}  // namespace

EliminationOrder estimate_treewidth_min_fill_dense(const TensorNetworkGraph& graph,
                                                   std::uint64_t max_peak_bytes,
    const MinFillStopPredicate& stop_after_step) {
    BitAdjacency adjacency(graph.num_nodes);
    for (const auto& [a, b] : graph.edges) {
        common::ensure(a < graph.num_nodes && b < graph.num_nodes,
                       "estimate_treewidth_min_fill: edge endpoint out of range");
        adjacency.set(a, b);
        adjacency.set(b, a);
    }
    std::vector<bool> eliminated(graph.num_nodes, false);

    std::vector<std::size_t> fill(graph.num_nodes);
    for (std::size_t v = 0; v < graph.num_nodes; ++v) {
        fill[v] = compute_fill(adjacency, v);
    }

    EliminationOrder result;
    result.order.reserve(graph.num_nodes);
    result.per_step_degree.reserve(graph.num_nodes);
    result.width = 0;

    std::vector<std::size_t> neighbors;
    std::vector<bool> is_affected(graph.num_nodes, false);
    std::vector<std::size_t> affected;

    for (std::size_t step = 0; step < graph.num_nodes; ++step) {
        std::size_t best_vertex = 0;
        std::size_t best_fill = std::numeric_limits<std::size_t>::max();
        bool found = false;

        for (std::size_t v = 0; v < graph.num_nodes; ++v) {
            if (eliminated[v]) {
                continue;
            }
            // Strict less-than: iterating v in increasing order and only
            // replacing best_vertex on a strictly smaller fill count
            // means ties resolve to the lowest vertex index,
            // deterministically (never hash order).
            if (!found || fill[v] < best_fill) {
                best_fill = fill[v];
                best_vertex = v;
                found = true;
            }
        }

        common::ensure(found, "estimate_treewidth_min_fill: no remaining vertex found");
        neighbors.clear();
        adjacency.for_each_neighbor(best_vertex, [&](std::size_t u) { neighbors.push_back(u); });
        const std::size_t degree = neighbors.size();
        if (degree_exceeds_budget(degree, max_peak_bytes)) {
            // width is defined as the max degree over every step
            // (including this one, had it been recorded), so the final
            // width the completed search would report can only be >=
            // degree -- already enough on its own to exceed max_peak_bytes,
            // regardless of what any later step's own degree turns out to
            // be. Finishing the search cannot change this conclusion, only
            // spend more time reaching it.
            result.exceeded_budget = true;
            break;
        }
        result.width = std::max(result.width, degree);
        result.order.push_back(best_vertex);
        result.per_step_degree.push_back(degree);
        if (stop_after_step && stop_after_step(degree)) {
            result.stopped_by_caller = true;
            break;
        }

        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            for (std::size_t j = i + 1; j < neighbors.size(); ++j) {
                adjacency.set(neighbors[i], neighbors[j]);
                adjacency.set(neighbors[j], neighbors[i]);
            }
        }
        for (const std::size_t nb : neighbors) {
            adjacency.reset(nb, best_vertex);
        }
        adjacency.clear_row(best_vertex);
        eliminated[best_vertex] = true;

        // A new fill-in edge is always between two members of
        // `neighbors`, so only a vertex with two or more of them as its
        // own neighbors could have its own fill count change. Every
        // neighbor of a member of `neighbors` is a safe (if slightly
        // generous) superset of that set -- cheap to build, and correct
        // regardless of how generous, since compute_fill is idempotent.
        affected.clear();
        const auto mark = [&](std::size_t u) {
            if (!is_affected[u]) {
                is_affected[u] = true;
                affected.push_back(u);
            }
        };
        for (const std::size_t nb : neighbors) {
            mark(nb);
            adjacency.for_each_neighbor(nb, mark);
        }
        for (const std::size_t v : affected) {
            fill[v] = compute_fill(adjacency, v);
            is_affected[v] = false;
        }
    }
    return result;
}

EliminationOrder estimate_treewidth_min_fill_sparse(const TensorNetworkGraph& graph,
                                                    std::uint64_t max_peak_bytes,
    const MinFillStopPredicate& stop_after_step) {
    std::vector<std::set<std::size_t>> adjacency(graph.num_nodes);
    for (const auto& [a, b] : graph.edges) {
        adjacency.at(a).insert(b);
        adjacency.at(b).insert(a);
    }
    std::vector<bool> eliminated(graph.num_nodes, false);

    std::vector<std::size_t> fill(graph.num_nodes);
    for (std::size_t v = 0; v < graph.num_nodes; ++v) {
        fill[v] = compute_fill_sparse(adjacency, v);
    }

    EliminationOrder result;
    result.order.reserve(graph.num_nodes);
    result.per_step_degree.reserve(graph.num_nodes);
    result.width = 0;

    for (std::size_t step = 0; step < graph.num_nodes; ++step) {
        std::size_t best_vertex = 0;
        std::size_t best_fill = std::numeric_limits<std::size_t>::max();
        bool found = false;

        for (std::size_t v = 0; v < graph.num_nodes; ++v) {
            if (eliminated[v]) {
                continue;
            }
            // Strict less-than: iterating v in increasing order and only
            // replacing best_vertex on a strictly smaller fill count
            // means ties resolve to the lowest vertex index,
            // deterministically (never hash order).
            if (!found || fill[v] < best_fill) {
                best_fill = fill[v];
                best_vertex = v;
                found = true;
            }
        }

        common::ensure(found, "estimate_treewidth_min_fill: no remaining vertex found");
        const std::size_t degree = adjacency[best_vertex].size();
        if (degree_exceeds_budget(degree, max_peak_bytes)) {
            // width is defined as the max degree over every step
            // (including this one, had it been recorded), so the final
            // width the completed search would report can only be >=
            // degree -- already enough on its own to exceed max_peak_bytes,
            // regardless of what any later step's own degree turns out to
            // be. Finishing the search cannot change this conclusion, only
            // spend more time reaching it.
            result.exceeded_budget = true;
            break;
        }
        result.width = std::max(result.width, degree);
        result.order.push_back(best_vertex);
        result.per_step_degree.push_back(degree);
        if (stop_after_step && stop_after_step(degree)) {
            result.stopped_by_caller = true;
            break;
        }

        const std::vector<std::size_t> neighbors(adjacency[best_vertex].begin(),
                                                 adjacency[best_vertex].end());
        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            for (std::size_t j = i + 1; j < neighbors.size(); ++j) {
                adjacency[neighbors[i]].insert(neighbors[j]);
                adjacency[neighbors[j]].insert(neighbors[i]);
            }
        }
        for (const std::size_t nb : neighbors) {
            adjacency[nb].erase(best_vertex);
        }
        adjacency[best_vertex].clear();
        eliminated[best_vertex] = true;

        // A new fill-in edge is always between two members of
        // `neighbors`, so only a vertex with two or more of them as its
        // own neighbors could have its own fill count change. Every
        // neighbor of a member of `neighbors` is a safe (if slightly
        // generous) superset of that set -- cheap to build, and correct
        // regardless of how generous, since compute_fill is idempotent.
        std::set<std::size_t> affected(neighbors.begin(), neighbors.end());
        for (const std::size_t nb : neighbors) {
            for (const std::size_t nb2 : adjacency[nb]) {
                affected.insert(nb2);
            }
        }
        for (const std::size_t v : affected) {
            fill[v] = compute_fill_sparse(adjacency, v);
        }
    }
    return result;
}

// Above this many nodes the dense bitset (n * ceil(n / 64) words) would
// exceed 64 MiB, so the search falls back to the original sparse
// representation, whose memory is proportional to the edge count. Both give
// the identical EliminationOrder; only time and memory differ.
constexpr std::size_t kDenseMinFillMaxNodes = 23000;

EliminationOrder estimate_treewidth_min_fill(const TensorNetworkGraph& graph,
                                             std::uint64_t max_peak_bytes,
    const MinFillStopPredicate& stop_after_step) {
    return graph.num_nodes <= kDenseMinFillMaxNodes
               ? estimate_treewidth_min_fill_dense(graph, max_peak_bytes, stop_after_step)
               : estimate_treewidth_min_fill_sparse(graph, max_peak_bytes, stop_after_step);
}

}  // namespace menos::analysis
