#include "analysis/treewidth.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
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
    std::map<std::size_t, std::vector<std::size_t>> wires_of_tensor_node;

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
            wires_of_tensor_node[a].push_back(wire_id);
            wires_of_tensor_node[b].push_back(wire_id);
        }
    }

    // Every tensor node's own incident wires become a clique in the line
    // graph (a k-legged tensor mixes all k of its own indices together).
    std::set<std::pair<std::size_t, std::size_t>> unique_edges;
    for (const auto& [tensor_node, wires] : wires_of_tensor_node) {
        for (std::size_t i = 0; i < wires.size(); ++i) {
            for (std::size_t j = i + 1; j < wires.size(); ++j) {
                unique_edges.emplace(std::min(wires[i], wires[j]), std::max(wires[i], wires[j]));
            }
        }
    }

    return LineGraph{TensorNetworkGraph{wire_of.size(),
                                        std::vector<std::pair<std::size_t, std::size_t>>(
                                            unique_edges.begin(), unique_edges.end())},
                     std::move(wire_of)};
}

namespace {

// One std::set<std::size_t> node: the element, plus the links and colour that
// both libstdc++ and the MSVC STL pack into four pointer-sized words.
// One allocation each (common::tree_node_bytes, written out because it is
// needed as a constant).
constexpr std::uint64_t kSetNodeBytes =
    sizeof(std::size_t) + (4 * sizeof(void*)) + common::kHeapBlockOverheadBytes;

// fill(v) = C(deg(v), 2) - |{edges already present among v's neighbors}|
// -- the number of NEW fill-in edges eliminating v right now would add.
// O(deg(v)^2) (a std::set lookup per candidate pair); called once per
// vertex up front, then again only for the small set of vertices whose
// own neighbor set (or the connectivity within it) actually changed,
// never for every remaining vertex at every step (see the "affected"
// set below) -- that per-step full rescan was this function's original
// implementation, and its own O(V) per-step cost, repeated for O(V)
// steps, each recomputing O(V) vertices' O(width^2) fill from scratch
// regardless of whether their neighborhood had changed at all, made a
// single estimate() call take minutes on an ordinary few-hundred-node
// closed network (measured directly: a fixture-sized ~80-node graph
// took over 300ms, extrapolating to unusable on the several-hundred-node
// networks real circuits produce) -- a real, benchmark-confirmed
// bottleneck, not a premature optimization.
std::size_t compute_fill(const std::vector<std::set<std::size_t>>& adjacency, std::size_t v) {
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

EliminationOrder estimate_treewidth_min_fill(const TensorNetworkGraph& graph,
                                             std::uint64_t max_peak_bytes) {
    std::vector<std::set<std::size_t>> adjacency(graph.num_nodes);
    std::uint64_t entries = 0;  // the adjacency sets' total size
    for (const auto& [a, b] : graph.edges) {
        entries += adjacency.at(a).insert(b).second ? 1U : 0U;
        entries += adjacency.at(b).insert(a).second ? 1U : 0U;
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

    // Held throughout: each vertex's set (with its sentinel node), fill
    // count and two result slots, the elimination flags, and the affected
    // set's sentinel. A step adds its neighbor list and affected set.
    // Five arrays (the sets, the fill counts, the flags and the result's two
    // lists) are one allocation each.
    const std::uint64_t fixed =
        common::checked_add(common::checked_mul(graph.num_nodes,
                                                sizeof(std::set<std::size_t>) + kSetNodeBytes +
                                                    (3 * sizeof(std::size_t))),
                            ((graph.num_nodes + 63) / 64 * sizeof(std::uint64_t)) + kSetNodeBytes +
                                (5 * common::kHeapBlockOverheadBytes));
    const auto note = [&](std::uint64_t step_bytes) {
        result.search_peak_bytes = std::max(
            result.search_peak_bytes,
            common::checked_add(common::checked_add(fixed, entries * kSetNodeBytes), step_bytes));
    };
    note(0);

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

        const std::vector<std::size_t> neighbors(adjacency[best_vertex].begin(),
                                                 adjacency[best_vertex].end());
        const std::uint64_t neighbor_bytes = neighbors.size() * sizeof(std::size_t);
        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            for (std::size_t j = i + 1; j < neighbors.size(); ++j) {
                entries += adjacency[neighbors[i]].insert(neighbors[j]).second ? 1U : 0U;
                entries += adjacency[neighbors[j]].insert(neighbors[i]).second ? 1U : 0U;
            }
        }
        note(common::heap_block_bytes(neighbor_bytes));
        for (const std::size_t nb : neighbors) {
            entries -= adjacency[nb].erase(best_vertex);
        }
        entries -= adjacency[best_vertex].size();
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
        note(common::heap_block_bytes(neighbor_bytes) + (affected.size() * kSetNodeBytes));
        for (const std::size_t v : affected) {
            fill[v] = compute_fill(adjacency, v);
        }
    }
    return result;
}

std::uint64_t closed_timelines_bytes(const ClosedTimelines& timelines) {
    std::uint64_t bytes = common::heap_block_bytes(timelines.per_qubit.size() *
                                                   sizeof(std::vector<ClosedTimelineEntry>));
    for (const std::vector<ClosedTimelineEntry>& combined : timelines.per_qubit) {
        bytes = common::checked_add(
            bytes, common::grown_vector_bytes(combined.size(), sizeof(ClosedTimelineEntry)));
    }
    return bytes;
}

std::uint64_t closed_timelines_peak_bytes(const ClosedTimelines& timelines,
                                          std::size_t num_operations) {
    // ket_timelines(): node_index, and per qubit its ket entries, each list
    // grown one entry at a time; then the result beside them.
    std::uint64_t bytes = common::checked_add(
        common::heap_block_bytes(num_operations * sizeof(std::size_t)),
        common::heap_block_bytes(timelines.per_qubit.size() *
                                 sizeof(std::vector<std::pair<std::size_t, std::size_t>>)));
    for (const std::vector<ClosedTimelineEntry>& combined : timelines.per_qubit) {
        std::uint64_t ket = 0;
        for (const ClosedTimelineEntry& entry : combined) {
            ket += entry.kind == ClosedTimelineEntry::Kind::KetGate ? 1U : 0U;
        }
        bytes = common::checked_add(
            bytes, common::grown_vector_bytes(ket, sizeof(std::pair<std::size_t, std::size_t>)));
    }
    return common::checked_add(bytes, closed_timelines_bytes(timelines));
}

std::uint64_t line_graph_bytes(const LineGraph& line_graph) {
    return common::checked_add(
        common::heap_block_bytes(line_graph.graph.edges.size() *
                                 sizeof(std::pair<std::size_t, std::size_t>)),
        common::grown_vector_bytes(line_graph.wire_of.size(), sizeof(WireId)));
}

std::uint64_t line_graph_peak_bytes(const ClosedTimelines& timelines, const LineGraph& line_graph) {
    const std::uint64_t wires = line_graph.wire_of.size();
    const std::uint64_t edges = line_graph.graph.edges.size();
    // wires_of_tensor_node: one map node per tensor node and its wire list
    // (2 entries per wire in total, each list grown one at a time); the
    // unique_edges set; then the result's edge vector and wire_of.
    const std::uint64_t map_node =
        common::tree_node_bytes(sizeof(std::pair<const std::size_t, std::vector<std::size_t>>)) +
        (2 * common::kHeapBlockOverheadBytes);  // and its own wire list, grown
    const std::uint64_t edge_node =
        common::tree_node_bytes(sizeof(std::pair<std::size_t, std::size_t>));
    std::uint64_t bytes = common::checked_mul(timelines.num_nodes + 1, map_node);
    bytes = common::checked_add(bytes, common::grown_vector_bytes(2 * wires, sizeof(std::size_t)));
    bytes = common::checked_add(bytes, common::checked_mul(edges + 1, edge_node));
    return common::checked_add(bytes, line_graph_bytes(line_graph));
}

std::uint64_t elimination_order_bytes(const EliminationOrder& order) {
    return common::checked_add(
        common::heap_block_bytes(common::checked_mul(order.order.capacity(), sizeof(std::size_t))),
        common::heap_block_bytes(
            common::checked_mul(order.per_step_degree.capacity(), sizeof(std::size_t))));
}

}  // namespace menos::analysis
