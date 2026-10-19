#include "engines/tensor_network/contraction.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <numeric>
#include <type_traits>
#include <variant>

#include "analysis/treewidth.h"
#include "common/error.h"
#include "common/pauli_row.h"
#include "ir/gate_table.h"

namespace menos::engines::tensor_network {
namespace {

using Complex = std::complex<double>;
using analysis::ClosedTimelineEntry;
using Kind = ClosedTimelineEntry::Kind;

// This node's own operand-indexed leg assignment, accumulated across
// every qubit timeline that touches it (a k-operand gate is touched by
// exactly k of the num_qubits per-qubit timelines, one per operand).
// left/right follow build_contraction_network's own positional
// definition: left = this node's neighbor earlier in its own qubit's
// combined timeline (toward the |0> boundary's own position in that
// array), right = later (toward the <0| boundary's position) -- see
// contraction.h's own doc comment on ContractionNetwork's tensor
// convention for the full derivation.
struct LegAssignment {
    std::vector<LegId> left;
    std::vector<LegId> right;
};

// Exhaustive over Operation, like operation_matrix below: a new alternative
// fails the static_assert until it is given its own rule (docs/adr/0016
// section 7).
const std::vector<QubitId>* operand_qubits(const Operation& op) {
    return std::visit(
        [](const auto& o) -> const std::vector<QubitId>* {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                return &o.qubits;
            } else if constexpr (std::is_same_v<T, ChannelOp>) {
                // Unreachable: TensorNetworkEngine::estimate refuses a noisy job.
                throw common::InternalError(
                    "contraction: a noise channel reached a pure-state tensor network");
            } else {
                static_assert(std::is_same_v<T, MeasureOp>);
                return nullptr;
            }
        },
        op);
}

std::vector<Complex> operation_matrix(const Operation& op) {
    return std::visit(
        [](const auto& o) -> std::vector<Complex> {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                return ir::generate_matrix(o.opcode, o.params);
            } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                return o.matrix;
            } else {
                static_assert(std::is_same_v<T, MeasureOp> || std::is_same_v<T, ChannelOp>);
                throw common::InternalError(
                    "operation_matrix: operation has no unitary matrix (MeasureOp or ChannelOp)");
            }
        },
        op);
}

// Builds a rank-2k tensor from a 2^k x 2^k row-major matrix (bit j of the
// matrix's row/column corresponds to operand j, CLAUDE.md's operand-order
// rule): legs = left_legs ++ right_legs (2k legs total). For a ket-style
// tensor (bra=false, matches a ket gate or the observable node): row bits
// come from the right legs, column bits from the left legs, value taken
// as-is. For a bra-style tensor (bra=true): the roles are swapped (row
// from left legs, column from right legs) and the value is conjugated --
// this exact convention (conjugate only, not transposed; left/right
// roles reversed relative to ket) was derived and numerically verified
// this session against a random unitary's own <0|U^dagger U|0> = 1
// identity, both without and with an inserted observable (see this
// module's docs/adr or the plan this phase was approved from).
LabeledTensor build_matrix_tensor(std::span<const Complex> matrix,
                                  const std::vector<LegId>& left_legs,
                                  const std::vector<LegId>& right_legs,
                                  bool bra) {
    const std::size_t k = left_legs.size();
    common::ensure(right_legs.size() == k, "build_matrix_tensor: left/right leg count mismatch");
    const std::size_t dim = std::size_t{1} << k;
    common::ensure(matrix.size() == dim * dim, "build_matrix_tensor: matrix size mismatch");

    LabeledTensor tensor;
    tensor.legs.reserve(2 * k);
    tensor.legs.insert(tensor.legs.end(), left_legs.begin(), left_legs.end());
    tensor.legs.insert(tensor.legs.end(), right_legs.begin(), right_legs.end());
    tensor.data.assign(std::size_t{1} << (2 * k), Complex{0.0, 0.0});

    for (std::size_t left_bits = 0; left_bits < dim; ++left_bits) {
        for (std::size_t right_bits = 0; right_bits < dim; ++right_bits) {
            const std::size_t row = bra ? left_bits : right_bits;
            const std::size_t col = bra ? right_bits : left_bits;
            Complex value = matrix[row * dim + col];
            if (bra) {
                value = std::conj(value);
            }
            const std::size_t flat = left_bits | (right_bits << k);
            tensor.data[flat] = value;
        }
    }
    return tensor;
}

LabeledTensor build_boundary_tensor(LegId leg) {
    return LabeledTensor{{Complex{1.0, 0.0}, Complex{0.0, 0.0}}, {leg}};
}

// Contracts two tensors over every leg they share (there must be at
// least one, checked by the caller via shares_a_leg), summing over the
// shared legs' 2 values each in ascending shared-leg order -- fixed, not
// hash or iteration order, CLAUDE.md's contraction-order determinism
// domain invariant (floating-point addition is not associative).
LabeledTensor contract_pair(const LabeledTensor& a, const LabeledTensor& b) {
    std::vector<std::size_t> a_shared_pos;
    std::vector<std::size_t> b_shared_pos;
    std::vector<std::size_t> a_free_pos;
    std::vector<std::size_t> b_free_pos;
    std::vector<LegId> result_legs;

    for (std::size_t i = 0; i < a.legs.size(); ++i) {
        const auto it = std::find(b.legs.begin(), b.legs.end(), a.legs[i]);
        if (it != b.legs.end()) {
            a_shared_pos.push_back(i);
            b_shared_pos.push_back(static_cast<std::size_t>(it - b.legs.begin()));
        } else {
            a_free_pos.push_back(i);
            result_legs.push_back(a.legs[i]);
        }
    }
    for (std::size_t j = 0; j < b.legs.size(); ++j) {
        if (std::find(b_shared_pos.begin(), b_shared_pos.end(), j) == b_shared_pos.end()) {
            b_free_pos.push_back(j);
            result_legs.push_back(b.legs[j]);
        }
    }

    common::ensure(!a_shared_pos.empty(), "contract_pair: no shared leg between the two tensors");

    const std::size_t num_shared = a_shared_pos.size();
    LabeledTensor result;
    result.legs = std::move(result_legs);
    result.data.assign(std::size_t{1} << result.legs.size(), Complex{0.0, 0.0});

    std::vector<int> a_bit(a.legs.size(), 0);
    std::vector<int> b_bit(b.legs.size(), 0);
    for (std::size_t free_flat = 0; free_flat < result.data.size(); ++free_flat) {
        for (std::size_t fi = 0; fi < a_free_pos.size(); ++fi) {
            a_bit[a_free_pos[fi]] = static_cast<int>((free_flat >> fi) & 1U);
        }
        for (std::size_t fj = 0; fj < b_free_pos.size(); ++fj) {
            b_bit[b_free_pos[fj]] = static_cast<int>((free_flat >> (a_free_pos.size() + fj)) & 1U);
        }

        Complex sum{0.0, 0.0};
        for (std::size_t shared_flat = 0; shared_flat < (std::size_t{1} << num_shared);
             ++shared_flat) {
            for (std::size_t s = 0; s < num_shared; ++s) {
                const int bit = static_cast<int>((shared_flat >> s) & 1U);
                a_bit[a_shared_pos[s]] = bit;
                b_bit[b_shared_pos[s]] = bit;
            }
            std::size_t a_flat = 0;
            for (std::size_t i = 0; i < a.legs.size(); ++i) {
                a_flat |= static_cast<std::size_t>(a_bit[i]) << i;
            }
            std::size_t b_flat = 0;
            for (std::size_t j = 0; j < b.legs.size(); ++j) {
                b_flat |= static_cast<std::size_t>(b_bit[j]) << j;
            }
            sum += a.data[a_flat] * b.data[b_flat];
        }
        result.data[free_flat] = sum;
    }
    return result;
}

// Union-find "find" with path compression: redirect[i] is the slot that
// currently holds node i's own contribution (i itself, until some other
// node absorbs it). A node's own slot can itself later be absorbed by a
// third node, so a lookup must follow the chain to its current root, not
// stop at the first redirect (see contract()'s own doc comment for why
// this indirection is needed at all).
std::size_t find_root(std::vector<std::size_t>& redirect, std::size_t i) {
    std::size_t root = i;
    while (redirect[root] != root) {
        root = redirect[root];
    }
    while (redirect[i] != root) {
        const std::size_t next = redirect[i];
        redirect[i] = root;
        i = next;
    }
    return root;
}

// Sums a tensor over one of its own legs appearing twice (a self-loop:
// both original endpoints of that wire have already been folded into
// this same tensor via some other path -- a cycle in the original
// network, e.g. two gates sharing more than one qubit pair). Positions
// p0 < p1 identify the two occurrences of `leg` in t.legs; the result
// drops both.
LabeledTensor trace_leg(const LabeledTensor& t, std::size_t p0, std::size_t p1) {
    std::vector<std::size_t> free_pos;
    std::vector<LegId> result_legs;
    for (std::size_t i = 0; i < t.legs.size(); ++i) {
        if (i != p0 && i != p1) {
            free_pos.push_back(i);
            result_legs.push_back(t.legs[i]);
        }
    }

    LabeledTensor result;
    result.legs = std::move(result_legs);
    result.data.assign(std::size_t{1} << result.legs.size(), Complex{0.0, 0.0});

    std::vector<int> bit(t.legs.size(), 0);
    for (std::size_t free_flat = 0; free_flat < result.data.size(); ++free_flat) {
        for (std::size_t fi = 0; fi < free_pos.size(); ++fi) {
            bit[free_pos[fi]] = static_cast<int>((free_flat >> fi) & 1U);
        }
        Complex sum{0.0, 0.0};
        for (int shared_bit = 0; shared_bit < 2; ++shared_bit) {
            bit[p0] = shared_bit;
            bit[p1] = shared_bit;
            std::size_t flat = 0;
            for (std::size_t i = 0; i < t.legs.size(); ++i) {
                flat |= static_cast<std::size_t>(bit[i]) << i;
            }
            sum += t.data[flat];
        }
        result.data[free_flat] = sum;
    }
    return result;
}

}  // namespace

ContractionNetwork build_contraction_network(std::uint32_t num_qubits,
                                             std::span<const Operation> operations,
                                             const PauliString& term) {
    const analysis::ClosedTimelines timelines =
        analysis::build_closed_timelines(num_qubits, operations, term);

    // node_id -> Operation*, for every ket-side gate node (node ids
    // [0, num_gate_nodes)); populated by walking `operations` in the
    // same order build_closed_timelines used to assign ket node ids
    // (every GateOp/UnitaryOp, consecutively, skipping MeasureOp).
    std::vector<const Operation*> gate_ops_in_order;
    for (const Operation& op : operations) {
        if (operand_qubits(op) != nullptr) {
            gate_ops_in_order.push_back(&op);
        }
    }

    // Every node's own leg assignment, gathered across whichever
    // per-qubit timelines touch it (a k-operand gate node is touched by
    // exactly k timelines, one per operand; boundary/observable nodes by
    // exactly one).
    std::map<std::size_t, LegAssignment> legs_of;
    for (std::size_t q = 0; q < timelines.per_qubit.size(); ++q) {
        const std::vector<ClosedTimelineEntry>& combined = timelines.per_qubit[q];
        for (std::size_t i = 0; i + 1 < combined.size(); ++i) {
            const ClosedTimelineEntry& a = combined[i];
            const ClosedTimelineEntry& b = combined[i + 1];
            if (a.node_id == b.node_id) {
                continue;
            }
            const LegId leg{std::min(a.node_id, b.node_id),
                            std::max(a.node_id, b.node_id),
                            static_cast<std::uint32_t>(q)};

            LegAssignment& left_owner = legs_of[a.node_id];
            if (left_owner.right.size() <= a.operand_index) {
                left_owner.right.resize(a.operand_index + 1);
            }
            left_owner.right[a.operand_index] = leg;

            LegAssignment& right_owner = legs_of[b.node_id];
            if (right_owner.left.size() <= b.operand_index) {
                right_owner.left.resize(b.operand_index + 1);
            }
            right_owner.left[b.operand_index] = leg;
        }
    }

    ContractionNetwork network;
    network.tensors.resize(timelines.num_nodes);

    // Kind, and (for the single-qubit-owned kinds -- boundary and
    // observable nodes only) which qubit owns it, from the same
    // timelines (a node may appear in several per-qubit timelines --
    // once per operand, for a KetGate/BraGate node -- so its Kind/owner
    // is read off the first entry seen for it; both are the same in
    // every entry for a given node id by construction).
    std::vector<Kind> kind_of(timelines.num_nodes);
    std::vector<bool> kind_known(timelines.num_nodes, false);
    std::vector<QubitId> owner_qubit_of(timelines.num_nodes, 0);
    for (std::size_t q = 0; q < timelines.per_qubit.size(); ++q) {
        for (const ClosedTimelineEntry& entry : timelines.per_qubit[q]) {
            if (!kind_known[entry.node_id]) {
                kind_of[entry.node_id] = entry.kind;
                kind_known[entry.node_id] = true;
                owner_qubit_of[entry.node_id] = static_cast<QubitId>(q);
            }
        }
    }

    for (std::size_t node_id = 0; node_id < timelines.num_nodes; ++node_id) {
        const Kind kind = kind_of[node_id];
        const LegAssignment& assignment = legs_of.at(node_id);
        switch (kind) {
            case Kind::BoundaryKet:
                network.tensors[node_id] = build_boundary_tensor(assignment.right.at(0));
                break;
            case Kind::BoundaryBra:
                network.tensors[node_id] = build_boundary_tensor(assignment.left.at(0));
                break;
            case Kind::Observable: {
                const QubitId q = owner_qubit_of[node_id];
                const std::array<Complex, 4> pauli = common::pauli_matrix_2x2(
                    common::get_bit(term.x_words, q), common::get_bit(term.z_words, q));
                network.tensors[node_id] =
                    build_matrix_tensor(pauli, assignment.left, assignment.right, /*bra=*/false);
                break;
            }
            case Kind::KetGate: {
                const std::vector<Complex> matrix =
                    operation_matrix(*gate_ops_in_order.at(node_id));
                network.tensors[node_id] =
                    build_matrix_tensor(matrix, assignment.left, assignment.right, /*bra=*/false);
                break;
            }
            case Kind::BraGate: {
                const std::size_t ket_node_id = node_id - gate_ops_in_order.size();
                const std::vector<Complex> matrix =
                    operation_matrix(*gate_ops_in_order.at(ket_node_id));
                network.tensors[node_id] =
                    build_matrix_tensor(matrix, assignment.left, assignment.right, /*bra=*/true);
                break;
            }
        }
    }

    return network;
}

ContractionResult contract(ContractionNetwork network,
                           const analysis::LineGraph& line_graph,
                           const std::vector<std::size_t>& elimination_order) {
    const std::size_t num_nodes = network.tensors.size();
    common::ensure(elimination_order.size() == line_graph.wire_of.size(),
                   "contract: elimination order does not cover every wire");

    std::vector<LabeledTensor> live = std::move(network.tensors);
    // redirect[i]: which slot currently holds original tensor node i's
    // own contribution (i itself, until some wire-elimination step folds
    // it into another node's slot).
    std::vector<std::size_t> redirect(num_nodes);
    std::iota(redirect.begin(), redirect.end(), std::size_t{0});
    std::size_t peak_legs = 0;
    for (const LabeledTensor& t : live) {
        peak_legs = std::max(peak_legs, t.legs.size());
    }

    for (const std::size_t wire_node : elimination_order) {
        const LegId& leg = line_graph.wire_of.at(wire_node);
        const std::size_t node_a = std::get<0>(leg);
        const std::size_t node_b = std::get<1>(leg);
        const std::size_t root_a = find_root(redirect, node_a);
        const std::size_t root_b = find_root(redirect, node_b);

        if (root_a == root_b) {
            // Both of this wire's original endpoints already live in the
            // same slot. Count how many of its own legs still equal
            // `leg`: 0 means an earlier contract_pair between the same
            // two (now-merged) slots already summed this wire away as
            // one of several legs they happened to share at once --
            // nothing left to do. Exactly 2 is a genuine trace (a cycle
            // in the original network reached this same slot from two
            // different directions).
            std::vector<std::size_t> positions;
            for (std::size_t i = 0; i < live[root_a].legs.size(); ++i) {
                if (live[root_a].legs[i] == leg) {
                    positions.push_back(i);
                }
            }
            if (positions.empty()) {
                continue;
            }
            common::ensure(positions.size() == 2,
                           "contract: a wire appeared neither 0 nor 2 times on its own merged "
                           "slot -- internal invariant violated");
            live[root_a] = trace_leg(live[root_a], positions[0], positions[1]);
            peak_legs = std::max(peak_legs, live[root_a].legs.size());
            continue;
        }

        live[root_a] = contract_pair(live[root_a], live[root_b]);
        redirect[root_b] = root_a;
        // root_b's slot is never read again (find_root now leads past it), so
        // its tensor is released here rather than when contract() returns
        // (docs/adr/0025 section 3.3).
        live[root_b] = LabeledTensor{};
        peak_legs = std::max(peak_legs, live[root_a].legs.size());
    }

    // Every original tensor node is now either folded into some other
    // node's slot, or is a surviving root holding a fully self-contained
    // connected-component result with no remaining open legs (a circuit
    // with an idle qubit, or one only ever touched by gates/an
    // observable entirely disjoint from the rest, yields more than one
    // such component -- the closed network's own boundary nodes close
    // each component independently, so nothing is ever left dangling).
    // The true scalar is the product of every surviving root's own
    // scalar.
    Complex scalar{1.0, 0.0};
    bool found_any = false;
    for (std::size_t i = 0; i < num_nodes; ++i) {
        if (find_root(redirect, i) != i) {
            continue;
        }
        common::ensure(live[i].legs.empty(),
                       "contract: a node survived the full elimination order with open legs");
        scalar *= live[i].data.at(0);
        found_any = true;
    }
    common::ensure(found_any, "contract: no surviving component -- empty network");

    return ContractionResult{scalar, peak_legs};
}

}  // namespace menos::engines::tensor_network
