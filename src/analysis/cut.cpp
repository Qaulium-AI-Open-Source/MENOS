#include "analysis/cut.h"

#include <algorithm>
#include <array>
#include <complex>
#include <map>
#include <numeric>
#include <set>
#include <span>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/pauli_row.h"

namespace menos::analysis {
namespace {

using common::checked_mul;

// Bounds on the search itself. The Phase 8 lesson applies directly: a
// candidate whose estimate() runs an unbounded combinatorial search slows
// every job that will never choose it, so the search is capped rather than
// gated out of the planner (docs/adr/0013 section 8).
constexpr std::size_t kMaxCandidatePositions = 48;
constexpr std::size_t kMaxPairEvaluations = 600;

// The quantum operands of a gate or unitary; nullptr for a MeasureOp or a
// ChannelOp, either of which puts a circuit outside the cut search's scope:
// the cut decomposition is exact for unitary fragments only, and cutting a
// noisy circuit is out of Phase 12's scope (docs/adr/0016 section 5).
// Exhaustive over Operation, like every dispatch below.
const std::vector<QubitId>* operands_of(const Operation& op) {
    return std::visit(
        [](const auto& o) -> const std::vector<QubitId>* {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                return &o.qubits;
            } else {
                static_assert(std::is_same_v<T, MeasureOp> || std::is_same_v<T, ChannelOp>);
                return nullptr;
            }
        },
        op);
}

// op with its operands replaced by qubits. A fragment holds only gates and
// unitaries: find_cut_plan returns no plan for a circuit with any MeasureOp
// or ChannelOp.
Operation with_qubits(const Operation& op, std::vector<QubitId> qubits) {
    return std::visit(
        [&](const auto& o) -> Operation {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                return GateOp{o.opcode, std::move(qubits), o.params};
            } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                return UnitaryOp{std::move(qubits), o.matrix};
            } else {
                static_assert(std::is_same_v<T, MeasureOp> || std::is_same_v<T, ChannelOp>);
                throw common::InternalError("cut: a non-unitary operation reached a fragment");
            }
        },
        op);
}

// Union-find over (qubit, segment) nodes.
class DisjointSet {
public:
    explicit DisjointSet(std::size_t n) : parent_(n) {
        std::iota(parent_.begin(), parent_.end(), std::size_t{0});
    }
    std::size_t find(std::size_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }
    void unite(std::size_t a, std::size_t b) {
        const std::size_t ra = find(a);
        const std::size_t rb = find(b);
        if (ra != rb) {
            parent_[ra] = rb;
        }
    }

private:
    std::vector<std::size_t> parent_;
};

// For each qubit, the sorted op indices at which it is cut.
std::vector<std::vector<std::size_t>> cuts_by_qubit(const Circuit& circuit,
                                                    const std::vector<WireCut>& cuts) {
    std::vector<std::vector<std::size_t>> out(circuit.num_qubits());
    for (const WireCut& cut : cuts) {
        out[cut.qubit].push_back(cut.op_index);
    }
    for (std::vector<std::size_t>& positions : out) {
        std::sort(positions.begin(), positions.end());
    }
    return out;
}

// Which timeline segment of `qubit` operation index `op_index` belongs to.
std::size_t segment_of(const std::vector<std::size_t>& qubit_cuts, std::size_t op_index) {
    std::size_t segment = 0;
    for (const std::size_t position : qubit_cuts) {
        if (op_index >= position) {
            segment += 1;
        }
    }
    return segment;
}

}  // namespace

std::optional<CutPlan> build_cut_plan_for(const Circuit& circuit, const std::vector<WireCut>& cuts);

// Builds the fragment decomposition induced by `cuts`, or nullopt when the
// result is not usable for Phase 9 (fewer than two fragments, or a fragment
// holding two segments of one qubit).
std::optional<CutPlan> build_cut_plan_for(const Circuit& circuit,
                                          const std::vector<WireCut>& cuts) {
    const std::vector<std::vector<std::size_t>> per_qubit = cuts_by_qubit(circuit, cuts);

    // Node id for (qubit, segment): offsets[q] + segment.
    std::vector<std::size_t> offsets(circuit.num_qubits() + 1, 0);
    for (std::uint32_t q = 0; q < circuit.num_qubits(); ++q) {
        offsets[q + 1] = offsets[q] + per_qubit[q].size() + 1;
    }
    const std::size_t node_count = offsets[circuit.num_qubits()];
    DisjointSet sets(node_count);

    const std::vector<Operation>& ops = circuit.operations();
    std::vector<std::size_t> op_node(ops.size(), 0);
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const std::vector<QubitId>* operands = operands_of(ops[i]);
        if (operands == nullptr || operands->empty()) {
            return std::nullopt;  // MeasureOp: outside Phase 9's scope
        }
        const std::size_t first =
            offsets[(*operands)[0]] + segment_of(per_qubit[(*operands)[0]], i);
        op_node[i] = first;
        for (const QubitId q : *operands) {
            sets.unite(first, offsets[q] + segment_of(per_qubit[q], i));
        }
    }

    // Group nodes into fragments by their component root, in a deterministic
    // order (first appearance by node id), never by hash order.
    std::map<std::size_t, std::size_t> root_to_fragment;
    std::vector<std::vector<std::size_t>> fragment_nodes;
    for (std::size_t node = 0; node < node_count; ++node) {
        const std::size_t root = sets.find(node);
        auto it = root_to_fragment.find(root);
        if (it == root_to_fragment.end()) {
            root_to_fragment.emplace(root, fragment_nodes.size());
            fragment_nodes.emplace_back();
            it = root_to_fragment.find(root);
        }
        fragment_nodes[it->second].push_back(node);
    }
    if (fragment_nodes.size() < 2) {
        return std::nullopt;
    }

    // Decode nodes back to (qubit, segment) and build each fragment.
    CutPlan plan;
    plan.cuts = cuts;
    plan.fragments.resize(fragment_nodes.size());
    std::vector<std::size_t> node_to_fragment(node_count, 0);
    std::vector<QubitId> node_to_local(node_count, 0);

    for (std::size_t f = 0; f < fragment_nodes.size(); ++f) {
        CutFragment& fragment = plan.fragments[f];
        std::set<QubitId> seen_qubits;
        for (const std::size_t node : fragment_nodes[f]) {
            // Recover the qubit owning this node.
            const auto upper = std::upper_bound(offsets.begin(), offsets.end(), node);
            const auto qubit = static_cast<QubitId>(std::distance(offsets.begin(), upper) - 1);
            if (!seen_qubits.insert(qubit).second) {
                // Two segments of one qubit in one fragment: the reconstruction
                // below assumes one local wire per global qubit per fragment.
                return std::nullopt;
            }
            const std::size_t segment = node - offsets[qubit];
            node_to_fragment[node] = f;
            node_to_local[node] = static_cast<QubitId>(fragment.local_to_global.size());
            fragment.local_to_global.push_back(qubit);

            // A segment after the first begins at a cut: this fragment
            // prepares it. A segment before the last is followed by a cut:
            // this fragment measures it. The last segment carries the qubit's
            // final state, so the job observable's factor lives here.
            if (segment > 0) {
                const std::size_t position = per_qubit[qubit][segment - 1];
                for (std::size_t c = 0; c < cuts.size(); ++c) {
                    if (cuts[c].qubit == qubit && cuts[c].op_index == position) {
                        fragment.prepared.push_back(CutEndpoint{c, node_to_local[node]});
                        break;
                    }
                }
            }
            if (segment < per_qubit[qubit].size()) {
                const std::size_t position = per_qubit[qubit][segment];
                for (std::size_t c = 0; c < cuts.size(); ++c) {
                    if (cuts[c].qubit == qubit && cuts[c].op_index == position) {
                        fragment.measured.push_back(CutEndpoint{c, node_to_local[node]});
                        break;
                    }
                }
            } else {
                fragment.final_locals.push_back(node_to_local[node]);
            }
        }
    }

    // Distribute operations, remapped to each fragment's local indices.
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const std::size_t root = sets.find(op_node[i]);
        const std::size_t f = root_to_fragment.at(root);
        CutFragment& fragment = plan.fragments[f];
        const std::vector<QubitId>* operands = operands_of(ops[i]);
        std::vector<QubitId> local(operands->size());
        for (std::size_t j = 0; j < operands->size(); ++j) {
            const QubitId q = (*operands)[j];
            local[j] = node_to_local[offsets[q] + segment_of(per_qubit[q], i)];
        }
        fragment.operations.push_back(with_qubits(ops[i], std::move(local)));
    }
    return plan;
}

std::optional<CutPlan> find_cut_plan(const Circuit& circuit, std::size_t max_cuts) {
    if (max_cuts == 0 || circuit.operations().empty()) {
        return std::nullopt;
    }
    for (const Operation& op : circuit.operations()) {
        if (operands_of(op) == nullptr) {
            return std::nullopt;  // expectation-only, unitary scope: no MeasureOp or ChannelOp
        }
    }

    // Candidate positions: the gap before each operation a qubit takes part
    // in, after its first. Cutting anywhere else on that qubit produces the
    // same segmentation.
    std::vector<WireCut> candidates;
    std::vector<bool> seen_qubit(circuit.num_qubits(), false);
    const std::vector<Operation>& ops = circuit.operations();
    for (std::size_t i = 0; i < ops.size() && candidates.size() < kMaxCandidatePositions; ++i) {
        for (const QubitId q : *operands_of(ops[i])) {
            if (seen_qubit[q]) {
                candidates.push_back(WireCut{q, i});
            }
            seen_qubit[q] = true;
        }
    }
    if (candidates.empty()) {
        return std::nullopt;
    }

    // Prefer the plan whose largest fragment is smallest, since peak memory is
    // the max over fragments (docs/adr/0013 Eq. 4.3) and that is what makes a
    // cut worth taking. Ties break on fewer cuts, then on earlier candidate
    // order, so the choice is deterministic.
    std::optional<CutPlan> best;
    std::size_t best_widest = 0;
    const auto consider = [&](const std::vector<WireCut>& cuts) {
        std::optional<CutPlan> plan = build_cut_plan_for(circuit, cuts);
        if (!plan.has_value()) {
            return;
        }
        std::size_t widest = 0;
        for (const CutFragment& fragment : plan->fragments) {
            widest = std::max(widest, fragment.local_to_global.size());
        }
        if (!best.has_value() || widest < best_widest ||
            (widest == best_widest && cuts.size() < best->cuts.size())) {
            best_widest = widest;
            best = std::move(plan);
        }
    };

    for (const WireCut& candidate : candidates) {
        consider({candidate});
    }
    if (best.has_value() || max_cuts < 2) {
        return best;
    }

    std::size_t evaluations = 0;
    for (std::size_t a = 0; a < candidates.size() && evaluations < kMaxPairEvaluations; ++a) {
        for (std::size_t b = a + 1; b < candidates.size() && evaluations < kMaxPairEvaluations;
             ++b) {
            evaluations += 1;
            consider({candidates[a], candidates[b]});
        }
    }
    return best;
}

Circuit build_fragment_circuit(const CutFragment& fragment, const FragmentAssignment& assignment) {
    common::ensure(
        assignment.prepared_states.size() == fragment.prepared.size(),
        "build_fragment_circuit: assignment does not match the fragment's prepared cuts");
    Circuit circuit(static_cast<std::uint32_t>(fragment.local_to_global.size()), 0);
    for (std::size_t i = 0; i < fragment.prepared.size(); ++i) {
        for (const Opcode op : ir::preparation_gates(assignment.prepared_states[i])) {
            circuit.append(op, {fragment.prepared[i].local_qubit});
        }
    }
    for (const Operation& op : fragment.operations) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    circuit.append(o.opcode, o.qubits, o.params);
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    circuit.append_unitary(o.qubits, o.matrix);
                } else {
                    static_assert(std::is_same_v<T, MeasureOp> || std::is_same_v<T, ChannelOp>);
                    throw common::InternalError("cut: a non-unitary operation reached a fragment");
                }
            },
            op);
    }
    return circuit;
}

PauliString build_fragment_term(const CutFragment& fragment,
                                const PauliString& job_term,
                                const FragmentAssignment& assignment) {
    common::ensure(assignment.measured_observables.size() == fragment.measured.size(),
                   "build_fragment_term: assignment does not match the fragment's measured cuts");
    const auto width = static_cast<std::uint32_t>(fragment.local_to_global.size());
    PauliString term;
    term.num_qubits = width;
    term.x_words.assign((width + 63) / 64, 0);
    term.z_words.assign((width + 63) / 64, 0);

    // The job observable's own factors, on the qubits whose final state this
    // fragment holds.
    for (const QubitId local : fragment.final_locals) {
        const QubitId global = fragment.local_to_global[local];
        if (global < job_term.num_qubits) {
            if (common::get_bit(job_term.x_words, global)) {
                term.x_words[local / 64] |= (std::uint64_t{1} << (local % 64));
            }
            if (common::get_bit(job_term.z_words, global)) {
                term.z_words[local / 64] |= (std::uint64_t{1} << (local % 64));
            }
        }
    }
    // The inserted cut observables.
    for (std::size_t i = 0; i < fragment.measured.size(); ++i) {
        const ir::PauliBits bits = ir::observable_bits(assignment.measured_observables[i]);
        const QubitId local = fragment.measured[i].local_qubit;
        if (bits.x) {
            term.x_words[local / 64] |= (std::uint64_t{1} << (local % 64));
        }
        if (bits.z) {
            term.z_words[local / 64] |= (std::uint64_t{1} << (local % 64));
        }
    }
    return term;
}

std::uint64_t distinct_evaluation_count(const CutFragment& fragment) {
    std::uint64_t count = 1;
    for (std::size_t i = 0; i < fragment.measured.size(); ++i) {
        count = checked_mul(count, ir::kDistinctCutObservables);
    }
    for (std::size_t i = 0; i < fragment.prepared.size(); ++i) {
        count = checked_mul(count, ir::kDistinctCutPrepStates);
    }
    return count;
}

std::vector<FragmentAssignment> enumerate_assignments(const CutFragment& fragment) {
    const std::array<ir::CutObservable, 4> observables = {
        ir::CutObservable::I, ir::CutObservable::X, ir::CutObservable::Y, ir::CutObservable::Z};
    const std::array<ir::CutPrepState, 6> states = {ir::CutPrepState::Zero,
                                                    ir::CutPrepState::One,
                                                    ir::CutPrepState::Plus,
                                                    ir::CutPrepState::Minus,
                                                    ir::CutPrepState::PlusI,
                                                    ir::CutPrepState::MinusI};

    std::vector<FragmentAssignment> out;
    const std::uint64_t total = distinct_evaluation_count(fragment);
    out.reserve(total);
    for (std::uint64_t index = 0; index < total; ++index) {
        std::uint64_t rest = index;
        FragmentAssignment assignment;
        assignment.measured_observables.reserve(fragment.measured.size());
        assignment.prepared_states.reserve(fragment.prepared.size());
        for (std::size_t i = 0; i < fragment.measured.size(); ++i) {
            assignment.measured_observables.push_back(observables[rest % observables.size()]);
            rest /= observables.size();
        }
        for (std::size_t i = 0; i < fragment.prepared.size(); ++i) {
            assignment.prepared_states.push_back(states[rest % states.size()]);
            rest /= states.size();
        }
        out.push_back(std::move(assignment));
    }
    return out;
}

namespace {

// An operation's own heap: its operand list, and its parameters or matrix.
std::uint64_t operation_heap_bytes(const Operation& op) {
    return std::visit(
        [](const auto& o) -> std::uint64_t {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                return common::heap_block_bytes(o.qubits.size() * sizeof(QubitId)) +
                       common::heap_block_bytes(o.params.size() * sizeof(double));
            } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                return common::heap_block_bytes(o.qubits.size() * sizeof(QubitId)) +
                       common::heap_block_bytes(o.matrix.size() * sizeof(std::complex<double>));
            } else {
                static_assert(std::is_same_v<T, MeasureOp> || std::is_same_v<T, ChannelOp>);
                return 0;  // never in a cut circuit: find_cut_plan refuses both
            }
        },
        op);
}

}  // namespace

std::uint64_t find_cut_plan_peak_bytes(const Circuit& circuit) {
    using common::checked_add;
    using common::grown_vector_bytes;
    const std::uint64_t n = circuit.num_qubits();
    const std::uint64_t ops = circuit.operations().size();
    const std::uint64_t nodes = n + kMaxCuts;  // (qubit, segment) nodes
    // Each array below is one allocation, charged this overhead; every list
    // grown one element at a time is charged two by grown_vector_bytes.
    const std::uint64_t block = common::kHeapBlockOverheadBytes;
    std::uint64_t op_heap = 0;
    for (const Operation& op : circuit.operations()) {
        op_heap = checked_add(op_heap, operation_heap_bytes(op));
    }

    // One CutPlan for this circuit: its cuts, at most `nodes` fragments, and
    // across them every (qubit, segment) node once, every cut endpoint once on
    // each side, every qubit's final local once and every operation once,
    // each list grown one element at a time.
    std::uint64_t plan = (kMaxCuts * sizeof(WireCut)) + block;
    plan = checked_add(plan, (nodes * sizeof(CutFragment)) + block);
    plan = checked_add(plan, grown_vector_bytes(nodes, sizeof(QubitId)));
    plan = checked_add(plan, 2 * grown_vector_bytes(kMaxCuts, sizeof(CutEndpoint)));
    plan = checked_add(plan, grown_vector_bytes(n, sizeof(QubitId)));
    plan = checked_add(plan, grown_vector_bytes(ops, sizeof(Operation)));
    plan = checked_add(plan, op_heap);
    // Each fragment's five lists are separate allocations, two while growing.
    plan = checked_add(plan, nodes * 10 * block);

    // build_cut_plan_for()'s own arrays while it builds one.
    std::uint64_t build = (n * sizeof(std::vector<std::size_t>)) + block +
                          grown_vector_bytes(kMaxCuts, sizeof(std::size_t));  // per_qubit
    build = checked_add(build, ((n + 1) * sizeof(std::size_t)) + block);      // offsets
    build = checked_add(build, (nodes * sizeof(std::size_t)) + block);        // DisjointSet
    build = checked_add(build, (ops * sizeof(std::size_t)) + block);          // op_node
    build = checked_add(
        build,
        (nodes + 1) * common::tree_node_bytes(sizeof(std::pair<const std::size_t, std::size_t>)));
    build = checked_add(build, (n + nodes) * 2 * block);  // per-qubit and per-fragment lists
    build = checked_add(build,
                        grown_vector_bytes(nodes, sizeof(std::vector<std::size_t>)) +
                            grown_vector_bytes(nodes, sizeof(std::size_t)));  // fragment_nodes
    build = checked_add(build, (nodes * (sizeof(std::size_t) + sizeof(QubitId))) + (2 * block));
    build =
        checked_add(build, (2 * nodes) * common::tree_node_bytes(sizeof(QubitId)));  // seen_qubits
    build = checked_add(build, (5 * sizeof(QubitId)) + block);  // one op's local list

    // find_cut_plan(): the candidate positions, the per-qubit flags, the
    // cut list being considered, and the best plan beside the one built.
    std::uint64_t search = grown_vector_bytes(kMaxCandidatePositions, sizeof(WireCut));
    search = checked_add(search, ((n + 63) / 64 * sizeof(std::uint64_t)) + block);
    search = checked_add(search, (kMaxCuts * sizeof(WireCut)) + block);
    return checked_add(search, checked_add(build, 2 * plan));
}

std::uint64_t cut_plan_heap_bytes(const CutPlan& plan) {
    std::uint64_t bytes = common::heap_block_bytes(plan.cuts.capacity() * sizeof(WireCut)) +
                          common::heap_block_bytes(plan.fragments.capacity() * sizeof(CutFragment));
    for (const CutFragment& fragment : plan.fragments) {
        bytes += common::heap_block_bytes(fragment.local_to_global.capacity() * sizeof(QubitId));
        bytes += common::heap_block_bytes(fragment.operations.capacity() * sizeof(Operation));
        for (const Operation& op : fragment.operations) {
            bytes += operation_heap_bytes(op);
        }
        bytes += common::heap_block_bytes(fragment.measured.capacity() * sizeof(CutEndpoint));
        bytes += common::heap_block_bytes(fragment.prepared.capacity() * sizeof(CutEndpoint));
        bytes += common::heap_block_bytes(fragment.final_locals.capacity() * sizeof(QubitId));
    }
    return bytes;
}

std::uint64_t assignments_heap_bytes(const std::vector<FragmentAssignment>& assignments) {
    std::uint64_t bytes =
        common::heap_block_bytes(assignments.capacity() * sizeof(FragmentAssignment));
    for (const FragmentAssignment& assignment : assignments) {
        bytes += common::heap_block_bytes(assignment.measured_observables.capacity() *
                                          sizeof(ir::CutObservable)) +
                 common::heap_block_bytes(assignment.prepared_states.capacity() *
                                          sizeof(ir::CutPrepState));
    }
    return bytes;
}

std::size_t assignment_index_for_branch(const CutFragment& fragment,
                                        const std::vector<std::size_t>& branch_terms) {
    const std::span<const ir::WireCutTerm> terms = ir::wire_cut_terms();
    // The same mixed-radix encoding enumerate_assignments() decodes: measured
    // cuts first (radix 4), then prepared cuts (radix 6).
    std::size_t index = 0;
    std::size_t place = 1;
    for (const CutEndpoint& endpoint : fragment.measured) {
        const ir::CutObservable observable = terms[branch_terms[endpoint.cut_index]].observable;
        index += static_cast<std::size_t>(observable) * place;
        place *= ir::kDistinctCutObservables;
    }
    for (const CutEndpoint& endpoint : fragment.prepared) {
        const ir::CutPrepState state = terms[branch_terms[endpoint.cut_index]].prepared;
        index += static_cast<std::size_t>(state) * place;
        place *= ir::kDistinctCutPrepStates;
    }
    return index;
}

}  // namespace menos::analysis
