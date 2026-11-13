#pragma once

#include <complex>
#include <cstdint>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "analysis/treewidth.h"
#include "menos/circuit.h"
#include "menos/observable.h"

namespace menos::engines::tensor_network {

// A leg is exactly one wire of the closed network's own line graph
// (analysis::WireId, analysis::LineGraph) -- the same identity, reused
// rather than re-declared, since this module's own elimination order
// (from analysis::estimate_treewidth_min_fill run on the LINE graph) is
// indexed by wires, and contract() below needs to look up which two
// tensor nodes a given wire-elimination-order entry actually connects.
// Every leg has dimension 2 (one qubit's Hilbert space per wire).
using LegId = analysis::WireId;

// data.size() == 1 << legs.size(); bit i of a flat index selects
// legs[i]'s own value (0 or 1).
struct LabeledTensor {
    std::vector<std::complex<double>> data;
    std::vector<LegId> legs;
};

// One initial tensor per node of the closed bra-ket-observable network
// (analysis::build_closed_contraction_graph's own node numbering,
// reproduced here independently -- see this file's .cpp for why, and
// how the two are kept from drifting apart). tensors.size() ==
// analysis::build_closed_contraction_graph(...).num_nodes for the same
// arguments.
struct ContractionNetwork {
    std::vector<LabeledTensor> tensors;
};

// "Left"/"right" below are always positional, per operand qubit, within
// that qubit's own combined boundary+ket+observable+bra timeline
// (analysis::ClosedTimelines): left = this node's neighbor earlier in
// that timeline (toward the |0> boundary's own array position), right =
// later (toward the <0| boundary's position). This is the SAME leg for
// two adjacent nodes (one's right is the other's left) but does not mean
// "input side" for both kinds of node -- a ket node's left neighbor sits
// closer to |0> (so left = input, right = output), while a bra node's
// left neighbor sits closer to the *observable* (the bra sequence runs
// in reverse circuit order), which is the reverse assignment. Builds
// every node's own initial tensor:
// - A |0>/<0| boundary: rank 1, data = {1, 0}.
// - A ket gate (GateOp/UnitaryOp): rank 2k for a k-qubit gate, legs
//   ordered [operand_0_left, ..., operand_{k-1}_left, operand_0_right,
//   ..., operand_{k-1}_right], data[combined] = M[right_bits][left_bits]
//   (M = ir::generate_matrix(opcode, params); row = right = output,
//   column = left = input, the standard convention, verified this
//   session against a random unitary's own <0|Udag U|0> = 1 identity).
// - A bra gate (the mirrored copy of a ket gate, at the same operand
//   qubits): same legs arrangement, data[combined] =
//   conj(M[left_bits][right_bits]) -- row/column roles SWAPPED relative
//   to the ket copy (row = left, column = right) because the bra
//   sequence runs the same gates backward, value conjugated, never
//   transposed (verified this session against the same random-unitary
//   identity, and against a full observable-sandwich case).
// - A Pauli-operator node (only for a qubit `term` is non-trivial on):
//   rank 2, same row(=right)/column(=left) convention as a ket gate
//   (never conjugated), data from common::pauli_matrix_2x2.
[[nodiscard]] ContractionNetwork build_contraction_network(std::uint32_t num_qubits,
                                                           std::span<const Operation> operations,
                                                           const PauliString& term);

// The result of contracting a ContractionNetwork in a validated
// elimination order (analysis::EliminationOrder::order): the final
// scalar (<0|U^dagger P U|0>, the raw term value -- coefficient NOT
// applied, matching EngineExecutionResult::expectation_terms's existing
// convention) and the actual peak number of legs any intermediate
// tensor reached during contraction, for comparison against the
// pre-execution treewidth estimate (docs/BUILD_PLAN.md Phase 8's
// estimated-versus-actual explain() requirement).
struct ContractionResult {
    std::complex<double> scalar;
    std::size_t actual_peak_legs;
};

// Contracts network via bucket elimination over WIRES (analysis::
// LineGraph, analysis::EliminationOrder::order computed by running
// estimate_treewidth_min_fill on line_graph.graph -- never on a bare
// tensor-node graph: contraction complexity is bounded by the line
// graph's own treewidth, not the tensor-node graph's, verified against
// two independent published sources, see analysis::LineGraph's own doc
// comment for the full citation and for why an earlier, tensor-node-
// graph-driven attempt at this function did not terminate in reasonable
// time on a realistic circuit).
//
// At each step, wire w's own two original endpoints (line_graph.
// wire_of[w]) are resolved to whichever live tensor slot currently holds
// each one's contribution (a node is only ever absorbed once, via
// whichever wire between it and something else is eliminated first, so
// later references to it redirect to that slot). If both endpoints
// already share the same slot (both sides of this wire were already
// folded together via some other path -- a cycle in the original
// network, e.g. two gates sharing more than one qubit pair), it is
// summed out as a trace within that one tensor; otherwise the two slots
// are multiplied together over every leg they currently share (normally
// just this one wire, occasionally more if two nodes happen to share
// several wires that reach elimination via the same step) and merged
// into one slot. Fixed ascending loop order throughout (CLAUDE.md's
// contraction-order determinism domain invariant -- floating-point
// addition is not associative, so this loop order is part of the
// result's determinism contract, not free to vary).
[[nodiscard]] ContractionResult contract(ContractionNetwork network,
                                         const analysis::LineGraph& line_graph,
                                         const std::vector<std::size_t>& elimination_order);

// The network build_contraction_network() builds, as leg lists only, and its
// memory (Phase 21, docs/adr/0025 section 3.4): legs[i] is node i's tensor's
// legs in that tensor's own order; network_bytes the heap bytes of the
// network it returns; build_peak_bytes the most it holds at once while
// building, including that network. Builds no tensor data.
struct NetworkShape {
    std::vector<std::vector<LegId>> legs;
    std::uint64_t network_bytes = 0;
    std::uint64_t build_peak_bytes = 0;
};
[[nodiscard]] NetworkShape network_shape(std::uint32_t num_qubits,
                                         std::span<const Operation> operations,
                                         const PauliString& term);

// The most heap bytes contract() holds at once for a network of this shape,
// including the network it consumes: replayed on leg lists, so every
// intermediate's size is exact, and every tensor already absorbed is released
// as contract() releases it.
[[nodiscard]] std::uint64_t contraction_peak_bytes(
    const NetworkShape& shape,
    const analysis::LineGraph& line_graph,
    const std::vector<std::size_t>& elimination_order);

}  // namespace menos::engines::tensor_network
