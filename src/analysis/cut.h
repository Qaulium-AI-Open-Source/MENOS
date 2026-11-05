#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ir/wire_cut_basis.h"
#include "menos/circuit.h"
#include "menos/observable.h"

namespace menos::analysis {

// Phase 9 wire cutting (docs/adr/0013). Internal to the plan module: none of
// this is public API.
//
// A cut severs one qubit's world line at one instant, so the same global qubit
// can belong to two different fragments -- before the cut and after it. That
// is the single thing Phase 3's connected-component regions cannot express,
// and the reason this module exists. Note that `Region` itself needs no time
// range: a fragment's time extent is already implied by which operations it
// owns (docs/adr/0013 section 6, as refined during implementation).

// Sever `qubit`'s wire immediately BEFORE operation `op_index` of the
// circuit's operation list.
struct WireCut {
    QubitId qubit;
    std::size_t op_index;
};

// One side of a cut as seen by a fragment: which cut, and which of the
// fragment's own local qubits carries it.
struct CutEndpoint {
    std::size_t cut_index;  // index into CutPlan::cuts
    QubitId local_qubit;
};

struct CutFragment {
    // Global qubit for each local index. A global qubit appears in at most
    // one fragment per timeline segment; a migrating qubit therefore appears
    // in two fragments, at possibly different local indices.
    std::vector<QubitId> local_to_global;
    // This fragment's own operations, already remapped to local indices, in
    // program order.
    std::vector<Operation> operations;
    // Cuts whose UPSTREAM side ends here: this fragment measures the cut
    // observable on `local_qubit`.
    std::vector<CutEndpoint> measured;
    // Cuts whose DOWNSTREAM side starts here: this fragment prepares
    // `local_qubit` in the cut's eigenstate.
    std::vector<CutEndpoint> prepared;
    // Local qubits holding the FINAL state of their global qubit, so the
    // job's own observable factor for that global qubit belongs to this
    // fragment.
    std::vector<QubitId> final_locals;
};

struct CutPlan {
    std::vector<WireCut> cuts;
    std::vector<CutFragment> fragments;
};

// Searches for a cut plan that splits `circuit` into two or more fragments,
// bounded by `max_cuts`.
//
// Scope, stated rather than implied: the search considers cut sets of size 1
// and 2 over the positions where a qubit's operation timeline has a gap, and
// rejects any plan in which a fragment would hold two segments of the same
// qubit. Circuits containing a MeasureOp are rejected outright, since Phase 9
// reconstructs expectation values only (docs/adr/0013 section 7). A richer
// search is deliberately out of scope; the planner's cost comparison, not the
// finder's cleverness, decides whether a cut is used.
// The cut-count bound the planner and the runtime must both use, so the
// runtime's recomputation of the plan is guaranteed to reproduce the planner's
// (find_cut_plan is deterministic given the same circuit and bound).
inline constexpr std::size_t kMaxCuts = 2;

[[nodiscard]] std::optional<CutPlan> find_cut_plan(const Circuit& circuit, std::size_t max_cuts);

// Upper bounds on heap bytes (Phase 21, docs/adr/0025 section 3.4).
// find_cut_plan_peak_bytes: the most find_cut_plan(circuit, kMaxCuts) holds at
// once, including its result -- the best plan so far, the candidate plan
// being built and that build's own working arrays -- from the circuit's sizes
// alone. cut_plan_heap_bytes: what one CutPlan holds, from its own
// containers.
[[nodiscard]] std::uint64_t find_cut_plan_peak_bytes(const Circuit& circuit);
[[nodiscard]] std::uint64_t cut_plan_heap_bytes(const CutPlan& plan);

// One fragment evaluation: which Pauli each of its measured cuts carries, and
// which eigenstate each of its prepared cuts starts in. The vectors are
// parallel to CutFragment::measured / ::prepared.
struct FragmentAssignment {
    std::vector<ir::CutObservable> measured_observables;
    std::vector<ir::CutPrepState> prepared_states;
};

// Builds the concrete circuit this fragment runs for a given assignment: the
// preparation gates for each prepared cut, then the fragment's own operations.
[[nodiscard]] Circuit build_fragment_circuit(const CutFragment& fragment,
                                             const FragmentAssignment& assignment);

// Builds the Pauli string this fragment measures for a given assignment: the
// job term's own factors on this fragment's final qubits, times the inserted
// cut observable on each measured cut (docs/adr/0013 Eq. 2.2).
[[nodiscard]] PauliString build_fragment_term(const CutFragment& fragment,
                                              const PauliString& job_term,
                                              const FragmentAssignment& assignment);

// The number of distinct evaluations this fragment needs: 4^(measured cuts) *
// 6^(prepared cuts), per docs/adr/0013 Eq. (4.1). This is strictly smaller
// than the 8^K branch count, and the gap is the reason a cut plan is
// affordable at all.
[[nodiscard]] std::uint64_t distinct_evaluation_count(const CutFragment& fragment);

// Enumerates this fragment's distinct assignments in a fixed order, so the
// evaluation order -- and therefore the floating-point summation order of the
// reconstruction -- is deterministic.
[[nodiscard]] std::vector<FragmentAssignment> enumerate_assignments(const CutFragment& fragment);

// The heap bytes enumerate_assignments() returns, from the vector itself.
[[nodiscard]] std::uint64_t assignments_heap_bytes(
    const std::vector<FragmentAssignment>& assignments);

// The index into enumerate_assignments() that a given branch selects for this
// fragment. `branch_terms[c]` is the wire-cut term index chosen for cut c.
[[nodiscard]] std::size_t assignment_index_for_branch(const CutFragment& fragment,
                                                      const std::vector<std::size_t>& branch_terms);

}  // namespace menos::analysis
