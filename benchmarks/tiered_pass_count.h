#pragma once

// The pass count of the chunked executor docs/adr/0027 section 3.2 proposes,
// as a pure function of a circuit, the number of local qubits L and the
// window's global-qubit capacity w. Phase 23 step 1's analysis (iv) ran this
// greedy walk from a benchmark-local copy before any engine code existed
// (commit 88294be, docs/benchmarks/phase-23-baseline); step 2d moved the walk
// into src/engines/dense_tiered/pass_plan.h with its tests, and this header
// is now a view of that plan in the benchmark's own terms, so the benchmark's
// table is produced by the code the executor runs.

#include <cstdint>
#include <span>

#include "engines/dense_tiered/pass_plan.h"
#include "menos/circuit.h"

namespace menos::benchmarks {

struct PassCount {
    // Passes over the whole state; 0 for a circuit with no gate.
    std::uint64_t passes = 0;
    // Gates with at least one global operand.
    std::uint64_t global_gates = 0;
    // The most global operands any one gate has: the window must hold 2^that
    // chunks, so a window with w below it cannot run the circuit.
    std::uint32_t max_global_operands = 0;
    bool window_too_small = false;
};

struct PassCountRequest {
    std::uint32_t local_qubits = 0;
    std::uint32_t window_globals = 0;
};

inline PassCount count_passes(std::span<const menos::Operation> operations,
                              PassCountRequest request) {
    // The qubit count does not enter the count: only which operands are at
    // or above local_qubits does.
    const engines::dense_tiered::PassPlan plan =
        engines::dense_tiered::build_pass_plan(operations,
                                               /*num_qubits=*/64,
                                               request.local_qubits,
                                               request.local_qubits + request.window_globals);
    return PassCount{
        plan.pass_count(), plan.global_gates, plan.max_global_operands, plan.window_too_small};
}

}  // namespace menos::benchmarks
