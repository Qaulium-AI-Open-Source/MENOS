#pragma once

// The pass count of the chunked executor docs/adr/0027 section 3.2 proposes,
// as a pure function of a circuit, the number of local qubits L and the
// window's global-qubit capacity w -- Phase 23 step 1's analysis (iv), before
// any engine code exists. Step 2 moves this walk into src/ with tests.
//
// Qubits 0..L-1 are local (index bits inside a chunk), L..n-1 global (the
// chunk number). A pass holds a set G of at most w global qubits and absorbs
// every consecutive operation whose global operands lie in G; the greedy rule
// extends G with a gate's global operands while that fits, and otherwise
// closes the pass and opens the next with that gate's global operands.

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <span>
#include <variant>
#include <vector>

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
    PassCount result;
    std::vector<std::uint32_t> group;  // G, sorted
    bool open = false;
    for (const menos::Operation& operation : operations) {
        const std::vector<menos::QubitId>* qubits = nullptr;
        if (const auto* gate = std::get_if<menos::GateOp>(&operation)) {
            qubits = &gate->qubits;
        } else if (const auto* unitary = std::get_if<menos::UnitaryOp>(&operation)) {
            qubits = &unitary->qubits;
        } else {
            continue;  // measurements and channels are not passes
        }
        std::vector<std::uint32_t> globals;
        for (const menos::QubitId q : *qubits) {
            if (q >= request.local_qubits) {
                globals.push_back(q);
            }
        }
        std::sort(globals.begin(), globals.end());
        result.max_global_operands =
            std::max(result.max_global_operands, static_cast<std::uint32_t>(globals.size()));
        if (globals.size() > request.window_globals) {
            result.window_too_small = true;
        }
        if (!globals.empty()) {
            ++result.global_gates;
        }
        if (!open) {
            open = true;
            group = globals;
            continue;
        }
        std::vector<std::uint32_t> merged;
        std::set_union(
            group.begin(), group.end(), globals.begin(), globals.end(), std::back_inserter(merged));
        if (merged.size() <= request.window_globals) {
            group = std::move(merged);
        } else {
            ++result.passes;
            group = globals;
        }
    }
    if (open) {
        ++result.passes;
    }
    return result;
}

}  // namespace menos::benchmarks
