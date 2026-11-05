#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "menos/circuit.h"

// The pass plan of the chunked executor (Phase 23, docs/adr/0027 section 3.2).
//
// A state of n qubits is 2^(n - L) chunks of 2^L amplitudes: qubits 0 .. L-1
// are LOCAL (the index bits inside a chunk), qubits L .. n-1 are GLOBAL (the
// chunk number). The executor holds a window of 2^W amplitudes in memory,
// W = L + w, so a pass can bring 2^g chunks together for any set G of at most
// w global qubits: the window's slot bits L .. L+g-1 stand for the qubits of
// G, in increasing order, and a gate on local qubits and qubits of G is applied
// to the window by the unchanged kernels with its operands remapped. Gates on
// a chunk's own amplitudes are the same values multiplied by the same matrix
// in the same order whichever window they sit in, so the result is bitwise
// the in-memory engine's.
//
// The greedy rule (section 3.2): a pass opens with the first gate's global
// operands as G and absorbs each following operation while its global operands
// fit in G with at most w qubits in all; otherwise the pass closes and the next
// opens on that gate. Every pass is one read and one write of the whole state,
// so the pass count is what the tier's seconds are made of. Measurements and
// other operations without operands belong to whichever pass is open.
namespace menos::engines::dense_tiered {

// One pass: the operations [first_operation, end_operation) of the circuit,
// whose global operands all lie in G. global_mask has bit (g - L) set for each
// global qubit g in G, so popcount(global_mask) chunks' worth of index bits
// are in play and the pass runs over 2^(n - L - popcount) chunk groups.
struct Pass {
    std::uint64_t global_mask;
    std::size_t first_operation;
    std::size_t end_operation;
};

struct PassPlan {
    std::uint32_t num_qubits = 0;
    // L: a chunk is 2^chunk_qubits amplitudes.
    std::uint32_t chunk_qubits = 0;
    // L + w: the window is 2^window_qubits amplitudes, so a pass may hold at
    // most w = window_qubits - chunk_qubits global qubits.
    std::uint32_t window_qubits = 0;
    // The passes, in circuit order; they partition the operations of a
    // circuit with at least one gate, and are empty for a circuit with none.
    std::vector<Pass> passes;
    // Gates with at least one global operand.
    std::uint64_t global_gates = 0;
    // The most global operands any one gate has: a window whose w is below it
    // cannot run the circuit, and the plan says so.
    std::uint32_t max_global_operands = 0;
    bool window_too_small = false;

    [[nodiscard]] std::uint64_t pass_count() const {
        return passes.size();
    }
};

// The greedy pass plan of `operations` on num_qubits qubits with chunk_qubits
// local qubits and a window of window_qubits qubits (window_qubits >=
// chunk_qubits; both at most num_qubits). Pure: a function of its arguments
// only. A gate with more global operands than the window holds leaves
// window_too_small set and the plan unusable, rather than throwing, so the
// caller can report the reason as an infeasibility.
[[nodiscard]] PassPlan build_pass_plan(std::span<const Operation> operations,
                                       std::uint32_t num_qubits,
                                       std::uint32_t chunk_qubits,
                                       std::uint32_t window_qubits);

// The smallest chunk size the executor prefers when it has the choice:
// 2^24 amplitudes, 256 MiB. Step 1 measured the scratch file's bandwidth flat
// from 256 MiB chunks upward on NTFS and ext4 (docs/adr/0027 section 10.1),
// and a smaller chunk costs more I/O calls for no gain.
inline constexpr std::uint32_t kPreferredMinChunkQubits = 24;

// The pass plan the executor runs with a window of window_qubits qubits
// (window_qubits <= num_qubits): among the chunk sizes L from
// min(kPreferredMinChunkQubits, window_qubits) to window_qubits -- or every
// L from 0 when the window itself is below the preferred minimum -- the one
// with the FEWEST passes, the larger chunk on a tie. Step 1 found this is
// not the largest chunk that fits: a smaller chunk gives the window more
// global qubits, and the crossover circuit at n = 31 takes 18 passes at
// L = 25 against 24 at L = 27 in the same 2^29 window (docs/adr/0027 section
// 10.1, analysis (iv)). Chunks below the preferred minimum are tried only when
// no chunk at or above it fits the gates. needs_chunk_pair asks for w >= 1
// whatever the gates need: an expectation term whose X mask has a global bit
// pairs each chunk with another, so the window must hold two. nullopt when no
// L fits every gate's global operands.
[[nodiscard]] std::optional<PassPlan> choose_pass_plan(std::span<const Operation> operations,
                                                       std::uint32_t num_qubits,
                                                       std::uint32_t window_qubits,
                                                       bool needs_chunk_pair);

}  // namespace menos::engines::dense_tiered
