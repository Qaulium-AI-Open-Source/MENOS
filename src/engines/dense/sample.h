#pragma once

#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <utility>

#include "engines/dense/state_vector.h"
#include "menos/bitstring.h"

namespace menos::engines::dense {

// Draws `shots` uniforms from a seeded RNG, sorts them, and walks the
// state once accumulating probability -- no extra array of size 2^n.
// measure_targets is the already-resolved (last-write-wins applied)
// qubit-to-classical-bit mapping; bit `clbit` of a returned key is set
// from bit `qubit` of the sampled basis index. bitstring_width is the
// width of every returned key (docs/BUILD_PLAN.md Phase 3: a global
// clbit index, not a region-local clbit count, since MeasureOp::clbit is
// never remapped -- see include/menos/plan.h's Region comment). Returned
// counts are ordered by key (std::map, BitString's numeric ordering), so
// output is deterministic. This sort-then-walk algorithm is exact for a
// single region with no cross-region combination; see
// src/engines/dense/cumulative_distribution.h and
// src/runtime/combine_counts.h for the multi-region factorized case,
// which cannot reuse this algorithm directly (pairing two independently
// sorted per-region sequences by shared shot index introduces spurious
// correlation -- see docs/adr/0005).
[[nodiscard]] std::map<BitString, std::uint64_t> sample(
    const StateVector& state,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width);

// The same, over amplitudes held outside a StateVector: the CUDA backend's
// host copy, which is the result's own statevector buffer (docs/adr/0025).
[[nodiscard]] std::map<BitString, std::uint64_t> sample(
    std::span<const std::complex<double>> amplitudes,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width);

// The same algorithm over explicit outcome probabilities: probabilities[b] is
// the probability of basis state b (the density-matrix engine's diagonal,
// docs/adr/0016 section 4). Both overloads share one implementation, so they
// sum the same values in the same order.
[[nodiscard]] std::map<BitString, std::uint64_t> sample(
    std::span<const double> probabilities,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width);

// The same algorithm over a state that is not in memory all at once (Phase
// 23, docs/adr/0027 section 3.4): num_amplitudes amplitudes in chunks of
// chunk_amplitudes (a power of two dividing num_amplitudes), where read_chunk(c)
// returns the amplitudes of chunk c, basis states c * chunk_amplitudes onward,
// valid until the next call. The walk visits the basis states in increasing
// order and asks for each chunk once, in order, only as far as the last shot
// needs; it sums the same |amplitude|^2 in the same order as the span
// overload, so the counts are bit for bit those of sample(span) over the whole
// state. The histogram and the uniforms are charged as the other overloads
// charge them.
[[nodiscard]] std::map<BitString, std::uint64_t> sample_streamed(
    std::uint64_t num_amplitudes,
    std::uint64_t chunk_amplitudes,
    const std::function<std::span<const std::complex<double>>(std::uint64_t)>& read_chunk,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width);

// The heap bytes either sample() overload allocates for `shots` draws whose
// outcomes are measured on `measured_qubits` qubits, into bitstring_width-bit
// keys: the sorted uniforms (8 bytes per shot) and the histogram it returns,
// which outlives the call (docs/adr/0025 section 3.4).
[[nodiscard]] std::uint64_t sample_bytes(std::uint64_t shots,
                                         std::uint32_t measured_qubits,
                                         std::uint32_t bitstring_width);

}  // namespace menos::engines::dense
