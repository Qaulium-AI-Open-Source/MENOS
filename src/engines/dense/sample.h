#pragma once

#include <complex>
#include <cstdint>
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

}  // namespace menos::engines::dense
