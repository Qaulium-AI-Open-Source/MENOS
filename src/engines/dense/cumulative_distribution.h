#pragma once

#include <complex>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "engines/dense/state_vector.h"
#include "menos/bitstring.h"
#include "menos/simulation_engine.h"

namespace menos::engines::dense {

// A region's outcome distribution, precomputed once (O(2^m)) so many
// independent draws can each be resolved in O(log distinct outcomes) via
// binary search, without re-sorting or otherwise correlating draws across
// shots or across regions. Used only for the multi-region factorized
// counts-combination path (src/runtime/combine_counts.h) -- the
// single-region case keeps engines::dense::sample()'s existing sort-then-
// walk algorithm, which is exact there but would introduce spurious
// cross-region correlation if reused for combination (docs/adr/0005).
// Implements RegionCountsSampler (Phase 4) so DenseEngine::execute() can
// hand it back through the engine-agnostic multi-region path unchanged.
//
// Holds only the cumulative sums, 8 bytes per basis state: a drawn index is
// turned into its BitString when it is drawn (docs/adr/0025 section 3.3).
// Until Phase 21 every basis state's BitString was built up front, about 48
// bytes and one heap allocation each.
class CumulativeDistribution : public RegionCountsSampler {
public:
    CumulativeDistribution(const StateVector& state,
                           std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
                           std::uint32_t bitstring_width);

    // The same distribution over amplitudes held outside a StateVector (the
    // CUDA backend's host copy).
    CumulativeDistribution(std::span<const std::complex<double>> amplitudes,
                           std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
                           std::uint32_t bitstring_width);

    // The same distribution over explicit outcome probabilities:
    // probabilities[b] is the probability of basis state b (the density-matrix
    // engine's diagonal, docs/adr/0016 section 4). Built by the same code as
    // the statevector constructor.
    CumulativeDistribution(std::span<const double> probabilities,
                           std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
                           std::uint32_t bitstring_width);

    // uniform must be in [0, 1). Returns the outcome whose cumulative
    // probability interval contains it.
    [[nodiscard]] BitString sample_one(double uniform) const;

    // RegionCountsSampler: draws exactly one word from rng and converts
    // it to [0, 1) via the same formula common::Rng::next_uniform() uses
    // internally, so this is exactly the double-based overload above
    // reached through a shared conversion, not a second algorithm.
    [[nodiscard]] BitString sample_one(common::Rng& rng) const override;

    // One draw from the distribution of amplitudes without building it: one
    // cumulative pass, stopping at the drawn outcome. Draws one word from rng
    // and returns exactly what constructing a CumulativeDistribution over the
    // same amplitudes and calling sample_one(rng) would, in O(1) memory: the
    // same doubles are summed in the same order, and the first index whose
    // running sum exceeds the uniform is the index upper_bound finds (the
    // trajectory engine, which draws one shot per trajectory; docs/adr/0025).
    [[nodiscard]] static BitString sample_once(
        std::span<const std::complex<double>> amplitudes,
        std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
        std::uint32_t bitstring_width,
        common::Rng& rng);

private:
    template <typename ProbabilityOf>
    void build(std::size_t size, ProbabilityOf probability_of);

    std::vector<double> cumulative_;  // ascending, back() ~= 1.0
    std::vector<std::pair<std::uint32_t, std::uint32_t>> measure_targets_;
    std::uint32_t bitstring_width_;
};

}  // namespace menos::engines::dense
