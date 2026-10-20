#include "engines/dense/cumulative_distribution.h"

#include <algorithm>
#include <complex>
#include <utility>

#include "common/checked_math.h"
#include "common/memory_bounds.h"
#include "common/rng.h"

namespace menos::engines::dense {

template <typename ProbabilityOf>
void CumulativeDistribution::build(std::size_t size, ProbabilityOf probability_of) {
    cumulative_.reserve(size);
    double cumulative = 0.0;
    for (std::size_t basis = 0; basis < size; ++basis) {
        cumulative += probability_of(basis);
        cumulative_.push_back(cumulative);
    }
    charge_.set(common::buffer_bytes(cumulative_));
}

CumulativeDistribution::CumulativeDistribution(
    const StateVector& state,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width)
    : CumulativeDistribution(state.amplitudes(), measure_targets, bitstring_width) {}

CumulativeDistribution::CumulativeDistribution(
    std::span<const std::complex<double>> amplitudes,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width)
    : measure_targets_(measure_targets.begin(), measure_targets.end()),
      bitstring_width_(bitstring_width) {
    build(amplitudes.size(),
          [amplitudes](std::size_t basis) { return std::norm(amplitudes[basis]); });
}

CumulativeDistribution::CumulativeDistribution(
    std::span<const double> probabilities,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width)
    : measure_targets_(measure_targets.begin(), measure_targets.end()),
      bitstring_width_(bitstring_width) {
    build(probabilities.size(),
          [probabilities](std::size_t basis) { return probabilities[basis]; });
}

CumulativeDistribution::CumulativeDistribution(
    std::vector<double>&& probabilities,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width)
    : cumulative_(std::move(probabilities)),
      measure_targets_(measure_targets.begin(), measure_targets.end()),
      bitstring_width_(bitstring_width) {
    // The buffer was charged by its producer until now; it is this
    // sampler's from here on.
    charge_.set(common::buffer_bytes(cumulative_));
    double cumulative = 0.0;
    for (double& entry : cumulative_) {
        cumulative += entry;
        entry = cumulative;
    }
}

namespace {

BitString outcome_of(std::size_t index,
                     std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
                     std::uint32_t bitstring_width) {
    BitString outcome(bitstring_width);
    for (const auto& [qubit, clbit] : measure_targets) {
        if ((index >> qubit) & std::size_t{1}) {
            outcome.set(clbit, true);
        }
    }
    return outcome;
}

}  // namespace

BitString CumulativeDistribution::sample_one(double uniform) const {
    const auto it = std::upper_bound(cumulative_.begin(), cumulative_.end(), uniform);
    // Floating point rounding may leave the cumulative sum just short of
    // 1.0; fall back to the last outcome rather than reading past the end
    // (mirrors engines::dense::sample()'s equivalent fallback).
    const std::size_t index = (it == cumulative_.end())
                                  ? cumulative_.size() - 1
                                  : static_cast<std::size_t>(it - cumulative_.begin());
    return outcome_of(index, measure_targets_, bitstring_width_);
}

BitString CumulativeDistribution::sample_once(
    std::span<const std::complex<double>> amplitudes,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width,
    common::Rng& rng) {
    const double uniform = common::Rng::word_to_uniform(rng.next_word());
    double cumulative = 0.0;
    std::size_t index = amplitudes.size() - 1;
    for (std::size_t basis = 0; basis < amplitudes.size(); ++basis) {
        cumulative += std::norm(amplitudes[basis]);
        if (uniform < cumulative) {
            index = basis;
            break;
        }
    }
    return outcome_of(index, measure_targets, bitstring_width);
}

std::uint64_t CumulativeDistribution::heap_bytes(std::uint64_t size, std::uint64_t num_targets) {
    return common::checked_add(
        common::checked_add(common::heap_block_bytes(sizeof(CumulativeDistribution)),
                            common::heap_block_bytes(common::checked_mul(size, sizeof(double)))),
        common::heap_block_bytes(
            common::checked_mul(num_targets, sizeof(std::pair<std::uint32_t, std::uint32_t>))));
}

BitString CumulativeDistribution::sample_one(common::Rng& rng) const {
    return sample_one(common::Rng::word_to_uniform(rng.next_word()));
}

}  // namespace menos::engines::dense
