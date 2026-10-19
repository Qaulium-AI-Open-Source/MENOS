#include "engines/dense/sample.h"

#include <algorithm>
#include <complex>
#include <vector>

#include "common/rng.h"

namespace menos::engines::dense {

using common::Rng;

namespace {

// The sort-then-walk algorithm over `size` basis states whose probabilities are
// probability_of(basis). Shared by the statevector overload (|amplitude|^2) and
// the explicit-probability overload (the density-matrix engine's diagonal), so
// both sum the same doubles in the same order (docs/adr/0016 section 4).
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <typename ProbabilityOf>
std::map<BitString, std::uint64_t> sample_impl(
    std::size_t size,
    ProbabilityOf probability_of,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    std::map<BitString, std::uint64_t> counts;
    if (shots == 0) {
        return counts;
    }

    Rng rng(seed);
    std::vector<double> uniforms(shots);
    for (std::uint64_t i = 0; i < shots; ++i) {
        uniforms[i] = rng.next_uniform();
    }
    std::sort(uniforms.begin(), uniforms.end());

    const auto outcome_for = [&](std::size_t basis) {
        BitString outcome(bitstring_width);
        for (const auto& [qubit, clbit] : measure_targets) {
            if ((basis >> qubit) & std::size_t{1}) {
                outcome.set(clbit, true);
            }
        }
        return outcome;
    };

    double cumulative = 0.0;
    std::size_t shot_index = 0;
    for (std::size_t basis = 0; basis < size && shot_index < uniforms.size(); ++basis) {
        cumulative += probability_of(basis);
        while (shot_index < uniforms.size() && uniforms[shot_index] < cumulative) {
            counts[outcome_for(basis)] += 1;
            ++shot_index;
        }
    }
    // Floating point rounding may leave the cumulative sum just short of
    // 1.0; assign any remaining shots to the last basis state rather than
    // dropping them.
    if (shot_index < uniforms.size() && size != 0) {
        counts[outcome_for(size - 1)] += (uniforms.size() - shot_index);
    }
    return counts;
}

}  // namespace

// shots (a shot count, typically in the thousands) and seed (an arbitrary
// 64-bit RNG seed) are not realistically confusable at any call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::map<BitString, std::uint64_t> sample(
    const StateVector& state,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width) {
    return sample(state.amplitudes(), shots, seed, measure_targets, bitstring_width);
}

std::map<BitString, std::uint64_t> sample(
    std::span<const std::complex<double>> amps,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width) {
    return sample_impl(
        amps.size(),
        [amps](std::size_t basis) { return std::norm(amps[basis]); },
        shots,
        seed,
        measure_targets,
        bitstring_width);
}

std::map<BitString, std::uint64_t> sample(
    std::span<const double> probabilities,
    std::uint64_t shots,
    std::uint64_t seed,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> measure_targets,
    std::uint32_t bitstring_width) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return sample_impl(
        probabilities.size(),
        [probabilities](std::size_t basis) { return probabilities[basis]; },
        shots,
        seed,
        measure_targets,
        bitstring_width);
}

}  // namespace menos::engines::dense
