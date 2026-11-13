// Phase 23 (docs/adr/0027 section 3.4): the two outputs the chunked executor
// produces without the whole state in memory must be BITWISE those of the
// in-memory engine. The expectation is accumulated chunk by chunk through
// expectation_of_term_accumulate, and the counts are drawn through
// sample_streamed; both are compared against the whole-span functions by
// their bytes, for every chunk size from one amplitude to the whole state.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "engines/dense/sample.h"
#include "kernels/dense_kernels.h"
#include "menos/bitstring.h"

namespace menos {
namespace {

using Complex = std::complex<double>;

std::vector<Complex> random_state(std::uint32_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> v(std::size_t{1} << n);
    double norm = 0.0;
    for (Complex& c : v) {
        c = Complex{dist(rng), dist(rng)};
        norm += std::norm(c);
    }
    const double scale = 1.0 / std::sqrt(norm);
    for (Complex& c : v) {
        c *= scale;
    }
    return v;
}

bool same_bits(Complex a, Complex b) {
    return std::bit_cast<std::uint64_t>(a.real()) == std::bit_cast<std::uint64_t>(b.real()) &&
           std::bit_cast<std::uint64_t>(a.imag()) == std::bit_cast<std::uint64_t>(b.imag());
}

// expectation_of_term over the state, computed the way the chunked executor
// computes it: chunks of 2^local_qubits amplitudes in increasing order, each
// paired with the chunk the X mask's high bits select, the Z mask's high
// bits folded into one sign per chunk, the phase applied once at the end.
Complex chunked_expectation(std::span<const Complex> state,
                            std::uint32_t local_qubits,
                            std::uint64_t x_mask,
                            std::uint64_t z_mask) {
    const std::uint64_t chunk_size = std::uint64_t{1} << local_qubits;
    const std::uint64_t local_mask = chunk_size - 1;
    const std::uint64_t chunks = state.size() / chunk_size;
    const std::uint64_t x_high = x_mask >> local_qubits;
    const std::uint64_t z_high = z_mask >> local_qubits;
    Complex acc{0.0, 0.0};
    for (std::uint64_t c = 0; c < chunks; ++c) {
        const std::uint64_t partner = c ^ x_high;
        const bool global_negative = (std::popcount(z_high & partner) & 1) != 0;
        kernels::cpu::expectation_of_term_accumulate(
            state.subspan(c * chunk_size, chunk_size),
            state.subspan(partner * chunk_size, chunk_size),
            x_mask & local_mask,
            z_mask & local_mask,
            global_negative,
            acc);
    }
    return kernels::cpu::expectation_term_phase(x_mask, z_mask) * acc;
}

TEST(ChunkedKernelsTest, ChunkedExpectationIsBitwiseTheWholeSpanOneForEveryChunkSize) {
    std::mt19937_64 rng(2301);
    for (std::uint32_t n = 1; n <= 12; ++n) {
        const std::vector<Complex> state = random_state(n, rng);
        const std::uint64_t all = (std::uint64_t{1} << n) - 1;
        // Terms with no X bit, one X bit (low and high), every X bit, and
        // random masks: the Z mask alone, the X mask alone, and both (Y's).
        std::vector<std::pair<std::uint64_t, std::uint64_t>> terms = {
            {0, 0},
            {0, all},
            {1, 0},
            {std::uint64_t{1} << (n - 1), 0},
            {std::uint64_t{1} << (n - 1), std::uint64_t{1} << (n - 1)},
            {all, 0},
            {all, all},
        };
        std::uniform_int_distribution<std::uint64_t> pick(0, all);
        for (int i = 0; i < 8; ++i) {
            terms.emplace_back(pick(rng), pick(rng));
        }
        for (const auto& [x, z] : terms) {
            const Complex whole = kernels::cpu::expectation_of_term(state, x, z);
            for (std::uint32_t local = 0; local <= n; ++local) {
                const Complex chunked = chunked_expectation(state, local, x, z);
                EXPECT_TRUE(same_bits(whole, chunked))
                    << "n=" << n << " L=" << local << " x=" << x << " z=" << z << " whole=" << whole
                    << " chunked=" << chunked;
            }
        }
    }
}

TEST(ChunkedKernelsTest, StreamedCountsAreExactlyTheSpanCountsForEveryChunkSize) {
    std::mt19937_64 rng(2302);
    for (std::uint32_t n = 1; n <= 10; ++n) {
        const std::vector<Complex> state = random_state(n, rng);
        // Every qubit measured to its own clbit, and the high half measured
        // to reversed clbits.
        std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> target_sets;
        target_sets.emplace_back();
        for (std::uint32_t q = 0; q < n; ++q) {
            target_sets.back().emplace_back(q, q);
        }
        target_sets.emplace_back();
        for (std::uint32_t q = n / 2; q < n; ++q) {
            target_sets.back().emplace_back(q, n - 1 - q);
        }
        for (const auto& targets : target_sets) {
            for (const std::uint64_t shots :
                 {std::uint64_t{1}, std::uint64_t{7}, std::uint64_t{1000}}) {
                for (const std::uint64_t seed : {std::uint64_t{1}, std::uint64_t{99}}) {
                    const std::map<BitString, std::uint64_t> expected = engines::dense::sample(
                        std::span<const Complex>(state), shots, seed, targets, n);
                    for (std::uint32_t local = 0; local <= n; ++local) {
                        const std::uint64_t chunk = std::uint64_t{1} << local;
                        std::uint64_t reads = 0;
                        std::uint64_t last_chunk = 0;
                        const std::function<std::span<const Complex>(std::uint64_t)> read_chunk =
                            [&](std::uint64_t c) {
                                // Chunks are asked for in increasing order, each once.
                                if (reads > 0) {
                                    EXPECT_EQ(c, last_chunk + 1);
                                }
                                last_chunk = c;
                                ++reads;
                                return std::span<const Complex>(state).subspan(c * chunk, chunk);
                            };
                        const std::map<BitString, std::uint64_t> streamed =
                            engines::dense::sample_streamed(
                                state.size(), chunk, read_chunk, shots, seed, targets, n);
                        EXPECT_EQ(streamed, expected) << "n=" << n << " L=" << local
                                                      << " shots=" << shots << " seed=" << seed;
                        EXPECT_LE(reads, state.size() / chunk);
                    }
                }
            }
        }
    }
}

TEST(ChunkedKernelsTest, StreamedCountsWithNoShotsReadNothing) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> targets = {{0, 0}};
    const std::function<std::span<const Complex>(std::uint64_t)> read_chunk = [](std::uint64_t) {
        ADD_FAILURE() << "no shots, so no chunk should be read";
        return std::span<const Complex>();
    };
    const auto counts =
        engines::dense::sample_streamed(1024, 64, read_chunk, /*shots=*/0, /*seed=*/1, targets, 1);
    EXPECT_TRUE(counts.empty());
}

}  // namespace
}  // namespace menos
