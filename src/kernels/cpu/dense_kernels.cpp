#include "kernels/dense_kernels.h"

#include <array>
#include <bit>
#include <cmath>
#include <vector>

#include "common/error.h"
#include "kernels/cpu/thread_pool.h"

namespace menos::kernels::cpu {

void initialize(std::span<std::complex<double>> state, double global_phase) {
    const std::complex<double> phase = std::exp(std::complex<double>{0.0, global_phase});
    for (std::complex<double>& a : state) {
        a = {0.0, 0.0};
    }
    state[0] = phase;
}

namespace {

// The team for a sweep over `groups` groups: the whole of num_threads from the
// measured cut-off on, one thread below it.
std::uint32_t team_size(std::size_t groups, std::uint32_t num_threads) {
    return groups >= kParallelMinGroups ? num_threads : 1;
}

// Inserts a zero bit at position `pos` of `i`, shifting the bits at and above
// it up by one.
// An index and a bit position: distinct roles, named at every call.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::size_t insert_zero_bit(std::size_t i, std::uint32_t pos) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const std::size_t low = (std::size_t{1} << pos) - 1;
    return ((i & ~low) << 1) | (i & low);
}

// k = 1. For each pair (i0, i1) differing only in bit q, the generic kernel
// gathers {state[i0], state[i1]} as local indices 0 and 1 and computes
// out[r] = 0 + m[r][0] * a0 + m[r][1] * a1, in that order. This computes the
// same expression in the same order, so the result is bitwise identical;
// only the index arithmetic (one bit insertion instead of an n-step loop
// and two heap vectors) differs.
// q is the operand and num_threads the team cap; both named at the one call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void apply_matrix_1q(std::span<std::complex<double>> state,
                     std::span<const std::complex<double>> matrix,
                     std::uint32_t q,
                     std::uint32_t num_threads) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const std::size_t bit = std::size_t{1} << q;
    const std::complex<double> m00 = matrix[0];
    const std::complex<double> m01 = matrix[1];
    const std::complex<double> m10 = matrix[2];
    const std::complex<double> m11 = matrix[3];
    const std::size_t pairs = state.size() >> 1;
    auto sweep = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const std::size_t i0 = insert_zero_bit(i, q);
            const std::size_t i1 = i0 | bit;
            const std::complex<double> a0 = state[i0];
            const std::complex<double> a1 = state[i1];
            std::complex<double> out0{0.0, 0.0};
            out0 += m00 * a0;
            out0 += m01 * a1;
            std::complex<double> out1{0.0, 0.0};
            out1 += m10 * a0;
            out1 += m11 * a1;
            state[i0] = out0;
            state[i1] = out1;
        }
    };
    parallel_for(pairs, team_size(pairs, num_threads), sweep);
}

// k = 2. Local index bit j is operand j (operands[0] is local bit 0), exactly
// as in the generic kernel, so a reversed or non-adjacent operand pair keeps
// the generic kernel's local ordering. Zero bits are inserted at the lower
// operand position first, then the higher, which leaves both positions clear.
// Each output sums in = 0..3 in order from zero: bitwise identical.
// q0 and q1 keep operands[0] and operands[1]'s order, which is the kernel's
// local-bit order; num_threads is the team cap. One call site passes them in
// that order.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void apply_matrix_2q(std::span<std::complex<double>> state,
                     std::span<const std::complex<double>> matrix,
                     std::uint32_t q0,
                     std::uint32_t q1,
                     std::uint32_t num_threads) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const std::uint32_t lo = q0 < q1 ? q0 : q1;
    const std::uint32_t hi = q0 < q1 ? q1 : q0;
    const std::size_t b0 = std::size_t{1} << q0;
    const std::size_t b1 = std::size_t{1} << q1;
    const std::size_t groups = state.size() >> 2;
    auto sweep = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const std::size_t base = insert_zero_bit(insert_zero_bit(i, lo), hi);
            const std::array<std::size_t, 4> idx = {base, base | b0, base | b1, base | b0 | b1};
            const std::array<std::complex<double>, 4> a = {
                state[idx[0]], state[idx[1]], state[idx[2]], state[idx[3]]};
            for (std::size_t out = 0; out < 4; ++out) {
                std::complex<double> sum{0.0, 0.0};
                for (std::size_t in = 0; in < 4; ++in) {
                    sum += matrix[out * 4 + in] * a[in];
                }
                state[idx[out]] = sum;
            }
        }
    };
    parallel_for(groups, team_size(groups, num_threads), sweep);
}

}  // namespace

void apply_matrix(std::span<std::complex<double>> state,
                  std::span<const std::complex<double>> matrix,
                  std::span<const std::uint32_t> operands,
                  std::uint32_t num_threads) {
    common::ensure(num_threads >= 1, "apply_matrix: num_threads must be at least 1");
    switch (operands.size()) {
        case 1:
            common::ensure(matrix.size() == 4,
                           "apply_matrix: matrix size does not match 2^|operands|");
            apply_matrix_1q(state, matrix, operands[0], num_threads);
            return;
        case 2:
            common::ensure(matrix.size() == 16,
                           "apply_matrix: matrix size does not match 2^|operands|");
            apply_matrix_2q(state, matrix, operands[0], operands[1], num_threads);
            return;
        default:
            apply_matrix_generic(state, matrix, operands, num_threads);
            return;
    }
}

void apply_matrix_generic(std::span<std::complex<double>> state,
                          std::span<const std::complex<double>> matrix,
                          std::span<const std::uint32_t> operands,
                          std::uint32_t num_threads) {
    const std::size_t k = operands.size();
    const std::size_t dim = std::size_t{1} << k;
    common::ensure(matrix.size() == dim * dim,
                   "apply_matrix: matrix size does not match 2^|operands|");

    const std::size_t total = state.size();
    const int n = std::countr_zero(total);

    std::uint64_t operand_mask = 0;
    for (std::uint32_t q : operands) {
        operand_mask |= (std::uint64_t{1} << q);
    }

    const std::size_t n_outer = total >> k;
    // Each chunk owns its gather scratch (2^k amplitudes and indices, k <= 5).
    // Groups are disjoint, so no two chunks touch the same amplitude.
    auto sweep = [&](std::size_t begin, std::size_t end) {
        std::vector<std::size_t> full_indices(dim);
        std::vector<std::complex<double>> gathered(dim);
        for (std::size_t compact = begin; compact < end; ++compact) {
            // Expand compact's (n-k) bits into an n-bit base index, leaving
            // zeros at the operand positions.
            std::size_t base = 0;
            int bit_src = 0;
            for (int p = 0; p < n; ++p) {
                if ((operand_mask >> p) & std::uint64_t{1}) {
                    continue;
                }
                base |= ((compact >> bit_src) & std::size_t{1}) << p;
                ++bit_src;
            }

            // Set operand bits per each local sub-index and gather.
            for (std::size_t local = 0; local < dim; ++local) {
                std::size_t idx = base;
                for (std::size_t j = 0; j < k; ++j) {
                    if ((local >> j) & std::size_t{1}) {
                        idx |= (std::size_t{1} << operands[j]);
                    }
                }
                full_indices[local] = idx;
                gathered[local] = state[idx];
            }

            // matrix * gathered, scattered back to the same indices.
            for (std::size_t out = 0; out < dim; ++out) {
                std::complex<double> sum{0.0, 0.0};
                for (std::size_t in = 0; in < dim; ++in) {
                    sum += matrix[out * dim + in] * gathered[in];
                }
                state[full_indices[out]] = sum;
            }
        }
    };
    parallel_for(n_outer, team_size(n_outer, num_threads), sweep);
}

namespace {

// i^k for k in {0,1,2,3} (k taken mod 4 by the caller).
std::complex<double> i_power(int k) {
    switch (k & 3) {
        case 0:
            return {1.0, 0.0};
        case 1:
            return {0.0, 1.0};
        case 2:
            return {-1.0, 0.0};
        default:
            return {0.0, -1.0};
    }
}

}  // namespace

void expectation_of_term_accumulate(std::span<const std::complex<double>> chunk,
                                    std::span<const std::complex<double>> partner,
                                    std::uint64_t x_local,
                                    std::uint64_t z_local,
                                    bool global_negative,
                                    std::complex<double>& acc) {
    for (std::size_t j = 0; j < chunk.size(); ++j) {
        const std::uint64_t k = j ^ x_local;
        const bool negative = ((std::popcount(z_local & k) & 1) != 0) != global_negative;
        const std::complex<double> term = std::conj(chunk[j]) * partner[k];
        acc += negative ? -term : term;
    }
}

std::complex<double> expectation_term_phase(std::uint64_t x_mask, std::uint64_t z_mask) {
    return i_power(std::popcount(x_mask & z_mask));
}

std::complex<double> expectation_of_term(std::span<const std::complex<double>> state,
                                         std::uint64_t x_mask,
                                         std::uint64_t z_mask) {
    // The whole state is one chunk: no high bits, so no global sign.
    std::complex<double> sum{0.0, 0.0};
    expectation_of_term_accumulate(state, state, x_mask, z_mask, /*global_negative=*/false, sum);
    return expectation_term_phase(x_mask, z_mask) * sum;
}

}  // namespace menos::kernels::cpu
