#include "kernels/dense_kernels.h"

#include <algorithm>
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

// (a.r*b.r - a.i*b.i) + i (a.r*b.i + a.i*b.r): the products and sums of
// std::complex<double>'s operator* in its order, written out so the loops
// below vectorise. operator* (C99 Annex G) computes exactly this and only
// then, if the result is NaN, recovers through __muldc3; for the finite
// amplitudes and matrices the kernels are specified for the two are bitwise
// identical, and the generic kernel keeps operator* so that it stays the
// oracle.
inline std::complex<double> cmul(std::complex<double> a, std::complex<double> b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

// `out = (0, 0); out += p` as the generic kernel starts each output: +0.0 +
// x normalises a -0.0 component of x to +0.0 and changes nothing else.
inline std::complex<double> from_zero(std::complex<double> p) {
    return {0.0 + p.real(), 0.0 + p.imag()};
}

// The two unit-stride inner loops, as free functions so that
// target_clones applies to the loop itself (a lambda handed to parallel_for
// is compiled separately and would not be cloned). "avx2" is dispatched at
// load time where the CPU has it, "default" is baseline x86-64; neither
// enables FMA, and the target is built with -ffp-contract=off, so the vector
// and scalar clones perform the same IEEE multiplies and adds in the same
// order: bitwise identical (measured: the same bytes, ~3x faster, docs
// pending). MSVC has no target_clones; it compiles the plain loop.
#if defined(__GNUC__) && !defined(__clang__)
#define MENOS_KERNEL_CLONES [[gnu::target_clones("avx2", "default")]]
#else
#define MENOS_KERNEL_CLONES
#endif

// lo[i], hi[i] <- m * (lo[i], hi[i]) for i in [0, n): one 1-qubit gate on n
// pairs whose two members are n apart.
MENOS_KERNEL_CLONES
void pair_run(std::complex<double>* lo,
              std::complex<double>* hi,
              std::size_t n,
              std::complex<double> m00,
              std::complex<double> m01,
              std::complex<double> m10,
              std::complex<double> m11) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::complex<double> a0 = lo[i];
        const std::complex<double> a1 = hi[i];
        lo[i] = from_zero(cmul(m00, a0)) + cmul(m01, a1);
        hi[i] = from_zero(cmul(m10, a0)) + cmul(m11, a1);
    }
}

// The same gate for a low operand q (< kRunKernelMinQubit): the runs are
// too short to amortise a call per run, so this walks whole blocks of
// 2^(q+1) amplitudes with a compile-time inner trip count instead.
template <std::uint32_t Q>
MENOS_KERNEL_CLONES void pair_blocks(std::complex<double>* amps,
                                     std::size_t total,
                                     std::complex<double> m00,
                                     std::complex<double> m01,
                                     std::complex<double> m10,
                                     std::complex<double> m11) {
    constexpr std::size_t kHalf = std::size_t{1} << Q;
    for (std::size_t base = 0; base < total; base += 2 * kHalf) {
        std::complex<double>* const lo = amps + base;
        std::complex<double>* const hi = lo + kHalf;
        for (std::size_t i = 0; i < kHalf; ++i) {
            const std::complex<double> a0 = lo[i];
            const std::complex<double> a1 = hi[i];
            lo[i] = from_zero(cmul(m00, a0)) + cmul(m01, a1);
            hi[i] = from_zero(cmul(m10, a0)) + cmul(m11, a1);
        }
    }
}

// Runs shorter than 2^kRunKernelMinQubit are not worth a call each.
inline constexpr std::uint32_t kRunKernelMinQubit = 3;

// a[i] <- scale * a[i] for i in [0, n), as `(+0, +0) += scale * a[i]`.
MENOS_KERNEL_CLONES
void scale_run(std::complex<double>* a, std::size_t n, std::complex<double> scale) {
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = from_zero(cmul(scale, a[i]));
    }
}

// amps[i] <- d[local(i)] * amps[i] for i in [begin, end), local(i) assembled
// from the operand bits of i: the diagonal kernel for a block whose lowest
// operand is below kRunKernelMinQubit.
MENOS_KERNEL_CLONES
void scale_elements(std::complex<double>* amps,
                    std::size_t begin,
                    std::size_t end,
                    const std::uint32_t* operands,
                    std::size_t k,
                    const std::complex<double>* d) {
    for (std::size_t i = begin; i < end; ++i) {
        std::size_t local = 0;
        for (std::size_t j = 0; j < k; ++j) {
            local |= ((i >> operands[j]) & std::size_t{1}) << j;
        }
        amps[i] = from_zero(cmul(d[local], amps[i]));
    }
}

// k = 1. For each pair (i0, i1) differing only in bit q, the generic kernel
// gathers {state[i0], state[i1]} as local indices 0 and 1 and computes
// out[r] = 0 + m[r][0] * a0 + m[r][1] * a1, in that order. This computes the
// same expression in the same order, so the result is bitwise identical;
// only the loop shape differs: the pairs are walked as contiguous runs of
// 2^q indices with bit q clear, each paired with the run 2^q above it, so
// the inner loop has unit stride and vectorises.
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
    std::complex<double>* const amps = state.data();
    if (q < kRunKernelMinQubit && team_size(pairs, num_threads) == 1) {
        switch (q) {
            case 0:
                pair_blocks<0>(amps, state.size(), m00, m01, m10, m11);
                return;
            case 1:
                pair_blocks<1>(amps, state.size(), m00, m01, m10, m11);
                return;
            default:
                pair_blocks<2>(amps, state.size(), m00, m01, m10, m11);
                return;
        }
    }
    // Pair p lives at i0 = insert_zero_bit(p, q) = (p >> q) * 2 * bit + (p & (bit - 1)).
    auto sweep = [&](std::size_t begin, std::size_t end) {
        std::size_t p = begin;
        while (p < end) {
            const std::size_t run_start = p & ~(bit - 1);
            const std::size_t run_end = std::min(run_start + bit, end);
            const std::size_t i0_base = ((p >> q) << (q + 1));
            std::complex<double>* const lo = amps + i0_base;
            std::complex<double>* const hi = lo + bit;
            const std::size_t from = p - run_start;
            const std::size_t to = run_end - run_start;
            pair_run(lo + from, hi + from, to - from, m00, m01, m10, m11);
            p = run_end;
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

// Diagonal matrices (RZ, RZZ, CPhase, Z, S, T, a fused diagonal block ...):
// every off-diagonal entry is exactly (0, 0). The dense kernels above gather
// each 2^k group and add matrix[out][in] * a[in] for every `in`; for a
// diagonal matrix all but one of those products is (0, 0) * a[in]. Adding a
// signed zero to a sum leaves it unchanged unless the sum is itself a zero,
// and the generic sum can never be -0.0 (it starts at +0.0, and
// +0.0 + -0.0 == +0.0), so every output equals
// `(+0.0, +0.0) += matrix[out][out] * a[out]` bit for bit. These kernels
// compute exactly that expression and skip the gather: one complex multiply
// per amplitude instead of 2^k, streamed in index order. Bitwise identical to
// apply_matrix_generic for finite amplitudes (the kernel precondition).
//
// Loop shape: the operand bits are constant over every run of 2^q_min
// consecutive indices (q_min the lowest operand), so the diagonal entry is
// looked up once per run and the run itself is a unit-stride scale.
bool is_diagonal(std::span<const std::complex<double>> matrix, std::size_t dim) {
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            if (r != c &&
                (matrix[r * dim + c].real() != 0.0 || matrix[r * dim + c].imag() != 0.0)) {
                return false;
            }
        }
    }
    return true;
}

void apply_diagonal(std::span<std::complex<double>> state,
                    std::span<const std::complex<double>> matrix,
                    std::span<const std::uint32_t> operands,
                    std::uint32_t num_threads) {
    const std::size_t k = operands.size();
    const std::size_t dim = std::size_t{1} << k;
    std::vector<std::complex<double>> d(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        d[i] = matrix[i * dim + i];
    }
    std::uint32_t q_min = operands[0];
    for (const std::uint32_t q : operands) {
        q_min = std::min(q_min, q);
    }
    const std::size_t run = std::size_t{1} << q_min;
    const std::size_t runs = state.size() >> q_min;
    std::complex<double>* const amps = state.data();
    if (q_min < kRunKernelMinQubit) {
        auto elements = [&](std::size_t begin, std::size_t end) {
            scale_elements(amps, begin, end, operands.data(), k, d.data());
        };
        parallel_for(state.size(), team_size(state.size() >> k, num_threads), elements);
        return;
    }
    auto sweep = [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; ++r) {
            const std::size_t base = r << q_min;
            std::size_t local = 0;
            for (std::size_t j = 0; j < k; ++j) {
                local |= ((base >> operands[j]) & std::size_t{1}) << j;
            }
            scale_run(amps + base, run, d[local]);
        }
    };
    parallel_for(runs, team_size(state.size() >> k, num_threads), sweep);
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
            if (is_diagonal(matrix, 2)) {
                apply_diagonal(state, matrix, operands, num_threads);
            } else {
                apply_matrix_1q(state, matrix, operands[0], num_threads);
            }
            return;
        case 2:
            common::ensure(matrix.size() == 16,
                           "apply_matrix: matrix size does not match 2^|operands|");
            if (is_diagonal(matrix, 4)) {
                apply_diagonal(state, matrix, operands, num_threads);
            } else {
                apply_matrix_2q(state, matrix, operands[0], operands[1], num_threads);
            }
            return;
        default: {
            const std::size_t dim = std::size_t{1} << operands.size();
            common::ensure(matrix.size() == dim * dim,
                           "apply_matrix: matrix size does not match 2^|operands|");
            if (is_diagonal(matrix, dim)) {
                apply_diagonal(state, matrix, operands, num_threads);
            } else {
                apply_matrix_generic(state, matrix, operands, num_threads);
            }
            return;
        }
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

std::complex<double> expectation_of_term(std::span<const std::complex<double>> state,
                                         std::uint64_t x_mask,
                                         std::uint64_t z_mask) {
    const int y_count = std::popcount(x_mask & z_mask);

    std::complex<double> sum{0.0, 0.0};
    for (std::size_t m = 0; m < state.size(); ++m) {
        const std::uint64_t k = m ^ x_mask;
        const bool negative = (std::popcount(z_mask & k) & 1) != 0;
        const std::complex<double> term = std::conj(state[m]) * state[k];
        sum += negative ? -term : term;
    }
    return i_power(y_count) * sum;
}

std::vector<std::complex<double>> expectation_of_diagonal_terms(
    std::span<std::complex<double>> state, std::span<const std::uint64_t> z_masks) {
    const std::size_t total = state.size();
    common::ensure(total > 0 && (total & (total - 1)) == 0,
                   "expectation_of_diagonal_terms: state size must be a power of two");
    // Probabilities p[m] = re^2 + im^2, written over the state in place. The
    // double view of a std::complex<double> array is standard-blessed, and
    // p[m] lands in complex slot m / 2, which was read at iteration m / 2 <= m.
    double* p = reinterpret_cast<double*>(state.data());
    for (std::size_t m = 0; m < total; ++m) {
        const double re = state[m].real();
        const double im = state[m].imag();
        p[m] = re * re + im * im;
    }
    // In-place Walsh-Hadamard transform: afterwards p[z] = sum over m of
    // p[m] * (-1)^popcount(z & m), which is <Z-string z> for every z at once.
    for (std::size_t len = 1; len < total; len <<= 1) {
        for (std::size_t i = 0; i < total; i += len << 1) {
            for (std::size_t j = i; j < i + len; ++j) {
                const double a = p[j];
                const double b = p[j + len];
                p[j] = a + b;
                p[j + len] = a - b;
            }
        }
    }
    std::vector<std::complex<double>> out;
    out.reserve(z_masks.size());
    for (const std::uint64_t z : z_masks) {
        out.emplace_back(p[z], 0.0);
    }
    return out;
}

}  // namespace menos::kernels::cpu
