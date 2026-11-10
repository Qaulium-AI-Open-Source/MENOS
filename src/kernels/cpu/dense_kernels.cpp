#include "kernels/dense_kernels.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <vector>

#if defined(_MSC_VER) && defined(_M_X64)
#include <immintrin.h>
#include <intrin.h>
#endif

#include "common/error.h"
#include "kernels/cpu/dense_kernel_table.h"
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

// ---------------------------------------------------------------------------
// The baseline table: plain x86-64, scalar. These are the loops the kernels
// ran before item 8 (docs/adr/0028 section 19.1), unchanged in arithmetic;
// they serve every CPU without AVX2, and the tests force them on the AVX2
// machines that verify the build so both tables stay pinned to the oracle.
// ---------------------------------------------------------------------------

std::complex<double>* as_complex(double* p) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<std::complex<double>*>(p);
}

std::complex<double> load_complex(const double* p) {
    return {p[0], p[1]};
}

// (a.r*b.r - a.i*b.i) + i (a.r*b.i + a.i*b.r): the products and sums of
// std::complex<double>'s operator* in its order. operator* (C99 Annex G)
// computes exactly this and only then, if the result is NaN, recovers through
// __muldc3; for the finite amplitudes and matrices the kernels are specified
// for the two are bitwise identical, and the generic kernel keeps operator*
// so that it stays the oracle.
inline std::complex<double> cmul(std::complex<double> a, std::complex<double> b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

// `out = (0, 0); out += p` as the generic kernel starts each output: +0.0 +
// x normalises a -0.0 component of x to +0.0 and changes nothing else.
inline std::complex<double> from_zero(std::complex<double> p) {
    return {0.0 + p.real(), 0.0 + p.imag()};
}

// The (lo, hi) pointers and the matrix follow the fixed row-major order of
// every gate kernel in this file.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void baseline_pair_run(double* lo_d, double* hi_d, std::size_t n, const double* m) {
    std::complex<double>* const lo = as_complex(lo_d);
    std::complex<double>* const hi = as_complex(hi_d);
    const std::complex<double> m00 = load_complex(m);
    const std::complex<double> m01 = load_complex(m + 2);
    const std::complex<double> m10 = load_complex(m + 4);
    const std::complex<double> m11 = load_complex(m + 6);
    for (std::size_t i = 0; i < n; ++i) {
        const std::complex<double> a0 = lo[i];
        const std::complex<double> a1 = hi[i];
        lo[i] = from_zero(cmul(m00, a0)) + cmul(m01, a1);
        hi[i] = from_zero(cmul(m10, a0)) + cmul(m11, a1);
    }
}

// Whole blocks of 2^(Q+1) amplitudes with a compile-time inner trip count.
template <std::uint32_t Q>
void baseline_pair_blocks_q(double* amps_d, std::size_t total, const double* m) {
    std::complex<double>* const amps = as_complex(amps_d);
    const std::complex<double> m00 = load_complex(m);
    const std::complex<double> m01 = load_complex(m + 2);
    const std::complex<double> m10 = load_complex(m + 4);
    const std::complex<double> m11 = load_complex(m + 6);
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

void baseline_pair_blocks(double* amps, std::size_t total, std::uint32_t q, const double* m) {
    switch (q) {
        case 0:
            baseline_pair_blocks_q<0>(amps, total, m);
            return;
        case 1:
            baseline_pair_blocks_q<1>(amps, total, m);
            return;
        default:
            baseline_pair_blocks_q<2>(amps, total, m);
            return;
    }
}

void baseline_scale_run(double* a_d, std::size_t n, const double* s_d) {
    std::complex<double>* const a = as_complex(a_d);
    const std::complex<double> scale = load_complex(s_d);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = from_zero(cmul(scale, a[i]));
    }
}

void baseline_scale_elements(double* amps_d,
                             std::size_t begin,
                             std::size_t end,
                             const std::uint32_t* operands,
                             std::size_t k,
                             const double* d_d) {
    std::complex<double>* const amps = as_complex(amps_d);
    for (std::size_t i = begin; i < end; ++i) {
        std::size_t local = 0;
        for (std::size_t j = 0; j < k; ++j) {
            local |= ((i >> operands[j]) & std::size_t{1}) << j;
        }
        amps[i] = from_zero(cmul(load_complex(d_d + 2 * local), amps[i]));
    }
}

void baseline_two_qubit(double* amps_d,
                        std::size_t begin,
                        std::size_t end,
                        std::uint32_t q0,
                        std::uint32_t q1,
                        const double* m) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    std::complex<double>* const state = as_complex(amps_d);
    const std::uint32_t lo = q0 < q1 ? q0 : q1;
    const std::uint32_t hi = q0 < q1 ? q1 : q0;
    const std::size_t b0 = std::size_t{1} << q0;
    const std::size_t b1 = std::size_t{1} << q1;
    std::array<std::complex<double>, 16> matrix{};
    for (std::size_t e = 0; e < 16; ++e) {
        matrix[e] = load_complex(m + 2 * e);
    }
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
}

constexpr DenseKernelTable kBaselineTable{
    .pair_run = baseline_pair_run,
    .pair_blocks = baseline_pair_blocks,
    .scale_run = baseline_scale_run,
    .scale_elements = baseline_scale_elements,
    .two_qubit = baseline_two_qubit,
};

// A test's forced table, or nullptr for the detected one.
std::atomic<const DenseKernelTable*> g_forced_table{nullptr};

const DenseKernelTable* detected_table() {
    static const DenseKernelTable* const table =
        (avx2_dense_kernels() != nullptr && cpu_supports_avx2()) ? avx2_dense_kernels()
                                                                 : &kBaselineTable;
    return table;
}

const double* as_doubles(const std::complex<double>* p) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const double*>(p);
}

double* as_doubles(std::complex<double>* p) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<double*>(p);
}

// Runs shorter than 2^kRunKernelMinQubit are not worth a call each.
inline constexpr std::uint32_t kRunKernelMinQubit = 3;

// k = 1. For each pair (i0, i1) differing only in bit q, the generic kernel
// gathers {state[i0], state[i1]} as local indices 0 and 1 and computes
// out[r] = 0 + m[r][0] * a0 + m[r][1] * a1, in that order. This computes the
// same expression in the same order, so the result is bitwise identical;
// only the loop shape differs: the pairs are walked as contiguous runs of
// 2^q indices with bit q clear, each paired with the run 2^q above it, so
// the inner loop has unit stride.
// q is the operand and num_threads the team cap; both named at the one call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void apply_matrix_1q(std::span<std::complex<double>> state,
                     std::span<const std::complex<double>> matrix,
                     std::uint32_t q,
                     std::uint32_t num_threads) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const DenseKernelTable& kernels = active_dense_kernels();
    const std::size_t bit = std::size_t{1} << q;
    const double* const m = as_doubles(matrix.data());
    const std::size_t pairs = state.size() >> 1;
    std::complex<double>* const amps = state.data();
    if (q < kRunKernelMinQubit && team_size(pairs, num_threads) == 1) {
        kernels.pair_blocks(as_doubles(amps), state.size(), q, m);
        return;
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
            kernels.pair_run(as_doubles(lo + from), as_doubles(hi + from), to - from, m);
            p = run_end;
        }
    };
    parallel_for(pairs, team_size(pairs, num_threads), sweep);
}

// k = 2. Local index bit j is operand j (operands[0] is local bit 0), exactly
// as in the generic kernel, so a reversed or non-adjacent operand pair keeps
// the generic kernel's local ordering. Each output sums in = 0..3 in order
// from zero: bitwise identical.
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
    const DenseKernelTable& kernels = active_dense_kernels();
    const std::size_t groups = state.size() >> 2;
    double* const amps = as_doubles(state.data());
    const double* const m = as_doubles(matrix.data());
    auto sweep = [&](std::size_t begin, std::size_t end) {
        kernels.two_qubit(amps, begin, end, q0, q1, m);
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
    const DenseKernelTable& kernels = active_dense_kernels();
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
    double* const amps = as_doubles(state.data());
    const double* const entries = as_doubles(d.data());
    if (q_min < kRunKernelMinQubit) {
        auto elements = [&](std::size_t begin, std::size_t end) {
            kernels.scale_elements(amps, begin, end, operands.data(), k, entries);
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
            kernels.scale_run(amps + 2 * base, run, entries + 2 * local);
        }
    };
    parallel_for(runs, team_size(state.size() >> k, num_threads), sweep);
}

}  // namespace

const DenseKernelTable& baseline_dense_kernels() {
    return kBaselineTable;
}

#if !defined(MENOS_HAVE_AVX2_KERNELS)
const DenseKernelTable* avx2_dense_kernels() {
    return nullptr;
}
#endif

bool cpu_supports_avx2() {
#if defined(_MSC_VER) && defined(_M_X64)
    // CPUID.1:ECX.OSXSAVE[27] and AVX[28], XCR0 bits 1 and 2 (the OS saves
    // XMM and YMM state), then CPUID.(7,0):EBX.AVX2[5].
    std::array<int, 4> r{};
    __cpuid(r.data(), 0);
    if (r[0] < 7) {
        return false;
    }
    __cpuid(r.data(), 1);
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    const bool avx = (r[2] & (1 << 28)) != 0;
    if (!osxsave || !avx || (_xgetbv(0) & 0x6U) != 0x6U) {
        return false;
    }
    __cpuidex(r.data(), 7, 0);
    return (r[1] & (1 << 5)) != 0;
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    // The compiler runtime's CPU model, which reports AVX2 only when the OS
    // has enabled YMM state (it reads XCR0 itself).
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}

const DenseKernelTable& active_dense_kernels() {
    const DenseKernelTable* const forced = g_forced_table.load(std::memory_order_acquire);
    return forced != nullptr ? *forced : *detected_table();
}

KernelIsa active_dense_kernel_isa() {
    return &active_dense_kernels() == &kBaselineTable ? KernelIsa::Baseline : KernelIsa::Avx2;
}

void force_dense_kernel_isa(KernelIsa isa) {
    if (isa == KernelIsa::Baseline) {
        g_forced_table.store(&kBaselineTable, std::memory_order_release);
        return;
    }
    common::ensure(avx2_dense_kernels() != nullptr && cpu_supports_avx2(),
                   "force_dense_kernel_isa: AVX2 is not available on this build or CPU");
    g_forced_table.store(avx2_dense_kernels(), std::memory_order_release);
}

void reset_dense_kernel_isa() {
    g_forced_table.store(nullptr, std::memory_order_release);
}

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
