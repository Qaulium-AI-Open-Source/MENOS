// The AVX2 table of the CPU dense kernels (Phase 24 item 8, docs/adr/0028
// section 19). Compiled with -mavx2 (GCC, Clang) or /arch:AVX2 (MSVC), never
// with FMA enabled, with contraction off (-ffp-contract=off, /fp:precise), and
// selected at run time only on a CPU and OS that support AVX2
// (cpu_supports_avx2, dense_kernels.cpp).
//
// Bitwise rule. Each lane holds one amplitude's real or imaginary part and
// performs exactly the scalar kernels' operations in their order:
//   cmul(m, x) = (m.r*x.r - m.i*x.i, m.r*x.i + m.i*x.r)
// as four separate multiplies (vmulpd) and one vaddsubpd, never a fused
// multiply-add; every output starts as (+0, +0) + first product, then adds
// the later products in increasing input order (vaddpd). Lanes run across
// amplitudes, never across a sum. So the results are the baseline table's,
// and apply_matrix_generic's, bit for bit (tests/engines/dense/
// simd_kernels_test.cpp pins this on every operand placement and thread
// count).
//
// Linkage rule. This file calls no inline function from a shared header
// (std::complex, std::min, ...): an out-of-line copy of such a function
// compiled here would hold AVX2 instructions, and the linker may keep that
// copy for every caller, including those that run on CPUs without AVX2.
// Everything below is intrinsics and plain arithmetic on double pointers, at
// internal linkage.

#include "kernels/cpu/dense_kernel_table.h"

#if defined(MENOS_HAVE_AVX2_KERNELS)

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace menos::kernels::cpu {
namespace {

// [x0, x1] two complex values; m broadcast as (mr, mi). Per lane cmul(m, x).
inline __m256d cmul2(__m256d mr, __m256d mi, __m256d x) {
    const __m256d t1 = _mm256_mul_pd(mr, x);                         // mr*xr, mr*xi
    const __m256d t2 = _mm256_mul_pd(mi, _mm256_permute_pd(x, 0x5));  // mi*xi, mi*xr
    return _mm256_addsub_pd(t1, t2);  // mr*xr - mi*xi, mr*xi + mi*xr
}

// One complex value, the same operations.
inline __m128d cmul1(__m128d mr, __m128d mi, __m128d x) {
    const __m128d t1 = _mm_mul_pd(mr, x);
    const __m128d t2 = _mm_mul_pd(mi, _mm_shuffle_pd(x, x, 0x1));
    return _mm_addsub_pd(t1, t2);
}

// +0.0 read through a volatile, once per kernel call. Every output starts as
// (+0) + first product, which turns a -0.0 product into +0.0 exactly as the
// scalar kernels and the generic sum do; that add is not an identity, but
// MSVC 14.50 /arch:AVX2 folds _mm256_add_pd(_mm256_setzero_pd(), p) to p
// anyway (seen in item 8 step 2: SimdKernelsTest.SignedZeros* failed, and
// the disassembly had no add). A zero the optimiser cannot see keeps the add.
double opaque_zero() {
    static volatile double zero = 0.0;
    return zero;
}

inline __m256d zero4() {
    return _mm256_set1_pd(opaque_zero());
}

inline __m128d zero2() {
    return _mm_set1_pd(opaque_zero());
}

struct Bcast2 {
    __m256d r;
    __m256d i;
};

struct Bcast1 {
    __m128d r;
    __m128d i;
};

inline Bcast2 bcast2(const double* c) {
    return {_mm256_set1_pd(c[0]), _mm256_set1_pd(c[1])};
}

inline Bcast1 bcast1(const double* c) {
    return {_mm_set1_pd(c[0]), _mm_set1_pd(c[1])};
}

// A 2x2 gate's entries broadcast once per call: the state pointers may alias
// the matrix pointer as far as the compiler knows, so a broadcast inside a
// loop that stores to the state would be re-read every iteration.
struct Gate1 {
    Bcast2 m00;
    Bcast2 m01;
    Bcast2 m10;
    Bcast2 m11;
    __m256d zero;
};

inline Gate1 gate1(const double* m) {
    return {bcast2(m), bcast2(m + 2), bcast2(m + 4), bcast2(m + 6), zero4()};
}

// (l, h) <- m * (a0, a1) for the two pairs in a0 = [lo values], a1 = [hi values].
inline void pair2(const Gate1& g, __m256d a0, __m256d a1, __m256d& l, __m256d& h) {
    l = _mm256_add_pd(_mm256_add_pd(g.zero, cmul2(g.m00.r, g.m00.i, a0)), cmul2(g.m01.r, g.m01.i, a1));
    h = _mm256_add_pd(_mm256_add_pd(g.zero, cmul2(g.m10.r, g.m10.i, a0)), cmul2(g.m11.r, g.m11.i, a1));
}

// One pair, 128-bit: a run's odd last pair.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline void pair1(double* lo, double* hi, const double* m) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const Bcast1 s00 = bcast1(m);
    const Bcast1 s01 = bcast1(m + 2);
    const Bcast1 s10 = bcast1(m + 4);
    const Bcast1 s11 = bcast1(m + 6);
    const __m128d z = zero2();
    const __m128d a0 = _mm_loadu_pd(lo);
    const __m128d a1 = _mm_loadu_pd(hi);
    _mm_storeu_pd(lo, _mm_add_pd(_mm_add_pd(z, cmul1(s00.r, s00.i, a0)), cmul1(s01.r, s01.i, a1)));
    _mm_storeu_pd(hi, _mm_add_pd(_mm_add_pd(z, cmul1(s10.r, s10.i, a0)), cmul1(s11.r, s11.i, a1)));
}

// lo, hi <- m * (lo, hi) for n pairs: four pairs per iteration (two
// independent vectors, for the in-cache case), then two, then one.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline void pair_run_body(double* lo, double* hi, std::size_t n, const Gate1& g, const double* m) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d a0 = _mm256_loadu_pd(lo + 2 * i);
        const __m256d a1 = _mm256_loadu_pd(hi + 2 * i);
        const __m256d b0 = _mm256_loadu_pd(lo + 2 * i + 4);
        const __m256d b1 = _mm256_loadu_pd(hi + 2 * i + 4);
        __m256d la;
        __m256d ha;
        __m256d lb;
        __m256d hb;
        pair2(g, a0, a1, la, ha);
        pair2(g, b0, b1, lb, hb);
        _mm256_storeu_pd(lo + 2 * i, la);
        _mm256_storeu_pd(hi + 2 * i, ha);
        _mm256_storeu_pd(lo + 2 * i + 4, lb);
        _mm256_storeu_pd(hi + 2 * i + 4, hb);
    }
    for (; i + 2 <= n; i += 2) {
        __m256d l;
        __m256d h;
        pair2(g, _mm256_loadu_pd(lo + 2 * i), _mm256_loadu_pd(hi + 2 * i), l, h);
        _mm256_storeu_pd(lo + 2 * i, l);
        _mm256_storeu_pd(hi + 2 * i, h);
    }
    if (i < n) {
        pair1(lo + 2 * i, hi + 2 * i, m);
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void avx2_pair_run(double* lo, double* hi, std::size_t n, const double* m) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    pair_run_body(lo, hi, n, gate1(m), m);
}

// Operand 0: each pair is two adjacent amplitudes. Two pairs per step:
// X = [p.a0, p.a1], Y = [q.a0, q.a1] regrouped into A0 = [p.a0, q.a0] and
// A1 = [p.a1, q.a1], computed, and regrouped back.
void pair_blocks_q0(double* amps, std::size_t total, const double* m) {
    const Gate1 g = gate1(m);
    std::size_t base = 0;
    for (; base + 4 <= total; base += 4) {
        const __m256d x = _mm256_loadu_pd(amps + 2 * base);
        const __m256d y = _mm256_loadu_pd(amps + 2 * base + 4);
        __m256d l;
        __m256d h;
        pair2(g, _mm256_permute2f128_pd(x, y, 0x20), _mm256_permute2f128_pd(x, y, 0x31), l, h);
        _mm256_storeu_pd(amps + 2 * base, _mm256_permute2f128_pd(l, h, 0x20));
        _mm256_storeu_pd(amps + 2 * base + 4, _mm256_permute2f128_pd(l, h, 0x31));
    }
    if (base < total) {  // a single pair: total == 2
        pair1(amps + 2 * base, amps + 2 * base + 2, m);
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void avx2_pair_blocks(double* amps, std::size_t total, std::uint32_t q, const double* m) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const Gate1 g = gate1(m);
    switch (q) {
        case 0:
            pair_blocks_q0(amps, total, m);
            return;
        case 1:  // blocks of 4: lo = 2 amplitudes, hi = the next 2
            for (std::size_t base = 0; base < total; base += 4) {
                __m256d l;
                __m256d h;
                pair2(g, _mm256_loadu_pd(amps + 2 * base), _mm256_loadu_pd(amps + 2 * base + 4), l, h);
                _mm256_storeu_pd(amps + 2 * base, l);
                _mm256_storeu_pd(amps + 2 * base + 4, h);
            }
            return;
        default:  // q == 2, blocks of 8: lo = 4 amplitudes, hi = the next 4
            for (std::size_t base = 0; base < total; base += 8) {
                pair_run_body(amps + 2 * base, amps + 2 * base + 8, 4, g, m);
            }
            return;
    }
}

void avx2_scale_run(double* a, std::size_t n, const double* s) {
    const Bcast2 v = bcast2(s);
    const __m256d zero = zero4();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d x = _mm256_loadu_pd(a + 2 * i);
        const __m256d y = _mm256_loadu_pd(a + 2 * i + 4);
        _mm256_storeu_pd(a + 2 * i, _mm256_add_pd(zero, cmul2(v.r, v.i, x)));
        _mm256_storeu_pd(a + 2 * i + 4, _mm256_add_pd(zero, cmul2(v.r, v.i, y)));
    }
    for (; i + 2 <= n; i += 2) {
        const __m256d x = _mm256_loadu_pd(a + 2 * i);
        _mm256_storeu_pd(a + 2 * i, _mm256_add_pd(zero, cmul2(v.r, v.i, x)));
    }
    if (i < n) {
        const Bcast1 w = bcast1(s);
        const __m128d x = _mm_loadu_pd(a + 2 * i);
        _mm_storeu_pd(a + 2 * i, _mm_add_pd(zero2(), cmul1(w.r, w.i, x)));
    }
}

// One element: amps[i] <- (+0, +0) + d[local] * amps[i].
inline void scale_one(double* amps, std::size_t i, const double* d, std::size_t local) {
    const Bcast1 w = bcast1(d + 2 * local);
    const __m128d x = _mm_loadu_pd(amps + 2 * i);
    _mm_storeu_pd(amps + 2 * i, _mm_add_pd(zero2(), cmul1(w.r, w.i, x)));
}

// The low-operand diagonal kernel. local(i) is assembled from the operands
// below bit 3 by an 8-entry table indexed by i & 7, and from the others once
// per aligned block of 8 (their bits are constant over it): the same local
// index as the bit-by-bit loop, so the same entry and the same bits. Two
// elements per vector: D = [d[l0], d[l1]], broadcast per lane.
void avx2_scale_elements(double* amps,
                         std::size_t begin,
                         std::size_t end,
                         const std::uint32_t* operands,
                         std::size_t k,
                         const double* d) {
    std::size_t low_table[8] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    for (std::size_t t = 0; t < 8; ++t) {
        for (std::size_t j = 0; j < k; ++j) {
            if (operands[j] < 3) {
                low_table[t] |= ((t >> operands[j]) & std::size_t{1}) << j;
            }
        }
    }
    const __m256d zero = zero4();
    std::size_t i = begin;
    while (i < end) {
        const std::size_t block = i & ~std::size_t{7};
        const std::size_t block_end = (block + 8 < end) ? block + 8 : end;
        std::size_t high = 0;
        for (std::size_t j = 0; j < k; ++j) {
            if (operands[j] >= 3) {
                high |= ((block >> operands[j]) & std::size_t{1}) << j;
            }
        }
        if ((i & 1) != 0) {
            scale_one(amps, i, d, high | low_table[i & 7]);
            ++i;
        }
        for (; i + 2 <= block_end; i += 2) {
            const std::size_t l0 = high | low_table[i & 7];
            const std::size_t l1 = high | low_table[(i + 1) & 7];
            const __m256d e = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm_loadu_pd(d + 2 * l0)), _mm_loadu_pd(d + 2 * l1), 1);
            const __m256d er = _mm256_movedup_pd(e);           // d0.r, d0.r, d1.r, d1.r
            const __m256d ei = _mm256_permute_pd(e, 0xF);      // d0.i, d0.i, d1.i, d1.i
            const __m256d x = _mm256_loadu_pd(amps + 2 * i);
            _mm256_storeu_pd(amps + 2 * i, _mm256_add_pd(zero, cmul2(er, ei, x)));
        }
        if (i < block_end) {
            scale_one(amps, i, d, high | low_table[i & 7]);
            ++i;
        }
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline std::size_t insert_zero_bit(std::size_t i, std::uint32_t pos) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const std::size_t low = (std::size_t{1} << pos) - 1;
    return ((i & ~low) << 1) | (i & low);
}

// One 2-qubit group at base, 128-bit: the tail and odd-start case.
inline void two_qubit_one(double* amps,
                          std::size_t base,
                          const std::size_t* off,
                          const double* m) {
    __m128d a[4];  // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    for (std::size_t in = 0; in < 4; ++in) {
        a[in] = _mm_loadu_pd(amps + 2 * (base + off[in]));
    }
    for (std::size_t out = 0; out < 4; ++out) {
        const Bcast1 c0 = bcast1(m + 2 * (out * 4 + 0));
        __m128d sum = _mm_add_pd(zero2(), cmul1(c0.r, c0.i, a[0]));
        for (std::size_t in = 1; in < 4; ++in) {
            const Bcast1 c = bcast1(m + 2 * (out * 4 + in));
            sum = _mm_add_pd(sum, cmul1(c.r, c.i, a[in]));
        }
        _mm_storeu_pd(amps + 2 * (base + off[out]), sum);
    }
}

// Groups [begin, end) of the 2-qubit gate m on (q0, q1). Two groups per
// vector: when both operands are above bit 0, an even group i and i + 1 sit
// at base and base + 1 for every local index, so each of the four inputs is
// one 256-bit load; otherwise the two groups' amplitudes are gathered and
// scattered as 128-bit halves.
void avx2_two_qubit(double* amps,
                    std::size_t begin,
                    std::size_t end,
                    std::uint32_t q0,
                    std::uint32_t q1,
                    const double* m) {
    const std::uint32_t lo = q0 < q1 ? q0 : q1;
    const std::uint32_t hi = q0 < q1 ? q1 : q0;
    const std::size_t b0 = std::size_t{1} << q0;
    const std::size_t b1 = std::size_t{1} << q1;
    const std::size_t off[4] = {0, b0, b1, b0 | b1};  // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    Bcast2 c[16];                                     // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    for (std::size_t e = 0; e < 16; ++e) {
        c[e] = bcast2(m + 2 * e);
    }
    const __m256d zero = zero4();
    std::size_t i = begin;
    if (lo >= 1) {
        if (i < end && (i & 1) != 0) {
            two_qubit_one(amps, insert_zero_bit(insert_zero_bit(i, lo), hi), off, m);
            ++i;
        }
        for (; i + 2 <= end; i += 2) {
            const std::size_t base = insert_zero_bit(insert_zero_bit(i, lo), hi);
            __m256d a[4];  // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
            for (std::size_t in = 0; in < 4; ++in) {
                a[in] = _mm256_loadu_pd(amps + 2 * (base + off[in]));
            }
            for (std::size_t out = 0; out < 4; ++out) {
                __m256d sum = _mm256_add_pd(zero, cmul2(c[out * 4].r, c[out * 4].i, a[0]));
                for (std::size_t in = 1; in < 4; ++in) {
                    sum = _mm256_add_pd(sum, cmul2(c[out * 4 + in].r, c[out * 4 + in].i, a[in]));
                }
                _mm256_storeu_pd(amps + 2 * (base + off[out]), sum);
            }
        }
    } else {
        for (; i + 2 <= end; i += 2) {
            const std::size_t base_a = insert_zero_bit(insert_zero_bit(i, lo), hi);
            const std::size_t base_b = insert_zero_bit(insert_zero_bit(i + 1, lo), hi);
            __m256d a[4];  // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
            for (std::size_t in = 0; in < 4; ++in) {
                a[in] = _mm256_insertf128_pd(
                    _mm256_castpd128_pd256(_mm_loadu_pd(amps + 2 * (base_a + off[in]))),
                    _mm_loadu_pd(amps + 2 * (base_b + off[in])),
                    1);
            }
            for (std::size_t out = 0; out < 4; ++out) {
                __m256d sum = _mm256_add_pd(zero, cmul2(c[out * 4].r, c[out * 4].i, a[0]));
                for (std::size_t in = 1; in < 4; ++in) {
                    sum = _mm256_add_pd(sum, cmul2(c[out * 4 + in].r, c[out * 4 + in].i, a[in]));
                }
                _mm_storeu_pd(amps + 2 * (base_a + off[out]), _mm256_castpd256_pd128(sum));
                _mm_storeu_pd(amps + 2 * (base_b + off[out]), _mm256_extractf128_pd(sum, 1));
            }
        }
    }
    if (i < end) {
        two_qubit_one(amps, insert_zero_bit(insert_zero_bit(i, lo), hi), off, m);
    }
}

constexpr DenseKernelTable kAvx2Table{
    .pair_run = avx2_pair_run,
    .pair_blocks = avx2_pair_blocks,
    .scale_run = avx2_scale_run,
    .scale_elements = avx2_scale_elements,
    .two_qubit = avx2_two_qubit,
};

}  // namespace

const DenseKernelTable* avx2_dense_kernels() {
    return &kAvx2Table;
}

}  // namespace menos::kernels::cpu

#endif  // MENOS_HAVE_AVX2_KERNELS
