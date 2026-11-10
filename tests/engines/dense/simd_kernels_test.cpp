// Phase 24 item 8 (docs/adr/0028 section 19): both kernel tables, the
// baseline and the AVX2 one, must be BITWISE identical to
// apply_matrix_generic on every operand placement and every thread count.
// The AVX2 table is the one production uses on the verification machines, and
// the baseline one serves every CPU without AVX2; forcing each in turn tests
// both on one machine. Tolerance is not enough: the comparison is of bytes,
// because the CPU/CUDA, in-memory/disk and Phase 21 baseline identities all
// rest on it.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/cpu/dense_kernel_table.h"
#include "kernels/dense_kernels.h"

namespace menos::kernels::cpu {
namespace {

using Complex = std::complex<double>;

// Draws from raw mt19937_64 output, so the states are the same under every
// standard library (std::uniform_real_distribution is not).
double unit(std::mt19937_64& rng) {
    return (static_cast<double>(rng() >> 11) * 0x1.0p-53 * 2.0) - 1.0;
}

std::vector<Complex> random_vector(std::size_t size, std::mt19937_64& rng) {
    std::vector<Complex> v(size);
    for (Complex& c : v) {
        const double re = unit(rng);
        const double im = unit(rng);
        c = Complex{re, im};
    }
    return v;
}

std::vector<Complex> random_dense(std::size_t dim, std::mt19937_64& rng) {
    return random_vector(dim * dim, rng);
}

std::vector<Complex> random_diagonal(std::size_t dim, std::mt19937_64& rng) {
    std::vector<Complex> m(dim * dim, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < dim; ++i) {
        const double re = unit(rng);
        const double im = unit(rng);
        m[i * dim + i] = Complex{re, im};
    }
    return m;
}

bool bytes_equal(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(Complex)) == 0;
}

std::vector<KernelIsa> isas() {
    std::vector<KernelIsa> out = {KernelIsa::Baseline};
    if (avx2_dense_kernels() != nullptr && cpu_supports_avx2()) {
        out.push_back(KernelIsa::Avx2);
    }
    return out;
}

const char* name(KernelIsa isa) {
    return isa == KernelIsa::Baseline ? "baseline" : "avx2";
}

// Forces one table for a scope and restores detection after.
class ForcedIsa {
public:
    explicit ForcedIsa(KernelIsa isa) {
        force_dense_kernel_isa(isa);
    }
    ForcedIsa(const ForcedIsa&) = delete;
    ForcedIsa& operator=(const ForcedIsa&) = delete;
    ForcedIsa(ForcedIsa&&) = delete;
    ForcedIsa& operator=(ForcedIsa&&) = delete;
    ~ForcedIsa() {
        reset_dense_kernel_isa();
    }
};

std::string describe(const std::vector<std::uint32_t>& operands,
                     std::uint32_t n,
                     std::uint32_t threads,
                     KernelIsa isa) {
    std::string s = std::string(name(isa)) + " n=" + std::to_string(n) +
                    " threads=" + std::to_string(threads) + " operands=";
    for (const std::uint32_t q : operands) {
        s += std::to_string(q) + ",";
    }
    return s;
}

// apply_matrix under every table and every thread count against
// apply_matrix_generic on one thread, from the same state.
void expect_all_tables_bitwise(const std::vector<Complex>& matrix,
                               const std::vector<std::uint32_t>& operands,
                               std::uint32_t n,
                               const std::vector<std::uint32_t>& thread_counts,
                               std::mt19937_64& rng) {
    const std::vector<Complex> initial = random_vector(std::size_t{1} << n, rng);
    std::vector<Complex> oracle = initial;
    apply_matrix_generic(oracle, matrix, operands, /*num_threads=*/1);
    for (const KernelIsa isa : isas()) {
        const ForcedIsa forced(isa);
        for (const std::uint32_t threads : thread_counts) {
            std::vector<Complex> got = initial;
            apply_matrix(got, matrix, operands, threads);
            ASSERT_TRUE(bytes_equal(got, oracle)) << describe(operands, n, threads, isa);
        }
    }
}

TEST(SimdKernelsTest, ProductionUsesAvx2WhereTheCpuHasIt) {
    reset_dense_kernel_isa();
    if (avx2_dense_kernels() != nullptr && cpu_supports_avx2()) {
        EXPECT_EQ(active_dense_kernel_isa(), KernelIsa::Avx2);
    } else {
        EXPECT_EQ(active_dense_kernel_isa(), KernelIsa::Baseline);
        GTEST_SKIP() << "no AVX2 on this build or CPU -- the AVX2 table did NOT run";
    }
}

TEST(SimdKernelsTest, ForcingTheBaselineTableIsUndoneByReset) {
    {
        const ForcedIsa forced(KernelIsa::Baseline);
        EXPECT_EQ(active_dense_kernel_isa(), KernelIsa::Baseline);
    }
    if (avx2_dense_kernels() != nullptr && cpu_supports_avx2()) {
        EXPECT_EQ(active_dense_kernel_isa(), KernelIsa::Avx2);
    }
}

TEST(SimdKernelsTest, OneQubitEveryQubitBothTables) {
    std::mt19937_64 rng(2421);
    for (std::uint32_t n = 1; n <= 10; ++n) {
        for (std::uint32_t q = 0; q < n; ++q) {
            expect_all_tables_bitwise(random_dense(2, rng), {q}, n, {1}, rng);
        }
    }
}

TEST(SimdKernelsTest, TwoQubitEveryOrderedPairBothTables) {
    std::mt19937_64 rng(2422);
    for (std::uint32_t n = 2; n <= 8; ++n) {
        for (std::uint32_t a = 0; a < n; ++a) {
            for (std::uint32_t b = 0; b < n; ++b) {
                if (a != b) {
                    expect_all_tables_bitwise(random_dense(4, rng), {a, b}, n, {1}, rng);
                }
            }
        }
    }
}

TEST(SimdKernelsTest, DiagonalsOfEveryWidthAndPlacementBothTables) {
    std::mt19937_64 rng(2423);
    const std::vector<std::vector<std::uint32_t>> placements = {
        {0},
        {1},
        {2},
        {3},
        {8},
        {0, 1},
        {1, 0},
        {2, 7},
        {6, 7},
        {0, 4, 8},
        {5, 1, 3},
        {0, 1, 2, 3},
        {8, 2, 5, 0},
        {0, 2, 4, 6, 8},
        {7, 6, 5, 4, 3},
    };
    for (const auto& operands : placements) {
        expect_all_tables_bitwise(
            random_diagonal(std::size_t{1} << operands.size(), rng), operands, 9, {1}, rng);
    }
}

// Signed zeros in the matrix and the state: the kernels' (+0, +0) start must
// normalise them exactly as the generic sum does.
TEST(SimdKernelsTest, SignedZerosStayBitwiseIdenticalBothTables) {
    std::mt19937_64 rng(2424);
    const std::vector<Complex> x = {{0.0, -0.0}, {1.0, 0.0}, {1.0, -0.0}, {-0.0, 0.0}};
    std::vector<Complex> cx(16, Complex{-0.0, 0.0});
    for (const std::size_t e : {0U, 7U, 10U, 13U}) {  // CX, control local bit 0
        cx[e] = Complex{1.0, -0.0};
    }
    const std::vector<Complex> z = {{1.0, 0.0}, {-0.0, -0.0}, {0.0, 0.0}, {-1.0, -0.0}};
    for (const KernelIsa isa : isas()) {
        const ForcedIsa forced(isa);
        for (std::uint32_t n = 2; n <= 6; ++n) {
            std::vector<Complex> initial = random_vector(std::size_t{1} << n, rng);
            for (std::size_t i = 0; i < initial.size(); i += 3) {
                initial[i] = Complex{-0.0, 0.0};
            }
            const auto check = [&](const std::vector<Complex>& matrix,
                                   const std::vector<std::uint32_t>& operands) {
                std::vector<Complex> got = initial;
                std::vector<Complex> oracle = initial;
                apply_matrix(got, matrix, operands, 1);
                apply_matrix_generic(oracle, matrix, operands, 1);
                ASSERT_TRUE(bytes_equal(got, oracle)) << describe(operands, n, 1, isa);
            };
            for (std::uint32_t q = 0; q < n; ++q) {
                check(x, {q});
                check(z, {q});
            }
            check(cx, {0, 1});
            check(cx, {n - 1, 0});
        }
    }
}

// n = 18: 2^17 pairs and 2^16 two-qubit groups, so every kernel is threaded,
// and 3 and 7 threads split the groups at odd boundaries.
TEST(SimdKernelsTest, ThreadedSweepsAreBitwiseIdenticalBothTables) {
    std::mt19937_64 rng(2425);
    constexpr std::uint32_t kN = 18;
    const std::vector<std::uint32_t> threads = {1, 2, 3, 7};
    for (const std::uint32_t q : {0U, 1U, 2U, 3U, 9U, 17U}) {
        expect_all_tables_bitwise(random_dense(2, rng), {q}, kN, threads, rng);
        expect_all_tables_bitwise(random_diagonal(2, rng), {q}, kN, threads, rng);
    }
    const std::vector<std::vector<std::uint32_t>> pairs = {
        {0, 1}, {1, 0}, {0, 17}, {17, 0}, {8, 9}, {16, 17}, {17, 3}, {2, 1}};
    for (const auto& p : pairs) {
        expect_all_tables_bitwise(random_dense(4, rng), p, kN, threads, rng);
        expect_all_tables_bitwise(random_diagonal(4, rng), p, kN, threads, rng);
    }
    expect_all_tables_bitwise(random_diagonal(8, rng), {0, 1, 5}, kN, threads, rng);
}

}  // namespace
}  // namespace menos::kernels::cpu
