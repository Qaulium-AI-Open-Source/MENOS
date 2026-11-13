// common::svd calls zgesdd directly, column-major, with buffers it owns
// (Phase 21, docs/adr/0025 section 3.3). Until then it called LAPACKE_zgesdd
// row-major. This test keeps that call as the reference and requires the two to
// agree bit for bit, over every shape class the MPS engine and the Schmidt-rank
// bound use: square, tall, wide, rank-deficient and 1 x n / m x 1.

#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <complex>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4190)
#endif
#include <lapacke.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "common/svd.h"

namespace menos::common {
namespace {

using Complex = std::complex<double>;

SvdResult lapacke_reference(const std::vector<Complex>& a, std::size_t m, std::size_t n) {
    const std::size_t k = std::min(m, n);
    std::vector<Complex> a_copy = a;
    SvdResult result;
    result.singular_values.resize(k);
    result.u.resize(m * k);
    result.vt.resize(k * n);
    const lapack_int info = LAPACKE_zgesdd(LAPACK_ROW_MAJOR,
                                           'S',
                                           static_cast<lapack_int>(m),
                                           static_cast<lapack_int>(n),
                                           a_copy.data(),
                                           static_cast<lapack_int>(n),
                                           result.singular_values.data(),
                                           result.u.data(),
                                           static_cast<lapack_int>(k),
                                           result.vt.data(),
                                           static_cast<lapack_int>(n));
    EXPECT_EQ(info, 0);
    return result;
}

template <typename T>
bool bitwise_equal(const std::vector<T>& x, const std::vector<T>& y) {
    return x.size() == y.size() &&
           (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0);
}

std::vector<Complex> random_matrix(std::size_t m, std::size_t n, std::mt19937_64& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Complex> a(m * n);
    for (Complex& z : a) {
        z = Complex{normal(rng), normal(rng)};
    }
    return a;
}

// A rank-r matrix: the MPS engine's Exact splits are mostly rank-deficient.
std::vector<Complex> low_rank_matrix(std::size_t m,
                                     std::size_t n,
                                     std::size_t r,
                                     std::mt19937_64& rng) {
    const std::vector<Complex> left = random_matrix(m, r, rng);
    const std::vector<Complex> right = random_matrix(r, n, rng);
    std::vector<Complex> a(m * n, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t t = 0; t < r; ++t) {
                a[(i * n) + j] += left[(i * r) + t] * right[(t * n) + j];
            }
        }
    }
    return a;
}

void expect_bitwise_equal_to_lapacke(const std::vector<Complex>& a, std::size_t m, std::size_t n) {
    const SvdResult expected = lapacke_reference(a, m, n);
    const SvdResult actual = svd(a, m, n);
    EXPECT_TRUE(bitwise_equal(actual.singular_values, expected.singular_values)) << m << " x " << n;
    EXPECT_TRUE(bitwise_equal(actual.u, expected.u)) << m << " x " << n;
    EXPECT_TRUE(bitwise_equal(actual.vt, expected.vt)) << m << " x " << n;
}

TEST(SvdLapackeReferenceTest, BitwiseIdenticalToLapackeOnRandomShapes) {
    std::mt19937_64 rng(20261020);
    const std::vector<std::size_t> sizes = {1, 2, 3, 4, 7, 8, 16, 31, 64, 65, 128};
    for (const std::size_t m : sizes) {
        for (const std::size_t n : sizes) {
            expect_bitwise_equal_to_lapacke(random_matrix(m, n, rng), m, n);
        }
    }
}

TEST(SvdLapackeReferenceTest, BitwiseIdenticalToLapackeOnRankDeficientMatrices) {
    std::mt19937_64 rng(7);
    for (const std::size_t m : {std::size_t{4}, std::size_t{8}, std::size_t{32}, std::size_t{64}}) {
        for (const std::size_t n : {std::size_t{4}, std::size_t{16}, std::size_t{64}}) {
            for (const std::size_t r : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
                expect_bitwise_equal_to_lapacke(low_rank_matrix(m, n, r, rng), m, n);
            }
        }
    }
}

TEST(SvdLapackeReferenceTest, BitwiseIdenticalToLapackeOnAllZeroMatrix) {
    const std::vector<Complex> zero(std::size_t{6} * 4, Complex{0.0, 0.0});
    expect_bitwise_equal_to_lapacke(zero, 6, 4);
}

}  // namespace
}  // namespace menos::common
