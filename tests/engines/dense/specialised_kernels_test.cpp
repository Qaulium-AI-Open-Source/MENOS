// The specialised 1- and 2-qubit CPU kernels must be BITWISE identical to the
// generic k-operand kernel (docs/adr/0019 section 4, owner decision Q4): the
// CPU/CUDA statevector identity of docs/adr/0014 section 10, and with it
// identical counts across backends, rests on every CPU path summing in the
// generic kernel's order. Tolerance is not enough here; the comparison is of
// the bytes. Agreement with an independent oracle is apply_matrix_test.cpp's
// job.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "kernels/dense_kernels.h"

namespace menos::kernels::cpu {
namespace {

using Complex = std::complex<double>;

std::vector<Complex> random_vector(std::size_t size, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> v(size);
    for (Complex& c : v) {
        c = Complex{dist(rng), dist(rng)};
    }
    return v;
}

bool bytes_equal(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(Complex)) == 0;
}

// Applies matrix at operands to the same random state through apply_matrix
// (which dispatches to the specialised kernel) and apply_matrix_generic, and
// requires identical bytes.
void expect_bitwise_equal_to_generic(const std::vector<Complex>& matrix,
                                     const std::vector<std::uint32_t>& operands,
                                     std::uint32_t n,
                                     std::mt19937_64& rng) {
    const std::vector<Complex> initial = random_vector(std::size_t{1} << n, rng);
    std::vector<Complex> specialised = initial;
    std::vector<Complex> generic = initial;
    apply_matrix(specialised, matrix, operands, /*num_threads=*/1);
    apply_matrix_generic(generic, matrix, operands, /*num_threads=*/1);
    ASSERT_TRUE(bytes_equal(specialised, generic))
        << "n=" << n << " operands=" << operands[0] << (operands.size() > 1 ? "," : "")
        << (operands.size() > 1 ? operands[1] : 0);
}

TEST(SpecialisedKernelsTest, OneQubitIsBitwiseIdenticalToGenericOnEveryQubit) {
    std::mt19937_64 rng(1501);
    for (std::uint32_t n = 1; n <= 10; ++n) {
        for (std::uint32_t q = 0; q < n; ++q) {
            expect_bitwise_equal_to_generic(random_vector(4, rng), {q}, n, rng);
        }
    }
}

TEST(SpecialisedKernelsTest, TwoQubitIsBitwiseIdenticalToGenericOnEveryOrderedPair) {
    // Every ordered pair covers adjacent and non-adjacent operands in both
    // orders; the order decides which operand is local bit 0.
    std::mt19937_64 rng(1502);
    for (std::uint32_t n = 2; n <= 9; ++n) {
        for (std::uint32_t q0 = 0; q0 < n; ++q0) {
            for (std::uint32_t q1 = 0; q1 < n; ++q1) {
                if (q0 != q1) {
                    expect_bitwise_equal_to_generic(random_vector(16, rng), {q0, q1}, n, rng);
                }
            }
        }
    }
}

TEST(SpecialisedKernelsTest, SparseMatricesWithSignedZerosStayBitwiseIdentical) {
    // Named gates are mostly zeros (and a negated zero is a different bit
    // pattern), which is where a summation that skipped or reordered terms
    // would first show.
    std::mt19937_64 rng(1503);
    const std::vector<Complex> x = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
    const std::vector<Complex> neg = {{-0.0, -0.0}, {-1.0, 0.0}, {0.0, -1.0}, {-0.0, 0.0}};
    std::vector<Complex> cx(16, Complex{0.0, 0.0});
    cx[0] = cx[5] = cx[11] = cx[14] = Complex{1.0, 0.0};
    std::vector<Complex> neg_cx(16, Complex{-0.0, -0.0});
    neg_cx[0] = neg_cx[7] = neg_cx[10] = neg_cx[13] = Complex{-1.0, 0.0};
    for (std::uint32_t q = 0; q < 6; ++q) {
        expect_bitwise_equal_to_generic(x, {q}, 6, rng);
        expect_bitwise_equal_to_generic(neg, {q}, 6, rng);
    }
    for (const auto& operands :
         std::vector<std::vector<std::uint32_t>>{{0, 1}, {1, 0}, {0, 5}, {5, 2}}) {
        expect_bitwise_equal_to_generic(cx, operands, 6, rng);
        expect_bitwise_equal_to_generic(neg_cx, operands, 6, rng);
    }
}

TEST(SpecialisedKernelsTest, ThreeQubitGatesStillUseTheGenericKernel) {
    // k >= 3 is the generic kernel itself, so this is equality by
    // construction; it pins the dispatch so a later specialisation of k = 3
    // has to come with its own bitwise test.
    std::mt19937_64 rng(1504);
    expect_bitwise_equal_to_generic(random_vector(64, rng), {4, 0, 2}, 6, rng);
}

// Diagonal matrices take the streaming kernels (apply_diagonal_1q/2q). Random
// complex diagonals, including entries with negative real or imaginary parts
// whose products produce signed zeros, on every qubit and operand pair.
TEST(SpecialisedKernelsTest, DiagonalMatricesAreBitwiseIdenticalToGeneric) {
    std::mt19937_64 rng(1505);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (std::uint32_t n = 1; n <= 9; ++n) {
        for (std::uint32_t q = 0; q < n; ++q) {
            const std::vector<Complex> d1 = {Complex{dist(rng), dist(rng)},
                                             Complex{0.0, 0.0},
                                             Complex{0.0, 0.0},
                                             Complex{dist(rng), dist(rng)}};
            expect_bitwise_equal_to_generic(d1, {q}, n, rng);
            // RZ(theta)-shaped: pure phases, the common case.
            const double theta = dist(rng) * 3.0;
            const std::vector<Complex> rz = {std::polar(1.0, -theta / 2),
                                             Complex{0.0, 0.0},
                                             Complex{0.0, 0.0},
                                             std::polar(1.0, theta / 2)};
            expect_bitwise_equal_to_generic(rz, {q}, n, rng);
        }
        if (n < 2) {
            continue;
        }
        for (std::uint32_t a = 0; a < n; ++a) {
            for (std::uint32_t b = 0; b < n; ++b) {
                if (a == b) {
                    continue;
                }
                std::vector<Complex> d2(16, Complex{0.0, 0.0});
                for (std::size_t i = 0; i < 4; ++i) {
                    d2[i * 4 + i] = Complex{dist(rng), dist(rng)};
                }
                expect_bitwise_equal_to_generic(d2, {a, b}, n, rng);
            }
        }
    }
}

// k >= 3 diagonal matrices take apply_diagonal_generic.
TEST(SpecialisedKernelsTest, WideDiagonalMatricesAreBitwiseIdenticalToGeneric) {
    std::mt19937_64 rng(1508);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (std::uint32_t k = 3; k <= 5; ++k) {
        const std::size_t dim = std::size_t{1} << k;
        for (int trial = 0; trial < 6; ++trial) {
            std::vector<Complex> d(dim * dim, Complex{0.0, 0.0});
            for (std::size_t i = 0; i < dim; ++i) {
                d[i * dim + i] = Complex{dist(rng), dist(rng)};
            }
            std::vector<std::uint32_t> operands(7);
            std::iota(operands.begin(), operands.end(), 0u);
            std::shuffle(operands.begin(), operands.end(), rng);
            operands.resize(k);
            expect_bitwise_equal_to_generic(d, operands, 7, rng);
        }
    }
}

// A matrix with a single non-zero off-diagonal entry must NOT take the
// diagonal path (and stays bitwise identical either way).
TEST(SpecialisedKernelsTest, AlmostDiagonalMatricesStillMatchGeneric) {
    std::mt19937_64 rng(1506);
    std::vector<Complex> m(16, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < 4; ++i) {
        m[i * 4 + i] = Complex{0.5, -0.5};
    }
    m[1] = Complex{0.0, 1e-300};
    expect_bitwise_equal_to_generic(m, {2, 0}, 5, rng);
    std::vector<Complex> one = {
        Complex{1.0, 0.0}, Complex{-0.0, 0.0}, Complex{0.0, -0.0}, Complex{0.0, 1.0}};
    expect_bitwise_equal_to_generic(one, {3}, 5, rng);  // signed zeros ARE zero: diagonal path
}

// The batched Z-term path agrees with the per-term kernel within the
// expectation tolerance, for every Z-string on random states, and reports a
// +0.0 imaginary part as the per-term kernel does.
TEST(SpecialisedKernelsTest, DiagonalTermBatchMatchesPerTermExpectation) {
    std::mt19937_64 rng(1507);
    for (std::uint32_t n = 1; n <= 10; ++n) {
        std::vector<Complex> state = random_vector(std::size_t{1} << n, rng);
        double norm = 0.0;
        for (const Complex& c : state) {
            norm += std::norm(c);
        }
        for (Complex& c : state) {
            c /= std::sqrt(norm);
        }
        std::vector<std::uint64_t> masks;
        for (std::uint64_t z = 0; z < (std::uint64_t{1} << n); ++z) {
            masks.push_back(z);
        }
        std::vector<Complex> expected;
        for (const std::uint64_t z : masks) {
            expected.push_back(expectation_of_term(state, 0, z));
        }
        std::vector<Complex> scratch = state;
        const std::vector<Complex> got = expectation_of_diagonal_terms(scratch, masks);
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
            EXPECT_NEAR(got[i].real(), expected[i].real(), 1e-12) << "n=" << n << " z=" << masks[i];
            EXPECT_EQ(got[i].imag(), 0.0);
            EXPECT_FALSE(std::signbit(got[i].imag()));
        }
    }
}

}  // namespace
}  // namespace menos::kernels::cpu
